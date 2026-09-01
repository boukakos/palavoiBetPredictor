import argparse
import io
import json
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_tools.download_historical_data import resolve_active_season_candidates, sync_historical_data

DATA_DIR = ROOT / "data"
RAW_HISTORY_FILE = DATA_DIR / "historical_raw" / "premier_league_10_years.csv"
PROCESSED_FILE = DATA_DIR / "processed" / "model_features_10_years.csv"
PREDICTION_DIR = DATA_DIR / "predictions"
UPCOMING_FIXTURES_FILE = DATA_DIR / "upcoming_fixtures.csv"
OUTPUT_CSV = PREDICTION_DIR / "live_upcoming_predictions.csv"
OUTPUT_JSON = PREDICTION_DIR / "live_upcoming_predictions.json"

FEATURES_1X2 = [
    "h_form_pts",
    "a_form_pts",
    "h_form_gdiff",
    "a_form_gdiff",
    "h_form_sot_diff",
    "a_form_sot_diff",
    "h_venue_pts",
    "a_venue_pts",
    "Home_xG_L5",
    "Away_xG_L5",
    "xG_Diff",
    "xG_Ratio",
    "elo_diff",
    "prob_home_market",
    "prob_draw_market",
    "prob_away_market",
    "h_rest_days",
    "a_rest_days",
    "rest_diff",
    "h2h_home_win_rate",
    "h_cards_avg",
    "a_cards_avg",
    "ref_home_bias",
]

FEATURES_OU = [
    "h_goals_scored_avg",
    "h_goals_conceded_avg",
    "a_goals_scored_avg",
    "a_goals_conceded_avg",
    "expected_total_goals_proxy",
    "Home_xG_L5",
    "Away_xG_L5",
    "xG_Diff",
    "xG_Ratio",
    "h_over25_rate",
    "a_over25_rate",
    "h_form_sot_diff",
    "a_form_sot_diff",
    "h_clean_sheet_rate",
    "a_clean_sheet_rate",
    "elo_diff",
]

FEATURES_BTTS = [
    "h_goals_scored_avg",
    "h_goals_conceded_avg",
    "a_goals_scored_avg",
    "a_goals_conceded_avg",
    "h_btts_rate",
    "a_btts_rate",
    "Home_xG_L5",
    "Away_xG_L5",
    "xG_Diff",
    "xG_Ratio",
    "h_clean_sheet_rate",
    "a_clean_sheet_rate",
    "expected_total_goals_proxy",
    "elo_diff",
]

TARGET_MAP = {"H": 0, "D": 1, "A": 2}
INITIAL_BANKROLL = 1000.0
UNIT_STAKE = 10.0
STANDARD_KELLY_FRACTION = 0.125
STANDARD_KELLY_CAP = INITIAL_BANKROLL * 0.015
LONGSHOT_KELLY_FRACTION = 0.0625
LONGSHOT_KELLY_CAP = INITIAL_BANKROLL * 0.005
MIN_EDGE_PERCENT = 4.0
SEASON_CANDIDATES = resolve_active_season_candidates()
CURRENT_EPL_CLUBS = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Ipswich Town",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Southampton",
    "Tottenham Hotspur",
    "West Ham United",
    "Wolverhampton Wanderers",
]


def safe_float(value):
    if value is None:
        return np.nan
    converted = pd.to_numeric(value, errors="coerce")
    if pd.isna(converted):
        return np.nan
    return float(converted)


def safe_odd(value):
    odd = safe_float(value)
    if pd.isna(odd) or odd <= 1.0:
        return np.nan
    return float(odd)


def parse_date_series(values):
    values = pd.Series(values)
    if values.empty:
        return pd.to_datetime(pd.Series([], dtype="datetime64[ns]"))
    string_values = values.astype(str).str.strip()
    parsed = pd.to_datetime(string_values, format="%Y-%m-%d", errors="coerce")
    fallback_mask = parsed.isna()
    if fallback_mask.any():
        parsed[fallback_mask] = pd.to_datetime(string_values[fallback_mask], dayfirst=True, errors="coerce")
    return parsed


def calculate_elo(r_home, r_away, outcome, k=20, home_adv=50):
    dr = (r_home + home_adv) - r_away
    e_home = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
    e_away = 1.0 - e_home
    new_r_home = r_home + k * (outcome - e_home)
    new_r_away = r_away + k * ((1.0 - outcome) - e_away)
    return new_r_home, new_r_away


def build_xgb_model(market: str) -> XGBClassifier:
    if market == "1x2":
        return XGBClassifier(
            objective="multi:softprob",
            n_estimators=350,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            gamma=0.05,
            reg_alpha=0.4,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.05,
        reg_alpha=0.5,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def compute_multiclass_brier_score(y_true, probs):
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    one_hot = np.eye(probs.shape[1], dtype=float)[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_metric_summary(y_true, probs, market: str):
    if market == "1x2":
        predictions = probs.argmax(axis=1)
        accuracy = accuracy_score(y_true, predictions)
        loss = log_loss(y_true, probs, labels=[0, 1, 2])
        brier = compute_multiclass_brier_score(y_true, probs)
    else:
        positive_probs = probs[:, 1]
        predictions = (positive_probs >= 0.5).astype(int)
        accuracy = accuracy_score(y_true, predictions)
        loss = log_loss(y_true, positive_probs)
        brier = float(np.mean((positive_probs - y_true) ** 2))

    return {
        "accuracy": float(accuracy),
        "log_loss": float(loss),
        "brier_score": float(brier),
    }


def fit_calibrated_model(X, y, market: str):
    unique_values = np.unique(y)
    n_splits = min(3, max(2, len(unique_values)))
    if len(unique_values) < 2:
        n_splits = 2
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_method = None
    best_metrics = None
    best_model = None

    for method in ("sigmoid", "isotonic"):
        calibrator = CalibratedClassifierCV(
            estimator=build_xgb_model(market),
            method=method,
            cv=inner_cv,
            ensemble=True,
        )
        probs = cross_val_predict(calibrator, X, y, cv=inner_cv, method="predict_proba")
        metrics = compute_metric_summary(y, probs, market)
        if best_metrics is None or metrics["log_loss"] < best_metrics["log_loss"]:
            best_method = method
            best_metrics = metrics

    final_model = CalibratedClassifierCV(
        estimator=build_xgb_model(market),
        method=best_method,
        cv=inner_cv,
        ensemble=True,
    )
    final_model.fit(X, y)
    return final_model, best_method, best_metrics


def extract_form(history, venue_filter=None):
    if venue_filter is not None:
        relevant = [match for match in history if match.get("Venue") == venue_filter]
    else:
        relevant = list(history)

    if not relevant:
        return {
            "pts_avg": 1.0,
            "goal_diff_avg": 0.0,
            "sot_diff_avg": 0.0,
            "corners_avg": 4.5,
            "clean_sheets": 0.0,
            "cards_avg": 1.5,
            "goals_scored_avg": 1.2,
            "goals_conceded_avg": 1.2,
            "over25_rate": 0.50,
            "btts_rate": 0.50,
        }

    last5 = relevant[-5:] if len(relevant) >= 5 else relevant
    weights = np.arange(1, len(last5) + 1, dtype=float)
    pts = np.asarray([match["Pts"] for match in last5], dtype=float)
    g_diff = np.asarray([match["GF"] - match["GA"] for match in last5], dtype=float)
    sot_diff = np.asarray([match["SoT_F"] - match["SoT_A"] for match in last5], dtype=float)
    goal_for = np.asarray([match["GF"] for match in last5], dtype=float)
    goal_against = np.asarray([match["GA"] for match in last5], dtype=float)
    clean_sheets = np.asarray([1.0 if match["GA"] == 0 else 0.0 for match in last5], dtype=float)
    cards = np.asarray([match["Yellow"] + (2.0 * match["Red"]) for match in last5], dtype=float)
    over25 = np.asarray([1.0 if (match["GF"] + match["GA"]) > 2.5 else 0.0 for match in last5], dtype=float)
    btts = np.asarray([1.0 if (match["GF"] > 0 and match["GA"] > 0) else 0.0 for match in last5], dtype=float)

    return {
        "pts_avg": float(np.average(pts, weights=weights)),
        "goal_diff_avg": float(np.average(g_diff, weights=weights)),
        "sot_diff_avg": float(np.average(sot_diff, weights=weights)),
        "corners_avg": float(np.mean([match["Corners_F"] for match in last5])),
        "clean_sheets": float(np.mean(clean_sheets)),
        "cards_avg": float(np.mean(cards)),
        "goals_scored_avg": float(np.average(goal_for, weights=weights)),
        "goals_conceded_avg": float(np.average(goal_against, weights=weights)),
        "over25_rate": float(np.mean(over25)),
        "btts_rate": float(np.mean(btts)),
    }


def build_team_history(raw_history_df, team_name, as_of_date):
    team_matches = []
    for row in raw_history_df.sort_values("Date").itertuples(index=False):
        match_date = pd.to_datetime(row.Date)
        if match_date > as_of_date:
            continue
        if row.HomeTeam == team_name:
            venue = "H"
            opponent = row.AwayTeam
            pts = 3 if row.FTR == "H" else (1 if row.FTR == "D" else 0)
            gf, ga = row.FTHG, row.FTAG
            sot_f, sot_a = row.HST, row.AST
            yellow, red = row.HY, row.HR
            corners = row.HC
        elif row.AwayTeam == team_name:
            venue = "A"
            opponent = row.HomeTeam
            pts = 3 if row.FTR == "A" else (1 if row.FTR == "D" else 0)
            gf, ga = row.FTAG, row.FTHG
            sot_f, sot_a = row.AST, row.HST
            yellow, red = row.AY, row.AR
            corners = row.AC
        else:
            continue

        team_matches.append(
            {
                "Date": match_date,
                "Opponent": opponent,
                "Venue": venue,
                "Pts": pts,
                "GF": int(gf),
                "GA": int(ga),
                "SoT_F": int(sot_f),
                "SoT_A": int(sot_a),
                "Corners_F": int(corners),
                "Yellow": int(yellow),
                "Red": int(red),
            }
        )

    return team_matches


def compute_current_elo_state(raw_history_df, team_name, as_of_date):
    elo_ratings = {}
    for team in set(raw_history_df["HomeTeam"].unique()) | set(raw_history_df["AwayTeam"].unique()):
        elo_ratings[team] = 1500.0

    for row in raw_history_df.sort_values("Date").itertuples(index=False):
        match_date = pd.to_datetime(row.Date)
        if match_date > as_of_date:
            break
        h_team = row.HomeTeam
        a_team = row.AwayTeam
        h_elo = elo_ratings.get(h_team, 1500.0)
        a_elo = elo_ratings.get(a_team, 1500.0)
        result = 1.0 if row.FTR == "H" else (0.5 if row.FTR == "D" else 0.0)
        new_h, new_a = calculate_elo(h_elo, a_elo, result)
        elo_ratings[h_team], elo_ratings[a_team] = new_h, new_a

    return elo_ratings.get(team_name, 1500.0)


def make_fixture_feature_row(raw_history_df, fixture_row):
    fixture_date = pd.to_datetime(fixture_row["Date"])
    home_team = str(fixture_row["HomeTeam"]).strip()
    away_team = str(fixture_row["AwayTeam"]).strip()

    h_history = build_team_history(raw_history_df, home_team, fixture_date)
    a_history = build_team_history(raw_history_df, away_team, fixture_date)

    h_elo = compute_current_elo_state(raw_history_df, home_team, fixture_date)
    a_elo = compute_current_elo_state(raw_history_df, away_team, fixture_date)
    elo_diff = h_elo - a_elo

    h_form = extract_form(h_history)
    a_form = extract_form(a_history)
    h_home = extract_form(h_history, venue_filter="H")
    a_away = extract_form(a_history, venue_filter="A")

    latest_h_match = h_history[-1] if h_history else None
    latest_a_match = a_history[-1] if a_history else None
    h_rest = 14 if latest_h_match is None else min((fixture_date - latest_h_match["Date"]).days, 30)
    a_rest = 14 if latest_a_match is None else min((fixture_date - latest_a_match["Date"]).days, 30)

    h2h_matches = []
    for match in h_history:
        if match["Opponent"] == away_team:
            h2h_matches.append(match)
    h2h_home_win_rate = 0.33 if not h2h_matches else sum(1 for match in h2h_matches[-5:] if match["Pts"] == 3) / len(h2h_matches[-5:])

    b365_h = safe_odd(fixture_row.get("B365H", np.nan))
    b365_d = safe_odd(fixture_row.get("B365D", np.nan))
    b365_a = safe_odd(fixture_row.get("B365A", np.nan))
    if pd.notna(b365_h) and pd.notna(b365_d) and pd.notna(b365_a):
        raw_h = 1.0 / b365_h
        raw_d = 1.0 / b365_d
        raw_a = 1.0 / b365_a
        margin = raw_h + raw_d + raw_a
        prob_h, prob_d, prob_a = raw_h / margin, raw_d / margin, raw_a / margin
    else:
        prob_h, prob_d, prob_a = 0.45, 0.25, 0.30

    home_recent = {
        "gf_avg": np.mean([match["GF"] for match in h_history[-5:]]) if h_history else 1.2,
        "ga_avg": np.mean([match["GA"] for match in h_history[-5:]]) if h_history else 1.2,
    }
    away_recent = {
        "gf_avg": np.mean([match["GF"] for match in a_history[-5:]]) if a_history else 1.2,
        "ga_avg": np.mean([match["GA"] for match in a_history[-5:]]) if a_history else 1.2,
    }
    league_home_goal_avg = max((h_form["goals_scored_avg"] + a_form["goals_conceded_avg"]) / 2.0, 1.2)
    league_away_goal_avg = max((a_form["goals_scored_avg"] + h_form["goals_conceded_avg"]) / 2.0, 1.2)
    home_attack = (home_recent["gf_avg"] + 0.5) / max(league_home_goal_avg + 0.5, 0.5)
    away_defense = (away_recent["ga_avg"] + 0.5) / max(league_away_goal_avg + 0.5, 0.5)
    away_attack = (away_recent["gf_avg"] + 0.5) / max(league_away_goal_avg + 0.5, 0.5)
    home_defense = (home_recent["ga_avg"] + 0.5) / max(league_home_goal_avg + 0.5, 0.5)
    home_xg = float(np.clip(home_attack * away_defense * (league_home_goal_avg * 1.15), 0.05, 3.2))
    away_xg = float(np.clip(away_attack * home_defense * (league_away_goal_avg * 1.10), 0.05, 3.2))
    xg_diff = home_xg - away_xg
    xg_ratio = home_xg / max(away_xg, 1e-6)

    row = {
        "h_form_pts": h_form["pts_avg"],
        "a_form_pts": a_form["pts_avg"],
        "h_form_gdiff": h_form["goal_diff_avg"],
        "a_form_gdiff": a_form["goal_diff_avg"],
        "h_form_sot_diff": h_form["sot_diff_avg"],
        "a_form_sot_diff": a_form["sot_diff_avg"],
        "h_venue_pts": h_home["pts_avg"],
        "a_venue_pts": a_away["pts_avg"],
        "Home_xG_L5": home_xg,
        "Away_xG_L5": away_xg,
        "xG_Diff": xg_diff,
        "xG_Ratio": xg_ratio,
        "elo_diff": elo_diff,
        "prob_home_market": prob_h,
        "prob_draw_market": prob_d,
        "prob_away_market": prob_a,
        "h_rest_days": h_rest,
        "a_rest_days": a_rest,
        "rest_diff": h_rest - a_rest,
        "h2h_home_win_rate": h2h_home_win_rate,
        "h_cards_avg": h_form["cards_avg"],
        "a_cards_avg": a_form["cards_avg"],
        "ref_home_bias": 0.45,
        "h_goals_scored_avg": h_form["goals_scored_avg"],
        "h_goals_conceded_avg": h_form["goals_conceded_avg"],
        "a_goals_scored_avg": a_form["goals_scored_avg"],
        "a_goals_conceded_avg": a_form["goals_conceded_avg"],
        "expected_total_goals_proxy": (
            h_form["goals_scored_avg"] + a_form["goals_conceded_avg"] + a_form["goals_scored_avg"] + h_form["goals_conceded_avg"]
        ) / 2.0,
        "h_over25_rate": h_form["over25_rate"],
        "a_over25_rate": a_form["over25_rate"],
        "h_btts_rate": h_form["btts_rate"],
        "a_btts_rate": a_form["btts_rate"],
        "h_clean_sheet_rate": h_form["clean_sheets"],
        "a_clean_sheet_rate": a_form["clean_sheets"],
    }
    return row


def build_fixture_feature_frame(raw_history_df, upcoming_fixtures_df):
    if upcoming_fixtures_df.empty:
        return pd.DataFrame()

    rows = []
    for _, fixture in upcoming_fixtures_df.iterrows():
        row = make_fixture_feature_row(raw_history_df, fixture.to_dict())
        feature_row = {**fixture.to_dict(), **row}
        rows.append(feature_row)

    feature_frame = pd.DataFrame(rows)
    return feature_frame


def fetch_football_data_uk_fixtures():
    season_candidates = resolve_active_season_candidates()
    candidates = [
        "https://www.football-data.co.uk/fixtures.csv",
        "https://www.football-data.co.uk/new/E0.csv",
    ]
    for season in season_candidates[:4]:
        candidates.append(f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv")
    candidates.extend([
        "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    ])

    for url in candidates:
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=20) as response:
                payload = response.read()
            if not payload:
                continue
            df = pd.read_csv(io.BytesIO(payload), low_memory=False)
            if df.empty:
                continue
            normalized = normalize_fixture_columns(df)
            filtered = filter_epl_upcoming_fixtures(normalized)
            if not filtered.empty:
                return filtered
        except Exception:
            continue
    return pd.DataFrame()


def normalize_fixture_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Home": "HomeTeam",
        "Away": "AwayTeam",
        "HomeTeam": "HomeTeam",
        "AwayTeam": "AwayTeam",
        "Date": "Date",
        "MatchDate": "Date",
        "FixtureDate": "Date",
        "MatchDateTime": "Date",
        "Match_Date": "Date",
        "DateTime": "Date",
        "Time": "Time",
        "MatchTime": "Time",
        "Div": "Div",
        "League": "Div",
        "B365H": "B365H",
        "B365D": "B365D",
        "B365A": "B365A",
        "BbAv>2.5": "BbAv>2.5",
        "BbAv<2.5": "BbAv<2.5",
        "BbAH": "BbAH",
        "BTTSYesOdds": "BTTSYesOdds",
        "BTTSNoOdds": "BTTSNoOdds",
        "BTTSYes": "BTTSYesOdds",
        "BttsYes": "BTTSYesOdds",
        "BTTSNo": "BTTSNoOdds",
        "BttsNo": "BTTSNoOdds",
    }

    columns = {}
    for col in df.columns:
        cleaned = str(col).strip()
        normalized = cleaned.replace(" ", "").replace("-", "").replace("_", "")
        if normalized in rename_map:
            columns[col] = rename_map[normalized]
    output = df.rename(columns=columns).copy()

    if "Date" in output.columns:
        output["Date"] = parse_date_series(output["Date"])
    if "Time" in output.columns and "Date" in output.columns:
        combined = output["Date"].astype(str).str.replace("NaT", "", regex=False) + " " + output["Time"].astype(str)
        output["Date"] = pd.to_datetime(combined, errors="coerce")
        output = output.drop(columns=["Time"])

    for col in ["HomeTeam", "AwayTeam"]:
        if col in output.columns:
            output[col] = output[col].astype(str).str.strip()
    if "Div" in output.columns:
        output["Div"] = output["Div"].astype(str).str.strip().str.upper()

    for col_name, default_value in {
        "BTTSYesOdds": 2.05,
        "BTTSNoOdds": 1.70,
    }.items():
        if col_name not in output.columns:
            output[col_name] = default_value
    return output


def filter_epl_upcoming_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    output = df.copy()
    if "Div" in output.columns:
        output = output[output["Div"].astype(str).str.strip().str.upper() == "E0"].copy()
    else:
        return pd.DataFrame()
    if output.empty:
        return output
    if "Date" in output.columns:
        output["Date"] = parse_date_series(output["Date"])
    output = output.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).copy()
    if output.empty:
        return output
    today_start = pd.Timestamp.now().normalize()
    output = output[output["Date"] >= today_start].copy()
    output = output.sort_values("Date").reset_index(drop=True)
    return output


def generate_mock_upcoming_fixtures(team_names=None, matchday_offset_days=3):
    return pd.DataFrame(columns=["Div", "Date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A", "BbAv>2.5", "BbAv<2.5", "BTTSYesOdds", "BTTSNoOdds"])


def load_upcoming_fixtures():
    if UPCOMING_FIXTURES_FILE.exists():
        tmp = pd.read_csv(UPCOMING_FIXTURES_FILE, low_memory=False)
        if not tmp.empty:
            normalized = normalize_fixture_columns(tmp)
            filtered = filter_epl_upcoming_fixtures(normalized)
            if not filtered.empty:
                print(f"Loaded {len(filtered)} fixtures from local CSV: {UPCOMING_FIXTURES_FILE.name}")
                return filtered

    fetched = fetch_football_data_uk_fixtures()
    if not fetched.empty:
        return fetched

    return pd.DataFrame()


def prepare_model_data(processed_df):
    merged = processed_df.copy()
    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
    merged = merged.sort_values("Date").reset_index(drop=True)
    merged["target_1x2"] = merged["Target_FTR"].map(TARGET_MAP)
    return merged


def as_probabilities_from_market_odds(row):
    outcome_probs = {
        "H": safe_odd(row.get("B365H", np.nan)),
        "D": safe_odd(row.get("B365D", np.nan)),
        "A": safe_odd(row.get("B365A", np.nan)),
    }
    odds = [v for v in outcome_probs.values() if pd.notna(v)]
    if len(odds) != 3:
        return {"H": 0.45, "D": 0.25, "A": 0.30}
    inv = [1.0 / x for x in odds]
    margin = sum(inv)
    return {
        "H": inv[0] / margin,
        "D": inv[1] / margin,
        "A": inv[2] / margin,
    }


def fractional_kelly(probability: float, odds: float, fraction: float) -> float:
    if pd.isna(probability) or pd.isna(odds):
        return 0.0
    if probability <= 0.0 or odds <= 1.0:
        return 0.0
    b = odds - 1.0
    if b <= 0.0:
        return 0.0
    kelly = ((b * probability) - (1.0 - probability)) / b
    return max(0.0, fraction * kelly)


def recommended_stake_for_value(probability: float, odds: float, risk_tier: str) -> float:
    fraction = STANDARD_KELLY_FRACTION if risk_tier == "Standard Value" else LONGSHOT_KELLY_FRACTION
    cap = STANDARD_KELLY_CAP if risk_tier == "Standard Value" else LONGSHOT_KELLY_CAP
    raw_fraction = fractional_kelly(probability, odds, fraction)
    raw_stake = INITIAL_BANKROLL * raw_fraction
    return min(max(raw_stake, 0.0), cap)


def edge_pct(probability: float, odds: float) -> float:
    return max(0.0, (probability * odds - 1.0) * 100.0)


def classify_risk(probability: float, odds: float):
    if probability >= 0.40 and odds <= 3.40:
        return "Standard Value"
    return "[HIGH RISK / LONGSHOT]"


def summarize_value_table(value_records):
    if value_records is None:
        return pd.DataFrame(columns=["RiskTier", "Bets", "WinRate_%", "NetProfit_EUR", "ROI_%"])
    if isinstance(value_records, pd.DataFrame):
        if value_records.empty:
            return pd.DataFrame(columns=["RiskTier", "Bets", "WinRate_%", "NetProfit_EUR", "ROI_%"])
        records = value_records
    else:
        if len(value_records) == 0:
            return pd.DataFrame(columns=["RiskTier", "Bets", "WinRate_%", "NetProfit_EUR", "ROI_%"])
        records = pd.DataFrame(value_records)

    frames = []
    for tier_name in ["Standard Value", "[HIGH RISK / LONGSHOT]"]:
        tier_df = records[records["RiskTier"] == tier_name].copy()
        if tier_df.empty:
            frames.append({
                "RiskTier": tier_name,
                "Bets": 0,
                "WinRate_%": 0.0,
                "NetProfit_EUR": 0.0,
                "ROI_%": 0.0,
            })
            continue

        tier_df["ProjectedWinRate_%"] = tier_df["ModelProbability"] * 100.0
        tier_df["ExpectedProfit_EUR"] = tier_df["ModelProbability"] * tier_df["RecommendedStakeEUR"] * (tier_df["Odds"] - 1.0) - tier_df["RecommendedStakeEUR"] * (1.0 - tier_df["ModelProbability"])

        total_bets = len(tier_df)
        win_rate = float(tier_df["ProjectedWinRate_%"].mean())
        expected_profit = float(tier_df["ExpectedProfit_EUR"].sum())
        staked = float(tier_df["RecommendedStakeEUR"].sum())
        roi = (expected_profit / staked) * 100.0 if staked > 0 else 0.0

        frames.append({
            "RiskTier": tier_name,
            "Bets": total_bets,
            "WinRate_%": win_rate,
            "NetProfit_EUR": expected_profit,
            "ROI_%": roi,
        })

    combined = records.copy()
    if combined.empty:
        combined_summary = {
            "RiskTier": "Combined",
            "Bets": 0,
            "WinRate_%": 0.0,
            "NetProfit_EUR": 0.0,
            "ROI_%": 0.0,
        }
    else:
        combined["ProjectedWinRate_%"] = combined["ModelProbability"] * 100.0
        combined["ExpectedProfit_EUR"] = combined["ModelProbability"] * combined["RecommendedStakeEUR"] * (combined["Odds"] - 1.0) - combined["RecommendedStakeEUR"] * (1.0 - combined["ModelProbability"])
        staked = float(combined["RecommendedStakeEUR"].sum())
        expected_profit = float(combined["ExpectedProfit_EUR"].sum())
        combined_summary = {
            "RiskTier": "Combined",
            "Bets": len(combined),
            "WinRate_%": float(combined["ProjectedWinRate_%"].mean()),
            "NetProfit_EUR": expected_profit,
            "ROI_%": (expected_profit / staked) * 100.0 if staked > 0 else 0.0,
        }

    frames.append(combined_summary)
    summary_df = pd.DataFrame(frames, columns=["RiskTier", "Bets", "WinRate_%", "NetProfit_EUR", "ROI_%"])
    return summary_df


def generate_value_bets(prediction_rows):
    value_records = []

    for _, row in prediction_rows.iterrows():
        home_prob = float(row["Prob_Home"])
        draw_prob = float(row["Prob_Draw"])
        away_prob = float(row["Prob_Away"])
        market_probs = {"H": home_prob, "D": draw_prob, "A": away_prob}
        market_odds = {
            "H": safe_odd(row.get("B365H", np.nan)),
            "D": safe_odd(row.get("B365D", np.nan)),
            "A": safe_odd(row.get("B365A", np.nan)),
        }

        for outcome, probability in market_probs.items():
            odd = market_odds.get(outcome)
            if pd.isna(odd):
                continue
            current_edge = edge_pct(probability, odd)
            if current_edge < MIN_EDGE_PERCENT:
                continue
            risk_tier = classify_risk(probability, odd)
            stake = recommended_stake_for_value(probability, odd, risk_tier)
            value_records.append(
                {
                    "MatchDate": row["Date"],
                    "HomeTeam": row["HomeTeam"],
                    "AwayTeam": row["AwayTeam"],
                    "Market": "1X2",
                    "Outcome": outcome,
                    "ModelProbability": float(probability),
                    "Odds": float(odd),
                    "EdgePct": float(current_edge),
                    "RiskTier": risk_tier,
                    "StakeRecommendationEUR": float(stake),
                    "RecommendedStakeEUR": float(stake),
                    "Prob_Home": home_prob,
                    "Prob_Draw": draw_prob,
                    "Prob_Away": away_prob,
                }
            )

        over_prob = float(row["Prob_Over25"])
        under_prob = 1.0 - over_prob
        over_odd = safe_odd(row.get("BbAv>2.5", np.nan))
        under_odd = safe_odd(row.get("BbAv<2.5", np.nan))
        for outcome, probability, odd in [("Over", over_prob, over_odd), ("Under", under_prob, under_odd)]:
            if pd.isna(odd):
                continue
            current_edge = edge_pct(probability, odd)
            if current_edge < MIN_EDGE_PERCENT:
                continue
            risk_tier = classify_risk(probability, odd)
            stake = recommended_stake_for_value(probability, odd, risk_tier)
            value_records.append(
                {
                    "MatchDate": row["Date"],
                    "HomeTeam": row["HomeTeam"],
                    "AwayTeam": row["AwayTeam"],
                    "Market": "Over/Under 2.5",
                    "Outcome": outcome,
                    "ModelProbability": float(probability),
                    "Odds": float(odd),
                    "EdgePct": float(current_edge),
                    "RiskTier": risk_tier,
                    "StakeRecommendationEUR": float(stake),
                    "RecommendedStakeEUR": float(stake),
                    "Prob_Over25": over_prob,
                    "Prob_Under25": under_prob,
                }
            )

        btts_yes_prob = float(row["Prob_BTTS_Yes"])
        btts_no_prob = 1.0 - btts_yes_prob
        btts_yes_odd = safe_odd(row.get("BTTSYesOdds", np.nan))
        btts_no_odd = safe_odd(row.get("BTTSNoOdds", np.nan))
        for outcome, probability, odd in [("Yes", btts_yes_prob, btts_yes_odd), ("No", btts_no_prob, btts_no_odd)]:
            if pd.isna(odd):
                continue
            current_edge = edge_pct(probability, odd)
            if current_edge < MIN_EDGE_PERCENT:
                continue
            risk_tier = classify_risk(probability, odd)
            stake = recommended_stake_for_value(probability, odd, risk_tier)
            value_records.append(
                {
                    "MatchDate": row["Date"],
                    "HomeTeam": row["HomeTeam"],
                    "AwayTeam": row["AwayTeam"],
                    "Market": "BTTS",
                    "Outcome": outcome,
                    "ModelProbability": float(probability),
                    "Odds": float(odd),
                    "EdgePct": float(current_edge),
                    "RiskTier": risk_tier,
                    "StakeRecommendationEUR": float(stake),
                    "RecommendedStakeEUR": float(stake),
                    "Prob_BTTS_Yes": btts_yes_prob,
                    "Prob_BTTS_No": btts_no_prob,
                }
            )

    return pd.DataFrame(value_records)


def print_summary_table(title: str, summary_df: pd.DataFrame):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    if summary_df.empty:
        print("No qualifying value bets available.")
        return
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


def fit_models_for_live_inference(processed_df):
    model_map = {}
    calibration_map = {}
    metric_map = {}

    market_specs = {
        "1x2": (FEATURES_1X2, processed_df["Target_FTR"].map(TARGET_MAP)),
        "OU": (FEATURES_OU, processed_df["Target_Over25"].astype(int)),
        "BTTS": (FEATURES_BTTS, processed_df["Target_BTTS"].astype(int)),
    }

    for market_name, (feature_cols, target_values) in market_specs.items():
        current_df = processed_df[feature_cols].copy()
        current_df = current_df.replace([np.inf, -np.inf], np.nan)
        current_df = current_df.fillna(current_df.median())
        model, method, metrics = fit_calibrated_model(current_df.to_numpy(), target_values.to_numpy(), market_name)
        model_map[market_name] = model
        calibration_map[market_name] = method
        metric_map[market_name] = metrics

    return model_map, calibration_map, metric_map


def predict_fixture_market(model, market_name, feature_row):
    if market_name == "1x2":
        feature_array = pd.DataFrame([feature_row], columns=FEATURES_1X2)
        probs = model.predict_proba(feature_array)
        class_order = getattr(model, "classes_", np.array([0, 1, 2]))
        class_lookup = {int(label): idx for idx, label in enumerate(class_order)}
        return {
            "Prob_Home": float(probs[0, class_lookup.get(0, 0)]),
            "Prob_Draw": float(probs[0, class_lookup.get(1, 1)]),
            "Prob_Away": float(probs[0, class_lookup.get(2, 2)]),
        }

    if market_name == "OU":
        feature_array = pd.DataFrame([feature_row], columns=FEATURES_OU)
        probs = model.predict_proba(feature_array)
        class_order = getattr(model, "classes_", np.array([0, 1]))
        class_lookup = {int(label): idx for idx, label in enumerate(class_order)}
        over_idx = class_lookup.get(1, 1 if probs.shape[1] > 1 else 0)
        under_idx = class_lookup.get(0, 0)
        return {
            "Prob_Over25": float(probs[0, over_idx]),
            "Prob_Under25": float(probs[0, under_idx]),
        }

    feature_array = pd.DataFrame([feature_row], columns=FEATURES_BTTS)
    probs = model.predict_proba(feature_array)
    class_order = getattr(model, "classes_", np.array([0, 1]))
    class_lookup = {int(label): idx for idx, label in enumerate(class_order)}
    yes_idx = class_lookup.get(1, 1 if probs.shape[1] > 1 else 0)
    no_idx = class_lookup.get(0, 0)
    return {
        "Prob_BTTS_Yes": float(probs[0, yes_idx]),
        "Prob_BTTS_No": float(probs[0, no_idx]),
    }


def score_upcoming_fixtures(raw_history_df, fixtures_df, models):
    if fixtures_df.empty:
        return pd.DataFrame()

    prediction_rows = []

    for _, fixture in fixtures_df.iterrows():
        row = fixture.to_dict()
        row["Date"] = pd.to_datetime(row["Date"], errors="coerce")
        feature_row = make_fixture_feature_row(raw_history_df, row)
        match_record = {
            "Date": row["Date"],
            "MatchDate": row["Date"],
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],
            "B365H": safe_odd(row.get("B365H", np.nan)),
            "B365D": safe_odd(row.get("B365D", np.nan)),
            "B365A": safe_odd(row.get("B365A", np.nan)),
            "BbAv>2.5": safe_odd(row.get("BbAv>2.5", np.nan)),
            "BbAv<2.5": safe_odd(row.get("BbAv<2.5", np.nan)),
            "BTTSYesOdds": safe_odd(row.get("BTTSYesOdds", np.nan)),
            "BTTSNoOdds": safe_odd(row.get("BTTSNoOdds", np.nan)),
        }

        x_1x2 = {key: feature_row.get(key, np.nan) for key in FEATURES_1X2}
        x_ou = {key: feature_row.get(key, np.nan) for key in FEATURES_OU}
        x_btts = {key: feature_row.get(key, np.nan) for key in FEATURES_BTTS}

        pred_1x2 = predict_fixture_market(models["1x2"], "1x2", x_1x2)
        pred_ou = predict_fixture_market(models["OU"], "OU", x_ou)
        pred_btts = predict_fixture_market(models["BTTS"], "BTTS", x_btts)

        match_record.update(pred_1x2)
        match_record.update(pred_ou)
        match_record.update(pred_btts)

        # add explicit edge calculations so the CSV and dashboard can display each market's EV clearly
        for label, prob_key, odds_key, edge_key in [
            ("H", "Prob_Home", "B365H", "Edge_1X2_H"),
            ("D", "Prob_Draw", "B365D", "Edge_1X2_D"),
            ("A", "Prob_Away", "B365A", "Edge_1X2_A"),
            ("Over", "Prob_Over25", "BbAv>2.5", "Edge_OU_Over"),
            ("Under", "Prob_Under25", "BbAv<2.5", "Edge_OU_Under"),
            ("Yes", "Prob_BTTS_Yes", "BTTSYesOdds", "Edge_BTTS_Yes"),
            ("No", "Prob_BTTS_No", "BTTSNoOdds", "Edge_BTTS_No"),
        ]:
            prob_val = match_record.get(prob_key)
            odd_val = safe_odd(match_record.get(odds_key, np.nan))
            if pd.isna(prob_val) or pd.isna(odd_val):
                match_record[edge_key] = np.nan
            else:
                match_record[edge_key] = float((float(prob_val) * float(odd_val) - 1.0) * 100.0)

        prediction_rows.append(match_record)

    return pd.DataFrame(prediction_rows)


def save_predictions(prediction_df):
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    if prediction_df.empty:
        empty_csv = pd.DataFrame(columns=[
            "MatchDate",
            "HomeTeam",
            "AwayTeam",
            "B365H",
            "B365D",
            "B365A",
            "BbAv>2.5",
            "BbAv<2.5",
            "BTTSYesOdds",
            "BTTSNoOdds",
            "Prob_Home",
            "Prob_Draw",
            "Prob_Away",
            "Prob_Over25",
            "Prob_Under25",
            "Prob_BTTS_Yes",
            "Prob_BTTS_No",
            "Edge_1X2_H",
            "Edge_1X2_D",
            "Edge_1X2_A",
            "Edge_OU_Over",
            "Edge_OU_Under",
            "Edge_BTTS_Yes",
            "Edge_BTTS_No",
        ])
        empty_csv.to_csv(OUTPUT_CSV, index=False)
        OUTPUT_JSON.write_text(json.dumps([], indent=2), encoding="utf-8")
        return

    ordered = [
        "MatchDate",
        "HomeTeam",
        "AwayTeam",
        "B365H",
        "B365D",
        "B365A",
        "BbAv>2.5",
        "BbAv<2.5",
        "BTTSYesOdds",
        "BTTSNoOdds",
        "Prob_Home",
        "Prob_Draw",
        "Prob_Away",
        "Prob_Over25",
        "Prob_Under25",
        "Prob_BTTS_Yes",
        "Prob_BTTS_No",
        "Edge_1X2_H",
        "Edge_1X2_D",
        "Edge_1X2_A",
        "Edge_OU_Over",
        "Edge_OU_Under",
        "Edge_BTTS_Yes",
        "Edge_BTTS_No",
    ]
    for column in ordered:
        if column not in prediction_df.columns:
            prediction_df[column] = np.nan

    output_df = prediction_df[ordered].copy()
    if "MatchDate" in output_df.columns:
        output_df["MatchDate"] = output_df["MatchDate"].apply(lambda value: value.isoformat() if hasattr(value, "isoformat") else str(value))
    output_df.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_JSON.write_text(json.dumps(output_df.to_dict(orient="records"), indent=2, default=str), encoding="utf-8")


def sync_and_prepare_data():
    RAW_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        sync_historical_data(raw_file=RAW_HISTORY_FILE, seasons=SEASON_CANDIDATES[:2], verbose=True)
    except Exception as exc:
        print(f"Warning: latest-season sync failed: {exc}")
    build_features_script = ROOT / "data_tools" / "build_features.py"
    subprocess.run([sys.executable, str(build_features_script)], cwd=str(ROOT), check=False)
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(f"Processed feature file not found after rebuild: {PROCESSED_FILE}")


def print_fixture_verification(fixtures_df):
    if fixtures_df.empty:
        return
    print("\nLIVE EPL FIXTURE VERIFICATION")
    print("-" * 160)
    preview = fixtures_df[["Date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A", "BbAv>2.5", "BbAv<2.5"]].copy()
    preview["Date"] = preview["Date"].dt.strftime("%Y-%m-%d %H:%M")
    print(preview.head(10).to_string(index=False))


def print_prediction_dashboard(scored_df, value_df):
    if scored_df.empty:
        print("No upcoming fixtures scored.")
        return

    best_by_match = {}
    if not value_df.empty:
        for _, bet in value_df.iterrows():
            key = (str(bet["HomeTeam"]), str(bet["AwayTeam"]))
            if key not in best_by_match or bet["EdgePct"] > best_by_match[key]["EdgePct"]:
                best_by_match[key] = bet.to_dict()

    rows = []
    for _, row in scored_df.iterrows():
        key = (str(row["HomeTeam"]), str(row["AwayTeam"]))
        best_bet = best_by_match.get(key)
        rows.append(
            {
                "Match": f"{row['HomeTeam']} vs {row['AwayTeam']}",
                "H%": float(row["Prob_Home"] * 100.0),
                "D%": float(row["Prob_Draw"] * 100.0),
                "A%": float(row["Prob_Away"] * 100.0),
                "Over%": float(row["Prob_Over25"] * 100.0),
                "Under%": float(row["Prob_Under25"] * 100.0),
                "BTTS Yes%": float(row["Prob_BTTS_Yes"] * 100.0),
                "BTTS No%": float(row["Prob_BTTS_No"] * 100.0),
                "Value": best_bet["RiskTier"] if best_bet is not None else "None",
                "StakeEUR": round(float(best_bet["StakeRecommendationEUR"]), 2) if best_bet is not None else 0.0,
                "Edge%": round(float(best_bet["EdgePct"]), 2) if best_bet is not None else 0.0,
            }
        )

    dashboard = pd.DataFrame(rows)
    print("\n" + "=" * 180)
    print("LIVE PREDICTION DASHBOARD")
    print("=" * 180)
    print(dashboard.to_string(index=False, formatters={
        "H%": lambda x: f"{x:.2f}",
        "D%": lambda x: f"{x:.2f}",
        "A%": lambda x: f"{x:.2f}",
        "Over%": lambda x: f"{x:.2f}",
        "Under%": lambda x: f"{x:.2f}",
        "BTTS Yes%": lambda x: f"{x:.2f}",
        "BTTS No%": lambda x: f"{x:.2f}",
        "StakeEUR": lambda x: f"{x:.2f}",
        "Edge%": lambda x: f"{x:.2f}",
    }))


def print_value_bet_table(value_df):
    if value_df.empty:
        print("No value bets met the EV threshold.")
        return

    display = value_df[["HomeTeam", "AwayTeam", "Market", "Outcome", "ModelProbability", "Odds", "EdgePct", "RiskTier", "StakeRecommendationEUR"]].copy()
    display["ModelProbability"] = display["ModelProbability"] * 100.0
    print("\n" + "=" * 160)
    print("IDENTIFIED VALUE BETS")
    print("=" * 160)
    print(display.sort_values(["EdgePct", "ModelProbability"], ascending=False).to_string(index=False, formatters={
        "ModelProbability": lambda x: f"{x:.2f}",
        "Odds": lambda x: f"{x:.2f}",
        "EdgePct": lambda x: f"{x:.2f}",
        "StakeRecommendationEUR": lambda x: f"{x:.2f}",
    }))


def main():
    parser = argparse.ArgumentParser(description="Synchronize football-data.co.uk EPL data and predict upcoming fixtures.")
    parser.add_argument("--sync", action="store_true", help="Refresh raw data from football-data.co.uk and rebuild the processed feature set before prediction.")
    args = parser.parse_args()

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    if args.sync:
        try:
            sync_and_prepare_data()
            print("Sync completed successfully and feature pipeline has been refreshed.")
        except Exception as exc:
            print(f"Warning: sync/build step failed: {exc}")

    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(f"Missing processed feature file: {PROCESSED_FILE}")
    if not RAW_HISTORY_FILE.exists():
        raise FileNotFoundError(f"Missing raw history file: {RAW_HISTORY_FILE}")

    processed_df = pd.read_csv(PROCESSED_FILE, low_memory=False)
    processed_df = prepare_model_data(processed_df)
    raw_history_df = pd.read_csv(RAW_HISTORY_FILE, encoding="latin1", low_memory=False)
    raw_history_df["Date"] = parse_date_series(raw_history_df["Date"])

    fixtures_df = load_upcoming_fixtures()
    if fixtures_df.empty:
        print("No live EPL fixtures are currently published. Checked local schedule and football-data.co.uk feed. No stale mock fixtures are being injected.")
        save_predictions(pd.DataFrame())
        return

    fixtures_df = fixtures_df.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).copy()
    fixtures_df = fixtures_df[fixtures_df["Date"] >= pd.Timestamp.now().normalize()].copy()
    if fixtures_df.empty:
        print("No live EPL fixtures are currently published for the upcoming schedule. Please refresh when fixtures are released by football-data.co.uk.")
        save_predictions(pd.DataFrame())
        return

    fixtures_df = fixtures_df.sort_values("Date").reset_index(drop=True)
    print(f"Loaded {len(fixtures_df)} upcoming EPL fixtures.")
    print_fixture_verification(fixtures_df)

    models, calibration_map, metric_map = fit_models_for_live_inference(processed_df)
    print("Calibration methods selected:")
    for market, method in calibration_map.items():
        print(f" - {market}: {method}")

    scored_df = score_upcoming_fixtures(raw_history_df, fixtures_df, models)
    if scored_df.empty:
        print("No valid fixture rows were scored.")
        save_predictions(pd.DataFrame())
        return

    value_df = generate_value_bets(scored_df)
    save_predictions(scored_df)
    print_prediction_dashboard(scored_df, value_df)
    print_value_bet_table(value_df)

    if value_df.empty:
        print("No value bets met the threshold Edge >= 4.0%.")
        return

    standard_df = value_df[value_df["RiskTier"] == "Standard Value"].copy()
    longshot_df = value_df[value_df["RiskTier"] == "[HIGH RISK / LONGSHOT]"].copy()

    print_summary_table("Standard Value bets (1/8 Kelly, capped at 1.5% bankroll)", summarize_value_table(standard_df))
    print_summary_table("High-Odds / Longshot alerts (1/16 Kelly, capped at 0.5% bankroll)", summarize_value_table(longshot_df))
    print_summary_table("Total Combined Value Bets (1X2 + Over/Under 2.5 + BTTS)", summarize_value_table(value_df))


if __name__ == "__main__":
    main()
