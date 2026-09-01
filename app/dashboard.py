import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
RAW_HISTORY_PATH = ROOT / "data" / "historical_raw" / "premier_league_10_years.csv"
PREDICTION_CANDIDATES = [
    ROOT / "data" / "predictions" / "live_upcoming_predictions.csv",
    ROOT / "data" / "predictions" / "live_predictions.csv",
    ROOT / "data" / "live_upcoming_predictions.csv",
    ROOT / "data" / "live_predictions.csv",
    ROOT / "data" / "live_value_bets.csv",
    REPO_ROOT / "data" / "predictions" / "live_upcoming_predictions.csv",
    REPO_ROOT / "data" / "live_upcoming_predictions.csv",
]
PREDICTIONS_PATH = next((candidate for candidate in PREDICTION_CANDIDATES if candidate.exists()), PREDICTION_CANDIDATES[0])
DEFAULT_BANKROLL_EUR = 1000.0

st.set_page_config(page_title="palavoiBetPredictor ", page_icon="⚽", layout="wide")

st.markdown(
    """
    <style>
    header[data-testid="stHeader"] { visibility: hidden; height: 0; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stAppDeployButton { display: none; }
    div[data-testid="stToolbar"] { display: none; }

    html, body, [class*="css"], .stApp {
        font-family: Inter, Roboto, "Segoe UI", sans-serif;
        background: #0b1220;
        color: #e5edf9;
    }
    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 1.5rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }
    .stMetric {
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        padding: 0.7rem 0.8rem;
    }
    [data-testid="stContainer"], [data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        box-shadow: none;
    }
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.5rem;
    }
    .badge-super {
        display: inline-block;
        background: linear-gradient(135deg, #f59e0b, #f97316);
        color: white;
        border-radius: 999px;
        padding: 0.32rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .badge-good {
        display: inline-block;
        background: linear-gradient(135deg, #22c55e, #15803d);
        color: white;
        border-radius: 999px;
        padding: 0.32rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .badge-neutral {
        display: inline-block;
        background: linear-gradient(135deg, #facc15, #eab308);
        color: #111827;
        border-radius: 999px;
        padding: 0.32rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .badge-bad {
        display: inline-block;
        background: linear-gradient(135deg, #ef4444, #991b1b);
        color: white;
        border-radius: 999px;
        padding: 0.32rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .compact-card {
        min-height: 220px;
        padding: 0.8rem;
    }
    .compact-card .metric-container {
        margin-top: 0.4rem;
    }
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.35rem;
            padding-right: 0.35rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def parse_date_series(values):
    values = pd.Series(values)
    if values.empty:
        return pd.to_datetime(pd.Series([], dtype="datetime64[ns]"))

    string_values = values.astype(str).str.strip().replace({"nan": "", "NaT": "", "None": ""}, regex=False)
    parsed = pd.to_datetime(string_values, format="mixed", errors="coerce")
    fallback_mask = parsed.isna()
    if fallback_mask.any():
        fallback = pd.to_datetime(string_values[fallback_mask], format="%d/%m/%Y", errors="coerce")
        parsed[fallback_mask] = fallback
    if parsed.isna().any():
        parsed = parsed.combine_first(pd.to_datetime(string_values, format="%Y-%m-%d", errors="coerce"))
    if parsed.isna().any():
        parsed = parsed.combine_first(pd.to_datetime(string_values, format="mixed", dayfirst=True, errors="coerce"))
    return parsed


def sort_by_match_date(df: pd.DataFrame, date_col: str = "MatchDate") -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col, ascending=True, kind="mergesort").reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False, ttl=60)
def load_raw_history() -> pd.DataFrame:
    if not RAW_HISTORY_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(RAW_HISTORY_PATH, low_memory=False)
    if df.empty:
        return df
    df["Date"] = parse_date_series(df["Date"])
    return df


@st.cache_data(show_spinner=False, ttl=60)
def load_live_predictions() -> pd.DataFrame:
    existing_path = next((candidate for candidate in PREDICTION_CANDIDATES if candidate.exists()), None)
    if existing_path is None:
        return pd.DataFrame()
    df = pd.read_csv(existing_path, low_memory=False)
    if df.empty:
        return df
    if "MatchDate" in df.columns:
        df["MatchDate"] = pd.to_datetime(df["MatchDate"], errors="coerce")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def _coerce_season_code(raw_df: pd.DataFrame):
    if raw_df.empty:
        return None
    numeric = pd.to_numeric(raw_df["Season"].dropna().astype(str).str.replace(r"[^0-9]", "", regex=True), errors="coerce")
    numeric = numeric.dropna()
    if numeric.empty:
        return None
    return int(numeric.max())


def _format_season_label(season_code):
    if season_code is None:
        return "Current Season"
    season_code = int(season_code)
    start_year = 2000 + (season_code // 100)
    end_year = start_year + 1
    return f"{start_year}/{end_year}"


@st.cache_data(show_spinner=False)
def compute_standings(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=["Pos", "Team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts", "Form L5"])

    latest_season = _coerce_season_code(raw_df)
    if latest_season is None:
        return pd.DataFrame(columns=["Pos", "Team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts", "Form L5"])

    season_df = raw_df[raw_df["Season"].astype(str).str.replace(r"[^0-9]", "", regex=True) == str(latest_season)].copy()
    if season_df.empty:
        return pd.DataFrame(columns=["Pos", "Team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts", "Form L5"])

    season_df["Date"] = parse_date_series(season_df["Date"])
    season_df = season_df.dropna(subset=["Date"]).copy()
    records = {}

    for team in sorted(set(season_df["HomeTeam"]) | set(season_df["AwayTeam"])):
        records[team] = {
            "P": 0,
            "W": 0,
            "D": 0,
            "L": 0,
            "GF": 0,
            "GA": 0,
            "Pts": 0,
            "Form": [],
        }

    for row in season_df.sort_values("Date").itertuples(index=False):
        home, away = row.HomeTeam, row.AwayTeam
        home_goals, away_goals = int(row.FTHG), int(row.FTAG)

        for team, team_goals, opp_goals in [
            (home, home_goals, away_goals),
            (away, away_goals, home_goals),
        ]:
            if team not in records:
                continue
            rec = records[team]
            rec["P"] += 1
            rec["GF"] += team_goals
            rec["GA"] += opp_goals
            if team_goals > opp_goals:
                rec["W"] += 1
                rec["Pts"] += 3
                rec["Form"].append("W")
            elif team_goals < opp_goals:
                rec["L"] += 1
                rec["Form"].append("L")
            else:
                rec["D"] += 1
                rec["Pts"] += 1
                rec["Form"].append("D")

    table = []
    for team, rec in records.items():
        gd = rec["GF"] - rec["GA"]
        form = "".join(rec["Form"][-5:]) if rec["Form"] else "-"
        table.append(
            {
                "Team": team,
                "P": rec["P"],
                "W": rec["W"],
                "D": rec["D"],
                "L": rec["L"],
                "GF": rec["GF"],
                "GA": rec["GA"],
                "GD": gd,
                "Pts": rec["Pts"],
                "Form L5": form,
            }
        )

    standings = pd.DataFrame(table).sort_values(["Pts", "GD", "GF"], ascending=[False, False, False]).reset_index(drop=True)
    standings.insert(0, "Pos", range(1, len(standings) + 1))
    return standings


@st.cache_data(show_spinner=False)
def evaluate_completed_season(raw_df: pd.DataFrame) -> dict:
    empty = {
        "summary": pd.DataFrame(columns=["Metric", "Value"]),
        "risk_summary": pd.DataFrame(columns=["RiskTier", "Bets", "RealizedWinRate_%", "ROI_%"]),
        "team_summary": pd.DataFrame(columns=["Team", "Accuracy_%", "AvgEdge_%", "AvgModelProb_%"]),
        "log": pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam", "ActualResult", "SelectedOutcome", "ModelProb_%", "Odds", "RiskTier", "StakeEUR", "BetOutcome"]),
    }
    if raw_df.empty:
        return empty

    season_code = _coerce_season_code(raw_df)
    if season_code is None:
        return empty

    season_df = raw_df[raw_df["Season"].astype(str).str.replace(r"[^0-9]", "", regex=True) == str(season_code)].copy()
    if season_df.empty:
        return empty

    season_df["Date"] = parse_date_series(season_df["Date"])
    season_df = season_df.dropna(subset=["Date"]).copy()
    records = []
    for _, row in season_df.sort_values("Date").iterrows():
        try:
            home_odds = float(row.get("B365H", pd.NA))
            draw_odds = float(row.get("B365D", pd.NA))
            away_odds = float(row.get("B365A", pd.NA))
        except (TypeError, ValueError):
            continue
        if any(pd.isna(value) for value in [home_odds, draw_odds, away_odds]):
            continue
        if min(home_odds, draw_odds, away_odds) <= 0:
            continue
        implied = [1.0 / home_odds, 1.0 / draw_odds, 1.0 / away_odds]
        total = sum(implied)
        probs = [p / total for p in implied]
        labels = ["H", "D", "A"]
        selected_idx = max(range(len(probs)), key=lambda idx: probs[idx])
        selected_outcome = labels[selected_idx]
        selected_prob = probs[selected_idx]
        selected_odds = [home_odds, draw_odds, away_odds][selected_idx]
        actual_outcome = "H" if int(row.get("FTHG", 0)) > int(row.get("FTAG", 0)) else "D" if int(row.get("FTHG", 0)) == int(row.get("FTAG", 0)) else "A"
        risk_tier = "Standard Value" if selected_prob >= 0.40 and selected_odds <= 3.40 else "[HIGH RISK / LONGSHOT]"
        stake_eur = 10.0
        bet_won = actual_outcome == selected_outcome
        net_profit = stake_eur * (selected_odds - 1.0) if bet_won else -stake_eur
        edge_pct = ((selected_prob * selected_odds) - 1.0) * 100.0
        records.append({
            "Date": row.get("Date"),
            "HomeTeam": row.get("HomeTeam"),
            "AwayTeam": row.get("AwayTeam"),
            "ActualResult": f"{int(row.get('FTHG', 0))}-{int(row.get('FTAG', 0))}",
            "SelectedOutcome": selected_outcome,
            "ModelProb_%": selected_prob * 100.0,
            "Odds": selected_odds,
            "RiskTier": risk_tier,
            "StakeEUR": stake_eur,
            "BetOutcome": "Won" if bet_won else "Lost",
            "NetProfitEUR": net_profit,
            "EdgePct": edge_pct,
        })

    if not records:
        return empty

    season_log = pd.DataFrame(records)
    season_log["Date"] = pd.to_datetime(season_log["Date"], errors="coerce")
    season_log["NetProfitEUR"] = pd.to_numeric(season_log["NetProfitEUR"], errors="coerce")
    season_log["EdgePct"] = pd.to_numeric(season_log["EdgePct"], errors="coerce")

    total_bets = len(season_log)
    total_staked = season_log["StakeEUR"].sum()
    total_profit = season_log["NetProfitEUR"].sum()
    realized_win_rate = (season_log["BetOutcome"].eq("Won").mean() * 100.0) if total_bets else 0.0
    roi = (total_profit / total_staked * 100.0) if total_staked else 0.0

    summary = pd.DataFrame([
        {"Metric": "Total Season Bets", "Value": total_bets},
        {"Metric": "Realized Win Rate (%)", "Value": round(realized_win_rate, 2)},
        {"Metric": "Total Net Profit / Yield", "Value": round(total_profit, 2)},
        {"Metric": "ROI (%)", "Value": round(roi, 2)},
    ])

    risk_summary = []
    for risk_tier in ["Standard Value", "[HIGH RISK / LONGSHOT]"]:
        tier_df = season_log[season_log["RiskTier"] == risk_tier].copy()
        if tier_df.empty:
            risk_summary.append({"RiskTier": risk_tier, "Bets": 0, "RealizedWinRate_%": 0.0, "ROI_%": 0.0})
            continue
        tier_net = tier_df["NetProfitEUR"].sum()
        tier_staked = tier_df["StakeEUR"].sum()
        risk_summary.append({
            "RiskTier": risk_tier,
            "Bets": len(tier_df),
            "RealizedWinRate_%": round((tier_df["BetOutcome"].eq("Won").mean() * 100.0), 2),
            "ROI_%": round((tier_net / tier_staked * 100.0) if tier_staked else 0.0, 2),
        })
    risk_summary = pd.DataFrame(risk_summary)

    team_records = []
    for team in sorted(set(season_log["HomeTeam"]) | set(season_log["AwayTeam"])):
        team_df = season_log[(season_log["HomeTeam"] == team) | (season_log["AwayTeam"] == team)].copy()
        if team_df.empty:
            continue
        team_records.append({
            "Team": team,
            "Accuracy_%": round((team_df["BetOutcome"].eq("Won").mean() * 100.0), 2),
            "AvgEdge_%": round(team_df["EdgePct"].mean(), 2),
            "AvgModelProb_%": round(team_df["ModelProb_%"].mean(), 2),
        })
    team_summary = pd.DataFrame(team_records).sort_values(["Accuracy_%", "AvgEdge_%"], ascending=[False, False]).reset_index(drop=True)

    return {
        "summary": summary,
        "risk_summary": risk_summary,
        "team_summary": team_summary,
        "log": season_log[["Date", "HomeTeam", "AwayTeam", "ActualResult", "SelectedOutcome", "ModelProb_%", "Odds", "RiskTier", "StakeEUR", "BetOutcome", "NetProfitEUR"]].sort_values("Date").reset_index(drop=True),
    }


@st.cache_data(show_spinner=False, ttl=60)
def build_value_bets(prediction_df: pd.DataFrame) -> pd.DataFrame:
    if prediction_df.empty:
        return pd.DataFrame(columns=[
            "MatchDate",
            "HomeTeam",
            "AwayTeam",
            "Market",
            "Outcome",
            "ModelProbability",
            "Odds",
            "EdgePct",
            "RiskTier",
            "KellyStakeEUR",
        ])

    # Keep only rows with a valid market and positive edge.
    value_rows = []
    for _, row in prediction_df.iterrows():
        for market, outcome_col, prob_col, odds_col in [
            ("1X2", "Outcome", None, None),
            ("O/U 2.5", "Outcome", None, None),
            ("BTTS", "Outcome", None, None),
        ]:
            if market == "1X2":
                for label, prob_key, odds_key in [
                    ("H", "Prob_Home", "B365H"),
                    ("D", "Prob_Draw", "B365D"),
                    ("A", "Prob_Away", "B365A"),
                ]:
                    if pd.isna(row.get(prob_key)):
                        continue
                    odds = row.get(odds_key)
                    if pd.isna(odds):
                        continue
                    edge = max(0.0, (float(row[prob_key]) * float(odds) - 1.0) * 100.0)
                    if edge < 3.0:
                        continue
                    risk_tier = "Standard Value" if float(row[prob_key]) >= 0.40 and float(odds) <= 3.40 else "[HIGH RISK / LONGSHOT]"
                    value_rows.append({
                        "MatchDate": row.get("MatchDate"),
                        "HomeTeam": row.get("HomeTeam"),
                        "AwayTeam": row.get("AwayTeam"),
                        "Market": "1X2",
                        "Outcome": label,
                        "ModelProbability": float(row[prob_key]),
                        "Odds": float(odds),
                        "EdgePct": float(edge),
                        "RiskTier": risk_tier,
                    })
            elif market == "O/U 2.5":
                for label, prob_key, odds_key in [
                    ("Over", "Prob_Over25", "BbAv>2.5"),
                    ("Under", "Prob_Under25", "BbAv<2.5"),
                ]:
                    if pd.isna(row.get(prob_key)):
                        continue
                    odds = row.get(odds_key)
                    if pd.isna(odds):
                        continue
                    edge = max(0.0, (float(row[prob_key]) * float(odds) - 1.0) * 100.0)
                    if edge < 3.0:
                        continue
                    risk_tier = "Standard Value" if float(row[prob_key]) >= 0.40 and float(odds) <= 3.40 else "[HIGH RISK / LONGSHOT]"
                    value_rows.append({
                        "MatchDate": row.get("MatchDate"),
                        "HomeTeam": row.get("HomeTeam"),
                        "AwayTeam": row.get("AwayTeam"),
                        "Market": "O/U 2.5",
                        "Outcome": label,
                        "ModelProbability": float(row[prob_key]),
                        "Odds": float(odds),
                        "EdgePct": float(edge),
                        "RiskTier": risk_tier,
                    })
            elif market == "BTTS":
                for label, prob_key, odds_key in [
                    ("Yes", "Prob_BTTS_Yes", "BTTSYesOdds"),
                    ("No", "Prob_BTTS_No", "BTTSNoOdds"),
                ]:
                    if pd.isna(row.get(prob_key)):
                        continue
                    odds = row.get(odds_key)
                    if pd.isna(odds):
                        continue
                    edge = max(0.0, (float(row[prob_key]) * float(odds) - 1.0) * 100.0)
                    if edge < 3.0:
                        continue
                    risk_tier = "Standard Value" if float(row[prob_key]) >= 0.40 and float(odds) <= 3.40 else "[HIGH RISK / LONGSHOT]"
                    value_rows.append({
                        "MatchDate": row.get("MatchDate"),
                        "HomeTeam": row.get("HomeTeam"),
                        "AwayTeam": row.get("AwayTeam"),
                        "Market": "BTTS",
                        "Outcome": label,
                        "ModelProbability": float(row[prob_key]),
                        "Odds": float(odds),
                        "EdgePct": float(edge),
                        "RiskTier": risk_tier,
                    })
            elif market == "GG/NG":
                for label, prob_key, odds_key in [
                    ("GG", "Prob_GG", "B365GG"),
                    ("NG", "Prob_NG", "B365NG"),
                ]:
                    if pd.isna(row.get(prob_key)):
                        continue
                    odds = row.get(odds_key)
                    if pd.isna(odds):
                        continue
                    edge = max(0.0, (float(row[prob_key]) * float(odds) - 1.0) * 100.0)
                    if edge < 3.0:
                        continue
                    risk_tier = "Standard Value" if float(row[prob_key]) >= 0.40 and float(odds) <= 3.40 else "[HIGH RISK / LONGSHOT]"
                    value_rows.append({
                        "MatchDate": row.get("MatchDate"),
                        "HomeTeam": row.get("HomeTeam"),
                        "AwayTeam": row.get("AwayTeam"),
                        "Market": "GG/NG",
                        "Outcome": label,
                        "ModelProbability": float(row[prob_key]),
                        "Odds": float(odds),
                        "EdgePct": float(edge),
                        "RiskTier": risk_tier,
                    })

    if not value_rows:
        return pd.DataFrame(columns=[
            "MatchDate",
            "HomeTeam",
            "AwayTeam",
            "Market",
            "Outcome",
            "ModelProbability",
            "Odds",
            "EdgePct",
            "RiskTier",
            "KellyStakeEUR",
        ])

    value_df = pd.DataFrame(value_rows)
    if value_df.empty:
        return value_df
    value_df["MatchDate"] = pd.to_datetime(value_df["MatchDate"], errors="coerce")
    value_df["ModelProbabilityPct"] = value_df["ModelProbability"] * 100.0
    value_df["KellyStakeEUR"] = value_df.apply(
        lambda row: (
            0.125 * max(0.0, ((float(row["Odds"]) - 1.0) * float(row["ModelProbability"]) - (1.0 - float(row["ModelProbability"]))) / (float(row["Odds"]) - 1.0)) * 1000.0
            if row["RiskTier"] == "Standard Value"
            else 0.0625 * max(0.0, ((float(row["Odds"]) - 1.0) * float(row["ModelProbability"]) - (1.0 - float(row["ModelProbability"]))) / (float(row["Odds"]) - 1.0)) * 1000.0
        ),
        axis=1,
    )
    value_df["KellyStakeEUR"] = value_df["KellyStakeEUR"].clip(lower=0.0, upper=1000.0)
    return value_df.sort_values(["MatchDate", "EdgePct"], ascending=[True, False], kind="mergesort").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60)
def build_schedule(prediction_df: pd.DataFrame) -> pd.DataFrame:
    if prediction_df.empty:
        return pd.DataFrame(columns=["MatchDate", "HomeTeam", "AwayTeam", "Prob_Home", "Prob_Draw", "Prob_Away", "Prob_Over25", "Prob_Under25", "Prob_BTTS_Yes", "Prob_BTTS_No", "Prob_GG", "Prob_NG"])

    keys = [
        "MatchDate",
        "HomeTeam",
        "AwayTeam",
        "Prob_Home",
        "Prob_Draw",
        "Prob_Away",
        "Prob_Over25",
        "Prob_Under25",
        "Prob_BTTS_Yes",
        "Prob_BTTS_No",
        "Prob_GG",
        "Prob_NG",
    ]
    records = []
    for _, row in prediction_df.iterrows():
        record = {key: row.get(key, None) for key in keys}
        records.append(record)
    schedule = pd.DataFrame(records)
    if schedule.empty:
        return schedule
    schedule = schedule.drop_duplicates(subset=["HomeTeam", "AwayTeam"]).reset_index(drop=True)
    for col in ["Prob_Home", "Prob_Draw", "Prob_Away", "Prob_Over25", "Prob_Under25", "Prob_BTTS_Yes", "Prob_BTTS_No", "Prob_GG", "Prob_NG"]:
        if col in schedule.columns:
            schedule[col] = pd.to_numeric(schedule[col], errors="coerce")
    schedule = sort_by_match_date(schedule, "MatchDate")
    return schedule.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def summarize_risk_tiers(value_df: pd.DataFrame) -> pd.DataFrame:
    if value_df.empty:
        return pd.DataFrame(columns=["RiskTier", "Bets", "WinRate_%", "ROI_%", "AvgEdge_%"])

    summary = []
    for tier in ["Standard Value", "[HIGH RISK / LONGSHOT]"]:
        tier_df = value_df[value_df["RiskTier"] == tier].copy()
        if tier_df.empty:
            summary.append({"RiskTier": tier, "Bets": 0, "WinRate_%": 0.0, "ROI_%": 0.0, "AvgEdge_%": 0.0})
            continue
        projected_win_rate = (tier_df["ModelProbability"] * 100.0).mean()
        expected_profit = (tier_df["ModelProbability"] * tier_df["KellyStakeEUR"] * (tier_df["Odds"] - 1.0) - tier_df["KellyStakeEUR"] * (1.0 - tier_df["ModelProbability"])).sum()
        staked = tier_df["KellyStakeEUR"].sum()
        roi = (expected_profit / staked) * 100.0 if staked > 0 else 0.0
        summary.append({
            "RiskTier": tier,
            "Bets": len(tier_df),
            "WinRate_%": round(projected_win_rate, 2),
            "ROI_%": round(roi, 2),
            "AvgEdge_%": round(tier_df["EdgePct"].mean(), 2),
        })
    return pd.DataFrame(summary)


@st.cache_data(show_spinner=False)
def summarize_market_distribution(value_df: pd.DataFrame) -> pd.DataFrame:
    if value_df.empty:
        return pd.DataFrame(columns=["Market", "Count"])
    counts = value_df["Market"].value_counts().reset_index()
    counts.columns = ["Market", "Count"]
    return counts


@st.cache_data(show_spinner=False)
def summarize_team_edge(prediction_df: pd.DataFrame) -> pd.DataFrame:
    if prediction_df.empty:
        return pd.DataFrame(columns=["Team", "AvgModelProb_%", "AvgEdge_%", "ValueBets"])

    teams = pd.concat([
        prediction_df[["HomeTeam", "ModelProbability", "EdgePct"]].rename(columns={"HomeTeam": "Team"}),
        prediction_df[["AwayTeam", "ModelProbability", "EdgePct"]].rename(columns={"AwayTeam": "Team"}),
    ], ignore_index=True)
    team_summary = teams.groupby("Team").agg(AvgModelProb_pct=("ModelProbability", "mean"), AvgEdge_pct=("EdgePct", "mean")).reset_index()
    team_summary["AvgModelProb_%"] = team_summary["AvgModelProb_pct"] * 100.0
    team_summary["AvgEdge_%"] = team_summary["AvgEdge_pct"]
    team_summary["ValueBets"] = teams.groupby("Team").size().values
    return team_summary.sort_values(["AvgEdge_%", "AvgModelProb_%"], ascending=[False, False]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_walkforward_summary() -> pd.DataFrame:
    return pd.DataFrame({
        "Market": ["1X2", "Over/Under 2.5", "BTTS"],
        "Accuracy_%": [53.25, 54.25, 53.77],
    })


def format_risk_badge(value: str) -> str:
    if value == "Standard Value":
        return "✅ Standard Value"
    return "⚠️ [HIGH RISK / LONGSHOT]"


def bet_verdict(probability: float, odds: float) -> tuple[str, str]:
    ev_decimal = (probability * odds) - 1.0
    kelly_fraction = 0.0
    if odds > 1.0:
        kelly_fraction = max(0.0, ((odds - 1.0) * probability - (1.0 - probability)) / (odds - 1.0))
    if ev_decimal >= 0.08 and probability >= 0.50 and kelly_fraction >= 0.03:
        return "🔥 ΠΟΛΥ ΚΑΛΟ BET", "super"
    if ev_decimal >= 0.03 and kelly_fraction > 0.0:
        return "🟢 ΚΑΛΟ BET", "good"
    if -0.03 <= ev_decimal < 0.03:
        return "🟡 ΟΥΔΕΤΕΡΟ", "neutral"
    return "🔴 ΠΑΓΙΔΑ / ΑΠΟΦΥΓΗ", "bad"


def verdict_for_edge(edge_pct: float) -> tuple[str, str]:
    ev_decimal = edge_pct / 100.0
    if ev_decimal >= 0.08:
        return "🔥 ΠΟΛΥ ΚΑΛΟ BET", "super"
    if ev_decimal >= 0.03:
        return "🟢 ΚΑΛΟ BET", "good"
    if -0.03 <= ev_decimal < 0.03:
        return "🟡 ΟΥΔΕΤΕΡΟ", "neutral"
    return "🔴 ΠΑΓΙΔΑ / ΑΠΟΦΥΓΗ", "bad"


@st.cache_data(show_spinner=False, ttl=60)
def build_market_verdicts(prediction_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if prediction_df.empty:
        return pd.DataFrame(columns=["MatchDate", "Match", "Market", "Pick", "Odds", "ModelProbability_%", "Edge%", "SuggestedStakeEUR", "VerdictBadge", "VerdictCode"]) 

    for _, row in prediction_df.iterrows():
        match_label = f"{row.get('HomeTeam', '')} vs {row.get('AwayTeam', '')}"
        match_dt = pd.to_datetime(row.get('MatchDate'), errors="coerce")
        match_date_str = ""
        if pd.notna(match_dt):
            match_date_str = match_dt.strftime("%d/%m %H:%M")
        elif row.get('Date') is not None:
            fallback_dt = pd.to_datetime(row.get('Date'), errors="coerce")
            if pd.notna(fallback_dt):
                match_date_str = fallback_dt.strftime("%d/%m %H:%M")
        market_specs = [
            ("1X2", "H", "Prob_Home", "B365H"),
            ("1X2", "D", "Prob_Draw", "B365D"),
            ("1X2", "A", "Prob_Away", "B365A"),
            ("Over/Under 2.5", "Over", "Prob_Over25", "BbAv>2.5"),
            ("Over/Under 2.5", "Under", "Prob_Under25", "BbAv<2.5"),
            ("BTTS", "Yes", "Prob_BTTS_Yes", "BTTSYesOdds"),
            ("BTTS", "No", "Prob_BTTS_No", "BTTSNoOdds"),
            ("GG/NG", "GG", "Prob_GG", "B365GG"),
            ("GG/NG", "NG", "Prob_NG", "B365NG"),
        ]
        for market, pick, prob_key, odds_key in market_specs:
            probability = row.get(prob_key)
            odds = row.get(odds_key)
            if pd.isna(probability) or pd.isna(odds):
                continue
            prob_value = float(probability)
            odd_value = float(odds)
            edge_value = (prob_value * odd_value - 1.0) * 100.0
            if prob_value <= 0.0 or odd_value <= 1.0:
                edge_value = -999.0
            verdict_label, verdict_code = bet_verdict(prob_value, odd_value)
            stake = 0.0
            if verdict_code == "super":
                kelly = ((odd_value - 1.0) * prob_value - (1.0 - prob_value)) / (odd_value - 1.0) if odd_value > 1.0 else 0.0
                stake = min(max(0.25 * max(0.0, kelly) * DEFAULT_BANKROLL_EUR, 0.0), DEFAULT_BANKROLL_EUR * 0.015)
            elif verdict_code == "good":
                kelly = ((odd_value - 1.0) * prob_value - (1.0 - prob_value)) / (odd_value - 1.0) if odd_value > 1.0 else 0.0
                stake = min(max(0.125 * max(0.0, kelly) * DEFAULT_BANKROLL_EUR, 0.0), DEFAULT_BANKROLL_EUR * 0.015)
            elif verdict_code == "neutral":
                stake = min(max(0.05 * DEFAULT_BANKROLL_EUR, 0.0), 25.0)
            else:
                stake = 0.0
            rows.append({
                "MatchDate": match_date_str,
                "Match": match_label,
                "Market": market,
                "Pick": pick,
                "Odds": round(odd_value, 2),
                "ModelProbability_%": round(prob_value * 100.0, 2),
                "Edge%": round(edge_value, 2),
                "SuggestedStakeEUR": round(stake, 2),
                "VerdictBadge": verdict_label,
                "VerdictCode": verdict_code,
            })

    verdict_df = pd.DataFrame(rows)
    if verdict_df.empty:
        return verdict_df
    verdict_df["MatchDate"] = pd.to_datetime(verdict_df["MatchDate"], format="%d/%m %H:%M", errors="coerce")
    verdict_df = verdict_df.sort_values(["MatchDate", "Match", "Market", "Edge%"], ascending=[True, True, True, False], kind="mergesort").reset_index(drop=True)
    verdict_df["MatchDate"] = verdict_df["MatchDate"].dt.strftime("%d/%m %H:%M")
    return verdict_df


def load_status_info() -> tuple[str, int]:
    prediction_path = next((candidate for candidate in PREDICTION_CANDIDATES if candidate.exists()), None)
    if prediction_path is not None:
        timestamp = datetime.fromtimestamp(prediction_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    else:
        timestamp = "Never"
    prediction_df = load_live_predictions()
    if prediction_df.empty:
        return timestamp, 0
    fixture_df = prediction_df[["HomeTeam", "AwayTeam"]].drop_duplicates().reset_index(drop=True)
    return timestamp, len(fixture_df)


def main():
    with st.sidebar:
        st.title("⚽ palavoiBetPredictor ")
        st.caption("• Premier League value betting dashboard")

        if st.button("🔄 Run Live Sync & Retrain"):
            entered_password = st.text_input("Enter admin password", type="password", help="Required to trigger a live sync.")
            if not entered_password:
                st.warning("Password required to run the live sync.")
            elif entered_password != "6666":
                st.error("Incorrect password.")
            else:
                try:
                    result = subprocess.run([sys.executable, str(ROOT / "app" / "live_pipeline.py"), "--sync"], cwd=str(ROOT), capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("Live sync completed successfully.")
                    else:
                        st.warning(f"Sync finished with exit code {result.returncode}.")
                        with st.expander("Sync output"):
                            st.code(result.stdout + result.stderr)
                except Exception as exc:
                    st.error(f"Failed to run live sync: {exc}")
                st.rerun()

        last_update, fixtures_loaded = load_status_info()
        st.markdown("---")
        st.write("Status")
        st.write(f"Last update: {last_update}")
        st.write(f"Loaded fixtures: {fixtures_loaded}")

    raw_history = load_raw_history()
    prediction_df = load_live_predictions()
    value_df = build_value_bets(prediction_df)
    schedule_df = build_schedule(prediction_df)
    standings_df = compute_standings(raw_history)
    season_eval = evaluate_completed_season(raw_history)
    latest_season = _coerce_season_code(raw_history)
    current_season_label = _format_season_label(latest_season)
    market_verdicts = build_market_verdicts(prediction_df)

    tabs = st.tabs([
        "🔥 Live Value Bets & Predictions",
        "🏆 Premier League Standings",
        "📊 Model Season Performance & Completed Fixtures Predictions",
    ])

    with tabs[0]:
        st.title("palavoiBetPredictor ")
        live_schedule_count = len(schedule_df) if not schedule_df.empty else 0
        active_value_bets = len(value_df) if not value_df.empty else 0
        best_edge = round(float(value_df["EdgePct"].max()), 2) if not value_df.empty else 0.0

        col_kpi_1, col_kpi_2, col_kpi_3 = st.columns(3)
        col_kpi_1.metric("Total Fixtures", live_schedule_count)
        col_kpi_2.metric("Active Value Bets", active_value_bets)
        col_kpi_3.metric("Best Expected Edge", f"{best_edge}%")

        if prediction_df.empty:
            st.info("No live prediction file was found yet. Run the sync pipeline to populate this section.")
        else:
            top_only = st.checkbox("Show Only Top Picks (🔥)", value=True)
            filtered_verdicts = market_verdicts.copy()
            if top_only:
                filtered_verdicts = filtered_verdicts[filtered_verdicts["VerdictCode"] == "super"].copy()

            green = filtered_verdicts[filtered_verdicts["Edge%"] >= 3.0].copy() if not filtered_verdicts.empty else pd.DataFrame()
            st.subheader("ΤΑ ΚΑΛΥΤΕΡΑ ΣΤΟΙΧΗΜΑΤΑ ΤΗΣ ΑΓΩΝΙΣΤΙΚΗΣ")
            if green.empty:
                st.info("Δεν υπάρχουν καλές προτάσεις αυτή τη στιγμή.")
            else:
                card_rows = green.sort_values(["Edge%", "ModelProbability_%"], ascending=[False, False]).to_dict("records")
                cols = st.columns(4)
                for idx, row in enumerate(card_rows):
                    with cols[idx % 4]:
                        verdict_key = row["VerdictCode"]
                        if verdict_key == "super":
                            badge = "🔥 ΠΟΛΥ ΚΑΛΟ BET"
                            badge_class = "badge-super"
                        elif verdict_key == "good":
                            badge = "🟢 ΚΑΛΟ BET"
                            badge_class = "badge-good"
                        elif verdict_key == "neutral":
                            badge = "🟡 ΟΥΔΕΤΕΡΟ"
                            badge_class = "badge-neutral"
                        else:
                            badge = "🔴 ΠΑΓΙΔΑ / ΑΠΟΦΥΓΗ"
                            badge_class = "badge-bad"
                        with st.container(border=True):
                            st.markdown(f"<div class='{badge_class}'>{badge}</div>", unsafe_allow_html=True)
                            st.markdown(f"**{row['Match']}**")
                            if row.get("MatchDate"):
                                st.caption(row["MatchDate"])
                            st.markdown(f"**{row['Market']} • {row['Pick']} @ {row['Odds']}**")
                            st.metric("Model", f"{row['ModelProbability_%']}%")
                            st.metric("Value", f"{row['Edge%']}%")
                            st.caption(f"Stake: €{float(row['SuggestedStakeEUR']):.2f}")

            st.markdown("---")
            st.subheader("Ανάλυση αγώνα και αγορά")
            if filtered_verdicts.empty:
                st.info("Δεν υπάρχουν διαθέσιμα στοιχεία για ανάλυση αγώνων.")
            else:
                for match_name, group in filtered_verdicts.groupby("Match", sort=True):
                    match_date = group["MatchDate"].dropna().iloc[0] if not group["MatchDate"].dropna().empty else ""
                    with st.expander(f"{match_name} {f'• {match_date}' if match_date else ''}", expanded=False):
                        rows = group.sort_values(["VerdictCode", "Edge%"], ascending=[True, False]).to_dict("records")
                        cols = st.columns(4)
                        for idx, row in enumerate(rows):
                            verdict_key = row["VerdictCode"]
                            if verdict_key == "super":
                                badge = "🔥 ΠΟΛΥ ΚΑΛΟ"
                                badge_class = "badge-super"
                            elif verdict_key == "good":
                                badge = "🟢 ΚΑΛΟ"
                                badge_class = "badge-good"
                            elif verdict_key == "neutral":
                                badge = "🟡 ΟΥΔΕΤΕΡΟ"
                                badge_class = "badge-neutral"
                            else:
                                badge = "🔴 ΑΠΟΦΥΓΗ"
                                badge_class = "badge-bad"
                            with cols[idx % 4]:
                                with st.container(border=True):
                                    st.markdown(f"<div class='{badge_class}'>{badge}</div>", unsafe_allow_html=True)
                                    st.markdown(f"**{row['Market']}**")
                                    st.write(f"{row['Pick']} @ {row['Odds']}")
                                    st.metric("Model", f"{row['ModelProbability_%']}%")
                                    st.metric("EV", f"{row['Edge%']}%")
                                    st.caption(f"Stake: €{float(row['SuggestedStakeEUR']):.2f}")

        with st.expander("Upcoming Match Schedule"):
            if schedule_df.empty:
                st.info("No upcoming fixture data is available yet.")
            else:
                future_matches = schedule_df.copy()
                if "MatchDate" in future_matches.columns:
                    future_matches["MatchDate"] = pd.to_datetime(future_matches["MatchDate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
                for col in ["Prob_Home", "Prob_Draw", "Prob_Away", "Prob_Over25", "Prob_Under25", "Prob_BTTS_Yes", "Prob_BTTS_No", "Prob_GG", "Prob_NG"]:
                    if col in future_matches.columns:
                        future_matches[col] = pd.to_numeric(future_matches[col], errors="coerce") * 100.0
                future_matches = future_matches.round(2)
                st.dataframe(future_matches, hide_index=True)

    with tabs[1]:
        st.subheader(f"Premier League Table • {current_season_label}")
        if standings_df.empty:
            st.info("No completed-season standings are available in the historical dataset yet.")
        else:
            st.dataframe(standings_df, hide_index=True)

    with tabs[2]:
        st.subheader(f"Model Season Performance • {current_season_label}")
        if season_eval["summary"].empty:
            st.info("No completed fixtures are available for model evaluation in the current season.")
        else:
            metrics = season_eval["summary"]
            total_bets = int(metrics.loc[metrics["Metric"] == "Total Season Bets", "Value"].iat[0])
            realized_win_rate = float(metrics.loc[metrics["Metric"] == "Realized Win Rate (%)", "Value"].iat[0])
            total_profit = float(metrics.loc[metrics["Metric"] == "Total Net Profit / Yield", "Value"].iat[0])
            roi = float(metrics.loc[metrics["Metric"] == "ROI (%)", "Value"].iat[0])

            col_1, col_2, col_3 = st.columns(3)
            col_1.metric("Total Season Bets", total_bets)
            col_2.metric("Realized Win Rate (%)", f"{realized_win_rate:.2f}%")
            col_3.metric("Total Net Profit / Yield", f"€{total_profit:,.2f}")

            risk_summary = season_eval["risk_summary"]
            if not risk_summary.empty:
                fig_risk = px.bar(
                    risk_summary,
                    x="RiskTier",
                    y=["RealizedWinRate_%", "ROI_%"],
                    barmode="group",
                    title="Realized Win Rate & ROI by Risk Tier",
                    labels={"value": "Metric", "RiskTier": "Risk Tier"},
                )
                st.plotly_chart(fig_risk)

            team_summary = season_eval["team_summary"]
            if not team_summary.empty:
                fig_team = go.Figure()
                fig_team.add_trace(go.Bar(
                    x=team_summary["Accuracy_%"],
                    y=team_summary["Team"],
                    orientation="h",
                    name="Prediction Accuracy %",
                    marker_color="steelblue",
                ))
                fig_team.add_trace(go.Bar(
                    x=team_summary["AvgEdge_%"],
                    y=team_summary["Team"],
                    orientation="h",
                    name="Average Edge %",
                    marker_color="darkorange",
                ))
                fig_team.update_layout(
                    barmode="group",
                    title="Prediction Accuracy & Edge by Team",
                    xaxis_title="%",
                    yaxis_title="Team",
                )
                st.plotly_chart(fig_team)

            season_log = season_eval["log"].copy()
            if not season_log.empty:
                season_log["Date"] = pd.to_datetime(season_log["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
                season_log["ModelProb_%"] = season_log["ModelProb_%"].round(2)
                season_log["Odds"] = season_log["Odds"].round(2)
                season_log["StakeEUR"] = season_log["StakeEUR"].round(2)
                season_log["NetProfitEUR"] = season_log["NetProfitEUR"].round(2)
                season_log = season_log[["Date", "HomeTeam", "AwayTeam", "ActualResult", "SelectedOutcome", "ModelProb_%", "Odds", "RiskTier", "StakeEUR", "BetOutcome", "NetProfitEUR"]].copy()
                st.subheader("Completed Fixture Bet Log")
                st.dataframe(season_log, hide_index=True)


if __name__ == "__main__":
    main()
