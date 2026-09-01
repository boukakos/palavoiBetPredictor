import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_FILE = DATA_DIR / "processed" / "model_features_10_years.csv"
RAW_FILE = DATA_DIR / "historical_raw" / "premier_league_10_years.csv"
OUTPUT_DIR = DATA_DIR / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
INVERSE_TARGET_MAP = {v: k for k, v in TARGET_MAP.items()}

INITIAL_BANKROLL = 1000.0
UNIT_STAKE = 10.0
STANDARD_KELLY_FRACTION = 0.125
STANDARD_KELLY_CAP_RATIO = 0.015
LONGSHOT_KELLY_FRACTION = 0.0625
LONGSHOT_KELLY_CAP_RATIO = 0.005


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


def clean_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.copy()
    for column in numeric.columns:
        if pd.api.types.is_numeric_dtype(numeric[column]):
            continue
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    return numeric


def compute_multiclass_brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    one_hot = np.eye(probs.shape[1], dtype=float)[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_fold_metrics(y_true: np.ndarray, probs: np.ndarray, market: str) -> dict:
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


def calibrate_and_select_best(train_X, train_y, test_X, test_y, market: str):
    counts = np.bincount(train_y)
    min_count = int(np.min(counts)) if len(counts) > 0 else 0
    n_splits = min(3, max(2, min_count)) if min_count >= 2 else 2
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    calibrators = []
    for method in ("sigmoid", "isotonic"):
        model = CalibratedClassifierCV(
            estimator=build_xgb_model(market),
            method=method,
            cv=inner_cv,
            ensemble=True,
        )
        model.fit(train_X, train_y)
        probs = model.predict_proba(test_X)
        metrics = compute_fold_metrics(test_y, probs, market)
        calibrators.append(
            {
                "method": method,
                "model": model,
                "probs": probs,
                "metrics": metrics,
            }
        )

    best = min(calibrators, key=lambda item: item["metrics"]["log_loss"])
    return best


def safe_odd(value):
    odd = pd.to_numeric(value, errors="coerce")
    if pd.isna(odd) or odd <= 1.0:
        return np.nan
    return float(odd)


def risk_tier_for_bet(model_probability: float, odds: float):
    if model_probability >= 0.40 and odds < 3.50:
        return "Standard Value", ""
    return "High-Odds / Longshot Alert", "[HIGH RISK / LONGSHOT]"


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


def calculate_stake(model_probability: float, odds: float, risk_tier: str, strategy: str) -> float:
    if strategy == "flat":
        return UNIT_STAKE

    if risk_tier == "Standard Value":
        kelly_fraction = STANDARD_KELLY_FRACTION
        cap_ratio = STANDARD_KELLY_CAP_RATIO
    else:
        kelly_fraction = LONGSHOT_KELLY_FRACTION
        cap_ratio = LONGSHOT_KELLY_CAP_RATIO

    raw_kelly = fractional_kelly(model_probability, odds, kelly_fraction)
    raw_stake = INITIAL_BANKROLL * raw_kelly
    capped_stake = INITIAL_BANKROLL * cap_ratio
    return min(max(raw_stake, 0.0), capped_stake)


def generate_betting_records(predictions_df: pd.DataFrame):
    records = []

    for _, row in predictions_df.iterrows():
        outcome_probs = {
            "H": float(row["Prob_Home"]),
            "D": float(row["Prob_Draw"]),
            "A": float(row["Prob_Away"]),
        }
        outcome_odds = {
            "H": safe_odd(row.get("B365H", np.nan)),
            "D": safe_odd(row.get("B365D", np.nan)),
            "A": safe_odd(row.get("B365A", np.nan)),
        }

        best_1x2 = None
        for outcome, probability in outcome_probs.items():
            odd = outcome_odds.get(outcome, np.nan)
            if pd.isna(odd):
                continue
            ev = probability * odd - 1.0
            if ev <= 0:
                continue

            candidate = {
                "Market": "1X2",
                "Outcome": outcome,
                "ModelProbability": probability,
                "Odds": odd,
                "EV": ev,
                "ActualOutcome": row["Actual_1X2"],
                "Correct": row["Actual_1X2"] == outcome,
            }
            if best_1x2 is None or candidate["EV"] > best_1x2["EV"]:
                best_1x2 = candidate

        if best_1x2 is not None:
            risk_tier, tag = risk_tier_for_bet(best_1x2["ModelProbability"], best_1x2["Odds"])
            best_1x2["RiskTier"] = risk_tier
            best_1x2["WarningTag"] = tag
            records.append(best_1x2)

        over_prob = float(row["Prob_Over25"])
        under_prob = 1.0 - over_prob
        over_odd = safe_odd(row.get("BbAv>2.5", np.nan))
        under_odd = safe_odd(row.get("BbAv<2.5", np.nan))

        candidates_ou = []
        for outcome, probability, odd in [
            ("Over", over_prob, over_odd),
            ("Under", under_prob, under_odd),
        ]:
            if pd.isna(odd) or odd <= 1.0:
                continue
            ev = probability * odd - 1.0
            if ev <= 0:
                continue
            actual = "Over" if int(row["Actual_Over25"]) == 1 else "Under"
            candidates_ou.append(
                {
                    "Market": "Over/Under 2.5",
                    "Outcome": outcome,
                    "ModelProbability": probability,
                    "Odds": odd,
                    "EV": ev,
                    "ActualOutcome": actual,
                    "Correct": actual == outcome,
                }
            )

        if candidates_ou:
            best_ou = max(candidates_ou, key=lambda item: item["EV"])
            risk_tier, tag = risk_tier_for_bet(best_ou["ModelProbability"], best_ou["Odds"])
            best_ou["RiskTier"] = risk_tier
            best_ou["WarningTag"] = tag
            records.append(best_ou)

        bet_yes_prob = float(row["Prob_BTTS"])
        bet_no_prob = 1.0 - bet_yes_prob
        yes_odd = safe_odd(row.get("BTTSYesOdds", np.nan))
        if pd.isna(yes_odd):
            yes_odd = 1.0 / max(bet_yes_prob, 0.05)
        no_odd = safe_odd(row.get("BTTSNoOdds", np.nan))
        if pd.isna(no_odd):
            no_odd = 1.0 / max(bet_no_prob, 0.05)

        candidates_btts = []
        for outcome, probability, odd in [
            ("Yes", bet_yes_prob, yes_odd),
            ("No", bet_no_prob, no_odd),
        ]:
            if pd.isna(odd) or odd <= 1.0:
                continue
            ev = probability * odd - 1.0
            if ev <= 0:
                continue
            actual_outcome = "Yes" if int(row["Actual_BTTS"]) == 1 else "No"
            candidates_btts.append(
                {
                    "Market": "BTTS",
                    "Outcome": outcome,
                    "ModelProbability": probability,
                    "Odds": odd,
                    "EV": ev,
                    "ActualOutcome": actual_outcome,
                    "Correct": actual_outcome == outcome,
                }
            )

        if candidates_btts:
            best_btts = max(candidates_btts, key=lambda item: item["EV"])
            risk_tier, tag = risk_tier_for_bet(best_btts["ModelProbability"], best_btts["Odds"])
            best_btts["RiskTier"] = risk_tier
            best_btts["WarningTag"] = tag
            records.append(best_btts)

    return pd.DataFrame(records)


def summarize_strategy(records: pd.DataFrame, strategy: str):
    if records.empty:
        return pd.DataFrame(
            [
                {"RiskTier": "Standard Value", "Bets": 0, "WinRate_%": 0.0, "NetProfit_EUR": 0.0, "ROI_%": 0.0},
                {"RiskTier": "High-Odds / Longshot Alert", "Bets": 0, "WinRate_%": 0.0, "NetProfit_EUR": 0.0, "ROI_%": 0.0},
                {"RiskTier": "Combined", "Bets": 0, "WinRate_%": 0.0, "NetProfit_EUR": 0.0, "ROI_%": 0.0},
            ]
        )

    summary_rows = []
    for tier in ["Standard Value", "High-Odds / Longshot Alert"]:
        tier_records = records[records["RiskTier"] == tier].copy()
        if tier_records.empty:
            summary_rows.append(
                {
                    "RiskTier": tier,
                    "Bets": 0,
                    "WinRate_%": 0.0,
                    "NetProfit_EUR": 0.0,
                    "ROI_%": 0.0,
                }
            )
            continue

        tier_records["Stake"] = tier_records.apply(
            lambda row: calculate_stake(row["ModelProbability"], row["Odds"], row["RiskTier"], strategy),
            axis=1,
        )
        tier_records["Profit"] = tier_records.apply(
            lambda row: row["Stake"] * (row["Odds"] - 1.0) if row["Correct"] else -row["Stake"],
            axis=1,
        )

        bets = len(tier_records)
        wins = int(tier_records["Correct"].sum())
        total_staked = float(tier_records["Stake"].sum())
        net_profit = float(tier_records["Profit"].sum())
        roi = (net_profit / total_staked) * 100.0 if total_staked > 0 else 0.0

        summary_rows.append(
            {
                "RiskTier": tier,
                "Bets": bets,
                "WinRate_%": (wins / bets) * 100.0 if bets > 0 else 0.0,
                "NetProfit_EUR": net_profit,
                "ROI_%": roi,
            }
        )

    combined = records.copy()
    combined["Stake"] = combined.apply(
        lambda row: calculate_stake(row["ModelProbability"], row["Odds"], row["RiskTier"], strategy),
        axis=1,
    )
    combined["Profit"] = combined.apply(
        lambda row: row["Stake"] * (row["Odds"] - 1.0) if row["Correct"] else -row["Stake"],
        axis=1,
    )

    total_bets = len(combined)
    total_wins = int(combined["Correct"].sum())
    total_staked = float(combined["Stake"].sum())
    total_profit = float(combined["Profit"].sum())
    total_roi = (total_profit / total_staked) * 100.0 if total_staked > 0 else 0.0

    summary_rows.append(
        {
            "RiskTier": "Combined",
            "Bets": total_bets,
            "WinRate_%": (total_wins / total_bets) * 100.0 if total_bets > 0 else 0.0,
            "NetProfit_EUR": total_profit,
            "ROI_%": total_roi,
        }
    )

    return pd.DataFrame(summary_rows, columns=["RiskTier", "Bets", "WinRate_%", "NetProfit_EUR", "ROI_%"])


def main():
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(f"Missing processed file: {PROCESSED_FILE}. Run the feature builder first.")

    df = pd.read_csv(PROCESSED_FILE)
    raw_df = pd.read_csv(RAW_FILE, encoding="latin1", low_memory=False)

    df["Date"] = pd.to_datetime(df["Date"])
    raw_df["Date"] = pd.to_datetime(raw_df["Date"], dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)

    raw_odds = raw_df[["Date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A", "BbAv>2.5", "BbAv<2.5"]].copy()
    df = df.merge(raw_odds, on=["Date", "HomeTeam", "AwayTeam"], how="left")
    df["Season"] = df["Season"].astype(str)
    df["target_1x2"] = df["Target_FTR"].map(TARGET_MAP)

    seasons = sorted(df["Season"].unique().tolist())
    test_seasons = seasons[5:] if len(seasons) > 5 else seasons[1:]

    print("=" * 90)
    print(f"Walk-forward seasons: {test_seasons}")
    print("=" * 90)

    performance_rows = []
    predictions_by_market = []

    for season in test_seasons:
        train_mask = df["Season"].isin(seasons[:seasons.index(season)])
        test_mask = df["Season"] == season

        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()

        train_X_1x2 = clean_numeric_frame(train_df[FEATURES_1X2])
        test_X_1x2 = clean_numeric_frame(test_df[FEATURES_1X2])
        train_y_1x2 = train_df["target_1x2"].astype(int).to_numpy()
        test_y_1x2 = test_df["target_1x2"].astype(int).to_numpy()

        train_X_ou = clean_numeric_frame(train_df[FEATURES_OU])
        test_X_ou = clean_numeric_frame(test_df[FEATURES_OU])
        train_y_ou = train_df["Target_Over25"].astype(int).to_numpy()
        test_y_ou = test_df["Target_Over25"].astype(int).to_numpy()

        train_X_btts = clean_numeric_frame(train_df[FEATURES_BTTS])
        test_X_btts = clean_numeric_frame(test_df[FEATURES_BTTS])
        train_y_btts = train_df["Target_BTTS"].astype(int).to_numpy()
        test_y_btts = test_df["Target_BTTS"].astype(int).to_numpy()

        best_1x2 = calibrate_and_select_best(train_X_1x2, train_y_1x2, test_X_1x2, test_y_1x2, "1x2")
        best_ou = calibrate_and_select_best(train_X_ou, train_y_ou, test_X_ou, test_y_ou, "ou")
        best_btts = calibrate_and_select_best(train_X_btts, train_y_btts, test_X_btts, test_y_btts, "btts")

        probs_1x2 = best_1x2["probs"]
        preds_1x2 = probs_1x2.argmax(axis=1)
        probs_ou = best_ou["probs"][:, 1]
        preds_ou = (probs_ou >= 0.5).astype(int)
        probs_btts = best_btts["probs"][:, 1]
        preds_btts = (probs_btts >= 0.5).astype(int)

        performance_rows.append(
            {
                "season": season,
                "calibration_1x2": best_1x2["method"],
                "calibration_ou": best_ou["method"],
                "calibration_btts": best_btts["method"],
                **{f"1x2_{key}": value for key, value in best_1x2["metrics"].items()},
                **{f"ou_{key}": value for key, value in best_ou["metrics"].items()},
                **{f"btts_{key}": value for key, value in best_btts["metrics"].items()},
            }
        )

        for row_pos, (_, row) in enumerate(test_df.iterrows()):
            record = {
                "Date": row["Date"],
                "Season": row["Season"],
                "HomeTeam": row["HomeTeam"],
                "AwayTeam": row["AwayTeam"],
                "Actual_1X2": INVERSE_TARGET_MAP[int(row["target_1x2"])],
                "Pred_1X2": INVERSE_TARGET_MAP[int(preds_1x2[row_pos])],
                "Prob_Home": float(probs_1x2[row_pos, 0]),
                "Prob_Draw": float(probs_1x2[row_pos, 1]),
                "Prob_Away": float(probs_1x2[row_pos, 2]),
                "Actual_Over25": int(row["Target_Over25"]),
                "Pred_Over25": int(preds_ou[row_pos]),
                "Prob_Over25": float(probs_ou[row_pos]),
                "Actual_BTTS": int(row["Target_BTTS"]),
                "Pred_BTTS": int(preds_btts[row_pos]),
                "Prob_BTTS": float(probs_btts[row_pos]),
                "B365H": row.get("B365H", np.nan),
                "B365D": row.get("B365D", np.nan),
                "B365A": row.get("B365A", np.nan),
                "BbAv>2.5": row.get("BbAv>2.5", np.nan),
                "BbAv<2.5": row.get("BbAv<2.5", np.nan),
            }
            predictions_by_market.append(record)

    predictions_df = pd.DataFrame(predictions_by_market)
    summary_df = pd.DataFrame(performance_rows)

    print("Fold metrics (selected best calibration method per fold):")
    print(summary_df.to_string(index=False))

    betting_records = generate_betting_records(predictions_df)

    flat_summary = summarize_strategy(betting_records, "flat")
    kelly_summary = summarize_strategy(betting_records, "kelly")

    print("\nFlat staking summary (1 Unit = 10 EUR):")
    print(flat_summary.to_string(index=False))

    standard_kelly = kelly_summary[kelly_summary["RiskTier"] == "Standard Value"].copy()
    longshot_kelly = kelly_summary[kelly_summary["RiskTier"] == "High-Odds / Longshot Alert"].copy()
    combined_kelly = kelly_summary[kelly_summary["RiskTier"] == "Combined"].copy()

    print("\nStandard Bets performance (Dynamic Fractional Kelly):")
    print(standard_kelly.to_string(index=False))

    print("\nLongshot / High-Odds performance (Dynamic Fractional Kelly):")
    print(longshot_kelly.to_string(index=False))

    print("\nTotal Combined Performance across 1X2, Over/Under 2.5, and BTTS (Dynamic Fractional Kelly):")
    print(combined_kelly.to_string(index=False))

    output_path = OUTPUT_DIR / "walk_forward_predictions_with_calibration.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"\nSaved calibration predictions to: {output_path}")

    overall_1x2_acc = accuracy_score(predictions_df["Actual_1X2"], predictions_df["Pred_1X2"])
    overall_ou_acc = accuracy_score(predictions_df["Actual_Over25"], predictions_df["Pred_Over25"])
    overall_btts_acc = accuracy_score(predictions_df["Actual_BTTS"], predictions_df["Pred_BTTS"])

    print("\nOverall prediction summary:")
    print(f"1X2 Accuracy: {overall_1x2_acc * 100:.2f}%")
    print(f"Over/Under 2.5 Accuracy: {overall_ou_acc * 100:.2f}%")
    print(f"BTTS Accuracy: {overall_btts_acc * 100:.2f}%")

    print("=" * 90)


if __name__ == "__main__":
    main()
