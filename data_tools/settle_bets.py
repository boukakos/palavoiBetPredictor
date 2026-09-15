from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_HISTORY_FILE = ROOT / "data" / "historical_raw" / "premier_league_10_years.csv"
BETS_LOG_FILE = ROOT / "data" / "predictions" / "bets_log.csv"


def settle_bets(raw_file: Path = RAW_HISTORY_FILE, ledger_file: Path = BETS_LOG_FILE, verbose: bool = True) -> int:
    """Fill in ActualResult/BetOutcome for any ledger row whose match now has
    a completed result in the raw history file. Mirrors the same H/D/A- and
    total-goals-vs-2.5 comparison already used for the season backtest in
    dashboard.py's evaluate_completed_season(), extended here to O/U 2.5.
    Rows whose match hasn't been played/synced yet are left pending and
    retried on the next run."""
    if not ledger_file.exists():
        if verbose:
            print(f"No bets ledger found at {ledger_file}; nothing to settle.")
        return 0

    ledger = pd.read_csv(ledger_file, low_memory=False)
    if ledger.empty:
        if verbose:
            print("Bets ledger is empty; nothing to settle.")
        return 0

    ledger["BetOutcome"] = ledger["BetOutcome"].fillna("")
    ledger["ActualResult"] = ledger["ActualResult"].fillna("")
    unsettled_mask = ledger["BetOutcome"].astype(str).str.strip().eq("")
    if not unsettled_mask.any():
        if verbose:
            print("No unsettled bets in the ledger.")
        return 0

    if not raw_file.exists():
        if verbose:
            print(f"Raw history file not found: {raw_file}")
        return 0

    raw_df = pd.read_csv(raw_file, encoding="latin1", low_memory=False)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"], format="%d/%m/%Y", errors="coerce")
    results = raw_df.dropna(subset=["Date", "FTHG", "FTAG"]).copy()
    results["MatchDateKey"] = results["Date"].dt.date.astype(str)

    ledger["MatchDate"] = ledger["MatchDate"].astype(str)
    settled_count = 0

    for idx in ledger[unsettled_mask].index:
        row = ledger.loc[idx]
        match = results[
            (results["MatchDateKey"] == row["MatchDate"])
            & (results["HomeTeam"] == row["HomeTeam"])
            & (results["AwayTeam"] == row["AwayTeam"])
        ]
        if match.empty:
            continue

        result_row = match.iloc[0]
        fthg = int(result_row["FTHG"])
        ftag = int(result_row["FTAG"])
        actual_result = f"{fthg}-{ftag}"

        if row["Market"] == "1X2":
            actual_outcome = "H" if fthg > ftag else ("A" if fthg < ftag else "D")
            bet_outcome = "Won" if row["Pick"] == actual_outcome else "Lost"
        elif row["Market"] == "O/U 2.5":
            actual_side = "Over" if (fthg + ftag) > 2.5 else "Under"
            bet_outcome = "Won" if row["Pick"] == actual_side else "Lost"
        else:
            continue

        ledger.at[idx, "ActualResult"] = actual_result
        ledger.at[idx, "BetOutcome"] = bet_outcome
        settled_count += 1

    if settled_count:
        ledger.to_csv(ledger_file, index=False)
        if verbose:
            print(f"Settled {settled_count} bet(s) in {ledger_file}")
    elif verbose:
        print("No matching completed results found for any unsettled bets yet.")

    return settled_count


def main():
    settle_bets(verbose=True)


if __name__ == "__main__":
    main()
