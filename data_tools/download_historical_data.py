import argparse
import io
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "historical_raw"
RAW_FILE = RAW_DIR / "premier_league_10_years.csv"

DEFAULT_SEASONS = [
    "2526", "2425", "2324", "2223", "2122", "2021", "1920", "1819", "1718", "1617", "1516", "1415"
]
BASE_URL = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"


def resolve_active_season_candidates():
    current_year = datetime.now().year
    current_code = f"{str(current_year)[-2:]}{str(current_year + 1)[-2:]}"
    previous_code = f"{str(current_year - 1)[-2:]}{str(current_year)[-2:]}"
    candidates = [current_code, previous_code]
    for season in DEFAULT_SEASONS:
        if season not in candidates:
            candidates.append(season)
    return candidates


def normalize_match_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    normalized = df.copy()
    normalized.columns = [str(col).strip() for col in normalized.columns]
    for col in ["Date", "HomeTeam", "AwayTeam"]:
        if col in normalized.columns:
            normalized[col] = normalized[col].astype(str)
    return normalized


def fetch_season_csv(season: str):
    url = BASE_URL.format(season)
    try:
        with urlopen(url, timeout=20) as response:
            payload = response.read()
        if not payload:
            return None
        df = pd.read_csv(io.BytesIO(payload), encoding="latin1", low_memory=False)
        return df
    except Exception:
        return None


def merge_latest_season_rows(existing_df: pd.DataFrame, fresh_df: pd.DataFrame) -> pd.DataFrame:
    if existing_df.empty and fresh_df.empty:
        return pd.DataFrame()

    if existing_df.empty:
        combined = fresh_df.copy()
    elif fresh_df.empty:
        combined = existing_df.copy()
    else:
        existing_norm = existing_df.copy()
        fresh_norm = fresh_df.copy()
        for col in ["Date", "HomeTeam", "AwayTeam"]:
            if col in existing_norm.columns:
                existing_norm[col] = existing_norm[col].astype(str)
            if col in fresh_norm.columns:
                fresh_norm[col] = fresh_norm[col].astype(str)
        combined = pd.concat([existing_norm, fresh_norm], ignore_index=True)

    required = {"Date", "HomeTeam", "AwayTeam"}
    if not required.issubset(set(combined.columns)):
        return combined

    combined["Date"] = combined["Date"].astype(str).str.strip()
    combined["HomeTeam"] = combined["HomeTeam"].astype(str).str.strip()
    combined["AwayTeam"] = combined["AwayTeam"].astype(str).str.strip()
    combined = combined.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="last")
    combined = combined.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    return combined


def sync_historical_data(raw_file: Path = RAW_FILE, seasons=None, verbose: bool = True):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    seasons = seasons or resolve_active_season_candidates()

    if raw_file.exists():
        existing = pd.read_csv(raw_file, low_memory=False)
    else:
        existing = pd.DataFrame()

    if verbose:
        print(f"Loading existing data: {len(existing)} rows")

    season_frames = []
    for season in seasons:
        url = BASE_URL.format(season)
        if verbose:
            print(f"Checking season {season}: {url}")
        try:
            with urlopen(url, timeout=20) as response:
                payload = response.read()
            if not payload:
                if verbose:
                    print(f"  No payload for {season}")
                continue
            df = pd.read_csv(io.BytesIO(payload), encoding="latin1", low_memory=False)
            if df.empty:
                continue
            df = normalize_match_columns(df)
            df["Season"] = season
            season_frames.append(df)
            if verbose:
                print(f"  Fetched {len(df)} rows for season {season}")
        except Exception as exc:
            if verbose:
                print(f"  Failed to fetch {season}: {exc}")
            continue

    if not season_frames:
        if verbose:
            print("No season data was fetched.")
        return existing

    fresh = pd.concat(season_frames, ignore_index=True)
    merged = merge_latest_season_rows(existing, fresh)
    merged.to_csv(raw_file, index=False)

    if verbose:
        print(f"Saved merged raw dataset to {raw_file} with {len(merged)} rows.")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Download and merge latest English Premier League historical data.")
    parser.add_argument("--season", action="append", default=None, help="Season code to fetch, e.g. 2526.")
    parser.add_argument("--sync", action="store_true", help="Fetch the latest active season and merge it into raw historical data.")
    args = parser.parse_args()

    seasons = args.season or resolve_active_season_candidates()
    if args.sync:
        seasons = [season for season in seasons if season in resolve_active_season_candidates()]
        if not seasons:
            seasons = resolve_active_season_candidates()[:1]
        sync_historical_data(raw_file=RAW_FILE, seasons=seasons, verbose=True)
        return

    dfs = []
    for season in seasons:
        df = fetch_season_csv(season)
        if df is None:
            continue
        df = normalize_match_columns(df)
        df["Season"] = season
        dfs.append(df)

    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(RAW_FILE, index=False)
        print(f"SUCCESS: Downloaded {len(final_df)} matches into {RAW_FILE}")
    else:
        print("No season data downloaded successfully.")


if __name__ == "__main__":
    import io
    main()