import os
import numpy as np
import pandas as pd


def calculate_elo(r_home, r_away, outcome, k=20, home_adv=50):
    dr = (r_home + home_adv) - r_away
    e_home = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
    e_away = 1.0 - e_home
    new_r_home = r_home + k * (outcome - e_home)
    new_r_away = r_away + k * ((1.0 - outcome) - e_away)
    return new_r_home, new_r_away


def _recent_team_stats(history, lookback=5):
    if not history:
        return {
            "gf_avg": 1.2,
            "ga_avg": 1.2,
            "pts_avg": 1.0,
            "goal_diff_avg": 0.0,
            "sot_diff_avg": 0.0,
        }
    recent = history[-lookback:]
    gf = np.asarray([match["GF"] for match in recent], dtype=float)
    ga = np.asarray([match["GA"] for match in recent], dtype=float)
    pts = np.asarray([match.get("Pts", 1.0) for match in recent], dtype=float)
    gdiff = np.asarray([match["GF"] - match["GA"] for match in recent], dtype=float)
    sot_diff = np.asarray([match.get("SoT_F", 0) - match.get("SoT_A", 0) for match in recent], dtype=float)
    return {
        "gf_avg": float(np.mean(gf)),
        "ga_avg": float(np.mean(ga)),
        "pts_avg": float(np.mean(pts)),
        "goal_diff_avg": float(np.mean(gdiff)),
        "sot_diff_avg": float(np.mean(sot_diff)),
    }


def _calculate_poisson_xg(home_stats, away_stats, league_home_goal_avg, league_away_goal_avg):
    home_attack = (home_stats["gf_avg"] + 0.5) / max(league_home_goal_avg + 0.5, 0.5)
    away_defense = (away_stats["ga_avg"] + 0.5) / max(league_away_goal_avg + 0.5, 0.5)
    away_attack = (away_stats["gf_avg"] + 0.5) / max(league_away_goal_avg + 0.5, 0.5)
    home_defense = (home_stats["ga_avg"] + 0.5) / max(league_home_goal_avg + 0.5, 0.5)

    home_lambda = home_attack * away_defense * (league_home_goal_avg * 1.15)
    away_lambda = away_attack * home_defense * (league_away_goal_avg * 1.10)

    return float(np.clip(home_lambda, 0.05, 3.2)), float(np.clip(away_lambda, 0.05, 3.2))


def main():
    input_file = "./data/historical_raw/premier_league_10_years.csv"
    output_dir = "./data/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "model_features_10_years.csv")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing {input_file}. Run download script first.")

    df = pd.read_csv(input_file, encoding="latin1", low_memory=False)
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values(by=["Date"]).reset_index(drop=True)

    teams = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))
    elo_ratings = {team: 1500.0 for team in teams}
    team_matches = {team: [] for team in teams}
    referee_history = {}

    feature_rows = []
    league_home_goal_avg = float(df["FTHG"].mean())
    league_away_goal_avg = float(df["FTAG"].mean())

    for _, row in df.iterrows():
        h_team = row["HomeTeam"]
        a_team = row["AwayTeam"]
        ref = str(row.get("Referee", "Unknown")).strip()
        match_date = row["Date"]

        h_elo = elo_ratings.get(h_team, 1500.0)
        a_elo = elo_ratings.get(a_team, 1500.0)
        elo_diff = h_elo - a_elo

        h_history = team_matches[h_team]
        a_history = team_matches[a_team]

        h_rest = min((match_date - h_history[-1]["Date"]).days if h_history else 14, 30)
        a_rest = min((match_date - a_history[-1]["Date"]).days if a_history else 14, 30)

        def extract_form(history, venue_filter=None):
            sub = [match for match in history if venue_filter is None or match["Venue"] == venue_filter]
            last5 = sub[-5:] if len(sub) >= 5 else sub
            if not last5:
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

            weights = np.arange(1, len(last5) + 1, dtype=float)
            pts = np.asarray([match["Pts"] for match in last5], dtype=float)
            g_diff = np.asarray([match["GF"] - match["GA"] for match in last5], dtype=float)
            sot_diff = np.asarray([match["SoT_F"] - match["SoT_A"] for match in last5], dtype=float)
            gf = np.asarray([match["GF"] for match in last5], dtype=float)
            ga = np.asarray([match["GA"] for match in last5], dtype=float)
            corners = np.asarray([match["Corners_F"] for match in last5], dtype=float)
            cs = np.asarray([1.0 if match["GA"] == 0 else 0.0 for match in last5], dtype=float)
            cards = np.asarray([match["Yellow"] + (2.0 * match["Red"]) for match in last5], dtype=float)
            over25 = np.asarray([1.0 if (match["GF"] + match["GA"]) > 2.5 else 0.0 for match in last5], dtype=float)
            btts = np.asarray([1.0 if (match["GF"] > 0 and match["GA"] > 0) else 0.0 for match in last5], dtype=float)

            return {
                "pts_avg": float(np.average(pts, weights=weights)),
                "goal_diff_avg": float(np.average(g_diff, weights=weights)),
                "sot_diff_avg": float(np.average(sot_diff, weights=weights)),
                "goals_scored_avg": float(np.average(gf, weights=weights)),
                "goals_conceded_avg": float(np.average(ga, weights=weights)),
                "corners_avg": float(np.mean(corners)),
                "clean_sheets": float(np.mean(cs)),
                "cards_avg": float(np.mean(cards)),
                "over25_rate": float(np.mean(over25)),
                "btts_rate": float(np.mean(btts)),
            }

        h_overall = extract_form(h_history)
        a_overall = extract_form(a_history)
        h_home = extract_form(h_history, venue_filter="H")
        a_away = extract_form(a_history, venue_filter="A")

        home_recent = _recent_team_stats(h_history)
        away_recent = _recent_team_stats(a_history)
        h_home_recent = _recent_team_stats([match for match in h_history if match["Venue"] == "H"])
        a_away_recent = _recent_team_stats([match for match in a_history if match["Venue"] == "A"])
        home_xg_l5, away_xg_l5 = _calculate_poisson_xg(home_recent, away_recent, league_home_goal_avg, league_away_goal_avg)
        home_xg_home, away_xg_away = _calculate_poisson_xg(h_home_recent, a_away_recent, league_home_goal_avg, league_away_goal_avg)
        home_xg_value = float(np.clip((home_xg_l5 + home_xg_home) / 2.0, 0.05, 3.2))
        away_xg_value = float(np.clip((away_xg_l5 + away_xg_away) / 2.0, 0.05, 3.2))
        xg_diff = home_xg_value - away_xg_value
        xg_ratio = home_xg_value / max(away_xg_value, 1e-6)

        h2h_matches = [match for match in h_history if match["Opponent"] == a_team][-5:]
        h2h_win_rate = (sum(1 for match in h2h_matches if match["Pts"] == 3) / len(h2h_matches)) if h2h_matches else 0.33

        if ref in referee_history and len(referee_history[ref]) >= 5:
            ref_home_win_rate = np.mean([1 if outcome == "H" else 0 for outcome in referee_history[ref][-20:]])
        else:
            ref_home_win_rate = 0.45

        b365_h, b365_d, b365_a = row.get("B365H", np.nan), row.get("B365D", np.nan), row.get("B365A", np.nan)
        if pd.notna(b365_h) and pd.notna(b365_d) and pd.notna(b365_a) and b365_h > 0:
            raw_h, raw_d, raw_a = 1.0 / b365_h, 1.0 / b365_d, 1.0 / b365_a
            margin = raw_h + raw_d + raw_a
            prob_h, prob_d, prob_a = raw_h / margin, raw_d / margin, raw_a / margin
        else:
            prob_h, prob_d, prob_a = 0.45, 0.25, 0.30

        fthg, ftag = row["FTHG"], row["FTAG"]
        target_over25 = 1 if (fthg + ftag) > 2.5 else 0
        target_btts = 1 if (fthg > 0 and ftag > 0) else 0

        feature_rows.append({
            "Season": str(row["Season"]),
            "Date": match_date,
            "HomeTeam": h_team,
            "AwayTeam": a_team,
            "Target_FTR": row["FTR"],
            "Target_Over25": target_over25,
            "Target_BTTS": target_btts,
            "h_form_pts": h_overall["pts_avg"],
            "a_form_pts": a_overall["pts_avg"],
            "h_form_gdiff": h_overall["goal_diff_avg"],
            "a_form_gdiff": a_overall["goal_diff_avg"],
            "h_form_sot_diff": h_overall["sot_diff_avg"],
            "a_form_sot_diff": a_overall["sot_diff_avg"],
            "h_venue_pts": h_home["pts_avg"],
            "a_venue_pts": a_away["pts_avg"],
            "h_goals_scored_avg": h_overall["goals_scored_avg"],
            "h_goals_conceded_avg": h_overall["goals_conceded_avg"],
            "a_goals_scored_avg": a_overall["goals_scored_avg"],
            "a_goals_conceded_avg": a_overall["goals_conceded_avg"],
            "expected_total_goals_proxy": (h_overall["goals_scored_avg"] + a_overall["goals_conceded_avg"] + a_overall["goals_scored_avg"] + h_overall["goals_conceded_avg"]) / 2.0,
            "h_over25_rate": h_overall["over25_rate"],
            "a_over25_rate": a_overall["over25_rate"],
            "h_btts_rate": h_overall["btts_rate"],
            "a_btts_rate": a_overall["btts_rate"],
            "Home_xG_L5": home_xg_value,
            "Away_xG_L5": away_xg_value,
            "xG_Diff": xg_diff,
            "xG_Ratio": xg_ratio,
            "elo_diff": elo_diff,
            "prob_home_market": prob_h,
            "prob_draw_market": prob_d,
            "prob_away_market": prob_a,
            "h_rest_days": h_rest,
            "a_rest_days": a_rest,
            "rest_diff": h_rest - a_rest,
            "h_clean_sheet_rate": h_overall["clean_sheets"],
            "a_clean_sheet_rate": a_overall["clean_sheets"],
            "h2h_home_win_rate": h2h_win_rate,
            "h_cards_avg": h_overall["cards_avg"],
            "a_cards_avg": a_overall["cards_avg"],
            "ref_home_bias": ref_home_win_rate,
        })

        outcome_val = 1.0 if row["FTR"] == "H" else (0.5 if row["FTR"] == "D" else 0.0)
        new_h_elo, new_a_elo = calculate_elo(h_elo, a_elo, outcome_val)
        elo_ratings[h_team], elo_ratings[a_team] = new_h_elo, new_a_elo

        h_pts = 3 if row["FTR"] == "H" else (1 if row["FTR"] == "D" else 0)
        a_pts = 3 if row["FTR"] == "A" else (1 if row["FTR"] == "D" else 0)

        team_matches[h_team].append({
            "Date": match_date,
            "Opponent": a_team,
            "Venue": "H",
            "Pts": h_pts,
            "GF": fthg,
            "GA": ftag,
            "SoT_F": row.get("HST", 0),
            "SoT_A": row.get("AST", 0),
            "Corners_F": row.get("HC", 0),
            "Yellow": row.get("HY", 0),
            "Red": row.get("HR", 0),
        })
        team_matches[a_team].append({
            "Date": match_date,
            "Opponent": h_team,
            "Venue": "A",
            "Pts": a_pts,
            "GF": ftag,
            "GA": fthg,
            "SoT_F": row.get("AST", 0),
            "SoT_A": row.get("HST", 0),
            "Corners_F": row.get("AC", 0),
            "Yellow": row.get("AY", 0),
            "Red": row.get("AR", 0),
        })
        referee_history.setdefault(ref, []).append(row["FTR"])

    processed_df = pd.DataFrame(feature_rows)
    processed_df.to_csv(output_file, index=False)
    print(f"\nSUCCESS: Generated {len(processed_df)} feature rows into {output_file}")


if __name__ == "__main__":
    main()
