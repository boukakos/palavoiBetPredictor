import os
import pandas as pd
import requests

API_KEY = 'ea47f1a77d912002b64fff86502d7197'
SPORT = 'soccer_epl'
REGIONS = 'eu,uk,eu2'
MARKETS = 'h2h,totals'
TARGET_KEY = 'betano_uk'

# Αντιστοίχιση ονομάτων API στα ονόματα του εκπαιδευμένου μοντέλου
TEAM_MAPPING = {
    "Brighton and Hove Albion": "Brighton",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nottingham",
    "Sheffield United": "Sheffield Utd",
    "Luton Town": "Luton",
    "Leicester City": "Leicester",
    "Ipswich Town": "Ipswich",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Leeds United": "Leeds"
}

def clean_team(name):
    return TEAM_MAPPING.get(name, name)

url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds'
params = {
    'apiKey': API_KEY,
    'regions': REGIONS,
    'markets': MARKETS,
    'oddsFormat': 'decimal',
    'dateFormat': 'iso'
}

print("[...] Τραβάμε αποδόσεις από The Odds API...")
response = requests.get(url, params=params)

if response.status_code != 200:
    print(f"[!] Σφάλμα API: {response.text}")
    exit()

matches = response.json()
rows = []

for match in matches:
    home_raw = match.get('home_team')
    away_raw = match.get('away_team')
    commence_time = match.get('commence_time')

    home = clean_team(home_raw)
    away = clean_team(away_raw)

    stoiximan = None
    for bm in match.get('bookmakers', []):
        if bm.get('key').lower() == TARGET_KEY:
            stoiximan = bm
            break

    if not stoiximan:
        continue

    h2h_1, h2h_x, h2h_2 = None, None, None
    o25, u25 = None, None

    # 1. 1X2 & Totals από Stoiximan
    for market in stoiximan.get('markets', []):
        m_key = market.get('key')
        if m_key == 'h2h':
            for out in market.get('outcomes', []):
                if out.get('name') == home_raw:
                    h2h_1 = out.get('price')
                elif out.get('name') == away_raw:
                    h2h_2 = out.get('price')
                elif out.get('name') == 'Draw':
                    h2h_x = out.get('price')

        elif m_key == 'totals':
            for out in market.get('outcomes', []):
                try:
                    pt = float(out.get('point', 0))
                except (ValueError, TypeError):
                    pt = 0
                if pt == 2.5:
                    if out.get('name').lower() == 'over':
                        o25 = out.get('price')
                    elif out.get('name').lower() == 'under':
                        u25 = out.get('price')

    # 2. Fallback Totals από την υπόλοιπη αγορά αν λείπει από Stoiximan
    if o25 is None or u25 is None:
        for bm in match.get('bookmakers', []):
            if bm.get('key').lower() == TARGET_KEY:
                continue
            for market in bm.get('markets', []):
                if market.get('key') == 'totals':
                    for out in market.get('outcomes', []):
                        try:
                            pt = float(out.get('point', 0))
                        except (ValueError, TypeError):
                            pt = 0
                        if pt == 2.5:
                            if out.get('name').lower() == 'over' and o25 is None:
                                o25 = out.get('price')
                            elif out.get('name').lower() == 'under' and u25 is None:
                                u25 = out.get('price')
            if o25 is not None and u25 is not None:
                break

    # Κρατάμε μόνο αγώνες με πλήρεις αποδόσεις
    if all(v is not None for v in [h2h_1, h2h_x, h2h_2, o25, u25]):
        # Προεπιλογή GG/NG αν δεν παρέχεται ξεχωριστά (χρησιμοποιείται κυρίως το Poisson model)
        rows.append({
            'HomeTeam': home,
            'AwayTeam': away,
            'B365H': h2h_1,
            'B365D': h2h_x,
            'B365A': h2h_2,
            'B365>2.5': o25,
            'B365<2.5': u25,
            'GG_odds': 1.85,
            'NG_odds': 1.95,
            'Date': commence_time[:10] if commence_time else ''
        })

df = pd.DataFrame(rows)

# Περιορισμός στην τρέχουσα αγωνιστική (τα πρώτα 10 ματς)
df = df.head(10)

# Αποθήκευση στο CSV εισόδου του pipeline
target_paths = [
    os.path.join("data", "upcoming_fixtures.csv"),
    "upcoming_fixtures.csv"
]

saved = False
for path in target_paths:
    if os.path.exists(os.path.dirname(path)) or not os.path.dirname(path):
        df.to_csv(path, index=False)
        print(f"[✓] Αποθηκεύτηκαν επιτυχώς {len(df)} αγώνες στο: {path}")
        saved = True
        break

if not saved:
    df.to_csv("upcoming_fixtures.csv", index=False)
    print(f"[✓] Αποθηκεύτηκαν {len(df)} αγώνες στο: upcoming_fixtures.csv")