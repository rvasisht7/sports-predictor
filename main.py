"""Usage:
    python main.py backtest     # grade the model on past seasons
    python main.py predict      # win probabilities for the upcoming week
    python main.py ratings      # current Elo power rankings
"""
import sys

import numpy as np
import pandas as pd

from data import current_week, load_games, vegas_prob_row
from elo import EloModel


def build(refresh=False):
    games = load_games(refresh=refresh)
    model = EloModel()
    games = model.run(games)
    games["vegas_home_prob"] = [vegas_prob_row(h, a, s) for h, a, s in
                                zip(games.home_moneyline, games.away_moneyline, games.spread_line)]
    return model, games


def score(df, col):
    d = df.dropna(subset=[col, "home_win"])
    d = d[d["home_win"] != 0.5]
    p, y = d[col].clip(0.01, 0.99), d["home_win"]
    return {
        "games": len(d),
        "accuracy": round(float(((p > 0.5) == (y == 1)).mean()), 3),
        "brier": round(float(((p - y) ** 2).mean()), 4),
        "log_loss": round(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), 4),
    }


def backtest(games, seasons=range(2015, 2027)):
    rows = []
    for s in seasons:
        d = games[(games.season == s) & games.completed]
        if d.empty:
            continue
        rows.append({"season": s, "model": "Elo", **score(d, "elo_home_prob")})
        rows.append({"season": s, "model": "Vegas", **score(d, "vegas_home_prob")})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    d = games[games.season.isin(seasons) & games.completed]
    print("\nOverall:")
    print("  Elo  ", score(d, "elo_home_prob"))
    print("  Vegas", score(d, "vegas_home_prob"))
    print("  (Lower Brier/log loss = better. Always picking home = Brier 0.25.)")


def predict(games):
    season, week = current_week(games)
    wk = games[(games.season == season) & (games.week == week) & ~games.completed]
    print(f"\n{season} Week {week} predictions\n")
    for g in wk.itertuples():
        p = g.elo_home_prob
        pick = g.home_team if p >= 0.5 else g.away_team
        conf = max(p, 1 - p)
        v = g.vegas_home_prob
        vtxt = f"  Vegas: {g.home_team} {v:.0%}" if not pd.isna(v) else ""
        print(f"{g.away_team:>4} @ {g.home_team:<4}  Elo: {g.home_team} {p:.0%}  -> {pick} ({conf:.0%}){vtxt}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "predict"
    model, games = build(refresh="--refresh" in sys.argv)
    {"backtest": lambda: backtest(games),
     "predict": lambda: predict(games),
     "ratings": lambda: print(model.table().to_string())}[cmd]()
