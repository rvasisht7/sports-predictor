"""Data layer: pulls NFL schedules/results from nflverse and caches them locally."""
import math
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)


def load_games(refresh: bool = False) -> pd.DataFrame:
    """All games since 1999 (completed and upcoming), cached to parquet."""
    path = CACHE / "schedules.parquet"
    if refresh or not path.exists():
        df = nfl.load_schedules().to_pandas()
        df.to_parquet(path)
    df = pd.read_parquet(path)
    df = df.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)
    df["completed"] = df["home_score"].notna()
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(float)
    df.loc[df["home_score"] == df["away_score"], "home_win"] = 0.5  # ties
    df.loc[~df["completed"], "home_win"] = float("nan")
    return df


def vegas_home_prob(spread_line: float) -> float:
    """Convert nflverse spread_line (positive = home favored) into a win probability.
    NFL margins vs. the spread are roughly normal with SD ~13.5 points."""
    if spread_line is None or pd.isna(spread_line):
        return float("nan")
    return 0.5 * (1 + math.erf(spread_line / (13.5 * math.sqrt(2))))


def moneyline_prob(ml: float) -> float:
    """American odds -> implied probability (includes the book's margin)."""
    if ml is None or pd.isna(ml):
        return float("nan")
    return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)


def vegas_prob_row(home_ml, away_ml, spread_line) -> float:
    """Market home-win probability: moneylines with the margin (vig) removed,
    falling back to the spread when moneylines are missing."""
    h, a = moneyline_prob(home_ml), moneyline_prob(away_ml)
    if not (pd.isna(h) or pd.isna(a)):
        return h / (h + a)
    return vegas_home_prob(spread_line)


def current_week(df: pd.DataFrame) -> tuple[int, int]:
    """(season, week) of the next week with unplayed games."""
    upcoming = df[~df["completed"]]
    row = upcoming.iloc[0]
    return int(row["season"]), int(row["week"])
