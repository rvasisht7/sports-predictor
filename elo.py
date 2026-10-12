"""Elo rating model for NFL teams (538-style: margin-of-victory multiplier,
home-field advantage, and regression to the mean between seasons)."""
import math
from dataclasses import dataclass, field

import pandas as pd

# Franchise moves: map old abbreviations to current ones so ratings carry over.
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


def norm(team: str) -> str:
    return TEAM_ALIASES.get(team, team)


@dataclass
class EloModel:
    k: float = 20.0
    home_adv: float = 48.0       # Elo points (~2 pts on the spread; 25 Elo ~ 1 point)
    season_regress: float = 1 / 3
    base: float = 1500.0
    ratings: dict = field(default_factory=dict)
    history: list = field(default_factory=list)  # (season, week, team, rating) after each game
    _season: int | None = None

    def rating(self, team: str) -> float:
        return self.ratings.get(norm(team), self.base)

    def win_prob(self, home: str, away: str, neutral: bool = False) -> float:
        diff = self.rating(home) - self.rating(away) + (0 if neutral else self.home_adv)
        return 1 / (1 + 10 ** (-diff / 400))

    def _new_season(self, season: int):
        if self._season is not None and season != self._season:
            for t in self.ratings:
                self.ratings[t] = self.base + (self.ratings[t] - self.base) * (1 - self.season_regress)
        self._season = season

    def update(self, home, away, home_score, away_score, neutral=False):
        p = self.win_prob(home, away, neutral)
        result = 1.0 if home_score > away_score else 0.0 if home_score < away_score else 0.5
        margin = abs(home_score - away_score)
        elo_diff = self.rating(home) - self.rating(away) + (0 if neutral else self.home_adv)
        winner_diff = elo_diff if result == 1 else -elo_diff
        mov_mult = math.log(margin + 1) * (2.2 / (winner_diff * 0.001 + 2.2))
        shift = self.k * mov_mult * (result - p)
        h, a = norm(home), norm(away)
        self.ratings[h] = self.rating(home) + shift
        self.ratings[a] = self.rating(away) - shift
        return shift

    def run(self, games: pd.DataFrame) -> pd.DataFrame:
        """Walk through games in order. For each game, record the pre-game
        probability (no leakage), then update ratings if it has been played."""
        probs, shifts = [], []
        for g in games.itertuples():
            self._new_season(g.season)
            neutral = g.location == "Neutral"
            probs.append(self.win_prob(g.home_team, g.away_team, neutral))
            shift = float("nan")
            if g.completed:
                shift = self.update(g.home_team, g.away_team, g.home_score, g.away_score, neutral)
                for t in (g.home_team, g.away_team):
                    self.history.append((g.season, g.week, norm(t), self.rating(t)))
            shifts.append(shift)
        out = games.copy()
        out["elo_home_prob"] = probs
        out["elo_shift"] = shifts  # rating points the home team gained (away lost)
        return out

    def table(self) -> pd.Series:
        return pd.Series(self.ratings).sort_values(ascending=False).round(0)
