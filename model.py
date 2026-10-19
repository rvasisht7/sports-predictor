"""Upgraded prediction model.

For every game, in date order, it records what was known *before* kickoff:
  elo       Elo win probability (as log-odds)
  epa       team efficiency: offense EPA/play minus defense EPA/play allowed (recent games weighted more)
  qb        starting QB quality: EPA per dropback, shrunk toward backup level for QBs with few snaps
  qb_change starter vs. the QB who started the team's previous game (catches injuries/benchings)
  injuries  missing non-QB starters (Out/Doubtful), weighted by position and snap share
  rest      rest-day difference
Then a logistic regression learns how much each one matters, trained only on
earlier seasons (walk-forward), so every backtest number is out-of-sample."""
import math
from collections import defaultdict

import numpy as np
import pandas as pd

from elo import norm

FEATURES = ["elo", "epa", "qb", "qb_change", "injuries", "rest", "neutral"]
LABELS = {"elo": "Elo rating", "epa": "Team efficiency (EPA)", "qb": "Quarterback matchup",
          "qb_change": "QB change", "injuries": "Injuries", "rest": "Rest", "neutral": "Neutral site"}

EPA_ALPHA = 0.12          # weight of the newest game in team EPA averages
EPA_SEASON_KEEP = 0.55    # share of team EPA carried into a new season
QB_DECAY = 0.985          # per-game decay of a QB's past dropbacks
QB_PRIOR_DB = 250         # dropbacks of "backup level" prior
QB_REPLACEMENT = -0.10    # backup-level EPA per dropback
FIRST_TRAIN_SEASON = 2007


def _rest(h, a):
    h = 7 if h is None or h != h else h
    a = 7 if a is None or a != a else a
    return float(np.clip(h - a, -7, 7))


def build_features(games: pd.DataFrame, team_games: pd.DataFrame, qb_games: pd.DataFrame,
                   inj_values: pd.DataFrame) -> pd.DataFrame:
    tg = {(r.game_id, r.posteam): r for r in team_games.itertuples()}
    qbg = defaultdict(list)
    for r in qb_games.itertuples():
        qbg[(r.game_id, r.posteam)].append(r)
    inj = {(r.season, r.week, r.team): r.inj_out for r in inj_values.itertuples()}

    off, dfn = defaultdict(float), defaultdict(float)
    qS, qW = defaultdict(float), defaultdict(float)
    qb_names, last_starter = {}, {}
    season = None
    rows = []

    def qb_val(qid):
        if qid is None:
            return QB_REPLACEMENT
        return (qS[qid] + QB_PRIOR_DB * QB_REPLACEMENT) / (qW[qid] + QB_PRIOR_DB)

    for g in games.itertuples():
        if g.season != season:
            for t in list(off):
                off[t] *= EPA_SEASON_KEEP
                dfn[t] *= EPA_SEASON_KEEP
            season = g.season
        h, a = norm(g.home_team), norm(g.away_team)
        starters = {}
        for side, t, qid in (("home", h, g.home_qb_id), ("away", a, g.away_qb_id)):
            qid = qid if isinstance(qid, str) and qid else last_starter.get(t)
            starters[side] = qid
        hq, aq = starters["home"], starters["away"]
        p = min(max(g.elo_home_prob, 1e-4), 1 - 1e-4)
        row = {
            "elo": math.log(p / (1 - p)),
            "epa": (off[h] - dfn[h]) - (off[a] - dfn[a]),
            "qb": qb_val(hq) - qb_val(aq),
            "qb_change": (qb_val(hq) - qb_val(last_starter.get(h, hq)))
                         - (qb_val(aq) - qb_val(last_starter.get(a, aq))),
            "injuries": inj.get((g.season, g.week, h), 0.0) - inj.get((g.season, g.week, a), 0.0),
            "rest": _rest(g.home_rest, g.away_rest),
            "neutral": 1.0 if g.location == "Neutral" else 0.0,
            "home_qb": qb_names.get(hq, g.home_qb_name), "away_qb": qb_names.get(aq, g.away_qb_name),
            "home_qb_epa": qb_val(hq), "away_qb_epa": qb_val(aq),
            "home_net_epa": off[h] - dfn[h], "away_net_epa": off[a] - dfn[a],
        }
        rows.append(row)

        if g.completed:
            for t, o in ((h, a), (a, h)):
                r = tg.get((g.game_id, t))
                if r is not None:
                    off[t] += EPA_ALPHA * (r.off_epa - off[t])
                    dfn[o] += EPA_ALPHA * (r.off_epa - dfn[o])
                plays = qbg.get((g.game_id, t), [])
                for q in plays:
                    qS[q.qb_id] = qS[q.qb_id] * QB_DECAY + q.qb_epa_sum
                    qW[q.qb_id] = qW[q.qb_id] * QB_DECAY + q.dropbacks
                    qb_names[q.qb_id] = q.qb_name
                if plays:
                    last_starter[t] = max(plays, key=lambda q: q.dropbacks).qb_id
    feats = pd.DataFrame(rows, index=games.index)
    return games.join(feats)


class Logit:
    """Logistic regression with standardized inputs and L2 regularization, fit with
    Newton's method. Same math as scikit-learn's LogisticRegression(C=0.5) with a
    StandardScaler, in a few lines of numpy (scikit-learn alone needs ~160 MB of memory)."""

    def __init__(self, C: float = 0.5, iters: int = 50):
        self.C, self.iters = C, iters

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.mean_, self.scale_ = X.mean(0), X.std(0)
        self.scale_[self.scale_ == 0] = 1.0
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mean_) / self.scale_])
        w = np.zeros(Z.shape[1])
        reg = np.full(Z.shape[1], 1 / self.C); reg[0] = 0.0      # intercept isn't penalized
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-Z @ w))
            grad = Z.T @ (p - y) + reg * w
            hess = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(reg)
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.intercept_, self.coef_ = np.array([w[0]]), w[None, 1:]
        return self

    def predict_proba(self, X):
        z = ((np.asarray(X, float) - self.mean_) / self.scale_) @ self.coef_[0] + self.intercept_[0]
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])


def _fit(train: pd.DataFrame):
    d = train[train.completed & (train.home_win != 0.5)]
    return Logit(C=0.5).fit(d[FEATURES], d.home_win)


def walk_forward(df: pd.DataFrame, first_test_season: int = 2015) -> pd.DataFrame:
    """Predict each season with a model trained only on earlier seasons."""
    df = df.copy()
    df["model_home_prob"] = np.nan
    pool = df[df.season >= FIRST_TRAIN_SEASON]
    seasons = sorted(df.season.unique())
    for s in seasons:
        if s < first_test_season:
            continue
        m = _fit(pool[pool.season < s])
        idx = df.index[df.season == s]
        df.loc[idx, "model_home_prob"] = m.predict_proba(df.loc[idx, FEATURES])[:, 1]
    final = _fit(pool[pool.season < seasons[-1]])
    return df, final


def explain(model, row) -> list[dict]:
    """How much each factor pushed the home team's chances (in log-odds),
    turned into percentage-point contributions that add up to the final prob."""
    x = (np.array([row[f] for f in FEATURES], dtype=float) - model.mean_) / model.scale_
    contrib = model.coef_[0] * x
    base = model.intercept_[0]
    sig = lambda z: 1 / (1 + math.exp(-z))
    out, running = [{"factor": "Home field", "points": round((sig(base) - 0.5) * 100, 1)}], base
    for f, c in sorted(zip(FEATURES, contrib), key=lambda t: -abs(t[1])):
        before = sig(running); running += c
        out.append({"factor": LABELS[f], "points": round((sig(running) - before) * 100, 1)})
    return out


def coefficients(model) -> dict:
    return {LABELS[f]: round(float(c), 3) for f, c in zip(FEATURES, model.coef_[0])}
