"""Web app. Run locally:   uvicorn app:app --reload
Then open http://127.0.0.1:8000"""
import asyncio
import math
import random
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

from data import current_week
from elo import norm
from live import fetch_scoreboard, live_win_prob, minutes_left
from main import build, score
from model import build_features, coefficients, explain, walk_forward
import news
import players
from teams import CONF_OF, DIVISION_OF, DIVISIONS, TEAM_NAME
import season as seasonsim
import tendencies
from data import load_games as load_raw_games
from elo import EloModel
from stats import load_epa, load_injuries_current

DATA_REFRESH_SECONDS = 6 * 60 * 60   # re-pull results, play-by-play and injuries every 6 hours
LIVE_FAST, LIVE_SLOW = 30, 300       # poll ESPN every 30s during games, 5 min otherwise

S = {"elo": None, "model": None, "games": None, "injuries": pd.DataFrame(), "team_games": None,
     "player_weeks": None, "totals": None, "fantasy": {}, "data_updated": None,
     "qb_games": None, "inj_values": None, "tend": pd.DataFrame(), "sim": None, "actual": None, "snapshots": {}}
SCORING = {"ppr": "fantasy_points_ppr", "half": "fp_half", "standard": "fantasy_points"}
LIVE = {"games": [], "updated": None, "error": None, "history": defaultdict(list)}


def clean(o):
    """Make pandas/numpy values JSON-safe (NaN -> null)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if math.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ---------------- data loading ----------------

def _build_all(refresh: bool):
    elo, games = build(refresh)
    cs = int(games.season.max())
    tg, qg = load_epa(cs, refresh)
    rep, injv = load_injuries_current(cs, refresh)
    df = build_features(games, tg, qg, injv)
    df, model = walk_forward(df)
    df["model_home_prob"] = df["model_home_prob"].fillna(df["elo_home_prob"])
    pw = players.load_player_weeks([cs - 1, cs], refresh)
    tend = tendencies.compute(cs)
    sim = seasonsim.simulate(df, cs, elo.ratings)
    actual = seasonsim.actual_bracket(df, cs, elo.ratings)
    return elo, model, df, rep, tg, pw, qg, injv, tend, sim, actual


def _fantasy(df, pw, rep):
    """Projections for every team's next game, for each scoring format."""
    upcoming = df[~df.completed & (df.game_type == "REG")].sort_values("gameday")
    first = {}   # each team's next game
    for r in upcoming.itertuples():
        for t in (norm(r.home_team), norm(r.away_team)):
            first.setdefault(t, r.Index)
    games = df.loc[sorted(set(first.values()))]
    if games.empty:
        return {}
    week, season = int(games.week.mode().iloc[0]), int(games.season.max())
    byes = sorted(t for t, i in first.items() if df.loc[i, "week"] > week)
    games = games[games.week <= week]
    inj = {}
    next_week = {t: int(df.loc[i, "week"]) for t, i in first.items()}   # each team's own next game
    if rep is not None and not rep.empty:
        cur = rep[(rep.season == season) & rep.report_status.notna()]
        for r in cur.itertuples():
            if r.week == next_week.get(r.team):
                inj[(r.full_name, r.team)] = {"status": r.report_status, "injury": r.report_primary_injury}
    out = {}
    for key, col in SCORING.items():
        p = players.project(pw, season, week, games, col, inj, schedules=df)
        out[key] = {"season": season, "week": week, "players": p, "byes": byes,
                    "dvp": players.defense_vs_position(pw, season, week, col)}
    return out


async def load_data(refresh: bool):
    try:
        elo, model, df, rep, tg, pw, qg, injv, tend, sim, actual = await asyncio.to_thread(_build_all, refresh)
    except Exception as e:
        print("refresh failed, using cache:", e)
        elo, model, df, rep, tg, pw, qg, injv, tend, sim, actual = await asyncio.to_thread(_build_all, False)
    cs = int(df.season.max())
    totals = players.season_totals(pw, cs)
    fantasy = await asyncio.to_thread(_fantasy, df, pw, rep)
    S.update(elo=elo, model=model, games=df, injuries=rep, team_games=tg, player_weeks=pw,
             totals=totals, fantasy=fantasy, qb_games=qg, inj_values=injv, tend=tend, sim=sim, actual=actual,
             snapshots={}, data_updated=time.time())


async def data_loop():
    while True:
        await asyncio.sleep(DATA_REFRESH_SECONDS)
        await load_data(refresh=True)


def find_game(home: str, away: str):
    g = S["games"]
    m = g[(g.home_team.map(norm) == home) & (g.away_team.map(norm) == away) & (g.season == g.season.max())]
    return next(m.tail(1).itertuples()) if len(m) else None


def pregame_prob(home: str, away: str, neutral: bool) -> float:
    r = find_game(home, away)
    return float(r.model_home_prob) if r is not None else S["elo"].win_prob(home, away, neutral)


async def live_loop():
    async with httpx.AsyncClient() as client:
        while True:
            delay = LIVE_SLOW
            try:
                games = await fetch_scoreboard(client)
                for g in games:
                    pre = pregame_prob(g["home"]["abbr"], g["away"]["abbr"], g["neutral"])
                    g["pregame_home_prob"] = round(pre, 4)
                    if g["state"] == "pre":
                        g["live_home_prob"] = g["pregame_home_prob"]
                    else:
                        margin = g["home"]["score"] - g["away"]["score"]
                        mins = 0 if g["state"] == "post" else minutes_left(g["period"], g["clock"])
                        g["minutes_left"] = round(mins, 2)
                        g["live_home_prob"] = round(live_win_prob(pre, margin, mins), 4)
                    hist = LIVE["history"][g["id"]]   # [[minutes_left, home_prob], ...]
                    if g["state"] != "pre":
                        if not hist:
                            hist.append([60.0, g["pregame_home_prob"]])
                        point = [g["minutes_left"], g["live_home_prob"]]
                        if hist[-1] != point:
                            hist.append(point)
                    g["history"] = hist
                    if g["state"] == "in":
                        delay = LIVE_FAST
                LIVE.update(games=games, updated=time.time(), error=None)
            except Exception as e:
                LIVE["error"] = "Can't reach ESPN for live scores. Retrying automatically."
                print(f"live feed error: {e!r}")
            await asyncio.sleep(delay)


@asynccontextmanager
async def lifespan(app):
    await load_data(refresh=True)
    tasks = [asyncio.create_task(t()) for t in (data_loop, live_loop)]
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(title="Sports Predictor", lifespan=lifespan)


@app.middleware("http")
async def always_fresh_pages(request, call_next):
    """Make browsers check for updated pages, scripts and styles on every load."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# ---------------- helpers ----------------

def season_games(season=None):
    g = S["games"]
    season = season or int(g.season.max())
    return g[g.season == season]


def records(season):
    rec = defaultdict(lambda: [0, 0, 0])
    for r in season_games(season).itertuples():
        if not r.completed or r.game_type != "REG":
            continue
        h, a = norm(r.home_team), norm(r.away_team)
        if r.home_score == r.away_score:
            rec[h][2] += 1; rec[a][2] += 1
        else:
            w, l = (h, a) if r.home_score > r.away_score else (a, h)
            rec[w][0] += 1; rec[l][1] += 1
    return {t: f"{w}-{l}" + (f"-{tie}" if tie else "") for t, (w, l, tie) in rec.items()}


def team_epa_table(season):
    tg = S["team_games"]
    d = tg[tg.season == season]
    if d.empty:
        return pd.DataFrame()
    off = d.groupby("posteam")[["off_epa", "pass_epa", "rush_epa", "off_sr"]].mean()
    dfn = d.groupby("defteam")[["off_epa", "pass_epa", "rush_epa"]].mean().add_prefix("def_")
    t = off.join(dfn)
    t["net_epa"] = t.off_epa - t.def_off_epa
    t["off_rank"] = t.off_epa.rank(ascending=False).astype(int)
    t["def_rank"] = t.def_off_epa.rank(ascending=True).astype(int)
    t["net_rank"] = t.net_epa.rank(ascending=False).astype(int)
    return t


def injuries_for(team: str, season: int, week: int, limit: int = 12):
    rep = S["injuries"]
    if rep is None or rep.empty:
        return []
    d = rep[(rep.season == season) & (rep.week == week) & (rep.team == team)
            & rep.report_status.isin(["Out", "Doubtful", "Questionable"])]
    d = d.sort_values(["impact", "usual_snap_share"], ascending=False).head(limit)
    return [{"name": r.full_name, "pos": r.position, "status": r.report_status,
             "injury": r.report_primary_injury, "snap_share": r.usual_snap_share,
             "impact": round(float(r.impact), 2), "qb": r.position == "QB"} for r in d.itertuples()]


def game_row(r, with_detail=False):
    p = float(r.model_home_prob)
    pick = r.home_team if p >= 0.5 else r.away_team
    row = {"game_id": r.game_id, "season": int(r.season), "week": int(r.week), "gameday": r.gameday,
           "home": norm(r.home_team), "away": norm(r.away_team),
           "model_home_prob": round(p, 3), "elo_home_prob": round(float(r.elo_home_prob), 3),
           "vegas_home_prob": None if pd.isna(r.vegas_home_prob) else round(float(r.vegas_home_prob), 3),
           "pick": norm(pick), "completed": bool(r.completed),
           "home_qb": r.home_qb, "away_qb": r.away_qb}
    if r.completed:
        tie = r.home_score == r.away_score
        winner = r.home_team if r.home_score > r.away_score else r.away_team
        row.update(home_score=int(r.home_score), away_score=int(r.away_score),
                   correct=None if tie else norm(winner) == norm(pick),
                   elo_shift=None if pd.isna(r.elo_shift) else round(float(r.elo_shift), 1))
    if with_detail:
        row["factors"] = explain(S["model"], r._asdict())
        row["keys"] = (tendencies.matchup_keys(S["tend"], row["home"], row["away"], 2)
                       + tendencies.matchup_keys(S["tend"], row["away"], row["home"], 2))
        row["home_injuries"] = injuries_for(row["home"], r.season, r.week, 6)
        row["away_injuries"] = injuries_for(row["away"], r.season, r.week, 6)
    return row


# ---------------- API ----------------

@app.get("/api/live")
def api_live():
    return clean({"updated": LIVE["updated"], "error": LIVE["error"], "games": LIVE["games"]})


@app.get("/api/predictions")
def api_predictions():
    g = S["games"]
    season, week = current_week(g)
    wk = g[(g.season == season) & (g.week == week)]
    return clean({"season": season, "week": week, "games": [game_row(r, True) for r in wk.itertuples()]})


@app.get("/api/ratings")
def api_ratings():
    return [{"team": k, "rating": float(v)} for k, v in S["elo"].table().items()]


@app.get("/api/elo-history")
def api_elo_history(teams: str = "KC,BUF,PHI,DET", seasons: int = 2):
    h = pd.DataFrame(S["elo"].history, columns=["season", "week", "team", "rating"])
    h = h[h.season > h.season.max() - seasons]
    out = {}
    for t in [x.strip().upper() for x in teams.split(",") if x.strip()][:6]:
        d = h[h.team == norm(t)]
        out[t] = [{"label": f"{s} W{w}", "rating": round(r, 1)} for s, w, _, r in d.itertuples(index=False)]
    return out


@app.get("/api/backtest")
def api_backtest():
    g = S["games"]
    rows = []
    for s in range(2015, int(g.season.max()) + 1):
        d = g[(g.season == s) & g.completed]
        if len(d):
            rows.append({"season": s, "model": score(d, "model_home_prob"),
                         "elo": score(d, "elo_home_prob"), "vegas": score(d, "vegas_home_prob")})
    return {"seasons": rows, "weights": coefficients(S["model"])}


@app.get("/api/teams")
def api_teams():
    season = int(S["games"].season.max())
    rec, epa = records(season), team_epa_table(season)
    ratings = S["elo"].table()
    out = []
    for rank, (t, rating) in enumerate(ratings.items(), 1):
        e = epa.loc[t] if t in epa.index else None
        out.append({"team": t, "name": TEAM_NAME.get(t, t), "elo": float(rating), "elo_rank": rank,
                    "record": rec.get(t, "0-0"),
                    "net_epa": None if e is None else round(float(e.net_epa), 3),
                    "net_rank": None if e is None else int(e.net_rank),
                    "off_rank": None if e is None else int(e.off_rank),
                    "def_rank": None if e is None else int(e.def_rank)})
    return clean({"season": season, "teams": out})


@app.get("/api/team/{abbr}")
def api_team(abbr: str):
    t = norm(abbr.upper())
    if t not in S["elo"].ratings:
        raise HTTPException(404, f"No team called {abbr}")
    g = S["games"]
    season = int(g.season.max())
    sg = season_games(season)
    mine = sg[(sg.home_team.map(norm) == t) | (sg.away_team.map(norm) == t)]
    schedule = []
    for r in mine.itertuples():
        row = game_row(r)
        home = row["home"] == t
        row.update(opponent=row["away"] if home else row["home"], at_home=home,
                   team_prob=row["model_home_prob"] if home else round(1 - row["model_home_prob"], 3))
        if r.completed:
            us, them = (row["home_score"], row["away_score"]) if home else (row["away_score"], row["home_score"])
            row["result"] = ("W" if us > them else "L" if us < them else "T") + f" {us}-{them}"
        schedule.append(row)
    tg = S["team_games"]
    weekly = []
    for r in tg[(tg.season == season) & (tg.posteam == t)].sort_values("week").itertuples():
        opp = tg[(tg.game_id == r.game_id) & (tg.posteam == r.defteam)]
        weekly.append({"week": int(r.week), "off_epa": round(r.off_epa, 3),
                       "def_epa": round(float(opp.off_epa.iloc[0]), 3) if len(opp) else None})
    epa = team_epa_table(season)
    ranks = epa.loc[t].to_dict() if t in epa.index else {}
    _, week = current_week(g)
    hist = pd.DataFrame(S["elo"].history, columns=["season", "week", "team", "rating"])
    hist = hist[(hist.team == t) & (hist.season >= season - 1)]
    nxt = next((s for s in schedule if not s["completed"]), None)
    last_qb = next((s["home_qb"] if s["at_home"] else s["away_qb"] for s in reversed(schedule) if s["completed"]), None)
    return clean({
        "team": t, "name": TEAM_NAME.get(t, t), "season": season, "record": records(season).get(t, "0-0"),
        "elo": round(S["elo"].rating(t)), "elo_rank": list(S["elo"].table().index).index(t) + 1,
        "qb": last_qb, "stats": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ranks.items()},
        "schedule": schedule, "weekly_epa": weekly,
        "elo_history": [{"label": f"{s} W{w}", "rating": round(r, 1)} for s, w, _, r in hist.itertuples(index=False)],
        "injuries": injuries_for(t, season, week, 20), "injury_week": week,
        "tendencies": tendencies.team_profile(S["tend"], t),
        "outlook": S["sim"]["teams"].get(t) if S["sim"] else None,
        "division": DIVISION_OF.get(t),
        "next_game": game_row(find_game(nxt["home"], nxt["away"]), True) if nxt else None,
    })


@app.get("/api/tracker")
def api_tracker(season: int | None = None):
    g = S["games"]
    season = season or int(g.season.max())
    d = g[(g.season == season) & g.completed & (g.home_score != g.away_score)]
    picks = []
    for r in d.itertuples():
        hw = r.home_score > r.away_score
        row = game_row(r)
        row["elo_correct"] = (r.elo_home_prob >= 0.5) == hw
        row["vegas_correct"] = None if pd.isna(r.vegas_home_prob) else bool((r.vegas_home_prob >= 0.5) == hw)
        picks.append(row)
    weeks = []
    for w, x in d.groupby("week"):
        weeks.append({"week": int(w), "games": len(x),
                      **{k: score(x, c)["accuracy"] for k, c in
                         (("model", "model_home_prob"), ("elo", "elo_home_prob"), ("vegas", "vegas_home_prob"))}})
    # calibration: when the model said X%, how often did that team win?
    p = d.model_home_prob.values
    y = (d.home_score > d.away_score).values.astype(float)
    fav_p, fav_won = np.where(p >= .5, p, 1 - p), np.where(p >= .5, y, 1 - y)
    bins = []
    for lo in (0.5, 0.6, 0.7, 0.8):
        m = (fav_p >= lo) & (fav_p < lo + 0.1 if lo < 0.8 else fav_p <= 1)
        if m.sum():
            bins.append({"range": f"{int(lo*100)}–{int(lo*100)+9 if lo < 0.8 else 100}%", "games": int(m.sum()),
                         "predicted": float(fav_p[m].mean()), "actual": float(fav_won[m].mean())})
    return clean({"season": season, "seasons": sorted(int(s) for s in g[g.completed].season.unique() if s >= 2015),
                  "totals": {k: score(d, c) for k, c in (("model", "model_home_prob"), ("elo", "elo_home_prob"),
                                                          ("vegas", "vegas_home_prob"))},
                  "weeks": weeks, "calibration": bins, "picks": picks[::-1]})


@app.get("/api/stats")
def api_stats():
    g = S["games"]
    season = int(g.season.max())
    d = g[(g.season == season) & g.completed & (g.home_score != g.away_score)].copy()
    if d.empty:
        return {"season": season, "facts": []}
    d["winner"] = np.where(d.home_score > d.away_score, d.home_team, d.away_team)
    d["loser"] = np.where(d.home_score > d.away_score, d.away_team, d.home_team)
    d["winner_prob"] = np.where(d.winner == d.home_team, d.model_home_prob, 1 - d.model_home_prob)
    d["total"] = d.home_score + d.away_score
    d["margin"] = (d.home_score - d.away_score).abs()
    correct = int((d.winner_prob > 0.5).sum())
    facts = [f"The model is {correct}-{len(d) - correct} picking winners in {season} ({correct / len(d):.0%})."]
    u = d.loc[d.winner_prob.idxmin()]
    facts.append(f"Biggest upset by the model's numbers: {u.winner} beat {u.loser} in week {u.week} "
                 f"with only a {u.winner_prob:.0%} chance.")
    h = d.loc[d.total.idxmax()]
    facts.append(f"Highest-scoring game: {h.away_team} {int(h.away_score)}, {h.home_team} {int(h.home_score)} "
                 f"in week {h.week} ({int(h.total)} points).")
    b = d.loc[d.margin.idxmax()]
    facts.append(f"Biggest blowout: {b.winner} over {b.loser} by {int(b.margin)} in week {b.week}.")
    facts.append(f"Home teams have won {(d.winner == d.home_team).mean():.0%} of games this season.")
    v = d.dropna(subset=["vegas_home_prob"])
    dis = v[(v.model_home_prob > 0.5) != (v.vegas_home_prob > 0.5)]
    if len(dis):
        right = int(((dis.model_home_prob > 0.5) == (dis.winner == dis.home_team)).sum())
        facts.append(f"When the model and Vegas picked different winners ({len(dis)} games), "
                     f"the model was right {right} times.")
    epa = team_epa_table(season)
    if not epa.empty:
        facts.append(f"Most efficient offense: {epa.off_epa.idxmax()} at {epa.off_epa.max():+.2f} EPA per play.")
        facts.append(f"Stingiest defense: {epa.def_off_epa.idxmin()}, allowing {epa.def_off_epa.min():+.2f} EPA per play.")
    hist = pd.DataFrame(S["elo"].history, columns=["season", "week", "team", "rating"])
    hs = hist[hist.season == season]
    if len(hs):
        delta = (hs.groupby("team").rating.last() - hs.groupby("team").rating.first()).sort_values()
        facts.append(f"Biggest riser since week 1: {delta.index[-1]} (+{delta.iloc[-1]:.0f} Elo). "
                     f"Biggest faller: {delta.index[0]} ({delta.iloc[0]:.0f}).")
    random.shuffle(facts)
    return {"season": season, "facts": facts}


# ---------------- leaders + fantasy ----------------

@app.get("/api/leaders")
def api_leaders(team: str | None = None, n: int = 5):
    g = S["games"]
    season = int(g.season.max())
    done = g[(g.season == season) & g.completed & (g.game_type == "REG")]
    weeks = int(done.week.max()) if len(done) else 1
    t = norm(team.upper()) if team else None
    return clean({"season": season, "through_week": weeks,
                  "groups": players.leaders(S["totals"], weeks, t, max(1, min(n, 10)))})


@app.get("/api/fantasy")
def api_fantasy(scoring: str = "ppr"):
    f = S["fantasy"].get(scoring if scoring in SCORING else "ppr")
    if not f:
        return {"players": [], "matchups": {}, "week": None}
    p = f["players"]
    # best and worst matchups by position: the defenses that give up the most/least
    teams_playing = {r.team: r.opp for r in p.itertuples()}
    matchups = {}
    for pos in players.FANTASY_POS:
        rows = [{"offense": off, "defense": dfn, "mult": v["mult"], "allowed": v["allowed"], "league": v["league"]}
                for off, dfn in teams_playing.items() if (v := f["dvp"].get((dfn, pos)))]
        rows.sort(key=lambda x: -x["mult"])
        matchups[pos] = {"best": rows[:5], "worst": rows[-5:][::-1]}
    return clean({"season": f["season"], "week": f["week"], "scoring": scoring, "byes": f["byes"],
                  "players": p.to_dict(orient="records"), "matchups": matchups,
                  "accuracy": players.BACKTEST, "cutoffs": players.STARTER_CUTOFF, "flex": players.FLEX_CUTOFF})


# ---------------- standings + playoffs ----------------

def _picture(rec):
    """Playoff seeds if the season ended today."""
    score = np.array([[rec[t]["pct"] + rec[t]["conf_pct"] * 1e-3 + rec[t]["diff"] * 1e-7 for t in seasonsim.TEAMS]])
    seeds = seasonsim._seed(score, 1)[0]
    return {conf: [seasonsim.TEAMS[i] for i in seeds[ci]] for ci, conf in enumerate(("AFC", "NFC"))}


@app.get("/api/standings")
def api_standings():
    g = S["games"]
    season = int(g.season.max())
    rec = seasonsim.standings(g, season)
    odds = S["sim"]["teams"] if S["sim"] else {}
    divs = []
    for d, teams in DIVISIONS.items():
        rows = sorted((dict(rec[t], name=TEAM_NAME[t], odds=odds.get(t)) for t in teams),
                      key=lambda x: (-x["pct"], -(x["div"][0] - x["div"][1]), -x["conf_pct"], -x["diff"]))
        divs.append({"division": d, "conference": d.split()[0], "teams": rows})
    return clean({"season": season, "divisions": divs, "picture": _picture(rec)})


def _snapshot(week: int):
    """Playoff outlook as it looked after a given week (0 = before the season)."""
    cs = int(S["games"].season.max())
    raw = load_raw_games()
    mask = (raw.season == cs) & (raw.week > week)
    raw.loc[mask, ["home_score", "away_score", "home_win"]] = np.nan
    raw.loc[mask, "completed"] = False
    elo = EloModel()
    g = elo.run(raw)
    g["vegas_home_prob"] = np.nan
    f = build_features(g, S["team_games"], S["qb_games"], S["inj_values"])
    cur = f.season == cs
    f.loc[cur, "model_home_prob"] = S["model"].predict_proba(f.loc[cur, __import__("model").FEATURES])[:, 1]
    sim = seasonsim.simulate(f, cs, elo.ratings)
    actual = seasonsim.actual_bracket(f, cs, elo.ratings)
    return {"sim": sim, "actual": actual}


def _weeks_done():
    g = S["games"]
    cs = int(g.season.max())
    d = g[(g.season == cs)]
    done = []
    for w in sorted(d.week.unique()):
        wk = d[d.week == w]
        if wk.completed.all():
            done.append(int(w))
    return cs, done


@app.get("/api/playoffs")
async def api_playoffs(week: int | None = None):
    cs, done = _weeks_done()
    labels = {19: "After the Wild Card round", 20: "After the Divisional round", 21: "After the Conference Championships",
              22: "After the Super Bowl"}
    options = [{"week": 0, "label": "Before the season"}] + [{"week": w, "label": labels.get(w, f"After week {w}")} for w in done]
    latest = {"week": None, "label": "Right now"}
    if week is None:
        snap, label = {"sim": S["sim"], "actual": S["actual"]}, "Right now"
    else:
        if week not in S["snapshots"]:
            S["snapshots"][week] = await asyncio.to_thread(_snapshot, week)
        snap, label = S["snapshots"][week], next((o["label"] for o in options if o["week"] == week), f"After week {week}")
    bracket = snap["actual"] or snap["sim"]["projected"]
    odds = sorted(snap["sim"]["teams"].values(), key=lambda x: (-x["champion"], -x["playoffs"]))
    for o in odds:
        o["conference"] = CONF_OF[o["team"]]
    return clean({"season": cs, "week": week, "label": label, "options": options + [latest],
                  "mode": "actual" if snap["actual"] else "projected", "bracket": bracket, "odds": odds})


@app.get("/api/favorites")
def api_favorites(teams: str = ""):
    g = S["games"]
    season = int(g.season.max())
    rec = seasonsim.standings(g, season)
    out = []
    for t in [norm(x.strip().upper()) for x in teams.split(",") if x.strip()][:8]:
        if t not in rec:
            continue
        sg = g[(g.season == season) & ((g.home_team.map(norm) == t) | (g.away_team.map(norm) == t))]
        nxt = sg[~sg.completed].sort_values("gameday").head(1)
        last = sg[sg.completed].sort_values("gameday").tail(1)
        item = {"team": t, "name": TEAM_NAME.get(t, t), "record": f'{rec[t]["w"]}-{rec[t]["l"]}' + (f'-{rec[t]["t"]}' if rec[t]["t"] else ""),
                "division": DIVISION_OF[t], "outlook": S["sim"]["teams"].get(t) if S["sim"] else None}
        if len(nxt):
            r = next(nxt.itertuples()); home = norm(r.home_team) == t
            item["next"] = {"game_id": r.game_id, "week": int(r.week), "gameday": r.gameday, "gametime": r.gametime,
                            "opp": norm(r.away_team if home else r.home_team), "home": home,
                            "win_prob": float(r.model_home_prob if home else 1 - r.model_home_prob)}
        if len(last):
            r = next(last.itertuples()); home = norm(r.home_team) == t
            us, them = (r.home_score, r.away_score) if home else (r.away_score, r.home_score)
            item["last"] = {"opp": norm(r.away_team if home else r.home_team), "home": home,
                            "result": ("W" if us > them else "L" if us < them else "T") + f" {int(us)}-{int(them)}"}
        live = next((x for x in LIVE["games"] if t in (x["home"]["abbr"], x["away"]["abbr"]) and x["state"] == "in"), None)
        if live:
            home = live["home"]["abbr"] == t
            item["live"] = {"detail": live["detail"], "us": live["home" if home else "away"]["score"],
                            "them": live["away" if home else "home"]["score"],
                            "win_prob": live["live_home_prob"] if home else 1 - live["live_home_prob"]}
        out.append(item)
    return clean(out)


@app.get("/api/player/{pid}")
def api_player(pid: str):
    season = int(S["games"].season.max())
    card = players.player_card(S["player_weeks"], S["totals"], pid, season, S["games"])
    if card is None:
        raise HTTPException(404, "Player not found")
    f = S["fantasy"].get("ppr")
    if f is not None and len(f["players"]):
        m = f["players"][f["players"].id == pid]
        if len(m):
            p = m.iloc[0]
            card["next"] = {"opp": p.opp, "home": bool(p.home), "week": int(p.week), "proj": float(p.proj),
                            "verdict": p.verdict, "pos_rank": int(p.pos_rank), "reasons": p.reasons}
    rep = S["injuries"]
    if rep is not None and not rep.empty:
        latest = rep[rep.season == season].week.max()   # only the most recent report counts
        i = rep[(rep.season == season) & (rep.week == latest) & (rep.full_name == card["name"])
                & (rep.team == card["team"]) & rep.report_status.notna()]
        if len(i):
            i = i.sort_values("week").iloc[-1]
            card["injury"] = {"status": i.report_status, "injury": i.report_primary_injury, "week": int(i.week)}
    return clean(card)


@app.get("/api/roster/{abbr}")
def api_roster(abbr: str):
    return clean(players.roster(S["totals"], norm(abbr.upper())))


@app.get("/api/schedule")
def api_schedule(week: int | None = None, season: int | None = None):
    g = S["games"]
    season = season or int(g.season.max())
    sg = g[g.season == season]
    reg_weeks = sorted(int(w) for w in sg.week.unique())
    if week is None:
        week = current_week(g)[1] if season == int(g.season.max()) else reg_weeks[-1]
    wk = sg[sg.week == week].sort_values(["gameday", "gametime"])
    rows = []
    for r in wk.itertuples():
        row = game_row(r)
        row.update(weekday=r.weekday, gametime=r.gametime, stadium=r.stadium, roof=r.roof,
                   game_type=r.game_type, spread=None if pd.isna(r.spread_line) else float(r.spread_line),
                   total=None if pd.isna(r.total_line) else float(r.total_line))
        rows.append(row)
    labels = {"WC": "Wild Card", "DIV": "Divisional", "CON": "Conference", "SB": "Super Bowl"}
    weeks = [{"week": w, "label": labels.get(sg[sg.week == w].game_type.iloc[0], f"Week {w}")} for w in reg_weeks]
    return clean({"season": season, "week": week, "weeks": weeks, "games": rows})


# ---------------- news ----------------

def _local(request: Request) -> bool:
    """Stories can only be added/removed from your own computer, so visitors
    to the deployed site can't edit them. Add locally, then git push."""
    return request.client is not None and request.client.host in ("127.0.0.1", "::1", "localhost")


class Story(BaseModel):
    title: str
    url: str
    summary: str = ""
    source: str = ""
    teams: list[str] = []


@app.get("/api/news")
async def api_news(request: Request, team: str | None = None):
    async with httpx.AsyncClient() as client:
        espn = await news.espn_headlines(client)
    mine, auto = news.my_stories(), espn["items"]
    if team:
        t = norm(team.upper())
        mine = [s for s in mine if t in s.get("teams", [])]
        auto = [s for s in auto if t in s.get("teams", [])]
    return {"mine": mine, "headlines": auto, "error": espn["error"], "can_edit": _local(request)}


@app.post("/api/news")
def api_add_story(story: Story, request: Request):
    if not _local(request):
        raise HTTPException(403, "Stories can only be added from your own computer.")
    if not story.title.strip() or not story.url.startswith(("http://", "https://")):
        raise HTTPException(400, "A story needs a title and a link starting with http:// or https://.")
    return news.add_story(story.title, story.url, story.summary, story.source, story.teams)


@app.delete("/api/news/{story_id}")
def api_delete_story(story_id: str, request: Request):
    if not _local(request):
        raise HTTPException(403, "Stories can only be removed from your own computer.")
    if not news.delete_story(story_id):
        raise HTTPException(404, "That story wasn't found.")
    return {"deleted": story_id}


@app.get("/api/status")
def api_status():
    return clean({"data_updated": S["data_updated"], "live_updated": LIVE["updated"], "live_error": LIVE["error"]})


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
