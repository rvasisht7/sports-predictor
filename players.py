"""Player layer: season leaders and fantasy football projections.

Fantasy projection for a player's next game =
    recent production (recent weeks weighted more, blended with last season early on)
  x matchup   (fantasy points the opponent allows to that position vs. league average)
  x game script (points Vegas expects the player's team to score)
with injury status on top. Each factor is shrunk toward average when it's based
on only a few games, so one fluky week doesn't swing everything."""
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd

from elo import norm

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)
FANTASY_POS = ["QB", "RB", "WR", "TE"]

# Tuned on 2024-2025 weekly data (see backtest()): average miss of 5.85 PPR points
# per player-week, vs. 6.00 for just using the season average.
DECAY = 0.9             # weight of a game one week older
PRIOR_GAMES = 2.0       # last season's average counts like this many games
MATCHUP_SHRINK = 6.0    # games of "league average" added to each defense's numbers
MATCHUP_POWER = 0.3     # dampens the matchup multiplier (matchups are noisy)
ENV_POWER = 0.5         # dampens the team-total multiplier
BACKTEST = {"model": 5.85, "naive": 6.00, "seasons": "2024-2025"}
STARTER_CUTOFF = {"QB": 12, "RB": 24, "WR": 30, "TE": 12}
FLEX_CUTOFF = {"QB": 18, "RB": 36, "WR": 42, "TE": 18}

KEEP = ["player_id", "player_display_name", "position", "headshot_url", "season", "week", "season_type", "game_id",
        "team", "opponent_team", "completions", "attempts", "passing_yards", "passing_tds",
        "passing_interceptions", "passing_epa", "sacks_suffered", "carries", "rushing_yards", "rushing_tds",
        "receptions", "targets", "receiving_yards", "receiving_tds", "target_share",
        "def_tackles_solo", "def_tackle_assists", "def_tackles_for_loss", "def_sacks", "def_interceptions",
        "def_fumbles_forced", "def_pass_defended", "fg_made", "fg_att", "fg_long",
        "fantasy_points", "fantasy_points_ppr"]


def load_player_weeks(seasons, refresh=False) -> pd.DataFrame:
    frames = []
    for s in seasons:
        path = CACHE / f"player_weeks_{s}.parquet"
        if refresh or not path.exists():
            try:
                df = nfl.load_player_stats([s], summary_level="week").to_pandas()
                df[[c for c in KEEP if c in df.columns]].to_parquet(path)
            except Exception as e:
                print(f"player stats {s} failed:", e)
        if path.exists():
            frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    df = df[df.season_type == "REG"].copy()
    df["team"] = df.team.map(norm)
    df["opponent_team"] = df.opponent_team.map(norm)
    df["fp_half"] = (df.fantasy_points + df.fantasy_points_ppr) / 2
    return df


# ---------------- leaders ----------------

def _qualify(min_per_game):
    return lambda d, weeks: d >= min_per_game * weeks

LEADER_GROUPS = [
    ("Passing", [
        ("Passing yards", "passing_yards", "sum", None),
        ("Passing TDs", "passing_tds", "sum", None),
        ("Completion %", "comp_pct", "rate", ("attempts", 14)),
        ("EPA per dropback", "epa_db", "rate", ("attempts", 14)),
        ("Interceptions thrown", "passing_interceptions", "sum", None),
    ]),
    ("Rushing", [
        ("Rushing yards", "rushing_yards", "sum", None),
        ("Rushing TDs", "rushing_tds", "sum", None),
        ("Yards per carry", "ypc", "rate", ("carries", 6.25)),
        ("Carries", "carries", "sum", None),
    ]),
    ("Receiving", [
        ("Receiving yards", "receiving_yards", "sum", None),
        ("Receptions", "receptions", "sum", None),
        ("Receiving TDs", "receiving_tds", "sum", None),
        ("Targets", "targets", "sum", None),
        ("Yards per catch", "ypr", "rate", ("receptions", 1.875)),
    ]),
    ("Defense", [
        ("Sacks", "def_sacks", "sum", None),
        ("Interceptions", "def_interceptions", "sum", None),
        ("Tackles", "tackles", "sum", None),
        ("Tackles for loss", "def_tackles_for_loss", "sum", None),
        ("Forced fumbles", "def_fumbles_forced", "sum", None),
        ("Passes defended", "def_pass_defended", "sum", None),
    ]),
    ("Kicking", [
        ("Field goals made", "fg_made", "sum", None),
        ("Field goal %", "fg_pct", "rate", ("fg_att", 1)),
        ("Longest field goal", "fg_long", "max", None),
    ]),
    ("Fantasy (PPR)", [
        ("Total points", "fantasy_points_ppr", "sum", None),
        ("Points per game", "ppg", "rate", ("games", 0.5)),
    ]),
]


def season_totals(weeks: pd.DataFrame, season: int) -> pd.DataFrame:
    d = weeks[weeks.season == season]
    num = [c for c in KEEP if c in d.columns and d[c].dtype.kind in "if" and c not in ("season", "week", "target_share", "game_id")]
    g = d.groupby("player_id")
    t = g[num].sum(min_count=1)
    t["fg_long"] = g.fg_long.max()
    t["games"] = g.week.nunique()
    last = d.sort_values("week").groupby("player_id").last()
    for c in ("player_display_name", "position", "team", "headshot_url"):
        t[c] = last[c]
    t["tackles"] = t.def_tackles_solo.fillna(0) + t.def_tackle_assists.fillna(0)
    t["comp_pct"] = t.completions / t.attempts.replace(0, np.nan)
    t["epa_db"] = t.passing_epa / (t.attempts + t.sacks_suffered.fillna(0)).replace(0, np.nan)
    t["ypc"] = t.rushing_yards / t.carries.replace(0, np.nan)
    t["ypr"] = t.receiving_yards / t.receptions.replace(0, np.nan)
    t["fg_pct"] = t.fg_made / t.fg_att.replace(0, np.nan)
    t["ppg"] = t.fantasy_points_ppr / t.games
    return t.reset_index()


def leaders(totals: pd.DataFrame, team_games: int, team: str | None = None, n: int = 5) -> list[dict]:
    t = totals if team is None else totals[totals.team == team]
    out = []
    for group, stats in LEADER_GROUPS:
        cats = []
        for label, col, kind, qual in stats:
            d = t
            if qual:   # rate stats need volume; a quarter of the league bar when looking at one team
                base, per_game = qual
                d = d[d[base].fillna(0) >= per_game * team_games * (1 if team is None else 0.25)]
            d = d.dropna(subset=[col])
            d = d[d[col] > 0] if col != "epa_db" else d
            top = d.sort_values(col, ascending=False).head(n)
            if top.empty:
                continue
            pct = col in ("comp_pct", "fg_pct")
            cats.append({"stat": label, "rate": kind == "rate", "players": [{
                "id": r.player_id, "name": r.player_display_name, "pos": r.position, "team": r.team,
                "photo": r.headshot_url, "games": int(r.games),
                "value": round(float(getattr(r, col)) * (100 if pct else 1), 1 if pct or kind == "rate" else 0)
                         if col != "epa_db" else round(float(r.epa_db), 2),
                "unit": "%" if pct else ""} for r in top.itertuples()]})
        if cats:
            out.append({"group": group, "categories": cats})
    return out


# ---------------- fantasy ----------------

def defense_vs_position(weeks: pd.DataFrame, season: int, before_week: int, pts: str) -> dict:
    """Multiplier per (defense, position): how many fantasy points the defense
    allows to that position compared with the league average, shrunk toward 1."""
    d = weeks[(weeks.season == season) & (weeks.week < before_week) & weeks.position.isin(FANTASY_POS)]
    if d.empty:
        return {}
    per_game = d.groupby(["opponent_team", "position", "week"])[pts].sum().reset_index()
    league = per_game.groupby("position")[pts].mean()
    agg = per_game.groupby(["opponent_team", "position"])[pts].agg(["sum", "count"])
    out = {}
    for (dfn, pos), r in agg.iterrows():
        avg = (r["sum"] + MATCHUP_SHRINK * league[pos]) / (r["count"] + MATCHUP_SHRINK)
        out[(dfn, pos)] = {"mult": float(avg / league[pos]), "allowed": float(r["sum"] / r["count"]),
                           "league": float(league[pos])}
    return out


def player_baselines(weeks: pd.DataFrame, season: int, before_week: int, pts: str) -> pd.DataFrame:
    cur = weeks[(weeks.season == season) & (weeks.week < before_week) & weeks.position.isin(FANTASY_POS)]
    prev = weeks[(weeks.season == season - 1) & weeks.position.isin(FANTASY_POS)]
    prior = prev.groupby("player_id")[pts].mean()
    cur = cur.assign(w=DECAY ** (before_week - 1 - cur.week))
    g = cur.groupby("player_id")
    wsum = g.w.sum()
    wavg = (cur[pts] * cur.w).groupby(cur.player_id).sum() / wsum
    last = cur.sort_values("week").groupby("player_id").last()
    b = pd.DataFrame({"wavg": wavg, "wsum": wsum, "games": g.week.nunique(), "season_avg": g[pts].mean(),
                      "last_week": last.week, "name": last.player_display_name, "pos": last.position,
                      "team": last.team, "photo": last.headshot_url,
                      "tgt_share": g.target_share.mean(),
                      "tgt_share_recent": cur[cur.week >= before_week - 3].groupby("player_id").target_share.mean()})
    b["prior"] = prior.reindex(b.index)
    has_prior = b.prior.notna()
    b["base"] = b.wavg
    b.loc[has_prior, "base"] = ((b.wavg * b.wsum + b.prior * PRIOR_GAMES) / (b.wsum + PRIOR_GAMES))[has_prior]
    return b


def implied_totals(sched_rows: pd.DataFrame, all_rows: bool = False):
    """Points Vegas expects each team to score, from the total and the spread."""
    out = []
    for r in sched_rows.itertuples():
        if pd.isna(r.total_line) or pd.isna(r.spread_line):
            continue
        out.append((norm(r.home_team), r.total_line / 2 + r.spread_line / 2))
        out.append((norm(r.away_team), r.total_line / 2 - r.spread_line / 2))
    return out if all_rows else dict(out)


def usual_totals(schedules, season, week) -> dict:
    """Each team's average implied total in its earlier games this season (shrunk toward 22)."""
    past = schedules[(schedules.season == season) & (schedules.week < week) & (schedules.game_type == "REG")]
    vals = {}
    for t, v in implied_totals(past, all_rows=True):
        vals.setdefault(t, []).append(v)
    return {t: (sum(v) + 2 * 22.0) / (len(v) + 2) for t, v in vals.items()}


def project(weeks, season, week, games_for_week, pts="fantasy_points_ppr", injuries=None, schedules=None):
    """Projections for the given week. games_for_week: schedule rows (one per game)."""
    dvp = defense_vs_position(weeks, season, week, pts)
    base = player_baselines(weeks, season, week, pts)
    totals = implied_totals(games_for_week)
    usual = usual_totals(schedules, season, week) if schedules is not None else {}
    opp, home, game_info = {}, {}, {}
    # latest first, so a team's earliest listed game wins if it appears twice
    for r in games_for_week.sort_values("gameday", ascending=False).itertuples():
        h, a = norm(r.home_team), norm(r.away_team)
        opp[h], opp[a] = a, h
        home[h], home[a] = True, False
        for t in (h, a):
            game_info[t] = {"game_id": r.game_id, "week": int(r.week), "gameday": r.gameday}
    # only players still active: appeared in one of their team's last 2 games
    team_last = weeks[(weeks.season == season) & (weeks.week < week)].groupby("team").week.max()
    base = base[base.team.isin(opp.keys())]
    base = base[base.last_week >= base.team.map(team_last) - 2]
    rows = []
    for pid, b in base.iterrows():
        o = opp[b.team]
        m = dvp.get((o, b.pos), {"mult": 1.0, "allowed": None, "league": None})
        mm = m["mult"] ** MATCHUP_POWER
        tt = totals.get(b.team)
        em = (tt / usual.get(b.team, 22.0)) ** ENV_POWER if tt else 1.0
        proj = b.base * mm * em
        status = injuries.get((b["name"], b.team)) if injuries else None
        if status and status["status"] in ("Out", "Doubtful"):
            proj = 0.0
        rows.append({"id": pid, "name": b["name"], "pos": b.pos, "team": b.team, "opp": o, "home": home[b.team],
                     "photo": b.photo, **game_info[b.team],
                     "proj": round(float(proj), 1), "base": round(float(b.base), 1),
                     "season_avg": round(float(b.season_avg), 1), "games": int(b.games),
                     "matchup_mult": round(float(m["mult"]), 3), "opp_allowed": m["allowed"], "league_allowed": m["league"],
                     "team_total": None if tt is None else round(float(tt), 1), "env_mult": round(float(em), 3),
                     "tgt_share": None if pd.isna(b.tgt_share) else round(float(b.tgt_share), 3),
                     "tgt_share_recent": None if pd.isna(b.tgt_share_recent) else round(float(b.tgt_share_recent), 3),
                     "injury": status})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["pos_rank"] = df.groupby("pos").proj.rank(ascending=False, method="first").astype(int)
    df["verdict"] = [("Sit" if (r.injury and r.injury["status"] in ("Out", "Doubtful")) else
                      "Start" if r.pos_rank <= STARTER_CUTOFF[r.pos] else
                      "Flex" if r.pos_rank <= FLEX_CUTOFF[r.pos] else "Sit") for r in df.itertuples()]
    df["reasons"] = [reasons(r) for r in df.itertuples()]
    return df.sort_values("proj", ascending=False)


def reasons(r) -> list[str]:
    out = []
    if r.injury:
        out.append(f"{r.injury['status']}{' (' + r.injury['injury'] + ')' if r.injury.get('injury') else ''} on the injury report")
    if r.opp_allowed is not None and abs(r.opp_allowed / r.league_allowed - 1) >= 0.10:
        diff = (r.opp_allowed / r.league_allowed - 1) * 100
        word = "more" if diff > 0 else "fewer"
        out.append(f"{r.opp} allows {abs(diff):.0f}% {word} fantasy points to {r.pos}s than average")
    if r.team_total is not None:
        if r.env_mult >= 1.03:
            out.append(f"{r.team} expected to score {r.team_total:.1f} points, more than usual")
        elif r.env_mult <= 0.97:
            out.append(f"{r.team} expected to score only {r.team_total:.1f} points, less than usual")
    if r.tgt_share_recent and r.tgt_share and r.pos in ("WR", "TE", "RB"):
        if r.tgt_share_recent - r.tgt_share >= 0.04:
            out.append(f"Target share rising: {r.tgt_share_recent:.0%} over the last 3 weeks")
        elif r.tgt_share - r.tgt_share_recent >= 0.04:
            out.append(f"Target share falling: {r.tgt_share_recent:.0%} over the last 3 weeks")
    out.append(f"Averaging {r.season_avg:.1f} points over {r.games} game{'s' if r.games != 1 else ''}")
    return out


def backtest(weeks, schedules, seasons, pts="fantasy_points_ppr"):
    """Compare projections with simply using the player's season average."""
    errs = {"model": [], "naive": []}
    for s in seasons:
        for w in range(3, 19):
            g = schedules[(schedules.season == s) & (schedules.week == w) & (schedules.game_type == "REG")]
            if g.empty:
                continue
            p = project(weeks, s, w, g, pts, schedules=schedules)
            if p.empty:
                continue
            act = weeks[(weeks.season == s) & (weeks.week == w)].set_index("player_id")[pts]
            p = p[p.id.isin(act.index)]
            # same fantasy-relevant players every time: ranked by season average, not by projection
            p = p.assign(r=p.groupby("pos").season_avg.rank(ascending=False))
            p = p[p.r <= p.pos.map(FLEX_CUTOFF) + 12]
            a = act.reindex(p.id).values
            errs["model"].extend(np.abs(p.proj.values - a))
            errs["naive"].extend(np.abs(p.season_avg.values - a))
    return {k: round(float(np.mean(v)), 3) for k, v in errs.items()}, len(errs["model"])


# ---------------- player cards + rosters ----------------
POS_GROUP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE", "K": "K", "P": "K"}
STAT_LINES = {
    "QB": [("Pass yds", "passing_yards"), ("Pass TD", "passing_tds"), ("INT", "passing_interceptions"),
           ("Comp %", "comp_pct"), ("Rush yds", "rushing_yards"), ("Rush TD", "rushing_tds")],
    "RB": [("Carries", "carries"), ("Rush yds", "rushing_yards"), ("Rush TD", "rushing_tds"),
           ("Rec", "receptions"), ("Rec yds", "receiving_yards"), ("Rec TD", "receiving_tds")],
    "WR": [("Targets", "targets"), ("Rec", "receptions"), ("Rec yds", "receiving_yards"),
           ("Rec TD", "receiving_tds"), ("Yds/catch", "ypr"), ("Rush yds", "rushing_yards")],
    "K":  [("FG made", "fg_made"), ("FG att", "fg_att"), ("FG %", "fg_pct"), ("Long", "fg_long")],
    "DEF": [("Tackles", "tackles"), ("Sacks", "def_sacks"), ("TFL", "def_tackles_for_loss"),
            ("INT", "def_interceptions"), ("Passes def.", "def_pass_defended"), ("FF", "def_fumbles_forced")],
}
STAT_LINES["TE"] = STAT_LINES["WR"]
GAME_COLS = {
    "QB": [("C/Att", None), ("Yds", "passing_yards"), ("TD", "passing_tds"), ("INT", "passing_interceptions"), ("Rush", "rushing_yards")],
    "RB": [("Car", "carries"), ("Rush", "rushing_yards"), ("TD", "rushing_tds"), ("Rec", "receptions"), ("Rec yds", "receiving_yards")],
    "WR": [("Tgt", "targets"), ("Rec", "receptions"), ("Yds", "receiving_yards"), ("TD", "receiving_tds")],
    "K":  [("FG", None), ("Long", "fg_long")],
    "DEF": [("Tkl", "tackles"), ("Sacks", "def_sacks"), ("INT", "def_interceptions"), ("PD", "def_pass_defended")],
}
GAME_COLS["TE"] = GAME_COLS["WR"]


def group_of(pos):
    return POS_GROUP.get(pos, "DEF")


def _num(v, pct=False, dec=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    v = float(v) * (100 if pct else 1)
    if dec is None:
        dec = 1 if pct or abs(v - round(v)) > 1e-9 else 0
    return round(v, dec)


def stat_line(row, grp):
    out = []
    for label, col in STAT_LINES[grp]:
        pct = col in ("comp_pct", "fg_pct")
        out.append({"label": label, "value": _num(row.get(col), pct), "unit": "%" if pct else ""})
    return out


def roster(totals, team):
    d = totals[totals.team == team].copy()
    d["grp"] = d.position.map(group_of)
    d = d.assign(tackles=d.def_tackles_solo.fillna(0) + d.def_tackle_assists.fillna(0))
    order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "K": 4, "DEF": 5}
    out = {}
    for grp, g in sorted(d.groupby("grp"), key=lambda x: order[x[0]]):
        key = {"QB": "passing_yards", "RB": "rushing_yards", "WR": "receiving_yards", "TE": "receiving_yards",
               "K": "fg_made", "DEF": "tackles"}[grp]
        g = g.sort_values([key, "games"], ascending=False)
        g = g[g[key].fillna(0) > 0] if grp != "DEF" else g[g.tackles > 0]
        out[grp] = {"columns": [l for l, _ in STAT_LINES[grp]],
                    "players": [{"id": r["player_id"], "name": r["player_display_name"], "pos": r["position"],
                                 "photo": r["headshot_url"], "games": int(r["games"]),
                                 "stats": [s["value"] for s in stat_line(r, grp)]} for _, r in g.iterrows()]}
    return out


def player_card(weeks, totals, pid, season, schedules):
    t = totals[totals.player_id == pid]
    if t.empty:
        return None
    r = t.iloc[0].to_dict()
    grp = group_of(r["position"])
    # where they rank among everyone in each stat on their card
    ranks = []
    for label, col in STAT_LINES[grp]:
        if col in ("passing_interceptions",) or r.get(col) is None or not (r.get(col) or 0) > 0:
            continue
        pool = totals[totals[col].fillna(0) > 0]
        if col in ("comp_pct", "ypr", "fg_pct"):
            continue
        rank = int((pool[col] > r[col]).sum()) + 1
        if rank <= 32:
            ranks.append({"label": label, "rank": rank})
    g = weeks[(weeks.player_id == pid) & (weeks.season == season)].sort_values("week")
    sched = schedules.set_index("game_id") if "game_id" in schedules.columns else None
    log = []
    for _, w in g.iterrows():
        row = {"week": int(w.week), "opp": w.opponent_team, "fp": _num(w.fantasy_points_ppr, dec=1)}
        if sched is not None and "game_id" in w and w.game_id in sched.index:
            s = sched.loc[w.game_id]
            home = norm(s.home_team) == w.team
            row["home"] = bool(home)
            if pd.notna(s.home_score):
                us, them = (s.home_score, s.away_score) if home else (s.away_score, s.home_score)
                row["result"] = ("W" if us > them else "L" if us < them else "T") + f" {int(us)}-{int(them)}"
        vals = []
        for label, col in GAME_COLS[grp]:
            if label == "C/Att":
                vals.append(f"{int(w.completions or 0)}/{int(w.attempts or 0)}")
            elif label == "FG":
                vals.append(f"{int(w.fg_made or 0)}/{int(w.fg_att or 0)}")
            elif col == "tackles":
                vals.append(_num((w.def_tackles_solo or 0) + (w.def_tackle_assists or 0)))
            else:
                vals.append(_num(w.get(col)))
        row["stats"] = vals
        log.append(row)
    return {"id": pid, "name": r["player_display_name"], "pos": r["position"], "team": r["team"],
            "photo": r["headshot_url"], "games": int(r["games"]), "group": grp,
            "season_line": stat_line(r, grp), "ranks": ranks,
            "fantasy": {"total": _num(r.get("fantasy_points_ppr"), dec=1), "ppg": _num(r.get("ppg"), dec=1)},
            "log_columns": [l for l, _ in GAME_COLS[grp]], "log": log}
