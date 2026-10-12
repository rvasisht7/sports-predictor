"""Team tendencies from this season's play-by-play, and matchup keys for each game.

Every metric is ranked 1-32 across the league (1 = the most of that thing).
Matchup keys compare one offense's strengths and habits with the other
defense's weaknesses, and flag the clearest mismatches."""
from pathlib import Path

import pandas as pd
import polars as pl

from elo import norm

CACHE = Path(__file__).parent / "cache"
COLS = ["game_id", "season_type", "posteam", "defteam", "play", "pass", "rush", "qb_dropback", "qb_scramble",
        "epa", "pass_oe", "down", "ydstogo", "air_yards", "shotgun", "no_huddle", "sack", "interception",
        "fumble_lost", "yardline_100", "yards_gained", "third_down_converted", "third_down_failed",
        "fixed_drive", "fixed_drive_result", "play_type", "run_location", "wp", "half_seconds_remaining"]

# key: (label, side, higher_is_good_for_that_side, format)
METRICS = {
    "proe": ("Pass rate over expected", "off", None, "pct_signed"),
    "early_pass": ("Early-down pass rate", "off", None, "pct"),
    "deep_rate": ("Deep pass rate (20+ air yards)", "off", None, "pct"),
    "outside_run": ("Runs to the outside", "off", None, "pct"),
    "no_huddle": ("No-huddle rate", "off", None, "pct"),
    "go_rate": ("4th-down go rate (short yardage)", "off", None, "pct"),
    "pass_epa": ("EPA per dropback", "off", True, "epa"),
    "rush_epa": ("EPA per rush", "off", True, "epa"),
    "explosive": ("Explosive play rate", "off", True, "pct"),
    "third": ("3rd-down conversion rate", "off", True, "pct"),
    "rz_td": ("Red-zone touchdown rate", "off", True, "pct"),
    "sack_allowed": ("Sack rate allowed", "off", False, "pct"),
    "giveaways": ("Giveaways per game", "off", False, "num"),
    "d_pass_epa": ("EPA per dropback allowed", "def", False, "epa"),
    "d_rush_epa": ("EPA per rush allowed", "def", False, "epa"),
    "d_explosive": ("Explosive plays allowed", "def", False, "pct"),
    "d_sack": ("Sack rate", "def", True, "pct"),
    "d_takeaways": ("Takeaways per game", "def", True, "num"),
    "d_third": ("3rd-down conversions allowed", "def", False, "pct"),
    "d_rz_td": ("Red-zone TD rate allowed", "def", False, "pct"),
}


def compute(season: int) -> pd.DataFrame:
    path = CACHE / f"pbp_{season}.parquet"
    if not path.exists():
        return pd.DataFrame()
    have = set(pl.scan_parquet(path).collect_schema().names())
    d = (pl.scan_parquet(path).select([c for c in COLS if c in have])
         .filter(pl.col("season_type") == "REG").collect().to_pandas())
    d = d[d.posteam.notna()].copy()
    d["posteam"] = d.posteam.map(norm); d["defteam"] = d.defteam.map(norm)
    plays = d[(d.play == 1) & ((d["pass"] == 1) | (d.rush == 1))]
    neutral = plays[(plays.wp.between(0.2, 0.8)) & (plays.half_seconds_remaining > 120)]
    drop = plays[plays.qb_dropback == 1]
    runs = plays[(plays.rush == 1) & (plays.qb_scramble != 1)]
    att = plays[(plays["pass"] == 1) & plays.air_yards.notna()]
    games = plays.groupby("posteam").game_id.nunique()
    dgames = plays.groupby("defteam").game_id.nunique()
    plays = plays.assign(expl=((plays["pass"] == 1) & (plays.yards_gained >= 20)) | ((plays.rush == 1) & (plays.yards_gained >= 10)),
                         give=plays.interception.fillna(0) + plays.fumble_lost.fillna(0))
    third = d[(d.down == 3) & (d.third_down_converted.notna())]
    fourth = d[(d.down == 4) & (d.ydstogo <= 2) & (d.yardline_100 <= 60) & (d.wp.between(0.1, 0.9))
               & d.play_type.isin(["pass", "run", "punt", "field_goal"])]
    drives = d[d.fixed_drive.notna()].groupby(["game_id", "posteam", "defteam", "fixed_drive"]).agg(
        min_yl=("yardline_100", "min"), result=("fixed_drive_result", "first")).reset_index()
    rz = drives[drives.min_yl <= 20].assign(td=lambda x: x.result == "Touchdown")

    def by(df, team_col, expr):
        return df.groupby(team_col).apply(expr, include_groups=False)

    t = pd.DataFrame(index=sorted(games.index))
    t["proe"] = neutral.groupby("posteam").pass_oe.mean() / 100
    t["early_pass"] = neutral[neutral.down <= 2].groupby("posteam")["pass"].mean()
    t["deep_rate"] = att.groupby("posteam").air_yards.apply(lambda s: (s >= 20).mean())
    t["outside_run"] = runs.groupby("posteam").run_location.apply(lambda s: s.isin(["left", "right"]).mean())
    t["no_huddle"] = plays.groupby("posteam").no_huddle.mean()
    go = fourth.groupby("posteam").play_type.agg(lambda s: s.isin(["pass", "run"]).mean())
    chances = fourth.groupby("posteam").size()
    t["go_rate"] = go.where(chances >= 4)   # too few chances to judge otherwise
    t["pass_epa"] = drop.groupby("posteam").epa.mean()
    t["rush_epa"] = runs.groupby("posteam").epa.mean()
    t["explosive"] = plays.groupby("posteam").expl.mean()
    t["third"] = third.groupby("posteam").third_down_converted.mean()
    t["rz_td"] = rz.groupby("posteam").td.mean()
    t["sack_allowed"] = drop.groupby("posteam").sack.mean()
    t["giveaways"] = plays.groupby("posteam").give.sum() / games
    t["d_pass_epa"] = drop.groupby("defteam").epa.mean()
    t["d_rush_epa"] = runs.groupby("defteam").epa.mean()
    t["d_explosive"] = plays.groupby("defteam").expl.mean()
    t["d_sack"] = drop.groupby("defteam").sack.mean()
    t["d_takeaways"] = plays.groupby("defteam").give.sum() / dgames
    t["d_third"] = third.groupby("defteam").third_down_converted.mean()
    t["d_rz_td"] = rz.groupby("defteam").td.mean()
    return t


def ranks(t: pd.DataFrame) -> pd.DataFrame:
    """1 = most of that thing in the league."""
    return t.rank(ascending=False, method="min")


def _fmt(v, kind):
    if v is None or pd.isna(v):
        return "n/a"
    v = round(float(v), 2) + 0.0   # avoids "-0.00"
    return {"pct": f"{v:.0%}", "pct_signed": f"{v:+.0%}", "epa": f"{v:+.2f}", "num": f"{v:.1f}"}[kind]


def team_profile(t: pd.DataFrame, team: str) -> dict:
    if t.empty or team not in t.index:
        return {}
    r = ranks(t)
    out = {"off": [], "def": []}
    for k, (label, side, good, kind) in METRICS.items():
        rank = int(r.loc[team, k]) if pd.notna(r.loc[team, k]) else None
        quality = None
        if good is not None and rank is not None:   # convert to "how good" rank: 1 = best
            quality = rank if good else 33 - rank
        out[side].append({"key": k, "label": label, "value": _fmt(t.loc[team, k], kind), "rank": rank,
                          "good_rank": quality, "style": good is None})
    out["summary"] = style_summary(t, r, team)
    return out


def _ord(n):
    n = int(n)
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _most(n):
    return "the most" if int(n) == 1 else f"the {_ord(n)}-most"


def _worst(n):
    return "the worst" if int(n) == 1 else f"the {_ord(n)}-worst"


def style_summary(t, r, team) -> list[str]:
    s = []
    p = t.loc[team, "proe"]
    if r.loc[team, "proe"] <= 6:
        s.append(f"One of the most pass-heavy offenses: throws {p:+.0%} more than expected in neutral situations ({_ord(r.loc[team, 'proe'])}).")
    elif r.loc[team, "proe"] >= 27:
        s.append(f"Run-first offense: passes {abs(p):.0%} less than expected in neutral situations ({_ord(33 - r.loc[team, 'proe'])} most run-heavy).")
    if r.loc[team, "deep_rate"] <= 5:
        s.append(f"Takes deep shots often: {t.loc[team, 'deep_rate']:.0%} of throws travel 20+ yards ({_ord(r.loc[team, 'deep_rate'])}).")
    if r.loc[team, "go_rate"] <= 5:
        s.append(f"Among the most aggressive on 4th-and-short: has gone for it {t.loc[team, 'go_rate']:.0%} of the time in reachable spots.")
    if r.loc[team, "no_huddle"] <= 4:
        s.append(f"Uses no-huddle a lot ({t.loc[team, 'no_huddle']:.0%} of plays, {_ord(r.loc[team, 'no_huddle'])}).")
    if r.loc[team, "outside_run"] <= 4:
        s.append(f"Likes to run outside: {t.loc[team, 'outside_run']:.0%} of designed runs go to the edges.")
    if r.loc[team, "d_sack"] <= 5:
        s.append(f"Defense gets home: sacks on {t.loc[team, 'd_sack']:.1%} of dropbacks ({_ord(r.loc[team, 'd_sack'])}).")
    if r.loc[team, "d_takeaways"] <= 5:
        s.append(f"Ball-hawking defense: {t.loc[team, 'd_takeaways']:.1f} takeaways per game ({_ord(r.loc[team, 'd_takeaways'])}).")
    return s


def matchup_keys(t: pd.DataFrame, off: str, dfn: str, limit: int = 4) -> list[dict]:
    """What the offense will likely try against this defense, and where the mismatches are."""
    if t.empty or off not in t.index or dfn not in t.index:
        return []
    r = ranks(t)
    keys = []

    def good(k):   # 1 = best at it
        return int(r.loc[off, k] if METRICS[k][2] else 33 - r.loc[off, k])

    def weak(k):   # 1 = worst defense at it
        return int(33 - r.loc[dfn, k] if METRICS[k][2] else r.loc[dfn, k])

    pairs = [
        ("pass_epa", "d_pass_epa", "Air attack", "{o}'s passing game ({og} in EPA per dropback) meets a defense allowing {dwm} EPA per dropback. Expect {o} to lean on the pass.",
         "{d}'s pass defense ({dg} best) is a tough test for {o}'s passing game ({oww})."),
        ("rush_epa", "d_rush_epa", "Ground game", "{o} runs efficiently ({og} in EPA per rush) and {d} has struggled against the run ({dww}). Expect a heavy dose of the run.",
         "{o}'s run game ({oww}) runs into a stout {d} run defense ({dg} best)."),
        ("explosive", "d_explosive", "Big plays", "{o} hits explosive plays often ({og}) and {d} gives them up ({dwm} allowed). Watch for shots downfield.", None),
        ("third", "d_third", "Third downs", "{o} converts 3rd downs well ({og}) against a {d} defense that allows conversions ({dww}).", None),
        ("rz_td", "d_rz_td", "Red zone", "{o} finishes drives ({og} in red-zone TD rate) and {d} struggles to hold in the red zone ({dww}).", None),
    ]
    for ok, dk, title, pos, neg in pairs:
        og, dw = good(ok), weak(dk)
        if og <= 10 and dw <= 10:
            keys.append({"title": title, "edge": off, "strength": 22 - og - dw,
                         "text": pos.format(o=off, d=dfn, og=_ord(og), dwm=_most(dw), dww=_worst(dw))})
        elif neg and og >= 23 and dw >= 23:
            keys.append({"title": title, "edge": dfn, "strength": og + dw - 44,
                         "text": neg.format(o=off, d=dfn, oww=_worst(33 - og), dg=_ord(33 - dw))})
    # pressure: defense that sacks a lot vs offense that allows sacks
    if r.loc[dfn, "d_sack"] <= 10 and r.loc[off, "sack_allowed"] <= 10:
        keys.append({"title": "Pressure", "edge": dfn, "strength": 22 - r.loc[dfn, "d_sack"] - r.loc[off, "sack_allowed"],
                     "text": f"{dfn}'s pass rush ({_ord(r.loc[dfn, 'd_sack'])} in sack rate) faces a {off} line that allows {_most(r.loc[off, 'sack_allowed'])} sacks. Pressure could decide drives."})
    # turnovers
    if r.loc[dfn, "d_takeaways"] <= 10 and r.loc[off, "giveaways"] <= 10:
        keys.append({"title": "Ball security", "edge": dfn, "strength": 22 - r.loc[dfn, "d_takeaways"] - r.loc[off, "giveaways"],
                     "text": f"{off} turns it over a lot ({_most(r.loc[off, 'giveaways'])} giveaways) and {dfn} takes it away ({_ord(r.loc[dfn, 'd_takeaways'])} in takeaways)."})
    # style notes that shape the plan
    if r.loc[off, "deep_rate"] <= 6 and weak("d_explosive") <= 12:
        keys.append({"title": "Deep shots", "edge": off, "strength": 8,
                     "text": f"{off} throws deep often ({_ord(r.loc[off, 'deep_rate'])} in deep-pass rate) at a defense that has allowed big plays."})
    if r.loc[off, "proe"] >= 27 and weak("d_rush_epa") <= 12:
        keys.append({"title": "Run-first plan", "edge": off, "strength": 7,
                     "text": f"{off} prefers to run, and {dfn}'s run defense gives it reason to keep doing so."})
    if r.loc[off, "go_rate"] <= 6:
        keys.append({"title": "4th downs", "edge": off, "strength": 4,
                     "text": f"{off} has gone for it on {t.loc[off, 'go_rate']:.0%} of 4th-and-short chances, among the most aggressive in the league. Expect it to go for it in plus territory."})
    keys.sort(key=lambda k: -k["strength"])
    for k in keys:
        k["offense"] = off
        k.pop("strength")
    return keys[:limit]
