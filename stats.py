"""Advanced stats layer: per-game team EPA, per-game QB EPA, and injury impact.

Past seasons are condensed into small files in data/ that ship with the code.
Only the current season is downloaded when the app refreshes."""
from pathlib import Path

import httpx
import nflreadpy as nfl
import pandas as pd
import polars as pl

from elo import norm

ROOT = Path(__file__).parent
HIST = ROOT / "data"          # committed to git (small)
CACHE = ROOT / "cache"        # not committed
HIST.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.parquet"
FIRST_SEASON = 2006
INJURY_FIRST_SEASON = 2012


# ---------------- play-by-play -> per-game EPA ----------------

def _pbp_season(season: int, keep: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = CACHE / f"pbp_{season}.parquet"
    with httpx.stream("GET", PBP_URL.format(season), follow_redirects=True, timeout=180) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    lf = pl.scan_parquet(path).filter(
        (pl.col("season_type") == "REG") | (pl.col("season_type") == "POST"))
    plays = lf.filter((pl.col("play") == 1) & ((pl.col("pass") == 1) | (pl.col("rush") == 1))
                      & pl.col("epa").is_not_null() & pl.col("posteam").is_not_null())
    team = (plays.group_by(["game_id", "season", "week", "posteam", "defteam"])
            .agg(pl.col("epa").mean().alias("off_epa"), pl.col("success").mean().alias("off_sr"),
                 pl.col("epa").filter(pl.col("pass") == 1).mean().alias("pass_epa"),
                 pl.col("epa").filter(pl.col("rush") == 1).mean().alias("rush_epa"),
                 pl.len().alias("plays"))
            .collect().to_pandas())
    qb = (lf.filter((pl.col("qb_dropback") == 1) & pl.col("id").is_not_null()
                    & pl.col("qb_epa").is_not_null())
          .group_by(["game_id", "season", "week", "posteam", "id"])
          .agg(pl.col("name").first().alias("qb_name"), pl.len().alias("dropbacks"),
               pl.col("qb_epa").sum().alias("qb_epa_sum"))
          .collect().to_pandas().rename(columns={"id": "qb_id"}))
    if not keep:
        path.unlink(missing_ok=True)
    for df in (team, qb):
        df["posteam"] = df["posteam"].map(norm)
    team["defteam"] = team["defteam"].map(norm)
    return team, qb


def build_history(last_season: int):
    """One-time build of past seasons (run: python stats.py). Takes a few minutes."""
    teams, qbs = [], []
    for s in range(FIRST_SEASON, last_season + 1):
        t, q = _pbp_season(s)
        teams.append(t); qbs.append(q)
        print(f"  pbp {s}: {len(t)} team-games")
    pd.concat(teams).to_parquet(HIST / "team_games.parquet")
    pd.concat(qbs).to_parquet(HIST / "qb_games.parquet")
    inj = load_injury_values(range(INJURY_FIRST_SEASON, last_season + 1))
    inj.to_parquet(HIST / "injury_values.parquet")
    print("  injuries:", len(inj), "team-weeks")


def load_epa(current_season: int, refresh: bool = False):
    """History from data/ plus the current season (downloaded and cached)."""
    t = pd.read_parquet(HIST / "team_games.parquet")
    q = pd.read_parquet(HIST / "qb_games.parquet")
    ct, cq = CACHE / f"team_games_{current_season}.parquet", CACHE / f"qb_games_{current_season}.parquet"
    if current_season > t.season.max():
        if refresh or not ct.exists():
            try:
                team, qb = _pbp_season(current_season, keep=True)   # kept for team tendencies
                team.to_parquet(ct); qb.to_parquet(cq)
            except Exception as e:
                print("pbp refresh failed:", e)
        if ct.exists():
            t = pd.concat([t, pd.read_parquet(ct)])
            q = pd.concat([q, pd.read_parquet(cq)])
    return t, q


# ---------------- injuries ----------------
# How much a missing starter matters, relative to each other (QBs are handled
# separately by the QB model). Scaled by how many snaps the player usually plays.
POS_WEIGHT = {"WR": 1.0, "TE": 0.6, "RB": 0.5, "FB": 0.1, "T": 0.8, "OT": 0.8, "G": 0.5, "OG": 0.5,
              "C": 0.5, "DE": 0.8, "EDGE": 0.8, "OLB": 0.6, "DT": 0.6, "NT": 0.5, "DL": 0.6,
              "LB": 0.4, "ILB": 0.4, "MLB": 0.4, "CB": 0.8, "DB": 0.5, "S": 0.5, "FS": 0.5, "SS": 0.5,
              "K": 0.2, "P": 0.1}
OUT = {"Out", "Doubtful"}


def _key(name: str) -> str:
    n = str(name).lower().replace(".", "").replace("'", "")
    for suf in (" jr", " sr", " iii", " ii", " iv"):
        n = n.replace(suf, "")
    return " ".join(n.split())


def injury_report(seasons) -> pd.DataFrame:
    """Players on the final injury report each week, with their usual snap share
    from earlier weeks and an impact value."""
    seasons = list(seasons)
    inj = nfl.load_injuries(seasons).to_pandas()
    snaps = nfl.load_snap_counts(seasons).to_pandas()
    inj = inj[inj.game_type == "REG"].copy()
    snaps["share"] = snaps[["offense_pct", "defense_pct"]].max(axis=1).fillna(0)
    snaps["key"] = snaps.player.map(_key)
    snaps["team"] = snaps.team.map(norm)
    snaps = snaps.sort_values(["season", "week"])
    inj["key"] = inj.full_name.map(_key)
    inj["team"] = inj.team.map(norm)
    # usual snap share = average over the player's prior games this season
    rows = []
    for (s, team, key), grp in inj.groupby(["season", "team", "key"]):
        hist = snaps[(snaps.season == s) & (snaps.team == team) & (snaps.key == key)]
        for r in grp.itertuples():
            prior = hist[hist.week < r.week]
            rows.append((r.Index, prior.share.mean() if len(prior) else 0.0, len(prior)))
    share = pd.DataFrame(rows, columns=["idx", "usual_snap_share", "games_played"]).set_index("idx")
    inj = inj.join(share)
    inj["impact"] = inj.position.map(POS_WEIGHT).fillna(0.3) * inj.usual_snap_share.fillna(0)
    inj.loc[inj.position == "QB", "impact"] = 0.0  # handled by the QB model
    return inj


def load_injury_values(seasons) -> pd.DataFrame:
    """Total impact of Out/Doubtful players per team-week."""
    inj = injury_report(seasons)
    out = inj[inj.report_status.isin(OUT)]
    return out.groupby(["season", "week", "team"]).impact.sum().rename("inj_out").reset_index()


def load_injuries_current(season: int, refresh: bool = False):
    """Current season: full report (for display) and team-week totals (for the model)."""
    path = CACHE / f"injuries_{season}.parquet"
    if refresh or not path.exists():
        try:
            injury_report([season]).to_parquet(path)
        except Exception as e:
            print("injury refresh failed:", e)
    hist = pd.read_parquet(HIST / "injury_values.parquet")
    if not path.exists():
        return pd.DataFrame(), hist
    rep = pd.read_parquet(path)
    cur = (rep[rep.report_status.isin(OUT)].groupby(["season", "week", "team"])
           .impact.sum().rename("inj_out").reset_index())
    return rep, pd.concat([hist[hist.season != season], cur])


if __name__ == "__main__":
    import sys
    build_history(int(sys.argv[1]) if len(sys.argv) > 1 else 2025)
