"""Standings, season simulation and playoff bracket.

simulate() plays the rest of the regular season many times using the model's
win probability for each remaining game, seeds each conference the NFL way
(4 division winners seeded 1-4, then 3 wild cards; the 1 seed gets a bye; teams
are re-seeded each round), then plays out the playoffs with Elo ratings.
Tiebreakers are simplified: win %, then conference win %, then a coin flip."""
import numpy as np
import pandas as pd

from elo import norm
from teams import CONF_OF, DIVISION_OF, DIVISIONS

TEAMS = sorted(DIVISION_OF)
IDX = {t: i for i, t in enumerate(TEAMS)}
HOME_ADV = 48.0
POST_TYPES = ["WC", "DIV", "CON", "SB"]
ROUND_NAMES = {"WC": "Wild Card", "DIV": "Divisional", "CON": "Conference Championship", "SB": "Super Bowl"}


# ---------------- standings ----------------

def standings(games: pd.DataFrame, season: int) -> dict:
    rec = {t: {"team": t, "w": 0, "l": 0, "t": 0, "pf": 0, "pa": 0, "home": [0, 0, 0], "away": [0, 0, 0],
               "home_diff": 0, "away_diff": 0, "div": [0, 0, 0], "conf": [0, 0, 0], "results": []} for t in TEAMS}
    d = games[(games.season == season) & (games.game_type == "REG") & games.completed].sort_values("gameday")
    for r in d.itertuples():
        h, a = norm(r.home_team), norm(r.away_team)
        if h not in rec or a not in rec:
            continue
        for team, us, them, where in ((h, r.home_score, r.away_score, "home"), (a, r.away_score, r.home_score, "away")):
            x = rec[team]
            k = 0 if us > them else 1 if us < them else 2
            x["wlt"[k]] += 1
            x["pf"] += int(us); x["pa"] += int(them)
            x[where][k] += 1
            x[where + "_diff"] += int(us) - int(them)
            other = a if team == h else h
            if DIVISION_OF[other] == DIVISION_OF[team]:
                x["div"][k] += 1
            if CONF_OF[other] == CONF_OF[team]:
                x["conf"][k] += 1
            x["results"].append("WLT"[k])
    for x in rec.values():
        g = x["w"] + x["l"] + x["t"]
        x["pct"] = (x["w"] + 0.5 * x["t"]) / g if g else 0.0
        x["diff"] = x["pf"] - x["pa"]
        res = x.pop("results")
        x["last5"] = "".join(res[-5:])
        if res:
            n = 1
            while n < len(res) and res[-n - 1] == res[-1]:
                n += 1
            x["streak"] = f"{res[-1]}{n}"
        else:
            x["streak"] = ""
        cg = sum(x["conf"])
        x["conf_pct"] = (x["conf"][0] + 0.5 * x["conf"][2]) / cg if cg else 0.0
    return rec


def _wins_float(rec):
    return np.array([rec[t]["w"] + 0.5 * rec[t]["t"] for t in TEAMS])


# ---------------- simulation ----------------

def elo_prob(rh, ra, neutral=False):
    return 1 / (1 + 10 ** (-(rh - ra + (0 if neutral else HOME_ADV)) / 400))


def _seed(score, n):
    """score: (n, 32) array (wins + small tiebreak noise). Returns (n, 2, 7) team indexes by seed."""
    out = np.zeros((n, 2, 7), dtype=int)
    for ci, conf in enumerate(("AFC", "NFC")):
        divs = [d for d in DIVISIONS if d.startswith(conf)]
        winners = np.stack([np.array([IDX[t] for t in DIVISIONS[d]])[np.argmax(score[:, [IDX[t] for t in DIVISIONS[d]]], axis=1)]
                            for d in divs], axis=1)                       # (n, 4)
        wscore = np.take_along_axis(score, winners, axis=1)
        order = np.argsort(-wscore, axis=1)
        out[:, ci, :4] = np.take_along_axis(winners, order, axis=1)
        conf_teams = np.array([IDX[t] for t in TEAMS if CONF_OF[t] == conf])
        cs = score[:, conf_teams].copy()
        is_winner = (conf_teams[None, :, None] == winners[:, None, :]).any(axis=2)
        cs[is_winner] = -1e9
        wc = np.argsort(-cs, axis=1)[:, :3]
        out[:, ci, 4:] = conf_teams[wc]
    return out


def _play(home, away, ratings, rng, neutral=False):
    p = elo_prob(ratings[home], ratings[away], neutral)
    return np.where(rng.random(home.shape) < p, home, away)


def _playoffs(seeds, ratings, rng):
    """seeds: (n, 2, 7). Returns dict of (n, ...) arrays of teams reaching each round."""
    n = seeds.shape[0]
    reach = {"DIV": [], "CON": [], "SB": []}
    champs = []
    seednum = np.tile(np.arange(1, 8), (n, 1))
    for ci in range(2):
        s = seeds[:, ci, :]
        # wild card: 2v7, 3v6, 4v5 (higher seed hosts)
        survivors, ssn = [s[:, 0]], [seednum[:, 0]]
        for hi, lo in ((1, 6), (2, 5), (3, 4)):
            w = _play(s[:, hi], s[:, lo], ratings, rng)
            survivors.append(w); ssn.append(np.where(w == s[:, hi], hi + 1, lo + 1))
        st, sn = np.stack(survivors, 1), np.stack(ssn, 1)
        reach["DIV"].append(st)
        o = np.argsort(sn, axis=1); st, sn = np.take_along_axis(st, o, 1), np.take_along_axis(sn, o, 1)
        # divisional (re-seeded): best seed vs worst remaining, other two
        w1 = _play(st[:, 0], st[:, 3], ratings, rng)
        w2 = _play(st[:, 1], st[:, 2], ratings, rng)
        n1 = np.where(w1 == st[:, 0], sn[:, 0], sn[:, 3]); n2 = np.where(w2 == st[:, 1], sn[:, 1], sn[:, 2])
        reach["CON"].append(np.stack([w1, w2], 1))
        home = np.where(n1 < n2, w1, w2); away = np.where(n1 < n2, w2, w1)
        champs.append(_play(home, away, ratings, rng))
    reach["SB"] = [np.stack(champs, 1)]
    sb = _play(champs[0], champs[1], ratings, rng, neutral=True)
    return {k: np.concatenate(v, 1) for k, v in reach.items()}, sb


def simulate(games: pd.DataFrame, season: int, ratings: dict, n: int = 10000, seed: int = 7) -> dict:
    """games must have model_home_prob for unplayed games."""
    rng = np.random.default_rng(seed)
    rec = standings(games, season)
    base = _wins_float(rec)
    conf_pct = np.array([rec[t]["conf_pct"] for t in TEAMS])
    rem = games[(games.season == season) & (games.game_type == "REG") & ~games.completed]
    hi = np.array([IDX[norm(t)] for t in rem.home_team]); ai = np.array([IDX[norm(t)] for t in rem.away_team])
    p = rem.model_home_prob.values.astype(float)
    m = len(rem)
    if m:
        hw = (rng.random((n, m)) < p).astype(np.float32)
        H = np.zeros((m, 32), np.float32); H[np.arange(m), hi] = 1
        A = np.zeros((m, 32), np.float32); A[np.arange(m), ai] = 1
        wins = base[None, :] + hw @ H + (1 - hw) @ A
    else:
        wins = np.tile(base, (n, 1))
    score = wins + conf_pct[None, :] * 1e-3 + rng.random((n, 32)) * 1e-6
    seeds = _seed(score, n)
    r = np.array([ratings.get(t, 1500.0) for t in TEAMS])
    reach, sb = _playoffs(seeds, r, rng)

    def share(arr):
        return np.bincount(arr.ravel(), minlength=32) / n
    made = share(seeds.reshape(n, -1))
    out = {}
    for t, i in IDX.items():
        out[t] = {"team": t, "proj_wins": float(wins[:, i].mean()), "proj_losses": float(17 - wins[:, i].mean()),
                  "playoffs": float(made[i]), "division": float(share(seeds[:, :, :4])[i]),
                  "bye": float(share(seeds[:, :, 0])[i]), "div_round": float(share(reach["DIV"])[i]),
                  "conf_game": float(share(reach["CON"])[i]), "super_bowl": float(share(reach["SB"])[i]),
                  "champion": float(np.bincount(sb, minlength=32)[i] / n)}
    return {"teams": out, "projected": _projected_bracket(out, r)}


def _projected_bracket(odds: dict, r) -> dict:
    """The single most likely bracket: division winners by division-title chance,
    wild cards by playoff chance, then the favorite advances in every game."""
    confs = {}
    for conf in ("AFC", "NFC"):
        winners = [max(DIVISIONS[d], key=lambda t: odds[t]["division"]) for d in DIVISIONS if d.startswith(conf)]
        winners.sort(key=lambda t: -odds[t]["proj_wins"])
        rest = sorted((t for t in TEAMS if CONF_OF[t] == conf and t not in winners),
                      key=lambda t: (-odds[t]["playoffs"], -odds[t]["proj_wins"]))[:3]
        rest.sort(key=lambda t: -odds[t]["proj_wins"])
        confs[conf] = winners + rest
    return bracket_from_seeds(confs, r, results={})


def _game(home, away, hs, as_, r, neutral=False, result=None):
    p = float(elo_prob(r[IDX[home]], r[IDX[away]], neutral))
    g = {"home": home, "away": away, "home_seed": hs, "away_seed": as_, "home_prob": round(p, 3)}
    if result:
        g.update(result)
        g["winner"] = home if result["home_score"] > result["away_score"] else away
    else:
        g["winner"] = home if p >= 0.5 else away
        g["projected"] = True
    return g


def bracket_from_seeds(confs: dict, r, results: dict) -> dict:
    """confs: {"AFC": [7 teams by seed], ...}. results: {(home, away): {...scores}} for played games."""
    out = {"conferences": {}}
    finalists = {}
    for conf, seeds in confs.items():
        seed_of = {t: i + 1 for i, t in enumerate(seeds)}
        rounds = {}
        wc = [_game(seeds[h], seeds[l], h + 1, l + 1, r, result=results.get((seeds[h], seeds[l])))
              for h, l in ((1, 6), (2, 5), (3, 4))]
        rounds["WC"] = wc
        alive = sorted([seeds[0]] + [g["winner"] for g in wc], key=lambda t: seed_of[t])
        div = []
        for h, l in ((alive[0], alive[3]), (alive[1], alive[2])):
            div.append(_game(h, l, seed_of[h], seed_of[l], r, result=results.get((h, l))))
        rounds["DIV"] = div
        a, b = sorted([g["winner"] for g in div], key=lambda t: seed_of[t])
        con = _game(a, b, seed_of[a], seed_of[b], r, result=results.get((a, b)))
        rounds["CON"] = [con]
        finalists[conf] = (con["winner"], seed_of[con["winner"]])
        out["conferences"][conf] = {"seeds": [{"seed": i + 1, "team": t} for i, t in enumerate(seeds)], "rounds": rounds}
    (afc, as_), (nfc, ns) = finalists["AFC"], finalists["NFC"]
    res = results.get((afc, nfc)) or results.get((nfc, afc))
    if res and results.get((nfc, afc)):
        sb = _game(nfc, afc, ns, as_, r, neutral=True, result=res)
    else:
        sb = _game(afc, nfc, as_, ns, r, neutral=True, result=res)
    out["super_bowl"] = sb
    out["champion"] = sb["winner"]
    return out


def actual_bracket(games: pd.DataFrame, season: int, ratings: dict):
    """Once the real playoffs are set, build the bracket from the actual matchups and results."""
    post = games[(games.season == season) & games.game_type.isin(POST_TYPES)]
    wc = post[post.game_type == "WC"]
    if len(wc) < 6:
        return None
    rec = standings(games, season)
    results = {(norm(g.home_team), norm(g.away_team)): {"home_score": int(g.home_score), "away_score": int(g.away_score),
                                                          "final": True}
               for g in post.itertuples() if g.completed}
    confs = {}
    for conf in ("AFC", "NFC"):
        games_c = [g for g in wc.itertuples() if CONF_OF[norm(g.home_team)] == conf]
        homes = sorted((norm(g.home_team) for g in games_c), key=lambda t: (-rec[t]["pct"], -rec[t]["conf_pct"]))
        opp = {norm(g.home_team): norm(g.away_team) for g in games_c}
        in_wc = set(homes) | set(opp.values())
        # the 1 seed skipped the wild card round: it shows up in the divisional round if that's scheduled,
        # otherwise it's the best record among conference teams that didn't play a wild card game
        div_teams = {norm(t) for g in post[post.game_type == "DIV"].itertuples() for t in (g.home_team, g.away_team)}
        bye = [t for t in div_teams if CONF_OF[t] == conf and t not in in_wc]
        top = bye[0] if bye else max((t for t in TEAMS if CONF_OF[t] == conf and t not in in_wc),
                                     key=lambda t: (rec[t]["pct"], rec[t]["conf_pct"]))
        seeds = [top] + homes + [opp[h] for h in reversed(homes)]
        confs[conf] = seeds
    r = np.array([ratings.get(t, 1500.0) for t in TEAMS])
    return bracket_from_seeds(confs, r, results)
