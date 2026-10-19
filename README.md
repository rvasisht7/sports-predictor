# Sports Predictor

NFL win probabilities with a live dashboard: live scores and in-game win chances,
weekly picks with a breakdown of why, team pages, season stat leaders, fantasy start/sit projections, a graded prediction tracker,
injury reports, standings, playoff odds with a week-by-week bracket, team tendencies,
favorite teams and notifications.

## Run locally
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    uvicorn app:app --reload
Open http://127.0.0.1:8000

## The model (model.py)
Logistic regression trained walk-forward (each season predicted only from earlier seasons) on:
Elo rating, team EPA per play, starting-QB EPA per dropback, QB changes, injuries
(Out/Doubtful starters weighted by position and snap share), rest, neutral site.
Backtest 2015-2026: Brier 0.218 (Elo alone 0.222, Vegas 0.213).

## Search and Ask
The search button (or the / key) finds teams, players and pages. Ask answers questions like
"which team is best on the road?" using Claude with tools that read this site's own data (ask.py).
Turn it on by setting ANTHROPIC_API_KEY (locally: export ANTHROPIC_API_KEY=...; on Render: Environment tab).
Limits: 8 questions per visitor per hour, 200 per day for the whole site (ASK_PER_HOUR, ASK_DAILY_LIMIT).

## Coaching staffs
data/coaches.json holds every team's head coach, coordinators and play callers, with scheme notes and sources.
Update it when a coordinator or play caller changes (the "change" field shows a banner in the pop-up).

## Your news stories
Open the News tab on your own computer to add stories with the form (it's hidden on the
deployed site). They're saved in data/news.json; commit and push that file to publish them.

## Files
- app.py: web server + API, refreshes data every 6 h, live scores every 30 s
- model.py: features + trained model     stats.py: play-by-play EPA, QB stats, injuries
- elo.py: Elo ratings    live.py: ESPN scores + in-game win probability
- season.py: standings, season simulation, playoff bracket    tendencies.py: team tendencies + matchup keys
- teams.py: team names, conferences, divisions
- news.py: ESPN headlines + your saved stories
- players.py: season leaders and fantasy projections (tuned and tested on 2024-2025)
- data.py, main.py: schedules + command-line backtest     agent.py: Claude agent (uses API tokens)
- data/: condensed 2006-2025 stats (commit this). Rebuild with: python stats.py 2025
- static/: the website pages

## Deploy (Render)
Push to GitHub, then on render.com: New > Blueprint > pick the repo (uses render.yaml).
Every `git push` redeploys. Uses about 300-400 MB of memory, under the free plan's limit.

## Install as an app
Open the deployed site on a phone. iPhone: Safari > Share > Add to Home Screen.
Android: Chrome > menu > Install app. It opens full screen with its own icon.
