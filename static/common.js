const $ = s => document.querySelector(s);
const pct = p => p == null ? "n/a" : Math.round(p * 100) + "%";
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const getJSON = u => fetch(u).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
const teamLink = t => `<a href="/team.html?t=${t}">${t}</a>`;
const ago = t => { if (!t) return "just now"; const s = Math.round(Date.now()/1000 - t);
  return s < 60 ? `${s}s ago` : s < 3600 ? `${Math.round(s/60)} min ago` : `${Math.round(s/3600)} h ago`; };
const PALETTE = ["#ff6a1f", "#8cc8ff", "#f4d35e", "#c39bff", "#7fdc8f", "#ff8a7a"];
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function navHTML(active){
  const links = [["/", "Today"], ["/schedule.html", "Schedule"], ["/standings.html", "Standings"], ["/playoffs.html", "Playoffs"],
                 ["/teams.html", "Teams"], ["/leaders.html", "Leaders"], ["/fantasy.html", "Fantasy"],
                 ["/tracker.html", "Tracker"], ["/news.html", "News"], ["/how.html", "How it works"]];
  return `<div class="notice" role="note"><strong>Student project, not betting advice.</strong>
    These predictions are built for learning and are often wrong. Please don't use them to place bets.
    If gambling stops being fun, call or text 1-800-GAMBLER.</div><nav><a class="brand" href="/">Sports Predictor</a>${links.map(([h,l]) =>
    `<a class="link" href="${h}" ${l===active?'aria-current="page"':""}>${l}</a>`).join("")}</nav>`;
}
if (window.Chart){
  Chart.defaults.color = css("--muted");
  Chart.defaults.borderColor = "rgba(238,240,230,.10)";
  Chart.defaults.font.family = css("--body");
}

/* factor breakdown: positive points help the home team */
function factorsHTML(g){
  const max = Math.max(8, ...g.factors.map(f => Math.abs(f.points)));
  return g.factors.filter(f => Math.abs(f.points) >= 0.1).map(f => {
    const w = Math.abs(f.points) / max * 50, home = f.points > 0;
    const style = home ? `left:50%;width:${w}%;background:${css('--pylon')}` : `right:50%;width:${w}%;background:${css('--sky')}`;
    return `<div class="factor"><span>${f.factor}</span><span class="bar"><i style="${style}"></i></span>
      <span class="v">${f.points > 0 ? "+" : ""}${f.points.toFixed(1)}</span></div>`;
  }).join("") + `<p class="muted" style="font-size:.8rem;margin-top:6px">Points added to ${g.home}'s chance (orange) or ${g.away}'s (blue), starting from 50%.</p>`;
}

function injuriesHTML(list){
  if (!list || !list.length) return `<p class="muted" style="font-size:.9rem">No players listed.</p>`;
  return list.map(p => `<div class="inj"><span class="tag ${p.status}">${p.status}</span>
    <span><strong>${esc(p.name)}</strong> <span class="muted">${p.pos}${p.injury ? ", " + esc(p.injury) : ""}${
      p.snap_share ? `, plays ${Math.round(p.snap_share*100)}% of snaps` : ""}</span></span></div>`).join("");
}
const ordinal = n => { if (n==null) return "n/a"; const s=["th","st","nd","rd"], v=n%100; return n+(s[(v-20)%10]||s[v]||s[0]); };

/* leaders */
function photoHTML(url, name){
  return url && /^https:\/\//.test(url) ? `<img class="ph" src="${url}" alt="" loading="lazy" onerror="this.outerHTML='<span class=&quot;ph&quot;>${esc((name||"?")[0])}</span>'">` : `<span class="ph ph-x">${esc((name||"?")[0])}</span>`;
}
function leaderCard(c, compact){
  const fmt = p => (p.value>0 && c.stat==="EPA per dropback" ? "+" : "") + p.value.toLocaleString() + (p.unit||"");
  return `<div class="lcard"><h3>${c.stat}</h3>${c.players.map((p,i)=>`<div class="lrow ${i===0&&!compact?"first":""}">
    <span class="rk">${i+1}</span>${photoHTML(p.photo,p.name)}
    <span class="who"><button class="plink" data-pid="${p.id}">${esc(p.name)}</button><span class="muted">${p.pos} ${teamLink(p.team)}</span></span>
    <span class="val">${fmt(p)}</span></div>`).join("")}</div>`;
}

/* ---------- player card (opens from any element with data-pid) ---------- */
const fmtStat = s => s.value == null ? "0" : s.value.toLocaleString() + (s.unit || "");
async function openPlayer(pid){
  let dlg = document.getElementById("player-dlg");
  if (!dlg){
    dlg = document.createElement("dialog"); dlg.id = "player-dlg"; dlg.className = "pcard";
    dlg.addEventListener("click", e => { if (e.target === dlg) dlg.close(); });
    document.body.appendChild(dlg);
  }
  dlg.innerHTML = `<div class="pc-in"><p class="muted">Loading player</p></div>`;
  dlg.showModal();
  let p;
  try { p = await getJSON(`/api/player/${encodeURIComponent(pid)}`); }
  catch(e){ dlg.innerHTML = `<div class="pc-in"><p>Couldn't load this player.</p><button onclick="this.closest('dialog').close()">Close</button></div>`; return; }
  const inj = p.injury ? ` <span class="tag ${p.injury.status}">${p.injury.status} week ${p.injury.week}${p.injury.injury ? ", " + esc(p.injury.injury) : ""}</span>` : "";
  const next = p.next ? `<div class="pc-next"><div><span class="lbl">Week ${p.next.week} ${p.next.home?"vs.":"at"} ${p.next.opp}</span>
      <b>${p.next.proj.toFixed(1)}</b> <span class="muted">projected PPR</span></div>
      <span class="verdict ${p.next.verdict}">${p.next.verdict}</span>
      <ul>${p.next.reasons.slice(0,3).map(r=>`<li>${esc(r)}</li>`).join("")}</ul></div>` : "";
  dlg.innerHTML = `<div class="pc-in">
    <button class="pc-x" aria-label="Close" onclick="this.closest('dialog').close()">×</button>
    <div class="pc-head">${photoHTML(p.photo, p.name).replace('class="ph', 'class="ph big')}
      <div><h2>${esc(p.name)}</h2><p class="muted">${p.pos}, <a href="/team.html?t=${p.team}">${p.team}</a>, ${p.games} game${p.games===1?"":"s"}${inj}</p>
      ${p.ranks.length ? `<p class="pc-ranks">${p.ranks.map(r=>`<span>${ordinal(r.rank)} in ${r.label.toLowerCase()}</span>`).join("")}</p>` : ""}</div></div>
    <div class="pc-stats">${p.season_line.map(s=>`<div><b>${fmtStat(s)}</b><span>${s.label}</span></div>`).join("")}
      ${p.fantasy.total ? `<div><b>${p.fantasy.ppg}</b><span>Fantasy pts/game</span></div>` : ""}</div>
    ${next}
    ${p.log.length > 1 && window.Chart && p.group !== "DEF" && p.group !== "K" ? `<h3 style="margin:16px 0 6px">Fantasy points by week</h3><div style="height:140px;position:relative"><canvas id="pc-chart"></canvas></div>` : ""}
    <h3 style="margin:16px 0 6px">Game log</h3>
    <div class="table-wrap"><table><thead><tr><th>Wk</th><th>Opp</th><th>Result</th>${p.log_columns.map(c=>`<th>${c}</th>`).join("")}${p.group!=="DEF"&&p.group!=="K"?"<th>Fantasy</th>":""}</tr></thead>
    <tbody>${p.log.map(g=>`<tr><td>${g.week}</td><td>${g.home===false?"at ":"vs. "}${g.opp}</td><td>${g.result||""}</td>
      ${g.stats.map(v=>`<td>${v ?? 0}</td>`).join("")}${p.group!=="DEF"&&p.group!=="K"?`<td>${g.fp ?? 0}</td>`:""}</tr>`).join("")}</tbody></table></div>
  </div>`;
  const c = document.getElementById("pc-chart");
  if (c) new Chart(c, {type:"bar", data:{labels:p.log.map(g=>"W"+g.week), datasets:[{data:p.log.map(g=>g.fp), backgroundColor:css("--pylon"), borderRadius:3}]},
    options:{maintainAspectRatio:false, plugins:{legend:{display:false}}, scales:{y:{beginAtZero:true}}}});
}
document.addEventListener("click", e => {
  const el = e.target.closest("[data-pid]");
  if (!el) return;
  e.preventDefault(); e.stopPropagation();
  openPlayer(el.dataset.pid);
}, true);

/* ---------- favorite teams (saved in this browser) ---------- */
const Fav = {
  get(){ try { return JSON.parse(localStorage.getItem("favTeams") || "[]"); } catch(e){ return []; } },
  has(t){ return this.get().includes(t); },
  toggle(t){
    const f = this.get(), i = f.indexOf(t);
    i >= 0 ? f.splice(i, 1) : f.push(t);
    try { localStorage.setItem("favTeams", JSON.stringify(f)); } catch(e){}
    document.querySelectorAll(`[data-star="${t}"]`).forEach(b => paintStar(b));
    document.dispatchEvent(new CustomEvent("favchange"));
  },
};
function paintStar(b){
  const on = Fav.has(b.dataset.star);
  b.textContent = on ? "★" : "☆";
  b.setAttribute("aria-pressed", on);
  b.setAttribute("aria-label", `${on ? "Remove" : "Add"} ${b.dataset.star} ${on ? "from" : "to"} favorites`);
  b.classList.toggle("on", on);
}
const starBtn = t => `<button class="star" data-star="${t}" aria-pressed="false">☆</button>`;
document.addEventListener("click", e => {
  const b = e.target.closest("[data-star]");
  if (!b) return;
  e.preventDefault(); e.stopPropagation();
  Fav.toggle(b.dataset.star);
}, true);
new MutationObserver(() => document.querySelectorAll("[data-star]:not([data-painted])").forEach(b => { b.dataset.painted = 1; paintStar(b); }))
  .observe(document.documentElement, {childList: true, subtree: true});

/* ---------- notifications for favorite teams ----------
   While any page of the site is open, checks every 30 seconds and notifies on:
   kickoff, every score, big swings in live win chance, final scores, and pregame
   changes of 4+ points in the model's win chance (injury or QB news). */
const Notify = {
  enabled(){ return localStorage.getItem("notify") === "on" && "Notification" in window && Notification.permission === "granted"; },
  async enable(){
    if (!("Notification" in window)) { toast("This browser doesn't support notifications."); return false; }
    const p = await Notification.requestPermission();
    if (p !== "granted") { toast("Notifications are blocked. Allow them in your browser's site settings."); return false; }
    localStorage.setItem("notify", "on"); this.start();
    this.send("Notifications are on", `You'll hear about ${Fav.get().join(", ") || "your starred teams"} while this site is open.`);
    return true;
  },
  disable(){ localStorage.setItem("notify", "off"); },
  send(title, body, tag){
    if (this.enabled()) { try { new Notification(title, {body, tag, icon: "/icon.svg"}); } catch(e){} }
    toast(`${title}: ${body}`);
  },
  timer: null,
  start(){ if (!this.timer) { this.check(); this.timer = setInterval(() => this.check(), 30000); } },
  async check(){
    const fav = Fav.get();
    if (!this.enabled() || !fav.length) return;
    let teams; try { teams = await getJSON(`/api/favorites?teams=${fav.join(",")}`); } catch(e){ return; }
    let prev = {}; try { prev = JSON.parse(localStorage.getItem("notifyState") || "{}"); } catch(e){}
    const next = {};
    for (const t of teams){
      const p = prev[t.team] || {}, n = {};
      if (t.live){
        n.live = true; n.us = t.live.us; n.them = t.live.them; n.prob = t.live.win_prob; n.notedProb = p.notedProb ?? t.live.win_prob;
        const opp = t.next ? t.next.opp : "";
        if (!p.live) this.send(`${t.team} has kicked off`, `${t.team} vs. ${opp}. Win chance ${pct(t.live.win_prob)}.`, `${t.team}-ko`);
        else if (p.us !== n.us || p.them !== n.them){
          const d = Math.round((t.live.win_prob - (p.prob ?? t.live.win_prob)) * 100);
          this.send(`${t.team} ${n.us}, ${opp} ${n.them}`, `${t.live.detail}. Win chance ${pct(t.live.win_prob)}${d ? ` (${d > 0 ? "up" : "down"} ${Math.abs(d)})` : ""}.`, `${t.team}-score`);
          n.notedProb = t.live.win_prob;
        } else if (Math.abs(t.live.win_prob - n.notedProb) >= 0.15){
          this.send(`Big swing for ${t.team}`, `Win chance now ${pct(t.live.win_prob)}, from ${pct(n.notedProb)}. ${t.live.detail}.`, `${t.team}-swing`);
          n.notedProb = t.live.win_prob;
        }
      } else if (p.live && t.last){
        this.send(`Final: ${t.team} ${t.last.result.startsWith("W") ? "wins" : t.last.result.startsWith("L") ? "loses" : "ties"}`, `${t.team} ${t.last.result} ${t.last.home ? "vs." : "at"} ${t.last.opp}.`, `${t.team}-final`);
      }
      if (t.next && !t.live){
        n.gameId = t.next.game_id; n.pre = t.next.win_prob;
        if (p.gameId === t.next.game_id && p.pre != null && Math.abs(t.next.win_prob - p.pre) >= 0.04){
          const up = t.next.win_prob > p.pre;
          this.send(`${t.team}'s chances ${up ? "rose" : "fell"}`, `Week ${t.next.week} ${t.next.home ? "vs." : "at"} ${t.next.opp}: ${pct(p.pre)} to ${pct(t.next.win_prob)}, likely after injury or quarterback news.`, `${t.team}-pre`);
        } else if (p.gameId === t.next.game_id && p.pre != null) n.pre = p.pre;   // keep the baseline until it moves enough
      }
      next[t.team] = n;
    }
    try { localStorage.setItem("notifyState", JSON.stringify(next)); } catch(e){}
  },
};
function toast(msg){
  let box = document.getElementById("toasts");
  if (!box){ box = document.createElement("div"); box.id = "toasts"; box.setAttribute("role", "status"); document.body.appendChild(box); }
  const el = document.createElement("div"); el.className = "toast"; el.textContent = msg; box.appendChild(el);
  setTimeout(() => el.remove(), 7000);
}
if (localStorage.getItem("notify") === "on") Notify.start();

function keysHTML(keys){
  if (!keys || !keys.length) return `<p class="muted" style="font-size:.9rem">No clear mismatches stand out in this matchup.</p>`;
  return `<div class="keys">${keys.map(k=>`<div class="key"><b>${esc(k.title)}: edge ${k.edge}</b>${esc(k.text)}</div>`).join("")}</div>`;
}
