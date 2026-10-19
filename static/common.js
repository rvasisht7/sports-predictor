const $ = s => document.querySelector(s);
const pct = p => p == null ? "n/a" : Math.round(p * 100) + "%";
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const getJSON = u => fetch(u).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
const teamLink = t => `<a href="/team.html?t=${t}">${t}</a>`;
const ago = t => { if (!t) return "just now"; const s = Math.round(Date.now()/1000 - t);
  return s < 60 ? `${s}s ago` : s < 3600 ? `${Math.round(s/60)} min ago` : `${Math.round(s/3600)} h ago`; };
const PALETTE = ["#ff6a1f", "#8cc8ff", "#f4d35e", "#c39bff", "#7fdc8f", "#ff8a7a"];
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

const ICONS = {
  today: '<path d="M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/>',
  schedule: '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/>',
  standings: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
  fantasy: '<path d="M12 3l2.6 5.5 6 .8-4.4 4.2 1.1 6L12 16.6 6.7 19.5l1.1-6L3.4 9.3l6-.8z"/>',
  more: '<circle cx="5" cy="12" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="19" cy="12" r="1.6"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/>',
};
const icon = (n, size=22) => `<svg class="ic" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[n]}</svg>`;
const PAGES = [["/", "Today"], ["/schedule.html", "Schedule"], ["/standings.html", "Standings"], ["/playoffs.html", "Playoffs"],
  ["/teams.html", "Teams"], ["/fantasy.html", "Fantasy"], ["/leaders.html", "Leaders"], ["/tracker.html", "Tracker"],
  ["/news.html", "News"], ["/how.html", "How it works"]];
const MAIN = ["Today", "Schedule", "Standings", "Playoffs", "Teams", "Fantasy"];
const TABS = [["/", "Today", "today"], ["/schedule.html", "Schedule", "schedule"], ["/standings.html", "Standings", "standings"],
  ["/fantasy.html", "Fantasy", "fantasy"]];

function navPages(){
  return Fav.get().length ? PAGES.map(p => p[1] === "How it works" ? ["/news.html?mine=1", "My news"] : p) : PAGES;
}
function navHTML(active){
  window.__navActive = active;
  const PAGES = navPages();
  const extra = PAGES.filter(([, l]) => !MAIN.includes(l));
  const inMore = extra.some(([, l]) => l === active);
  return `<div class="notice" role="note"><span class="long"><strong>Student project, not betting advice.</strong>
      Predictions are often wrong; please don't use them to bet. Gambling help: 1-800-GAMBLER</span>
      <span class="short"><strong>Not betting advice.</strong> Help: 1-800-GAMBLER</span></div>
    <div class="topbar">
      <a class="brand" href="/"><img src="/icon-192.png" alt="" width="26" height="26">Sports Predictor</a>
      <nav class="toplinks" aria-label="Main">
        ${PAGES.filter(([, l]) => MAIN.includes(l)).map(([h, l]) => `<a href="${h}" ${l===active?'aria-current="page"':""}>${l}</a>`).join("")}
        <div class="more-wrap"><button class="more-btn ${inMore?"cur":""}" aria-expanded="false" aria-haspopup="true">More ▾</button>
          <div class="more-menu" hidden>${extra.map(([h, l]) => `<a href="${h}" ${l===active?'aria-current="page"':""}>${l}</a>`).join("")}
            ${Fav.get().length ? `<a class="about" href="/how.html">About this site</a>` : ""}</div></div>
      </nav>
      <button class="search-btn" aria-label="Search or ask a question">${icon("search", 18)}<span>Search or ask</span><kbd>/</kbd></button>
    </div>
    <nav class="tabbar" aria-label="Main">
      ${TABS.map(([h, l, i]) => `<a href="${h}" ${l===active?'aria-current="page"':""}>${icon(i)}<span>${l}</span></a>`).join("")}
      <button class="tab-more ${inMore || active==="Playoffs" || active==="Teams" ? "cur" : ""}" aria-label="More pages">${icon("more")}<span>More</span></button>
    </nav>
    <dialog class="sheet" id="more-sheet"><div class="sheet-in"><div class="grab"></div><h3>More</h3>
      ${PAGES.filter(([, l]) => !TABS.some(t => t[1] === l)).map(([h, l]) => `<a href="${h}" ${l===active?'aria-current="page"':""}>${l}</a>`).join("")}
      ${Fav.get().length ? `<a class="about" href="/how.html">About this site</a>` : ""}
      <button class="sheet-close">Close</button></div></dialog>`;
}

/* wire up the nav once it's on the page */
document.addEventListener("click", e => {
  const mb = e.target.closest(".more-btn");
  const menu = document.querySelector(".more-menu");
  if (mb){ const open = menu.hidden; menu.hidden = !open; mb.setAttribute("aria-expanded", open); return; }
  if (menu && !e.target.closest(".more-wrap")) { menu.hidden = true; document.querySelector(".more-btn")?.setAttribute("aria-expanded", false); }
  if (e.target.closest(".tab-more")) document.getElementById("more-sheet").showModal();
  if (e.target.closest(".sheet-close") || e.target.id === "more-sheet") document.getElementById("more-sheet").close();
  if (e.target.closest(".search-btn")) Search.open();
});
document.addEventListener("keydown", e => {
  if (e.key === "/" && !/input|textarea|select/i.test(document.activeElement.tagName)) { e.preventDefault(); Search.open(); }
});

/* ---------- search + ask ---------- */
const Search = {
  index: null, history: [],
  async open(){
    let d = document.getElementById("search-dlg");
    if (!d){
      d = document.createElement("dialog"); d.id = "search-dlg"; d.className = "searchdlg";
      d.innerHTML = `<div class="sd-in">
        <div class="sd-bar">${icon("search", 20)}<input id="sq" type="search" placeholder="Search or ask a question" autocomplete="off" enterkeyhint="search" aria-label="Search or ask">
          <button class="sd-x" aria-label="Close search">Cancel</button></div>
        <div id="sres" class="sres"></div></div>`;
      document.body.appendChild(d);
      d.querySelector(".sd-x").onclick = () => d.close();
      d.addEventListener("click", e => { if (e.target === d) d.close(); });
      const q = d.querySelector("#sq");
      q.addEventListener("input", () => this.render(q.value));
      q.addEventListener("keydown", e => { if (e.key === "Enter"){ e.preventDefault(); const v = q.value.trim(); if (v) this.enter(v); } });
    }
    d.showModal();
    d.querySelector("#sq").focus();
    if (!this.index){ try { this.index = await getJSON("/api/search-index"); } catch(e){ this.index = {teams:[], players:[], ask_enabled:false}; } }
    this.render(d.querySelector("#sq").value);
  },
  isQuestion(v){ return /\?$/.test(v) || v.trim().split(/\s+/).length >= 3; },
  matches(v){
    const s = v.trim().toLowerCase(); if (!s || !this.index) return {teams:[], players:[], pages:[]};
    let teams = this.index.teams.filter(t => t.team.toLowerCase() === s || t.name.toLowerCase().includes(s));
    let players = this.index.players.filter(p => p[1].toLowerCase().includes(s));
    const pages = PAGES.filter(([, l]) => l.toLowerCase().includes(s)).slice(0, 3);
    if (!teams.length && !players.length){   // "is mahomes good" -> match the individual words
      const STOP = new Set("the and who what which how best good bad worst team teams player players does did will should start sit for with this week vs is are was his their".split(" "));
      const words = s.replace(/[^a-z0-9' .-]/g, " ").split(/\s+/).filter(w => w.length >= 3 && !STOP.has(w));
      const hit = (name, w) => name.toLowerCase().split(/[\s.'-]+/).some(part => part === w || (w.length >= 5 && part.startsWith(w)));
      teams = this.index.teams.filter(t => words.some(w => t.team.toLowerCase() === w || hit(t.name, w)));
      players = this.index.players.filter(p => words.some(w => hit(p[1], w)));
    }
    return {teams: teams.slice(0, 4), players: players.slice(0, 8), pages};
  },
  render(v){
    const box = document.getElementById("sres"), ask = this.index && this.index.ask_enabled;
    const m = this.matches(v), s = v.trim();
    let html = this.history.length ? `<div id="answers">${this.history.map(h => this.answerHTML(h)).reverse().join("")}</div>` : "";
    if (s && ask) html += `<button class="sr ask-row" data-ask="1"><b>Ask</b> "${esc(s)}"</button>`;
    if (m.teams.length) html += `<h4>Teams</h4>` + m.teams.map(t => `<a class="sr" href="/team.html?t=${t.team}"><b>${esc(t.name)}</b><span class="muted">${t.team}</span></a>`).join("");
    if (m.players.length) html += `<h4>Players</h4>` + m.players.map(p => `<button class="sr" data-player="${p[0]}"><b>${esc(p[1])}</b><span class="muted">${p[2]}, ${p[3]}</span></button>`).join("");
    if (m.pages.length) html += `<h4>Pages</h4>` + m.pages.map(([h, l]) => `<a class="sr" href="${h}"><b>${l}</b></a>`).join("");
    if (!s){
      html += ask ? `<h4>${this.history.length ? "Ask something else" : "Try asking"}</h4>` + ["Which team is best on the road?", "Who leads the NFL in sacks?", "Who has the best defense?",
        "What are the Chiefs' playoff chances?", "Should I start Puka Nacua or Chris Olave?"].map(x => `<button class="sr chip-q" data-q="${esc(x)}">${esc(x)}</button>`).join("")
        : `<p class="muted sd-tip">Search for any team, player or page.</p>`;
    } else if (!html) html = `<p class="muted sd-tip">No teams or players match "${esc(s)}".${ask ? "" : " Asking questions isn't turned on for this site."}</p>`;
    box.innerHTML = html;
    box.querySelectorAll("[data-ask]").forEach(b => b.onclick = () => this.ask(s));
    box.querySelectorAll("[data-q]").forEach(b => b.onclick = () => { document.getElementById("sq").value = b.dataset.q; this.ask(b.dataset.q); });
    box.querySelectorAll("[data-player]").forEach(b => b.onclick = () => { document.getElementById("search-dlg").close(); openPlayer(b.dataset.player); });
  },
  enter(v){
    const m = this.matches(v);
    if (this.index.ask_enabled && (this.isQuestion(v) || (!m.teams.length && !m.players.length))) return this.ask(v);
    if (m.teams.length === 1 && !m.players.length) location.href = `/team.html?t=${m.teams[0].team}`;
    else if (m.players.length === 1 && !m.teams.length){ document.getElementById("search-dlg").close(); openPlayer(m.players[0][0]); }
  },
  answerHTML(h){
    const fmt = t => esc(t).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .split("\n").map(l => /^\s*[-•]\s+/.test(l) ? `<li>${l.replace(/^\s*[-•]\s+/, "")}</li>` : (l.trim() ? `<p>${l}</p>` : "")).join("")
      .replace(/(<li>.*?<\/li>)+/gs, x => `<ul>${x}</ul>`);
    return `<div class="answer"><p class="aq">${esc(h.q)}</p>${h.loading ? `<p class="muted">Looking that up…</p>` : h.error ? `<p class="no">${esc(h.error)}</p>` : fmt(h.a)}</div>`;
  },
  async ask(q){
    const h = {q, loading: true}; this.history.push(h);
    document.getElementById("sq").value = ""; this.render("");
    try {
      const r = await fetch("/api/ask", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question: q, history: this.history.filter(x => x.a).slice(-3).map(x => ({q: x.q, a: x.a}))})});
      const d = await r.json().catch(() => ({}));
      if (!r.ok) h.error = typeof d.detail === "string" ? d.detail : "Couldn't get an answer right now."; else h.a = d.answer;
    } catch(e){ h.error = "Can't reach the server."; }
    h.loading = false; this.render(document.getElementById("sq").value);
    document.getElementById("sres").scrollTop = 0;
  },
};

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
    const nav = document.getElementById("nav");
    if (nav && window.__navActive) nav.innerHTML = navHTML(window.__navActive);
    if (i < 0) toast(`Following ${t}. Their news is now under My news.`);
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

/* installable app: register the service worker */
if ("serviceWorker" in navigator) window.addEventListener("load", () => navigator.serviceWorker.register("/sw.js").catch(() => {}));
