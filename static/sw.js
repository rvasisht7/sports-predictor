/* Service worker: lets the site install as an app and still open without signal.
   Always tries the network first so data is fresh; falls back to the last saved copy. */
const CACHE = "predictor-v10";
const SHELL = ["/", "/style.css?v=10", "/common.js?v=10", "/icon-192.png", "/offline.html"];
self.addEventListener("install", e => { e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL))); self.skipWaiting(); });
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))));
  self.clients.claim();
});
self.addEventListener("fetch", e => {
  const req = e.request;
  if (req.method !== "GET" || new URL(req.url).origin !== location.origin) return;
  e.respondWith(fetch(req).then(res => {
    if (res.ok) { const copy = res.clone(); caches.open(CACHE).then(c => c.put(req, copy)); }
    return res;
  }).catch(async () => (await caches.match(req)) || (req.mode === "navigate" ? caches.match("/offline.html") : Response.error())));
});
