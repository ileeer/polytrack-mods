/*
  AI Driver - a PolyModLoader mod for PolyTrack 0.6.3

  A small neural network learns to drive the current track by trial and error.
  It starts out mashing buttons at random, earns reward when it does the right thing
  (closing in on the next checkpoint, passing checkpoints, finishing), and the best
  attempts pass their "brain" on to the next generation.

  How it hooks into the game (see init() at the bottom):
    1. one mixin in Car.update() hands the car, its controls and the track to onCarUpdate()
    2. one mixin in submitLeaderboard() refuses to upload runs that the AI drove
*/

// ---------------------------------------------------------------- settings
const CFG = {
  popSize: 16,          // attempts per generation
  elites: 4,            // best brains copied unchanged into the next generation
  warmStart: true,      // seed a few brains with "steer toward the checkpoint" so learning starts sooner
  mashStart: 0.9,       // share of time spent mashing random buttons in generation 0...
  mashDecay: 0.8,       // ...multiplied by this every generation
  mashMin: 0.03,
  maxAttemptS: 90,      // give up on an attempt after this long
  stuckS: 3.5,          // ...or after this long without getting closer to the next checkpoint
  graceS: 1.5,
  settleMs: 700,        // wait after a restart before starting the next attempt
  toggleCode: "KeyK",   // Shift + this forgets what was learned on the current track
  restartCode: "KeyT",  // the game's default "start over" key
  startCode: "KeyW",    // the game's default accelerate key (fallback for starting a run)
};

// network: 8 inputs -> 10 tanh units -> 4 buttons [up, right, down, left]
const NI = 8, NH = 10, NO = 4, NW = NI * NH + NH + NH * NO + NO;
const PART = 5;         // world units per track grid cell (PolyTrack's partSize)
const NS = "ai_driver:";

const lsGet = (k, fallback) => { try { const v = localStorage.getItem(NS + k); return v == null ? fallback : JSON.parse(v); } catch { return fallback; } };
const lsSet = (k, v) => { try { localStorage.setItem(NS + k, JSON.stringify(v)); } catch { /* storage full or blocked */ } };
const gauss = () => (Math.random() + Math.random() + Math.random() - 1.5) * 2;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

// ---------------------------------------------------------------- brains
function randomBrain() { const w = new Float32Array(NW); for (let i = 0; i < NW; i++) w[i] = gauss() * 0.8; return w; }
function warmBrain() {
  const w = new Float32Array(NW); for (let i = 0; i < NW; i++) w[i] = gauss() * 0.05;
  const b1 = NI * NH, w2 = b1 + NH, b2 = w2 + NH * NO;
  w[0 * NI + 0] = 4; w[1 * NI + 0] = -4;      // hidden 0 = "checkpoint is on my right", hidden 1 = "on my left"
  w[b2 + 0] = 1;                              // gas by default
  w[w2 + 1 * NH + 0] = 1.5; w[b2 + 1] = -0.2; // right when the checkpoint is on the right
  w[w2 + 3 * NH + 1] = 1.5; w[b2 + 3] = -0.2; // left when it is on the left
  w[b2 + 2] = -1;                             // no brake by default
  return w;
}
function mutate(w, p, s) { const c = Float32Array.from(w); for (let i = 0; i < NW; i++) if (Math.random() < p) c[i] += gauss() * s; return c; }
function think(w, inp, out) {
  const b1 = NI * NH, w2 = b1 + NH, b2 = w2 + NH * NO, h = new Float32Array(NH);
  for (let j = 0; j < NH; j++) { let s = w[b1 + j]; for (let i = 0; i < NI; i++) s += w[j * NI + i] * inp[i]; h[j] = Math.tanh(s); }
  for (let o = 0; o < NO; o++) { let s = w[b2 + o]; for (let j = 0; j < NH; j++) s += w[w2 + o * NH + j] * h[j]; out[o] = Math.tanh(s); }
}

// ---------------------------------------------------------------- state
const AI = {
  on: false, phase: "idle", usedThisRun: false, unsupported: false, error: null,
  gen: 0, idx: 0, pop: [], fit: [], best: { fit: -Infinity, w: null }, trk: null,
  fwd: null, cal: { s: [0, 0, 0, 0], n: 0 },   // which local axis of the car points forward (self-calibrated)
  att: null, t0: 0, lastHud: 0, feed: [], buttons: [0, 0, 0, 0],
};
const eps = () => Math.max(CFG.mashMin, CFG.mashStart * Math.pow(CFG.mashDecay, AI.gen));

function note(text) { AI.feed.push({ text, t: performance.now() }); if (AI.feed.length > 4) AI.feed.shift(); }

function initPopulation() {
  const saved = lsGet("best:" + AI.trk.sig, null);
  AI.pop = []; AI.fit = []; AI.gen = 0;
  if (saved && saved.w && saved.w.length === NW) {
    AI.best = { fit: saved.fit, w: Float32Array.from(saved.w) }; AI.gen = saved.gen || 0;
    AI.pop.push(Float32Array.from(saved.w));
    while (AI.pop.length < CFG.popSize) AI.pop.push(AI.pop.length < CFG.popSize - 3 ? mutate(AI.best.w, 0.15, 0.25) : randomBrain());
    note("Loaded saved brain, generation " + AI.gen);
  } else {
    AI.best = { fit: -Infinity, w: null };
    for (let i = 0; i < CFG.popSize; i++) AI.pop.push(CFG.warmStart && i < 4 ? mutate(warmBrain(), 0.2, 0.15 * i) : randomBrain());
  }
  AI.idx = 0;
}

function nextGeneration() {
  const order = AI.fit.map((f, i) => i).sort((a, b) => AI.fit[b] - AI.fit[a]);
  const pool = Math.max(2, CFG.popSize >> 1), next = [];
  for (let i = 0; i < CFG.elites; i++) next.push(Float32Array.from(AI.pop[order[i]]));
  while (next.length < CFG.popSize) {
    const a = AI.pop[order[Math.random() * pool | 0]], b = AI.pop[order[Math.random() * pool | 0]], c = new Float32Array(NW);
    for (let i = 0; i < NW; i++) c[i] = Math.random() < 0.5 ? a[i] : b[i];
    next.push(mutate(c, 0.12, 0.3));
  }
  AI.pop = next; AI.fit = []; AI.idx = 0; AI.gen++;
  lsSet("best:" + AI.trk.sig, { fit: AI.best.fit, w: Array.from(AI.best.w), gen: AI.gen });
  note("Generation " + AI.gen + " (random mashing " + Math.round(eps() * 100) + "%)");
}

// ---------------------------------------------------------------- track + geometry
function rot(q, v) { // rotate vector v by quaternion q
  const ix = q.w * v[0] + q.y * v[2] - q.z * v[1], iy = q.w * v[1] + q.z * v[0] - q.x * v[2],
        iz = q.w * v[2] + q.x * v[1] - q.y * v[0], iw = -q.x * v[0] - q.y * v[1] - q.z * v[2];
  return [ix * q.w + iw * -q.x + iy * -q.z - iz * -q.y, iy * q.w + iw * -q.y + iz * -q.x - ix * -q.z, iz * q.w + iw * -q.z + ix * -q.y - iy * -q.x];
}
function toWorld(c) {
  let x = c.x * PART, z = c.z * PART;
  if (String(c.type).includes("Wide")) { const a = (c.rotation || 0) * Math.PI / 2; x += 10 * Math.cos(a); z += -10 * Math.sin(a); }
  return [x, 0, z];
}
function getTrack(objs) {
  const t = objs.find((o) => o && typeof o.getCheckpoints === "function");
  if (!t) return null;
  if (AI.trk && AI.trk.src === t) return AI.trk;
  const cps = t.getCheckpoints(), fins = typeof t.getFinishes === "function" ? t.getFinishes() : [];
  const orders = [...new Set(cps.map((c) => c.checkpointOrder))].sort((a, b) => a - b);
  const trk = { src: t, orders, cp: orders.map((o) => cps.filter((c) => c.checkpointOrder === o).map(toWorld)), fin: fins.map(toWorld) };
  if (!trk.cp.length && !trk.fin.length) return null;
  const str = JSON.stringify([trk.cp, trk.fin]); let h = 0; for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) | 0;
  trk.sig = (h >>> 0).toString(16);
  AI.trk = trk; initPopulation(); AI.phase = AI.on ? "wait" : "idle"; AI.t0 = performance.now();
  return trk;
}
function target(trk, i, pos) {
  const list = i < trk.cp.length ? trk.cp[i] : trk.fin;
  if (!list.length) return null;
  let best = list[0], bd = Infinity;
  for (const p of list) { const d = (p[0] - pos[0]) ** 2 + (p[2] - pos[2]) ** 2; if (d < bd) { bd = d; best = p; } }
  return best;
}

// ---------------------------------------------------------------- driving
function setControls(c, up, right, down, left) { c.up = !!up; c.right = !!right; c.down = !!down; c.left = !!left; }
function tapKey(code, type) { window.dispatchEvent(new KeyboardEvent(type, { code, key: code, bubbles: true })); }
function pressKey(code) { tapKey(code, "keydown"); setTimeout(() => tapKey(code, "keyup"), 120); }

function observe(car, trk, A, dt) {
  const p = car.getPosition(), q = car.getQuaternion(), pos = [p.x, p.y, p.z];
  const vel = A.prev ? [(pos[0] - A.prev[0]) / dt, (pos[1] - A.prev[1]) / dt, (pos[2] - A.prev[2]) / dt] : [0, 0, 0];
  A.prev = pos;
  const speed = Math.abs(car.getSpeedKmh()), st = car.getCarState(), wc = st && st.wheelContact;
  const air = !!wc && wc.every((w) => w == null);
  const up = rot(q, [0, 1, 0]);
  const fwd3 = rot(q, AI.fwd || [0, 0, 1]), fl = Math.hypot(fwd3[0], fwd3[2]) || 1;
  const f = [fwd3[0] / fl, fwd3[2] / fl], r = [-f[1], f[0]];       // right = forward x up
  const idx = car.getNextCheckpointIndex();
  const t1 = target(trk, idx, pos), t2 = target(trk, idx + 1, pos) || t1;
  const bear = (t) => { const dx = t[0] - pos[0], dz = t[2] - pos[2]; return Math.atan2(dx * r[0] + dz * r[1], dx * f[0] + dz * f[1]); };
  const dist = t1 ? Math.hypot(t1[0] - pos[0], t1[2] - pos[2]) : 0;
  const heading = Math.atan2(f[1], f[0]);
  let dh = A.heading == null ? 0 : heading - A.heading; dh = Math.atan2(Math.sin(dh), Math.cos(dh)); A.heading = heading;
  const hs = Math.hypot(vel[0], vel[2]), slip = (vel[0] * r[0] + vel[2] * r[1]) / (hs + 2);
  const inp = [
    t1 ? clamp(bear(t1) / (Math.PI / 2), -1, 1) : 0, Math.min(dist / 80, 1.5), t2 ? clamp(bear(t2) / (Math.PI / 2), -1, 1) : 0,
    Math.min(speed / 150, 1.5), clamp(slip, -1, 1), clamp(dh / dt / 3, -1, 1), air ? 1 : 0, clamp(fwd3[1], -1, 1),
  ];
  return { inp, pos, vel, speed, air, upY: up[1], idx, dist, q };
}

function calibrate(o) {
  if (o.speed < 12 || o.air) return;
  const cand = [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]];
  cand.forEach((a, i) => { const w = rot(o.q, a); AI.cal.s[i] += w[0] * o.vel[0] + w[2] * o.vel[2]; });
  if (++AI.cal.n >= 25) {
    let bi = 0; for (let i = 1; i < 4; i++) if (AI.cal.s[i] > AI.cal.s[bi]) bi = i;
    AI.fwd = cand[bi]; lsSet("fwd", bi); note("Calibrated: the car's forward axis is " + ["+X", "-X", "+Z", "-Z"][bi]);
  }
}

function beginAttempt() {
  const now = performance.now();
  AI.att = { t0: now, last: now, prev: null, heading: null, R: 0, idx: 0, dist: null, bestDist: Infinity, lastImprove: now,
    flipSince: null, startY: null, mashUntil: 0, mash: [0, 0, 0, 0], tried: false, ended: false };
  AI.phase = "run"; AI.usedThisRun = true;
}

function endAttempt(controls, why, bonus) {
  const A = AI.att; A.R += bonus || 0; setControls(controls, 0, 0, 0, 0);
  AI.fit[AI.idx] = A.R;
  if (A.R > AI.best.fit) { AI.best = { fit: A.R, w: Float32Array.from(AI.pop[AI.idx]) }; note("New best reward: " + A.R.toFixed(1)); }
  note("Attempt " + (AI.idx + 1) + "/" + CFG.popSize + ": " + why + ", reward " + A.R.toFixed(1));
  AI.idx++;
  if (AI.idx >= CFG.popSize) nextGeneration();
  AI.att = null; AI.phase = "wait"; AI.t0 = performance.now(); pressKey(CFG.restartCode);
}

function runFrame(car, controls, trk, now) {
  const A = AI.att; const dt = clamp((now - A.last) / 1000, 0.004, 0.1); A.last = now;
  if (!car.hasStarted()) {                            // press gas to start the run
    setControls(controls, 1, 0, 0, 0);
    if (!A.tried && now - A.t0 > 600) { A.tried = true; pressKey(CFG.startCode); }
    if (now - A.t0 > 4000) { AI.phase = "wait"; AI.t0 = now; pressKey(CFG.restartCode); }
    return;
  }
  const o = observe(car, trk, A, dt), t = (now - A.t0) / 1000;
  if (A.startY == null) A.startY = o.pos[1];
  if (car.hasFinished()) { note("FINISHED!"); return endAttempt(controls, "finished", 100 + Math.max(0, 60 - t) * 2); }

  // reward: getting closer to the next checkpoint, passing checkpoints
  if (o.idx > A.idx) { const n = o.idx - A.idx; A.R += 20 * n; A.idx = o.idx; A.bestDist = Infinity; A.dist = null; A.lastImprove = now; note("+" + 20 * n + " checkpoint"); }
  if (A.dist != null && Math.abs(A.dist - o.dist) < 30) A.R += (A.dist - o.dist) * 0.1;
  A.dist = o.dist;
  if (o.dist < A.bestDist - 0.5) { A.bestDist = o.dist; A.lastImprove = now; }

  // decide which buttons to press
  let b;
  if (!AI.fwd) { calibrate(o); b = [1, 0, 0, 0]; }                    // drive straight until we know which way is forward
  else if (now < A.mashUntil) b = A.mash;                              // random button mashing
  else {
    const e = eps(), rate = (e / (1 - e + 0.05)) / 0.35;               // mashing starts a few times per second early on
    if (Math.random() < 1 - Math.exp(-rate * dt)) {
      A.mashUntil = now + 150 + Math.random() * 400;
      A.mash = [Math.random() < 0.65, Math.random() < 0.4, Math.random() < 0.2, Math.random() < 0.4].map(Number); b = A.mash;
    } else { const out = new Float32Array(NO); think(AI.pop[AI.idx], o.inp, out); b = [out[0] > 0, out[1] > 0, out[2] > 0, out[3] > 0].map(Number); }
  }
  AI.buttons = b; setControls(controls, b[0], b[1], b[2], b[3]);

  // end conditions
  if (o.upY < 0.25) { A.flipSince = A.flipSince ?? now; if (now - A.flipSince > 1200) return endAttempt(controls, "flipped over", -5); } else A.flipSince = null;
  if (o.pos[1] < A.startY - 60) return endAttempt(controls, "fell off", -5);
  if (t > CFG.graceS && (now - A.lastImprove) / 1000 > CFG.stuckS) return endAttempt(controls, "stuck", 0);
  if (t > CFG.maxAttemptS) return endAttempt(controls, "ran out of time", 0);
}

function onCarUpdate(car, controls, trackObjs) {
  if (AI.unsupported || !controls || typeof controls.up === "undefined" || car.isPaused) return;
  try {
    const now = performance.now();
    if (!car.hasStarted() && AI.phase !== "run") AI.usedThisRun = false;
    const trk = getTrack(trackObjs || []);
    if (AI.on && trk) {
      if (!AI.fwd) { const bi = lsGet("fwd", null); if (bi != null) AI.fwd = [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]][bi]; }
      if (AI.phase === "idle") { AI.phase = "wait"; AI.t0 = now; if (car.hasStarted()) pressKey(CFG.restartCode); }
      else if (AI.phase === "wait") {
        if (car.hasStarted()) { if (now - AI.t0 > 1500) { AI.t0 = now; pressKey(CFG.restartCode); } }
        else if (now - AI.t0 > CFG.settleMs) beginAttempt();
      } else if (AI.phase === "run") runFrame(car, controls, trk, now);
    }
    if (now - AI.lastHud > 100) { AI.lastHud = now; renderHud(); }
  } catch (e) { AI.on = false; AI.error = String(e && e.message || e); console.error("[AI Driver]", e); renderHud(); }
}

// ---------------------------------------------------------------- HUD + keys
let hud = null;
function ensureHud() {
  if (hud) return;
  const css = document.createElement("style");
  css.textContent = `#aid{position:fixed;left:12px;bottom:12px;z-index:99999;font:12px/1.35 system-ui,sans-serif;color:#fff;background:rgba(10,14,28,.72);padding:8px 10px;border-radius:10px;pointer-events:none;min-width:190px}
#aid b{font-size:13px}#aid .k{display:flex;gap:4px;margin:6px 0}#aid .k span{width:26px;height:26px;border-radius:6px;display:grid;place-items:center;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.35)}
#aid .k span.on{background:#ffd23f;color:#111;border-color:#fff}#aid .f{opacity:.9}#aid small{opacity:.65}`;
  document.head.appendChild(css);
  hud = document.createElement("div"); hud.id = "aid";
  hud.innerHTML = `<b>AI Driver</b> <span class="s"></span><div class="l"></div><div class="k"><span>▲</span><span>▶</span><span>▼</span><span>◀</span></div><div class="f"></div><small>K on/off · Shift+K forget</small>`;
  document.body.appendChild(hud);
}
function renderHud() {
  ensureHud(); const q = (s) => hud.querySelector(s), now = performance.now();
  q(".s").textContent = AI.error ? "error" : AI.unsupported ? "unsupported game version" : AI.on ? "ON" : "off (press K)";
  q(".l").textContent = AI.error ? AI.error : AI.on && AI.trk
    ? `Gen ${AI.gen + 1} · attempt ${Math.min(AI.idx + 1, CFG.popSize)}/${CFG.popSize} · mashing ${Math.round(eps() * 100)}%`
      + (AI.att ? ` · reward ${AI.att.R.toFixed(1)}` : "") + (AI.best.fit > -Infinity ? ` · best ${AI.best.fit.toFixed(1)}` : "")
    : "";
  [...hud.querySelectorAll(".k span")].forEach((e, i) => e.classList.toggle("on", !!AI.buttons[i] && AI.on));
  q(".f").innerHTML = AI.on ? AI.feed.filter((f) => now - f.t < 6000).map((f) => f.text).join("<br>") : (AI.usedThisRun ? "Leaderboard upload blocked for this run" : "");
}
function onKey(e) {
  if (e.code !== CFG.toggleCode || e.repeat || !e.isTrusted) return;     // only real key presses, never our own synthetic ones
  if (e.shiftKey) { if (AI.trk) { try { localStorage.removeItem(NS + "best:" + AI.trk.sig); } catch { /* ignore */ } initPopulation(); note("Forgot everything on this track"); } return; }
  AI.on = !AI.on; AI.error = null; AI.phase = "idle";
  if (!AI.on) { AI.buttons = [0, 0, 0, 0]; AI.att = null; }
  renderHud();
}

// ---------------------------------------------------------------- the mod object PolyModLoader expects
const CAR_TOKEN = '(0, l.gn)(this, Ue, "f")?.update(e),';
const CAR_HOOK = 'window.__aiDriver&&window.__aiDriver.onCarUpdate(this,(0, l.gn)(this, ne, "f"),[(0, l.gn)(this, Ce, "f"),(0, l.gn)(this, _e, "f")]),';
const SUBMIT_TOKEN = "submitLeaderboard(e, t, n, i, r, a, s, o) {";
const SUBMIT_GUARD = 'if(window.__aiDriver&&window.__aiDriver.blockSubmit()){return Promise.reject(new Error("Submit not allowed"));}';

class AiDriverMod {
  touchingPhysics = true;   // makes the loader report the game as modded, so multiplayer matchmaking keeps it away from vanilla lobbies
  loaded = false; offlineMode = false; IconSrc; modBaseUrl; latestSaved; modInitialized;
  modAuthor; modID; modName; modVersion; modDependencies; modDescription; polyVersion; assetFolder; manifest;
  get iconSrc() { return this.IconSrc; } set iconSrc(v) { this.IconSrc = v; }
  set setLoaded(s) { this.loaded = s; } get isLoaded() { return this.loaded; }
  get baseUrl() { return this.modBaseUrl; } set baseUrl(u) { this.modBaseUrl = u; }
  get savedLatest() { return this.latestSaved; } set savedLatest(l) { this.latestSaved = l; }
  get initialized() { return this.modInitialized; } set initialized(v) { this.modInitialized = v; }
  preInit = () => {};
  postInit = () => {};
  errorInit = () => {};
  onGameLoad = () => { window.addEventListener("keydown", onKey); };
  init = (pml) => {
    window.__aiDriver = { onCarUpdate, blockSubmit: () => AI.usedThisRun, state: AI };
    try {
      pml.registerGlobalMixin({ type: 3 /* INSERT */, token: CAR_TOKEN, func: CAR_HOOK });
      pml.registerGlobalMixin({ type: 3 /* INSERT */, token: SUBMIT_TOKEN, func: SUBMIT_GUARD });
    } catch (e) {
      AI.unsupported = true; console.error("[AI Driver] could not hook this PolyTrack build:", e);
      alert("AI Driver: this version of PolyTrack isn't supported (the game code it hooks into has changed).");
    }
  };
}
export const polyMod = new AiDriverMod();
export const _test = { AI, CFG, CAR_TOKEN, CAR_HOOK, SUBMIT_TOKEN, SUBMIT_GUARD, onCarUpdate, think, NW };
