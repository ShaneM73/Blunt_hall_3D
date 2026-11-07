// Blunt Hall Guide: block model of the scanned floors + the route from the front entrance to one room.
// Data: data/2026-11-04/guide.json (scripts/pipeline/guide/g03_blocks.py + g04_route.py), building frame, metres, z up.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

const DATA = 'data/2026-11-04/guide.json';
const WALL_H = 2.2, ROUTE_R = 0.16, WALK = 1.3, STAIR = 0.55, SPEEDUP = 6;   // wall height (m), route radius, walking / stair speed (m/s), animation speed-up
const $ = (id) => document.getElementById(id);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const b2w = (p) => new THREE.Vector3(p[0], p[2], -p[1]);   // building (x, y, z-up) -> three.js (x, y-up, -y)

const stage = $('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
stage.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 0.5, 2000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.maxPolarAngle = Math.PI * 0.49; controls.minDistance = 4; controls.maxDistance = 400;
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8780, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(-60, 120, 80); scene.add(sun);
const root = new THREE.Group(); root.rotation.x = -Math.PI / 2; scene.add(root);

const mats = {
  wall: new THREE.MeshLambertMaterial({ color: 0xffffff, transparent: true }),
  edge: new THREE.LineBasicMaterial({ color: 0x999999, transparent: true }),
  slab: new THREE.MeshLambertMaterial({ color: 0xdddddd, transparent: true }),
  stair: new THREE.MeshLambertMaterial({ color: 0xeeeeee, transparent: true }),
  route: new THREE.MeshBasicMaterial({ color: 0x7a1e2c }),
  room: new THREE.MeshLambertMaterial({ color: 0x7a1e2c, transparent: true, opacity: 0.55, depthWrite: false }),
  walker: new THREE.MeshBasicMaterial({ color: 0x7a1e2c }),
};
function applyTheme() {
  scene.background = new THREE.Color(css('--stage'));
  mats.wall.color.set(css('--wall')); mats.edge.color.set(css('--wall-edge')); mats.slab.color.set(css('--slab'));
  mats.stair.color.set(css('--wall')); for (const k of ['route', 'room', 'walker']) mats[k].color.set(css('--accent'));
  for (const fl of floors) for (const o of fl.g.children) if (o.userData.ghostMat) o.userData.ghostMat.color.copy(o.userData.base.color);
}

// see-through building: walls, floors and stairs are translucent so the route shows from outside
function setXray(on) {
  for (const [k, o] of [['wall', 0.28], ['slab', 0.5], ['stair', 0.6], ['edge', 0.5]]) {
    mats[k].opacity = on ? o : 1; mats[k].depthWrite = !on; mats[k].needsUpdate = true;
  }
  $('xray').setAttribute('aria-pressed', on);
}

let D, floors = [], tube, tubeSegs, curve, cum, cumT, simT = 0, walkStage = 0, walker, pins = [], steps = [], cur = 0, focusFloor = 3, playing = false;

function boxGeom(x0, y0, x1, y1, z0, z1) {
  const g = new THREE.BoxGeometry(x1 - x0, y1 - y0, z1 - z0);
  g.translate((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2); return g;
}

function buildFloor(f, i) {
  const g = new THREE.Group(); g.position.z = f.z; root.add(g);
  const shapes = f.slab.map((rings) => {
    const s = new THREE.Shape(rings[0].map((p) => new THREE.Vector2(p[0], p[1])));
    for (const h of rings.slice(1)) s.holes.push(new THREE.Path(h.map((p) => new THREE.Vector2(p[0], p[1]))));
    return s;
  });
  if (shapes.length) {
    const sg = new THREE.ExtrudeGeometry(shapes, { depth: 0.18, bevelEnabled: false }); sg.translate(0, 0, -0.18);
    g.add(new THREE.Mesh(sg, mats.slab));
  }
  const hW = Math.min(WALL_H, f.h);
  const wg = f.walls.length ? mergeGeometries(f.walls.map((w) => boxGeom(w[0], w[1], w[2], w[3], 0, hW)), false) : null;
  if (wg) {
    g.add(new THREE.Mesh(wg, mats.wall));
    g.add(new THREE.LineSegments(new THREE.EdgesGeometry(wg, 30), mats.edge));
  }
  return { g, f, i };
}

function buildStairs() {
  // straight flights (solid block steps, 17 cm rise) joined by flat landing plates
  const geoms = [], st = D.stairs;
  for (const f of st.flights) {
    const run = (f.s1 - f.s0) / f.n, rise = (f.z1 - f.z0) / f.n;
    for (let k = 0; k < f.n; k++) {
      const a = f.s0 + k * run, b = a + run, top = f.z0 + (k + 1) * rise, z0 = Math.max(f.z0 - 0.25, top - 0.6);
      const [x0, x1] = f.axis === 0 ? [Math.min(a, b), Math.max(a, b)] : [f.lat - f.w / 2, f.lat + f.w / 2];
      const [y0, y1] = f.axis === 0 ? [f.lat - f.w / 2, f.lat + f.w / 2] : [Math.min(a, b), Math.max(a, b)];
      geoms.push(boxGeom(x0, y0, x1, y1, z0, top));
    }
  }
  for (const l of st.landings) geoms.push(boxGeom(l[0], l[1], l[2], l[3], l[4] - 0.25, l[4]));
  if (!geoms.length) return;
  const g = mergeGeometries(geoms, false);
  root.add(new THREE.Mesh(g, mats.stair)); root.add(new THREE.LineSegments(new THREE.EdgesGeometry(g, 30), mats.edge));
}

function portrait(url, z) {          // round headshot standing in the destination room, always facing the camera
  const r = D.room, b = r.box, tex = new THREE.CanvasTexture(document.createElement('canvas'));
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
  sp.position.set((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, z); sp.scale.set(1.9, 1.9, 1); sp.renderOrder = 5; root.add(sp);
  const img = new Image();
  img.onload = () => {
    const c = tex.image; c.width = c.height = 256; const g = c.getContext('2d');
    g.beginPath(); g.arc(128, 128, 124, 0, 2 * Math.PI); g.fillStyle = css('--accent'); g.fill();
    g.save(); g.beginPath(); g.arc(128, 128, 112, 0, 2 * Math.PI); g.clip(); g.drawImage(img, 16, 16, 224, 224); g.restore();
    tex.colorSpace = THREE.SRGBColorSpace; tex.needsUpdate = true;
  };
  img.src = url; return sp;
}

function buildRoute() {
  const pts = D.route.map((p) => new THREE.Vector3(p[0], p[1], p[2] + 0.15));
  curve = new THREE.CurvePath(); for (let i = 1; i < pts.length; i++) curve.add(new THREE.LineCurve3(pts[i - 1], pts[i]));
  tubeSegs = Math.max(200, Math.round(D.route_len * 4));
  tube = new THREE.Mesh(new THREE.TubeGeometry(curve, tubeSegs, ROUTE_R, 8, false), mats.route); root.add(tube);
  cum = [0]; cumT = [0];                       // distance (m) and walking time (s) along the route; stairs are walked slower
  for (let i = 1; i < D.route.length; i++) {
    const a = D.route[i - 1], b = D.route[i], L = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]), h = Math.hypot(b[0] - a[0], b[1] - a[1]);
    cum.push(cum[i - 1] + L); cumT.push(cumT[i - 1] + L / (Math.abs(b[2] - a[2]) > 0.2 * Math.max(h, 1e-6) ? STAIR : WALK));
  }
  walker = new THREE.Mesh(new THREE.SphereGeometry(0.35, 20, 14), mats.walker); walker.visible = false; root.add(walker);
  // destination block
  const r = D.room, fz = D.floors[r.floor].z, b = r.box;
  const rg = boxGeom(b[0], b[1], b[2], b[3], fz, fz + Math.min(WALL_H, D.floors[r.floor].h) + 0.05);
  root.add(new THREE.Mesh(rg, mats.room));
  portrait('assets/shane.jpg', fz + 1.35);
}
function setProgress(t) {             // 0..1 of the route drawn
  const segs = Math.round(tubeSegs * Math.max(0, Math.min(1, t)));
  tube.geometry.setDrawRange(0, segs * 8 * 6);
}

// ---------- directions from the route geometry
const fmtT = (sec) => `${Math.floor(Math.round(sec) / 60)}:${String(Math.round(sec) % 60).padStart(2, '0')}`;
const dOf = (a, b) => Math.round(cum[b]) - Math.round(cum[a]);          // rounded so the steps add up to the total exactly
const tOf = (a, b) => Math.round(cumT[b]) - Math.round(cumT[a]);
function interpAt(arr, v) {        // route position (index + fraction) where arr (cum or cumT) reaches v
  let i = 0; while (i < arr.length - 2 && arr[i + 1] < v) i++;
  return i + Math.max(0, Math.min(1, (v - arr[i]) / Math.max(1e-9, arr[i + 1] - arr[i])));
}
function routeIndexAt(d) { let i = 0; while (i < cum.length - 1 && cum[i + 1] < d) i++; return i; }
function turnsOf(a, b) {             // left / right turns (> 50 deg) between route points a..b, with distance from a
  const out = []; const P = D.route.slice(a, b + 1); let dir = null, acc = 0, last = P[0];
  for (let i = 1; i < P.length; i++) {
    const v = [P[i][0] - last[0], P[i][1] - last[1]]; const L = Math.hypot(...v);
    if (L < 1.5) continue;
    const u = [v[0] / L, v[1] / L];
    if (dir) {
      const ang = Math.atan2(dir[0] * u[1] - dir[1] * u[0], dir[0] * u[0] + dir[1] * u[1]) * 180 / Math.PI;
      if (Math.abs(ang) > 50) out.push({ side: ang > 0 ? 'left' : 'right', at: Math.round(acc) });
    }
    acc += L; dir = u; last = P[i];
  }
  return out;
}
function makeSteps() {
  const R = D.route, z0 = R[0][2], zf = D.floors[D.room.floor].z;
  let iS = R.findIndex((p) => p[2] > z0 + 0.3); iS = Math.max(1, iS - 1);
  let iE = R.findIndex((p) => p[2] > zf - 0.15); if (iE < 0) iE = R.length - 2;
  const n = R.length - 1, d = dOf, i12 = Math.min(iS - 1, routeIndexAt(12) + 1);
  const climbed = D.room.floor; // floors from the 1st
  const end = R[n], prev = R[Math.max(0, n - 3)], dir = [end[0] - prev[0], end[1] - prev[1]], s = D.room.sign;
  const side = (dir[0] * (s[1] - end[1]) - dir[1] * (s[0] - end[0])) > 0 ? 'left' : 'right';
  const tc = turnsOf(iE, n);
  const corr = tc.length ? tc.map((t) => `turn ${t.side} after ${t.at} m`).join(', then ') : 'go straight';
  const near = D.labels.filter((l) => l.floor === D.room.floor && l.num !== D.room.num)
    .map((l) => ({ ...l, dd: Math.hypot(l.p[0] - s[0], l.p[1] - s[1]) })).filter((l) => l.dd < 10).sort((a, b) => a.dd - b.dd).slice(0, 2).map((l) => l.num);
  const vEntr = D.entrance;
  steps = [
    { title: 'Overview', text: `The whole route: ${d(0, n)} m from the front entrance, up ${climbed} floors by the central stairs, to Room ${D.room.num}.`, a: 0, b: n, floor: 3, view: 'all' },
    { title: 'Front entrance', text: 'Come in through the glass front doors and the small vestibule on the 1st floor.', a: 0, b: i12, floor: 0 },
    { title: 'Lobby to the stairs', text: `Cross the lobby to the central stairwell (${d(0, iS)} m from the doors).`, a: i12, b: iS, floor: 0 },
    { title: `Up to the ${D.floors[D.room.floor].name}`, text: `Take the stairs up ${climbed} floors, past the 2nd-floor landing.`, a: iS, b: iE, floor: D.room.floor, stair: true },
    { title: `${D.floors[D.room.floor].name} corridor`, text: `Leave the stairs and ${corr}. Walk about ${d(iE, n)} m.`, a: iE, b: n, floor: D.room.floor },
    { title: `Room ${D.room.num}`, text: `Room ${D.room.num} is on your ${side}${near.length && !D.room.ref ? `, near room${near.length > 1 ? 's' : ''} ${near.join(' and ')}` : ''}.${D.room.ref ? ` Its door is ${D.room.ref_doors === 2 ? 'two' : D.room.ref_doors} doors before Room ${D.room.ref}, on the same side.` : (D.room.estimated ? ' (Position estimated from the neighbouring door signs.)' : '')}`, a: n, b: n, floor: D.room.floor, dest: true },
  ];
  steps.forEach((s, k) => {
    s.meta = k === 0 ? `${d(0, n)} m · ${fmtT(tOf(0, n))} walk` : (k === steps.length - 1 ? `Arrive after ${d(0, n)} m · ${fmtT(tOf(0, n))}` : `${d(s.a, s.b)} m · ${fmtT(tOf(s.a, s.b))}`);
  });
  $('steps').innerHTML = steps.map((s, k) => `<li class="step" data-k="${k}"><span class="n">${k}</span><div><h3>${s.title}</h3><p>${s.text}</p><div class="meta">${s.meta}</div></div></li>`).join('');
  $('steps').querySelectorAll('.step').forEach((el) => el.onclick = () => go(+el.dataset.k));
  $('lede').textContent = `A 3D guide built from a LiDAR scan of the building. Follow the maroon line from the front doors to Room ${D.room.num}, step by step.`;
  $('stats').innerHTML = `<div class="stat"><b>${d(0, n)} m</b><span>walking route</span></div>`
    + `<div class="stat"><b>${climbed}</b><span>floors up by the stairs</span></div>`
    + `<div class="stat"><b>${fmtT(tOf(0, n))}</b><span title="${WALK} m/s on the level, ${STAIR} m/s on the stairs">walking time</span></div>`;
  $('h-room').textContent = `Room ${D.room.num}`;
}

// ---------- pins
function addPin(p, html, cls, floor) {
  const el = document.createElement('div'); el.className = 'pin ' + cls;
  el.innerHTML = `<div class="pin-label">${html}</div><div class="pin-stem"></div><div class="pin-dot"></div>`;
  $('pins').appendChild(el); pins.push({ el, w: b2w(p), floor, cls });
}
function makePins() {
  const R = D.route, r = D.room;
  addPin([R[0][0], R[0][1], R[0][2] + 0.2], 'Front entrance<small>1st floor</small>', 'mark', 0);
  const iS = Math.max(0, R.findIndex((p) => p[2] > R[0][2] + 0.3) - 1);
  addPin([R[iS][0], R[iS][1], R[iS][2] + 0.2], 'Stairs<small>to upper floors</small>', 'mark', 0);
  const bc = [(r.box[0] + r.box[2]) / 2, (r.box[1] + r.box[3]) / 2, D.floors[r.floor].z + WALL_H + 0.3];
  addPin(bc, `Room ${r.num}<small>${D.floors[r.floor].name}</small>`, 'dest', r.floor);
  for (const l of D.labels) if (l.num !== r.num) addPin([l.p[0], l.p[1], D.floors[l.floor].z + WALL_H], l.num, 'room', l.floor);
}
const tmp = new THREE.Vector3();
function updatePins() {
  const w = renderer.domElement.clientWidth, h = renderer.domElement.clientHeight;
  const dist = camera.position.distanceTo(controls.target);
  for (const p of pins) {
    tmp.copy(p.w).project(camera);
    let hide = tmp.z > 1 || Math.abs(tmp.x) > 1.1 || Math.abs(tmp.y) > 1.1 || p.floor > focusFloor;
    if (p.cls === 'room') hide = hide || p.floor !== focusFloor || dist > 70;
    p.el.classList.toggle('hide', hide);
    const x = (tmp.x + 1) / 2 * w, y = (1 - tmp.y) / 2 * h;
    p.el.style.transform = `translate(${x}px, ${y}px) translate(-50%, -100%)`;
  }
}

// ---------- floors shown
function showFloors(top) {
  focusFloor = top;
  // floors above the focus one: the next one is a faint ghost, higher ones are hidden
  for (const fl of floors) {
    fl.g.visible = fl.i <= top + 1;
    for (const o of fl.g.children) {
      o.userData.base ??= o.material;
      if (fl.i > top) { o.material = o.userData.ghostMat ??= Object.assign(o.userData.base.clone(), { opacity: 0.1, depthWrite: false }); }
      else o.material = o.userData.base;
    }
  }
  document.querySelectorAll('#floors button').forEach((b) => b.setAttribute('aria-pressed', +b.dataset.f === top));
}

// ---------- camera
let tween = null;
function fly(target, pos, ms = 1200) {
  tween = { t0: performance.now(), ms, fromT: controls.target.clone(), fromP: camera.position.clone(), toT: target, toP: pos };
}
function viewFor(a, b, all) {
  const box = new THREE.Box3();
  const pts = all ? D.floors.flatMap((f) => f.walls.map((w) => [w[0], w[1], f.z])) : D.route.slice(a, b + 1);
  for (const p of pts) box.expandByPoint(b2w(p));
  const c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3());
  const r = Math.max(all ? 10 : 16, Math.hypot(s.x, s.z) * 0.62 + (all ? 0 : 6));
  const az = 0.75, el = all ? 0.72 : 0.95;
  const d = r / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * (camera.aspect < 1 ? 1.4 / camera.aspect : 1) * 0.55;
  return [c, c.clone().add(new THREE.Vector3(Math.cos(az) * Math.cos(el), Math.sin(el), Math.sin(az) * Math.cos(el)).multiplyScalar(d))];
}
function go(k, keepHud = false) {
  cur = Math.max(0, Math.min(steps.length - 1, k)); const s = steps[cur];
  stopWalk(); if (!keepHud) $('hud').hidden = true;
  showFloors(s.floor);
  const [t, p] = viewFor(s.dest ? Math.max(0, s.a - 6) : s.a, s.b, s.view === 'all'); fly(t, p);
  setProgress(cur === 0 ? 1 : cum[s.b] / D.route_len);
  markStep(cur);
  const el = document.querySelector('.step[aria-current]'), list = $('steps');        // scroll the list only, never the page
  if (el && list.scrollHeight > list.clientHeight) list.scrollTo({ top: el.offsetTop - list.offsetTop - 8, behavior: 'smooth' });
  $('count').textContent = `${cur} / ${steps.length - 1}`;
  $('back').disabled = cur === 0; $('next').textContent = cur === steps.length - 1 ? 'Start over' : 'Next →';
}

// ---------- walk animation: simulated walking time (SPEEDUP x real time), live distance / time, a report at each stage
function markStep(k) {
  $('count').textContent = `${k} / ${steps.length - 1}`;
  document.querySelectorAll('.step').forEach((el, i) => (i === k ? el.setAttribute('aria-current', 'step') : el.removeAttribute('aria-current')));
  const el = document.querySelector('.step[aria-current]'), list = $('steps');
  if (el && list.scrollHeight > list.clientHeight) list.scrollTo({ top: el.offsetTop - list.offsetTop - 8, behavior: 'smooth' });
}
function hudUpdate(d, t, k) {
  const n = D.route.length - 1;
  $('hud-step').textContent = k < steps.length - 1 ? `Step ${k} of ${steps.length - 1} · ${steps[k].title}` : `Arrived · Room ${D.room.num}`;
  $('hud-d').textContent = `${Math.round(d)} / ${dOf(0, n)} m`; $('hud-t').textContent = `${fmtT(t)} / ${fmtT(tOf(0, n))}`;
  $('hud-bar').style.width = `${(100 * t / cumT[n]).toFixed(1)}%`;
}
function hudReport(k) {             // stage k finished: same numbers as its panel entry; flashes in the walker's panel
  const s = steps[k], li = document.createElement('li'); li.className = 'flash'; li.dataset.t = performance.now();
  li.innerHTML = `<b>${k}</b> ${s.title}<span>${dOf(s.a, s.b)} m · ${fmtT(tOf(s.a, s.b))}</span>`; $('hud-log').appendChild(li);
}
const hudAt = new THREE.Vector3();
function placeHud() {               // the info panel rides beside the walker (or the end of the route once arrived)
  const el = $('hud'); if (el.hidden) return;
  const now = performance.now();
  el.querySelectorAll('.hud-log li.flash').forEach((e) => { if (now - e.dataset.t > 3000) e.classList.remove('flash'); });
  const p = walker.visible ? walker.position : (() => { const q = D.route[D.route.length - 1]; return hudAt.set(q[0], q[1], q[2] + 0.4); })();
  tmp.copy(p); root.localToWorld(tmp); tmp.project(camera);
  const W = stage.clientWidth, H = stage.clientHeight, w = el.offsetWidth, h = el.offsetHeight;
  let x = (tmp.x + 1) / 2 * W, y = (1 - tmp.y) / 2 * H, flip = x + 26 + w > W - 8;
  x = flip ? x - 26 - w : x + 26; x = Math.max(8, Math.min(W - w - 8, x)); y = Math.max(60, Math.min(H - h - 8, y - h / 2));
  el.classList.toggle('flip', flip); el.style.visibility = tmp.z > 1 ? 'hidden' : 'visible';
  el.style.transform = `translate(${x.toFixed(0)}px, ${y.toFixed(0)}px)`;
}
function startWalk() {
  playing = true; simT = 0; walkStage = 1; walker.visible = true; markStep(1);
  $('hud').hidden = false; $('hud').classList.remove('arrived'); $('hud-log').innerHTML = ''; $('hud-speed').textContent = `${SPEEDUP}× speed`; hudUpdate(0, 0, 1);
  $('play').setAttribute('aria-pressed', 'true'); $('play').textContent = '■ Stop';
}
function stopWalk() { playing = false; if (walker) walker.visible = false; $('play').setAttribute('aria-pressed', 'false'); $('play').textContent = '▶ Walk the route'; }
function walkStep(dt) {
  const n = D.route.length - 1; simT = Math.min(cumT[n], simT + dt * SPEEDUP);
  const x = interpAt(cumT, simT), i = Math.floor(x), f = x - i, dist = cum[i] + f * ((cum[i + 1] ?? cum[i]) - cum[i]);
  while (walkStage < steps.length - 1 && simT >= cumT[steps[walkStage].b] - 1e-6) { hudReport(walkStage); walkStage++; markStep(walkStage); }
  hudUpdate(dist, simT, walkStage);
  const p = curve.getPointAt(Math.min(1, dist / cum[n])); walker.position.copy(p).setZ(p.z + 0.25); setProgress(dist / cum[n]);
  if (simT >= cumT[n]) {            // arrived: final report stays on screen
    const li = document.createElement('li'); li.className = 'total';
    li.innerHTML = `Total<span>${dOf(0, n)} m · ${fmtT(tOf(0, n))}</span>`; $('hud-log').appendChild(li);
    $('hud-log').querySelectorAll('.flash').forEach((e) => e.classList.remove('flash')); $('hud').classList.add('arrived');
    go(steps.length - 1, true); hudUpdate(cum[n], cumT[n], steps.length - 1); return;
  }
  const w = b2w([p.x, p.y, p.z]); const fl = D.floors.reduce((m, f2, j) => (p.z > f2.z - 0.5 ? j : m), 0);
  if (fl !== focusFloor) showFloors(Math.max(fl, 0));
  const off = camera.position.clone().sub(controls.target); controls.target.lerp(w, 0.08); camera.position.copy(controls.target).add(off.setLength(Math.max(18, Math.min(off.length(), 30))));
}
let lastT = performance.now();
function tick(now) {
  const dt = Math.min(0.1, (now - lastT) / 1000); lastT = now;
  if (tween) {
    const u = Math.min(1, (now - tween.t0) / tween.ms), e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
    controls.target.lerpVectors(tween.fromT, tween.toT, e); camera.position.lerpVectors(tween.fromP, tween.toP, e);
    if (u >= 1) tween = null;
  }
  if (playing) walkStep(dt);
  controls.update(); renderer.render(scene, camera); updatePins(); placeHud(); requestAnimationFrame(tick);
}
function resize() {
  const w = stage.clientWidth, h = stage.clientHeight; renderer.setSize(w, h, false);
  renderer.domElement.style.width = w + 'px'; renderer.domElement.style.height = h + 'px';
  camera.aspect = w / h; camera.updateProjectionMatrix();
}

async function main() {
  D = await (await fetch(DATA)).json();
  applyTheme();
  floors = D.floors.map(buildFloor); buildStairs(); buildRoute(); makePins(); makeSteps();
  $('floors').innerHTML = D.floors.map((f, i) => `<button data-f="${i}" title="Show up to the ${f.name}">${i + 1}F</button>`).reverse().join('');
  $('floors').querySelectorAll('button').forEach((b) => b.onclick = () => showFloors(+b.dataset.f));
  $('next').onclick = () => go(cur === steps.length - 1 ? 0 : cur + 1); $('back').onclick = () => go(cur - 1);
  $('reset').onclick = () => go(cur);
  $('play').onclick = () => (playing ? stopWalk() : (go(1), showFloors(0), startWalk()));
  $('xray').onclick = () => setXray($('xray').getAttribute('aria-pressed') !== 'true'); setXray(true);
  $('orbit').onclick = () => { controls.autoRotate = !controls.autoRotate; $('orbit').setAttribute('aria-pressed', controls.autoRotate); };
  $('theme').onclick = () => {
    const dark = matchMedia('(prefers-color-scheme: dark)').matches, t = document.documentElement.dataset.theme;
    document.documentElement.dataset.theme = (t ? t === 'dark' : dark) ? 'light' : 'dark'; applyTheme();
  };
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
  addEventListener('keydown', (e) => { if (e.key === 'ArrowRight') go(cur + 1); if (e.key === 'ArrowLeft') go(cur - 1); });
  new ResizeObserver(resize).observe(stage); resize();
  const [t, p] = viewFor(0, 0, true); controls.target.copy(t); camera.position.copy(p);
  go(0); $('loader').remove(); requestAnimationFrame(tick);
}
main().catch((e) => { $('loader').textContent = 'Could not load the model: ' + e.message; console.error(e); });
