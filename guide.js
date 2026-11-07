// Blunt Hall Guide: block model of the scanned floors + the route from the front entrance to one room.
// Data: data/2026-11-04/guide.json (scripts/pipeline/guide/g03_blocks.py + g04_route.py), building frame, metres, z up.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

const DATA = 'data/2026-11-04/guide.json';
const WALL_H = 2.2, ROUTE_R = 0.13, WALK = 1.3;          // wall height shown (m), route tube radius, walking speed (m/s)
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
  stair: new THREE.MeshLambertMaterial({ color: 0xeeeeee }),
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

let D, floors = [], tube, tubeSegs, curve, cum, walker, pins = [], steps = [], cur = 0, focusFloor = 3, playing = false, walkT = 0;

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
  const geoms = D.stairs.map((s) => {
    const g = new THREE.BoxGeometry(s.len + 0.02, 1.3, 0.3);
    g.rotateZ(Math.atan2(s.dir[1], s.dir[0])); g.translate(s.c[0], s.c[1], s.top - 0.15); return g;
  });
  if (!geoms.length) return;
  const g = mergeGeometries(geoms, false);
  root.add(new THREE.Mesh(g, mats.stair)); root.add(new THREE.LineSegments(new THREE.EdgesGeometry(g, 30), mats.edge));
}

function buildRoute() {
  const pts = D.route.map((p) => new THREE.Vector3(p[0], p[1], p[2] + 0.15));
  curve = new THREE.CurvePath(); for (let i = 1; i < pts.length; i++) curve.add(new THREE.LineCurve3(pts[i - 1], pts[i]));
  tubeSegs = Math.max(200, Math.round(D.route_len * 4));
  tube = new THREE.Mesh(new THREE.TubeGeometry(curve, tubeSegs, ROUTE_R, 8, false), mats.route); root.add(tube);
  cum = [0]; for (let i = 1; i < D.route.length; i++) cum.push(cum[i - 1] + Math.hypot(...[0, 1, 2].map((k) => D.route[i][k] - D.route[i - 1][k])));
  walker = new THREE.Mesh(new THREE.SphereGeometry(0.35, 20, 14), mats.walker); walker.visible = false; root.add(walker);
  // destination block
  const r = D.room, fz = D.floors[r.floor].z, b = r.box;
  const rg = boxGeom(b[0], b[1], b[2], b[3], fz, fz + Math.min(WALL_H, D.floors[r.floor].h) + 0.05);
  root.add(new THREE.Mesh(rg, mats.room));
}
function setProgress(t) {             // 0..1 of the route drawn
  const segs = Math.round(tubeSegs * Math.max(0, Math.min(1, t)));
  tube.geometry.setDrawRange(0, segs * 8 * 6);
}

// ---------- directions from the route geometry
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
  const n = R.length - 1, d = (a, b) => Math.round(cum[b] - cum[a]);
  const climbed = D.room.floor; // floors from the 1st
  const end = R[n], prev = R[Math.max(0, n - 3)], dir = [end[0] - prev[0], end[1] - prev[1]], s = D.room.sign;
  const side = (dir[0] * (s[1] - end[1]) - dir[1] * (s[0] - end[0])) > 0 ? 'left' : 'right';
  const tc = turnsOf(iE, n);
  const corr = tc.length ? tc.map((t) => `turn ${t.side} after ${t.at} m`).join(', then ') : 'go straight';
  const near = D.labels.filter((l) => l.floor === D.room.floor && l.num !== D.room.num)
    .map((l) => ({ ...l, dd: Math.hypot(l.p[0] - s[0], l.p[1] - s[1]) })).filter((l) => l.dd < 10).sort((a, b) => a.dd - b.dd).slice(0, 2).map((l) => l.num);
  const vEntr = D.entrance;
  steps = [
    { title: 'Overview', text: `The whole route: ${Math.round(D.route_len)} m from the front entrance, up ${climbed} floors by the central stairs, to Room ${D.room.num}.`, a: 0, b: n, floor: 3, view: 'all' },
    { title: 'Front entrance', text: 'Come in through the glass front doors and the small vestibule on the 1st floor.', a: 0, b: Math.min(n, routeIndexAt(12)), floor: 0 },
    { title: 'Lobby to the stairs', text: `Cross the lobby to the central stairwell (${d(0, iS)} m from the doors).`, a: 0, b: iS, floor: 0 },
    { title: `Up to the ${D.floors[D.room.floor].name}`, text: `Take the stairs up ${climbed} floors, past the 2nd-floor landing.`, a: iS, b: iE, floor: D.room.floor, stair: true },
    { title: `${D.floors[D.room.floor].name} corridor`, text: `Leave the stairs and ${corr}. Walk about ${d(iE, n)} m.`, a: iE, b: n, floor: D.room.floor },
    { title: `Room ${D.room.num}`, text: `Room ${D.room.num} is on your ${side}${near.length && !D.room.ref ? `, near room${near.length > 1 ? 's' : ''} ${near.join(' and ')}` : ''}.${D.room.ref ? ` Its door is ${D.room.ref_doors === 2 ? 'two' : D.room.ref_doors} doors before Room ${D.room.ref}, on the same side.` : (D.room.estimated ? ' (Position estimated from the neighbouring door signs.)' : '')}`, a: Math.max(0, n - 6), b: n, floor: D.room.floor, dest: true },
  ];
  steps.forEach((s, k) => { s.meta = k === 0 ? `${Math.ceil(D.route_len / WALK / 60)} min walk` : (k === steps.length - 1 ? 'Destination' : `${d(s.a, s.b)} m`); });
  $('steps').innerHTML = steps.map((s, k) => `<li class="step" data-k="${k}"><span class="n">${k}</span><div><h3>${s.title}</h3><p>${s.text}</p><div class="meta">${s.meta}</div></div></li>`).join('');
  $('steps').querySelectorAll('.step').forEach((el) => el.onclick = () => go(+el.dataset.k));
  $('lede').textContent = `A 3D guide built from a LiDAR scan of the building. Follow the maroon line from the front doors to Room ${D.room.num}, step by step.`;
  $('stats').innerHTML = `<div class="stat"><b>${Math.round(D.route_len)} m</b><span>walking route</span></div>`
    + `<div class="stat"><b>${climbed}</b><span>floors up by the stairs</span></div>`
    + `<div class="stat"><b>${Math.ceil(D.route_len / WALK / 60)} min</b><span>at an easy pace</span></div>`;
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
  const bc = [(r.box[0] + r.box[2]) / 2, (r.box[1] + r.box[3]) / 2, D.floors[r.floor].z + WALL_H];
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
function go(k) {
  cur = Math.max(0, Math.min(steps.length - 1, k)); const s = steps[cur];
  stopWalk();
  showFloors(s.floor);
  const [t, p] = viewFor(s.a, s.b, s.view === 'all'); fly(t, p);
  setProgress(cur === 0 ? 1 : cum[s.b] / D.route_len);
  document.querySelectorAll('.step').forEach((el, i) => el.toggleAttribute('aria-current', false) || (i === cur && el.setAttribute('aria-current', 'step')));
  const el = document.querySelector('.step[aria-current]'), list = $('steps');        // scroll the list only, never the page
  if (el && list.scrollHeight > list.clientHeight) list.scrollTo({ top: el.offsetTop - list.offsetTop - 8, behavior: 'smooth' });
  $('count').textContent = `${cur} / ${steps.length - 1}`;
  $('back').disabled = cur === 0; $('next').textContent = cur === steps.length - 1 ? 'Start over' : 'Next →';
}

// ---------- walk animation
function startWalk() { playing = true; walkT = 0; walker.visible = true; $('play').setAttribute('aria-pressed', 'true'); $('play').textContent = '■ Stop'; }
function stopWalk() { playing = false; if (walker) walker.visible = false; $('play').setAttribute('aria-pressed', 'false'); $('play').textContent = '▶ Walk the route'; }
let lastT = performance.now();
function tick(now) {
  const dt = Math.min(0.1, (now - lastT) / 1000); lastT = now;
  if (tween) {
    const u = Math.min(1, (now - tween.t0) / tween.ms), e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
    controls.target.lerpVectors(tween.fromT, tween.toT, e); camera.position.lerpVectors(tween.fromP, tween.toP, e);
    if (u >= 1) tween = null;
  }
  if (playing) {
    walkT += dt * WALK * 6 / D.route_len;            // 6x real time
    if (walkT >= 1) { walkT = 1; stopWalk(); go(steps.length - 1); }
    else {
      const p = curve.getPointAt(walkT); walker.position.copy(p).setZ(p.z + 0.25); setProgress(walkT);
      const w = b2w([p.x, p.y, p.z]); const fl = D.floors.reduce((m, f, i) => (p.z > f.z - 0.5 ? i : m), 0);
      if (fl !== focusFloor) showFloors(Math.max(fl, 0));
      const off = camera.position.clone().sub(controls.target); controls.target.lerp(w, 0.08); camera.position.copy(controls.target).add(off.setLength(Math.max(18, Math.min(off.length(), 30))));
    }
  }
  controls.update(); renderer.render(scene, camera); updatePins(); requestAnimationFrame(tick);
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
  $('play').onclick = () => (playing ? stopWalk() : (go(0), showFloors(0), startWalk()));
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
