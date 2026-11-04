import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';

const $ = (s) => document.querySelector(s);
const viewport = $('#viewport');
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

// ---------- three.js scene (SLAM frame is Z-up) ----------
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.localClippingEnabled = true;
viewport.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, 1, 0.05, 800);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
let splatActive = false;
const world = new THREE.Group();
scene.add(world);
scene.add(new THREE.HemisphereLight(0xffffff, 0x666666, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.2); sun.position.set(10, -6, 20); scene.add(sun);

function setBg() { scene.background = new THREE.Color(css('--viewer-bg') || '#e9e7e1'); }
setBg();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', setBg);

const pointMats = new Set();
function resize() {
  const w = viewport.clientWidth, h = viewport.clientHeight;
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
  for (const m of pointMats) m.uniforms.vh.value = h * renderer.getPixelRatio();
}
addEventListener('resize', resize); resize();
const clock = new THREE.Clock();
let walkRef = null;   // set once the walk module section has run (avoids a temporal-dead-zone error in the first frames)
renderer.setAnimationLoop(() => { const dt = Math.min(clock.getDelta(), 0.1); if (walkRef?.active) walkRef.step(dt); else controls.update(); if (!splatActive) renderer.render(scene, camera); });

// ---------- loading helpers ----------
const box = $('#loading');
function busy(label, frac) { box.hidden = false; box.querySelector('.msg').textContent = label; box.querySelector('.fill').style.width = `${(100 * (frac || 0)).toFixed(0)}%`; }
function idle() { box.hidden = true; }
async function fetchBuf(url, label) {
  busy(label, 0);
  const res = await fetch(url);
  if (!res.ok) { idle(); throw new Error(`${url}: ${res.status}`); }
  const total = +res.headers.get('content-length') || 0;
  const reader = res.body.getReader(); const chunks = []; let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break;
    chunks.push(value); got += value.length; if (total) busy(label, got / total); }
  idle();
  const out = new Uint8Array(got); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; }
  return out.buffer;
}

// ---------- datasets & floors ----------
const datasets = (await (await fetch('data/datasets.json')).json()).datasets;
const params = new URLSearchParams(location.hash.slice(1));
let ds = datasets.find(d => d.id === params.get('scan')) || datasets.find(d => d.default) || datasets.at(-1);
let meta, base, floorIx;
const cache = new Map();                       // url -> parsed object (per dataset files are distinct urls)
async function loadDataset(d) {
  ds = d; base = `data/${d.id}/`;
  meta = await (await fetch(base + 'meta.json')).json();
  const f = params.get('floor'); floorIx = f === 'all' ? -1 : (f !== null && meta.floors[+f] ? +f : meta.main_floor);
  framed = false;
}
const levels = () => meta.floors.map((f, j) => [f, j]).filter(([f]) => f.kind !== 'area');
const FL = () => floorIx < 0 ? { name: 'All floors', zmin: Math.min(...levels().map(([f]) => f.zmin)), zmax: Math.max(...levels().map(([f]) => f.zmax)),
  z: levels()[0][0].z, ceiling: Math.max(...levels().map(([f]) => f.ceiling)) } : meta.floors[floorIx];
const floorsShown = () => floorIx < 0 ? levels().map(([, j]) => j) : [floorIx];

// ---------- point clouds ----------
// points_*.bin: 'BHPC' | u32 N | u32 flags(1=rgb) | f32 origin[3] | f32 scale | u16 xyz[N*3] | u8 intensity[N] | u8 rgb[N*3]
const cloudMat = new THREE.ShaderMaterial({
  uniforms: { mode: { value: 0 }, size: { value: 1.6 }, vh: { value: 800 }, spacing: { value: 0.02 }, zmin: { value: 0 }, zmax: { value: 3 },
              clipTop: { value: 99 }, clipBot: { value: -99 }, gain: { value: 1.5 }, boost: { value: 1.3 } },
  vertexShader: `
    attribute float intensity; attribute vec3 rgb;
    uniform int mode; uniform float size, zmin, zmax, gain, clipTop, clipBot, boost, vh, spacing;
    varying vec3 vCol; varying float vClip;
    vec3 turbo(float x){ x=clamp(x,0.,1.);
      vec4 kr=vec4(.13572138,4.61539260,-42.66032258,132.13108234), kg=vec4(.09140261,2.19418839,4.84296658,-14.18503333), kb=vec4(.10667330,12.64194608,-60.58204836,110.36276771);
      vec2 kr2=vec2(-152.94239396,59.28637943), kg2=vec2(4.27729857,2.82956604), kb2=vec2(-89.90310912,27.34824973);
      vec4 v4=vec4(1.,x,x*x,x*x*x); vec2 v2=v4.zw*v4.z;
      return vec3(dot(v4,kr)+dot(v2,kr2), dot(v4,kg)+dot(v2,kg2), dot(v4,kb)+dot(v2,kb2)); }
    void main(){
      vClip = (position.z > clipTop || position.z < clipBot) ? 1. : 0.;
      if (mode==0) { float v=clamp(intensity*gain,0.,1.); vCol=vec3(v); }
      else if (mode==1) { vec3 c=(rgb-0.5)*(1.0+0.6*(boost-1.0))+0.5; float l=dot(c,vec3(.299,.587,.114)); vCol=clamp(mix(vec3(l),c,boost*boost),0.,1.); }
      else vCol=turbo((position.z-zmin)/(zmax-zmin));
      vec4 mv=modelViewMatrix*vec4(position,1.);
      gl_PointSize = clamp(size * spacing * projectionMatrix[1][1] * vh * 0.5 / max(-mv.z, 0.05), 1.0, 14.0);
      gl_Position = projectionMatrix*mv; }`,
  fragmentShader: `varying vec3 vCol; varying float vClip;
    void main(){ if (vClip>0.5) discard; vec2 c=gl_PointCoord-.5; if(dot(c,c)>.25) discard; gl_FragColor=vec4(vCol,1.); }`,
});
pointMats.add(cloudMat); resize();

function parsePoints(buf) {
  const dv = new DataView(buf);
  const N = dv.getUint32(4, true), flags = dv.getUint32(8, true);
  const ox = dv.getFloat32(12, true), oy = dv.getFloat32(16, true), oz = dv.getFloat32(20, true), sc = dv.getFloat32(24, true);
  let off = 28;
  const q = new Uint16Array(buf, off, N * 3); off += N * 6;
  const inten = new Uint8Array(buf, off, N); off += N;
  const rgb = flags & 1 ? new Uint8Array(buf, off, N * 3) : null;
  const pos = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) { pos[3*i] = ox + q[3*i]*sc; pos[3*i+1] = oy + q[3*i+1]*sc; pos[3*i+2] = oz + q[3*i+2]*sc; }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('intensity', new THREE.BufferAttribute(inten, 1, true));
  if (rgb) g.setAttribute('rgb', new THREE.BufferAttribute(rgb, 3, true));
  g.computeBoundingSphere();
  return { points: new THREE.Points(g, cloudMat), N, hasRGB: !!rgb };
}
async function loadCloud() {
  const files = floorsShown().flatMap(j => meta.floors[j].points);
  const parts = [];
  for (const [i, f] of files.entries()) {
    const url = base + f;
    if (!cache.has(url)) cache.set(url, parsePoints(await fetchBuf(url, `Loading points ${i + 1} / ${files.length}…`)));
    parts.push(cache.get(url));
  }
  cloudMat.uniforms.spacing.value = (meta.voxel_cm || 2) / 100;
  return { parts, N: parts.reduce((a, p) => a + p.N, 0), hasRGB: parts.every(p => p.hasRGB) };
}

// ---------- trajectory & camera stations ----------
async function loadTraj() {
  const url = base + 'trajectory.json';
  if (!cache.has(url)) {
    const tj = await (await fetch(url)).json();
    const grp = new THREE.Group();
    const lg = new THREE.BufferGeometry().setFromPoints(tj.p.map(p => new THREE.Vector3(...p)));
    const lineMat = new THREE.LineBasicMaterial({ color: 0xd62728 });
    grp.add(new THREE.Line(lg, lineMat));
    const sg = new THREE.SphereGeometry(0.08, 12, 8);
    const s0 = new THREE.Mesh(sg, new THREE.MeshBasicMaterial({ color: 0x2ca02c })); s0.position.set(...tj.p[0]); grp.add(s0);
    const s1 = new THREE.Mesh(sg, new THREE.MeshBasicMaterial({ color: 0x1f77b4 })); s1.position.set(...tj.p.at(-1)); grp.add(s1);
    let camMat = null;
    if (tj.cams) {
      const cg = new THREE.BufferGeometry().setFromPoints(tj.cams.map(p => new THREE.Vector3(...p)));
      camMat = new THREE.PointsMaterial({ color: 0xff9f1c, size: 0.06 }); grp.add(new THREE.Points(cg, camMat));
    }
    cache.set(url, { group: grp, data: tj, mats: [lineMat, camMat, s0.material, s1.material].filter(Boolean) });
  }
  return cache.get(url);
}

// ---------- mesh model (one PLY per tile) ----------
const meshUnlit = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
const meshLit = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, metalness: 0, side: THREE.DoubleSide });
const tilesOf = () => meta.model.tiles ? meta.model.tiles.filter(t => !t.floors || t.floors.some(j => floorsShown().includes(j))) : [{ file: meta.model.file }];
async function loadModel() {
  const tiles = tilesOf(); const grp = new THREE.Group(); let tris = 0;
  for (const [i, t] of tiles.entries()) {
    const url = base + t.file;
    if (!cache.has(url)) {
      const geom = new PLYLoader().parse(await fetchBuf(url, `Loading 3D model ${i + 1} / ${tiles.length}…`));
      geom.computeVertexNormals(); cache.set(url, new THREE.Mesh(geom, meshUnlit));
    }
    const m = cache.get(url); grp.add(m); tris += m.geometry.index ? m.geometry.index.count / 3 : m.geometry.attributes.position.count / 3;
  }
  return { group: grp, tris };
}
const setMeshMat = (grp, mat) => grp.traverse(o => { if (o.isMesh) o.material = mat; });


// ---------- structural building model (textured glTF, Y-up -> rotate back to Z-up) ----------
let bvh = null;
async function loadBVH() { if (!bvh) { bvh = await import('three-mesh-bvh'); THREE.Mesh.prototype.raycast = bvh.acceleratedRaycast; } return bvh; }
const buildFloors = () => {
  const s = meta.structure; if (!s) return [];
  const f = FL(); const want = floorIx < 0 ? levels().map(([, j]) => j) : (f.kind === 'area' ? (f.box && f.zmax - f.zmin > 6 ? levels().map(([, j]) => j) : [f.floor]) : [floorIx]);
  return s.files.filter(x => want.includes(x.floor));
};
async function loadBuild() {
  const B = await loadBVH(); const files = buildFloors(); const grp = new THREE.Group(); let tris = 0;
  for (const [i, x] of files.entries()) {
    const url = base + x.file;
    if (!cache.has(url)) {
      const buf = await fetchBuf(url, `Loading building model ${i + 1} / ${files.length}…`);
      const gltf = await new GLTFLoader().parseAsync(buf, base);
      const root = gltf.scene; root.rotation.x = Math.PI / 2; root.updateMatrixWorld(true);
      root.traverse(o => { if (!o.isMesh) return;
        const m = o.material; o.material = new THREE.MeshBasicMaterial({ map: m.map || null, vertexColors: !m.map && !!o.geometry.attributes.color, side: THREE.DoubleSide });
        if (o.material.map) o.material.map.colorSpace = THREE.SRGBColorSpace;
        o.geometry.computeBoundsTree = B.computeBoundsTree; o.geometry.computeBoundsTree(); });
      cache.set(url, root);
    }
    const r = cache.get(url); grp.add(r); r.traverse(o => { if (o.isMesh) tris += o.geometry.index ? o.geometry.index.count / 3 : 0; });
  }
  return { group: grp, tris };
}

// ---------- walk mode: first person, collisions with the model, gravity onto floors / stairs ----------
const walk = { active: false, pl: null, keys: {}, vz: 0, eye: 1.6, colliders: [] }; walkRef = walk;
addEventListener('keydown', e => { walk.keys[e.code] = true; });
addEventListener('keyup', e => { walk.keys[e.code] = false; });
const _ray = new THREE.Raycaster(); _ray.firstHitOnly = true;
function hit(origin, dir, far) { _ray.set(origin, dir); _ray.far = far; const h = _ray.intersectObjects(walk.colliders, true); return h.length ? h[0] : null; }
walk.step = function (dt) {
  const k = walk.keys, sp = (k.ShiftLeft || k.ShiftRight ? 3.2 : 1.5) * dt;
  const fwd = new THREE.Vector3(); camera.getWorldDirection(fwd); fwd.z = 0; fwd.normalize();
  const right = new THREE.Vector3().crossVectors(fwd, new THREE.Vector3(0, 0, 1)).normalize();
  const mv = new THREE.Vector3();
  if (k.KeyW || k.ArrowUp) mv.add(fwd); if (k.KeyS || k.ArrowDown) mv.sub(fwd);
  if (k.KeyD || k.ArrowRight) mv.add(right); if (k.KeyA || k.ArrowLeft) mv.sub(right);
  const p = camera.position;
  if (mv.lengthSq() > 0) {
    mv.normalize();
    for (const hgt of [-1.2, -0.6, 0.1]) {                       // knee, waist, head: stop 0.3 m before walls
      const o = p.clone(); o.z += hgt; const h = hit(o, mv, sp + 0.3);
      if (h) { const n = h.face ? h.face.normal.clone().transformDirection(h.object.matrixWorld) : null;
        if (n && Math.abs(n.z) < 0.6 && hgt > -1.3) { n.z = 0; n.normalize(); mv.addScaledVector(n, -mv.dot(n)); } } }
    p.addScaledVector(mv, sp);
  }
  // ground: highest surface below the eye (steps up to 0.45 m)
  const g = hit(new THREE.Vector3(p.x, p.y, p.z - walk.eye + 0.45), new THREE.Vector3(0, 0, -1), 3);
  if (g) { const target = g.point.z + walk.eye; if (p.z <= target + 0.02) { p.z += (target - p.z) * Math.min(1, dt * 12); walk.vz = 0; }
    else { walk.vz -= 9.8 * dt; p.z = Math.max(target, p.z + walk.vz * dt); } }
};
function startWalk(model) {
  if (!walk.pl) {
    walk.pl = new PointerLockControls(camera, renderer.domElement);
    walk.pl.addEventListener('unlock', () => { walk.active = false; $('#walkhud').hidden = true; $('#panel').style.display = ''; $('#crosshair')?.remove(); controls.enabled = true;
      controls.target.copy(camera.position).add(camera.getWorldDirection(new THREE.Vector3()).multiplyScalar(3)); controls.update(); });
  }
  walk.colliders = [model.group];
  const f = FL(); const zf = f.kind === 'area' ? meta.floors[f.floor].z : f.z;
  const tj = cache.get(base + 'trajectory.json').data;
  const on = tj.p.filter(q => q[2] - zf > 0.5 && q[2] - zf < 2.0 && (!f.box || (q[0] > f.box[0] && q[0] < f.box[1] && q[1] > f.box[2] && q[1] < f.box[3])));
  // start at the straightest stretch of the walk on this floor (largest displacement over the next 8 samples ~ 4 s), looking along it
  const stairBoxes = meta.floors.filter(g => g.stairs && g.box).map(g => g.box);
  const notStair = (q) => !stairBoxes.some(b => q[0] > b[0] && q[0] < b[1] && q[1] > b[2] && q[1] < b[3]);
  const on2 = on.filter(notStair); let best = 0, bi = 0; const P = on2.length > 10 ? on2 : (on.length > 10 ? on : tj.p);
  for (let i = 0; i + 8 < P.length; i++) { const d = Math.hypot(P[i + 8][0] - P[i][0], P[i + 8][1] - P[i][1]); if (d > best) { best = d; bi = i; } }
  const q = P[bi], q2 = P[Math.min(bi + 8, P.length - 1)];
  camera.position.set(q[0], q[1], zf + walk.eye); camera.up.set(0, 0, 1); camera.lookAt(q2[0], q2[1], zf + walk.eye);
  controls.enabled = false; walk.active = true; walk.vz = 0; $('#walkhud').hidden = false; $('#panel').style.display = 'none';
  const ch = document.createElement('div'); ch.id = 'crosshair'; viewport.appendChild(ch);
  walk.pl.lock();
}

// ---------- gaussian splat ----------
let splatViewer = null, splatFor = null;
async function showSplat() {
  if (!meta.splat) return false;
  const GS = await import('@mkkellogg/gaussian-splats-3d');
  const key = ds.id + '/' + floorIx;
  if (splatViewer && splatFor !== key) { try { await splatViewer.dispose(); } catch (e) { console.warn(e); } splatViewer = null; }
  if (!splatViewer) {
    const sp = meta.splat; const v = sp.views_by_floor?.[floorIx] || sp;
    splatViewer = new GS.Viewer({ rootElement: viewport, selfDrivenMode: true, useBuiltInControls: true,
      cameraUp: [0, 0, 1], initialCameraPosition: v.cam_pos, initialCameraLookAt: v.cam_look,
      sharedMemoryForWorkers: false, gpuAcceleratedSort: false });
    const files = (sp.files || [{ file: sp.file }]).filter(f => f.floor === undefined || floorIx < 0 || f.floor === floorIx).map(f => f.file);
    busy('Loading Gaussian splat…', 0);
    for (const [i, f] of files.entries())
      await splatViewer.addSplatScene(f.startsWith('data/') ? f : base + f, { showLoadingUI: false, progressiveLoad: files.length === 1,
        onProgress: (p) => busy(`Loading Gaussian splat ${files.length > 1 ? `${i + 1} / ${files.length}` : ''}…`, (i + p / 100) / files.length) });
    idle(); splatFor = key;
  }
  splatViewer.renderer.domElement.style.display = '';
  splatViewer.start?.();
  return true;
}
function hideSplat() { if (splatViewer) { splatViewer.renderer.domElement.style.display = 'none'; splatViewer.stop?.(); } }

// ---------- SLAM 2D map ----------
async function drawSlam() {
  const m = meta.slam_map; const f = floorIx < 0 ? meta.floors[meta.main_floor] : meta.floors[floorIx];
  const img = new Image(); img.src = base + f.slam_map; await img.decode();
  const tj = (await loadTraj()).data;
  const cv = $('#slamcanvas'); const k = Math.min(1, 1800 / Math.max(img.width, img.height));
  cv.width = img.width * k; cv.height = img.height * k;
  const ctx = cv.getContext('2d'); ctx.imageSmoothingEnabled = false; ctx.drawImage(img, 0, 0, cv.width, cv.height);
  const px = (x, y) => [(x - m.xmin) / m.res * k, (m.ymax - y) / m.res * k];
  const onFloor = (p) => floorIx < 0 || (p[2] - f.z > 0.5 && p[2] - f.z < 2.0) || (f.box && p[2] > f.zmin && p[2] < f.zmax && p[0] > f.box[0] && p[0] < f.box[1] && p[1] > f.box[2] && p[1] < f.box[3]);
  ctx.lineWidth = 2; ctx.strokeStyle = '#d62728'; ctx.beginPath(); let pen = false;
  tj.p.forEach((p) => { const [u, v] = px(p[0], p[1]); if (!onFloor(p)) { pen = false; return; } pen ? ctx.lineTo(u, v) : ctx.moveTo(u, v); pen = true; }); ctx.stroke();
  const dot = (p, c) => { const [u, v] = px(p[0], p[1]); ctx.fillStyle = c; ctx.beginPath(); ctx.arc(u, v, 6, 0, 7); ctx.fill(); };
  if (onFloor(tj.p[0])) dot(tj.p[0], '#2ca02c'); if (onFloor(tj.p.at(-1))) dot(tj.p.at(-1), '#1f77b4');
  if (f.box) { const [a, b] = px(f.box[0], f.box[3]), [c2, d] = px(f.box[1], f.box[2]); ctx.strokeStyle = '#ff8c00'; ctx.lineWidth = 3; ctx.setLineDash([8, 6]); ctx.strokeRect(a, b, c2 - a, d - b); ctx.setLineDash([]); }
  const bar = cv.width > 1200 ? 10 : 5; const L = bar / m.res * k; ctx.fillStyle = '#1d1f22'; ctx.fillRect(20, cv.height - 30, L, 6);
  ctx.font = '13px system-ui'; ctx.fillText(`${bar} m`, 20, cv.height - 36);
}

// ---------- UI ----------
const info = $('#info'), ctl = $('#controls'), stats = $('#stats'), pick = $('#pick');
const fmt = (n) => n.toLocaleString();
function statRows(rows) { stats.innerHTML = rows.map(([a, b]) => `<div><span>${a}</span><span>${b}</span></div>`).join(''); }
let framed = false;
async function frame() {
  if (framed) return; framed = true;
  const f = FL(); let c = new THREE.Vector3(...(f.focus || meta.focus));
  if (f.box) c = new THREE.Vector3((f.box[0] + f.box[1]) / 2, (f.box[2] + f.box[3]) / 2, f.zmin + (f.zmax - f.zmin) * 0.35);
  else if (!f.focus && floorIx >= 0) {
    const tj = (await loadTraj()).data; const on = tj.p.filter(p => p[2] - f.z > 0.5 && p[2] - f.z < 2.0);
    if (on.length) { const med = (k) => on.map(p => p[k]).sort((a, b) => a - b)[on.length >> 1]; c = new THREE.Vector3(med(0), med(1), f.z + 0.5); }
  }
  const span = f.box ? Math.max(f.box[1] - f.box[0], f.box[3] - f.box[2], f.zmax - f.zmin) / 9 : (meta.floors.length > 1 || meta.path_m > 200 ? 3 : 1);
  controls.target.copy(c); camera.position.set(c.x - 3.5 * span, c.y - 7 * span, c.z + 6 * span); controls.update();
}
function clearWorld() { world.clear(); }
const CUT = () => FL().ceiling;
const clipRange = () => { const f = FL(); return `min="${(f.z + 0.5).toFixed(2)}" max="${(f.zmax + 3).toFixed(2)}" step="0.05" value="${CUT()}"`; };

function renderPicker() {
  const scanSel = datasets.length > 1 ? `<label>Scan <select id="scanSel">${datasets.map(d => `<option value="${d.id}" ${d.id === ds.id ? 'selected' : ''}>${d.title}</option>`).join('')}</select></label>` : '';
  const opt = ([f, j]) => `<option value="${j}" ${j === floorIx ? 'selected' : ''}>${f.name}</option>`;
  const areas = meta.floors.map((f, j) => [f, j]).filter(([f]) => f.kind === 'area');
  const floorSel = meta.floors.length > 1 ? `<label>View <select id="floorSel"><optgroup label="Floors">${levels().map(opt).join('')}<option value="-1" ${floorIx < 0 ? 'selected' : ''}>All floors</option></optgroup>
      ${areas.length ? `<optgroup label="Areas">${areas.map(opt).join('')}</optgroup>` : ''}</select></label>` : '';
  pick.innerHTML = scanSel + floorSel; pick.hidden = !(scanSel || floorSel);
  $('#scanSel') && ($('#scanSel').onchange = async (e) => { params.delete('floor'); await loadDataset(datasets.find(d => d.id === e.target.value)); renderPicker(); show(current); });
  $('#floorSel') && ($('#floorSel').onchange = (e) => { floorIx = +e.target.value; framed = false; show(current); });
  $('.brand .sub').textContent = ds.subtitle || '';
}

function cloudPanel(c, t, rgb) {
  const f = FL();
  clearWorld(); c.parts.forEach(p => world.add(p.points)); world.add(t.group); frame();
  const u = cloudMat.uniforms; u.mode.value = rgb ? (c.hasRGB ? 1 : 0) : 0; u.zmin.value = f.z; u.zmax.value = f.ceiling + 0.15; u.clipBot.value = f.zmin;
  ctl.innerHTML = (rgb ? '' : `<label>Color <select id="cmode"><option value="0">Intensity</option><option value="2">Height</option></select></label>`) + `
    <label>Point size <input id="psize" type="range" min="0.6" max="4" step="0.1" value="1.6"></label>
    ${rgb ? '<label>Color boost <input id="boost" type="range" min="1" max="2" step="0.05" value="1.3"></label>' : '<label>Intensity gain <input id="gain" type="range" min="0.5" max="4" step="0.1" value="1.5"></label>'}
    <label>Cut ceiling <input id="clip" type="range" ${clipRange()}></label>
    <label>Trajectory <input id="trj" type="checkbox" ${rgb ? '' : 'checked'}></label>
    <div id="leg"></div>` + (rgb ? `<p class="note">For photo-realistic texture, open the Gaussian splat or Building model tab — points ${meta.voxel_cm} cm apart cannot carry fine photo detail.</p>` : '');
  $('#psize').oninput = (e) => { u.size.value = +e.target.value; };
  $('#clip').oninput = (e) => { u.clipTop.value = +e.target.value; };
  $('#gain') && ($('#gain').oninput = (e) => { u.gain.value = +e.target.value; });
  $('#boost') && ($('#boost').oninput = (e) => { u.boost.value = +e.target.value; });
  u.clipTop.value = +$('#clip').value; u.size.value = +$('#psize').value;
  $('#trj').onchange = (e) => { t.group.visible = e.target.checked; }; t.group.visible = $('#trj').checked;
  const legend = () => { const leg = $('#leg'); if (!leg) return;
    leg.innerHTML = u.mode.value === 2 ? `<div class="legend" style="background:linear-gradient(90deg,#30123b,#4686fb,#1ae4b6,#a2fc3c,#fabb39,#e4460a,#7a0403)"></div>
      <div class="legend-labels"><span>${u.zmin.value.toFixed(1)} m</span><span>height</span><span>${u.zmax.value.toFixed(1)} m</span></div>` : ''; };
  $('#cmode') && ($('#cmode').onchange = (e) => { u.mode.value = +e.target.value; legend(); });
  legend();
}

const views = {
  async lidar() {
    const c = await loadCloud(); const t = await loadTraj();
    info.innerHTML = `<h2>LiDAR points</h2><p>${fmt(meta.n_raw)} raw returns from the Hesai 16-beam scanner, registered with the SLAM trajectory and thinned to a ${meta.voxel_cm} cm voxel grid.</p>`;
    cloudPanel(c, t, false);
    statRows([['Points shown', fmt(c.N)], ['Scan frames', fmt(meta.n_frames)], ['Duration', `${meta.duration_s} s`], ['Path length', `${meta.path_m} m`]]);
  },
  async rgb() {
    const c = await loadCloud(); const t = await loadTraj();
    info.innerHTML = `<h2>Colored points</h2><p>Each point takes its color from the three fisheye cameras (${fmt(meta.n_images)} photos), as a weighted blend of every photo that sees it unoccluded (closer and more central views count more).</p>
      <p class="note">${meta.rgb_note || ''}</p>`;
    cloudPanel(c, t, true);
    statRows([['Points shown', fmt(c.N)], ['Colored', meta.rgb_frac ? `${(100 * meta.rgb_frac).toFixed(1)} %` : '—'], ['Cameras', '3 × 5 MP fisheye']]);
  },
  async model() {
    const m = await loadModel(); const t = await loadTraj(); const f = FL();
    clearWorld(); world.add(m.group); world.add(t.group); t.group.visible = false; frame();
    info.innerHTML = `<h2>3D model</h2><p>${meta.model.method}</p><p class="note">Covers the well-scanned zone only (within ${meta.crop_m || 5} m of the walking path); areas seen just through doorways were left out rather than shown as fragments.</p>`;
    ctl.innerHTML = `
      <label>Shading <select id="shade"><option value="p">Photo color (unlit)</option><option value="c">Photo color + shading</option><option value="g">Gray (shaded)</option><option value="w">Wireframe</option></select></label>
      <label>Cut ceiling <input id="mclip" type="range" ${clipRange()}></label>
      <label>Trajectory <input id="trj" type="checkbox"></label>`;
    const top = new THREE.Plane(new THREE.Vector3(0, 0, -1), CUT()), bot = new THREE.Plane(new THREE.Vector3(0, 0, 1), -f.zmin);
    meshLit.clippingPlanes = [top, bot]; meshUnlit.clippingPlanes = [top, bot];
    $('#mclip').oninput = (e) => { top.constant = +e.target.value; };
    const setShade = (v) => { if (v === 'p') return setMeshMat(m.group, meshUnlit);
      meshLit.wireframe = v === 'w'; meshLit.vertexColors = v === 'c'; meshLit.color.set(v === 'c' ? 0xffffff : 0xcfcac0); meshLit.needsUpdate = true; setMeshMat(m.group, meshLit); };
    $('#shade').onchange = (e) => setShade(e.target.value); setShade('p');
    $('#trj').onchange = (e) => { t.group.visible = e.target.checked; };
    statRows([['Triangles', fmt(m.tris)], ['Method', meta.model.short]]);
  },
  async build() {
    if (!meta.structure) { clearWorld(); info.innerHTML = '<h2>Building model</h2><p class="note">The structural building model for this scan is not built yet.</p>'; ctl.innerHTML = ''; statRows([]); return; }
    const m = await loadBuild(); const t = await loadTraj(); const f = FL();
    clearWorld(); world.add(m.group); world.add(t.group); t.group.visible = false; frame();
    info.innerHTML = `<h2>Building model</h2><p>${meta.structure.note}</p>`;
    ctl.innerHTML = `<button class="btn" id="walkBtn">Walk in the building</button>
      <p class="note">Walk mode: WASD / arrow keys to move, Shift to run, mouse to look, Esc to exit. Desktop browsers only.</p>
      <label>Cut ceiling <input id="bclip" type="range" ${clipRange()}></label>
      <label>Trajectory <input id="trj" type="checkbox"></label>
      <p><a href="${base}${(buildFloors()[0] || {}).file || ''}" download>Download this floor (.glb)</a> · glTF 2.0, Y-up, metres</p>`;
    const top = new THREE.Plane(new THREE.Vector3(0, 0, -1), CUT() + 0.5);
    m.group.traverse(o => { if (o.isMesh) o.material.clippingPlanes = [top]; });
    $('#bclip').value = CUT() + 0.5; $('#bclip').oninput = (e) => { top.constant = +e.target.value; };
    $('#trj').onchange = (e) => { t.group.visible = e.target.checked; };
    $('#walkBtn').onclick = () => { top.constant = 99; startWalk(m); };
    statRows([['Faces', fmt(meta.structure.faces)], ['Texture', `${meta.structure.res_cm} cm / texel`], ['Triangles shown', fmt(Math.round(m.tris))]]);
  },
  async slam() {
    await drawSlam();
    info.innerHTML = `<h2>SLAM map</h2><p>Top-down occupancy map: wall hits in a horizontal slice ${meta.slam_map.slice}, with the SLAM-optimized walking path on this floor (green = start, blue = end).</p>`;
    ctl.innerHTML = `<p>${meta.slam_map.note}</p>`;
    statRows([['Poses', fmt(meta.n_frames)], ['Loop closures', meta.loop_closures ?? '—'], ['Grid', `${meta.slam_map.grid_cm} cm`], ['Path length', `${meta.path_m} m`]]);
  },
  async splat() {
    clearWorld();
    const ok = await showSplat().catch((e) => { console.error(e); return false; });
    info.innerHTML = ok ? `<h2>Gaussian splat</h2><p>${meta.splat.note}</p>` :
      `<h2>Gaussian splat</h2><p class="note">The Gaussian Splatting model for this scan is not built yet.</p>`;
    ctl.innerHTML = ''; statRows(ok ? [['Gaussians', fmt(meta.splat.n)], ['Training views', fmt(meta.splat.views)]] : []);
    splatActive = ok;
  },
};

// double-click to refocus orbit
const ray = new THREE.Raycaster(); ray.params.Points.threshold = 0.05;
renderer.domElement.addEventListener('dblclick', (e) => {
  const r = renderer.domElement.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
  const hit = ray.intersectObjects(world.children, true)[0]; if (hit) controls.target.copy(hit.point);
});

window.__bh = { camera, world, controls, scene, renderer, get walk() { return walkRef; } };
let current = 'lidar';
async function show(name) {
  current = name;
  document.querySelectorAll('#tabs button').forEach(b => b.classList.toggle('active', b.dataset.view === name));
  const is2d = name === 'slam';
  $('#slam2d').hidden = !is2d; viewport.style.visibility = is2d ? 'hidden' : 'visible';
  $('#help').style.display = is2d ? 'none' : '';
  if (name !== 'splat') { hideSplat(); splatActive = false; renderer.domElement.style.display = ''; }
  else renderer.domElement.style.display = 'none';
  try { await views[name](); } catch (e) { console.error(e); idle(); info.innerHTML = `<p class="note">Failed to load: ${e.message}</p>`; }
  const h = new URLSearchParams({ scan: ds.id, view: name }); if (meta.floors.length > 1) h.set('floor', floorIx < 0 ? 'all' : floorIx);
  history.replaceState(null, '', '#' + h.toString());
}
document.querySelectorAll('#tabs button').forEach(b => b.onclick = () => show(b.dataset.view));
await loadDataset(ds); renderPicker();
const legacyHash = location.hash.slice(1);
show(params.get('view') in views ? params.get('view') : (legacyHash in views ? legacyHash : 'lidar'));
