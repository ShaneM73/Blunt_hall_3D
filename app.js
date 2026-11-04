import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const $ = (s) => document.querySelector(s);
const viewport = $('#viewport');
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

// ---------- three.js scene (SLAM frame is Z-up; rotate into three's Y-up) ----------
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
viewport.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, 1, 0.05, 500);
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

function resize() {
  const w = viewport.clientWidth, h = viewport.clientHeight;
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
  if (window.__cloudMat) window.__cloudMat.uniforms.vh.value = h * renderer.getPixelRatio();
}
addEventListener('resize', resize); resize();
renderer.setAnimationLoop(() => { controls.update(); if (!splatActive) renderer.render(scene, camera); });

// ---------- loading helpers ----------
async function fetchBuf(url, label) {
  const box = $('#loading'); box.hidden = false; box.querySelector('.msg').textContent = label;
  const fill = box.querySelector('.fill'); fill.style.width = '0';
  const res = await fetch(url);
  if (!res.ok) { box.hidden = true; throw new Error(`${url}: ${res.status}`); }
  const total = +res.headers.get('content-length') || 0;
  const reader = res.body.getReader(); const chunks = []; let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break;
    chunks.push(value); got += value.length; if (total) fill.style.width = `${(100 * got / total).toFixed(0)}%`; }
  box.hidden = true;
  const out = new Uint8Array(got); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; }
  return out.buffer;
}
const meta = await (await fetch('data/meta.json')).json();

// ---------- point cloud ----------
// points.bin: 'BHPC' | u32 N | u32 flags(1=rgb) | f32 origin[3] | f32 scale | u16 xyz[N*3] | u8 intensity[N] | u8 rgb[N*3]
let cloud = null;
async function loadCloud() {
  if (cloud) return cloud;
  const buf = await fetchBuf('data/points.bin', 'Loading point cloud…');
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
  g.computeBoundingBox();
  const mat = new THREE.ShaderMaterial({
    uniforms: { mode: { value: 0 }, size: { value: 1.6 }, vh: { value: 800 }, spacing: { value: (meta.voxel_cm || 2) / 100 }, zmin: { value: meta.floor_z }, zmax: { value: meta.ceiling_z },
                clipTop: { value: 99 }, gain: { value: 1.5 }, boost: { value: 1.3 } },
    vertexShader: `
      attribute float intensity; attribute vec3 rgb;
      uniform int mode; uniform float size, zmin, zmax, gain, clipTop, boost, vh, spacing;
      varying vec3 vCol; varying float vClip;
      vec3 turbo(float x){ x=clamp(x,0.,1.);
        vec4 kr=vec4(.13572138,4.61539260,-42.66032258,132.13108234), kg=vec4(.09140261,2.19418839,4.84296658,-14.18503333), kb=vec4(.10667330,12.64194608,-60.58204836,110.36276771);
        vec2 kr2=vec2(-152.94239396,59.28637943), kg2=vec2(4.27729857,2.82956604), kb2=vec2(-89.90310912,27.34824973);
        vec4 v4=vec4(1.,x,x*x,x*x*x); vec2 v2=v4.zw*v4.z;
        return vec3(dot(v4,kr)+dot(v2,kr2), dot(v4,kg)+dot(v2,kg2), dot(v4,kb)+dot(v2,kb2)); }
      void main(){
        vClip = position.z > clipTop ? 1. : 0.;
        if (mode==0) { float v=clamp(intensity*gain,0.,1.); vCol=vec3(v); }
        else if (mode==1) { vec3 c=(rgb-0.5)*(1.0+0.6*(boost-1.0))+0.5; float l=dot(c,vec3(.299,.587,.114)); vCol=clamp(mix(vec3(l),c,boost*boost),0.,1.); }
        else vCol=turbo((position.z-zmin)/(zmax-zmin));
        vec4 mv=modelViewMatrix*vec4(position,1.);
        gl_PointSize = clamp(size * spacing * projectionMatrix[1][1] * vh * 0.5 / max(-mv.z, 0.05), 1.0, 14.0);
        gl_Position = projectionMatrix*mv; }`,
    fragmentShader: `varying vec3 vCol; varying float vClip;
      void main(){ if (vClip>0.5) discard; vec2 c=gl_PointCoord-.5; if(dot(c,c)>.25) discard; gl_FragColor=vec4(vCol,1.); }`,
  });
  cloud = { points: new THREE.Points(g, mat), N, hasRGB: !!rgb };
  mat.uniforms.vh.value = viewport.clientHeight * renderer.getPixelRatio(); window.__cloudMat = mat;
  return cloud;
}

// ---------- trajectory & camera stations ----------
let trajObj = null;
async function loadTraj() {
  if (trajObj) return trajObj;
  const tj = await (await fetch('data/trajectory.json')).json();
  const grp = new THREE.Group();
  const lg = new THREE.BufferGeometry().setFromPoints(tj.p.map(p => new THREE.Vector3(...p)));
  grp.add(new THREE.Line(lg, new THREE.LineBasicMaterial({ color: 0xd62728 })));
  const sg = new THREE.SphereGeometry(0.06, 12, 8);
  const s0 = new THREE.Mesh(sg, new THREE.MeshBasicMaterial({ color: 0x2ca02c })); s0.position.set(...tj.p[0]); grp.add(s0);
  const s1 = new THREE.Mesh(sg, new THREE.MeshBasicMaterial({ color: 0x1f77b4 })); s1.position.set(...tj.p.at(-1)); grp.add(s1);
  if (tj.cams) {
    const cg = new THREE.BufferGeometry().setFromPoints(tj.cams.map(p => new THREE.Vector3(...p)));
    grp.add(new THREE.Points(cg, new THREE.PointsMaterial({ color: 0xff9f1c, size: 0.06 })));
  }
  trajObj = { group: grp, data: tj };
  return trajObj;
}

// ---------- mesh model ----------
let modelObj = null;
async function loadModel() {
  if (modelObj) return modelObj;
  const buf = await fetchBuf(meta.model.file, 'Loading 3D model…');
  const geom = new PLYLoader().parse(buf);
  geom.computeVertexNormals();
  const hasColor = !!geom.getAttribute('color');
  const mat = new THREE.MeshStandardMaterial({ vertexColors: hasColor, color: hasColor ? 0xffffff : 0xcfcac0,
    roughness: 0.9, metalness: 0, side: THREE.DoubleSide });
  const unlit = new THREE.MeshBasicMaterial({ vertexColors: hasColor, side: THREE.DoubleSide });
  modelObj = { mesh: new THREE.Mesh(geom, hasColor ? unlit : mat), lit: mat, unlit, tris: geom.index ? geom.index.count / 3 : geom.attributes.position.count / 3, hasColor };
  return modelObj;
}

// ---------- gaussian splat ----------
let splatViewer = null;
async function showSplat() {
  if (!meta.splat) return false;
  const GS = await import('@mkkellogg/gaussian-splats-3d');
  if (!splatViewer) {
    splatViewer = new GS.Viewer({ rootElement: viewport, selfDrivenMode: true, useBuiltInControls: true,
      cameraUp: [0, 0, 1], initialCameraPosition: meta.splat.cam_pos, initialCameraLookAt: meta.splat.cam_look,
      sharedMemoryForWorkers: false, gpuAcceleratedSort: false });
    $('#loading').hidden = false; $('#loading .msg').textContent = 'Loading Gaussian splat…';
    await splatViewer.addSplatScene(meta.splat.file, { showLoadingUI: false, progressiveLoad: true,
      onProgress: (p) => { $('#loading .fill').style.width = `${p}%`; } });
    $('#loading').hidden = true;
  }
  splatViewer.renderer.domElement.style.display = '';
  splatViewer.start?.();
  return true;
}
function hideSplat() { if (splatViewer) { splatViewer.renderer.domElement.style.display = 'none'; splatViewer.stop?.(); } }

// ---------- SLAM 2D map ----------
async function drawSlam() {
  const m = meta.slam_map; const img = new Image(); img.src = m.image; await img.decode();
  const tj = (await loadTraj()).data;
  const cv = $('#slamcanvas'); const k = Math.min(1, 1600 / img.width);
  cv.width = img.width * k; cv.height = img.height * k;
  const ctx = cv.getContext('2d'); ctx.drawImage(img, 0, 0, cv.width, cv.height);
  const px = (x, y) => [ (x - m.xmin) / m.res * k, (m.ymax - y) / m.res * k ];
  ctx.lineWidth = 2; ctx.strokeStyle = '#d62728'; ctx.beginPath();
  tj.p.forEach((p, i) => { const [u, v] = px(p[0], p[1]); i ? ctx.lineTo(u, v) : ctx.moveTo(u, v); }); ctx.stroke();
  const dot = (p, c) => { const [u, v] = px(p[0], p[1]); ctx.fillStyle = c; ctx.beginPath(); ctx.arc(u, v, 6, 0, 7); ctx.fill(); };
  dot(tj.p[0], '#2ca02c'); dot(tj.p.at(-1), '#1f77b4');
  // scale bar 5 m
  const L = 5 / m.res * k; ctx.fillStyle = '#1d1f22'; ctx.fillRect(20, cv.height - 30, L, 6);
  ctx.font = '13px system-ui'; ctx.fillText('5 m', 20, cv.height - 36);
}

// ---------- UI ----------
const info = $('#info'), ctl = $('#controls'), stats = $('#stats');
const fmt = (n) => n.toLocaleString();
function statRows(rows) { stats.innerHTML = rows.map(([a, b]) => `<div><span>${a}</span><span>${b}</span></div>`).join(''); }
let framed = false;
function frame(obj) {
  if (framed) return; framed = true;
  const c = new THREE.Vector3(...meta.focus); controls.target.copy(c);
  camera.position.set(c.x - 3.5, c.y - 7, c.z + 6); controls.update();
}
function clearWorld() { world.clear(); }

const CUT = +(meta.ceiling_z - 0.15).toFixed(2);   // default: ceiling removed so the interior is visible
const views = {
  async lidar() {
    const c = await loadCloud(); const t = await loadTraj();
    clearWorld(); world.add(c.points); world.add(t.group); frame();
    c.points.material.uniforms.mode.value = 0;
    info.innerHTML = `<h2>LiDAR points</h2><p>${fmt(meta.n_raw)} raw returns from the Hesai 16-beam scanner, registered with the on-device SLAM trajectory and thinned to a ${meta.voxel_cm} cm voxel grid.</p>`;
    ctl.innerHTML = `
      <label>Color <select id="cmode"><option value="0">Intensity</option><option value="2">Height</option></select></label>
      <label>Point size <input id="psize" type="range" min="0.6" max="4" step="0.1" value="1.6"></label>
      <label>Intensity gain <input id="gain" type="range" min="0.5" max="4" step="0.1" value="1.5"></label>
      <label>Cut ceiling <input id="clip" type="range" min="${meta.floor_z+0.5}" max="${meta.ceiling_z+3}" step="0.05" value="${CUT}"></label>
      <label>Trajectory <input id="trj" type="checkbox" checked></label>
      <div id="leg"></div>`;
    bindCloudControls(c, t);
    statRows([['Points shown', fmt(c.N)], ['Scan frames', fmt(meta.n_frames)], ['Duration', `${meta.duration_s} s`], ['Path length', `${meta.path_m} m`]]);
  },
  async rgb() {
    const c = await loadCloud(); const t = await loadTraj();
    clearWorld(); world.add(c.points); world.add(t.group); frame();
    c.points.material.uniforms.mode.value = c.hasRGB ? 1 : 0;
    info.innerHTML = `<h2>Colored points</h2><p>Each point takes its color from the three fisheye cameras (${fmt(meta.n_images)} photos), as a weighted blend of every photo that sees it unoccluded (closer and more central views count more).</p>
      ${c.hasRGB ? '' : '<p class="note">RGB not available in this build — showing intensity.</p>'}
      <p class="note">${meta.rgb_note || ''}</p>`;
    ctl.innerHTML = `
      <label>Point size <input id="psize" type="range" min="0.6" max="4" step="0.1" value="1.6"></label>
      <label>Color boost <input id="boost" type="range" min="1" max="2" step="0.05" value="1.3"></label>
      <label>Cut ceiling <input id="clip" type="range" min="${meta.floor_z+0.5}" max="${meta.ceiling_z+3}" step="0.05" value="${CUT}"></label>
      <label>Trajectory <input id="trj" type="checkbox"></label>
      <p class="note">For photo-realistic texture, open the Gaussian splat tab — points 1.5 cm apart cannot carry fine photo detail.</p>`;
    bindCloudControls(c, t); t.group.visible = false;
    statRows([['Points shown', fmt(c.N)], ['Colored', meta.rgb_frac ? `${(100*meta.rgb_frac).toFixed(1)} %` : '—'], ['Cameras', '3 × 5 MP fisheye']]);
  },
  async model() {
    const m = await loadModel(); const t = await loadTraj();
    clearWorld(); world.add(m.mesh); world.add(t.group); t.group.visible = false; frame();
    info.innerHTML = `<h2>3D model</h2><p>${meta.model.method}</p><p class="note">Covers the well-scanned zone only; areas seen just through doorways were left out rather than shown as fragments.</p>`;
    ctl.innerHTML = `
      <label>Shading <select id="shade">${m.hasColor ? '<option value="p">Photo color (unlit)</option><option value="c">Photo color + shading</option>' : ''}<option value="g">Gray (shaded)</option><option value="w">Wireframe</option></select></label>
      <label>Cut ceiling <input id="mclip" type="range" min="${meta.floor_z+0.5}" max="${meta.ceiling_z+3}" step="0.05" value="${CUT}"></label>
      <label>Trajectory <input id="trj" type="checkbox"></label>`;
    renderer.localClippingEnabled = true;
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, -1), CUT);
    m.lit.clippingPlanes = [plane]; m.unlit.clippingPlanes = [plane];
    $('#mclip').oninput = (e) => { plane.constant = +e.target.value; };
    const setShade = (v) => { if (v === 'p') { m.mesh.material = m.unlit; return; }
      const mat = m.lit; m.mesh.material = mat; mat.wireframe = v === 'w'; mat.vertexColors = v === 'c' && m.hasColor;
      mat.color.set(mat.vertexColors ? 0xffffff : 0xcfcac0); mat.needsUpdate = true; };
    $('#shade').onchange = (e) => setShade(e.target.value); setShade(m.hasColor ? 'p' : 'g');
    $('#trj').onchange = (e) => { t.group.visible = e.target.checked; };
    statRows([['Triangles', fmt(m.tris)], ['Method', meta.model.short]]);
  },
  async slam() {
    await drawSlam(); const t = (await loadTraj()).data;
    info.innerHTML = `<h2>SLAM map</h2><p>Top-down occupancy map: wall hits in a horizontal slice ${meta.slam_map.slice}, with the SLAM-optimized walking path (green = start, blue = end).</p>`;
    ctl.innerHTML = `<p>${meta.slam_map.note}</p>`;
    statRows([['Poses', fmt(meta.n_frames)], ['Loop closures', meta.loop_closures], ['Grid', `${meta.slam_map.grid_cm} cm`], ['Path length', `${meta.path_m} m`]]);
  },
  async splat() {
    clearWorld();
    const ok = await showSplat().catch((e) => { console.error(e); return false; });
    info.innerHTML = ok ? `<h2>Gaussian splat</h2><p>${meta.splat.note}</p>` :
      `<h2>Gaussian splat</h2><p class="note">The Gaussian Splatting model is not built yet.</p>`;
    ctl.innerHTML = ''; statRows(ok ? [['Gaussians', fmt(meta.splat.n)], ['Training views', fmt(meta.splat.views)]] : []);
    splatActive = ok;
  },
};

function bindCloudControls(c, t) {
  const u = c.points.material.uniforms;
  $('#psize').oninput = (e) => { u.size.value = +e.target.value; };
  $('#clip').oninput = (e) => { u.clipTop.value = +e.target.value; };
  $('#gain') && ($('#gain').oninput = (e) => { u.gain.value = +e.target.value; });
  $('#boost') && ($('#boost').oninput = (e) => { u.boost.value = +e.target.value; });
  u.clipTop.value = +$('#clip').value; u.size.value = +$('#psize').value;
  $('#trj').onchange = (e) => { t.group.visible = e.target.checked; };
  t.group.visible = $('#trj').checked;
  const cm = $('#cmode');
  const legend = () => { const leg = $('#leg'); if (!leg) return;
    leg.innerHTML = u.mode.value === 2 ? `<div class="legend" style="background:linear-gradient(90deg,#30123b,#4686fb,#1ae4b6,#a2fc3c,#fabb39,#e4460a,#7a0403)"></div>
      <div class="legend-labels"><span>${meta.floor_z.toFixed(1)} m</span><span>height</span><span>${meta.ceiling_z.toFixed(1)} m</span></div>` : ''; };
  cm && (cm.onchange = (e) => { u.mode.value = +e.target.value; legend(); });
  legend();
}

// double-click to refocus orbit
const ray = new THREE.Raycaster(); ray.params.Points.threshold = 0.05;
renderer.domElement.addEventListener('dblclick', (e) => {
  const r = renderer.domElement.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
  const hit = ray.intersectObjects(world.children, false)[0]; if (hit) controls.target.copy(hit.point);
});

async function show(name) {
  document.querySelectorAll('#tabs button').forEach(b => b.classList.toggle('active', b.dataset.view === name));
  const is2d = name === 'slam';
  $('#slam2d').hidden = !is2d; viewport.style.visibility = is2d ? 'hidden' : 'visible';
  $('#help').style.display = is2d ? 'none' : '';
  if (name !== 'splat') { hideSplat(); splatActive = false; renderer.domElement.style.display = ''; }
  else renderer.domElement.style.display = 'none';
  try { await views[name](); } catch (e) { console.error(e); info.innerHTML = `<p class="note">Failed to load: ${e.message}</p>`; }
  history.replaceState(null, '', '#' + name);
}
document.querySelectorAll('#tabs button').forEach(b => b.onclick = () => show(b.dataset.view));
show((location.hash.slice(1) in views) ? location.hash.slice(1) : 'lidar');
