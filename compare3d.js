// Blunt Hall model comparison, 3D section: each method's wall geometry, coloured by whether each wall stands on a
// measured wall, over the storey's LiDAR. One model at a time, or all four side by side with one shared camera.
// Data: data/compare/compare3d.json, m3d_<method>.bin, lidar_3d.bin (scripts/pipeline/compare/c06_web3d.py).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const wrap = $('wrap3d');
const LABEL = { ours: 'Ours · LiDAR block model', cloud2bim: 'Cloud2BIM', scenescript: 'SceneScript · 28 m tiles', 'scenescript-t8-20k': 'SceneScript · 8 m tiles' };

async function init() {
  const M = await (await fetch('data/compare/compare3d.json')).json(), keys = Object.keys(M.methods);
  const renderer = new THREE.WebGLRenderer({ antialias: true }); renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setScissorTest(true); wrap.prepend(renderer.domElement);
  const camera = new THREE.PerspectiveCamera(35, 1, 0.3, 1500);
  const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.maxPolarAngle = Math.PI * 0.495;
  const mats = {
    ok: new THREE.MeshLambertMaterial({ color: 0xffffff, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }),
    bad: new THREE.MeshLambertMaterial({ color: 0xd03b3b, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }),
    edge: new THREE.LineBasicMaterial({ color: 0x999999, transparent: true, opacity: 0.55 }),
    door: new THREE.LineBasicMaterial({ color: 0x2a78d6 }),
    pts: new THREE.PointsMaterial({ color: 0x9a9890, size: 0.05, transparent: true, opacity: 0.45, depthWrite: false }),
  };
  function theme() {
    mats.ok.color.set(css('--wall3')); mats.bad.color.set(css('--bad')); mats.edge.color.set(css('--ref'));
    mats.door.color.set(css('--door')); mats.pts.color.set(css('--ref')); bg = new THREE.Color(css('--plan-bg'));
  }
  let bg; theme();
  // one scene per method, sharing the LiDAR points
  const buf = new Int16Array(await (await fetch('data/compare/lidar_3d.bin')).arrayBuffer());
  const pos = new Float32Array(buf.length); for (let i = 0; i < buf.length; i++) pos[i] = buf[i] / 100 + M.origin[i % 3];
  const pgeo = new THREE.BufferGeometry(); pgeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const scenes = {}, doorsObj = {}, ptsObj = {};
  for (const k of keys) {
    const s = new THREE.Scene(); s.add(new THREE.HemisphereLight(0xffffff, 0x8a8780, 1.7));
    const sun = new THREE.DirectionalLight(0xffffff, 1.2); sun.position.set(-40, 90, 60); s.add(sun);
    const root = new THREE.Group(); root.rotation.x = -Math.PI / 2; s.add(root);          // building frame (z up) -> three.js (y up)
    const n = M.methods[k].triangles, ab = await (await fetch(`data/compare/m3d_${k}.bin`)).arrayBuffer();
    const xyz = new Float32Array(ab, 0, n * 9), fl = new Uint8Array(ab, n * 36, n);
    for (const want of [1, 0]) {
      const sel = []; for (let t = 0; t < n; t++) if (fl[t] === want) sel.push(t);
      if (!sel.length) continue;
      const a = new Float32Array(sel.length * 9); sel.forEach((t, j) => a.set(xyz.subarray(t * 9, t * 9 + 9), j * 9));
      const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(a, 3)); g.computeVertexNormals();
      root.add(new THREE.Mesh(g, want ? mats.ok : mats.bad));
      root.add(new THREE.LineSegments(new THREE.EdgesGeometry(g, 25), mats.edge));
    }
    const dg = new THREE.Group();
    for (const d of M.methods[k].doors) {
      const b = new THREE.BoxGeometry(Math.max(0.1, d[3] - d[0]), Math.max(0.1, d[4] - d[1]), Math.max(0.1, d[5] - d[2]));
      const e = new THREE.LineSegments(new THREE.EdgesGeometry(b), mats.door); e.position.set((d[0] + d[3]) / 2, (d[1] + d[4]) / 2, (d[2] + d[5]) / 2); dg.add(e);
    }
    root.add(dg); doorsObj[k] = dg;
    const pts = new THREE.Points(pgeo, mats.pts); root.add(pts); ptsObj[k] = pts;
    scenes[k] = s;
  }
  // camera: the whole storey from the south-east, as in the route guide
  const box = new THREE.Box3().setFromBufferAttribute(pgeo.getAttribute('position'));
  const c3 = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
  const centre = new THREE.Vector3(c3.x, c3.z, -c3.y);
  function home() {
    const r = Math.hypot(size.x, size.y) * 0.4, az = 0.75, el = 0.85, f = four ? 1.1 : 1;
    const d = r / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * f * (camera.aspect < 1 ? 1.5 / camera.aspect : 1);
    controls.target.copy(centre); camera.position.copy(centre).add(new THREE.Vector3(Math.cos(az) * Math.cos(el), Math.sin(el), Math.sin(az) * Math.cos(el)).multiplyScalar(d));
  }
  let cur = keys[0], four = false;
  const tabs = $('tabs3d');
  tabs.innerHTML = keys.map((k) => `<button class="tab" role="tab" data-m="${k}" aria-selected="${k === cur}">${LABEL[k] || k}<small>${M.methods[k].walls_on_measured} / ${M.methods[k].walls} walls on a measured wall</small></button>`).join('');
  tabs.querySelectorAll('.tab').forEach((b) => (b.onclick = () => { cur = b.dataset.m; four = false; sync(); }));
  $('mode4').onclick = () => { four = !four; sync(); home(); };
  $('reset3d').onclick = home;
  $('showpts').onchange = (e) => keys.forEach((k) => (ptsObj[k].visible = e.target.checked));
  $('showdoors').onchange = (e) => keys.forEach((k) => (doorsObj[k].visible = e.target.checked));
  function sync() {
    tabs.querySelectorAll('.tab').forEach((t) => t.setAttribute('aria-selected', !four && t.dataset.m === cur));
    $('mode4').setAttribute('aria-pressed', four);
    const m = M.methods[cur];
    $('vstat3d').innerHTML = four ? 'All four models · one shared camera' : `<b>${m.walls_on_measured}</b> of <b>${m.walls}</b> walls on a measured wall · <b>${m.doors.length}</b> doors · <b>${m.triangles.toLocaleString()}</b> triangles`;
    layout();
  }
  function cells() {
    const W = wrap.clientWidth, H = wrap.clientHeight;
    if (!four) return [[cur, 0, 0, W, H]];
    const w = W / 2, h = H / 2; return keys.map((k, i) => [k, (i % 2) * w, Math.floor(i / 2) * h, w, h]);
  }
  function layout() {
    const W = wrap.clientWidth, H = wrap.clientHeight; renderer.setSize(W, H, false);
    renderer.domElement.style.width = W + 'px'; renderer.domElement.style.height = H + 'px';
    const cs = cells(); camera.aspect = cs[0][3] / cs[0][4]; camera.updateProjectionMatrix();
    $('lbl3d').innerHTML = four ? cs.map(([k, x, y]) => `<div style="left:${x + 8}px;top:${y + 8}px">${LABEL[k] || k}<small>${M.methods[k].walls_on_measured}/${M.methods[k].walls}</small></div>`).join('')
      + `<div style="left:50%;top:0;bottom:0;width:1px;padding:0;border:0;border-radius:0;background:var(--line)"></div><div style="top:50%;left:0;right:0;height:1px;padding:0;border:0;border-radius:0;background:var(--line)"></div>` : '';
  }
  new ResizeObserver(layout).observe(wrap);
  $('theme')?.addEventListener('click', () => setTimeout(theme, 0));
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', theme);
  sync(); home(); $('load3d').remove();
  (function tick() {
    controls.update(); const H = wrap.clientHeight;
    for (const [k, x, y, w, h] of cells()) {
      renderer.setViewport(x, H - y - h, w, h); renderer.setScissor(x, H - y - h, w, h);
      renderer.setClearColor(bg); renderer.render(scenes[k], camera);
    }
    requestAnimationFrame(tick);
  })();
}

// load only when the section comes into view
new IntersectionObserver((es, ob) => { if (es.some((e) => e.isIntersecting)) { ob.disconnect(); init().catch((e) => { $('load3d').textContent = 'Could not load the 3D models: ' + e.message; console.error(e); }); } }, { rootMargin: '300px' }).observe(wrap);
