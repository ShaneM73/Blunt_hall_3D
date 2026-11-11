// Blunt Hall model comparison, section 05: the LingBot-Map video mesh (aligned to the building frame) over the LiDAR
// wall points, coloured as filmed or by its distance to the LiDAR, with the camera path.
// Data: data/compare/video_mesh.bin, video.json (scripts/pipeline/compare/c09_video_web.py), lidar_3d.bin, compare3d.json.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const wrap = $('vwrap');
const ECAP = 50;                                              // error colour ramp: 0 cm -> ECAP cm and beyond

async function init() {
  const [M, V, mb, lb] = await Promise.all([
    fetch('data/compare/compare3d.json').then((r) => r.json()), fetch('data/compare/video.json').then((r) => r.json()),
    fetch('data/compare/video_mesh.bin').then((r) => r.arrayBuffer()), fetch('data/compare/lidar_3d.bin').then((r) => r.arrayBuffer())]);
  const renderer = new THREE.WebGLRenderer({ antialias: true }); renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  wrap.prepend(renderer.domElement);
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(40, 1, 0.2, 800);
  const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.maxPolarAngle = Math.PI * 0.495;
  const root = new THREE.Group(); root.rotation.x = -Math.PI / 2; scene.add(root);           // building frame (z up) -> three.js (y up)
  // mesh: [nv, nf] uint32, int16 xyz cm, uint8 rgb, uint8 error cm, pad to 4, uint32 indices
  const [nv, nf] = new Uint32Array(mb, 0, 2), q = new Int16Array(mb, 8, nv * 3), o = M.origin;
  const pos = new Float32Array(nv * 3); for (let i = 0; i < nv * 3; i++) pos[i] = q[i] / 100 + o[i % 3];
  const rgb = new Uint8Array(mb, 8 + nv * 6, nv * 3), err = new Uint8Array(mb, 8 + nv * 9, nv);
  const ioff = 8 + nv * 10 + ((4 - ((nv * 10) % 4)) % 4), idx = new Uint32Array(mb, ioff, nf * 3);
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setIndex(new THREE.BufferAttribute(idx, 1));
  const col = new Float32Array(nv * 3); g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const mat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }); root.add(new THREE.Mesh(g, mat));
  // LiDAR wall points + camera path
  const b = new Int16Array(lb), lp = new Float32Array(b.length); for (let i = 0; i < b.length; i++) lp[i] = b[i] / 100 + o[i % 3];
  const pg = new THREE.BufferGeometry(); pg.setAttribute('position', new THREE.BufferAttribute(lp, 3));
  const pmat = new THREE.PointsMaterial({ size: 0.05, transparent: true, opacity: 0.4, depthWrite: false }); const pts = new THREE.Points(pg, pmat); root.add(pts);
  const z = (M.floor_z ?? 7.3) + 1.55, cp = new Float32Array(V.path.length * 3); V.path.forEach(([x, y], i) => cp.set([x, y, z], i * 3));
  const cg = new THREE.BufferGeometry(); cg.setAttribute('position', new THREE.BufferAttribute(cp, 3));
  const lmat = new THREE.LineBasicMaterial(); const path = new THREE.Line(cg, lmat); root.add(path);
  let mode = 'rgb', bg;
  function paint() {
    if (mode === 'rgb') for (let i = 0; i < nv * 3; i++) col[i] = rgb[i] / 255;
    else {
      const a = new THREE.Color(css('--wall3')), c = new THREE.Color(css('--bad')), t = new THREE.Color();
      for (let i = 0; i < nv; i++) { t.copy(a).lerp(c, Math.min(1, err[i] / ECAP)); col.set([t.r, t.g, t.b], i * 3); }
    }
    g.attributes.color.needsUpdate = true;
  }
  function theme() { pmat.color.set(css('--ref')); lmat.color.set(css('--door')); bg = new THREE.Color(css('--plan-bg')); if (mode === 'err') paint(); }
  theme(); paint();
  // camera: the walked part of the floor from the south-east
  g.computeBoundingBox(); const bb = g.boundingBox, c3 = bb.getCenter(new THREE.Vector3()), size = bb.getSize(new THREE.Vector3());
  const centre = new THREE.Vector3(c3.x, c3.z, -c3.y);
  function home() {
    const r = Math.hypot(size.x, size.y) * 0.42, az = 0.75, el = 0.8;
    const d = r / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * (camera.aspect < 1 ? 1.4 / camera.aspect : 1);
    controls.target.copy(centre); camera.position.copy(centre).add(new THREE.Vector3(Math.cos(az) * Math.cos(el), Math.sin(el), Math.sin(az) * Math.cos(el)).multiplyScalar(d));
  }
  const tabs = $('vtabs').querySelectorAll('.tab');
  tabs.forEach((t) => (t.onclick = () => {
    mode = t.dataset.c; tabs.forEach((x) => x.setAttribute('aria-selected', x === t)); paint();
    $('vkeyrgb').hidden = mode !== 'rgb'; $('vkeyerr').hidden = mode !== 'err';
  }));
  $('vramp').textContent = `0 cm … ≥ ${ECAP} cm from the LiDAR`;
  $('vshowpts').onchange = (e) => (pts.visible = e.target.checked);
  $('vshowpath').onchange = (e) => (path.visible = e.target.checked);
  $('vreset').onclick = home;
  $('vstat3dv').innerHTML = `<b>${nf.toLocaleString()}</b> triangles (of ${V.mesh.triangles.toLocaleString()}) · vertex error median <b>${Math.round(V.vertex_error_cm.median)} cm</b>, 90 % within <b>${Math.round(V.vertex_error_cm.p90)} cm</b>`;
  function layout() {
    const W = wrap.clientWidth, H = wrap.clientHeight; renderer.setSize(W, H, false);
    renderer.domElement.style.width = W + 'px'; renderer.domElement.style.height = H + 'px'; camera.aspect = W / H; camera.updateProjectionMatrix();
  }
  new ResizeObserver(layout).observe(wrap); layout(); home();
  $('theme')?.addEventListener('click', () => setTimeout(theme, 0));
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', theme);
  $('vload').remove();
  (function tick() { controls.update(); renderer.setClearColor(bg); renderer.render(scene, camera); requestAnimationFrame(tick); })();
}

new IntersectionObserver((es, ob) => { if (es.some((e) => e.isIntersecting)) { ob.disconnect(); init().catch((e) => { $('vload').textContent = 'Could not load the video mesh: ' + e.message; console.error(e); }); } }, { rootMargin: '300px' }).observe(wrap);
