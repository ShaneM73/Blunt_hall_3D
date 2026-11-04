"""Stage 13: photo textures for the structural model (s12) + glTF export per floor.
- every face rectangle (split to <= 20 m) gets a texel grid at RES m; texel colors come from the fisheye photos:
  visibility by a z-buffer of the clean LiDAR cloud, person masks skipped, per-photo gains from s06, weights as s06;
  pass 1 = weighted mean (reference), pass 2 = best-view weights (p = 3) with outlier rejection against pass 1
- texels never seen take the color of the nearest colored LiDAR point
- faces are packed (shelf packing) into ATLAS x ATLAS JPEG pages per floor; the stairwell keeps the measured mesh
Out: data/<id>/model_f<j>.glb (+ textures embedded), mesh/structure_tex/*.jpg     Usage: s13_texture.py <id>"""
import os, sys, json, io
import numpy as np, torch, cv2
from scipy.spatial import cKDTree
from common import W, REPO, CAL, Rcb, Poser, traj, image_index, read_image, log, removed_mask

DID = sys.argv[1]; dev = 'cuda:' + os.environ.get('BH_GPU', '1')
RES = float(os.environ.get('BH_TEX_RES', 0.02)); ATLAS = 4096; MAXLEN = 20.0
S = json.load(open(W('mesh', 'structure.json'))); faces = []
for f in S['faces']:                                   # split long faces so they fit an atlas row comfortably
    nu = max(1, int(np.ceil(f['su'] / MAXLEN))); nv = max(1, int(np.ceil(f['sv'] / MAXLEN)))
    for a in range(nu):
        for b in range(nv):
            g = dict(f); o = np.array(f['o']) + np.array(f['u']) * f['su'] * a / nu + np.array(f['v']) * f['sv'] * b / nv
            g['o'] = o.tolist(); g['su'] = f['su'] / nu; g['sv'] = f['sv'] / nv; faces.append(g)
# texel grid
dims = [(max(1, int(round(f['su'] / RES))), max(1, int(round(f['sv'] / RES)))) for f in faces]
offs = np.cumsum([0] + [w * h for w, h in dims]); NT = int(offs[-1]); log('faces', len(faces), 'texels %.1f M' % (NT / 1e6))
X = np.empty((NT, 3), np.float32); Nn = np.empty((NT, 3), np.float32)
for i, (f, (w, h)) in enumerate(zip(faces, dims)):
    a, b = np.meshgrid((np.arange(w) + 0.5) / w * f['su'], (np.arange(h) + 0.5) / h * f['sv'], indexing='xy')
    p = np.array(f['o']) + a.reshape(-1, 1) * np.array(f['u']) + b.reshape(-1, 1) * np.array(f['v'])
    X[offs[i]:offs[i + 1]] = p; Nn[offs[i]:offs[i + 1]] = f['n']
# occluders: clean LiDAR cloud
d = np.load(W('color', 'hq_colored.npz')); k = d['has'] & ~removed_mask(); LP = d['xyz'][k]; LC = d['rgb'][k]
def blocks(Q, BS=3.0):
    key = np.floor(Q / BS).astype(np.int64); kmin = key.min(0); key -= kmin; dm = key.max(0) + 1
    lin = (key[:, 0] * dm[1] + key[:, 1]) * dm[2] + key[:, 2]; order = np.argsort(lin, kind='stable')
    ub, st = np.unique(lin[order], return_index=True); en = np.r_[st[1:], len(Q)]
    return np.stack(np.unravel_index(ub, dm), 1) * BS + kmin * BS + BS / 2, st, en, torch.tensor(order, device=dev)
Xb = blocks(X); Lb = blocks(LP)
Xg = torch.tensor(X, device=dev, dtype=torch.float64); Ng = torch.tensor(Nn, device=dev, dtype=torch.float64); Lg = torch.tensor(LP, device=dev, dtype=torch.float64)
def near(bk, c, r):
    bc, st, en, og = bk; nb = np.flatnonzero(np.linalg.norm(bc - c, axis=1) < r + 2.6)
    return og[torch.tensor(np.concatenate([np.arange(st[i], en[i]) for i in nb]), device=dev)] if len(nb) else None
pose = Poser(traj()); recs = image_index(); gains = np.load(W('calib', 'gains.npy'))
GM = json.load(open(W('images', 'cameras.json'))); KN = np.array(GM['K']); MW, MH = GM['W'], GM['H']
SS = 4; H4, W4 = 2592 // SS, 1944 // SS
acc = torch.zeros(NT, 3, device=dev); ws = torch.zeros(NT, device=dev); ref = None
from concurrent.futures import ThreadPoolExecutor
def load(r):
    im = read_image(r); mk = cv2.imread(W('images', 'masks', f"c{r['cam']}_{r['t']:.6f}.png"), 0)
    return cv2.cvtColor(cv2.resize(im, (W4, H4), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB), mk
for pas in (1, 2):
    acc.zero_(); ws.zero_(); pool = ThreadPoolExecutor(8); futs = [pool.submit(load, r) for r in recs[:12]]
    for kk, r in enumerate(recs):
        im, mk = futs[kk].result(); futs[kk] = None
        if kk + 12 < len(recs): futs.append(pool.submit(load, recs[kk + 12]))
        cam, t = r['cam'], r['t']; Rwb, twb = pose(t); cpos = twb + Rwb @ np.array(CAL[cam]['t'])
        sx = near(Xb, cpos, 6.0)
        if sx is None: continue
        R_cb = torch.tensor(Rcb(cam), device=dev); tb = torch.tensor(CAL[cam]['t'], device=dev, dtype=torch.float64)
        Rw = torch.tensor(Rwb, device=dev); tw = torch.tensor(twb, device=dev); fx, fy, cx, cy = CAL[cam]['K']; kd = CAL[cam]['D']
        def proj(Q):
            Pc = ((Q - tw) @ Rw - tb) @ R_cb.T; x, y, z = Pc.unbind(1); rr = torch.hypot(x, y) + 1e-9; th = torch.atan2(rr, z); t2 = th * th
            thd = th * (1 + kd[0] * t2 + kd[1] * t2 ** 2 + kd[2] * t2 ** 3 + kd[3] * t2 ** 4); return Pc, fx * thd * x / rr + cx, fy * thd * y / rr + cy, th
        sl = near(Lb, cpos, 6.0)
        zb = torch.full((H4 * W4,), 1e9, device=dev, dtype=torch.float64)
        if sl is not None:
            Qc, qu, qv, qt = proj(Lg[sl]); qn = Qc.norm(dim=1); qo = (qt < 1.25) & (qu >= 0) & (qu < 1944) & (qv >= 0) & (qv < 2592) & (qn > 0.3)
            zb.scatter_reduce_(0, (qv[qo] / SS).long() * W4 + (qu[qo] / SS).long(), qn[qo], 'amin')
        Pc, u, v, th = proj(Xg[sx]); dd = Pc.norm(dim=1)
        vd = torch.tensor(cpos, device=dev) - Xg[sx]; vd = vd / vd.norm(dim=1, keepdim=True); cosi = (vd * Ng[sx]).sum(1)
        ok = (th < 1.2) & (u >= 0) & (u < 1944) & (v >= 0) & (v < 2592) & (dd > 0.3) & (dd < 6) & (cosi > 0.2)
        li = ok.nonzero().squeeze(1)
        if not len(li): continue
        pix = (v[li] / SS).long() * W4 + (u[li] / SS).long()
        li = li[dd[li] <= zb[pix] * 1.03 + 0.06]                      # texel not behind a measured surface
        if mk is not None and len(li):
            x, y, z = Pc[li].unbind(1); zz = z.clamp(min=1e-3); pu = KN[0, 0] * x / zz + KN[0, 2]; pv = KN[1, 1] * y / zz + KN[1, 2]
            inside = (pu >= 0) & (pu < MW) & (pv >= 0) & (pv < MH) & (z > 0.05); M = torch.tensor(mk > 127, device=dev)
            good = ~inside; ii = inside.nonzero().squeeze(1); good[ii] = M[pv[ii].long(), pu[ii].long()]; li = li[good]
        if not len(li): continue
        img = torch.tensor(im, device=dev, dtype=torch.float32)
        uu = (u[li] / SS).clamp(0, W4 - 1.001); vv = (v[li] / SS).clamp(0, H4 - 1.001); u0 = uu.floor().long(); v0 = vv.floor().long(); au = (uu - u0)[:, None].float(); av = (vv - v0)[:, None].float()
        col = (img[v0, u0] * (1 - au) * (1 - av) + img[v0, u0 + 1] * au * (1 - av) + img[v0 + 1, u0] * (1 - au) * av + img[v0 + 1, u0 + 1] * au * av)
        col = col * torch.tensor(gains[kk], device=dev, dtype=torch.float32)
        w = (cosi[li] ** 1.5 / (dd[li] ** 2 + 0.05) * torch.cos(th[li]) ** 2).float()
        gi = sx[li]
        if pas == 2: w = w ** 3 * torch.exp(-((col - ref[gi]) ** 2).sum(1) / (2 * 25.0 ** 2))
        acc.index_add_(0, gi, col * w[:, None]); ws.index_add_(0, gi, w)
        if kk % 1000 == 0: log('pass', pas, 'image', kk, 'of', len(recs))
    pool.shutdown()
    ref = acc / ws.clamp(min=1e-12)[:, None]
seen = (ws > 0).cpu().numpy(); rgb = ref.clamp(0, 255).byte().cpu().numpy()
log('texels seen in photos %.1f%%' % (seen.mean() * 100))
if (~seen).any():
    _, j = cKDTree(LP).query(X[~seen], workers=64); rgb[~seen] = LC[j]
# atlas packing per floor + glTF
import trimesh
ZUP_TO_YUP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)   # glTF is Y-up: (x, y, z) -> (x, z, -y)
from PIL import Image
OUT = os.path.join(REPO, 'data', DID); os.makedirs(W('mesh', 'structure_tex', 'x'), exist_ok=True)
import time
while os.path.getmtime(W('mesh', 'tiles.json')) < os.path.getmtime(W('clouds', 'clutter_rm.npy')):   # wait for s08 on the clean cloud
    log('waiting for the clean mesh tiles (s08)'); time.sleep(60)
floors = sorted({f['floor'] for f in faces}); summary = []
for fl in floors:
    idx = [i for i, f in enumerate(faces) if f['floor'] == fl]
    idx.sort(key=lambda i: -dims[i][1]); pages = []; place = {}
    x = y = rowh = 0; pg = 0; PAD = 2
    for i in idx:
        w, h = dims[i]
        if x + w + PAD > ATLAS: x = 0; y += rowh + PAD; rowh = 0
        if y + h + PAD > ATLAS: pg += 1; x = y = rowh = 0
        place[i] = (pg, x, y); x += w + PAD; rowh = max(rowh, h)
    npg = pg + 1; atl = [np.zeros((ATLAS, ATLAS, 3), np.uint8) for _ in range(npg)]
    for i in idx:
        pg, x, y = place[i]; w, h = dims[i]; tex = rgb[offs[i]:offs[i + 1]].reshape(h, w, 3)
        atl[pg][y:y + h, x:x + w] = tex
        atl[pg][y:y + h, max(x - 1, 0)] = tex[:, 0]; atl[pg][y:y + h, min(x + w, ATLAS - 1)] = tex[:, -1]   # bleed against seams
        atl[pg][max(y - 1, 0), x:x + w] = tex[0]; atl[pg][min(y + h, ATLAS - 1), x:x + w] = tex[-1]
    # crop the last page to the used height
    used = max(place[i][2] + dims[i][1] for i in idx if place[i][0] == npg - 1) + PAD
    hs = [ATLAS] * (npg - 1) + [int(2 ** np.ceil(np.log2(max(used, 64))))]
    scene = trimesh.Scene()
    for pg in range(npg):
        im = Image.fromarray(atl[pg][:hs[pg]]); im.save(W('mesh', 'structure_tex', f'f{fl}_p{pg}.jpg'), quality=88)
        V, F, UV = [], [], []
        for i in idx:
            if place[i][0] != pg: continue
            f = faces[i]; w, h = dims[i]; _, x, y = place[i]; o = np.array(f['o']); a = np.array(f['u']) * f['su']; b = np.array(f['v']) * f['sv']
            base = len(V); V += [o, o + a, o + a + b, o + b]
            u0, u1 = x / ATLAS, (x + w) / ATLAS; v0, v1 = 1 - y / hs[pg], 1 - (y + h) / hs[pg]
            UV += [[u0, v0], [u1, v0], [u1, v1], [u0, v1]]
            n = np.array(f['n']); tri = [[0, 1, 2], [0, 2, 3]]
            if np.dot(np.cross(a, b), n) < 0: tri = [[0, 2, 1], [0, 3, 2]]
            F += [[base + t[0], base + t[1], base + t[2]] for t in tri]
        mat = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.open(W('mesh', 'structure_tex', f'f{fl}_p{pg}.jpg')), metallicFactor=0.0, roughnessFactor=1.0, doubleSided=True)
        m = trimesh.Trimesh(np.array(V), np.array(F), visual=trimesh.visual.TextureVisuals(uv=np.array(UV), material=mat), process=False)
        scene.add_geometry(m, node_name=f'structure_p{pg}', transform=ZUP_TO_YUP)
    # stairwell / measured parts: crop the full-resolution clean mesh tiles to stair boxes within this floor band
    fi = S['floors'][fl]; zlo, zhi = fi['zmin'], fi['zmax']
    for b in S.get('stairs', []):
        Vs, Fs, Cs = [], [], []; base = 0
        for t in json.load(open(W('mesh', 'tiles.json')))['tiles']:
            M = np.load(W('mesh', 'tiles', t['file'])); Vt, Ft, Ct = M['v'], M['f'], M['c']; cen = Vt[Ft].mean(1)
            sel = (cen[:, 0] >= b[0]) & (cen[:, 0] < b[1]) & (cen[:, 1] >= b[2]) & (cen[:, 1] < b[3]) & (cen[:, 2] >= zlo) & (cen[:, 2] < zhi)
            if sel.sum() == 0: continue
            Fu = Ft[sel]; uq, inv = np.unique(Fu, return_inverse=True); Vs.append(Vt[uq]); Cs.append(Ct[uq]); Fs.append(inv.reshape(-1, 3) + base); base += len(uq)
        if Vs:
            import open3d as o3d
            om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.vstack(Vs).astype(np.float64)), o3d.utility.Vector3iVector(np.vstack(Fs)))
            om.vertex_colors = o3d.utility.Vector3dVector(np.vstack(Cs) / 255.)
            if len(om.triangles) > 400_000: om = om.simplify_quadric_decimation(400_000)
            om.remove_degenerate_triangles(); om.remove_unreferenced_vertices()
            sm = trimesh.Trimesh(np.asarray(om.vertices), np.asarray(om.triangles), vertex_colors=np.c_[(np.asarray(om.vertex_colors) * 255).round().astype(np.uint8), np.full(len(om.vertices), 255, np.uint8)], process=False)
            scene.add_geometry(sm, node_name='stairwell_measured', transform=ZUP_TO_YUP)
    fn = f'model_f{fl}.glb'; scene.export(os.path.join(OUT, fn))
    summary.append(dict(floor=fl, file=fn, faces=len(idx), pages=npg, mb=round(os.path.getsize(os.path.join(OUT, fn)) / 1e6, 1)))
    log(summary[-1])
json.dump(dict(res_cm=RES * 100, files=summary), open(W('mesh', 'structure_export.json'), 'w'), indent=1)

mp = os.path.join(OUT, 'meta.json'); meta = json.load(open(mp))
meta['structure'] = dict(files=[dict(floor=x['floor'], file=x['file'], mb=x['mb']) for x in summary], faces=len(faces), res_cm=round(RES * 100, 1),
    note=f'Simplified structural model in the style of a SketchUp model: walls, floors and ceilings fitted as flat rectangles on the building axes from the cleaned LiDAR cloud, with door openings, and photo textures projected from {len(recs):,} photos at {RES * 100:.0f} cm per texel (exposure-equalized, people masked). The central stairwell keeps its measured mesh. Each floor is a glTF 2.0 file that can be opened in Blender, Unity or Unreal.')
json.dump(meta, open(mp, 'w'), indent=1); log('meta.structure written')
