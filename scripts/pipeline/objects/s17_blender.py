"""Stage 17: Blender scene files (headless).  ~/apps/blender-4.2.3-linux-x64/blender -b -P s17_blender.py -- <data_dir> <objects_dir> <out_dir>
Per floor j:  blunt_hall_f<j>.blend  with collections
   Structure   - textured walls / floors / ceilings (model_f<j>.glb, photo atlases packed into the .blend)
   Stairs      - the measured stairwell mesh (vertex colours)
   Furniture   - one object per detected item, grouped in sub-collections by label, origin at the bottom centre;
                 SAM 3D mesh fitted to the LiDAR when the fit is good (custom property source), else the LiDAR mesh cut-out
plus blunt_hall_all.blend (one collection per floor) and a Cycles preview render per floor.
Building coordinates: metres, Z up (the glTF importer converts the glb's Y-up back to Z-up)."""
import bpy, sys, os, json, math
from mathutils import Vector

DATA, OBJ, OUT = sys.argv[sys.argv.index('--') + 1:][:3]
os.makedirs(OUT, exist_ok=True)
meta = json.load(open(os.path.join(DATA, 'meta.json')))
index = json.load(open(os.path.join(OBJ, 'index.json')))['objects']
FIT = os.path.join(os.path.dirname(OBJ), 'sam3d_fit')                   # SAM 3D meshes fitted to the LiDAR (s20), when good enough
fits = json.load(open(os.path.join(FIT, 'fit.json'))) if os.path.exists(os.path.join(FIT, 'fit.json')) else {}
levels = [(j, f['file'] if 'file' in f else None) for j, f in enumerate(meta['structure']['files'])]
FL = meta['floors']


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene; sc.unit_settings.system = 'METRIC'
    return sc


def vcol_material(name='VertexColour'):
    m = bpy.data.materials.get(name)
    if m: return m
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree
    bsdf = nt.nodes['Principled BSDF']; bsdf.inputs['Roughness'].default_value = 1.0
    ca = nt.nodes.new('ShaderNodeVertexColor'); nt.links.new(ca.outputs['Color'], bsdf.inputs['Base Color'])
    return m


REG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'regions', os.environ.get('BH_REGIONS', '2026-11-04-142939') + '.json')
STAIRS = [r['box'] for r in json.load(open(REG)) if r.get('stairs')] if os.path.exists(REG) else []
TRAJ = json.load(open(os.path.join(DATA, 'trajectory.json')))['p']


def unlit_switch(sc):
    """Every material gets Mix(Principled, Emission(base colour)) driven by the scene property 'unlit' (1 = flat photo colours,
    no lights needed; 0 = normal shading, add lights).  Uses an Attribute node of type View Layer, so no drivers / scripts."""
    sc['unlit'] = 1.0
    sc.id_properties_ui('unlit').update(min=0.0, max=1.0, description='1 = flat photo colours (no lights needed), 0 = lit by scene lights')
    for m in bpy.data.materials:
        if not m.use_nodes or m.get('unlit_done'): continue
        nt = m.node_tree; b = nt.nodes.get('Principled BSDF'); mo = nt.nodes.get('Material Output')
        if not b or not mo: continue
        e = nt.nodes.new('ShaderNodeEmission'); src = b.inputs['Base Color']
        if src.links: nt.links.new(src.links[0].from_socket, e.inputs['Color'])
        else: e.inputs['Color'].default_value = src.default_value
        a = nt.nodes.new('ShaderNodeAttribute'); a.attribute_type = 'VIEW_LAYER'; a.attribute_name = 'unlit'
        mx = nt.nodes.new('ShaderNodeMixShader'); nt.links.new(a.outputs['Fac'], mx.inputs['Fac'])
        nt.links.new(b.outputs['BSDF'], mx.inputs[1]); nt.links.new(e.outputs['Emission'], mx.inputs[2]); nt.links.new(mx.outputs['Shader'], mo.inputs['Surface'])
        e.location = (b.location.x, b.location.y - 450); a.location = (b.location.x, b.location.y + 250); mx.location = (mo.location.x - 200, mo.location.y)
        m['unlit_done'] = 1


def corridor_camera(sc, j, name='Corridor camera'):
    """Camera at eye height (1.6 m) in the middle of the longest straight stretch of the walking path on floor j, looking along it."""
    import numpy as np
    T = np.array(TRAJ); f = FL[j]; ok = (T[:, 2] > f['zmin']) & (T[:, 2] < f['zmax'])
    for b in STAIRS: ok &= ~((T[:, 0] > b[0]) & (T[:, 0] < b[1]) & (T[:, 1] > b[2]) & (T[:, 1] < b[3]))
    best = (0, None)
    for i in np.flatnonzero(ok):
        k = i
        while k + 1 < len(T) and ok[k + 1]:
            d = T[k + 1, :2] - T[i, :2]; n = np.linalg.norm(d)
            if n > 0.5:                                    # straight: every point within 0.4 m of the chord
                u = d / n; q = T[i:k + 2, :2] - T[i, :2]; off = np.abs(q[:, 0] * u[1] - q[:, 1] * u[0])
                if off.max() > 0.4: break
            k += 1
        L = np.linalg.norm(T[k, :2] - T[i, :2])
        if L > best[0]: best = (L, (i, k))
    if best[1] is None or best[0] < 1.0:                   # small landing: stand at the path centre, look at the farthest path point
        band = (T[:, 2] > f['zmin']) & (T[:, 2] < f['zmax']); Q = T[band if band.any() else slice(None), :2]
        pos = np.median(Q, 0); far = Q[np.argmax(np.linalg.norm(Q - pos, axis=1))]; d = (far - pos) / max(np.linalg.norm(far - pos), 1e-3)
        pos = pos - d * 0.5
    else:
        i, k = best[1]; p0 = T[i, :2]; d = (T[k, :2] - p0) / best[0]; pos = p0 + d * min(1.0, best[0] * 0.1)
    cam = bpy.data.objects.new(name, bpy.data.cameras.new(name)); sc.collection.objects.link(cam)
    cam.location = (pos[0], pos[1], f['z'] + 1.6); cam.rotation_euler = (math.radians(88), 0, math.atan2(d[1], d[0]) - math.pi / 2)
    cam.data.lens = 18; cam.data.clip_start = 0.05; cam.data.clip_end = 300
    return cam


def render_ready(sc, j):
    unlit_switch(sc); cam = corridor_camera(sc, j)
    if cam: sc.camera = cam
    w = bpy.data.worlds.new('World'); sc.world = w; w.use_nodes = True; w.node_tree.nodes['Background'].inputs['Strength'].default_value = 1.0
    sc.render.engine = 'CYCLES'; sc.cycles.samples = 64; sc.cycles.use_denoising = True; sc.cycles.device = 'GPU'
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    sc.view_settings.view_transform = 'Standard'                         # photo colours as captured (Filmic/AgX would wash them out)


def coll(name, parent):
    c = bpy.data.collections.new(name); parent.children.link(c); return c


def import_floor(j, root):
    """Import structure + stairs + furniture of floor j under collection root."""
    st, sr, fu = coll(f'Structure', root), coll(f'Stairs', root), coll(f'Furniture', root)
    for c, n in ((st, 'Structure'), (sr, 'Stairs'), (fu, 'Furniture')): c.name = f'{n} - {FL[j]["name"]}' if root.name != 'Scene Collection' else n
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.join(DATA, meta['structure']['files'][j]['file']))
    for o in set(bpy.data.objects) - before:
        for c in list(o.users_collection): c.objects.unlink(o)
        if o.type != 'MESH': bpy.data.objects.remove(o); continue
        if 'stair' in o.name:
            o.name = f'stairwell_f{j}'; o.data.materials.clear(); o.data.materials.append(vcol_material()); sr.objects.link(o)
        else:
            o.name = f'structure_f{j}_{o.name.split("_")[-1]}'; st.objects.link(o)
            for m in o.data.materials:                                    # flat photo texture: no specular shine
                b = m.node_tree.nodes.get('Principled BSDF') if m and m.use_nodes else None
                if b: b.inputs['Roughness'].default_value = 1.0; b.inputs['Metallic'].default_value = 0.0
    sub, n = {}, 0
    for ob in index:
        if not (FL[j]['zmin'] <= ob['lo'][2] < FL[j]['zmax']): continue
        before = set(bpy.data.objects)
        ft = fits.get(f'{ob["id"]:03d}'); sam = bool(ft and ft['use'])
        bpy.ops.wm.ply_import(filepath=os.path.join(FIT, ft['file']) if sam else os.path.join(OBJ, ob['file']))
        o = (set(bpy.data.objects) - before).pop()
        for c in list(o.users_collection): c.objects.unlink(o)
        lab = ob['label']
        if lab not in sub: sub[lab] = coll(lab + ('' if root.name == 'Scene Collection' else f' f{j}'), fu)
        sub[lab].objects.link(o); o.name = f'{lab.replace(" ", "_")}_{ob["id"]:03d}'
        o.data.materials.append(vcol_material())
        o['label'] = lab; o['sam3_votes'] = ob['mean_votes']; o['lidar_points'] = ob['n']; o['source'] = 'SAM 3D' if sam else 'LiDAR mesh'
        if sam: o['fit_cover'] = ft['cover']; o['fit_rms_cm'] = ft['rms_cm']
        bpy.context.view_layer.objects.active = o; o.select_set(True); bpy.ops.object.shade_smooth() if sam else None; o.select_set(False)
        # origin at the bottom centre so items can be moved / swapped for library models
        bb = [o.matrix_world @ Vector(c) for c in o.bound_box]; cx = sum(v.x for v in bb) / 8; cy = sum(v.y for v in bb) / 8; cz = min(v.z for v in bb)
        bpy.context.scene.cursor.location = (cx, cy, cz)
        bpy.context.view_layer.objects.active = o; o.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR'); o.select_set(False); n += 1
    return n


def preview(j, path):
    sc = bpy.context.scene; objs = [o for o in sc.objects if o.type == 'MESH']
    if not objs: return
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts))); hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    cen = (lo + hi) / 2; ext = max(hi.x - lo.x, hi.y - lo.y)
    cam = bpy.data.objects.new('PreviewCam', bpy.data.cameras.new('PreviewCam')); sc.collection.objects.link(cam)
    cam.data.type = 'ORTHO'; cam.data.ortho_scale = ext * 1.05; cam.data.clip_end = 1000
    cam.location = (cen.x, cen.y, hi.z + 50); cam.rotation_euler = (0, 0, 0); sc.camera = cam
    cz = FL[j].get('ceiling_z') or (FL[j]['z'] + 2.4)                     # near clip plane just under the ceiling: plan view
    cam.data.clip_start = max(0.1, cam.location.z - (cz - 0.15))
    for o in objs: o.visible_shadow = False                               # the (clipped) ceiling would shade the whole floor
    sun = bpy.data.objects.new('Sun', bpy.data.lights.new('Sun', 'SUN')); sun.data.energy = 3; sun.rotation_euler = (math.radians(35), 0, math.radians(30)); sc.collection.objects.link(sun)
    sc.world = sc.world or bpy.data.worlds.new('World'); sc.world.use_nodes = True; sc.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 1.0
    sc.render.engine = 'CYCLES'; sc.cycles.samples = 32; sc.cycles.use_denoising = False
    try:
        pr = bpy.context.preferences.addons['cycles'].preferences; pr.compute_device_type = 'CUDA'; pr.get_devices()
        for d in pr.devices: d.use = d.type == 'CUDA'
        sc.cycles.device = 'GPU'
    except Exception as e: print('cycles gpu unavailable', e)
    sc.render.resolution_x = 1600; sc.render.resolution_y = int(1600 * (hi.y - lo.y) / max(hi.x - lo.x, 1e-3)) if hi.x - lo.x > hi.y - lo.y else 1600
    sc.render.resolution_y = max(400, min(sc.render.resolution_y, 2400)); sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam); bpy.data.objects.remove(sun)


bpy.context.preferences.filepaths.save_version = 0                     # no .blend1 backups
summary = []
for j, _ in levels:
    sc = reset(); n = import_floor(j, sc.collection)
    render_ready(sc, j); bpy.ops.file.pack_all(); fn = os.path.join(OUT, f'blunt_hall_f{j}.blend')
    bpy.ops.wm.save_as_mainfile(filepath=fn, compress=True)
    summary.append(dict(floor=j, name=FL[j]['name'], file=os.path.basename(fn), furniture=n, mb=round(os.path.getsize(fn) / 1e6, 1)))
    print('SUMMARY', summary[-1], flush=True)
    if os.environ.get('BH_PREVIEW', '1') == '1':
        sc.cycles.samples = 16; sc.render.resolution_x, sc.render.resolution_y = 960, 540; sc.render.filepath = os.path.join(OUT, f'preview_f{j}_corridor.png')
        pr = bpy.context.preferences.addons['cycles'].preferences; pr.compute_device_type = 'CUDA'; pr.get_devices()
        for d in pr.devices: d.use = d.type == 'CUDA'
        bpy.ops.render.render(write_still=True)
        sc['unlit'] = 0.0; preview(j, os.path.join(OUT, f'preview_f{j}.png')); sc['unlit'] = 1.0
sc = reset(); tot = 0
for j, _ in levels:
    r = coll(FL[j]['name'], sc.collection); tot += import_floor(j, r)
render_ready(sc, 2); bpy.ops.file.pack_all(); fn = os.path.join(OUT, 'blunt_hall_all.blend'); bpy.ops.wm.save_as_mainfile(filepath=fn, compress=True)
summary.append(dict(floor='all', file='blunt_hall_all.blend', furniture=tot, mb=round(os.path.getsize(fn) / 1e6, 1)))
print('SUMMARY', summary[-1], flush=True)
json.dump(summary, open(os.path.join(OUT, 'blender_summary.json'), 'w'), indent=1)
