"""Comparison stage 2b: Cloud2BIM's IFC (walls, doors, slabs) -> triangle mesh of the walls for scoring.
Cloud2BIM (VaclavNezerka/Cloud2BIM, MIT) was run on compare/f<k>/cloud2bim.xyz with cloud2bim_f2.yaml and the two
changes in cloud2bim.patch (slab-density threshold 0.6 -> 0.35 for handheld scans, LaTeX off in its plots).
Run with the Cloud2BIM env (has ifcopenshell):  BH_WORK=... BH_FLOOR=2 BH_C2B_SECONDS=<run time> python c02_cloud2bim.py
Out: compare/f<k>/cloud2bim_walls.npz (V, F, doors, seconds), cloud2bim_summary.json"""
import os, json
import numpy as np, ifcopenshell, ifcopenshell.geom

WORK = os.environ['BH_WORK']; FI = int(os.environ.get('BH_FLOOR', 2)); D = os.path.join(WORK, 'compare', os.environ.get('BH_CMP_SET', f'f{FI}'))
f = ifcopenshell.open(os.path.join(D, 'cloud2bim.ifc'))
st = ifcopenshell.geom.settings(); st.set(st.USE_WORLD_COORDS, True)


def mesh(e):
    s = ifcopenshell.geom.create_shape(st, e); g = s.geometry
    return np.array(g.verts, float).reshape(-1, 3), np.array(g.faces, int).reshape(-1, 3)


V, F, doors, counts = [], [], [], {}
for t in ('IfcWall', 'IfcWallStandardCase', 'IfcDoor', 'IfcWindow', 'IfcSlab', 'IfcSpace', 'IfcOpeningElement'):
    counts[t] = len(f.by_type(t, include_subtypes=False))
for w in f.by_type('IfcWall'):                                         # includes IfcWallStandardCase
    try: v, t = mesh(w)
    except Exception as e: print('skip wall', w.id(), e); continue
    F.append(t + sum(len(x) for x in V)); V.append(v)
for d in f.by_type('IfcDoor'):
    try: v, _ = mesh(d); doors.append([*v.min(0), *v.max(0)])
    except Exception as e: print('skip door', d.id(), e)
V = np.vstack(V) if V else np.zeros((0, 3)); F = np.vstack(F) if F else np.zeros((0, 3), int)
np.savez(os.path.join(D, 'cloud2bim_walls.npz'), V=V, F=F, doors=np.array(doors).reshape(-1, 6),
         seconds=float(os.environ.get('BH_C2B_SECONDS', 'nan')))
json.dump(dict(counts=counts, z_range=[float(V[:, 2].min()), float(V[:, 2].max())] if len(V) else None), open(os.path.join(D, 'cloud2bim_summary.json'), 'w'), indent=1)
print('Cloud2BIM IFC:', counts, '| wall triangles', len(F), '| doors', len(doors))
