"""Shared paths, readers and pose helpers for the dataset-parameterized pipeline.

Every stage reads the raw TrueView GO folder from TVGO_RAW and writes into BH_WORK, split by data type:
  decoded/  trajectory, registered world points, image index
  calib/    camera rotations, per-image exposure gains
  clouds/   denoised 1 cm cloud, transient (people) flags
  color/    colorized cloud
  mesh/     Poisson mesh tiles
  images/   undistorted pinhole photos and person masks
  splat/    Gaussian-splat training data and parameters
  renders/  check images
  logs/     stage logs
"""
import os, sys, glob, json
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'tvgo'))
from read_tvgo import read_slam_bin  # noqa: E402
from calib import CAL                 # noqa: E402  (identical CAM-N.CP for device SN-006X)

RAW = os.environ.get('TVGO_RAW')
WORK = os.environ.get('BH_WORK')
if not RAW or not WORK:
    sys.exit('set TVGO_RAW (raw project folder) and BH_WORK (work folder for this dataset)')
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))


def W(*parts):
    """Path inside the work folder; creates the parent folder."""
    p = os.path.join(WORK, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def slam_bins():
    return sorted(glob.glob(os.path.join(RAW, 'SLAM_DATA', 'TMP_[0-9]*.bin')))


def read_slam_all():
    """Concatenate all TMP_*.bin segments -> (index (nframes,2), list of per-segment point arrays, frame offsets)."""
    idxs, pts = [], []
    for f in slam_bins():
        i, p = read_slam_bin(f)
        idxs.append(i); pts.append(p)
    idx = np.vstack(idxs)
    return idx, pts


def frame_points(idx, pts):
    """Yield (frame_number, body-frame records) across all segments."""
    k = 0
    for p in pts:
        n = 0
        while n < len(p):
            c = int(idx[k, 0]); yield k, p[n:n + c]; n += c; k += 1


def load_tra(name='slam_optimized_trajectory.tra'):
    d = np.fromfile(os.path.join(RAW, 'SLAM_DATA', name), np.uint8) ^ 0x53
    T = np.frombuffer(d.tobytes(), np.float64)
    return T[:len(T) // 14 * 14].reshape(-1, 14).copy()


def traj(refined=True):
    f = W('decoded', 'traj_refined.npy')
    if refined and os.path.exists(f):
        return np.load(f)
    return np.load(W('decoded', 'slam_trajectory.npy'))


def quat_R(q):
    q = q / np.linalg.norm(q)
    return Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


class Poser:
    """Body->world pose at arbitrary time (slerp between trajectory frames)."""
    def __init__(self, T):
        self.T = T
        self.R = Rot.from_quat(T[:, [5, 6, 7, 4]])

    def __call__(self, t):
        T = self.T
        i = int(np.clip(np.searchsorted(T[:, 0], t), 1, len(T) - 1))
        a = float(np.clip((t - T[i - 1, 0]) / (T[i, 0] - T[i - 1, 0]), 0, 1))
        R = Slerp([0, 1], self.R[[i - 1, i]])([a])[0].as_matrix()
        return R, (1 - a) * T[i - 1, 1:4] + a * T[i, 1:4]


def Rcb(cam):
    """Body->camera rotation (intrinsic ZYX Euler from CAM-N.CP, verified photometrically on scan 1)."""
    return Rot.from_euler('ZYX', CAL[cam]['a']).as_matrix()


def image_index():
    return json.load(open(W('decoded', 'images.json')))


def read_image(rec):
    """Decode one fisheye JPEG straight from the .cam container (no extracted copies on disk)."""
    import cv2
    with open(os.path.join(RAW, 'IMG', f"Camera{rec['cam']}", f"Camera{rec['cam']}.cam"), 'rb') as f:
        f.seek(rec['off']); buf = f.read(rec['len'])
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


def log(*a):
    print(*a, flush=True)


def removed_mask(n=None):
    """People (s07) + clutter (s07b) flags over color/hq_colored.npz points."""
    rm = np.load(W('clouds', 'person_rm.npy'))
    f = W('clouds', 'clutter_rm.npy')
    if os.path.exists(f) and os.path.getmtime(f) >= os.path.getmtime(W('color', 'hq_colored.npz')):
        rm = rm | np.load(f)
    return rm


def detect_floors(P, Nr, T):
    """Floors = peaks of up-facing horizontal surfaces (normals face the path) that the walk stood on for >= 15 s
    (device 0.8-1.7 m above); each floor's band runs to the next floor; ceiling = mode of down-facing surfaces."""
    tz = T[:, 3]
    hz = P[Nr[:, 2] > 0.9, 2]           # up-facing only (normals face the path): ceilings are excluded
    h, e = np.histogram(hz, bins=np.arange(hz.min(), hz.max() + 0.05, 0.05))
    cand = [i for i in range(1, len(h) - 1) if h[i] >= h[i - 1] and h[i] >= h[i + 1] and h[i] > 0.01 * h.max()]
    floors = []
    for i in sorted(cand, key=lambda i: -h[i]):
        z = (e[i] + e[i + 1]) / 2; above = tz - z; on = ((above > 0.8) & (above < 1.7)).sum()
        if on > 150 and all(abs(z - f['z']) > 2.0 for f in floors):     # >= 15 s of walking on it
            floors.append(dict(z=round(float(z), 2), dwell_s=round(on / 10, 1)))
    floors.sort(key=lambda f: f['z'])
    for j, f in enumerate(floors):
        f['zmin'] = round(f['z'] - 0.3, 2); f['zmax'] = round(floors[j + 1]['z'] - 0.3, 2) if j + 1 < len(floors) else round(float(P[:, 2].max()) + 0.1, 2)
        dz = P[(Nr[:, 2] < -0.9) & (P[:, 2] > f['z'] + 2.0) & (P[:, 2] < min(f['zmax'] + 0.3, f['z'] + 6)), 2]   # down-facing = ceilings
        if len(dz) > 1000:
            hc, ec = np.histogram(dz, bins=np.arange(dz.min(), dz.max() + 0.1, 0.05)); cz_ = float(ec[np.argmax(hc)] + 0.025)
        else: cz_ = min(f['zmax'] + 0.3, f['z'] + 3.0)
        f['ceiling_z'] = round(cz_, 2); f['ceiling'] = round(cz_ - 0.15, 2)
    return floors
