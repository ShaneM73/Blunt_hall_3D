"""Readers for raw TrueView GO (Go116S) project folders.
SLAM_DATA/TMP_00000.bin : 128-B header ('ffaa', ver, nframes, rec_size) + nframes*(uint32 npts, uint32 unix_s) index,
                          zero pad to 24128, then npts*20-B records: f32 x,y,z, f32 t, u8 a, u8 b, u16 c.
IMG/CameraN/CameraN.cam : container of JPEGs, each preceded by 0xAAAAAAAA + '<unix_time>.jpg' name.
"""
import numpy as np, re, sys, os

def read_slam_bin(path):
    d = np.memmap(path, dtype=np.uint8, mode='r')
    hdr = np.frombuffer(d[:16], np.uint32)
    nfr, rec = int(hdr[2]), int(hdr[3])
    idx = np.frombuffer(d[128:128+8*nfr], np.uint32).reshape(-1, 2)
    N = int(idx[:, 0].sum())
    start = len(d) - rec*N
    dt = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('t','<f4'),('a','u1'),('b','u1'),('c','<u2')])
    pts = np.frombuffer(d[start:], dt)
    return idx, pts

def extract_cam(path, outdir):
    os.makedirs(outdir, exist_ok=True)
    d = open(path, 'rb').read()
    names = []
    for m in re.finditer(rb'\xaa\xaa\xaa\xaa([0-9.]+\.jpg)', d):
        s = d.find(b'\xff\xd8\xff', m.end()); e = d.find(b'\xff\xd9', s) + 2
        name = m.group(1).decode(); open(os.path.join(outdir, name), 'wb').write(d[s:e]); names.append(name)
    return names
