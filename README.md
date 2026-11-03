# Blunt Hall 3D

A web viewer for a handheld LiDAR scan of the inside of Blunt Hall. The scan was captured on 29 Oct 2026 with a GeoCue **TrueView GO** (Go116S: Hesai 16-beam rotating LiDAR, three 5 MP fisheye cameras).

**Live site:** https://shanem73.github.io/Blunt_hall_3D/

| View | What it shows |
|---|---|
| LiDAR points | 2.45 M people-free points (2 cm voxel grid) from 19.5 M raw returns, colored by intensity or height |
| Colored points | The same points colored from 1,487 photos (3 cameras × ~2 Hz) |
| SLAM map | Top-down occupancy floor plan with the SLAM-optimized walking path (66.8 m, 245 s) |
| 3D model | Preliminary colored mesh (screened Poisson, 700 k triangles) |
| Gaussian splat | 3D Gaussian Splatting model (460 k Gaussians), trained on the photos with LiDAR depth supervision |

No build step is needed: the site is static HTML + three.js. To preview it locally, run `python3 -m http.server` and open `http://localhost:8000`.

## How the raw data was read

TrueView GO projects are normally processed in GeoCue's TrueView EVO software. These viewers were built straight from the raw project folder by reverse-engineering its files:

| File | Format |
|---|---|
| `SLAM_DATA/TMP_00000.bin` | 128-B header (`ffaa`, n_frames, record size = 20) · per-frame index `(u32 n_points, u32 unix_s)` · point records at byte 24128: `f32 x,y,z` (body frame), `f32 t`, `u8 intensity`, `u8 ring`, `u16 0` |
| `SLAM_DATA/slam_optimized_trajectory.tra` | XOR 0x53, then 14 × float64 per frame: `t, x, y, z, qw, qx, qy, qz, …` (body → world) |
| `IMG/CameraN/CameraN.cam` | Concatenated JPEGs, each preceded by `0xAAAAAAAA` + `"<unix_time>.jpg"` |
| `IMG/CameraN/CAM-N.CP` | Protobuf: field 8 = `fx, fy, cx, cy`; field 9 = Kannala–Brandt fisheye `k1–k4`; field 10 = mount angles. Fields 10.4–6 are intrinsic **ZYX** Euler angles (rad) giving body→camera, and 10.7–9 is the camera offset in the body frame (m) |

The camera-extrinsic convention was confirmed by photometric consistency: the same 3D point should have the same color in different photos. All three cameras picked the same convention, and refinement moved the angles by less than 2°.

## Pipeline (`scripts/`)

Raw folder via `TVGO_RAW` (default `~/3DTemplehall`); run in a work directory:

1. `register.py`: decode the points and trajectory, then transform every frame into the world frame. Writes a LAZ file plus `world_*.npy`. A 2 cm voxel grid with statistical outlier removal gives `clean_2cm.ply`.
2. `gs_prep.py` and `gs_masks.py`: undistort the fisheye photos to pinhole (972×1296, f = 450 px), write the camera poses, and make person masks (Mask R-CNN) plus valid-area masks.
3. `colorize.py` (run twice): projects the points into every photo on the GPU, with z-buffer occlusion and distance/angle weights. Pixels where a person was detected are skipped. The second run down-weights color outliers.
4. `carve.py`: free-space carving. A point that a laser beam later passed through belongs to a moving person, so it is removed (12 % of points).
5. `mesh.py`: screened Poisson mesh of the people-free cloud.
6. `build_web.py` and `slam_map.py`: write `data/points.bin`, `trajectory.json`, `slam_map.png` and `meta.json`.
7. `gs_train.py`, then `gs_export.py`: 3D Gaussian Splatting with [gsplat](https://github.com/nerfstudio-project/gsplat). Starts from the LiDAR points and trains for 30k iterations at about 12 min on one A100. The loss is masked L1 + SSIM, plus **LiDAR depth supervision**, which removes the fog-like floaters typical of handheld captures. Per-image exposure correction is included. The model is exported to the 32-byte `.splat` format.

`read_tvgo.py`, `pb.py` (generic protobuf dump) and `calib.py` are helpers. `photo_calib.py` and `conv_search.py` are the calibration checks.

## Known limitations

- Coordinates are local SLAM metres (no GNSS indoors). Z is up and the floor is at about −1.0 m.
- People who walked alongside the operator were removed (photo masks + LiDAR carving), but faint smudges and one person standing still near the stairs remain.
- The Gaussian splat looks best from viewpoints near the walking path. Far from it (or in corners the cameras barely saw) it becomes blurry or hazy.
- The mesh has holes where the scanner had no line of sight, and blobs at open doorways.
