"""Comparison stage 7a: run LingBot-Map (Robbyant, Apache-2.0) on a walkthrough video without its viser viewer.
Env: ~/.conda/envs/lingbot-map (torch 2.8 cu128, flashinfer 0.6.17 + jit-cache), repo ~/apps/cmp/lingbot-map,
weights work/models/lingbot-map/ (lingbot-map-long.pt for long corridors). Takes demo.py arguments, e.g.
  PATH=~/.conda/envs/lingbot-map/bin:$PATH LB_OUT=out.npz python c07_lingbot_run.py --model_path .../lingbot-map-long.pt \\
      --video_path walk.mp4 --fps 10 --mode windowed --window_size 128 --keyframe_interval 4 --overlap_keyframes 16
Out: npz with depth, depth_conf, extrinsic (w2c), intrinsic, images per frame (arbitrary scale)."""
import os, sys, numpy as np, types
sys.argv[0] = 'demo.py'; sys.path.insert(0, os.path.expanduser('~/apps/cmp/lingbot-map'))
import lingbot_map.vis as V
class Saver:
    def __init__(self, pred_dict, **kw):
        out = {k: (v.numpy() if hasattr(v, 'numpy') else v) for k, v in pred_dict.items() if isinstance(v, np.ndarray) or hasattr(v, 'numpy')}
        np.savez_compressed(os.environ['LB_OUT'], **out)
        print('saved', os.environ['LB_OUT'], {k: (v.shape, v.dtype) for k, v in out.items()})
    def run(self): pass
V.PointCloudViewer = Saver
import demo; demo.main()
