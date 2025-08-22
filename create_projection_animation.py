#!/usr/bin/env python3
import argparse
from pathlib import Path

def find_stack_dataset(h5_path):
    import h5py
    import numpy as np

    cand = None
    with h5py.File(h5_path, "r") as f:
        def walk(name, obj):
            nonlocal cand
            if isinstance(obj, h5py.Dataset):
                if obj.ndim == 3 and np.issubdtype(obj.dtype, np.number):
                    # prefer the largest 3D dataset (likely the projections)
                    size = obj.size
                    if cand is None or size > cand[2]:
                        cand = (name, obj.shape, size)
        f.visititems(walk)
    return cand  # (path, shape, size) or None

def load_stack(h5_path, dset_path):
    import h5py
    import numpy as np
    with h5py.File(h5_path, "r") as f:
        data = f[dset_path][...]
    data = np.asarray(data)
    # Ensure shape = (N, H, W)
    if data.shape[0] == min(data.shape):
        pass
    else:
        # Try to interpret last dim as frames
        # If shape is (H,W,N) rotate axes to (N,H,W)
        import numpy as np
        if data.shape[-1] == min(data.shape):
            data = np.moveaxis(data, -1, 0)
    return data

def normalize_to_uint8(stack):
    import numpy as np
    x = stack.astype(np.float32)
    # robust normalization (ignore extreme outliers)
    lo = np.percentile(x, 1.0)
    hi = np.percentile(x, 99.0)
    if hi <= lo:
        lo, hi = x.min(), x.max()
    x = (x - lo) / (hi - lo + 1e-12)
    x = np.clip(x, 0, 1)
    x = (x * 255.0).round().astype(np.uint8)
    return x

def optionally_downscale(stack_u8, scale):
    if scale == 1.0:
        return stack_u8
    from PIL import Image
    import numpy as np
    out = []
    for frame in stack_u8:
        h, w = frame.shape
        nh, nw = max(1, int(h*scale)), max(1, int(w*scale))
        img = Image.fromarray(frame)
        img = img.resize((nw, nh), Image.BILINEAR)
        out.append(np.array(img, dtype=frame.dtype))
    return out

def save_mp4(frames_u8, path_mp4, fps):
    import imageio
    with imageio.get_writer(path_mp4, mode="I", fps=fps, codec="libx264", quality=6) as w:
        for fr in frames_u8:
            # convert grayscale -> 3ch
            w.append_data(fr)

def save_gif(frames_u8, path_gif, fps):
    import imageio
    duration = 1.0 / max(fps, 1)
    imageio.mimsave(path_gif, frames_u8, duration=duration, loop=0)

def write_index_html(out_dir, title="Projections", mp4_name="projections.mp4", gif_name="projections.gif"):
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
  body {{ margin:0; background:#0b0b0b; color:#eee; font-family:system-ui, sans-serif; }}
  .wrap {{ max-width: 900px; margin: 0 auto; padding: 16px; text-align:center; }}
  video, img {{ width: 100%; height: auto; border-radius: 12px; }}
  h1 {{ font-size: 1.1rem; font-weight:600; opacity:0.9; }}
  p {{ opacity:0.8; font-size:0.95rem; }}
</style>
</head>
<body>
  <div class="wrap">
    <h1>{title}</h1>
    <video id="vid" autoplay muted playsinline loop>
      <source src="{mp4_name}" type="video/mp4">
      <img src="{gif_name}" alt="Animated projections">
    </video>
    <p>Rotating/scrolling through HDF5 projections (autoplay, loop, muted for mobile compatibility).</p>
  </div>
<script>
  // Attempt to start playback if mobile blocked the first try.
  const v = document.getElementById('vid');
  v.addEventListener('canplay', () => v.play().catch(()=>{{}}));
</script>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Make autoplaying webpage from HDF5 projection stack.")
    parser.add_argument("--h5", required=True, help="Path to mz_projections.hdf (or any HDF5).")
    parser.add_argument("--dataset", default=None, help="Dataset path inside HDF5 (optional; auto-detect if omitted).")
    parser.add_argument("--outdir", default="site", help="Output directory (default: site)")
    parser.add_argument("--fps", type=int, default=12, help="Frames per second (default: 12)")
    parser.add_argument("--scale", type=float, default=1.0, help="Downscale factor (e.g. 0.5 to halve size)")
    parser.add_argument("--title", default="Projections Viewer", help="Title for the webpage")
    args = parser.parse_args()

    h5_path = Path(args.h5)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) find dataset if not given
    dset = args.dataset
    if dset is None:
        found = find_stack_dataset(h5_path)
        if not found:
            raise SystemExit("Could not auto-detect a 3D numeric dataset in the HDF5. Please pass --dataset /path/in/file")
        dset = found[0]
        print(f"[info] Auto-detected dataset: {dset} with shape {found[1]}")

    # 2) load and normalize
    stack = load_stack(h5_path, dset)
    print(f"[info] Loaded stack shape: {stack.shape} (expected N,H,W)")
    stack_u8 = normalize_to_uint8(stack)

    # 3) optional downscale
    frames = optionally_downscale(stack_u8, args.scale)

    # 4) save MP4 + GIF
    mp4_path = out_dir / "projections.mp4"
    gif_path = out_dir / "projections.gif"
    print(f"[info] Writing MP4 -> {mp4_path}")
    save_mp4(frames, mp4_path, fps=args.fps)
    print(f"[info] Writing GIF -> {gif_path}")
    save_gif(frames, gif_path, fps=args.fps)

    # 5) write index.html
    write_index_html(out_dir, title=args.title)
    print(f"[done] Wrote {out_dir/'index.html'}")
    print("[next] You can now host the 'site' folder with GitHub Pages (steps below).")

if __name__ == "__main__":
    main()

