import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import math
import sys

def infer_square_shape(length):
    s = int(math.sqrt(length))
    return (s, s) if s * s == length else None

def plot_depth_heatmap(depth_map, cmap='hot', outpath=None, show=True):
    if depth_map.ndim != 2:
        raise ValueError(f"plot_depth_heatmap expects 2D array, got shape {depth_map.shape}")
    plt.imshow(depth_map, cmap=cmap, interpolation='nearest')
    plt.colorbar()
    plt.title('Depth Heatmap')
    if outpath:
        plt.savefig(outpath, bbox_inches='tight', dpi=150)
        print(f"Saved heatmap to {outpath}")
    if show:
        plt.show()
    plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect .npz / .npy depth maps and plot heatmap.')
    parser.add_argument('--input_file', type=str, help='Path to the input .npz or .npy file')
    parser.add_argument('--key', '-k', default=None, help='Key inside .npz (default: first)')
    parser.add_argument('--index', '-i', type=int, default=0, help='Index for first dimension (N,H,W) to plot')
    parser.add_argument('--height', type=int, default=None, help='Height to reshape flattened maps')
    parser.add_argument('--width', type=int, default=None, help='Width to reshape flattened maps')
    parser.add_argument('--out', type=str, default=None, help='Output PNG path (optional)')
    parser.add_argument('--no_show', action='store_true', help="Don't call plt.show()")
    parser.add_argument('--cmap', type=str, default='hot', help='Matplotlib colormap (default: hot)')
    args = parser.parse_args()

    path = os.path.expanduser(args.input_file)
    if not os.path.exists(path):
        print("File not found:", path, file=sys.stderr)
        sys.exit(1)

    if path.endswith('.npz'):
        npz = np.load(path, allow_pickle=True)
        print("NPZ keys:", npz.files)
        if args.key is None:
            key = npz.files[0]
        else:
            key = args.key
            if key not in npz.files:
                print(f"Key '{key}' not found. Available keys: {npz.files}", file=sys.stderr)
                sys.exit(1)
        data = np.asarray(npz[key])
    else:
        # .npy or other single-array file
        data = np.load(path, allow_pickle=True)
        print(f"Loaded .npy shape: {data.shape}, dtype: {data.dtype}")

    print("raw shape:", data.shape, "dtype:", data.dtype)

    # handle common layouts
    depth_map = None
    if data.ndim == 3:
        # (N, H, W) or (H, W, C) ambiguous: assume (N, H, W) if first dim > 1 and second/third reasonable
        if data.shape[0] > 1 and (data.shape[1] > 1 and data.shape[2] > 1):
            depth_map = np.asarray(data[args.index])
        else:
            depth_map = data
            if depth_map.shape[2] in (3,4):
                # maybe color image stack -> convert first channel
                depth_map = depth_map[..., 0]
    elif data.ndim == 2:
        # if dimensions already match provided H,W, accept as (H,W)
        if args.height and args.width and data.shape == (args.height, args.width):
            depth_map = data
        elif data.shape[0] == 1 and data.shape[1] > 1:
            vec = data[0]
            if args.height and args.width:
                depth_map = vec.reshape((args.height, args.width))
            else:
                s = infer_square_shape(vec.size)
                if s:
                    depth_map = vec.reshape(s)
                else:
                    print("Cannot infer shape for single-row flattened map; provide --height and --width", file=sys.stderr)
                    sys.exit(1)
        elif data.shape[0] > 1 and data.shape[1] > 1 and (data.shape[1] != data.shape[0]):
            # maybe (N, L) where each row is flattened map
            if args.height and args.width:
                H, W = args.height, args.width
                if data.shape[1] != H * W:
                    print("Provided --height/--width do not match data shape", file=sys.stderr)
                    sys.exit(1)
                depth_map = data[args.index].reshape((H, W))
            else:
                s = infer_square_shape(data.shape[1])
                if s:
                    depth_map = data[args.index].reshape(s)
                else:
                    print("2D array of shape (N, L) where L not square; provide --height/--width", file=sys.stderr)
                    sys.exit(1)
        else:
            # assume single (H,W)
            depth_map = data
    elif data.ndim == 1:
        # flattened single map
        vec = data
        if args.height and args.width:
            depth_map = vec.reshape((args.height, args.width))
        else:
            s = infer_square_shape(vec.size)
            if s:
                depth_map = vec.reshape(s)
            else:
                print("Single 1D array found; provide --height/--width to reshape", file=sys.stderr)
                sys.exit(1)
    else:
        print(f"Unsupported data ndim: {data.ndim}", file=sys.stderr)
        sys.exit(1)
    if depth_map is None:
        raise SystemExit("Failed to extract a depth map from the input data.")
    if depth_map.ndim == 3 and depth_map.shape[0] == 1:
        depth_map = depth_map[0] 
    print("Final map shape:", depth_map.shape, "min/max on finite:", np.nanmin(depth_map), np.nanmax(depth_map))
    plot_depth_heatmap(depth_map, cmap=args.cmap, outpath=args.out, show=not args.no_show)