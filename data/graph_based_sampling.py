import os
import random
import pickle
import shutil
import threading
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor
import itertools


from scene import ScanNetScene, load_depth_png, load_depth, find_scene_paths


class ScanNetGraphDataset(Dataset):
    def __init__(
        self,
        root_dir,
        num_frames=6,
        num_samples=700,
        graph_cache=None,
        min_overlap=0.2,
        max_overlap=0.9,
        overlap_sample_step=64,
        depth_tolerance=0.05,
        max_frame_gap=50,
        transform=None,
        max_scenes=None,
    ):
        self.root_dir = root_dir
        self.num_samples = num_samples
        self.transform = transform
        self.seq_len = num_frames
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.overlap_sample_step = overlap_sample_step
        self.depth_tolerance = depth_tolerance
        self.max_frame_gap = max_frame_gap
        self.max_scenes = max_scenes

        # load or build graph
        if graph_cache and os.path.exists(graph_cache):
            print("Loading graph from cache...")
            with open(graph_cache, "rb") as f:
                self.scene_data = pickle.load(f)
        else:
            print("Building graph...")
            self.scene_data = self._build_graph()
            if graph_cache:
                with open(graph_cache, "wb") as f:
                    pickle.dump(self.scene_data, f)

    # --------------------------------------------------
    # GRAPH BUILDING
    # --------------------------------------------------
    def _compute_directed_overlap(
        self,
        depth_src,
        pose_src,
        depth_dst,
        world_to_cam_dst,
        intrinsics,
    ):
        sample_step = self.overlap_sample_step
        sampled_depth = depth_src[::sample_step, ::sample_step]
        valid_mask = sampled_depth > 0

        if not np.any(valid_mask):
            return 0.0

        grid_y, grid_x = np.mgrid[
            0:depth_src.shape[0]:sample_step,
            0:depth_src.shape[1]:sample_step,
        ]

        depths = sampled_depth[valid_mask]
        pixel_x = grid_x[valid_mask].astype(np.float32)
        pixel_y = grid_y[valid_mask].astype(np.float32)

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        cam_x = (pixel_x - cx) * depths / fx
        cam_y = (pixel_y - cy) * depths / fy
        cam_points = np.stack([cam_x, cam_y, depths, np.ones_like(depths)], axis=1)

        world_points = (pose_src @ cam_points.T).T
        dst_points = (world_to_cam_dst @ world_points.T).T[:, :3]

        positive_depth = dst_points[:, 2] > 0
        if not np.any(positive_depth):
            return 0.0

        dst_points = dst_points[positive_depth]
        projected_x = intrinsics[0, 0] * (dst_points[:, 0] / dst_points[:, 2]) + intrinsics[0, 2]
        projected_y = intrinsics[1, 1] * (dst_points[:, 1] / dst_points[:, 2]) + intrinsics[1, 2]

        dst_height, dst_width = depth_dst.shape
        inside = (
            (projected_x >= 0)
            & (projected_x < dst_width)
            & (projected_y >= 0)
            & (projected_y < dst_height)
        )
        if not np.any(inside):
            return 0.0

        dst_points = dst_points[inside]
        projected_x = projected_x[inside].astype(np.int32)
        projected_y = projected_y[inside].astype(np.int32)

        dst_depth_samples = depth_dst[projected_y, projected_x]
        valid_dst = dst_depth_samples > 0
        if not np.any(valid_dst):
            return 0.0

        dst_points = dst_points[valid_dst]
        dst_depth_samples = dst_depth_samples[valid_dst]
        depth_error = np.abs(dst_depth_samples - dst_points[:, 2])
        visible = depth_error <= self.depth_tolerance

        return float(np.count_nonzero(visible)) / float(np.count_nonzero(valid_mask))

    def _compute_pair_overlap(
        self,
        depth_i,
        pose_i,
        world_to_cam_i,
        depth_j,
        pose_j,
        world_to_cam_j,
        intrinsics,
    ):
        overlap_ij = self._compute_directed_overlap(
            depth_i,
            pose_i,
            depth_j,
            world_to_cam_j,
            intrinsics,
        )
        overlap_ji = self._compute_directed_overlap(
            depth_j,
            pose_j,
            depth_i,
            world_to_cam_i,
            intrinsics,
        )
        return 0.5 * (overlap_ij + overlap_ji)

    def _build_graph(self):
        scene_data = []

        scene_paths = find_scene_paths(self.root_dir)
        if self.max_scenes:
            scene_paths = scene_paths[:self.max_scenes]

        for scene_path in tqdm(scene_paths, desc="Scenes", unit="scene"):
            scene_name = os.path.relpath(scene_path, self.root_dir)
            scene = ScanNetScene(scene_path)

            if not scene.is_valid():
                continue

            if not scene.frame_ids:
                continue

            poses, valid_fids = scene.load_all_poses()

            if not valid_fids:
                continue

            world_to_camera = {
                fid: np.linalg.inv(pose)
                for fid, pose in poses.items()
                if fid in valid_fids
            }
            # within a scene the camera is fixed; intrinsics are shared across all frames
            intrinsics = scene.intrinsics

            color_dir = scene.color_dir
            depth_dir = scene.depth_dir

            depth_cache = {}

            # default-arg trick binds depth_dir and depth_cache at definition time,
            # avoiding the classic loop-closure bug
            _lock = threading.Lock()  # ensure thread safety if DataLoader uses multiple workers
            def get_depth(fid, _scene=scene, _cache=depth_cache):
                with _lock:
                    if fid not in _cache:
                        _cache[fid] = load_depth(_scene.depth_path(fid))
                    return _cache[fid]

            # Pre-load all depths once, sequentially, before threading to avoid
            # duplicate loads and potential races in ThreadPoolExecutor.
            for fid in valid_fids:
                if fid not in depth_cache:
                    depth_cache[fid] = load_depth(scene.depth_path(fid))

            # weighted adjacency: graph[fid_i][fid_j] = symmetric overlap score
            # nodes = individual frames; edge weight = how much the two views share
            graph = {fid: {} for fid in valid_fids}

            n = len(valid_fids)
            total_pairs = sum(min(self.max_frame_gap, n - 1 - i) for i in range(n))
            with tqdm(total=total_pairs, desc=f"  {scene} pairs", unit="pair", leave=False) as pbar:
                # parallelize pairwise overlap computation for speed
                graph = {fid: {} for fid in valid_fids}

                # build list of index pairs to evaluate
                pairs = [
                    (i, j)
                    for i in range(n)
                    for j in range(i + 1, min(i + 1 + self.max_frame_gap, n))
                ]

                if pairs:
                    # delegate to helper that uses ThreadPoolExecutor
                    graph = self._compute_all_pairs(valid_fids, poses, world_to_camera, get_depth, intrinsics)

                # advance progress bar by number of evaluated pairs
                pbar.update(len(pairs))

            scene_data.append({
                "scene": scene_name,
                "color_dir": color_dir,
                "depth_dir": depth_dir,
                "mde_depth_dir": scene.mde_depth_dir,
                "intrinsics": intrinsics,  # same camera for every frame in the scene
                "poses": poses,
                "graph": graph,
                "frame_ids": valid_fids    # only frames with valid poses
            })

        return scene_data

    def _compute_all_pairs(self, valid_fids, poses, world_to_camera, get_depth, intrinsics):
        """Compute all pairwise overlaps in parallel using ThreadPoolExecutor.

        Returns a graph dict mapping fid -> {other_fid: overlap}
        """
        n = len(valid_fids)
        pairs = [
            (i, j)
            for i in range(n)
            for j in range(i + 1, min(i + 1 + self.max_frame_gap, n))
        ]

        graph = {fid: {} for fid in valid_fids}

        def compute_pair(args):
            i, j = args
            fid_i, fid_j = valid_fids[i], valid_fids[j]
            overlap = self._compute_pair_overlap(
                get_depth(fid_i), poses[fid_i], world_to_camera[fid_i],
                get_depth(fid_j), poses[fid_j], world_to_camera[fid_j],
                intrinsics,
            )
            return fid_i, fid_j, overlap

        # ThreadPoolExecutor is fine here because numpy releases the GIL
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(compute_pair, pairs))

        for fid_i, fid_j, overlap in results:
            if self.min_overlap <= overlap <= self.max_overlap:
                graph[fid_i][fid_j] = overlap
                graph[fid_j][fid_i] = overlap

        return graph

    def resample_seq_len(self):
        """No-op when num_frames is fixed; kept for API compatibility."""
        pass

    # --------------------------------------------------
    # WEIGHTED RANDOM WALK SAMPLING
    # --------------------------------------------------
    def _random_walk(self, graph, start, length):
        seq = [start]
        current = start

        for _ in range(length - 1):
            neighbors = graph[current]  # {fid: overlap_weight}

            if not neighbors:
                break  # dead end

            nbr_fids = list(neighbors.keys())
            weights = [neighbors[n] for n in nbr_fids]

            # avoid immediate back-step when alternatives exist
            if len(seq) > 1 and len(nbr_fids) > 1:
                prev = seq[-2]
                pairs = [(n, w) for n, w in zip(nbr_fids, weights) if n != prev]
                if pairs:
                    nbr_fids, weights = zip(*pairs)
                    nbr_fids, weights = list(nbr_fids), list(weights)

            # sample next node weighted by overlap score:
            # higher overlap = more shared content = preferred transition
            weights = [max(w, 1e-6) for w in weights]  # ensure no zero/nan weights
            nxt = random.choices(nbr_fids, weights=weights, k=1)[0]
            seq.append(nxt)
            current = nxt

        return seq

    def _sample_sequence(self, scene_info):
        graph = scene_info["graph"]
        frame_ids = scene_info["frame_ids"]

        seq_len = self.seq_len

        # pick valid start (with neighbors)
        valid_starts = [f for f in frame_ids if len(graph[f]) > 0]
        if not valid_starts:
            # no edges at all — fall back to random frames with replacement
            return random.choices(frame_ids, k=seq_len)

        start = random.choice(valid_starts)

        seq = self._random_walk(graph, start, seq_len)

        # Pad if the walk hit a dead end before reaching seq_len
        while len(seq) < seq_len:
            seq.append(random.choice(valid_starts))

        return seq

    # --------------------------------------------------
    # LOADING
    # --------------------------------------------------
    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img

    def _load_depth(self, path):
        arr = np.load(path).astype(np.float32)
        t = torch.from_numpy(arr)
        return t if t.ndim == 3 else t.unsqueeze(0)  # ensure (1, H, W)

    def _load_mde_depth(self, path):
        """Load a ZoeDepth prediction (uint16 PNG, stored in mm → convert to metres)."""
        arr = np.array(Image.open(path)).astype(np.float32) / 1000.0
        t = torch.from_numpy(arr)
        return t if t.ndim == 3 else t.unsqueeze(0)  # ensure (1, H, W)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        scene_info = random.choice(self.scene_data)

        seq_ids = self._sample_sequence(scene_info)

        images, depths, mde_depths, poses = [], [], [], []

        for fid in seq_ids:
            img_path   = os.path.join(scene_info["color_dir"],     f"{fid}{ScanNetScene.COLOR_EXT}")
            depth_path = os.path.join(scene_info["depth_dir"],     f"{fid}{ScanNetScene.DEPTH_EXT}")
            mde_path   = os.path.join(scene_info["mde_depth_dir"], f"{fid}{ScanNetScene.MDE_DEPTH_EXT}")

            images.append(self._load_image(img_path))
            depths.append(self._load_depth(depth_path))
            mde_depths.append(self._load_mde_depth(mde_path))
            poses.append(torch.from_numpy(scene_info["poses"][fid]).float())
        images_t = torch.stack(images).contiguous()
        depths_t = torch.stack(depths).contiguous()
        mde_t = torch.stack(mde_depths).contiguous()
        poses_t = torch.stack(poses).contiguous()
        intr_t = torch.from_numpy(scene_info["intrinsics"]).float().contiguous()

        return {
            "images":     images_t,
            "depths":     depths_t,                            # GT depths
            "mde_depths": mde_t,                               # MDE prior
            "poses":      poses_t,
            "intrinsics": intr_t,
            "scene": scene_info["scene"],
            "frame_ids": seq_ids
        }


if __name__ == "__main__":
    import yaml
    import torchvision
    from torch.utils.data import DataLoader

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "dataset.yaml",
    )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["dataset"]
    out_cfg = cfg["graph_output"]
    graph_cfg = cfg.get("graph_sampling", {})

    root_dir = ds_cfg["root_dir"]
    out_dir = out_cfg["sample_output_dir"]
    sample_dir = out_cfg["sampled_data_dir"]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    dataset = ScanNetGraphDataset(
        root_dir=root_dir,
        num_frames=ds_cfg["num_frames"],
        num_samples=ds_cfg["num_samples"],
        graph_cache=graph_cfg.get("graph_cache"),
        min_overlap=graph_cfg["min_overlap"],
        max_overlap=graph_cfg["max_overlap"],
        overlap_sample_step=graph_cfg["overlap_sample_step"],
        depth_tolerance=graph_cfg["depth_tolerance"],
        max_frame_gap=graph_cfg.get("max_frame_gap", 50),
        max_scenes=ds_cfg["max_scenes"],
    )

    print(f"Loaded {len(dataset.scene_data)} scenes")
    print(f"Dataset length (virtual): {len(dataset)}")
    print("Sampler type            : graph")
    print(f"Fixed seq_len this run  : {dataset.seq_len}")
    print(f"Min overlap             : {dataset.min_overlap}")
    print(f"Max overlap             : {dataset.max_overlap}")
    print(f"Overlap sample step     : {dataset.overlap_sample_step}")
    print(f"Depth tolerance         : {dataset.depth_tolerance}")
    print(f"Max frame gap           : {dataset.max_frame_gap}")
    print(f"Saving visualisations to: {out_dir}/")
    print(f"Saving sampled frames to: {sample_dir}/\n")

    loader = DataLoader(
        dataset,
        batch_size=out_cfg["batch_size"],
        num_workers=out_cfg["num_workers"],
        shuffle=False,
    )

    total_batches = len(loader)
    milestone = max(1, total_batches // 4)
    print(f"Sampling {total_batches} batches — progress reported every {milestone} batches.\n")

    for i, batch in enumerate(loader):

        images = batch["images"]
        depths = batch["depths"]
        poses = batch["poses"]
        scenes = batch["scene"]
        all_fids = batch["frame_ids"]

        print(f"\nBatch {i + 1}:")
        print(f"  scenes : {list(scenes)}")

        for b, scene_name in enumerate(scenes):
            num_seq_frames = images.shape[1]
            fids = [all_fids[f][b] for f in range(num_seq_frames)]

            seq_dir = os.path.join(sample_dir, f"batch{i+1}", f"sample{b+1}", scene_name)
            rgb_raw_dir = os.path.join(seq_dir, "color")
            dep_raw_dir = os.path.join(seq_dir, "depth")
            pose_dir = os.path.join(seq_dir, "pose")
            intr_dir = os.path.join(seq_dir, "intrinsic")
            os.makedirs(rgb_raw_dir, exist_ok=True)
            os.makedirs(dep_raw_dir, exist_ok=True)
            os.makedirs(pose_dir, exist_ok=True)
            os.makedirs(intr_dir, exist_ok=True)

            for intr_filename, intr_dst in [
                ("intrinsic_color.txt", os.path.join(intr_dir, "intrinsic_color.txt")),
                ("intrinsic_depth.txt", os.path.join(intr_dir, "intrinsic_depth.txt")),
            ]:
                src = os.path.join(root_dir, scene_name, "intrinsic", intr_filename)
                if os.path.isfile(src):
                    shutil.copy2(src, intr_dst)
                else:
                    print(f"  warning: could not find {intr_filename} for scene {scene_name}")

            for f in range(num_seq_frames):
                fid = fids[f]
                torchvision.utils.save_image(
                    images[b, f],
                    os.path.join(rgb_raw_dir, f"{fid}.jpg"),
                )
                np.save(os.path.join(dep_raw_dir, f"{fid}.npy"), depths[b, f].numpy())
                np.savetxt(os.path.join(pose_dir, f"{fid}.txt"), poses[b, f].numpy())

            rgb_frames = images[b]
            rgb_grid = torchvision.utils.make_grid(rgb_frames, nrow=num_seq_frames, padding=4)
            rgb_path = os.path.join(out_dir, f"batch{i+1}_sample{b+1}_{scene_name}_rgb.png")
            torchvision.utils.save_image(rgb_grid, rgb_path)

            dep_frames = depths[b]
            dep_min, dep_max = dep_frames.min(), dep_frames.max()
            dep_norm = (dep_frames - dep_min) / (dep_max - dep_min + 1e-6)
            dep_rgb = dep_norm.repeat(1, 3, 1, 1)
            dep_grid = torchvision.utils.make_grid(dep_rgb, nrow=num_seq_frames, padding=4)
            dep_path = os.path.join(out_dir, f"batch{i+1}_sample{b+1}_{scene_name}_depth.png")
            torchvision.utils.save_image(dep_grid, dep_path)

            print(f"  saved frames     : {seq_dir}/")
            print(f"  saved rgb grid   : {rgb_path}")
            print(f"  saved depth grid : {dep_path}")

        if (i + 1) % milestone == 0 or (i + 1) == total_batches:
            print(f"\nProgress: {i + 1}/{total_batches} batches complete ({(i + 1) * 100 // total_batches}%)")

    print("\nDone.")