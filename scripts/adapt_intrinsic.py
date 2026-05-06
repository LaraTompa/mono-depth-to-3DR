import os
import glob
import numpy as np

# Original resolution (ScanNet RGB images before resize in temporal_sampling.py)
ORIG_W, ORIG_H = 1296, 968

# Target resolution (after resize in temporal_sampling.py)
IMAGE_W, IMAGE_H = 640, 480

# Compute scale factors
SCALE_X = IMAGE_W / ORIG_W
SCALE_Y = IMAGE_H / ORIG_H

def scale_intrinsics(K_orig):
    """Scale intrinsics matrix from original resolution to resized resolution."""
    K_scaled = K_orig.copy()
    K_scaled[0, 0] *= SCALE_X  # fx
    K_scaled[1, 1] *= SCALE_Y  # fy
    K_scaled[0, 2] *= SCALE_X  # cx
    K_scaled[1, 2] *= SCALE_Y  # cy
    return K_scaled

def main():
    sampled_data_dir = "datasets/sampled_data"
    batch_dirs = sorted(glob.glob(os.path.join(sampled_data_dir, "batch*")))
    
    total_processed = 0
    total_failed = 0
    
    print(f"Scaling intrinsics from {ORIG_W}×{ORIG_H} → {IMAGE_W}×{IMAGE_H}")
    print(f"Scale factors: X={SCALE_X:.6f}, Y={SCALE_Y:.6f}\n")
    
    for batch_dir in batch_dirs:
        batch_name = os.path.basename(batch_dir)
        sample_dirs = sorted(glob.glob(os.path.join(batch_dir, "sample*")))
        
        for sample_dir in sample_dirs:
            sample_name = os.path.basename(sample_dir)
            scene_dirs = [d for d in glob.glob(os.path.join(sample_dir, "*"))
                         if os.path.isdir(d) and os.path.basename(d).startswith("scene")]
            
            for scene_dir in scene_dirs:
                scene_name = os.path.basename(scene_dir)
                intrinsics_path = os.path.join(scene_dir, "intrinsic", "intrinsic_color.txt")
                
                if not os.path.isfile(intrinsics_path):
                    total_failed += 1
                    continue
                
                try:
                    # Load original intrinsics
                    K_orig = np.loadtxt(intrinsics_path)
                    
                    # Scale
                    K_scaled = scale_intrinsics(K_orig)
                    
                    # Save back (overwrite)
                    np.savetxt(intrinsics_path, K_scaled, fmt='%.6f')
                    
                    total_processed += 1
                    
                    if total_processed <= 3 or total_processed % 100 == 0:
                        print(f"[{total_processed}] {batch_name}/{sample_name}/{scene_name}")
                        print(f"  fx: {K_orig[0,0]:.2f} → {K_scaled[0,0]:.2f}")
                        print(f"  fy: {K_orig[1,1]:.2f} → {K_scaled[1,1]:.2f}")
                        print(f"  cx: {K_orig[0,2]:.2f} → {K_scaled[0,2]:.2f}")
                        print(f"  cy: {K_orig[1,2]:.2f} → {K_scaled[1,2]:.2f}\n")
                
                except Exception as e:
                    print(f"✗ Failed {batch_name}/{sample_name}/{scene_name}: {e}")
                    total_failed += 1
    
    print("=" * 60)
    print(f"✓ Scaled {total_processed} intrinsics files")
    print(f"✗ Failed/missing: {total_failed}")
    print("=" * 60)

if __name__ == "__main__":
    main()