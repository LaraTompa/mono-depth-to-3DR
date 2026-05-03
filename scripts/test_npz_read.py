import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_depth_heatmap(depth_map):
    plt.imshow(depth_map, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Depth Heatmap')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process depth maps.')
    parser.add_argument('input_file', type=str, help='Path to the input .npz file')
    args = parser.parse_args()

    data = np.load(args.input_file)
    for key in ["depth", "pred", "prediction", "arr_0"]:
            if key in data:
                depth = data[key]
                break
            else:
                # fallback: take first array
                depth = data[list(data.keys())[0]]

    plot_depth_heatmap(depth)