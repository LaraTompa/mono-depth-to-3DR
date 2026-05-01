# mono-depth-to-3DR
The idea of this project is with the given predicted monocular depth predictions, design and train a lightweight network to align these monocular depths and predict camera parameters to ensure multiview consistency.

## Setup

Create and activate a conda environment named `mono-depth-3dr`:

```bash
conda create -n mono-depth-3dr python=3.11 -y
conda activate mono-depth-3dr
```

Install the dependencies:

```bash
pip install -r requirements.txt
```
