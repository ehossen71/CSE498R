# Optimal Transport with POT

A working guide for solving optimal transport problems using Python. This project demonstrates practical implementations of exact and approximate optimal transport algorithms.

## What This Does

This project shows:
- Exact optimal transport using Earth Mover's Distance (EMD)
- Approximate optimal transport via Sinkhorn-Knopp algorithm
- How to compute Wasserstein distances between distributions
- Visualization of transport plans as heatmaps

## How It Works

### Transport Matrix (P)
The transport matrix P[i,j] tells you how much mass moves from source point i to target point j. That's the core solution to the optimal transport problem.

**Key constraints:**
- Each row sums to the source mass: P @ 1 = a
- Each column sums to the target mass: P.T @ 1 = b
- Everything is non-negative: P[i,j] ≥ 0

### Wasserstein Distance
The Wasserstein distance measures the minimum cost to transform one distribution into another. It's defined as:

```
W(a, b) = min_{P} ⟨P, M⟩ = min_P Σᵢⱼ P[i,j] × M[i,j]
```

Here, M is the cost matrix. The lower the Wasserstein distance, the more similar your distributions are.

## Getting Started

### 1. Set up dependencies

```bash
# Create a virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt
```

### 2. Optional: GPU acceleration with PyTorch

If you want to run on GPU:

```bash
pip install torch  # Get it from pytorch.org for your setup
```

Then uncomment the GPU lines in the script.

## Running the Script

```bash
python optimal_transport_demo.py
```

This will:
1. Generate random distributions and point clouds
2. Compute exact optimal transport using `ot.emd()`
3. Compute regularized OT using `ot.sinkhorn()`
4. Print detailed comparison statistics:
   - Transport costs
   - Wasserstein distances
   - Sparsity of solutions
   - Solution differences
5. Create a visualization showing:
   - Exact OT transport matrix (heatmap)
   - Sinkhorn OT transport matrix (heatmap)
   - Cost matrix visualization
   - Source and target distributions

## Code Structure

### Main Components

| Function | Purpose |
|----------|---------|
| `generate_random_distribution()` | Create random normalized histograms |
| `compute_euclidean_cost_matrix()` | Build cost matrix from point distances |
| `compute_exact_optimal_transport()` | Solve OT using LP (ot.emd) |
| `compute_wasserstein_distance()` | Calculate Wasserstein distance |
| `compute_sinkhorn_transport()` | Solve regularized OT (ot.sinkhorn) |
| `visualize_transport_plans()` | Plot results as heatmaps |
| `print_comparison_statistics()` | Display detailed metrics |

## Comparing Exact OT vs Sinkhorn

### Exact OT (`ot.emd()`)
Pros:
- Gives you the true optimal solution
- Creates sparse transport matrices (lots of zeros)
- Exact Wasserstein distance

Cons:
- Slower for large-scale problems (runs an LP solver)
- Needs an LP solver installed
- Can be numerically less stable

### Sinkhorn OT (`ot.sinkhorn()`)
Pros:
- Much faster (Sinkhorn iterations, no LP solver)
- Can run on GPUs
- More numerically stable
- Scales well

Cons:
- Slightly suboptimal (entropy regularization trade-off)
- Denser solutions (more nonzero entries)
- Needs tuning of regularization parameter

## Example Results

Here's what typical output looks like:

```
Metric                             Exact OT    Sinkhorn OT
Transport Cost                      2.345678      2.412456
Wasserstein Distance                2.345678      2.412456
Sparsity (% zeros)                   75.00%        20.00%
L2 Difference                        0.123456
Relative Cost Difference             2.85%
```

## Tweaking Things

### Adjust problem size and parameters:
```python
n_source = 20        # Increase source points
n_target = 15        # Increase target points
reg_param = 0.05     # Lower = more exact, higher = smoother
```

### Try different distance metrics:
```python
# Instead of Euclidean
from scipy.spatial.distance import cdist

# Use Manhattan distance
M = cdist(X, Y, metric='cityblock')

# Or cosine distance
M = cdist(X, Y, metric='cosine')
```

### Run on GPU with PyTorch:
```python
import torch

a_gpu = torch.tensor(a).cuda()
b_gpu = torch.tensor(b).cuda()
M_gpu = torch.tensor(M).cuda()

P = ot.emd_1d(a_gpu, b_gpu, M_gpu)  # Now on GPU
```

## Useful Resources

- [POT Documentation](https://pythonot.github.io/)
- [Wasserstein Metric on Wikipedia](https://en.wikipedia.org/wiki/Wasserstein_metric)
- [Sinkhorn's Theorem](https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem)
- Cuturi (2013): "Sinkhorn Distances: Lightspeed Optimal Transport"

## Troubleshooting

### POT not found?
```
pip install pot
```

### LP solver not found?
POT uses scipy by default, but you can install CVXOPT for better performance:
```
pip install cvxopt
```

### Visualization not showing?
On headless servers, save to a file instead by specifying `save_path` in the visualization function, or use:
```python
import matplotlib
matplotlib.use('Agg')
```

### Too slow?
- Use Sinkhorn instead of exact OT for large datasets
- Reduce the problem size
- Enable GPU acceleration

## Quick Tips

1. **Start small** - Use n_source=5, n_target=5 to learn how it works
2. **Always visualize** - Plot the transport matrix to understand what's happening
3. **Check the math** - Verify `P @ 1 ≈ a` and `P.T @ 1 ≈ b`
4. **Pick the right cost** - Make sure your cost matrix M represents what you actually want to measure
5. **Tune Sinkhorn** - Start with reg=0.1, then adjust if needed

## Where to Go From Here

- Experiment with different cost matrices beyond Euclidean distance
- Try it on real data (image distributions, document similarities, etc.)
- Look into more advanced techniques: sliced OT, Gromov-Wasserstein distance
- Use OT in machine learning pipelines (barycenters, alignment)

---

Feel free to modify and experiment!
