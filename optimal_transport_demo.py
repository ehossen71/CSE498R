"""
Optimal Transport with POT (Python Optimal Transport) Library


This script demonstrates how to solve optimal transport problems between
two discrete distributions using the POT library.

Key Concepts:
- Transport Matrix: A 2D matrix P where P[i,j] represents the amount of
  mass transported from source point i to target point j.
- Wasserstein Distance: A metric measuring the "cost" of optimally moving
  one distribution to match another (minimizes total transport cost).

Author: Developer
"""

import numpy as np
import matplotlib.pyplot as plt
import ot

# Optional: GPU support (uncomment if you want to use PyTorch)
# import torch
# HAS_TORCH = True
HAS_TORCH = False


# UTILITY FUNCTIONS

def generate_random_distribution(n_points, seed=42, use_torch=False):
    """
    Generate a random discrete distribution (histogram) that sums to 1.
    
    Parameters:
    n_points : int
        Number of points in the distribution
    seed : int
        Random seed for reproducibility
    use_torch : bool
        If True, returns a PyTorch tensor (requires PyTorch installed)
    
    Returns:
    np.ndarray or torch.Tensor
        Distribution of shape (n_points,) summing to 1.0
    """
    np.random.seed(seed)
    
    # Generate random values and normalize to sum to 1
    dist = np.random.rand(n_points)
    dist = dist / np.sum(dist)
    
    if use_torch and HAS_TORCH:
        import torch
        dist = torch.from_numpy(dist).float()
    
    return dist


def compute_euclidean_cost_matrix(X, Y=None):
    """
    Compute Euclidean distance-based cost matrix between two sets of points.
    
    The cost matrix C[i,j] = ||X[i] - Y[j]||_2 represents the cost of
    transporting mass from source point i to target point j.
    
    Parameters:
    X : np.ndarray
        Source points of shape (n_source, n_features)
    Y : np.ndarray, optional
        Target points of shape (n_target, n_features). If None, uses X.
    
    Returns:
    np.ndarray
        Cost matrix of shape (n_source, n_target)
    """
    if Y is None:
        Y = X
    
    # Compute pairwise Euclidean distances using scipy's cdist for efficiency
    from scipy.spatial.distance import cdist
    M = cdist(X, Y, metric='euclidean')
    
    return M


def compute_exact_optimal_transport(a, b, M):
    """
    Compute exact optimal transport solution using the Earth Mover's Distance.
    
    This solves the linear programming problem:
        min <P, M>  (minimize total transport cost)
    subject to:
        P @ 1 = a  (row constraints: source mass conservation)
        P.T @ 1 = b  (column constraints: target mass conservation)
        P >= 0
    
    The solution P is the optimal transport matrix.
    
    Parameters:
    a : np.ndarray
        Source distribution (histogram), shape (n_source,)
    b : np.ndarray
        Target distribution (histogram), shape (n_target,)
    M : np.ndarray
        Cost matrix, shape (n_source, n_target)
    
    Returns:
    np.ndarray
        Optimal transport matrix P, shape (n_source, n_target)
    """
    # ot.emd computes the exact solution using linear programming
    P = ot.emd(a, b, M)
    
    return P


def compute_wasserstein_distance(a, b, M):
    """
    Compute the Wasserstein distance between two distributions.
    
    The Wasserstein distance is the minimum total cost of transporting one
    distribution to match another. Mathematically:
        W(a, b) = min_{P} <P, M>
    
    This is equivalent to the value of the linear program solved by emd().
    ot.emd2 is more efficient than computing P then summing P*M.
    
    Parameters:
    a : np.ndarray
        Source distribution
    b : np.ndarray
        Target distribution
    M : np.ndarray
        Cost matrix
    
    Returns:
    float
        Wasserstein distance
    """
    # ot.emd2 returns the transport cost directly (faster than emd)
    wasserstein_dist = ot.emd2(a, b, M)
    
    return wasserstein_dist


def compute_sinkhorn_transport(a, b, M, reg=0.1, max_iter=100):
    """
    Compute regularized optimal transport using the Sinkhorn algorithm.
    
    Unlike exact OT, Sinkhorn adds an entropy regularization term:
        min <P, M> + reg * KL(P || ab^T)
    
    Benefits:
    - Faster computation (no LP solver needed)
    - Smoother transport matrices (entropy regularization)
    - Better for very large-scale problems
    
    Trade-off: Slightly less optimal than exact OT, but often acceptable.
    
    Parameters:
    a : np.ndarray
        Source distribution
    b : np.ndarray
        Target distribution
    M : np.ndarray
        Cost matrix
    reg : float
        Regularization parameter (entropy). Higher = smoother but less optimal.
        Typical range: [0.01, 1.0]
    max_iter : int
        Maximum Sinkhorn iterations
    
    Returns:
    np.ndarray
        Regularized transport matrix
    float
        Transport cost <P, M>
    """
    # ot.sinkhorn computes regularized OT using Sinkhorn-Knopp algorithm
    P = ot.sinkhorn(a, b, M, reg=reg, numItermax=max_iter)
    
    # Compute the transport cost
    cost = np.sum(P * M)
    
    return P, cost


def visualize_transport_plans(P_exact, P_sinkhorn, a, b, M, 
                              save_path=None):
    """
    Visualize optimal transport matrices and cost comparison.
    
    Parameters:
    P_exact : np.ndarray
        Exact transport matrix
    P_sinkhorn : np.ndarray
        Sinkhorn transport matrix
    a : np.ndarray
        Source distribution
    b : np.ndarray
        Target distribution
    M : np.ndarray
        Cost matrix
    save_path : str, optional
        Path to save figure. If None, displays on screen.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Exact OT Transport Matrix
    im1 = axes[0, 0].imshow(P_exact, cmap='YlOrRd', aspect='auto')
    axes[0, 0].set_title('Exact OT Transport Matrix P', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Target points')
    axes[0, 0].set_ylabel('Source points')
    plt.colorbar(im1, ax=axes[0, 0], label='Mass transported')
    
    # Plot 2: Sinkhorn OT Transport Matrix
    im2 = axes[0, 1].imshow(P_sinkhorn, cmap='YlOrRd', aspect='auto')
    axes[0, 1].set_title('Sinkhorn OT Transport Matrix P', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Target points')
    axes[0, 1].set_ylabel('Source points')
    plt.colorbar(im2, ax=axes[0, 1], label='Mass transported')
    
    # Plot 3: Cost Matrix
    im3 = axes[1, 0].imshow(M, cmap='Blues', aspect='auto')
    axes[1, 0].set_title('Cost Matrix M (Euclidean distances)', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Target points')
    axes[1, 0].set_ylabel('Source points')
    plt.colorbar(im3, ax=axes[1, 0], label='Distance cost')
    
    # Plot 4: Distribution and Statistics
    axes[1, 1].bar(np.arange(len(a)), a, alpha=0.6, label='Source (a)', width=0.4)
    axes[1, 1].bar(np.arange(len(b)) + 0.4, b, alpha=0.6, label='Target (b)', width=0.4)
    axes[1, 1].set_title('Source and Target Distributions', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Point index')
    axes[1, 1].set_ylabel('Mass')
    axes[1, 1].legend()
    axes[1, 1].set_xticks(np.arange(max(len(a), len(b))))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()


def print_comparison_statistics(P_exact, P_sinkhorn, M, wasserstein_exact, 
                                wasserstein_sinkhorn):
    """
    Print detailed statistics comparing exact OT and Sinkhorn OT.
    
    Parameters:
    P_exact : np.ndarray
        Exact transport matrix
    P_sinkhorn : np.ndarray
        Sinkhorn transport matrix
    M : np.ndarray
        Cost matrix
    wasserstein_exact : float
        Exact Wasserstein distance
    wasserstein_sinkhorn : float
        Sinkhorn transport cost
    """
    print("\nOPTIMAL TRANSPORT COMPARISON: EXACT vs SINKHORN")
    print(f"\n{'Metric':<35} {'Exact OT':<15} {'Sinkhorn OT':<15}")
    
    # Transport cost
    cost_exact = np.sum(P_exact * M)
    cost_sinkhorn = np.sum(P_sinkhorn * M)
    print(f"{'Transport Cost':<35} {cost_exact:>14.6f} {cost_sinkhorn:>14.6f}")
    
    # Wasserstein distance
    print(f"{'Wasserstein Distance':<35} {wasserstein_exact:>14.6f} {wasserstein_sinkhorn:>14.6f}")
    
    # Sparsity (fraction of zero entries)
    sparsity_exact = np.sum(P_exact < 1e-10) / P_exact.size * 100
    sparsity_sinkhorn = np.sum(P_sinkhorn < 1e-10) / P_sinkhorn.size * 100
    print(f"{'Sparsity (% zeros)':<35} {sparsity_exact:>13.2f}% {sparsity_sinkhorn:>13.2f}%")
    
    # Matrix norm
    norm_exact = np.linalg.norm(P_exact)
    norm_sinkhorn = np.linalg.norm(P_sinkhorn)
    print(f"{'Matrix Frobenius Norm':<35} {norm_exact:>14.6f} {norm_sinkhorn:>14.6f}")
    
    # Difference between solutions
    diff = np.linalg.norm(P_exact - P_sinkhorn)
    print(f"{'L2 Difference |P_exact - P_sink|':<35} {diff:>14.6f}")
    
    # Relative cost difference
    rel_diff = abs(cost_exact - cost_sinkhorn) / cost_exact * 100
    print(f"{'Relative Cost Difference':<35} {rel_diff:>13.2f}%")
    
    print("\nINTERPRETATION:")
    print("- Lower Transport Cost: More efficient transport plan")
    print("- Higher Sparsity: Matrix P has more zero entries (cleaner transport)")
    print("- Lower L2 Difference: Solutions are more similar")
    print("- Sinkhorn is typically faster but slightly suboptimal\n")


# MAIN EXECUTION

def main():
    """
    Main function: Demonstrate optimal transport with POT library.
    """
    
    print("\nOPTIMAL TRANSPORT DEMONSTRATION WITH POT")
    
    # STEP 1: Create random distributions and cost matrix
    
    print("\n[1] Generating random distributions")
    n_source = 10  # Number of source points
    n_target = 8   # Number of target points
    
    # Create two histograms that sum to 1
    a = generate_random_distribution(n_source, seed=42)  # Source distribution
    b = generate_random_distribution(n_target, seed=43)  # Target distribution
    
    print(f"Source distribution a: {n_source} points, sum={np.sum(a):.6f}")
    print(f"Target distribution b: {n_target} points, sum={np.sum(b):.6f}")
    print(f"a = {a[:5]}... (first 5)")
    print(f"b = {b[:5]}... (first 5)")
    
    # STEP 2: Create random point clouds and compute cost matrix
    
    print("\n[2] Creating point clouds and cost matrix")
    
    # Generate random points in 2D space
    X = np.random.randn(n_source, 2) * 2  # Source points
    Y = np.random.randn(n_target, 2) * 2  # Target points
    
    # Compute Euclidean distance-based cost matrix
    M = compute_euclidean_cost_matrix(X, Y)
    
    print(f"Source points shape: {X.shape}")
    print(f"Target points shape: {Y.shape}")
    print(f"Cost matrix shape: {M.shape}")
    print(f"Cost matrix stats: min={M.min():.4f}, mean={M.mean():.4f}, max={M.max():.4f}")
    
    # STEP 3: Compute exact optimal transport
    
    print("\n[3] Computing exact optimal transport (ot.emd)")
    
    P_exact = compute_exact_optimal_transport(a, b, M)
    wasserstein_exact = compute_wasserstein_distance(a, b, M)
    
    print(f"Transport matrix shape: {P_exact.shape}")
    print(f"Transport matrix sum: {np.sum(P_exact):.6f}")
    print(f"Wasserstein distance: {wasserstein_exact:.6f}")
    print(f"Nonzero entries: {np.sum(P_exact > 1e-10)}")
    
    # STEP 4: Compute Sinkhorn regularized optimal transport
    
    print("\n[4] Computing Sinkhorn regularized OT (ot.sinkhorn)")
    
    reg_param = 0.1  # Entropy regularization parameter
    P_sinkhorn, cost_sinkhorn = compute_sinkhorn_transport(a, b, M, reg=reg_param, max_iter=100)
    
    print(f"Regularization parameter: {reg_param}")
    print(f"Transport matrix shape: {P_sinkhorn.shape}")
    print(f"Transport matrix sum: {np.sum(P_sinkhorn):.6f}")
    print(f"Transport cost: {cost_sinkhorn:.6f}")
    print(f"Nonzero entries: {np.sum(P_sinkhorn > 1e-10)}")
    
    # STEP 5: Print comparison statistics
    
    print_comparison_statistics(P_exact, P_sinkhorn, M, wasserstein_exact, cost_sinkhorn)
    
    # STEP 6: Visualize results
    
    print("\n[5] Creating visualizations")
    save_path = "optimal_transport_visualization.png"
    visualize_transport_plans(P_exact, P_sinkhorn, a, b, M, save_path=save_path)
    
    print("Demonstration complete!")
    
    return {
        'a': a, 'b': b, 'X': X, 'Y': Y, 'M': M,
        'P_exact': P_exact, 'P_sinkhorn': P_sinkhorn,
        'wasserstein_exact': wasserstein_exact, 'cost_sinkhorn': cost_sinkhorn
    }


if __name__ == "__main__":
    results = main()
