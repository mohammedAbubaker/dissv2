import torch

import os
import glob
import gc
import math

from plyfile import PlyData, PlyElement

# from torch.utils.checkpoint import checkpoint
import pandas as pd
import numpy as np

# import scipy.sparse.linalg as spla
# import scipy.sparse as sp
# from tqdm import tqdm
import torch as to
# import threading

from torch_kdtree import build_kd_tree

device = "cuda"
N = None
input_opacities = None
L = None
A_sparse = None
df = None

gc.collect()
to.cuda.empty_cache()
# --- Logging and Saving ---


def evaluate_gaussian_density_at_points(points, means, inv_covs, opacities):
    """
    Evaluates density contribution of ALL Gaussians at given query points.

    Args:
        points (torch.Tensor): Query points (batch_P, 3), where batch_P could be batch_R * n_samples.
        means (torch.Tensor): Gaussian means (1, N, 3) - requires grad.
        inv_covs (torch.Tensor): Inverse covariances (1, N, 3, 3) - detached.
        opacities (torch.Tensor): Raw opacities (1, N, 1) - detached, sigmoid applied internally.

    Returns:
        torch.Tensor: Density contributions (batch_P, N).
    """
    batch_P = points.shape[0]
    N = means.shape[1]

    # Expand points for broadcasting: (batch_P, 1, 3)
    points_exp = points.unsqueeze(1)

    # Sigmoid activation for opacity
    activated_opacities = torch.sigmoid(opacities)  # (1, N, 1)

    # Mahalanobis distance calculation
    dist = points_exp - means  # (batch_P, N, 3)
    dist_vec = dist.unsqueeze(2)  # (batch_P, N, 1, 3)
    dist_vec_T = dist.unsqueeze(3)  # (batch_P, N, 3, 1)

    # mahalanobis_sq = dist_vec @ inv_covs @ dist_vec_T
    # inv_covs is (1, N, 3, 3), broadcasting works
    mahalanobis_sq = (
        torch.matmul(torch.matmul(dist_vec, inv_covs), dist_vec_T)
        .squeeze(-1)
        .squeeze(-1)
    )  # (batch_P, N)
    mahalanobis_sq = torch.clamp(mahalanobis_sq, min=0.0)  # Stability

    # Gaussian exponent
    exponent = torch.exp(-0.5 * mahalanobis_sq)  # (batch_P, N)

    # Final density contribution = opacity * exponent
    evaluations = (
        activated_opacities.squeeze(-1) * exponent
    )  # (1,N) * (batch_P, N) -> (batch_P, N)

    return evaluations


def nerf_style_render_rays(
    ray_oris_batch,
    ray_dirs_batch,  # (batch_R, 3)
    means,
    inv_covs,
    opacities,  # Gaussian params (means req grad)
    n_samples,  # Number of samples per ray
    tn,
    tf,  # Near and far bounds
    perturb=True,  # Use stratified sampling if True
):
    """
    Performs NeRF-style volumetric rendering for a batch of rays using Gaussian densities.

    Args: See above.

    Returns:
        torch.Tensor: Expected depth for each ray (batch_R,).
    """
    batch_R = ray_oris_batch.shape[0]
    if batch_R == 0:
        return torch.empty(0, device=ray_oris_batch.device)

    # 1. Sample points along rays
    t_vals = torch.linspace(0.0, 1.0, n_samples, device=device)  # (n_samples,)
    depth_vals = (
        tn * (1.0 - t_vals) + tf * t_vals
    )  # Linearly space in depth (n_samples,)

    if perturb:
        # Stratified sampling: add random offset within each interval
        mids = 0.5 * (depth_vals[1:] + depth_vals[:-1])
        upper = torch.cat([mids, depth_vals[-1:]], -1)
        lower = torch.cat([depth_vals[:1], mids], -1)
        # (batch_R, n_samples)
        t_rand = torch.rand(batch_R, n_samples, device=device)
        depth_vals = (
            lower.unsqueeze(0) + (upper - lower).unsqueeze(0) * t_rand
        )  # (batch_R, n_samples)
    else:
        depth_vals = depth_vals.unsqueeze(0).expand(
            batch_R, -1)  # (batch_R, n_samples)

    # Calculate sample points in 3D: (batch_R, n_samples, 3)
    # ro shape: (batch_R, 1, 3)
    # depth_vals shape: (batch_R, n_samples, 1)
    # rd shape: (batch_R, 1, 3)
    sample_points = ray_oris_batch.unsqueeze(1) + depth_vals.unsqueeze(
        -1
    ) * ray_dirs_batch.unsqueeze(1)

    # Reshape for density evaluation: (batch_R * n_samples, 3)
    sample_points_flat = sample_points.view(-1, 3)

    # 2. Evaluate density at sample points
    # means: (1, N, 3), inv_covs: (1, N, 3, 3), opacities: (1, N, 1)
    # Densities shape: (batch_R * n_samples, N)
    densities_flat = evaluate_gaussian_density_at_points(
        sample_points_flat,
        means.unsqueeze(0),  # Add batch dim for broadcasting
        inv_covs.unsqueeze(0),  # Add batch dim for broadcasting
        opacities.unsqueeze(0),  # Add batch dim for broadcasting
    )

    # Sum density contributions from all Gaussians for each point
    # sigma_flat shape: (batch_R * n_samples,)
    sigma_flat = torch.sum(densities_flat, dim=-1)
    # Add a small amount of noise or minimum density? Optional.
    # sigma_flat = torch.relu(sigma_flat) # Ensure non-negative

    # Reshape sigma back: (batch_R, n_samples)
    sigma = sigma_flat.view(batch_R, n_samples)

    # 3. Volumetric Integration (Numerical Quadrature)
    # Calculate distances between adjacent samples
    deltas = depth_vals[:, 1:] - depth_vals[:, :-1]  # (batch_R, n_samples-1)
    # Assume final segment extends infinitely or to a large value
    delta_inf = torch.full_like(deltas[:, :1], 1e10)  # (batch_R, 1)
    deltas = torch.cat([deltas, delta_inf], dim=-1)  # (batch_R, n_samples)

    # Calculate alpha (opacity) for each segment
    alpha = 1.0 - torch.exp(-sigma * deltas)  # (batch_R, n_samples)
    alpha = torch.clamp(alpha, 0.0, 1.0)  # Ensure valid alpha

    # Calculate transmittance T_i = product(1 - alpha_j + epsilon) for j < i
    # Add epsilon for numerical stability during cumprod backward pass
    transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    # Shift transmittance: T_0=1, T_1=(1-a0), T_2=(1-a0)(1-a1), ...
    transmittance = torch.cat(
        [torch.ones(batch_R, 1, device=device), transmittance[:, :-1]], dim=-1
    )  # (batch_R, n_samples)

    # Calculate weights for each sample point
    weights = transmittance * alpha  # (batch_R, n_samples)

    # 4. Calculate Expected Depth
    # Use midpoint depth values for integration accuracy
    depth_vals_mid = 0.5 * (depth_vals[:, 1:] + depth_vals[:, :-1])
    depth_vals_mid = torch.cat(
        [depth_vals[:, :1], depth_vals_mid], dim=-1
    )  # Add first depth

    expected_depth = torch.sum(weights * depth_vals_mid, dim=-1)  # (batch_R,)

    # Add contribution from background if needed (e.g., if ray doesn't hit anything)
    # weight_background = transmittance[:, -1]
    # expected_depth += weight_background * tf # Or some other background depth

    return expected_depth  # (batch_R,)


def surface_laplacian_criterion(points_on_surface, k=8, chunk_size=2048):
    """
    Computes Laplacian smoothing loss directly on a set of 3D points.

    Args:
        points_on_surface (torch.Tensor): Tensor of shape (N, 3) representing points in 3D space.
        k (int): Number of nearest neighbors to consider for smoothing.
        chunk_size (int): Chunk size for KNN calculation.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    if points_on_surface.shape[0] < k + 1:
        print(
            f"Warning: Not enough points ({points_on_surface.shape[0]}) for k={k}. Skipping loss."
        )
        return to.tensor(
            0.0, device=points_on_surface.device, requires_grad=True
        )  # Return zero loss but keep grad

    # Find k nearest neighbors in 3D space for each point
    # Note: Using points_on_surface for both query and source
    topk_distances, topk_indices = get_knn_distances_chunked(
        points_on_surface.detach(),  # Use detach for KNN search if grads not needed here
        points_on_surface.detach(),
        k=k + 1,  # Find k+1 to include self
        chunk_size=chunk_size,
    )

    # Exclude self (usually the closest point with distance 0)
    topk_indices = topk_indices[:, 1:]  # Shape: (N, k)
    # Optional: Use distances for weighting, similar to your original criterion
    # topk_distances = topk_distances[:, 1:]
    # weights = torch.exp(-(topk_distances**2)) # Example Gaussian weights
    # weights = weights / weights.sum(dim=1, keepdim=True) # Normalize

    # Gather the 3D positions of the neighbors
    neighbor_positions = points_on_surface[topk_indices]  # Shape: (N, k, 3)

    # Compute the average position of neighbors (simple average)
    # avg_neighbor_position = torch.sum(weights.unsqueeze(-1) * neighbor_positions, dim=1) # Weighted average
    avg_neighbor_position = to.mean(neighbor_positions, dim=1)  # Shape: (N, 3)

    # Compute the squared difference (Laplacian vector)
    laplacian_vector = points_on_surface - \
        avg_neighbor_position  # Shape: (N, 3)

    # Loss is the sum of squared magnitudes of the Laplacian vectors
    # Using mean instead of sum can make it less sensitive to the number of points
    loss = to.mean(to.sum(laplacian_vector**2, dim=1))
    # Or using sum:
    # loss = torch.sum(laplacian_vector**2)

    return loss


def get_knn_angular_chunked(query_points, source_points, k=16, chunk_size=1024):
    """
    Calculates distances to the k-nearest neighbors without computing the full distance matrix
    by processing query points in chunks.

    Args:
        query_points (torch.Tensor): Points for which to find neighbors (shape: N, D).
        source_points (torch.Tensor): Points to search neighbors from (shape: M, D).
        k (int): Number of nearest neighbors to find.
        chunk_size (int): Number of query points to process in each chunk.

    Returns: (twrpk_distances, topk_indices) - Distances and indices of k-NN.
               topk_distances: (N, k)
               topk_indices: (N, k)  (indices are relative to source_points)
    """
    num_query_points = query_points.shape[0]
    topk_distances = to.zeros(
        (num_query_points, k), dtype=query_points.dtype, device=query_points.device
    )
    topk_indices = to.zeros(
        (num_query_points, k), dtype=to.long, device=query_points.device
    )
    for start_idx in range(0, num_query_points, chunk_size):
        end_idx = min(start_idx + chunk_size, num_query_points)
        query_chunk = query_points[start_idx:end_idx]

        distances = to.matmul(query_chunk, source_points.T)

        # Find top k-nearest neighbors for each point in the chunk
        chunk_topk_distances, chunk_topk_indices = to.topk(
            distances, k=k, dim=1, largest=False
        )
        # chunk_topk_distances: (chunk_size, k)
        # chunk_topk_indices:  (chunk_size, k) (indices are relative to source_points)

        topk_distances[start_idx:end_idx, :] = chunk_topk_distances
        topk_indices[start_idx:end_idx, :] = chunk_topk_indices
    return topk_distances, topk_indices


def compute_weighted_neighbors(ray_oris, depth_values, k=4):
    """
    Computes the weighted average of k-nearest neighbors for each point.
    Returns the weighted neighbor positions.
    """

    # Get k nearest neighbors
    topk_distances, topk_indices = get_knn_distances_chunked(
        ray_oris.detach(), ray_oris.detach(), k=k, chunk_size=2048
    )

    topk_distances = topk_distances[:, 1:]  # Remove first column
    topk_indices = topk_indices[:, 1:]  # Remove first column

    # Compute Gaussian weights and normalize
    weights = to.exp(-(topk_distances**2))
    weights = weights / weights.sum(dim=1, keepdim=True)

    # Gather neighbor positions: shape (N, k, D)
    neighbor_positions = depth_values[topk_indices]  # Potential crash point

    # Compute weighted average of neighbor positions for each point
    weighted_neighbors = to.sum(
        weights.unsqueeze(-1) * neighbor_positions, dim=1)

    return (
        weighted_neighbors,
        topk_indices,
        weights,
    )  # Return weight


def compute_weighted_neighbors_3(ray_oris, depth_values, k=32):
    """
    Computes the weighted average of k-nearest neighbors for each point.
    Returns the weighted neighbor positions.
    """

    # Get k nearest neighbors
    topk_distances, topk_indices = get_knn_angular_chunked(
        ray_oris.detach(), ray_oris.detach(), k=k, chunk_size=2048
    )

    topk_distances = topk_distances[:, 1:]  # Remove first column
    topk_indices = topk_indices[:, 1:]  # Remove first column

    # Compute Gaussian weights and normalize
    weights = to.exp(-(topk_distances**2))
    weights = weights / weights.sum(dim=1, keepdim=True)

    # Gather neighbor positions: shape (N, k, D)
    neighbor_positions = depth_values[topk_indices]  # Potential crash point

    # Compute weighted average of neighbor positions for each point
    weighted_neighbors = to.sum(
        weights.unsqueeze(-1) * neighbor_positions, dim=1)

    return (
        weighted_neighbors,
        topk_indices,
        weights,
    )  # Return weight


def compute_weighted_neighbors_2(positions, k=32):
    """
    Computes the weighted average of k-nearest neighbors for each point.
    Returns the weighted neighbor positions.
    """

    # Get k nearest neighbors
    topk_distances, topk_indices = get_knn_distances_chunked(
        positions, positions, k=k, chunk_size=2048
    )

    topk_distances = topk_distances[:, 1:]  # Remove first column
    topk_indices = topk_indices[:, 1:]  # Remove first column

    # Compute Gaussian weights and normalize
    weights = to.exp(-(topk_distances**2))
    weights = weights / weights.sum(dim=1, keepdim=True)

    # Gather neighbor positions: shape (N, k, D)
    neighbor_positions = to.norm(
        positions[topk_indices])  # Potentia120crash point

    # Compute weighted average of neighbor positions for each point
    weighted_neighbors = to.sum(
        weights.unsqueeze(-1) * neighbor_positions, dim=1)

    return (
        weighted_neighbors,
        topk_indices,
        weights,
    )  # Return weight


def save_model(original_path, model, output_path="out.ply"):
    global df
    original_ply = PlyData.read(original_path)
    original_data = original_ply["vertex"].data

    new_means = model["means"].detach().cpu().numpy()
    new_quaternions = model["quaternions"].detach().cpu().numpy()
    new_scales = model["scales"].detach().cpu().numpy()
    # new_means = model["means"]
    # new_quaternions = model["quaternions"]
    # new_scales = model["scales"]

    df["x"] = new_means[:, 0]
    df["y"] = new_means[:, 1]
    df["z"] = new_means[:, 2]

    df["rot_0"] = new_quaternions[:, 0]
    df["rot_1"] = new_quaternions[:, 1]
    df["rot_2"] = new_quaternions[:, 2]
    df["rot_3"] = new_quaternions[:, 3]

    df["scale_0"] = new_scales[:, 0]
    df["scale_1"] = new_scales[:, 1]
    df["scale_2"] = new_scales[:, 2]

    new_data = df.to_records(index=False)
    ply_element = PlyElement.describe(new_data, "vertex")
    PlyData([ply_element], text=False).write(output_path)


def get_top_16_differentiable_batched(
    means, scales, quaternions, ray_oris, ray_dirs, K=8, gaussian_batch_size=4096
):
    """
    Batched version of the differentiable ray–box intersection.
    Splits the Gaussians into batches and computes t-hit values per batch.
    Finally, the hit values are concatenated and the top K overall are selected.
    """
    num_gauss = means.shape[0]
    all_t_hits = []
    all_idx = []
    for start in range(0, num_gauss, gaussian_batch_size):
        end = min(start + gaussian_batch_size, num_gauss)
        m_batch = means[start:end]
        s_batch = scales[start:end]
        q_batch = quaternions[start:end]
        # Compute bounding boxes for this batch

        min_corners, max_corners = create_bounding_boxes(
            m_batch, s_batch, q_batch)

        # Expand dimensions to broadcast with the rays:
        ro_exp = ray_oris.unsqueeze(1)  # [R, 1, 3]
        rd_exp = ray_dirs.unsqueeze(1)  # [R, 1, 3]
        min_exp = min_corners.unsqueeze(0)  # [1, B, 3]
        max_exp = max_corners.unsqueeze(0)  # [1, B, 3]

        safe_rd = to.where(rd_exp != 0, rd_exp, to.full_like(rd_exp, 1e-8))
        inv_dirs = 1.0 / safe_rd
        t1 = (min_exp - ro_exp) * inv_dirs
        t2 = (max_exp - ro_exp) * inv_dirs
        t_min = to.minimum(t1, t2)
        t_max = to.maximum(t1, t2)
        t_entry = to.max(t_min, dim=2)[0]  # [R, B]
        t_exit = to.min(t_max, dim=2)[0]  # [R, B]
        hit = (t_exit > t_entry) & (t_exit > 0)
        t_hit = to.where(t_entry > 0, t_entry, t_exit)
        t_hit = to.where(hit, t_hit, to.full_like(t_hit, float("inf")))
        all_t_hits.append(t_hit)
        # Save the global gaussian indices for this batch
        batch_idx = (
            to.arange(start, end, device=means.device)
            .unsqueeze(0)
            .expand(ray_oris.shape[0], -1)
        )
        all_idx.append(batch_idx)

    full_t_hits = to.cat(all_t_hits, dim=1)  # shape: [R, total_gauss]
    full_idx = to.cat(all_idx, dim=1)  # shape: [R, total_gauss]
    top_vals, top_idx = to.topk(full_t_hits, k=K, dim=1, largest=False)
    top_gauss_idx = full_idx.gather(1, top_idx)
    return top_gauss_idx


def sample_uniformly_from_boxes(min_corners, max_corners, num_samples):
    """
    Samples points uniformly from a set of axis-aligned bounding boxes,
    weighted by the volume of each box.

    Args:
        min_corners (to.Tensor): Tensor of shape (M, 3) with min corners (x, y, z).
        max_corners (to.Tensor): Tensor of shape (M, 3) with max corners (x, y, z).
        num_samples (int): The total number of points to sample across all boxes.

    Returns:
        to.Tensor: Tensor of shape (num_samples, 3) containing sampled points.
                   Returns an empty tensor if num_samples is 0 or if all boxes
                   have zero volume.
    """
    if num_samples <= 0:
        return to.empty((0, 3), device=min_corners.device, dtype=min_corners.dtype)

    if min_corners.shape[0] == 0:
        # No boxes to sample from
        return to.empty((0, 3), device=min_corners.device, dtype=min_corners.dtype)

    assert min_corners.shape == max_corners.shape
    assert min_corners.ndim == 2 and min_corners.shape[1] == 3
    device = min_corners.device
    dtype = min_corners.dtype

    # 1. Calculate volumes
    dims = max_corners - min_corners
    # Clamp dimensions to be non-negative in case of numerical issues or invalid boxes
    dims = to.clamp(dims, min=0.0)
    volumes = to.prod(dims, dim=1)  # Shape: (M,)

    # 2. Calculate volume probabilities
    total_volume = to.sum(volumes)

    # Handle edge case where total volume is zero or near zero
    if total_volume <= 1e-9:
        print(
            "Warning: Total volume of bounding boxes is near zero. Cannot sample uniformly by volume."
        )
        # Option 1: Return empty
        # return to.empty((0, 3), device=device, dtype=dtype)

        # Option 2: Sample uniformly from the *centers* of the boxes instead
        print("Sampling uniformly from box centers instead.")
        num_boxes = min_corners.shape[0]
        centers = (min_corners + max_corners) / 2.0
        random_indices = to.randint(
            0, num_boxes, (num_samples,), device=device)
        return centers[random_indices]

        # Option 3: Choose one box randomly and sample from it (less uniform)
        # ... implementation depends on desired fallback

    probabilities = volumes / total_volume  # Shape: (M,)

    # Add small epsilon for numerical stability if needed, though usually fine if total_volume > 0
    # probabilities = probabilities + 1e-10
    # probabilities = probabilities / probabilities.sum()

    # 3. Select Box Indices (weighted by volume)
    sampled_box_indices = to.multinomial(
        probabilities, num_samples, replacement=True
    )  # Shape: (num_samples,)

    # 4. Sample Points within Selected Boxes
    # Gather the min/max corners for the chosen boxes
    # Shape: (num_samples, 3)
    selected_min_corners = min_corners[sampled_box_indices]
    # Shape: (num_samples, 3)
    selected_max_corners = max_corners[sampled_box_indices]

    # Generate random scales [0, 1) for each dimension, for each point
    random_scales = to.rand(num_samples, 3, device=device, dtype=dtype)

    # Linearly interpolate between min and max corners using the random scales
    # point = min + scale * (max - min)
    sampled_points = selected_min_corners + random_scales * (
        selected_max_corners - selected_min_corners
    )  # Shape: (num_samples, 3)

    return sampled_points


def lol_diff(means, quaternions, scales):
    n_gaussians = means.shape[0]
    frequency_table = to.zeros(n_gaussians).to(device=device)
    for i in range(10):
        look_at = to.rand(3).to(device=device) - 0.5
        normalised_look_at = look_at / to.norm(look_at)

        ray_ori = normalised_look_at * 100
        ray_dirs = means - ray_ori
        ray_dirs = ray_dirs[to.randint(0, ray_dirs.shape[0], (10240,))]
        ray_oris = to.broadcast_to(ray_ori, (ray_dirs.shape[0], 3))

        indices = get_top_16_differentiable_batched(
            means, scales, quaternions, ray_oris, ray_dirs, 1
        )
        frequency_table[indices] += 1

    total = to.sum(frequency_table)
    normalised_freq = frequency_table / total
    sorted_values, sorted_indices = to.sort(normalised_freq, descending=True)
    cumulative_sum = to.cumsum(sorted_values, dim=0)
    threshold = 0.80
    mask_below_threshold = cumulative_sum < threshold
    original_indices = sorted_indices[mask_below_threshold]
    min_corners, max_corners = create_bounding_boxes(
        means[original_indices], scales[original_indices], quaternions[original_indices]
    )

    samples = sample_uniformly_from_boxes(
        min_corners, max_corners, n_gaussians)

    return samples


def sample_v2(means, quaternions, scales, num_samples=2048):
    # Initialise a frequency table
    frequency_table = to.zeros(means.shape[0]).to(device=device)

    new_ray_oris = []
    for i in range(100):
        look_at = to.rand(3).to(device=device) - 0.5
        normalised_look_at = look_at / to.norm(look_at)

        ray_ori = normalised_look_at * 100
        new_ray_oris.append(ray_ori[None, ...])
        ray_dirs = means - ray_ori
        ray_dirs = ray_dirs[to.randint(0, ray_dirs.shape[0], (num_samples,))]

        ray_oris = to.broadcast_to(ray_ori, (ray_dirs.shape[0], 3))

        with to.no_grad():
            indices = get_top_16_differentiable_batched(
                means,
                scales,
                quaternions,
                ray_oris,
                ray_dirs,
                1,
            )
            frequency_table[indices] += 1

    new_ray_oris = to.cat(new_ray_oris, dim=0)
    total = to.sum(frequency_table)
    normalised_freq = frequency_table / total
    sorted_values, sorted_indices = to.sort(normalised_freq, descending=True)
    cumulative_sum = to.cumsum(sorted_values, dim=0)
    threshold = 0.80
    mask_below_threshold = cumulative_sum < threshold
    original_indices = sorted_indices[mask_below_threshold]

    covariances = get_covariances(
        quaternions[original_indices], scales[original_indices]
    )
    L = to.linalg.cholesky(covariances).to(device=device)
    samples_per_gaussian = 10
    # Sample from a standard normal distribution (mean = 0, std = 1)
    std_normal_samples = to.randn(
        means[original_indices].shape[0], samples_per_gaussian, 3
    ).to(device=device)
    samples = to.bmm(std_normal_samples, L.transpose(1, 2))
    samples += means[original_indices].unsqueeze(1)
    samples = samples.reshape(-1, 3)
    indices = to.randint(0, samples.shape[0], (500_000,))
    return samples[indices]


def sample_v1(means, quaternions, scales, num_samples=1024):
    covariances = get_covariances(quaternions, scales)
    L = to.linalg.cholesky(covariances).to(device=device)
    samples_per_gaussian = 10
    # Sample from a standard normal distribution (mean = 0, std = 1)
    std_normal_samples = to.randn(means.shape[0], samples_per_gaussian, 3).to(
        device=device
    )
    samples = to.bmm(std_normal_samples, L.transpose(1, 2))
    samples += means.unsqueeze(1)
    samples = samples.reshape(-1, 3)
    indices = to.randint(0, samples.shape[0], (num_samples,))
    ray_dirs = samples[indices]
    ray_dirs = ray_dirs / to.norm(ray_dirs, dim=1, keepdim=True)
    ray_oris = ray_dirs * 10
    return ray_oris, -1.0 * ray_dirs


def project_points_to_sphere(points, normals, sphere_center, sphere_radius):
    """
    Projects points along their normals until they intersect with the bounding sphere.

    Args:()
        points (torch.Tensor): Nx3 tensor of points.
        normals (torch.Tensor): Nx3 tensor of normal vectors (assumed to be normalized).
        sphere_center (torch.Tensor): 1x3 tensor representing the center of the sphere.
        sphere_radius (float): Radius of the sphere.

    Returns:
        torch.Tensor: Nx3 tensor of intersection points on the sphere.
    """
    D = points - sphere_center  # Vector from sphere center to points
    d_dot_n = to.sum(D * normals, dim=1)  # Dot product of D and normal
    d_dot_d = to.sum(D * D, dim=1)  # Squared length of D

    # Quadratic formula components
    discriminant = d_dot_n**2 - (d_dot_d - sphere_radius**2)
    sqrt_discriminant = to.sqrt(discriminant)

    # Compute the two possible t values
    t1 = -d_dot_n + sqrt_discriminant
    t2 = -d_dot_n - sqrt_discriminant

    # Choose the positive t (ensuring forward intersection)
    t = to.where(t1 > 0, t1, t2)

    # Compute intersection points on the sphere
    intersection_points = points + t[:, None] * normals
    return intersection_points


def generate_rays_from_bounding_volume(points):
    # Initialise a bounding volume
    lower_corner_x = points[:, 0].min().unsqueeze(0)
    lower_corner_y = points[:, 1].min().unsqueeze(0)
    lower_corner_z = points[:, 2].min().unsqueeze(0)

    upper_corner_x = points[:, 0].max().unsqueeze(0)
    upper_corner_y = points[:, 1].max().unsqueeze(0)
    upper_corner_z = points[:, 2].max().unsqueeze(0)

    projected_points = points.clone()
    projected_points[:, 1] = 2
    inward_normal = to.tensor([0, -1.0, 0]).to(device=device).float()
    inward_normal = inward_normal.unsqueeze(0).expand(points.shape[0], -1)
    return projected_points, inward_normal


def generate_rays_from_points(points):
    centroid = to.mean(points, dim=0)
    radius = 10
    ray_dirs = points - centroid
    ray_dirs = ray_dirs / to.norm(ray_dirs, dim=1, keepdim=True)
    ray_oris = centroid + ray_dirs * radius
    return ray_oris, -1.0 * ray_dirs


def load_gaussians(path):
    global N
    global df
    plyfile = PlyData.read(path)
    # Load in data
    plyfile = PlyData.read(path)
    plydata = plyfile["vertex"].data
    df = pd.DataFrame(plydata)
    #   df = df[df["opacity"] > 0.9]
    print(df.shape)

    means_mask = ["x", "y", "z"]
    quaternions_mask = ["rot_0", "rot_1", "rot_2", "rot_3"]
    scales_mask = ["scale_0", "scale_1", "scale_2"]
    opacities_mask = ["opacity"]

    means = to.tensor(df[means_mask].values).to(device)
    quaternions = to.tensor(df[quaternions_mask].values).to(device)
    scales = to.tensor(df[scales_mask].values).to(device)
    N = means.shape[0]

    global input_opacities
    input_opacities = to.tensor(df[opacities_mask].values).to(device)

    return {
        "n": means.shape[0],
        "means": means,
        "quaternions": quaternions,
        "scales": scales,
        "opacities": input_opacities,
    }


def compute_laplacian(ray_oris, beta=3.9):
    """
    Computes a fully dense Laplacian matrix using d_0^T W d_0.

    Parameters:
    - ray_oris: Tensor of shape (N, 3), representing points.
    - beta: Scaling factor for edge weights.

    Returns:
    - L: Dense Laplacian matrix of shape (N, N).
    """
    N = ray_oris.shape[0]

    # Compute pairwise distances
    distances = to.cdist(ray_oris, ray_oris)  # (N, N)

    # Construct differential matrix d_0 (N x N)
    d_0 = to.eye(N, device=ray_oris.device) - 1 / N  # Centered difference

    # Compute weight matrix W (N x N) using Gaussian decay
    weights = to.exp(-beta * distances**2)
    # Instead of in-place fill_diagonal_, create a mask to zero the diagonal:
    mask = 1 - to.eye(N, device=ray_oris.device)
    weights = weights * mask

    # Compute Laplacian: L = d_0^T W d_0
    L = d_0.T @ weights @ d_0
    return L


def get_laplacian(ray_oris):
    K = 32
    distances = to.cdist(ray_oris, ray_oris)
    neighbour_distances, neighbour_indices = to.topk(
        distances, K, largest=False)

    weights = to.exp(-(neighbour_distances**2 * 1.0))  # Reduced scaling
    N = distances.shape[0]
    W = to.zeros((N, N), device=device)

    rows = to.arange(N).unsqueeze(-1).expand(-1, K)
    W[rows, neighbour_indices] = weights

    D = to.sum(W, dim=1)
    D_sqrt_inv = to.diag(1.0 / to.sqrt(D + 1e-6))  # Normalization

    L = D_sqrt_inv @ (to.diag(D) - W) @ D_sqrt_inv
    return L


def get_laplacian_2(ray_oris):
    K = 4
    distances = to.cdist(ray_oris, ray_oris)
    neighbour_distances, neighbour_indices = to.topk(
        distances, K, largest=False)
    weights = to.exp(-(neighbour_distances**2) * 0.001)
    N = distances.shape[0]
    W = to.zeros((N, N), device=device)
    rows = to.arange(N).unsqueeze(-1).expand(-1, K)
    W[rows, neighbour_indices] = weights
    D = to.sum(W, dim=1)
    L = to.diag(D) - W
    return L


def lq_get_rotation_matrices(quaternions):
    # Normalise quaternions
    quaternions_norm = quaternions / to.norm(quaternions, dim=1, keepdim=True)

    x = quaternions_norm[:, 1]
    y = quaternions_norm[:, 2]
    z = quaternions_norm[:, 3]
    w = quaternions_norm[:, 0]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    n = quaternions.shape[0]
    R = to.empty((n, 3, 3), dtype=quaternions.dtype)

    R[:, 0, 0] = 1 - 2 * (yy + zz)
    R[:, 0, 1] = 2 * (xy - zw)
    R[:, 0, 2] = 2 * (xz + yw)
    R[:, 1, 0] = 2 * (xy + zw)
    R[:, 1, 1] = 1 - 2 * (xx + zz)
    R[:, 1, 2] = 2 * (yz - xw)
    R[:, 2, 0] = 2 * (xz - yw)
    R[:, 2, 1] = 2 * (yz + xw)
    R[:, 2, 2] = 1 - 2 * (xx + yy)

    return R


def get_rotation_matrices(quaternions):
    # Normalise quaternions
    normed_quaternions = quaternions / \
        to.norm(quaternions, dim=1, keepdim=True)

    x = normed_quaternions[:, 1]
    y = normed_quaternions[:, 2]
    z = normed_quaternions[:, 3]
    w = normed_quaternions[:, 0]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    n = quaternions.shape[0]
    R = to.empty((n, 3, 3), dtype=quaternions.dtype)

    R[:, 0, 0] = 1 - 2 * (yy + zz)
    R[:, 0, 1] = 2 * (xy - zw)
    R[:, 0, 2] = 2 * (xz + yw)
    R[:, 1, 0] = 2 * (xy + zw)
    R[:, 1, 1] = 1 - 2 * (xx + zz)
    R[:, 1, 2] = 2 * (yz - xw)
    R[:, 2, 0] = 2 * (xz - yw)
    R[:, 2, 1] = 2 * (yz + xw)
    R[:, 2, 2] = 1 - 2 * (xx + yy)

    return R


def get_scale_matrices(scales):
    # Scales are stored in log form, so exponentiate them.
    scales_exp = to.exp(scales).to(device=device)
    scales_d = to.eye(3)[None, ...].to(device) * (scales_exp)[..., None]
    return scales_d


def lq_get_scale_matrices(scales):
    # Scales are stored in log form, so exponentiate them.
    scales_exp = to.exp(scales).to(device=device)
    scales_d = to.eye(3)[None, ...].to(device) * (scales_exp)[..., None]
    return scales_d


def lq_get_covariances(params):
    R = lq_get_rotation_matrices(params[:, 3:7]).to(device=device)
    S = lq_get_scale_matrices(params[:, 7:10]).to(device=device)
    return R @ S @ S @ R.transpose(-1, -2)


def get_covariances(quaternions, scales):
    R = get_rotation_matrices(quaternions).to(device=device)
    S = get_scale_matrices(scales).to(device=device)
    covariances = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)

    covariances += 1e-5 * to.eye(3).to(device=device)
    return covariances


def evaluate_points(positions, means, inv_covs, opacities):
    # Pass opacity into sigmoid function
    distance_to_mean = positions - means
    exponent = -0.5 * (
        distance_to_mean[:, :, None,
                         :] @ inv_covs @ distance_to_mean[..., None]
    ).squeeze(-1)

    evaluations = opacities * to.exp(exponent)
    return evaluations


def create_bounding_boxes(means, scales, quaternions):
    unit_cube = to.tensor(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ],
        device=device,
    )

    scaled_vertices = to.exp(scales)[:, None, :] * unit_cube[None, :, :]
    """
    new_rotations = normals_to_rot_matrix(
        self.gaussian_model.reference_normals[None,
                                            :], self.normals[None, :]
    )
    new_rotations = new_rotations.squeeze(0)
    """
    # Expand rotations to match the number of vertices (2)
    rotation_expanded = (
        get_rotation_matrices(quaternions).unsqueeze(1).to(device=device)
    )

    # rotation_expanded = rotation_expanded.expand(-1, 2, -1, -1)
    # Now do the matrix multiplication
    rotated_vertices = rotation_expanded @ scaled_vertices[..., None]

    rotated_vertices = rotated_vertices.squeeze(-1)  # [N, 2, 3]

    # Finally translate
    translated = rotated_vertices + means[:, None, :]

    return translated.min(dim=1).values, translated.max(dim=1).values


def get_top_16_differentiable(means, scales, quaternions, ray_oris, ray_dirs, K=16):
    """
    Differentiable PyTorch implementation of ray-box intersection that returns
    indices of the 16 closest bounding boxes for each ray.

    Returns:
        torch.Tensor: Indices of shape [num_rays, 16] with the closest box indices
    """
    # [B, 3], [B, 3]
    min_corners, max_corners = create_bounding_boxes(
        means, scales, quaternions)

    # Get dimensions
    num_rays = ray_oris.shape[0]
    num_boxes = min_corners.shape[0]

    # Expand dimensions for broadcasting
    ray_oris_exp = ray_oris.unsqueeze(1)  # [R, 1, 3]
    ray_dirs_exp = ray_dirs.unsqueeze(1)  # [R, 1, 3]
    min_corners_exp = min_corners.unsqueeze(0)  # [1, B, 3]
    max_corners_exp = max_corners.unsqueeze(0)  # [1, B, 3]

    # Compute safe reciprocal of ray directions (handle zeros)
    safe_ray_dirs = to.where(
        ray_dirs_exp != 0, ray_dirs_exp, to.full_like(ray_dirs_exp, 1e-8)
    )
    inv_dirs = 1.0 / safe_ray_dirs  # [R, 1, 3]

    # Compute t values for each corner
    t1 = (min_corners_exp - ray_oris_exp) * inv_dirs  # [R, B, 3]
    t2 = (max_corners_exp - ray_oris_exp) * inv_dirs  # [R, B, 3]

    # Get entry and exit t values for each axis
    t_min = to.minimum(t1, t2)  # [R, B, 3]
    t_max = to.maximum(t1, t2)  # [R, B, 3]

    # Calculate overall entry and exit times
    t_entry = to.max(t_min, dim=2)[0]  # [R, B]
    t_exit = to.min(t_max, dim=2)[0]  # [R, B]

    # A hit occurs when t_exit > max(0, t_entry)
    hit = (t_exit > t_entry) & (t_exit > 0)  # [R, B]

    # Use t_entry if positive, otherwise t_exit
    t_hit = to.where(t_entry > 0, t_entry, t_exit)  # [R, B]

    # Mark non-hit boxes with infinite distance
    t_hit = to.where(hit, t_hit, to.full_like(t_hit, float("inf")))

    # Get top 16 closest hit boxes (or fewer if there aren't 16 hits)
    k = min(K, num_boxes)
    top_values, top_indices = to.topk(t_hit, k=k, dim=1, largest=False)

    # If fewer than 16 boxes, pad with -1
    if k < K:
        padding = to.full(
            (num_rays, K - k),
            -1,
            device=top_indices.device,
            dtype=top_indices.dtype,
        )
        top_indices = to.cat([top_indices, padding], dim=1)

    return top_indices


def get_max_responses_and_depths(ray_oris, means, inv_covs, ray_dirs, opacities):
    rg_diff = means - ray_oris

    ray_dirs_T = ray_dirs.view(ray_dirs.shape[0], 1, 3, 1)  # [256, 1, 3, 1]
    # Perform matrix multiplication

    inv_cov_d = inv_covs @ ray_dirs_T
    numerator = (rg_diff[:, :, None, :] @ inv_cov_d).squeeze(-1)
    denominator = (ray_dirs[:, :, None, :] @ inv_cov_d).squeeze(-1)
    t_values = numerator / (denominator + 1e-5)

    max_positions = ray_oris + t_values * ray_dirs
    normalised_opacities = 1 / (1 + to.exp(-opacities))

    # Don't squeeze normalised_opacities to preserve dimensions
    # normalised_opacities = normalised_opacities.squeeze()

    max_responses = evaluate_points(
        max_positions, means, inv_covs, normalised_opacities
    )
    return max_responses, t_values


def cool_get_features(means, scales, quaternions, ray_oris, ray_dirs):
    get_max_responses_v2(ray_oris, ray_dirs, means, scales, quaternions)
    raise Exception
    covariances = get_covariances(quaternions, scales)
    # covariances += 1e-6 * to.eye(3).to(device=device)
    inv_covariances = to.linalg.inv(covariances)
    knn_indices = None

    knn_indices = get_top_16_differentiable_batched(
        means, scales, quaternions, ray_oris, ray_dirs
    )

    responses, depths = get_max_responses_and_depths(
        ray_oris.unsqueeze(1),
        means[knn_indices],
        inv_covariances[knn_indices],
        ray_dirs.unsqueeze(1),
        opacities[knn_indices],
    )

    # Sort by t-values
    _, sorted_idx = to.sort(depths, dim=1)

    sorted_alphas = responses.gather(dim=1, index=sorted_idx)

    # Calculate transmittance
    alphas_compliment = 1 - sorted_alphas
    transmittance = to.cumprod(alphas_compliment, dim=1)

    # Calculate contributions
    shifted = to.ones_like(transmittance)
    shifted[:, 1:] = transmittance[:, :-1]
    sorted_contribution = shifted - transmittance

    # Normalize contributions
    norm_factor = to.sum(sorted_contribution, dim=1, keepdim=True)
    # Add small epsilon to avoid division by zero
    sorted_contribution = sorted_contribution / (norm_factor + 1e-8)

    # Unsort the contribution
    inv_idx = sorted_idx.argsort(dim=1)
    contribution = sorted_contribution.gather(dim=1, index=inv_idx)

    blended_tvals = to.sum(contribution * depths, dim=1)
    return blended_tvals


def serialise_params(means, quaternions, scales, path_out):
    model = {
        "means": means,
        "quaternions": quaternions,
        "scales": scales,
    }

    save_model(path, model, path_out)


def pack_model(model):
    means = model["means"]
    quaternions = model["quaternions"]
    scales = model["scales"]

    return to.cat([means, quaternions, scales], dim=1)


def generate_fibonacci_sphere_rays(center, radius, n, jitter_scale=0.00000):
    to.manual_seed(42)
    """
    Generate rays using Fibonacci sphere sampling with PyTorch vectorization.
    Adds random jitter to ray directions for more natural variation.

    Args:
        center: The center point of the sphere (to tensor of shape [3])
        radius: The radius of the sphere
        n: Number of points/rays to generate
        jitter_scale: Scale factor for jitter (0.0 = no jitter)

    Returns:
        ray_oris: Ray origins on the sphere surface
        ray_dirs: Ray directions (normalized vectors pointing outward from center)
    """
    # Create indices tensor
    indices = to.arange(0, n, dtype=to.float32, device=device)

    # Calculate z coordinates (vectorized)
    z = 1 - (2 * indices) / (n - 1) if n > 1 else to.zeros(1)

    # Calculate radius at each height (vectorized)
    r = to.sqrt(1 - z * z)

    # Golden ratio for Fibonacci spiral
    phi = (1 + math.sqrt(5)) / 2

    # Calculate theta (vectorized)
    theta = 2 * math.pi * indices / phi

    # Calculate x and y coordinates (vectorized)
    x = r * to.cos(theta)
    y = r * to.sin(theta)

    # Stack to create ray origins
    ray_oris = to.stack([x, y, z], dim=car)

    # Scale by radius
    ray_oris = ray_oris * radius

    # Add center offset
    ray_oris = ray_oris + center

    # Ray directions pointing outward from the center
    ray_dirs = center - ray_oris

    # Add random jitter to ray directions
    if jitter_scale > 0:
        jitter = to.randn_like(ray_dirs) * jitter_scale
        ray_dirs = ray_dirs + jitter

    # Normalize ray directions
    ray_dirs = ray_dirs / to.linalg.norm(ray_dirs, dim=1, keepdim=True)

    return ray_oris, ray_dirs


def print_memory_stats(step_name=""):
    # Ensure calculations happen on CUDA device if available
    if to.cuda.is_available():
        device = to.cuda.current_device()
        allocated = to.cuda.memory_allocated(
            device) / 1024**2  # Convert bytes to MB
        reserved = to.cuda.memory_reserved(
            device) / 1024**2  # Convert bytes to MB
        max_allocated = to.cuda.max_memory_allocated(device) / 1024**2
        max_reserved = to.cuda.max_memory_reserved(device) / 1024**2
        free_driver, total_driver = to.cuda.mem_get_info(device)
        free_driver /= 1024**2
        total_driver /= 1024**2

        print(f"--- Memory Stats ({step_name}) ---")
        print(f"Allocated:        {allocated:.2f} MB")
        print(f"Reserved:         {reserved:.2f} MB")
        print(f"Max Allocated:    {max_allocated:.2f} MB")
        print(f"Max Reserved:     {max_reserved:.2f} MB")
        print(
            f"Driver Free/Total:{free_driver:.2f} MB / {total_driver:.2f} MB")
        # print(torch.cuda.memory_summary(device=device, abbreviated=True)) # Optional detailed summary
        print("-" * 30)
    else:
        print(f"--- Memory Stats ({step_name}) ---")
        print("CUDA not available. Cannot report GPU memory stats.")
        print("-" * 30)


def compute_global_features(
    means, scales, quaternions, ray_oris, ray_dirs, batch_size=2048
):
    subsample = to.randint(0, means.shape[0], (1000,))
    means = means[subsample]
    scales = scales[subsample]
    quaternions = quaternions[subsample]
    num_rays = ray_oris.shape[0]
    f_list = []
    num_batches = (num_rays + batch_size - 1) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_rays)
        # Compute features for the current batch
        f_batch = cool_get_features(
            means,
            scales,
            quaternions,
            ray_oris[start_idx:end_idx],
            ray_dirs[start_idx:end_idx],
        )
        f_list.append(f_batch)
    # Concatenate batch features to get the global feature tensor
    return to.cat(f_list, dim=0)


def find_normals(means, quaternions, scales):
    # centroid
    centroid = to.mean(means, dim=0)
    rotation_matrices = get_rotation_matrices(quaternions).to(device=device)

    # Extract the z-axis direction (third column) from each rotation matrix
    # This represents where the canonical z-axis [0,0,1] points after rotation
    normals = rotation_matrices[:, :, 2]

    # The normals from rotation matrices should already be unit length,
    # but normalize again for numerical stability
    normals = normals / to.norm(normals, dim=1, keepdim=True)

    # flip normals that point away from the centroid
    dot_products = to.sum((means - centroid) * normals, dim=1)
    flip_mask = dot_products < 0
    normals[flip_mask] = -normals[flip_mask]

    return normals


def generate_rays(means, quaternions, scales, sample_size=1024):
    covariances = get_covariances(quaternions, scales)
    L = to.linalg.cholesky(covariances).to(device=device)
    samples_per_gaussian = 10
    # Sample from a standard normal distribution (mean = 0, std = 1)
    std_normal_samples = to.randn(means.shape[0], samples_per_gaussian, 3).to(
        device=device
    )
    samples = to.bmm(std_normal_samples, L.transpose(1, 2))
    samples += means.unsqueeze(1)
    samples = samples.reshape(-1, 3)
    indices = to.randint(0, samples.shape[0], (sample_size,))
    ray_oris, ray_dirs = generate_rays_from_points(samples[indices])
    return ray_oris, ray_dirs


class Model(to.nn.Module):
    def __init__(self, means, quaternions, scales):
        super().__init__()
        self.means = to.nn.Parameter(means)
        self.quaternions = quaternions
        self.scales = scales
        self.opacities = input_opacities
        self.batch_size = 2048
        self.normals = find_normals(means, quaternions, scales)
        self.iter = 0
        self.ray_oris, self.ray_dirs = sample_v1(
            self.means,
            self.quaternions,
            self.scales,
            self.means.shape[0],  # Sample N rays
        )
        self.ray_oris = self.ray_oris.detach()
        self.ray_dirs = self.ray_dirs.detach()

    def forward(self):
        # Build tree
        kd_tree = build_kd_tree(self.means)
        # Set up query points per ray
        query_points_start = self.ray_oris
        query_points_end = self.ray_oris + self.ray_dirs * 2
        query_points = to.linspace(query_points_start, query_points_end, 10)
        raise Exception

    def forward_1(self):
        covariances = get_covariances(self.quaternions, self.scales)
        num_rays = self.ray_oris.shape[0]
        inv_covs = to.linalg.inv(covariances)
        features = []
        for start_idx in range(0, num_rays, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_rays)
            features.append(
                nerf_style_render_rays(
                    self.ray_oris[start_idx:end_idx],
                    self.ray_dirs[start_idx:end_idx],
                    self.means,
                    inv_covs,
                    self.opacities,
                    1,
                    0,
                    1,
                )
            )
        return to.cat(features)

    def forward_9(self):
        covariances = get_covariances(self.quaternions, self.scales)
        # Sample rays (keep this part)
        num_rays = self.ray_oris.shape[0]
        features = to.zeros(
            num_rays, 1, device=self.means.device)  # Result tensor
        k = 16
        knn_indices = to.zeros(num_rays, k).to(device=device)
        for start_idx in range(0, num_rays, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_rays)
            knn_indices_batch = get_top_16_differentiable_batched(
                self.means,
                self.scales,
                self.quaternions,
                self.ray_oris[start_idx:end_idx],
                self.ray_dirs[start_idx:end_idx],
                K=16,
            )  # Shape: (num_rays, K)
            knn_indices[start_idx:end_idx] = knn_indices_batch.long()

        # --- Process Rays in Batches, using KNN results ---
        for start_idx in range(0, num_rays, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_rays)

            # Get batches for rays and their corresponding KNN indices
            ray_oris_batch = self.ray_oris[start_idx:end_idx]  # (batch_R, 3)
            ray_dirs_batch = self.ray_dirs[start_idx:end_idx]  # (batch_R, 3)
            knn_indices_batch = knn_indices[start_idx:end_idx]  # (batch_R, K)
            batch_covariances = covariances[
                knn_indices_batch.long()
            ]  # (batch_R, K, 3, 3)
            inv_covs = to.linalg.inv(batch_covariances)  # (batch_R, K, 3, 3)
            max_responses, depths = get_max_responses_and_depths(
                ray_oris_batch.unsqueeze(1),
                self.means[knn_indices_batch.long()],
                inv_covs,
                ray_dirs_batch.unsqueeze(1),
                self.opacities[knn_indices_batch.long()],
            )

            blended_features = blend_features(max_responses, depths, depths)
            features[start_idx:end_idx] = blended_features

        return features

    def forward_6(self):
        with to.no_grad():
            self.ray_oris, self.ray_dirs = sample_v1(
                self.means, self.quaternions, self.scales, self.means.shape[0]
            )

        num_rays = self.ray_oris.shape[0]
        features = to.zeros(num_rays)
        for start_idx in range(0, num_rays, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_rays)
            ray_oris_batch = self.ray_oris[start_idx:end_idx]
            ray_dirs_batch = self.ray_dirs[start_idx:end_idx]
            print_memory_stats("idx: " + str(start_idx))

            # Rename get_max_responses_v2
            features_batch = get_max_responses_v2(
                ray_oris_batch,
                ray_dirs_batch,
                self.means,
                self.scales,
                self.quaternions,
                self.opacities,
            )
            print_memory_stats("idx: " + str(start_idx))

            features[start_idx:end_idx] = features_batch

        return features

    def forward_3(self):
        samples = sample_v1(self.means, self.quaternions, self.scales)
        print("done")
        return samples

    def forward_2(self):
        with to.no_grad():
            self.ray_oris, self.ray_dirs = generate_rays_from_points(
                self.means)
        num_rays = self.ray_oris.shape[0]
        features = []
        for i in range(0, num_rays, self.batch_size):
            ray_oris_batch = self.ray_oris[i: i + self.batch_size]
            ray_dirs_batch = self.ray_dirs[i: i + self.batch_size]

            sub_means = self.means[i: i + self.batch_size]
            sub_scales = self.scales[i: i + self.batch_size]
            sub_quaternions = self.quaternions[i: i + self.batch_size]

            # Compute feature values for the batch
            f_batch = cool_get_features(
                sub_means,
                sub_scales,
                sub_quaternions,
                ray_oris_batch,
                ray_dirs_batch,
            )

            features.append(f_batch)

        # Concatenate all feature values to ensure loss computation remains correct
        f = to.cat(features, dim=0)
        print("Forward  pass complete")

        return f


"""
def cool_train(means, quaternions, scales, ray_oris, ray_dirs):
    laplacian = compute_laplacian(ray_oris)
    
    for epoch in range(10):
        num_rays = ray_oris.shape[0]
        batch_size = 2048
        num_batches = (num_rays + batch_size - 1) // batch_size
        epoch_loss = 0.0
        
        lr = 0.01
        
        # Compute features in batches
        with to.no_grad():
            global_f = []
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_rays)
                ray_oris_batch = ray_oris[start_idx:end_idx]
                ray_dirs_batch = ray_dirs[start_idx:end_idx]
                global_f.append(
                    cool_get_features(means, scales, quaternions, ray_oris_batch, ray_dirs_batch)
                )
            global_f_tensor = to.cat(global_f, dim=0)
        
        # Process batches for optimization
        for i in range(num_batches):
            # Clear gradients properly
            means.grad = None
            quaternions.grad = None
            scales.grad = None
            
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_rays)
            
            # Extract batch information
            ray_oris_batch = ray_oris[start_idx:end_idx]
            ray_dirs_batch = ray_dirs[start_idx:end_idx]
            laplacian_batch = laplacian[start_idx:end_idx, start_idx:end_idx]
            
            # Get precomputed features for this batch
            f_precomputed = global_f_tensor[start_idx:end_idx].detach()
            
            # Recompute features for this batch to create a differentiable version
            f_batch = cool_get_features(means, scales, quaternions, ray_oris_batch, ray_dirs_batch)
            
            # Compute Laplacian loss
            laplacian_loss = f_batch.T @ laplacian_batch @ f_batch
            
            # Backpropagate
            laplacian_loss.backward()
            
            # Update parameters
            with to.no_grad():
                means -= lr * means.grad
                quaternions -= lr * quaternions.grad
                scales -= lr * scales.grad
            
            epoch_loss += laplacian_loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}")
"""
path = "chair.ply"
model = load_gaussians(path)


def load_uniform_sphere(
    ply_path,
    n_of_rays=1024,
    sphere_centre=to.tensor([0, 0, 0]).to(device=device),
    radius=1.0,
):
    sphere_ply = PlyData.read(ply_path)
    plydata = sphere_ply["vertex"].data
    df = pd.DataFrame(plydata)
    df = df.sample(n_of_rays)
    ray_oris = to.from_numpy(df.to_numpy()).to(
        device=device) * radius + sphere_centre
    ray_dirs = sphere_centre - ray_oris
    # Normalise ray dirs
    ray_dirs /= to.norm(ray_dirs, dim=1, keepdim=True)
    return ray_oris, ray_dirs


def sample_bbox_surface(bbox, n_samples):
    """
    Samples points uniformly from the surface of a 3D axis-aligned bounding box.

    Parameters:
      bbox: A numpy array of shape (2, 3) where bbox[0] is the minimum corner and
            bbox[1] is the maximum corner of the bounding box.
      n_samples: Number of samples to generate.

    Returns:
      A numpy array of shape (n_samples, 3) containing points uniformly distributed
      on the surface of the bounding box.
    """
    bbox = np.array(bbox)
    min_corner, max_corner = bbox[0], bbox[1]
    lengths = max_corner - min_corner

    # Define the areas of the 6 faces:
    # Two faces perpendicular to x (yz planes), two perpendicular to y (xz planes),
    # and two perpendicular to z (xy planes).
    face_areas = np.array(
        [
            lengths[1] * lengths[2],  # Face at min x
            lengths[1] * lengths[2],  # Face at max x
            lengths[0] * lengths[2],  # Face at min y
            lengths[0] * lengths[2],  # Face at max y
            lengths[0] * lengths[1],  # Face at min z
            lengths[0] * lengths[1],  # Face at max z
        ]
    )
    total_area = np.sum(face_areas)

    # Create a probability distribution for the faces proportional to their area.
    face_probs = face_areas / total_area

    # Pre-allocate array for samples.
    samples = np.zeros((n_samples, 3))

    # For each sample, choose a face based on its probability and sample uniformly on that face.
    for i in range(n_samples):
        face = np.random.choice(6, p=face_probs)

        if face == 0:  # Face at min x (x = min_corner[0])
            x = min_corner[0]
            y = np.random.uniform(min_corner[1], max_corner[1])
            z = np.random.uniform(min_corner[2], max_corner[2])
        elif face == 1:  # Face at max x (x = max_corner[0])
            x = max_corner[0]
            y = np.random.uniform(min_corner[1], max_corner[1])
            z = np.random.uniform(min_corner[2], max_corner[2])
        elif face == 2:  # Face at min y (y = min_corner[1])
            y = min_corner[1]
            x = np.random.uniform(min_corner[0], max_corner[0])
            z = np.random.uniform(min_corner[2], max_corner[2])
        elif face == 3:  # Face at max y (y = max_corner[1])
            y = max_corner[1]
            x = np.random.uniform(min_corner[0], max_corner[0])
            z = np.random.uniform(min_corner[2], max_corner[2])
        elif face == 4:  # Face at min z (z = min_corner[2])
            z = min_corner[2]
            x = np.random.uniform(min_corner[0], max_corner[0])
            y = np.random.uniform(min_corner[1], max_corner[1])
        elif face == 5:  # Face at max z (z = max_corner[2])
            z = max_corner[2]
            x = np.random.uniform(min_corner[0], max_corner[0])
            y = np.random.uniform(min_corner[1], max_corner[1])

        samples[i] = [x, y, z]

    return samples


model = Model(
    model["means"],
    model["quaternions"],
    model["scales"],
)


def sample_v3(means):
    new_ray_oris = []

    look_at = to.rand(3).to(device=device) - 0.5
    normalised_look_at = look_at / to.norm(look_at)

    ray_ori = normalised_look_at * 100
    new_ray_oris.append(ray_ori[None, ...])
    ray_dirs = means - ray_ori
    ray_dirs = ray_dirs / to.norm(ray_dirs, dim=1, keepdim=True)
    ray_oris = to.broadcast_to(ray_ori, (ray_dirs.shape[0], 3))
    return ray_oris, ray_dirs


def criterion(f, positions):
    weighted_neighbours, _, _ = compute_weighted_neighbors(positions, f)
    return to.norm(to.sum((f - weighted_neighbours) ** 2, dim=1))


def criterion_3(positions):
    weighted_neighbours, _, _ = compute_weighted_neighbors_2(positions)
    return to.norm(to.sum((f - weighted_neighbours) ** 2, dim=1))


def criterion_2(f, positions):
    weighted_neighbours, _, _ = compute_weighted_neighbors(positions, f)
    return to.sum((f - weighted_neighbours) ** 2)


def get_knn_distances_chunked(query_points, source_points, k=16, chunk_size=1024):
    """
    Calculates distances to the k-nearest neighbors without computing the full distance matrix
    by processing query points in chunks.

    Args:
        query_points (torch.Tensor): Points for which to find neighbors (shape: N, D).
        source_points (torch.Tensor): Points to search neighbors from (shape: M, D).
        k (int): Number of nearest neighbors to find.
        chunk_size (int): Number of query points to process in each chunk.

    Returns:
        tuple: (topk_distances, topk_indices) - Distances and indices of k-NN.
               topk_distances: (N, k)
               topk_indices: (N, k)  (indices are relative to source_points)
    """
    num_query_points = query_points.shape[0]
    topk_distances = to.zeros(
        (num_query_points, k), dtype=query_points.dtype, device=query_points.device
    )
    topk_indices = to.zeros(
        (num_query_points, k), dtype=to.long, device=query_points.device
    )
    for start_idx in range(0, num_query_points, chunk_size):
        end_idx = min(start_idx + chunk_size, num_query_points)
        query_chunk = query_points[start_idx:end_idx]

        # Calculate distances only for the current chunk of query points to *all* source points
        # Shape: (chunk_size, M)
        distances = to.cdist(query_chunk, source_points)

        # Find top k-nearest neighbors for each point in the chunk
        chunk_topk_distances, chunk_topk_indices = to.topk(
            distances, k=k, dim=1, largest=False
        )
        # chunk_topk_distances: (chunk_size, k)
        # chunk_topk_indices:  (chunk_size, k) (indices are relative to source_points)

        topk_distances[start_idx:end_idx, :] = chunk_topk_distances
        topk_indices[start_idx:end_idx, :] = chunk_topk_indices
    return topk_distances, topk_indices


def blend_features(responses, depths, features):
    # Sort by t-values
    _, sorted_idx = to.sort(depths, dim=1)
    sorted_alphas = responses.gather(dim=1, index=sorted_idx)
    # Calculate transmittance
    alphas_compliment = 1 - sorted_alphas
    transmittance = to.cumprod(alphas_compliment, dim=1)
    # Calculate contributions
    shifted = to.ones_like(transmittance)
    shifted[:, 1:] = transmittance[:, :-1]
    sorted_contribution = shifted - transmittance
    # Normalize contributions
    norm_factor = to.sum(sorted_contribution, dim=1, keepdim=True)
    # Add small epsilon to avoid division by zero
    sorted_contribution = sorted_contribution / (norm_factor + 1e-8)
    # Unsort the contribution
    inv_idx = sorted_idx.argsort(dim=1)
    contribution = sorted_contribution.gather(dim=1, index=inv_idx)
    blended_features = to.sum(contribution * features, dim=1)
    return blended_features


def get_max_responses_v2(ray_oris, ray_dirs, means, scales, quaternions, opacities):
    N = means.shape[0]
    R = ray_oris.shape[0]
    # Project each gaussian to each ray
    max_responses = to.zeros(R, N).to(device=device)
    depths = to.zeros(R, N).to(device=device)
    features = to.zeros(R, N).to(device=device)
    # Batch gaussians
    batch_size = 4096
    for i in range(0, N, batch_size):
        start_idx = i
        end_idx = min(i + batch_size, N)
        covs = get_covariances(
            quaternions[start_idx:end_idx], scales[start_idx:end_idx]
        )

        inv_covs = to.linalg.inv(covs + 1e-5 * to.eye(3).to(device=device))
        batch_max_responses, batch_depths = get_max_responses_and_depths(
            ray_oris.unsqueeze(1),
            means[start_idx:end_idx].unsqueeze(0),
            inv_covs.unsqueeze(0),
            ray_dirs.unsqueeze(1),
            opacities[start_idx:end_idx].unsqueeze(0),
        )

        max_responses[:, start_idx:end_idx] = batch_max_responses[:, 0]
        depths[:, start_idx:end_idx] = batch_depths[:, 0]
        features[:, start_idx:end_idx] = batch_depths[:, 0]

    max_responses_val, max_responses_idx = to.sort(
        max_responses, dim=1, descending=True
    )
    top_k_responses_idx = max_responses_idx[:, :16]
    top_k_responses_vals = max_responses_val[:, :16]
    top_k_responses_depths = depths.gather(dim=1, index=top_k_responses_idx)
    top_k_responses_features = features.gather(
        dim=1, index=top_k_responses_idx)

    blended_features = blend_features(
        top_k_responses_vals, top_k_responses_depths, top_k_responses_features
    )

    return blended_features


def serialise_point_cloud(points, output_path="cool.ply"):
    points_np = points.detach().cpu().numpy()
    # Create a structured array for the vertices
    vertex_dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertices = np.array([tuple(pt) for pt in points_np], dtype=vertex_dtype)

    # Create a PlyElement and write to file (ASCII format)
    ply_element = PlyElement.describe(vertices, "vertex")
    PlyData([ply_element], text=True).write(output_path)
    print(f"Point cloud saved to {output_path}")


def serialize_point_cloud(
    ray_oris, ray_dirs, t_values, output_path="output_point_cloud.ply"
):
    """
    Compute the point positions as: points = ray_oris + ray_dirs * t_values,
    then write them out as a PLY file.

    Args:
        ray_oris (torch.Tensor): Tensor of shape (N, 3) with ray origins.
        ray_dirs (torch.Tensor): Tensor of shape (N, 3) with ray directions.
        t_values (torch.Tensor): Tensor of shape (N,) or (N,1) with t values.
        output_path (str): Path to save the PLY file.
    """
    # Ensure t_values has shape (N, 1)
    if t_values.ndim == 1:
        t_values = t_values.unsqueeze(1)

    # Compute points
    points = ray_oris + ray_dirs * t_values

    # Convert points to NumPy
    points_np = points.detach().cpu().numpy()

    # Create a structured array for the vertices
    vertex_dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertices = np.array([tuple(pt) for pt in points_np], dtype=vertex_dtype)

    # Create a PlyElement and write to file (ASCII format)
    ply_element = PlyElement.describe(vertices, "vertex")
    PlyData([ply_element], text=True).write(output_path)
    print(f"Point cloud saved to {output_path}")


def serialise_with_intersection_info(model, output_path="intersection.ply"):
    with to.no_grad():
        # Extract model components
        means = model.means.detach()
        quaternions = model.quaternions.detach()
        scales = model.scales.detach()

        ray_dirs = model.ray_dirs.detach()
        ray_oris = model.ray_oris.detach()
        num_batches = (ray_oris.shape[0] +
                       model.batch_size - 1) // model.batch_size

        intersection_indices = []
        for i in range(num_batches):
            start_idx = i * model.batch_size
            end_idx = min((i + 1) * model.batch_size, ray_oris.shape[0])
            ray_oris_batch = ray_oris[start_idx:end_idx]
            ray_dirs_batch = ray_dirs[start_idx:end_idx]

            # Find intersecting Gaussians
            intersection_indices.append(
                get_top_16_differentiable(
                    means, scales, quaternions, ray_oris_batch, ray_dirs_batch
                )
            )
        intersection_indices = to.cat(intersection_indices, dim=0)

        # Get unique intersecting Gaussian indices
        flat_indices = intersection_indices.flatten()
        valid_indices = flat_indices[flat_indices >= 0]  # Remove padding -1s
        unique_indices = to.unique(valid_indices).cpu().numpy()

        # Get intersection counts for each Gaussian
        counts = to.zeros(means.shape[0], device=means.device, dtype=to.int)
        for idx in valid_indices:
            counts[idx] += 1

        # Create intersection mask (1 for intersected, 0 for not intersected)
        intersection_mask = to.zeros(
            means.shape[0], device=means.device, dtype=to.int)
        intersection_mask[unique_indices] = 1

        # Convert to numpy for plyfile
        means_np = means.cpu().numpy()
        mask_np = intersection_mask.cpu().numpy()
        counts_np = counts.cpu().numpy()

        # Create RGB colors: red for intersected, blue for non-intersected
        r = np.where(mask_np == 1, 255, 0).astype(np.uint8)
        g = np.zeros_like(r).astype(np.uint8)
        b = np.where(mask_np == 0, 255, 0).astype(np.uint8)

        # Create structured array for PLY
        vertex_dtype = np.dtype(
            [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
                ("intersection", "i4"),
                ("count", "i4"),
            ]
        )

        vertices = np.zeros(means_np.shape[0], dtype=vertex_dtype)
        vertices["x"] = means_np[:, 0]
        vertices["y"] = means_np[:, 1]
        vertices["z"] = means_np[:, 2]
        vertices["red"] = r
        vertices["green"] = g
        vertices["blue"] = b
        vertices["intersection"] = mask_np
        vertices["count"] = counts_np

        # Create PLY element and write to file
        ply_element = PlyElement.describe(vertices, "vertex")
        PlyData([ply_element], text=True).write(output_path)
        print(f"Intersection visualization saved to {output_path}")
        print(
            f"Total Gaussians: {means.shape[0]}, Intersected: {len(unique_indices)}")

        # Optional: also save a simplified PLY with just the intersected Gaussians
        if len(unique_indices) > 0:
            intersected_means = means_np[unique_indices]
            intersected_counts = counts_np[unique_indices]

            # Normalize counts for color intensity
            max_count = np.max(intersected_counts)
            if max_count > 0:
                color_intensity = np.minimum(
                    255, (intersected_counts * 255 // max_count)
                ).astype(np.uint8)
            else:
                color_intensity = (
                    np.ones_like(intersected_counts).astype(np.uint8) * 255
                )

            intersected_dtype = np.dtype(
                [
                    ("x", "f4"),
                    ("y", "f4"),
                    ("z", "f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                    ("count", "i4"),
                ]
            )

            intersected_vertices = np.zeros(
                len(unique_indices), dtype=intersected_dtype
            )
            intersected_vertices["x"] = intersected_means[:, 0]
            intersected_vertices["y"] = intersected_means[:, 1]
            intersected_vertices["z"] = intersected_means[:, 2]
            intersected_vertices["red"] = color_intensity
            intersected_vertices["green"] = 0
            intersected_vertices["blue"] = 0
            intersected_vertices["count"] = intersected_counts

            intersected_ply = PlyElement.describe(
                intersected_vertices, "vertex")
            PlyData([intersected_ply], text=True).write(
                output_path.replace(".ply", "_intersected_only.ply")
            )
            print(
                f"Intersected Gaussians only saved to {output_path.replace('.ply', '_intersected_only.ply')}"
            )


optimizer = to.optim.AdamW(model.parameters(), lr=5e-3, amsgrad=True)
files = glob.glob("./out/*")
for f in files:
    os.remove(f)


for epoch in range(5):
    optimizer.zero_grad()
    f = model()
    loss = criterion(f, f)
    loss.backward()
    optimizer.step()
    serialise_params(model.means, model.quaternions, model.scales, "pain.ply")
    print("fuck")
