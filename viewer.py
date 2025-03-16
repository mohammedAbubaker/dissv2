from matplotlib.colors import Normalize
from scipy.spatial.transform import Rotation
import scipy.sparse.linalg as spla
import scipy.sparse as sp
import torch.nn.functional as F
import threading
import cupy as cp
import time
from plyfile import PlyData, PlyElement
import torch.autograd.profiler as profiler
from torch.utils.data import TensorDataset, DataLoader
import math
import torch as to
import torch.nn as nn
import numpy as np
import pandas as pd
from vispy import scene, app
from tqdm import tqdm


import matplotlib.pyplot as plt
from blank import kernel_code

device = "cuda"


module = cp.RawModule(code=kernel_code)
kernel = module.get_function("ray_aabb_intersect_top16")


def generate_rays_from_points(points, radius, keep_percentage=0.55):
    # Sample a subset of points if keep_percentage < 1.0
    num_points = points.shape[0]
    num_keep = int(num_points * keep_percentage)
    # Randomly permute and select
    indices = to.randperm(num_points)[:num_keep]
    selected_points = points[indices]

    # Compute the center of the selected points
    center = selected_points.mean(dim=0)

    # Generate ray directions from the center to the selected points
    ray_dirs = selected_points - center
    ray_dirs = ray_dirs / to.norm(ray_dirs, dim=1, keepdim=True)
    jitter_strength = 0.3
    # Add some jitter to rays (noise to avoid perfect alignment)
    jitter = (
        to.rand_like(ray_dirs) - 0.5
    ) * jitter_strength  # Uniform noise in range [-jitter_strength, jitter_strength]
    ray_dirs += jitter
    ray_dirs = ray_dirs / to.norm(ray_dirs, dim=1, keepdim=True)

    # Compute ray origins
    ray_oris = center + ray_dirs * radius
    return ray_oris, -1.0 * ray_dirs


def generate_fibonacci_sphere_rays(center, radius, n, jitter_scale=0.003):
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
    ray_oris = to.stack([x, y, z], dim=1)

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


def generate_sphere_rays(center, radius, n):
    # Generate random angles for spherical coordinates
    theta = to.rand(n, 1) * 2 * to.pi  # Azimuthal angle
    phi = to.rand(n, 1) * to.pi  # Polar angle

    # Spherical to Cartesian conversion
    x = radius * to.sin(phi) * to.cos(theta)
    y = radius * to.sin(phi) * to.sin(theta)
    z = radius * to.cos(phi)

    # Combine into ray origins
    ray_oris = to.hstack((x, y, z))

    # Ray directions pointing outward from the center
    ray_dirs = ray_oris - center
    # Normalise ray dirs
    ray_dirs = ray_dirs / to.linalg.norm(ray_dirs)

    return ray_oris, ray_dirs


def compute_pairwise_great_circle(points, radius=1.0):
    # Normalize points to lie on the unit sphere
    points_normalized = points / points.norm(dim=1, keepdim=True)
    # Compute the pairwise dot product; for unit vectors, this equals cos(theta)
    dot_prod = to.mm(points_normalized, points_normalized.t())
    # Clamp to ensure numerical stability
    dot_prod = to.clamp(dot_prod, -1.0, 1.0)
    # Compute the great circle distance (angle in radians)
    distances = to.acos(dot_prod)
    # Scale by the sphere's radius if needed
    return distances * radius


def compute_pairwise_euclidean(points):
    """
    Compute pairwise Euclidean distances for an (N,3) tensor of points.
    """
    # (points^2).sum(1) => shape (N,)
    # Expand dims for row-column broadcast => shape (N,1), then (1,N)
    sum_sq = (points**2).sum(dim=1, keepdim=True)
    # Pairwise squared distances
    sq_dists = sum_sq + sum_sq.T - 2.0 * (points @ points.T)
    sq_dists = to.clamp(sq_dists, min=0.0)  # numerical stability
    return to.sqrt(sq_dists)


def based_v2_loss(
    points, predicted_depth_vals, ground_truth_depth_vals, lambda_laplacian=0.2
):
    """
    Computes the total loss as a combination of fidelity loss (e.g., MSE)
    and the Laplacian (smoothness) loss.

    predicted_depth_vals: Predicted depth values, tensor of shape [N, 1] or [N]
    ground_truth_depth_vals: Ground truth depth values, tensor of shape [N] (or [N, 1])
    lambda_laplacian: Weighting factor for the Laplacian loss.
    """
    # Data fidelity loss: mean squared error between predicted and ground truth depth\

    # fidelity_loss = F.mse_loss(predicted_depth_vals.squeeze(), ground_truth_depth_vals)
    fidelity_loss = 0
    # Laplacian smoothness loss
    laplacian_loss = compute_graph_laplacian_loss(points, predicted_depth_vals)

    # Total loss: combine both terms
    total_loss = fidelity_loss + lambda_laplacian * laplacian_loss
    return total_loss


def compute_weighted_neighbors(ray_oris, depth_values, k=4):
    """
    Computes the weighted average of k-nearest neighbors for each point.
    Returns the weighted neighbor positions.
    """

    # Compute pairwise distances between points
    distances = to.cdist(ray_oris, ray_oris)

    # Exclude self from nearest neighbors
    N = depth_values.shape[0]
    diag_mask = to.eye(N, device=depth_values.device, dtype=distances.dtype)
    distances = distances + diag_mask * 1e6

    # Get k nearest neighbors
    topk_distances, topk_indices = distances.topk(k=k, dim=1, largest=False)

    # Compute Gaussian weights and normalize
    weights = to.exp(-0.5 * topk_distances)
    weights = weights / weights.sum(dim=1, keepdim=True)

    # Gather neighbor positions: shape (N, k, D)
    neighbor_positions = depth_values[topk_indices]

    # Compute weighted average of neighbor positions for each point
    weighted_neighbors = to.sum(weights.unsqueeze(-1) * neighbor_positions, dim=1)

    return (
        weighted_neighbors,
        topk_indices,
        weights,
    )  # Return weighted neighbors, indices and weights


def implicit_backward_euler_laplacian_smooth(ray_oris, depth_value, step_size, k=4):
    N, D = depth_value.shape
    # Compute weighted neighbors (and get neighbor indices and weights)
    weighted_neighbors, neighbor_indices, weights = compute_weighted_neighbors(
        ray_oris, depth_value, k=k
    )

    # Build dense identity matrix
    I = to.eye(N, device=depth_value.device, dtype=depth_value.dtype)

    # Build dense weight matrix W of shape (N, N)
    # Start with zeros and scatter the weight values into their corresponding columns.
    W = to.zeros((N, N), device=depth_value.device, dtype=depth_value.dtype)
    # neighbor_indices has shape (N, k) and weights has the same shape.
    W.scatter_(1, neighbor_indices, weights)

    # Construct A = (1 + step_size) * I - step_size * W
    A = (1 + step_size) * I - step_size * W

    # Right-hand side b is current_positions, solve A X = current_positions for X.
    new_positions = to.linalg.solve(A, depth_value)
    return new_positions


def compute_graph_laplacian_loss(ray_oris, t_vals, k=16):
    """
    Compute the Laplacian differences for each point in the point cloud.
    The Laplacian is defined as the difference between the point and the
    weighted average of its k nearest neighbors (excluding itself).
    """
    # Compute pairwise distances between points
    distances = to.cdist(ray_oris, ray_oris)

    # Exclude self from nearest neighbors by setting diagonal to a large number
    N = ray_oris.shape[0]
    diag_mask = to.eye(N, device=ray_oris.device, dtype=distances.dtype)
    # large value ensures self is not selected
    distances = distances + diag_mask * 1e6

    # Get k nearest neighbors (excluding the point itself)
    topk_distances, topk_indices = distances.topk(k=k, dim=1, largest=False)

    # Compute Gaussian weights and normalize
    weights = to.exp(-0.5 * topk_distances)
    weights = weights / weights.sum(dim=1, keepdim=True)

    # Gather neighbor positions: shape (N, k, D)
    neighbor_positions = t_vals[topk_indices]

    # Compute weighted average of neighbor positions for each point
    weighted_neighbors = to.sum(weights.unsqueeze(-1) * neighbor_positions, dim=1)
    laplacian_diff = t_vals - weighted_neighbors
    l2_loss = to.sum(laplacian_diff**2)
    return l2_loss


"""
def compute_weighted_neighbors(points):
    # Compute pairwise distances (assuming compute_pairwise_euclidean exists)
    distances = compute_pairwise_euclidean(points)  # shape: [N, N]
    # Get indices of 16 nearest neighbors for each point (ignoring self if needed)
    get_top_16_distances, get_top_16_indices = distances.topk(
        16, dim=1, largest=False)
    # Compute weights inversely proportional to distance
    weights = 1 / (get_top_16_distances + 1e-4)
    # Normalize weights to sum to 1 for each point (L1 normalization)
    weights = weights / weights.sum(dim=1, keepdim=True)
    return get_top_16_indices, weights
"""


def build_operator_matrix(num_points, top_indices, weights, dt):
    """
    Constructs the matrix A = I + dt * (I - W)
    where W is defined through the provided top_indices and weights.
    """
    # We'll collect row indices, col indices, and corresponding values
    rows, cols, vals = [], [], []

    # Diagonal entries: for each point, we have A_ii = 1 + dt.
    for i in range(num_points):
        rows.append(i)
        cols.append(i)
        vals.append(1.0 + dt)

    # Off-diagonals: for each point i and each neighbor j,
    # A_ij = -dt * weight_ij.
    for i in range(num_points):
        for k in range(top_indices.shape[1]):
            j = top_indices[i, k].item()
            w_ij = weights[i, k].item()
            rows.append(i)
            cols.append(j)
            vals.append(-dt * w_ij)

    # Build a sparse tensor A of shape [num_points, num_points]
    indices = to.tensor([rows, cols], dtype=to.long)
    values = to.tensor(vals, dtype=to.float32)
    A_sparse = to.sparse_coo_tensor(
        indices, values=values, size=(num_points, num_points)
    )

    return A_sparse


def backward_euler_update(points, depth_vals, dt):
    """
    Performs one backward Euler update for depth_vals using a sparse solver
    that preserves the computation graph for backpropagation.

    depth_vals: tensor of shape [N] or [N, 1]
    dt: time step (scalar)
    """

    # Ensure depth_vals is of shape [N]
    num_points = depth_vals.shape[0]
    depth_vals = depth_vals.reshape(-1)

    # Compute neighbor indices and weights from points (graph remains fixed)
    top_indices, weights = compute_weighted_neighbors(points)

    # Build sparse system matrix A = I + dt * (I - W)
    A_sparse = build_operator_matrix(num_points, top_indices, weights, dt).to(
        device=device
    )
    return to.linalg.solve(A_sparse.to_dense(), depth_vals)


def quaternion_to_rotation_matrix(quaternions):
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]
    w = quaternions[:, 0]

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


class GaussianModel:
    def __init__(self, path):
        self.path = path  # store original path for later saving
        # Load in data
        plyfile = PlyData.read(path)
        plydata = plyfile["vertex"].data
        df = pd.DataFrame(plydata)
        means_mask = ["x", "y", "z"]
        quaternions_mask = ["rot_0", "rot_1", "rot_2", "rot_3"]
        scales_mask = ["scale_0", "scale_1", "scale_2"]
        opacities_mask = ["opacity"]

        self.means = to.tensor(df[means_mask].values).to(device)
        self.quaternions = to.tensor(df[quaternions_mask].values).to(device)
        self.scales = to.tensor(df[scales_mask].values).to(device)
        self.opacities = to.tensor(df[opacities_mask].values).to(device)

        self.n_gaussians = plydata.shape[0]

        # (repeat loading of data, activation, etc.)
        self.opacities = 1 / (1 + to.exp(-self.opacities))
        self.normalised_quaternions = self.quaternions / to.linalg.norm(
            self.quaternions
        )
        self.rotations = quaternion_to_rotation_matrix(self.normalised_quaternions).to(
            device
        )
        self.scales_exp = to.exp(self.scales)
        self.scales_d = to.eye(3)[None, :, :].to(device) * (self.scales_exp)[:, :, None]
        self.scales_d **= 2
        self.scales_i_d = (
            to.eye(3)[None, :, :].to(device) * (1 / self.scales_exp)[:, :, None]
        )
        self.scales_i_d **= 2
        self.rotations_t = self.rotations.transpose(-1, -2)
        self.scales_d_t = self.scales_d.transpose(-1, -2)
        self.covariances = self.rotations @ self.scales_d @ self.rotations_t

        min_indices = self.scales_exp.argmin(axis=1)
        self.normals = self.rotations[to.arange(self.n_gaussians), :, min_indices]
        self.normals = self.normals / to.linalg.norm(self.normals)
        centroid = self.means.mean(dim=0)
        vectors_to_centroid = centroid - self.means
        dot_products = (vectors_to_centroid * self.normals).sum(dim=1)
        flip_mask = dot_products < 0
        self.normals[flip_mask] = -self.normals[flip_mask]
        self.reference_normals = self.normals


def evaluate_points(points, gaussian_means, gaussian_inv_covs, gaussian_opacities):
    distance_to_mean = points - gaussian_means
    exponent = -0.5 * (
        distance_to_mean[:, :, None, :]
        @ gaussian_inv_covs
        @ distance_to_mean[..., None]
    ).squeeze(-1)
    exponent = to.clamp(exponent, min=-50, max=50)
    evaluations = gaussian_opacities * to.exp(exponent)
    return evaluations


def evaluate_points_2(points, gaussian_means, gaussian_inv_covs, gaussian_opacities):
    distance_to_mean = points - gaussian_means
    exponent = -0.5 * (
        distance_to_mean[:, :, None, :]
        @ gaussian_inv_covs
        @ distance_to_mean[..., None]
    )
    evaluations = gaussian_opacities * to.exp(exponent).squeeze(-1)
    return evaluations


def skew_symmetric(v):
    """
    v: shape (..., 3)
    Returns: shape (..., 3, 3)
    """
    # Each row of v is (vx, vy, vz)
    # K = [[ 0, -vz,  vy],
    #      [ vz,  0, -vx],
    #      [-vy, vx,   0]]
    zero = to.zeros_like(v[..., 0])
    K = to.stack(
        [
            to.stack([zero, -v[..., 2], v[..., 1]], dim=-1),
            to.stack([v[..., 2], zero, -v[..., 0]], dim=-1),
            to.stack([-v[..., 1], v[..., 0], zero], dim=-1),
        ],
        dim=-2,
    )
    return K


def normals_to_rot_matrix_2(a, b):
    # a and b are assumed to be unit vectors (with shape [..., 3])
    # Compute the angle between a and b
    a_dot_b = (a * b).sum(dim=-1, keepdim=True)  # shape [..., 1]
    theta = to.acos(to.clamp(a_dot_b, -1.0, 1.0))

    # Compute the normalized cross product (rotation axis)
    v = to.cross(a, b)
    v_norm = to.norm(v, dim=-1, keepdim=True)
    # To avoid division by zero, add a small epsilon
    eps = 1e-8
    v_unit = v / (v_norm + eps)

    # Build the skew-symmetric matrix from the normalized axis
    K = skew_symmetric(v_unit)

    # Rodrigues formula: R = I + sin(theta)*K + (1 - cos(theta)) * K^2
    I = to.eye(3, device=a.device).expand(a.shape[:-1] + (3, 3))
    sin_theta = to.sin(theta)[..., None]
    cos_theta = to.cos(theta)[..., None]

    R = I + sin_theta * K + (1 - cos_theta) * (K @ K)
    return R


def axis_angle_to_quaternion(axis, angle):
    """Convert an axis-angle rotation to a quaternion.
    Axis should be normalized.
    Returns a quaternion in (w, x, y, z) order.
    """
    half_angle = angle / 2.0
    w = to.cos(half_angle)
    xyz = axis * to.sin(half_angle)
    return to.cat([w, xyz], dim=-1)


def compute_difference_quaternion(ref_normals, current_normals, eps=1e-8):
    """Compute the quaternion that rotates ref_normals to current_normals.
    Both inputs are assumed to be normalized and of shape (N, 3).
    Returns a tensor of shape (N, 4) representing quaternions in (w, x, y, z) order.
    """
    # Compute the dot product and clamp for stability.
    dot = to.clamp((ref_normals * current_normals).sum(dim=-1), -1.0, 1.0)
    angle = to.acos(dot)

    # Compute the rotation axis.
    axis = to.cross(ref_normals, current_normals)
    axis_norm = to.norm(axis, dim=-1, keepdim=True)
    # Avoid division by zero by providing a default axis when the norm is very small.
    axis = to.where(
        axis_norm < eps,
        to.tensor([1.0, 0.0, 0.0], device=axis.device).expand_as(axis),
        axis / (axis_norm + eps),
    )

    return axis_angle_to_quaternion(axis, angle[..., None])


def quaternion_multiply(q, r):
    """
    Multiply two quaternions.
    Both q and r are tensors of shape (..., 4) in (w, x, y, z) order.
    Returns their product.
    """
    w1, x1, y1, z1 = q.unbind(dim=-1)
    w2, x2, y2, z2 = r.unbind(dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return to.stack([w, x, y, z], dim=-1)


def normals_to_rot_matrix(a, b):
    """
    a, b: shape (R, N, 3) or (N, 3) or (any_batch, 3)
        Each [i,j,:] is a normal vector.
    Returns: shape (R, N, 3, 3) or matching batch shape + (3,3).
            Rotation matrices that rotate each a[i,j] onto b[i,j].
    """
    # 1) Normalize
    a_norm = to.norm(a, dim=-1, keepdim=True)
    b_norm = to.norm(b, dim=-1, keepdim=True)
    a_unit = a / (a_norm + 1e-8)
    b_unit = b / (b_norm + 1e-8)

    # 2) Dot product => cos(theta)
    dot_ab = (a_unit * b_unit).sum(dim=-1).clamp(-1.0, 1.0)
    theta = to.acos(dot_ab)  # shape (R, N) or batch

    # 3) Rotation axis = cross(a, b) (unnormalized), then normalize
    axis = to.cross(a_unit, b_unit, dim=-1)
    axis_len = to.norm(axis, dim=-1, keepdim=True) + 1e-8
    axis_unit = axis / axis_len

    # 4) Build skew-symmetric matrix K of shape (..., 3, 3)
    K = skew_symmetric(axis_unit)

    # 5) Rodrigues formula: R = I + sin(theta)*K + (1 - cos(theta))*K^2
    sin_t = to.sin(theta)[..., None, None]  # shape (..., 1, 1)
    cos_t = to.cos(theta)[..., None, None]
    I = to.eye(3, device=a.device).expand(K.shape)  # same batch shape + (3, 3)

    K2 = K @ K
    R = I + sin_t * K + (1.0 - cos_t) * K2

    # Special case: if a and b are nearly collinear, cross(a,b) ~ 0 => axis is undefined.
    # This formula still works if a ≈ b (axis=0 => K=0 => R=I).
    # But if a ≈ -b, you get 180° rotation; handle that if needed.

    return R


def get_max_responses_and_tvals(
    ray_oris, means, covs, ray_dirs, opacities, normals, old_normals
):
    new_rotations = normals_to_rot_matrix(old_normals, normals)
    new_covs = new_rotations.transpose(-2, -1) @ covs @ new_rotations
    inv_covs = to.linalg.inv(new_covs)
    rg_diff = means - ray_oris
    inv_cov_d = inv_covs @ ray_dirs[..., None]
    numerator = (rg_diff[:, :, None, :] @ inv_cov_d).squeeze(-1)
    denominator = (ray_dirs[:, :, None, :] @ inv_cov_d).squeeze(-1)
    # Increase epsilon to avoid unstable division
    t_values = numerator / (denominator + 1e-5)
    t_values = to.clamp(t_values, -1000.0, 1000.0)
    best_positions = ray_oris + t_values * ray_dirs
    max_responses = evaluate_points(best_positions, means, inv_covs, opacities)

    return max_responses, t_values


def get_max_responses_and_tvals_2(
    ray_oris, means, covs, ray_dirs, opacities, normals, old_normals
):
    new_rotations = normals_to_rot_matrix(old_normals, normals)
    new_covs = new_rotations @ covs @ new_rotations.transpose(-2, -1)
    covs_reg = (
        covs + to.eye(3, device=covs.device).unsqueeze(0).unsqueeze(0) * 1e-3
    )  # Increase regularization

    inv_covs = to.linalg.inv(new_covs)
    rg_diff = means - ray_oris
    inv_cov_d = inv_covs @ ray_dirs[..., None]
    numerator = (rg_diff[:, :, None, :] @ inv_cov_d).squeeze(-1)
    denomenator = (ray_dirs[:, :, None, :] @ inv_cov_d).squeeze(-1)
    # Increase epsilon to 1e-5 or more
    t_values = numerator / (denomenator + 1e-5)
    t_values = to.clamp(t_values, -1000.0, 1000.0)
    best_positions = ray_oris + t_values * ray_dirs
    max_responses = evaluate_points(best_positions, means, inv_covs, opacities)

    return max_responses, t_values


class GaussianParameters(nn.Module):
    def __init__(self, path):
        super(GaussianParameters, self).__init__()
        self.gaussian_model = GaussianModel(path)
        self.means = nn.Parameter(self.gaussian_model.means)
        self.normals = self.gaussian_model.normals

    def based_2(self):
        depth_values = self.project_2().squeeze()
        predicted_depth_values = backward_euler_update(self.ray_oris, depth_values, 0.1)
        fidelity_loss = F.mse_loss(predicted_depth_values.squeeze(), depth_values)
        lambda_laplacian = 0.3
        # Laplacian smoothness loss
        laplacian_loss = compute_graph_laplacian_loss(self.ray_oris)

        total_loss = (
            lambda_laplacian * laplacian_loss + (1 - lambda_laplacian) * fidelity_loss
        )
        return (
            lambda_laplacian * laplacian_loss + (1 - lambda_laplacian) * fidelity_loss
        )

    def forward(self):
        return self.means, self.normals

    def create_bounding_boxes(self):
        unit_cube = to.tensor(
            [
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0],
            ],
            device=device,
        )

        # Shape: (N, 2, 3)
        scaled_vertices = (
            self.gaussian_model.scales_exp[:, None, :] * unit_cube[None, :, :]
        )
        """
        new_rotations = normals_to_rot_matrix(
            self.gaussian_model.reference_normals[None,
                                                :], self.normals[None, :]
        )
        new_rotations = new_rotations.squeeze(0)
        """
        # Expand rotations to match the number of vertices (2)
        rotation_expanded = self.gaussian_model.rotations.unsqueeze(1)  # [N, 1, 3, 3]
        # [N, 2, 3, 3]

        rotation_expanded = rotation_expanded.expand(-1, 2, -1, -1)

        # Now do the matrix multiplication
        rotated_vertices = (
            rotation_expanded @ scaled_vertices[..., None]
        )  # [N, 2, 3, 1]

        new_rotations = normals_to_rot_matrix(
            self.gaussian_model.reference_normals[None, ...], self.normals[None, ...]
        )
        new_rots = new_rotations.squeeze(0).unsqueeze(1)
        rotated_vertices = new_rots @ rotated_vertices
        rotated_vertices = rotated_vertices.squeeze(-1)  # [N, 2, 3]

        # Finally translate
        translated = rotated_vertices + self.means[:, None, :]

        return translated.min(dim=1).values, translated.max(dim=1).values

    def get_top_16(self, ray_oris, ray_dirs, min_corners, max_corners):
        num_rays = ray_oris.shape[0]
        num_boxes = min_corners.shape[0]

        # Create an output tensor for indices
        out_indices = to.full(
            (num_rays, 32), -1, device=ray_oris.device, dtype=to.int32
        )

        # Get device pointers
        ray_ori_ptr = ray_oris.data_ptr()
        ray_dir_ptr = ray_dirs.data_ptr()
        min_corners_ptr = min_corners.data_ptr()
        max_corners_ptr = max_corners.data_ptr()
        out_indices_ptr = out_indices.data_ptr()

        threads_per_block = 256
        blocks = (num_rays + threads_per_block - 1) // threads_per_block

        # Launch the kernel
        kernel(
            (blocks,),
            (threads_per_block,),
            (
                ray_ori_ptr,
                ray_dir_ptr,
                min_corners_ptr,
                max_corners_ptr,
                out_indices_ptr,
                num_rays,
                num_boxes,
            ),
        )

        valid_mask = out_indices != -1
        return out_indices.long(), valid_mask

    def get_spatial_hashes(self, x_idxs, y_idxs, z_idxs, spatial_args):
        h = (x_idxs * 8191) ^ (y_idxs * 131071) ^ (z_idxs * 524287)
        return to.abs(h) % spatial_args["num_cells"]

    def ray_box_intersections(self, spatial_args):
        inv_dir = 1.0 / to.where(self.ray_dirs != 0, self.ray_dirs, 1e-8)
        t1 = (spatial_args["min_corner"] - self.ray_oris) * inv_dir
        t2 = (spatial_args["max_corner"] - self.ray_oris) * inv_dir

        t_near, _ = to.max(to.minimum(t1, t2), dim=1)
        t_far, _ = to.min(to.maximum(t1, t2), dim=1)
        no_hit = t_near > t_far
        t_near[no_hit] = to.inf
        t_far[no_hit] = to.inf

        return t_near, t_far

    def ray_box_intersections_binary(self, box_mins, box_maxs, ray_oris, ray_dirs):
        """
        Given:
            box_mins: Tensor of shape [B, 3] with each box's minimum corner.
            box_maxs: Tensor of shape [B, 3] with each box's maximum corner.
            ray_oris: Tensor of shape [R, 3] with ray origins.
            ray_dirs: Tensor of shape [R, 3] with normalized ray directions.

        Returns:
            A tensor of shape [R, B] with 1 if ray i intersects box j, and 0 otherwise.
        """
        device = ray_oris.device
        R = ray_oris.shape[0]
        B = box_mins.shape[0]

        # Expand rays and boxes for vectorized computation.
        # ray_oris_exp: [R, 1, 3], ray_dirs_exp: [R, 1, 3]
        # box_mins_exp: [1, B, 3], box_maxs_exp: [1, B, 3]
        ray_oris_exp = ray_oris.unsqueeze(1)
        ray_dirs_exp = ray_dirs.unsqueeze(1)
        box_mins_exp = box_mins.unsqueeze(0)
        box_maxs_exp = box_maxs.unsqueeze(0)

        # Avoid division by zero by replacing zeros in ray_dirs with a small number.
        safe_ray_dirs = to.where(
            ray_dirs_exp != 0, ray_dirs_exp, to.tensor(1e-8, device=device)
        )

        t1 = (box_mins_exp - ray_oris_exp) / safe_ray_dirs  # [R, B, 3]
        t2 = (box_maxs_exp - ray_oris_exp) / safe_ray_dirs  # [R, B, 3]

        # For each dimension, the entry and exit times.
        t_min_dim = to.minimum(t1, t2)  # [R, B, 3]
        t_max_dim = to.maximum(t1, t2)  # [R, B, 3]

        # For each ray-box pair, the overall entry time (largest of the three t_min)
        # and exit time (smallest of the three t_max).
        t_min = t_min_dim.amax(dim=2)  # [R, B]
        t_max = t_max_dim.amin(dim=2)  # [R, B]

        # A ray intersects the box if t_max >= max(t_min, 0)
        # (i.e. the exit time is positive and occurs after the entry time).
        intersects = (t_max >= to.maximum(t_min, to.zeros_like(t_min))) & (t_max >= 0)
        return intersects

    def visualize_bounding_volumes_intersections(self):
        """
        Visualize all cell bounding volumes as points (using their centers) colored based
        on whether any ray intersects them. Cells intersected by at least one ray are rendered red,
        while cells with no intersection are rendered blue.

        Returns:
            RenderContext: A Vispy render context with the visualization.
        """
        spatial_args = self.get_spatial_args()
        box_mins, box_maxs = self.compute_cell_bounding_volumes(spatial_args)
        # Compute binary intersections (shape: [R, B] where B is the number of cells).
        intersections = self.ray_box_intersections_binary(
            box_mins, box_maxs, self.ray_oris, self.ray_dirs
        )
        # For each cell (axis=0), determine if at least one ray hit it.
        cell_intersected = intersections.sum(dim=0) > 0

        # Compute cell centers to represent each bounding volume.
        cell_centers = box_mins + 0.5 * spatial_args["spacings"]  # [B, 3]
        cell_centers_np = cell_centers.detach().cpu().numpy()
        cell_intersected_np = cell_intersected.detach().cpu().numpy()

        # Create colors: red for intersected cells, blue otherwise.
        colors = np.empty((cell_centers_np.shape[0], 4), dtype=np.float32)
        colors[cell_intersected_np] = np.array([1, 0, 0, 1], dtype=np.float32)
        colors[~cell_intersected_np] = np.array([0, 0, 1, 1], dtype=np.float32)

        # Create a RenderContext and set the cell centers (as points).
        context = RenderContext(point_size=8)
        context.scatter.set_data(
            cell_centers_np, face_color=colors, size=context.point_size
        )

        # Now, draw the rays as lines from their origin to the center.
        # Assume the center is at [0,0,0] (as used in your Fibonacci sphere generation).
        ray_oris_np = self.ray_oris.detach().cpu().numpy()
        N = ray_oris_np.shape[0]
        center_np = np.array([0, 0, 0], dtype=np.float32)
        # Create an array where each ray contributes 2 points: [origin, center]
        line_pos = np.empty((2 * N, 3), dtype=np.float32)
        line_pos[0::2] = ray_oris_np
        line_pos[1::2] = np.tile(center_np, (N, 1))
        # Create connectivity: each segment connects indices 0-1, 2-3, etc.
        connect = np.array([[i * 2, i * 2 + 1] for i in range(N)])

        # Create a Line visual for the rays.
        ray_lines = scene.visuals.Line(
            pos=line_pos,
            connect=connect,
            color=(0, 1, 0, 1),
            width=2,
            parent=context.view.scene,
        )

        context.canvas.update()
        return context

    def project(self, batch_size=2048, temperature=0.01, use_soft_indices=True):
        num_rays = self.ray_oris.shape[0]
        num_batches = (num_rays + batch_size - 1) // batch_size
        all_blended_tvals = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_rays)
            ray_oris_batch = self.ray_oris[start_idx:end_idx]
            ray_dirs_batch = self.ray_dirs[start_idx:end_idx]
            # Compute bounding boxes for all Gaussians
            min_corners, max_corners = self.create_bounding_boxes()

            if use_soft_indices:
                # Get soft t-values and weights for all gaussians
                soft_tvals, soft_weights = get_soft_top_candidates(
                    ray_oris_batch,
                    ray_dirs_batch,
                    min_corners,
                    max_corners,
                    temperature=temperature,
                )

                # Calculate responses and t-values for all gaussians
                # (reshape ray origins and directions for broadcasting)
                responses, tvals = get_max_responses_and_tvals(
                    ray_oris_batch[:, None, :],
                    self.means.unsqueeze(0).expand(ray_oris_batch.shape[0], -1, 3),
                    self.gaussian_model.covariances.unsqueeze(0).expand(
                        ray_oris_batch.shape[0], -1, 3, 3
                    ),
                    ray_dirs_batch[:, None, :],
                    self.gaussian_model.opacities.unsqueeze(0).expand(
                        ray_oris_batch.shape[0], -1, 1
                    ),
                    self.normals.unsqueeze(0).expand(ray_oris_batch.shape[0], -1, 3),
                    self.gaussian_model.normals.unsqueeze(0).expand(
                        ray_oris_batch.shape[0], -1, 3
                    ),
                )

                # Apply soft weights to responses
                responses = responses * soft_weights

                # Continue with the same alpha composition as the hard approach
                _, sorted_idx = to.sort(tvals, dim=1)
                sorted_alphas = responses.gather(dim=1, index=sorted_idx)
                alphas_compliment = 1 - sorted_alphas
                transmittance = to.cumprod(alphas_compliment, dim=1)
                shifted = to.ones_like(transmittance)
                shifted[:, 1:] = transmittance[:, :-1]
                sorted_contribution = shifted - transmittance
                norm_factor = to.sum(sorted_contribution, dim=1, keepdim=True)
                sorted_contribution = sorted_contribution / (norm_factor + 1e-8)
                inv_idx = sorted_idx.argsort(dim=1)
                contribution = sorted_contribution.gather(dim=1, index=inv_idx)
                batch_blended_tvals = to.sum(contribution * tvals, dim=1)
            else:
                # Existing hard selection method:
                ray_gaussian_candidates = self.get_top_16_differentiable_batch(
                    ray_oris_batch, ray_dirs_batch
                ).long()
                responses, tvals = get_max_responses_and_tvals(
                    ray_oris_batch[:, None, :],
                    self.means[ray_gaussian_candidates],
                    self.gaussian_model.covariances[ray_gaussian_candidates],
                    ray_dirs_batch[:, None, :],
                    self.gaussian_model.opacities[ray_gaussian_candidates],
                    self.normals[ray_gaussian_candidates],
                    self.gaussian_model.normals[ray_gaussian_candidates],
                )
                _, sorted_idx = to.sort(tvals, dim=1)
                sorted_alphas = responses.gather(dim=1, index=sorted_idx)
                alphas_compliment = 1 - sorted_alphas
                transmittance = to.cumprod(alphas_compliment, dim=1)
                shifted = to.ones_like(transmittance)
                shifted[:, 1:] = transmittance[:, :-1]
                sorted_contribution = shifted - transmittance
                norm_factor = to.sum(sorted_contribution, dim=1, keepdim=True)
                sorted_contribution = sorted_contribution / (norm_factor + 1e-8)
                inv_idx = sorted_idx.argsort(dim=1)
                contribution = sorted_contribution.gather(dim=1, index=inv_idx)
                batch_blended_tvals = to.sum(contribution * tvals, dim=1)

            all_blended_tvals.append(batch_blended_tvals)

        return to.cat(all_blended_tvals, dim=0)[..., None]

    def project_2(self, batch_size=2048):
        #       self.ray_oris, self.ray_dirs = generate_rays_from_points(self.means, 2)

        num_rays = self.ray_oris.shape[0]
        num_batches = (num_rays + batch_size - 1) // batch_size
        all_blended_tvals = []

        # self.ray_oris, self.ray_dirs = generate_rays_from_points(self.gaussian_model.means, 2, 0.1)
        for i in range(num_batches):
            # Get batch indices
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_rays)

            # Get ray data for this batch
            ray_oris_batch = self.ray_oris[start_idx:end_idx]
            ray_dirs_batch = self.ray_dirs[start_idx:end_idx]

            ray_gaussian_indices = self.get_top_16_differentiable_batch(
                ray_oris_batch, ray_dirs_batch
            )

            # Calculate responses and t-values
            responses, tvals = get_max_responses_and_tvals(
                ray_oris_batch[:, None, :],
                self.means[ray_gaussian_indices],
                self.gaussian_model.covariances[ray_gaussian_indices],
                ray_dirs_batch[:, None, :],
                self.gaussian_model.opacities[ray_gaussian_indices],
                self.normals[ray_gaussian_indices],
                self.gaussian_model.normals[ray_gaussian_indices],
            )

            # Sort by t-values
            _, sorted_idx = to.sort(tvals, dim=1)
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

            # Calculate blended t-values
            batch_blended_tvals = to.sum(contribution * tvals, dim=1)
            all_blended_tvals.append(batch_blended_tvals)

        return to.cat(all_blended_tvals, dim=0)

    def get_top_16_differentiable_batch(self, ray_oris, ray_dirs):
        """
        Batched version of get_top_16_differentiable.

        Args:
            ray_oris: Ray origins tensor of shape [batch_size, 3]
            ray_dirs: Ray directions tensor of shape [batch_size, 3]

        Returns:
            torch.Tensor: Indices of top 16 closest boxes for each ray
        """
        # Get bounding box corners
        min_corners, max_corners = self.create_bounding_boxes()

        # Get dimensions
        num_rays = ray_oris.shape[0]
        num_boxes = min_corners.shape[0]

        # Expand dimensions for broadcasting
        ray_oris_exp = ray_oris.unsqueeze(1)  # [R, 1, 3]
        ray_dirs_exp = ray_dirs.unsqueeze(1)  # [R, 1, 3]
        min_corners_exp = min_corners.unsqueeze(0)  # [1, B, 3]
        max_corners_exp = max_corners.unsqueeze(0)  # [1, B, 3]

        # Compute safe reciprocal of ray directions
        safe_ray_dirs = to.where(
            ray_dirs_exp != 0, ray_dirs_exp, to.full_like(ray_dirs_exp, 1e-8)
        )
        inv_dirs = 1.0 / safe_ray_dirs  # [R, 1, 3]

        # Compute intersection t-values
        t1 = (min_corners_exp - ray_oris_exp) * inv_dirs
        t2 = (max_corners_exp - ray_oris_exp) * inv_dirs

        t_min = to.minimum(t1, t2)
        t_max = to.maximum(t1, t2)

        t_entry = to.max(t_min, dim=2)[0]
        t_exit = to.min(t_max, dim=2)[0]

        hit = (t_exit > t_entry) & (t_exit > 0)
        t_hit = to.where(t_entry > 0, t_entry, t_exit)
        t_hit = to.where(hit, t_hit, to.full_like(t_hit, float("inf")))

        # Get top 16 closest hit boxes
        k = min(16, num_boxes)
        top_values, top_indices = to.topk(t_hit, k=k, dim=1, largest=False)

        # Pad with -1 if needed
        if k < 16:
            padding = to.full(
                (num_rays, 16 - k),
                -1,
                device=top_indices.device,
                dtype=top_indices.dtype,
            )
            top_indices = to.cat([top_indices, padding], dim=1)

        return top_indices

    def get_top_16_differentiable(self):
        """
        Differentiable PyTorch implementation of ray-box intersection that returns
        indices of the 16 closest bounding boxes for each ray.

        Returns:
            torch.Tensor: Indices of shape [num_rays, 16] with the closest box indices
        """
        # Get ray origins, directions, and bounding box corners
        ray_oris = self.ray_oris  # [R, 3]
        ray_dirs = self.ray_dirs  # [R, 3]
        # [B, 3], [B, 3]
        min_corners, max_corners = self.create_bounding_boxes()

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
        k = min(16, num_boxes)
        top_values, top_indices = to.topk(t_hit, k=k, dim=1, largest=False)

        # If fewer than 16 boxes, pad with -1
        if k < 16:
            padding = to.full(
                (num_rays, 16 - k),
                -1,
                device=top_indices.device,
                dtype=top_indices.dtype,
            )
            top_indices = to.cat([top_indices, padding], dim=1)

        return top_indices

    def implicit_depth_update(self, d_current, delta_t):
        # Solve the linear system (I + delta_t * laplacian) * d_new = d_current
        I = to.eye(self.laplacian.shape[0], device=self.laplacian.device)
        A = I + delta_t * self.laplacian
        d_new = to.linalg.solve(A, d_current)
        return d_new

    def based_loss(self):
        ray_oris, ray_dirs = generate_rays_from_points(self.gaussian_model.means, 2)
        self.ray_oris = ray_oris.to(device)
        self.ray_dirs = ray_dirs.to(device)
        self.num_rays = ray_oris.shape[0]
        return compute_graph_laplacian_loss(ray_oris, self.project_2())

    def harmonic_loss(self):
        ray_oris, ray_dirs = generate_rays_from_points(self.gaussian_model.means, 2)
        model.ray_oris = ray_oris.to(device)
        model.ray_dirs = ray_dirs.to(device)
        model.num_rays = ray_oris.shape[0]
        model.laplacian = compute_graph_laplacian_loss(ray_oris, model.project_2()).to(
            device
        )
        # Compute the main loss via the Laplacian quadratic form on the projected t-values.
        delta_t = 1.0
        d_current = self.project_2()
        d_new = self.implicit_depth_update(d_current, delta_t)
        feature_loss = to.sum((d_new - d_current) ** 2)
        return feature_loss

    def harmonic_loss_2(self):
        # Compute the current projection of t-values.
        blended_t_vals = self.project()

        if to.isnan(blended_t_vals).any() or to.isinf(blended_t_vals).any():
            raise ValueError("NaN or Inf detected in blended_t_vals")

        # Define hyperparameters.
        feature_factor = 0.03
        # Compute feature loss using the Laplacian quadratic form.
        feature_loss = feature_factor * (
            blended_t_vals.T @ self.laplacian @ blended_t_vals
        )

        """
        # Set up and solve the implicit update: (I + dt*dampening_factor*L) * f_new = f_prev.
        I = to.eye(self.laplacian.shape[0], device=self.laplacian.device)
        A = I + dt * dampening_factor * self.laplacian
        f_new = to.linalg.solve(A, self.f_prev)

        # Compute the implicit loss as the squared difference between the projection and the implicit update.
        implicit_loss = to.sum((blended_t_vals - f_new) ** 2)

        # Update f_prev for the next iteration without detaching f_new.
        self.f_prev = f_new.detach()
        """
        # Return the total loss combining implicit and feature losses.
        return feature_loss


class RenderContext:
    def __init__(self, point_size=5, fov=45, distance=10):
        # Create a Vispy canvas with an interactive background.
        self.canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="black")
        self.view = self.canvas.central_widget.add_view()
        # Use a TurntableCamera for 3D interaction.
        self.view.camera = scene.cameras.TurntableCamera(fov=fov, distance=distance)
        # Create a scatter visual to display points.
        self.scatter = scene.visuals.Markers(parent=self.view.scene)
        # Initialize with an empty dataset.
        self.scatter.set_data(
            np.empty((0, 3)), edge_color=None, face_color=(1, 1, 1, 1), size=point_size
        )
        self.point_size = point_size

    def update(self, positions, blended_tvals):
        """
        Update the scatter plot with new 3D positions and color them based on blended_tvals.

        Parameters:
            positions (np.ndarray): An (N, 3) array of 3D coordinates.
            blended_tvals (np.ndarray): A length-N array of scalar values for color mapping.
        """
        if positions is None or positions.size == 0:
            return

        # Ensure positions is of shape (N, 3)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must be a numpy array of shape (N, 3)")

        # Map blended_tvals to colors if provided.
        if blended_tvals is not None and len(blended_tvals) == positions.shape[0]:
            min_val = np.min(blended_tvals)
            max_val = np.max(blended_tvals)
            range_val = max_val - min_val if max_val != min_val else 1.0
            normalized = (blended_tvals - min_val) / range_val

            # Simple blue-to-red colormap:
            colors = np.empty((positions.shape[0], 4), dtype=np.float32)
            colors[:, 0] = normalized[:, 0]  # Red increases with value.
            colors[:, 1] = 0.2  # Fixed green for consistency.
            # Blue decreases with value.
            colors[:, 2] = 1 - normalized[:, 0]
            colors[:, 3] = 1.0  # Fully opaque.
        else:
            # Default to white if no valid scalar values are provided.
            colors = np.tile(
                np.array([1, 1, 1, 1], dtype=np.float32), (positions.shape[0], 1)
            )

        # Update scatter plot data.
        self.scatter.set_data(positions, face_color=colors, size=self.point_size)
        self.canvas.update()
        # Process pending GUI events to refresh the display immediately.
        app.process_events()


def matrix_to_quaternion(rotation_matrices):
    N = rotation_matrices.shape[0]
    q = to.zeros((N, 4), device=rotation_matrices.device)

    trace = to.einsum("nii->n", rotation_matrices)

    cond1 = trace > 0
    cond2 = (rotation_matrices[:, 0, 0] > rotation_matrices[:, 1, 1]) & ~cond1
    cond3 = (rotation_matrices[:, 1, 1] > rotation_matrices[:, 2, 2]) & ~(cond1 | cond2)
    cond4 = ~(cond1 | cond2 | cond3)

    S = to.zeros_like(trace)
    S[cond1] = to.sqrt(trace[cond1] + 1.0) * 2
    q[cond1, 0] = 0.25 * S[cond1]
    q[cond1, 1] = (rotation_matrices[cond1, 2, 1] - rotation_matrices[cond1, 1, 2]) / S[
        cond1
    ]
    q[cond1, 2] = (rotation_matrices[cond1, 0, 2] - rotation_matrices[cond1, 2, 0]) / S[
        cond1
    ]
    q[cond1, 3] = (rotation_matrices[cond1, 1, 0] - rotation_matrices[cond1, 0, 1]) / S[
        cond1
    ]

    S[cond2] = (
        to.sqrt(
            1.0
            + rotation_matrices[cond2, 0, 0]
            - rotation_matrices[cond2, 1, 1]
            - rotation_matrices[cond2, 2, 2]
        )
        * 2
    )
    q[cond2, 0] = (rotation_matrices[cond2, 2, 1] - rotation_matrices[cond2, 1, 2]) / S[
        cond2
    ]
    q[cond2, 1] = 0.25 * S[cond2]
    q[cond2, 2] = (rotation_matrices[cond2, 0, 1] + rotation_matrices[cond2, 1, 0]) / S[
        cond2
    ]
    q[cond2, 3] = (rotation_matrices[cond2, 0, 2] + rotation_matrices[cond2, 2, 0]) / S[
        cond2
    ]

    S[cond3] = (
        to.sqrt(
            1.0
            + rotation_matrices[cond3, 1, 1]
            - rotation_matrices[cond3, 0, 0]
            - rotation_matrices[cond3, 2, 2]
        )
        * 2
    )
    q[cond3, 0] = (rotation_matrices[cond3, 0, 2] - rotation_matrices[cond3, 2, 0]) / S[
        cond3
    ]
    q[cond3, 1] = (rotation_matrices[cond3, 0, 1] + rotation_matrices[cond3, 1, 0]) / S[
        cond3
    ]
    q[cond3, 2] = 0.25 * S[cond3]
    q[cond3, 3] = (rotation_matrices[cond3, 1, 2] + rotation_matrices[cond3, 2, 1]) / S[
        cond3
    ]

    S[cond4] = (
        to.sqrt(
            1.0
            + rotation_matrices[cond4, 2, 2]
            - rotation_matrices[cond4, 0, 0]
            - rotation_matrices[cond4, 1, 1]
        )
        * 2
    )
    q[cond4, 0] = (rotation_matrices[cond4, 1, 0] - rotation_matrices[cond4, 0, 1]) / S[
        cond4
    ]
    q[cond4, 1] = (rotation_matrices[cond4, 0, 2] + rotation_matrices[cond4, 2, 0]) / S[
        cond4
    ]
    q[cond4, 2] = (rotation_matrices[cond4, 1, 2] + rotation_matrices[cond4, 2, 1]) / S[
        cond4
    ]
    q[cond4, 3] = 0.25 * S[cond4]

    return q


def rot_matrix_to_quaternions(R):
    R = R[0, ...]
    """
    Convert an (n,3,3) batch of rotation matrices to (n,4) batch of quaternions.
    Args:
        R: to.Tensor of shape (n,3,3), where each (3,3) matrix is a rotation matrix.

    Returns:
        to.Tensor of shape (n,4), where each quaternion is (w, x, y, z).
    """
    n = R.shape[0]

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    q = to.zeros((n, 4), device=R.device)

    # Case w is largest
    w_large = trace > 0
    if w_large.any():
        S = to.sqrt(trace[w_large] + 1.0) * 2  # S=4w
        q[w_large, 0] = 0.25 * S
        q[w_large, 1] = (R[w_large, 2, 1] - R[w_large, 1, 2]) / S
        q[w_large, 2] = (R[w_large, 0, 2] - R[w_large, 2, 0]) / S
        q[w_large, 3] = (R[w_large, 1, 0] - R[w_large, 0, 1]) / S

    # Case x is largest
    x_large = (~w_large) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if x_large.any():
        S = to.sqrt(1.0 + R[x_large, 0, 0] - R[x_large, 1, 1] - R[x_large, 2, 2]) * 2
        q[x_large, 0] = (R[x_large, 2, 1] - R[x_large, 1, 2]) / S
        q[x_large, 1] = 0.25 * S
        q[x_large, 2] = (R[x_large, 0, 1] + R[x_large, 1, 0]) / S
        q[x_large, 3] = (R[x_large, 0, 2] + R[x_large, 2, 0]) / S

    # Case y is largest
    y_large = (~w_large) & (~x_large) & (R[:, 1, 1] > R[:, 2, 2])
    if y_large.any():
        S = to.sqrt(1.0 + R[y_large, 1, 1] - R[y_large, 0, 0] - R[y_large, 2, 2]) * 2
        q[y_large, 0] = (R[y_large, 0, 2] - R[y_large, 2, 0]) / S
        q[y_large, 1] = (R[y_large, 0, 1] + R[y_large, 1, 0]) / S
        q[y_large, 2] = 0.25 * S
        q[y_large, 3] = (R[y_large, 1, 2] + R[y_large, 2, 1]) / S

    # Case z is largest
    z_large = ~(w_large | x_large | y_large)
    if z_large.any():
        S = to.sqrt(1.0 + R[z_large, 2, 2] - R[z_large, 0, 0] - R[z_large, 1, 1]) * 2
        q[z_large, 0] = (R[z_large, 1, 0] - R[z_large, 0, 1]) / S
        q[z_large, 1] = (R[z_large, 0, 2] + R[z_large, 2, 0]) / S
        q[z_large, 2] = (R[z_large, 1, 2] + R[z_large, 2, 1]) / S
        q[z_large, 3] = 0.25 * S

    return q


def save_optimized_gaussian_model(model, output_path="optimized_point_cloud.ply"):
    """
    Save the optimized Gaussian model to a new PLY file.
    The saved file will have all columns identical to the original file,
    except that the 'x', 'y', and 'z' columns (means) are updated with the
    final optimized values and the quaternion columns ('rot_0', 'rot_1',
    'rot_2', 'rot_3') are recomputed from the optimized normals.
    """
    # Re-read the original PLY file to preserve all columns and order.
    original_ply = PlyData.read(model.gaussian_model.path)
    original_data = original_ply["vertex"].data
    df = pd.DataFrame(original_data)

    # Update mean coordinates (x, y, z) with optimized values.
    new_means = model.means.detach().cpu().numpy()  # shape (N, 3)
    df["x"] = new_means[:, 0]
    df["y"] = new_means[:, 1]
    df["z"] = new_means[:, 2]

    # Get rotation matrix that transforms old normals to new normals.
    diff_rot = normals_to_rot_matrix(
        model.gaussian_model.reference_normals[None, ...], model.normals[None, ...]
    ).squeeze(0)
    # Convert the diff rot int a diff quat
    diff_quats = matrix_to_quaternion(diff_rot)
    # Apply the diff quats to new quaternioons
    new_quats = (
        quaternion_multiply(diff_quats, model.gaussian_model.quaternions)
        .detach()
        .cpu()
        .numpy()
    )
    df["rot_0"] = new_quats[:, 0]
    df["rot_1"] = new_quats[:, 1]
    df["rot_2"] = new_quats[:, 2]
    df["rot_3"] = new_quats[:, 3]
    # Convert the DataFrame back into a structured numpy array with the original dtype.
    new_data = df.to_records(index=False)
    # Create a PlyElement and write out a binary little-endian PLY file.
    ply_element = PlyElement.describe(new_data, "vertex")
    PlyData([ply_element], text=False).write(output_path)


def visualize_depth_updates(model, ray_oris, ray_dirs, update_interval=0.1):
    """
    Creates a Vispy window which displays updated ray positions.
    Each frame, the ray positions are computed as:
        new_points = ray_oris + (projected_depth * ray_dirs)
    and the points are colored based on their current depth value.
    """
    canvas = scene.SceneCanvas(keys="interactive", bgcolor="black", show=True)
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera()

    # Create markers which will be updated each tick.
    scatter = scene.visuals.Markers(parent=view.scene)

    def update(event):
        depth = model.project_2().squeeze().detach().cpu().numpy()  # (N,1) or (N,)
        depth = depth[..., None]

        # Compute updated positions.
        new_points = ray_oris.detach().cpu().numpy()
        # Normalize depth values for color mapping.
        min_d = np.min(depth)
        max_d = np.max(depth)
        norm_depth = (depth - min_d) / (max_d - min_d + 1e-8)

        # Create a blue-to-red colormap: red increases with depth, blue decreases.
        colors = np.zeros((new_points.shape[0], 4), dtype=np.float32)
        colors[:, 0] = norm_depth[:, 0]  # Red channel
        colors[:, 1] = 0.2  # Fixed green channel
        colors[:, 2] = 1 - norm_depth[:, 0]  # Blue channel
        colors[:, 3] = 1.0  # Alpha

        # Update markers.
        scatter.set_data(new_points, face_color=colors, size=10)
        canvas.update()

    # Start timer to update every 'update_interval' seconds.
    timer = app.Timer(interval=100)
    timer.connect(update)
    timer.start()
    app.run()


def start_visualization(model, ray_oris, ray_dirs, update_interval=0.2):
    visualize_depth_updates(model, ray_oris, ray_dirs, update_interval=update_interval)


def visualize_means_evolution(model, update_interval=0.2):
    """
    Visualize the evolution of gaussian means over time.
    A Vispy scatter plot is periodically updated with the current values of `model.means`.

    Args:
        model: Your GaussianParameters model.
        update_interval: Time between updates in seconds.
    """
    from vispy import scene, app
    import numpy as np

    # Set up the Vispy canvas and view.
    canvas = scene.SceneCanvas(keys="interactive", bgcolor="black", show=True)
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(fov=45)
    # Initialize scatter visual for gaussian means.
    scatter = scene.visuals.Markers(parent=view.scene)

    # Update function: update scatter data with current means.
    def update(event):
        # Get current means (detach from graph to prevent interference with training)
        means_np = model.means.detach().cpu().numpy()
        # Update scatter: here we use red color; adjust size/color as needed.
        scatter.set_data(means_np, face_color=(1, 0, 0, 1), size=10)
        canvas.update()

    # Set up timer to trigger update periodically.
    timer = app.Timer(interval=update_interval, connect=update, start=True)
    view.camera.set_range()
    app.run()


def train_model(model, num_iterations=300, lr=3e-4):
    n_of_rays = 5000
    radius = 2

    sphere_centre = to.tensor(
        [to.mean(model.means[0]), to.mean(model.means[1]), to.mean(model.means[2])],
        device=device,
    )
    """
    # Start visualization in a separate thread - this could be interfering with the training
    sphere_params = (sphere_centre, radius, n_of_rays)
    model.ray_oris, model.ray_dirs = generate_rays_from_points(model.means, 3.0)
    thread = threading.Thread(
        target=start_visualization, args=(model, model.ray_oris, model.ray_dirs, 0.2)
    )
    thread.daemon = True  # Make the thread daemon so it exits when main program does
    thread.start()
    losses = []
    """
    model.ray_oris, model.ray_dirs = generate_rays_from_points(
        model.means.detach().clone(), 3.0
    )
    optimizer = to.optim.AdamW([model.means, model.normals], lr=lr)
    for iteration in tqdm(range(num_iterations)):
        if not (iteration % 50):
            model.ray_oris, model.ray_dirs = generate_rays_from_points(
                model.means.detach().clone(), 3.0
            )
        # Fix ray origins/directions once at the beginning
        features = model.project_2()
        loss = features.T @ get_laplacian(model.ray_oris.detach().clone()) @ features
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    save_optimized_gaussian_model(model, output_path="out.ply")


def get_box_corners(mins, maxs):
    """
    Compute t()he 8 corners of a 3D bounding box (vectorized version).

    Args:
        mins: (B, 3) tensor/array for minimum corner of the box.
        maxs: (B, 3) tensor/array for maximum corner of the box.


    Returns:
        (B, 8, 3) tensor/array with the 8 corners of the box.
    """
    B = mins.shape[0]
    device = mins.device

    # Create binary pattern for all 8 corners (use min or max for each dimension)
    # Format: [x, y, z] where 0=min, 1=max
    corner_pattern = to.tensor(
        [
            [0, 0, 0],  # min, min, min
            [1, 0, 0],  # max, min, min
            [1, 1, 0],  # max, max, min
            [0, 1, 0],  # min, max, min
            [0, 0, 1],  # min, min, max
            [1, 0, 1],  # max, min, max
            [1, 1, 1],  # max, max, max
            [0, 1, 1],  # min, max, max
        ],
        device=device,
    ).float()

    # Expand dimensions for broadcasting: (1, 8, 3)
    pattern = corner_pattern.unsqueeze(0)

    # Expand mins and maxs for broadcasting: (B, 1, 3)
    mins_exp = mins.unsqueeze(1)
    maxs_exp = maxs.unsqueeze(1)

    # Use pattern to select between min and max for each corner
    # (B, 8, 3) = (B, 1, 3) * (1, 8, 3) + (B, 1, 3) * (1 - (1, 8, 3))
    corners = maxs_exp * pattern + mins_exp * (1 - pattern)

    return corners


def test_scenario(old_mins, old_maxs, old_normals, new_normals):
    # Get all 8 corners of the bounding boxes.
    old_corners = get_box_corners(old_mins, old_maxs)
    rotation_matrix = normals_to_rot_matrix(old_normals, new_normals)
    # Translate the corners to the origin.
    old_centres = (old_mins + old_maxs) / 2
    new_corners = old_corners - old_centres[:, None, :]
    # Rotate the corners.
    new_corners = to.einsum("bij,bkj->bki", rotation_matrix, new_corners)
    # Translate the corners back to the original position.
    new_corners += old_centres[:, None, :]

    # Select a single box for visualization.
    box_idx = to.argmax(old_maxs - old_mins)
    render_bounding_boxes(
        old_corners[box_idx : box_idx + 1],
        new_corners[box_idx : box_idx + 1],
        old_normals[box_idx : box_idx + 1],
        new_normals[box_idx : box_idx + 1],
    )


def render_bounding_boxes(old_corners, new_corners, old_normals, new_normals):
    """
    Render bounding boxes and their normal vectors in 3D.

    Args:
        old_corners: (B, 8, 3) tensor with corners of old bounding boxes
        new_corners: (B, 8, 3) tensor with corners of new bounding boxes
        old_normals: (B, 3) tensor with old normal directions
        new_normals: (B, 3) tensor with new normal directions
    """
    from vispy import scene, app
    import numpy as np

    # Convert to NumPy arrays if tensors
    if isinstance(old_corners, to.Tensor):
        old_corners = old_corners.detach().cpu().numpy()
    if isinstance(new_corners, to.Tensor):
        new_corners = new_corners.detach().cpu().numpy()
    if isinstance(old_normals, to.Tensor):
        old_normals = old_normals.detach().cpu().numpy()
    if isinstance(new_normals, to.Tensor):
        new_normals = new_normals.detach().cpu().numpy()

    # Create scene
    canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="black")
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(center=(0, 0, 0), fov=40)

    # Calculate box centers
    old_centers = old_corners.mean(axis=1)  # (B, 3)
    new_centers = new_corners.mean(axis=1)  # (B, 3)

    # Define box wireframe connections (edges between vertices)
    box_connections = np.array(
        [
            [0, 1],
            [1, 5],
            [5, 4],
            [4, 0],  # bottom face
            [0, 3],
            [3, 2],
            [2, 1],
            [2, 6],
            [6, 5],  # sides
            [7, 6],
            [7, 3],
            [7, 4],  # remaining edges
        ]
    )

    # Loop through boxes (usually just one box in this case)
    for i in range(old_corners.shape[0]):
        # Create old box wireframe (blue)
        old_box = scene.visuals.Line(
            pos=old_corners[i],
            connect=box_connections,
            color=(0, 0, 1, 1),  # blue
            width=2,
            parent=view.scene,
        )

        # Create new box wireframe (pink)
        new_box = scene.visuals.Line(
            pos=new_corners[i],
            connect=box_connections,
            color=(1, 0.5, 0.8, 1),  # pink
            width=2,
            parent=view.scene,
        )

        # Scale factor for normal vectors (make them visible enough)
        scale = 1.0

        # Create old normal vector (red)
        old_normal_end = old_centers[i] + old_normals[i] * scale
        old_normal_line = scene.visuals.Line(
            pos=np.array([old_centers[i], old_normal_end]),
            color=(1, 0, 0, 1),  # red
            width=3,
            parent=view.scene,
        )

        # Create new normal vector (green)
        new_normal_end = new_centers[i] + new_normals[i] * scale
        new_normal_line = scene.visuals.Line(
            pos=np.array([new_centers[i], new_normal_end]),
            color=(0, 1, 0, 1),  # green
            width=3,
            parent=view.scene,
        )

    view.camera.set_range()
    view.camera.center = old_corners.mean(axis=1).mean(axis=0)
    app.run()


def get_soft_top_candidates(
    ray_oris, ray_dirs, min_corners, max_corners, temperature=0.1
):  # Increased temperature
    # Expand dimensions for broadcasting as in your hard selection method.
    ray_oris_exp = ray_oris.unsqueeze(1)  # [R, 1, 3]
    ray_dirs_exp = ray_dirs.unsqueeze(1)  # [R, 1, 3]
    min_corners_exp = min_corners.unsqueeze(0)  # [1, B, 3]
    max_corners_exp = max_corners.unsqueeze(0)  # [1, B, 3]

    # Check if any boxes are invalid (min > max)
    invalid_boxes = (min_corners > max_corners).any(dim=1)
    if invalid_boxes.any():
        print(f"Warning: {invalid_boxes.sum().item()} invalid bounding boxes detected!")

    safe_ray_dirs = to.where(
        ray_dirs_exp != 0, ray_dirs_exp, to.full_like(ray_dirs_exp, 1e-8)
    )
    inv_dirs = 1.0 / safe_ray_dirs

    t1 = (min_corners_exp - ray_oris_exp) * inv_dirs
    t2 = (max_corners_exp - ray_oris_exp) * inv_dirs

    t_min = to.minimum(t1, t2)
    t_max = to.maximum(t1, t2)

    t_entry = t_min.amax(dim=2)
    t_exit = t_max.amin(dim=2)

    # Determine hit condition and select t_hit
    hit = (t_exit > t_entry) & (t_exit > 0)
    t_hit = to.where(t_entry > 0, t_entry, t_exit)

    # Count the hits to see if there's a problem
    total_hits = hit.sum().item()
    if total_hits == 0:
        print("Warning: No ray-box intersections found!")
        # Return default values instead of NaNs
        return to.ones(ray_oris.shape[0], device=ray_oris.device), to.zeros(
            (ray_oris.shape[0], min_corners.shape[0]), device=ray_oris.device
        )

    # Replace inf values with a large but finite number
    max_valid = t_hit[hit].max().item() if hit.any() else 1000.0
    t_hit = to.where(hit, t_hit, to.full_like(t_hit, max_valid * 10))

    # Scale t_hit to avoid extremely large values in softmax
    t_scaled = t_hit / max(1.0, t_hit.abs().max().item())

    # Apply a softmax with more stable temperature
    weights = to.softmax(-t_scaled / temperature, dim=1)

    # Compute a weighted t-value for each ray, avoiding inf*0 which causes NaNs
    soft_tvals = to.sum(
        weights * to.where(to.isinf(t_hit), to.zeros_like(t_hit), t_hit), dim=1
    )

    return soft_tvals, weights


def rotation_matrix_from_vectors(a, b, eps=1e-8):
    """
    Compute the rotation matrix that maps vector a to vector b using Rodrigues' formula.
    a and b should be 1D tensors of shape (3,) and normalized.
    """
    a = a / to.norm(a)
    b = b / to.norm(b)
    v = to.cross(a, b)
    c = to.dot(a, b)
    s = to.norm(v)
    if s < eps:
        # if vectors are parallel or anti-parallel
        if c > 0:
            return to.eye(3, device=a.device)
        else:
            # If a and b are opposite, choose an arbitrary orthogonal axis.
            # Here, we choose an axis orthogonal to a.
            orthogonal = to.tensor([1.0, 0.0, 0.0], device=a.device)
            if abs(a[0]) > 0.9:
                orthogonal = to.tensor([0.0, 1.0, 0.0], device=a.device)
            v = to.cross(a, orthogonal)
            v = v / to.norm(v)
            vx = skew_symmetric(v)
            return to.eye(3, device=a.device) + 2 * vx @ vx
    vx = skew_symmetric(v)
    R = to.eye(3, device=a.device) + vx + (vx @ vx) * ((1 - c) / (s**2))
    return R


def batched_rotation_matrices(A, B, eps=1e-8):
    # A, B have shape (R, N, 3)
    R_mats = []
    R, N = A.shape[0], A.shape[1]
    for r in range(R):
        mats = []
        for n in range(N):
            mat = rotation_matrix_from_vectors(A[r, n], B[r, n], eps)
            mats.append(mat[None, ...])  # shape: (1, 3, 3)
        R_mats.append(to.cat(mats, dim=0)[None, ...])
    return to.cat(R_mats, dim=0)  # shape: (R, N, 3, 3)


def create_complete_rotation(v_original, v_target):
    # Normalize inputs
    v_original = v_original / to.norm(v_original)
    v_target = v_target / to.norm(v_target)

    # First, find the rotation axis and angle
    rotation_axis = to.cross(v_original, v_target)
    rotation_axis_norm = to.norm(rotation_axis)

    # If vectors are nearly parallel, choose a perpendicular axis
    if rotation_axis_norm < 1e-6:
        # Check if they're nearly the same or opposite
        if to.dot(v_original, v_target) > 0:
            # Nearly the same, no rotation needed
            return to.eye(3, device=v_original.device)
        else:
            # Nearly opposite, rotate 180° around perpendicular axis
            # Find a consistent perpendicular axis
            if abs(v_original[0]) < abs(v_original[1]):
                rotation_axis = to.tensor([1.0, 0.0, 0.0], device=v_original.device)
            else:
                rotation_axis = to.tensor([0.0, 1.0, 0.0], device=v_original.device)
            rotation_axis = (
                rotation_axis - to.dot(rotation_axis, v_original) * v_original
            )
            rotation_axis = rotation_axis / to.norm(rotation_axis)

    else:
        rotation_axis = rotation_axis / rotation_axis_norm

    # Compute rotation angle
    cos_angle = to.clamp(to.dot(v_original, v_target), -1.0, 1.0)
    angle = to.acos(cos_angle)

    # Build rotation matrix using Rodrigues' formula
    K = skew_symmetric(rotation_axis)
    R = (
        to.eye(3, device=v_original.device)
        + to.sin(angle) * K
        + (1 - cos_angle) * (K @ K)
    )

    return R


def align_normal_with_scipy(old_normal, new_normal):
    """
    Compute the rotation matrix that aligns old_normal to new_normal using SciPy.

    Args:
        old_normal: torch.Tensor of shape (3,) representing the reference normal.
        new_normal: torch.Tensor of shape (3,) representing the target normal.

    Returns:
        R: torch.Tensor of shape (3,3) rotation matrix.
        rmsd: Root mean squared deviation from the alignment (float).
    """
    # Ensure the vectors are normalized
    old_normal = old_normal / (old_normal.norm() + 1e-8)
    new_normal = new_normal / (new_normal.norm() + 1e-8)

    # Convert to numpy and reshape to (1,3) as align_vectors expects an array of vectors.
    old_np = old_normal.detach().cpu().numpy().reshape(1, 3)
    new_np = new_normal.detach().cpu().numpy().reshape(1, 3)

    # align_vectors returns the rotation that aligns the source (old) to the target (new)
    rot, rmsd = Rotation.align_vectors(new_np, old_np)
    return rot, rmsd


def plot_points(positions, colors=None, point_size=10):
    """
    Plot a set of 3D points using Vispy.

    Args:
        positions: torch.Tensor or numpy array of shape (N, 3) containing point positions
        colors: Optional color array. If None, points are white. Can be:
            - Single RGB/RGBA tuple (applies to all points)
            - (N, 3) or (N, 4) array of RGB/RGBA values
        point_size: Size of points to render

    Returns:
        context: The RenderContext object for further customization
    """
    from vispy import scene, app
    import numpy as np

    # Convert to numpy if tensor
    if isinstance(positions, to.Tensor):
        positions = positions.detach().cpu().numpy()

    # Default color (white) if none provided
    if colors is None:
        colors = (1.0, 1.0, 1.0, 1.0)  # White with full opacity

    # Create visualization context
    canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="black")
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(fov=45)

    # Create scatter plot
    scatter = scene.visuals.Markers(parent=view.scene)
    scatter.set_data(positions, edge_color=None, face_color=colors, size=point_size)

    # Add axes for reference
    axis = scene.visuals.XYZAxis(parent=view.scene)

    # Set view range
    view.camera.set_range()

    # Return the canvas so it stays alive
    app.run()

    return canvas


def huber_loss(predictions, targets, delta=1.0):
    """
    Compute the Huber loss between predictions and targets.

    Args:
        predictions (torch.Tensor): The predicted values.
        targets (torch.Tensor): The ground truth values.
        delta (float): The point where the loss changes from quadratic to linear.

    Returns:
        torch.Tensor: The computed Huber loss.
    """
    error = predictions - targets
    abs_error = to.abs(error)
    quadratic = to.minimum(abs_error, to.tensor(delta, device=error.device))
    linear = abs_error - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    return loss.mean()


def rotate_bounding_box(box_min: to.Tensor, box_max: to.Tensor, rotation: Rotation):
    """
    Apply a SciPy rotation to a bounding box defined by box_min and box_max.

    Args:
        box_min (torch.Tensor): Tensor of shape (3,) for the minimum corner.
        box_max (torch.Tensor): Tensor of shape (3,) for the maximum corner.
        rotation (Rotation): A SciPy Rotation object representing the rotation.

    Returns:
        new_box_min (torch.Tensor): Tensor of shape (3,) for the new minimum corner.
        new_box_max (torch.Tensor): Tensor of shape (3,) for the new maximum corner.
    """
    # Compute the center of the bounding box.
    center = (box_min + box_max) / 2

    # Compute all 8 vertices of the bounding box.
    vertices = to.tensor(
        [
            [box_min[0], box_min[1], box_min[2]],
            [box_min[0], box_min[1], box_max[2]],
            [box_min[0], box_max[1], box_min[2]],
            [box_min[0], box_max[1], box_max[2]],
            [box_max[0], box_min[1], box_min[2]],
            [box_max[0], box_min[1], box_max[2]],
            [box_max[0], box_max[1], box_min[2]],
            [box_max[0], box_max[1], box_max[2]],
        ],
        device=box_min.device,
        dtype=box_min.dtype,
    )

    # Translate vertices so that rotation happens around the center.
    vertices_centered = vertices - center

    # Convert vertices to NumPy for applying the SciPy rotation.
    vertices_np = vertices_centered.detach().cpu().numpy()
    rotated_vertices_np = rotation.apply(vertices_np)

    # Convert the rotated vertices back to a Torch tensor.
    rotated_vertices = to.from_numpy(rotated_vertices_np).to(
        box_min.device, dtype=box_min.dtype
    )

    # Translate the vertices back to the original coordinate space.
    rotated_vertices = rotated_vertices + center

    # Compute the new bounding box corners from the rotated vertices.
    new_box_min = rotated_vertices.min(dim=0)[0]
    new_box_max = rotated_vertices.max(dim=0)[0]
    return new_box_min, new_box_max


def get_new_t_values(t_values, ray_oris, weights, neighbour_indices):
    time_step = 0.05
    weighted_neighbours = to.sum(weights * t_values[neighbour_indices], dim=1)
    diff_neighbours = t_values - weighted_neighbours
    t_values -= diff_neighbours * time_step
    print(t_values)
    return t_values


def great_circle_distance(points1, points2, radius=2.0):
    # Normalize points to ensure they are on the unit sphere
    points1_normalized = points1 / to.norm(points1, dim=1, keepdim=True)
    points2_normalized = points2 / to.norm(points2, dim=1, keepdim=True)

    # Compute the dot product between all pairs
    # This gives the cosine of the angle between the points
    cos_angle = to.matmul(points1_normalized, points2_normalized.transpose(0, 1))

    # Clip values to avoid numerical issues
    cos_angle = to.clamp(cos_angle, -1.0, 1.0)

    # Convert to angle (in radians)
    angle = to.acos(cos_angle)

    # Convert angle to distance along great circle
    distance = radius * angle

    return distance


def get_laplacian(ray_oris):
    K = 16
    distances = compute_pairwise_great_circle(ray_oris, 3.0)
    neighbour_distances, neighbour_indices = to.topk(distances, K, largest=False)
    weights = to.exp(-(neighbour_distances**2 * 3.9))
    N = distances.shape[0]
    W = to.zeros((N, N), device=device)
    rows = to.arange(N).unsqueeze(-1).expand(-1, K)
    W[rows, neighbour_indices] = weights
    D = to.sum(W, dim=1)
    L = to.diag(D) - W
    return L


class SphereSignalVisualize:
    def __init__(self, t_values, ray_oris, ray_dirs):
        # Initialise canvas
        self.canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="white")
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera()
        self.scatter = scene.visuals.Markers()
        self.ray_oris = ray_oris.numpy()
        self.ray_dirs = ray_dirs.numpy()
        self.values = t_values.numpy()
        self.positions = self.ray_oris + self.values[..., None] * self.ray_dirs

        colors = self.get_colours(self.values)
        self.scatter.set_data(self.positions, face_color=colors)
        self.view.add(self.scatter)

        K = 16
        distances = compute_pairwise_great_circle(ray_oris, 3)
        neighbour_distances, neighbour_indices = to.topk(distances, K, largest=False)
        weights = to.exp(-((neighbour_distances * 3.9) ** 2))
        N = self.positions.shape[0]
        W = to.zeros((N, N))
        rows = to.arange(N).unsqueeze(-1).expand(-1, K)
        W[rows, neighbour_indices] = weights
        D = to.sum(W, dim=1)
        L = to.diag(D) - W
        I = to.eye(N)
        self.A = I + 0.01 * L
        self.laplacian = L
        self.A_sparse = sp.csr_matrix(self.A.cpu().numpy())
        self.render()

    def get_colours(self, values):
        norm = Normalize(vmin=np.min(values), vmax=np.max(values))
        normalized_values = norm(values)
        cmap = plt.get_cmap("viridis")
        colors = cmap(normalized_values)
        return colors

    def update(self):
        new_values = spla.spsolve(self.A_sparse, self.values)
        self.values = new_values
        self.positions = self.ray_oris + self.values[..., None] * self.ray_dirs
        self.render()

    def render(self):
        colors = self.get_colours(self.values)
        self.scatter.set_data(self.positions, face_color=colors)

        self.canvas.update()


if __name__ == "__main__":
    model = GaussianParameters("car.ply")
    train_model(model)
    raise Exception
    n_of_rays = 5000
    radius = 3

    sphere_centre = to.tensor(
        [to.mean(model.means[0]), to.mean(model.means[1]), to.mean(model.means[2])],
        device=device,
    )

    sphere_params = (sphere_centre, radius, n_of_rays)
    # model.ray_oris, model.ray_dirs = generate_fibonacci_sphere_rays(*sphere_params)
    model.ray_oris, model.ray_dirs = generate_rays_from_points(model.means, radius)
    t_values = model.project_2()
    app.create()  # Create the VisPy app without blocking
    vizzer = SphereSignalVisualize(
        t_values.squeeze().detach().cpu(),
        model.ray_oris.detach().cpu(),
        model.ray_dirs.detach().cpu(),
    )

    while True:
        app.process_events()  # Process events periodically
        vizzer.update()
        time.sleep(0.01)  # Small delay to avoid high CPU usageapp.run()

    # plot_sphere_signal(t_values.detach().cpu(), ray_oris.detach().cpu())
    # losses = compute_graph_laplacian_loss(ray_oris + model.project_2() * ray_dirs)

    """
    positions = model.ray_oris + model.ray_dirs * model.project_2()
    for i in range(300):
        new_positions = implicit_backward_euler_laplacian_smooth(
            positions, 0.005, 16)

        positions = new_positions
        print(to.sum(compute_graph_laplacian_loss(positions, 16)))

    print(positions)

    losses = to.ones(positions.shape[0])
    residual_np = losses.detach().cpu().numpy()
    scalar_residual = residual_np
    norm_res = (scalar_residual - scalar_residual.min()) / \
                (scalar_residual.max() - scalar_residual.min() + 1e-8).squeeze()
    colors = cm.viridis(norm_res)



    colors = np.array(colors)


    # Ensure colors shape matches positions (N, 4)
    positions_np = positions.detach().cpu().numpy()

    # Create your render context and update scatter data
    context = RenderContext(point_size=7)
    context.scatter.set_data(
        positions_np, face_color=colors, size=context.point_size)
    app.run()
    # Shape match visualisation

    raise Exception
    print(model.ray_oris.shape)


    """

    # test_scenario(min_corners, max_corners, model.gaussian_model.normals, new_normals)
    # train_model(model=model)
    # Get spatial grid parameters
    # Call optimized projection functi
