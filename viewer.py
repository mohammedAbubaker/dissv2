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

import cupy as cp
module = cp.RawModule(code=kernel_code)
kernel = module.get_function("ray_aabb_intersect_top16")
def broadcast(gauss_batch, ray_batch):
    # Split up gauss_batch
    means, covariances, opacities, normals, reference_normals = gauss_batch
    # Split up ray_batch
    ray_oris, ray_dirs = ray_batch

    R = ray_oris.shape[0]
    G = means.shape[0]

    bcast_ray_oris = ray_oris.unsqueeze(1)
    # (N_, dim=1rays, 1, 3)
    bcast_ray_dirs = ray_dirs.unsqueeze(1)
    # (1, N_gaussians, 3)
    bcast_means = means.unsqueeze(0)
    # (1, N_gaussians, 3)
    bcast_covariances = covariances.unsqueeze(0)
    # (1, N_gaussians)
    bcast_opacities = opacities.unsqueeze(0)
    # (1, N_gaussians, 3)
    bcast_normals = normals.unsqueeze(0)
    # (1, N_gaussians, 3)
    bcast_reference_normals = reference_normals.unsqueeze(0)

    return (
        bcast_means,
        bcast_covariances,
        bcast_opacities,
        bcast_normals,
        bcast_reference_normals,
        bcast_ray_oris,
        bcast_ray_dirs,
    )


def generate_fibonacci_sphere_rays(center, radius, n, jitter_scale=0.01):
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
    indices = to.arange(0, n, dtype=to.float32)

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


def compute_graph_laplacian(points, scale=5.0):
   # Get pairwise Euclidean distances (N x N)
    dists = compute_pairwise_euclidean(points)
    
    # Compute weight matrix, applying the exponential decay.
    W = to.exp(-dists / scale)
    
    # Remove self connections by zeroing the diagonal.
    n = points.shape[0]
    eye = to.eye(n, device=points.device)
    W = W * (1 - eye)
    
    # Build degree matrix D.
    D = to.diag(W.sum(dim=1))
    
    # Compute Laplacian L = D - W.
    L = D - W
    
    return L


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
        self.scales_d = to.eye(3)[None, :, :].to(
            device) * (self.scales_exp)[:, :, None]
        self.scales_d **= 2
        self.scales_i_d = (
            to.eye(3)[None, :, :].to(device) *
            (1 / self.scales_exp)[:, :, None]
        )
        self.scales_i_d **= 2
        self.rotations_t = self.rotations.transpose(-1, -2)
        self.scales_d_t = self.scales_d.transpose(-1, -2)
        self.covariances = self.rotations @ self.scales_d @ self.rotations_t

        min_indices = self.scales_exp.argmin(axis=1)
        self.normals = self.rotations[to.arange(
            self.n_gaussians), :, min_indices]
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
    )
    evaluations = gaussian_opacities * to.exp(exponent).squeeze(-1)
    return evaluations


def skew_symmetric(v):
    row1 = to.stack([to.zeros_like(v[..., 0]), -v[..., 2], v[..., 1]], dim=-1)
    row2 = to.stack([v[..., 2], to.zeros_like(v[..., 1]), -v[..., 0]], dim=-1)
    row3 = to.stack([-v[..., 1], v[..., 0], to.zeros_like(v[..., 2])], dim=-1)
    K = to.stack([row1, row2, row3], dim=-2)
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
    # Given 2 RxNx3 vectors a and b, return an RxNx3x3 rotation matrix
    a_dot_b = (a[:, :, None, :] @ b[..., None]).squeeze(-1).squeeze(-1)
    a_norm = to.linalg.norm(a)
    b_norm = to.linalg.norm(b, dim=2)
    angle = to.acos((a_dot_b / (a_norm * b_norm)))
    v = to.cross(a, b)
    s = to.norm(v, dim=2) * to.sin(angle)
    c = a_dot_b * to.cos(angle)
    i = to.eye(3).to(device="cuda").tile(a.shape[0], a.shape[1], 1, 1)
    v_skew = skew_symmetric(v)
    last_term = 1 / (1 + c)
    return i + v_skew + (v_skew @ v_skew) * last_term[..., None, None]


def get_max_responses_and_tvals(
    ray_oris, means, covs, ray_dirs, opacities, normals, old_normals
):
    new_rotations = normals_to_rot_matrix(old_normals, normals)
    new_covs = new_rotations @ covs @ new_rotations.transpose(-2, -1)
    covs_reg = covs + to.eye(3, device=covs.device).unsqueeze(0).unsqueeze(0) * 1e-4
    inv_covs = to.linalg.inv(covs_reg)
    rg_diff = means - ray_oris
    inv_cov_d = inv_covs @ ray_dirs[..., None]
    numerator = (rg_diff[:, :, None, :] @ inv_cov_d).squeeze(-1)
    denomenator = (ray_dirs[:, :, None, :] @ inv_cov_d).squeeze(-1)
    t_values = numerator / (denomenator + 1e-8)  # Increase epsilon to 1e-5 or more
    best_positions = ray_oris + t_values * ray_dirs
    max_responses = evaluate_points(best_positions, means, inv_covs, opacities)

    return max_responses, t_values


class GaussianParameters(nn.Module):
    def __init__(self, path):
        super(GaussianParameters, self).__init__()
        self.gaussian_model = GaussianModel(path)
        self.means = nn.Parameter(self.gaussian_model.means)
        self.normals = self.gaussian_model.normals
        ray_oris, ray_dirs = generate_fibonacci_sphere_rays(
            to.tensor([0.0, 0.0, 0.0]), 1, 20000
        )
        self.ray_oris = ray_oris.to(device)
        self.ray_dirs = ray_dirs.to(device)
        self.laplacian = compute_graph_laplacian(ray_oris).to(device)

    def forward(self):
        return self.means
    
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
        '''
        new_rotations = normals_to_rot_matrix(
            self.gaussian_model.reference_normals[None,
                                                  :], self.normals[None, :]
        )
        new_rotations = new_rotations.squeeze(0)
        '''
        # Expand rotations to match the number of vertices (2)
        rotation_expanded = self.gaussian_model.rotations.unsqueeze(
            1)  # [N, 1, 3, 3]
        # [N, 2, 3, 3]

        rotation_expanded = rotation_expanded.expand(-1, 2, -1, -1)

        # Now do the matrix multiplication
        rotated_vertices = (
            rotation_expanded @ scaled_vertices[..., None]
        )  # [N, 2, 3, 1]
        rotated_vertices = rotated_vertices.squeeze(-1)  # [N, 2, 3]

        # Finally translate
        translated = rotated_vertices + self.means[:, None, :]
        return translated.min(dim=1).values, translated.max(dim=1).values
    
    def get_top_16(self):
        ray_ori = self.ray_oris    # [num_rays, 3]
        ray_dir = self.ray_dirs     
        min_corners, max_corners = self.create_bounding_boxes()
        num_rays = ray_ori.shape[0]
        num_boxes = min_corners.shape[0]

        # Create an output tensor for indices
        out_indices = to.full((num_rays, 16), -1, device=ray_ori.device, dtype=to.int32)

        # Get device pointers
        ray_ori_ptr      = ray_ori.data_ptr()
        ray_dir_ptr      = ray_dir.data_ptr()
        min_corners_ptr  = min_corners.data_ptr()
        max_corners_ptr  = max_corners.data_ptr()
        out_indices_ptr  = out_indices.data_ptr()

        threads_per_block = 256
        blocks = (num_rays + threads_per_block - 1) // threads_per_block

        # Launch the kernel
        kernel((blocks,), (threads_per_block,), 
            (ray_ori_ptr, ray_dir_ptr, min_corners_ptr, max_corners_ptr,
                out_indices_ptr, num_rays, num_boxes))
        
        return out_indices

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

    def final_project(self):
        spatial_args = self.get_spatial_args()
        gx, gy, gz = self.get_spatial_index(self.means, spatial_args)
        g_hashes = self.get_spatial_hashes(gx, gy, gz, spatial_args)
        hash_bins = to.bincount(g_hashes)
        H = spatial_args["num_cells"]
        B_cell = int(to.max(hash_bins).item())
        unique_hashes, inverse_indices, counts = to.unique(
            g_hashes, return_inverse=True, return_counts=True
        )
        grouped_indices = {
            h.item(): to.where(inverse_indices == i)[0]
            for i, h in enumerate(unique_hashes)
        }
        hash_to_g = to.full((H, B_cell), -1, dtype=to.int, device=device)
        for h, indices in grouped_indices.items():
            hash_to_g[h, : indices.numel()] = indices
        cell_mins, cell_maxs = self.compute_cell_bounding_volumes(spatial_args)
        ray_cell_mask = self.ray_box_intersections_binary(
            cell_mins, cell_maxs, self.ray_oris, self.ray_dirs
        )
        R = self.ray_oris.shape[0]
        blended_tvals = to.zeros(R, device=device)
        for r in range(R):
            cell_indices = to.nonzero(ray_cell_mask[r], as_tuple=True)[0]
            gaussian_indices = hash_to_g[cell_indices].flatten(-2, -1)
            valid_gaussians = gaussian_indices[gaussian_indices != -1].unique()
            valid_gaussians = valid_gaussians[None, ...].long()
            blended_tvals[r] = self.blend_tvals(r, valid_gaussians)

        plt.plot(blended_tvals.detach().cpu())
        plt.plot(to.sum(ray_cell_mask, dim=1).detach().cpu())
        plt.show()

        return blended_tvals

    def get_contributions(self, alphas, tvals):
        # Sort by t-values (depth) to properly handle occlusion
        sorted_indices = to.argsort(tvals, dim=1)
        sorted_alphas = alphas.gather(dim=-1, index=sorted_indices)
        sorted_tvals = tvals.gather(dim=-1, index=sorted_indices)

        # Calculate transmittance (how much light passes through)
        alphas_complement = 1 - sorted_alphas
        transmittance = to.cumprod(alphas_complement, dim=1)

        # First value starts with full visibility
        shifted = to.ones_like(transmittance)
        shifted[:, 1:] = transmittance[:, :-1]

        # Weight is visibility * opacity
        weights = shifted * sorted_alphas

        # Normalize weights to sum to 1
        weights_sum = weights.sum(dim=1, keepdim=True)
        eps = 1e-8  # Prevent division by zero
        normalized_weights = weights / (weights_sum + eps)

        # Map back to original order
        inv_indices = sorted_indices.argsort(dim=1)
        return normalized_weights.gather(dim=1, index=inv_indices)

    def blend_tvals(self, ray_idx, ray_gaussian_indices):
        max_responses, t_vals = get_max_responses_and_tvals(
            self.ray_oris[ray_idx, None, None, :],
            self.means[ray_gaussian_indices],
            self.gaussian_model.covariances[ray_gaussian_indices],
            self.ray_dirs[ray_idx, None, None, :],
            self.gaussian_model.opacities[ray_gaussian_indices],
            self.gaussian_model.reference_normals[ray_gaussian_indices],
            self.gaussian_model.reference_normals[ray_gaussian_indices],
        )

        t_vals = t_vals.squeeze(-1)
        max_responses = max_responses.squeeze(-1)
        top_responses, top_responses_idxs = max_responses.squeeze(
            -1).topk(16, dim=-1)

        gathered_tvals = t_vals.gather(-1, top_responses_idxs)
        # Calculate contribution
        contributions = self.get_contributions(top_responses, gathered_tvals)
        blended_tvals = to.sum(contributions * gathered_tvals, dim=-1)
        return blended_tvals

    def compute_cell_bounding_volumes(self, spatial_args):
        """
        Given spatial_args containing grid dimensions, minimum corner, and spacings,
        compute the axis-aligned bounding boxes (AABBs) for each cell.

        Args:
            spatial_args (dict): A dictionary with keys:
                - "n_x", "n_y", "n_z": Grid dimensions.
                - "min_corner": Tensor of shape [3] for the overall minimum corner.
                - "spacings": Tensor of shape [3] representing cell sizes along each dimension.

        Returns:
            box_mins (Tensor): [num_cells, 3] tensor of minimum corners of each cell.
            box_maxs (Tensor): [num_cells, 3] tensor of maximum corners of each cell.
        """
        n_x, n_y, n_z = spatial_args["n_x"], spatial_args["n_y"], spatial_args["n_z"]
        device = spatial_args["min_corner"].device
        xs = to.arange(n_x, device=device)
        ys = to.arange(n_y, device=device)
        zs = to.arange(n_z, device=device)

        grid_i, grid_j, grid_k = to.meshgrid(xs, ys, zs, indexing="ij")
        grid_i = grid_i.flatten()
        grid_j = grid_j.flatten()
        grid_k = grid_k.flatten()

        # Each cell's minimum corner = global min_corner + (grid indices * cell spacings)
        cell_mins = (
            spatial_args["min_corner"].unsqueeze(0)
            + to.stack([grid_i, grid_j, grid_k], dim=1) *
            spatial_args["spacings"]
        )
        # Each cell's maximum corner = cell minimum + cell spacings
        cell_maxs = cell_mins + spatial_args["spacings"]

        return cell_mins, cell_maxs

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
        intersects = (t_max >= to.maximum(
            t_min, to.zeros_like(t_min))) & (t_max >= 0)
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

    def visualize_gaussians_by_hash(self):
        """
        Visualize Gaussian means where each one is colored based on its spatial hash.
        This helps debug/visualize the spatial partitioning scheme.
        """
        # Get spatial args for hashing
        spatial_args = self.get_spatial_args()

        # Get spatial indices and hashes for each Gaussian mean
        gx, gy, gz = self.get_spatial_index(self.means, spatial_args)
        g_hashes = self.get_spatial_hashes(gx, gy, gz, spatial_args)

        # Get unique hashes and assign a normalized value to each
        unique_hashes = to.unique(g_hashes)
        num_hashes = len(unique_hashes)
        hash_to_idx = {h.item(): i for i, h in enumerate(unique_hashes)}

        # Map each hash to a normalized index for coloring
        hash_indices = to.tensor(
            [hash_to_idx[h.item()] for h in g_hashes], device=device, dtype=to.float32
        )
        hash_indices = hash_indices / \
            max(1, num_hashes - 1)  # Normalize to [0,1]

        # Create colors using HSV color space for maximum distinction
        import colorsys

        colors = np.zeros((len(self.means), 4), dtype=np.float32)
        hash_indices_np = hash_indices.cpu().numpy()

        # Generate visually distinct colors for each hash value
        for i, idx in enumerate(hash_indices_np):
            # Convert to RGB from HSV (hue from normalized hash, full saturation/value)
            r, g, b = colorsys.hsv_to_rgb(idx, 0.8, 0.9)
            colors[i] = [r, g, b, 1.0]  # RGBA

        # Convert means to numpy for visualization
        means_np = self.means.detach().cpu().numpy()

        # Create and configure visualization context
        # Slightly larger points for visibility
        context = RenderContext(point_size=8)
        context.scatter.set_data(
            means_np, face_color=colors, size=context.point_size)
        context.canvas.update()

        print(
            f"Visualizing {len(self.means)} Gaussians in {num_hashes} hash buckets")
        return context

    def project_spatial_vectorized_2(self):
        """
        For each ray, uses the grid (from get_spatial_args) to
        – build a hash table mapping grid cells to gaussian indices,
        – sample points along each ray (using ray–box intersections),
        – query candidate Gaussians by hashing the sample positions,
        – and then compute a blended t-value via response weighting.

        Returns:
            out_blended_tvals: Tensor of shape [N_rays] with a blended t-value per ray.
        """
        # Get spatial grid parameters.
        spatial_args = self.get_spatial_args()
        num_cells = spatial_args["num_cells"]

        # -----------------------------
        # Build hash table for Gaussians.
        # -----------------------------
        # self.means is [G,3]. Get cell indices (assume get_spatial_index accepts [G,3]).
        gx, gy, gz = self.get_spatial_index(self.means, spatial_args)
        g_hashes = self.get_spatial_hashes(gx, gy, gz, spatial_args)  # [G]
        # Build a mapping from each cell (0...num_cells-1) to a list of gaussian indices.
        # We use a simple loop over unique cell values (num_cells is small, e.g. 1000 max).
        # Use bincount/unique to know how many Gaussians go in each cell.
        unique_hashes, inv, counts = to.unique(
            g_hashes, return_inverse=True, return_counts=True
        )
        hash_to_g = to.full((num_cells, 32), -1, dtype=to.int64, device=device)

        for h in unique_hashes.tolist():
            # Find indices in self.means assigned to cell h.
            idxs = to.where(g_hashes == h)[0]
            # Cap at 32 Gaussians per cell - take only the first 32
            if idxs.numel() > 32:
                idxs = idxs[:32]
            # Now this will never exceed 32 elements
            hash_to_g[h, : idxs.numel()] = idxs

        # -----------------------------
        # Query rays.
        # -----------------------------
        # Compute ray-box intersections (t_nears and t_fars for each ray)
        t_nears, t_fars = self.ray_box_intersections(spatial_args)  # [N]
        # Use a fixed number of steps along each ray.
        steps = 10
        lin_base = to.linspace(0, 1, steps, device=device)  # [steps]
        sample_t_vals = t_nears.unsqueeze(1) + (t_fars - t_nears).unsqueeze(
            1
        ) * lin_base.unsqueeze(0)  # [N,steps]
        # Compute sample positions along each ray.
        # self.ray_oris: [N,3], self.ray_dirs: [N,3]
        sample_positions = self.ray_oris.unsqueeze(1) + self.ray_dirs.unsqueeze(
            1
        ) * sample_t_vals.unsqueeze(-1)  # [N,steps,3]

        # Get cell indices for every sample position.
        rx, ry, rz = self.get_spatial_index(
            sample_positions, spatial_args
        )  # each of shape [N,steps]
        ray_hashes = self.get_spatial_hashes(
            rx, ry, rz, spatial_args)  # [N,steps]

        # For each sample, retrieve candidate gaussian indices.
        # hash_to_g has shape [num_cells, 32] so we index it with ray_hashes.
        # This gives: candidates: [N, steps, 32].
        candidates = hash_to_g[ray_hashes.long()]  # [N,steps,32]
        N = self.ray_oris.shape[0]
        num_candidates = steps * candidates.shape[-1]
        candidates = candidates.reshape(N, num_candidates)  # [N, steps*32]

        # Build a mask of valid candidates (those != -1)
        valid_mask = candidates != -1
        # Clamp candidates for safe indexing:
        candidates_clamped = candidates.clamp(min=0)

        # -----------------------------
        # Gather candidate attributes.
        # -----------------------------
        # For each candidate index, get Gaussian attributes
        cand_means = self.means[candidates_clamped]  # [N, num_candidates, 3]
        cand_covs = self.gaussian_model.covariances[
            candidates_clamped
        ]  # [N, num_candidates, 3, 3]
        cand_opacities = self.gaussian_model.opacities[
            candidates_clamped
        ]  # [N, num_candidates]
        # [N, num_candidates, 3]
        cand_normals = self.normals[candidates_clamped]
        cand_ref_normals = self.gaussian_model.reference_normals[
            candidates_clamped
        ]  # [N, num_candidates, 3]

        # Expand ray origins/directions to align with candidate dimension.
        rays_exp = self.ray_oris.unsqueeze(1).expand(
            -1, num_candidates, -1
        )  # [N, num_candidates, 3]
        dirs_exp = self.ray_dirs.unsqueeze(1).expand(
            -1, num_candidates, -1
        )  # [N, num_candidates, 3]

        # -----------------------------
        # Compute responses and t-values.
        # -----------------------------
        # get_max_responses_and_tvals expects:
        #   ray_oris [N, num_candidates, 3],
        #   candidate means, covariances, etc.
        responses, t_vals = get_max_responses_and_tvals(
            rays_exp,
            cand_means,
            cand_covs,
            dirs_exp,
            cand_opacities,
            cand_normals,
            cand_ref_normals,
        )  # both: [N, num_candidates]
        responses = responses.squeeze(-1)
        t_vals = t_vals.squeeze(-1)
        # Set response=0 for candidates that were not valid.
        responses = responses * valid_mask.to(responses.dtype)
        # Then compute contributions using the original responses
        contributions = self.get_contributions(responses, t_vals)
        blended_tvals = (contributions * t_vals).sum(dim=1)  # [N]

        # For rays that did not hit the box, output 0.
        out_blended_tvals = to.zeros(self.ray_oris.shape[0], device=device)
        valid_rays = to.isfinite(t_nears)
        out_blended_tvals[valid_rays] = blended_tvals[valid_rays]
        return out_blended_tvals
    
 
    def project_optimized(
        self,
        min_corner,
        grid_dims,
        spacings
    ):
        means = self.means
        ray_oris = self.ray_oris
        ray_dirs = self.ray_dirs
        covariances = self.gaussian_model.covariances
        opacities = self.gaussian_model.opacities
        normals = self.normals
        reference_normals = self.gaussian_model.reference_normals

        # 1. Setup spatial grid parameters
        num_cells = grid_dims[0] * grid_dims[1] * grid_dims[2]
        
        # 2. Hash all Gaussians into cells
        g_idxs_x = ((means[:, 0] - min_corner[0]) / spacings[0]).int().clamp(0, grid_dims[0]-1)
        g_idxs_y = ((means[:, 1] - min_corner[1]) / spacings[1]).int().clamp(0, grid_dims[1]-1)
        g_idxs_z = ((means[:, 2] - min_corner[2]) / spacings[2]).int().clamp(0, grid_dims[2]-1)
        
        # Hash function: Using prime number multiplication for good distribution
        g_hashes = ((g_idxs_x * 8191) ^ (g_idxs_y * 131071) ^ (g_idxs_z * 524287)) % num_cells
        
        # 3. Prepare output arrays
        R = ray_oris.shape[0]
        blended_tvals = to.zeros(R, device=ray_oris.device)
        
        # 4. Process each ray individually (parallelized by JIT)
        for r in range(R):
            # 5. Calculate ray-box intersections
            inv_dir = 1.0 / to.where(ray_dirs[r] != 0, ray_dirs[r], to.tensor(1e-8, device=ray_dirs.device))
            t1 = (min_corner - ray_oris[r]) * inv_dir
            t2 = (min_corner + spacings * grid_dims - ray_oris[r]) * inv_dir
            
            t_near = to.max(to.minimum(t1, t2))
            t_far = to.min(to.maximum(t1, t2))
            
            # Skip if no intersection
            if t_near > t_far or t_far < 0:
                continue
                
            # 6. Sample points along the ray within the grid
            num_steps = 10
            step_size = (t_far - t_near) / num_steps
            
            # Container for candidates
            relevant_gaussians = to.zeros(means.shape[0], dtype=to.bool, device=means.device)
            
            # 7. Find all cells intersected by this ray
            for step in range(num_steps):
                t = t_near + step * step_size
                pos = ray_oris[r] + t * ray_dirs[r]
                
                # Get cell indices
                cell_x = ((pos[0] - min_corner[0]) / spacings[0]).int().clamp(0, grid_dims[0]-1)
                cell_y = ((pos[1] - min_corner[1]) / spacings[1]).int().clamp(0, grid_dims[1]-1)
                cell_z = ((pos[2] - min_corner[2]) / spacings[2]).int().clamp(0, grid_dims[2]-1)
                
                # Get cell hash
                cell_hash = ((cell_x * 8191) ^ (cell_y * 131071) ^ (cell_z * 524287)) % num_cells
                
                # Mark all Gaussians in this cell
                relevant_gaussians = relevant_gaussians | (g_hashes == cell_hash)
            
            # 8. Extract relevant Gaussians
            indices = to.where(relevant_gaussians)[0]
            
            if indices.shape[0] == 0:
                continue
                
            # 9. Compute responses only for relevant Gaussians
            ray_ori_exp = ray_oris[r:r+1, None]
            ray_dir_exp = ray_dirs[r:r+1, None]
            relevant_means = means[indices][None]
            relevant_covs = covariances[indices][None]
            relevant_opacities = opacities[indices][None]
            relevant_normals = normals[indices][None]
            relevant_ref_normals = reference_normals[indices][None]
            
            # Get responses and t-values
            responses, tvals = get_max_responses_and_tvals(
                ray_ori_exp,
                relevant_means,
                relevant_covs,
                ray_dir_exp, 
                relevant_opacities,
                relevant_normals,
                relevant_ref_normals
            )
            
            responses = responses.squeeze()
            tvals = tvals.squeeze()
            
            # Add batch dimension if needed - this fixes the dimension error
            if len(tvals.shape) == 1:
                responses = responses.unsqueeze(0)
                tvals = tvals.unsqueeze(0)
                    
            # Sort by depth for proper compositing
            sorted_idx = to.argsort(tvals, dim=1)
            sorted_alphas = responses.gather(1, sorted_idx)
            sorted_tvals = tvals.gather(1, sorted_idx)
            
            alphas_complement = 1 - sorted_alphas
            transmittance = to.cumprod(alphas_complement, dim=1)
            shifted = to.ones_like(transmittance)
            shifted[:, 1:] = transmittance[:, :-1]
            
            weights = shifted * sorted_alphas
            weights_sum = weights.sum(dim=1, keepdim=True)
            normalized_weights = weights / (weights_sum + 1e-8)
            
            # Compute final blended t-value
            blended_tvals[r] = (normalized_weights * sorted_tvals).sum()
        
        return blended_tvals

    def project(self,):
        ray_gaussian_candidates = self.get_top_16().long()

        responses, tvals = get_max_responses_and_tvals(
            self.ray_oris[:, None, :],
            self.means[ray_gaussian_candidates],
            self.gaussian_model.covariances[ray_gaussian_candidates],
            self.ray_dirs[:, None, :],
            self.gaussian_model.opacities[ray_gaussian_candidates],
            self.normals[ray_gaussian_candidates],
            self.gaussian_model.normals[ray_gaussian_candidates],
        )
        _, sorted_idx = to.sort(tvals, dim=1)
        sorted_alphas = responses.gather(dim=1, index=sorted_idx)
        alphas_compliment = 1 - sorted_alphas
        transmittance = to.cumprod(alphas_compliment, dim=1)
        shifted = to.ones_like(transmittance)
        # Fill shifted starting from the second column with the values of x's columns 0 to N-2
        shifted[:, 1:] = transmittance[:, :-1]
        # Calculate contribution
        sorted_contribution = shifted - transmittance
        # unsort the contribution
        inv_idx = sorted_idx.argsort(dim=1)
        # Reorder contribution back to the original order:
        contribution = sorted_contribution.gather(dim=1, index=inv_idx)
        blended_tvals = to.sum(contribution * tvals, dim=1)
        return blended_tvals
    
    def harmonic_loss(self):
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
        
        '''
        # Set up and solve the implicit update: (I + dt*dampening_factor*L) * f_new = f_prev.
        I = to.eye(self.laplacian.shape[0], device=self.laplacian.device)
        A = I + dt * dampening_factor * self.laplacian
        f_new = to.linalg.solve(A, self.f_prev)

        # Compute the implicit loss as the squared difference between the projection and the implicit update.
        implicit_loss = to.sum((blended_t_vals - f_new) ** 2)

        # Update f_prev for the next iteration without detaching f_new.
        self.f_prev = f_new.detach()
        '''
        # Return the total loss combining implicit and feature losses.
        return feature_loss


class RenderContext:
    def __init__(self, point_size=5, fov=45, distance=10):
        # Create a Vispy canvas with an interactive background.
        self.canvas = scene.SceneCanvas(
            keys="interactive", show=True, bgcolor="black")
        self.view = self.canvas.central_widget.add_view()
        # Use a TurntableCamera for 3D interaction.
        self.view.camera = scene.cameras.TurntableCamera(
            fov=fov, distance=distance)
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
        self.scatter.set_data(
            positions, face_color=colors, size=self.point_size)
        self.canvas.update()
        # Process pending GUI events to refresh the display immediately.
        app.process_events()


def matrix_to_quaternion(rotation_matrices):
    N = rotation_matrices.shape[0]
    q = to.zeros((N, 4), device=rotation_matrices.device)

    trace = to.einsum("nii->n", rotation_matrices)

    cond1 = trace > 0
    cond2 = (rotation_matrices[:, 0, 0] > rotation_matrices[:, 1, 1]) & ~cond1
    cond3 = (rotation_matrices[:, 1, 1] >
             rotation_matrices[:, 2, 2]) & ~(cond1 | cond2)
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
    x_large = (~w_large) & (R[:, 0, 0] > R[:, 1, 1]) & (
        R[:, 0, 0] > R[:, 2, 2])
    if x_large.any():
        S = to.sqrt(1.0 + R[x_large, 0, 0] -
                    R[x_large, 1, 1] - R[x_large, 2, 2]) * 2
        q[x_large, 0] = (R[x_large, 2, 1] - R[x_large, 1, 2]) / S
        q[x_large, 1] = 0.25 * S
        q[x_large, 2] = (R[x_large, 0, 1] + R[x_large, 1, 0]) / S
        q[x_large, 3] = (R[x_large, 0, 2] + R[x_large, 2, 0]) / S

    # Case y is largest
    y_large = (~w_large) & (~x_large) & (R[:, 1, 1] > R[:, 2, 2])
    if y_large.any():
        S = to.sqrt(1.0 + R[y_large, 1, 1] -
                    R[y_large, 0, 0] - R[y_large, 2, 2]) * 2
        q[y_large, 0] = (R[y_large, 0, 2] - R[y_large, 2, 0]) / S
        q[y_large, 1] = (R[y_large, 0, 1] + R[y_large, 1, 0]) / S
        q[y_large, 2] = 0.25 * S
        q[y_large, 3] = (R[y_large, 1, 2] + R[y_large, 2, 1]) / S

    # Case z is largest
    z_large = ~(w_large | x_large | y_large)
    if z_large.any():
        S = to.sqrt(1.0 + R[z_large, 2, 2] -
                    R[z_large, 0, 0] - R[z_large, 1, 1]) * 2
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

import threading
def visualize_depth_updates(model, update_interval=0.1):
    """
    Visualize the current depth values by plotting the points
    computed as: ray_ori + (depth * ray_dir)
    using Vispy. This function updates every 'update_interval'
    seconds.
    """
    from vispy import scene, app
    import numpy as np
    import torch as to

    # Set up the canvas and view.
    canvas = scene.SceneCanvas(keys="interactive", bgcolor="black")
    view = canvas.central_widget.add_view()
    
    # Create a markers visual for the projected points.
    scatter = scene.visuals.Markers(parent=view.scene)
    # Initialize with the ray origins to avoid None data.
    initial_points = model.ray_oris.detach().cpu().numpy() + model.f_prev.detach().cpu().numpy() * model.ray_dirs.detach().cpu().numpy()
    scatter.set_data(initial_points, face_color='red', size=8)

    # Update function called every timer tick.
    def update(event):
        # Compute new positions as: ray_ori + (depth * ray_dir)     
        new_points = model.ray_oris.detach().cpu().numpy() + model.f_prev.detach().cpu().numpy() * model.ray_dirs.detach().cpu().numpy()
        # Update scatter data.
        scatter.set_data(new_points, face_color='red', size=8)
        canvas.update()

    # Set up a timer to update the visualization periodically.
    timer = app.Timer(interval=update_interval, connect=update, start=True)

    view.camera = scene.cameras.TurntableCamera()
    view.camera.set_range()
    
    canvas.show()
    app.run()
def start_visualization(model, update_interval=0.1):
    visualize_depth_updates(model, update_interval=update_interval)

def train_model(model, num_iterations=1000, lr=0.001):
    optimizer = to.optim.Adam(model.parameters(), lr=lr)
    model.f_prev = model.project().detach()
    print
    vis_thread = threading.Thread(target=start_visualization, args=(model,))
    vis_thread.start()
    for iteration in tqdm(range(num_iterations)):
        optimizer.zero_grad()
        loss = model.harmonic_loss()
        loss.backward()
        optimizer.step()
    save_optimized_gaussian_model(
        model, output_path="optimized_point_cloud.ply")


if __name__ == "__main__":
    model = GaussianParameters("point_cloud.ply")
    train_model(model=model, num_iterations=100)
    # Get spatial grid parameters
    # Call optimized projection function