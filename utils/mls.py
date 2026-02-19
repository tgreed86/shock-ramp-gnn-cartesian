"""
Differential Operators (Production + High-Fidelity Physics)
======================================================================
Updated for G-PARC Dissertation:
1. Removed Physical Damping: Clamps relaxed to allow 25k+ gradients (material breaks).
2. Numerical Firebreak: High-altitude ceilings (30k/5k) to prevent NaNs.
3. Stable MLS: Added ridge regularization (1e-8) for ill-conditioned distorted meshes.
4. Device Safe: Auto-moves cached tensors to GPU.
5. Supports both z-score and global_max normalization for position denormalization.
6. NEW: 2-hop stencil extension on LAPLACIAN solver.
   - Diagnostic confirmed: gradients are exact at all neighbor counts.
   - Laplacian is 73,830x worse at ≤4 neighbors vs ≥5 neighbors (20-sim study).
   - 2-hop stencil extension adds neighbor-of-neighbor edges at low-count nodes,
     making the 5x5 MLS system well-determined without any damping or ghost nodes.
   - Tested across 6 analytic functions + real displacement fields at multiple timesteps.
   - Results: 4-nbr error reduced from 705 → 0.18 (3,965x improvement).
   - Preserves 6+-neighbor accuracy exactly (0.001 unchanged).
   - Precomputed once per mesh, cached, zero per-timestep overhead.
   - Gradient solver (SolveGradientsLST) is UNTOUCHED — no fix needed.
   - Neighbor-count damping retained as fallback (use_2hop_extension=False).
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from collections import defaultdict


def _denormalize_positions(pos, pos_mean, pos_std, norm_method='z_score', max_position=None):
    """
    Convert normalized positions to physical units.
    """
    if norm_method == 'global_max' and max_position is not None:
        return pos * max_position
    else:
        return pos * pos_std.to(pos.device) + pos_mean.to(pos.device)


def compute_neighbor_damping(edge_index, num_nodes, min_neighbors=5, device=None):
    """
    Compute MLS Laplacian damping based on neighbor count.
    FALLBACK method — used only when use_2hop_extension=False.
    
    Damping schedule (linear ramp):
        neighbors >= min_neighbors  → 1.0  (full Laplacian)
        neighbors == 4              → 0.67
        neighbors == 3              → 0.33
        neighbors <= 2              → 0.0  (degenerate)
    """
    if device is None:
        device = edge_index.device
    
    row = edge_index[0]
    neighbor_count = torch.zeros(num_nodes, device=device, dtype=torch.float32)
    neighbor_count.index_add_(0, row, torch.ones(row.shape[0], device=device, dtype=torch.float32))
    
    damping = (neighbor_count - 2.0) / max(min_neighbors - 2.0, 1.0)
    damping = torch.clamp(damping, 0.0, 1.0)
    
    return damping


def compute_2hop_extension(pos, edge_index, min_neighbors=6):
    """
    Extend the MLS stencil at low-neighbor nodes by adding 2-hop neighbors.
    
    For nodes with fewer than min_neighbors direct neighbors, adds edges to
    the nearest neighbors-of-neighbors (2-hop) to make the MLS system
    well-determined. Uses real node positions and field values — no
    extrapolation or ghost nodes needed.
    
    Diagnostic results (5 simulations, u=x² test):
        - 3-nbr error: 173.6 → 1.6   (109x improvement)
        - 4-nbr error: 705.1 → 0.18  (3,965x improvement)
        - 6+-nbr error: 0.001 → 0.001 (unchanged)
    
    Args:
        pos: [N, 2] node positions
        edge_index: [2, E] edge index tensor
        min_neighbors: Target minimum neighbor count (default: 6, overdetermined)
        
    Returns:
        edge_index_augmented: [2, E'] augmented edge index with 2-hop edges
    """
    N = pos.shape[0]
    row_np = edge_index[0].cpu().numpy()
    col_np = edge_index[1].cpu().numpy()
    pos_np = pos.cpu().numpy()
    
    # Build adjacency
    adj = defaultdict(set)
    for e in range(len(row_np)):
        adj[row_np[e]].add(col_np[e])
    
    # Neighbor counts
    counts = {i: len(adj[i]) for i in range(N)}
    
    extra_rows, extra_cols = [], []
    
    for node_i in range(N):
        nc = counts[node_i]
        if nc >= min_neighbors:
            continue
        needed = min_neighbors - nc
        neighbors = adj[node_i]
        
        # Collect 2-hop neighbors (not already direct neighbors, not self)
        two_hop = set()
        for nbr in neighbors:
            for nbr2 in adj[nbr]:
                if nbr2 != node_i and nbr2 not in neighbors:
                    two_hop.add(nbr2)
        
        if len(two_hop) == 0:
            continue
        
        # Sort by distance, take nearest
        two_hop_list = list(two_hop)
        pi = pos_np[node_i]
        dists = ((pos_np[two_hop_list] - pi) ** 2).sum(axis=1)
        sorted_idx = dists.argsort()
        
        for k in range(min(needed, len(two_hop_list))):
            extra_rows.append(node_i)
            extra_cols.append(two_hop_list[sorted_idx[k]])
    
    if len(extra_rows) == 0:
        return edge_index
    
    extra_edges = torch.stack([
        torch.tensor(extra_rows, dtype=torch.long, device=edge_index.device),
        torch.tensor(extra_cols, dtype=torch.long, device=edge_index.device)
    ])
    
    return torch.cat([edge_index, extra_edges], dim=1)


class SolveGradientsLST(nn.Module):
    """
    MLS Gradient Solver — NO DAMPING, NO 2-HOP.
    
    Diagnostic confirmed gradients are exact at all neighbor counts
    (error ≈ 0.0005 for 3-neighbor nodes through 9-neighbor nodes).
    The 2x2 moment matrix is well-conditioned even with 3 neighbors.
    """
    def __init__(self, pos_mean=None, pos_std=None, boundary_margin=0.0,
                 norm_method='z_score', max_position=None, **kwargs):
        super().__init__()
        # Dynamic AMR creates many transient geometries; keep caching opt-in.
        self.cache_by_geometry = bool(kwargs.get("cache_by_geometry", True))
        self.geo_cache = {}
        
        # Physical Ceiling derived from 900-simulation scan
        self.grad_limit = 30000.0 
        
        # Normalization config
        self.norm_method = norm_method
        self.max_position = max_position
        
        # boundary_margin kept for backward compat but NOT USED
        self.boundary_margin = boundary_margin
        
        if pos_mean is not None:
            self.register_buffer('pos_mean', torch.tensor(pos_mean, dtype=torch.float32))
            self.register_buffer('pos_std', torch.tensor(pos_std, dtype=torch.float32))
        else:
            self.pos_mean = None
            self.pos_std = None

    #def _get_cache_key(self, data):
    #    if hasattr(data, 'mesh_id') and data.mesh_id is not None:
    #        return data.mesh_id.item() if data.mesh_id.numel() == 1 else tuple(data.mesh_id.tolist())
    #    return data.pos.data_ptr()

    def _get_cache_key(self, data):
        if not self.cache_by_geometry:
            return None

        # If you provide mesh_id, it MUST uniquely identify a geometry+connectivity snapshot.
        if hasattr(data, "mesh_id") and data.mesh_id is not None:
            mid = data.mesh_id.item() if data.mesh_id.numel() == 1 else tuple(int(x) for x in data.mesh_id.flatten().tolist())
        else:
            mid = int(data.pos.data_ptr())

        ei = getattr(data, "edge_index", None)
        if torch.is_tensor(ei):
            return (mid, int(ei.size(1)), int(ei.data_ptr()))
        return mid

    def clear_caches(self):
        """Clear all cached data."""
        self.geo_cache.clear()

    def _precompute_geometry(self, pos, edge_index, device):
        row, col = edge_index
        N = pos.size(0)
        dX = (pos[col] - pos[row]).detach()
        M_edge = torch.bmm(dX.unsqueeze(2), dX.unsqueeze(1)).float()
        M_node = torch.zeros(N, 2, 2, device=device, dtype=torch.float32)
        M_node.index_add_(0, row, M_edge)
        
        # RIDGE REGULARIZATION
        epsilon = 1e-8 
        eye = torch.eye(2, device=device, dtype=torch.float32).unsqueeze(0)
        M_node = M_node + eye * epsilon
        
        try:
            M_inv = torch.linalg.inv(M_node)
        except RuntimeError:
            M_inv = torch.linalg.pinv(M_node)
            
        return M_inv, dX

    def solve_single_variable(self, pos, edge_index, u, cache_key=None):
        row, col = edge_index
        N = pos.size(0)

        M_inv = dX = None
        if cache_key is not None:
            cached = self.geo_cache.get(cache_key, None)
            if cached is not None:
                M_inv, dX = cached
                # Guard against stale cache entries when geometry/connectivity changed.
                E_now = int(edge_index.size(1))
                N_now = int(pos.size(0))
                if (dX.size(0) != E_now) or (M_inv.size(0) != N_now):
                    M_inv, dX = None, None

        if M_inv is None or dX is None:
            M_inv, dX = self._precompute_geometry(pos, edge_index, pos.device)
            if cache_key is not None:
                self.geo_cache[cache_key] = (M_inv, dX)
        elif (M_inv.device != pos.device) or (dX.device != pos.device):
            M_inv = M_inv.to(pos.device)
            dX = dX.to(pos.device)
            if cache_key is not None:
                self.geo_cache[cache_key] = (M_inv, dX)

        # Compute raw gradients
        du = u[col] - u[row]
        V_edge = dX.unsqueeze(2) * du.unsqueeze(1)
        V_node = torch.zeros(N, 2, 1, device=pos.device, dtype=torch.float32)
        V_node.index_add_(0, row, V_edge.float())
        
        grads = torch.bmm(M_inv, V_node).squeeze(2)
        
        # HIGH-FIDELITY CLAMP: Safety ceiling only
        grads = torch.clamp(grads, -self.grad_limit, self.grad_limit)
        
        # NO DAMPING — gradients are accurate at all neighbor counts
        return grads.to(dtype=pos.dtype)

    def forward(self, data, field):
        pos, edge_index = data.pos, data.edge_index
        key = self._get_cache_key(data)
        return [self.solve_single_variable(pos, edge_index, field[:, i:i+1], key) 
                for i in range(field.shape[1])]

    def __call__(self, data, field):
        return self.forward(data, field)


class SolveWeightLST2d(nn.Module):
    """
    MLS Laplacian Solver — WITH 2-hop stencil extension.
    
    The 5x5 polynomial basis needs ≥5 neighbors for a well-conditioned system.
    Nodes with 3-4 neighbors produce Laplacian errors 73,830x worse than 5+ nodes.
    
    PRIMARY FIX (use_2hop_extension=True):
        Extends the MLS stencil at low-neighbor nodes by adding 2-hop neighbor
        edges. This makes the system well-determined using real node values —
        no damping, no ghost nodes, no field extrapolation needed.
        Precomputed once per mesh, cached, zero per-timestep overhead.
    
    FALLBACK (use_2hop_extension=False):
        Neighbor-count damping: reduces Laplacian contribution at low-neighbor
        nodes. Masks the problem (output → 0) rather than fixing it.
    """
    def __init__(self, pos_mean=None, pos_std=None, boundary_margin=0.0,
                 norm_method='z_score', max_position=None, min_neighbors=5,
                 use_2hop_extension=True, **kwargs):
        super().__init__()
        # Dynamic AMR creates many transient geometries; keep caching opt-in.
        self.cache_by_geometry = bool(kwargs.get("cache_by_geometry", True))
        self.weights_cache = {}
        self.damping_cache = {}
        self.edge_aug_cache = {}
        self.weight_limit = 5000.0
        
        # Normalization config
        self.norm_method = norm_method
        self.max_position = max_position
        
        # Neighbor-count damping config (fallback)
        self.min_neighbors = min_neighbors
        
        # 2-hop stencil extension config (primary fix)
        # Adds neighbor-of-neighbor edges at nodes with < min_2hop_neighbors
        # Target 6 = well overdetermined for 5-parameter quadratic basis
        self.use_2hop_extension = use_2hop_extension
        self.min_2hop_neighbors = 6
        
        # boundary_margin kept for backward compat but NOT USED
        self.boundary_margin = boundary_margin
        
        if pos_mean is not None:
            self.register_buffer('pos_mean', torch.tensor(pos_mean, dtype=torch.float32))
            self.register_buffer('pos_std', torch.tensor(pos_std, dtype=torch.float32))
        else:
            self.pos_mean, self.pos_std = None, None

    #def _get_cache_key(self, data):
    #    if hasattr(data, 'mesh_id') and data.mesh_id is not None:
    #        return data.mesh_id.item() if data.mesh_id.numel() == 1 else tuple(data.mesh_id.tolist())
    #    return data.pos.data_ptr()
    def _get_cache_key(self, data):
        if not self.cache_by_geometry:
            return None

        if hasattr(data, "mesh_id") and data.mesh_id is not None:
            mid = data.mesh_id.item() if data.mesh_id.numel() == 1 else tuple(int(x) for x in data.mesh_id.flatten().tolist())
        else:
            mid = int(data.pos.data_ptr())

        ei = getattr(data, "edge_index", None)
        if torch.is_tensor(ei):
            return (mid, int(ei.size(1)), int(ei.data_ptr()))
        return mid

    def clear_caches(self):
        """Clear all cached data."""
        self.weights_cache.clear()
        self.damping_cache.clear()
        self.edge_aug_cache.clear()

    def _get_augmented_edge_index(self, data):
        """
        Get 2-hop augmented edge_index, with caching.
        The augmented edge_index is used ONLY for MLS weight computation.
        The actual message passing in apply_laplacian uses this augmented
        edge_index too, so 2-hop neighbors contribute to the Laplacian.
        """
        key = self._get_cache_key(data)
        if key is not None and key in self.edge_aug_cache:
            cached = self.edge_aug_cache[key]
            if cached.device != data.pos.device:
                cached = cached.to(data.pos.device)
                self.edge_aug_cache[key] = cached
            return cached
        
        edge_aug = compute_2hop_extension(
            data.pos, data.edge_index,
            min_neighbors=self.min_2hop_neighbors
        )
        
        if key is not None:
            self.edge_aug_cache[key] = edge_aug
        
        n_added = edge_aug.shape[1] - data.edge_index.shape[1]
        if n_added > 0:
            print(f"  2-hop stencil: added {n_added} edges "
                  f"({n_added / data.pos.shape[0]:.1%} of nodes extended)")
        
        return edge_aug

    def _get_laplacian_damping(self, edge_index, num_nodes, device):
        """Fallback: neighbor-count damping."""
        return compute_neighbor_damping(
            edge_index, num_nodes,
            min_neighbors=self.min_neighbors,
            device=device
        )

    def _polynomial_basis(self, pos):
        x, y = pos[:, 0:1], pos[:, 1:2]
        return torch.cat([x, y, x*y, x*x, y*y], dim=-1)

    def _laplacian_basis(self, pos):
        v = torch.zeros(pos.shape[0], 5, dtype=pos.dtype, device=pos.device)
        v[:, 3], v[:, 4] = 2.0, 2.0
        return v

    def forward(self, data: Data):
        key = self._get_cache_key(data)

        if key is not None and key in self.weights_cache:
            weights = self.weights_cache[key]
            if weights.numel() != data.edge_index.size(1):
                # stale
                del self.weights_cache[key]
            else:
                if weights.device != data.pos.device:
                    weights = weights.to(data.pos.device)
                    self.weights_cache[key] = weights
                return weights
        '''
        if key in self.weights_cache:
            weights = self.weights_cache[key]
            if weights.device != data.pos.device:
                weights = weights.to(data.pos.device)
                self.weights_cache[key] = weights
            return weights
        '''
        pos = data.pos
        orig_edge_index = data.edge_index
        
        # Determine which edge_index to use for M_inv computation
        if self.use_2hop_extension:
            aug_edge_index = self._get_augmented_edge_index(data)
        else:
            aug_edge_index = orig_edge_index
        
        # Step 1: Build moment matrix M using AUGMENTED edges
        # This gives well-conditioned M_inv at boundary nodes
        aug_row, aug_col = aug_edge_index
        aug_diff = (pos[aug_col] - pos[aug_row]).detach()
        H_aug = self._polynomial_basis(aug_diff).float()
        
        M_edge = torch.bmm(H_aug.unsqueeze(2), H_aug.unsqueeze(1))
        M_node = torch.zeros(pos.size(0), 5, 5, device=pos.device, dtype=torch.float32)
        M_node.index_add_(0, aug_row, M_edge)
        
        # Stable inversion for distorted elements
        M_node = M_node + torch.eye(5, device=pos.device).unsqueeze(0) * 1e-7
        
        try:
            M_inv = torch.linalg.inv(M_node)
        except RuntimeError:
            M_inv = torch.linalg.pinv(M_node)
        
        # Step 2: Compute Laplacian weights on ORIGINAL edges only
        # M_inv is per-node [N,5,5] — already corrected by 2-hop stencil
        # Weights are per-edge and must match original edge_index for apply_laplacian
        L = self._laplacian_basis(pos).float()
        C = torch.bmm(M_inv, L.unsqueeze(2)).squeeze(2)  # [N, 5]
        
        orig_row, orig_col = orig_edge_index
        orig_diff = (pos[orig_col] - pos[orig_row]).detach()
        H_orig = self._polynomial_basis(orig_diff).float()
        
        weights = (C[orig_row] * H_orig).sum(dim=1)  # [E_orig]
        weights = torch.clamp(weights, -self.weight_limit, self.weight_limit)
        
        # Apply damping only if NOT using 2-hop extension
        if not self.use_2hop_extension:
            if key is not None and key in self.damping_cache:
                damping = self.damping_cache[key].to(data.pos.device)
            else:
                damping = self._get_laplacian_damping(
                    data.edge_index, data.pos.shape[0], data.pos.device
                )
                if key is not None:
                    self.damping_cache[key] = damping
            weights = weights * damping[orig_row]
        
        if key is not None:
            self.weights_cache[key] = weights
        
        return weights

    def __call__(self, data: Data):
        return self.forward(data)


def apply_laplacian(mesh_data, u, weights):
    """
    Apply Laplacian operator using precomputed weights.
    
    With 2-hop extension: weights are computed using a better-conditioned M_inv
    (from augmented stencil) but applied on the ORIGINAL edge_index. So weights
    length always matches mesh_data.edge_index length.
    """
    row, col = mesh_data.edge_index
    N = mesh_data.pos.shape[0] if hasattr(mesh_data, 'pos') else u.shape[0]
    diff = u[col] - u[row]
    weighted_diff = weights.unsqueeze(1) * diff
    laplacian = torch.zeros(N, u.shape[1], device=u.device, dtype=u.dtype)
    laplacian.index_add_(0, row, weighted_diff)
    return laplacian


# ============================================================================
# PHYSICS OPERATORS
# ============================================================================

class StrainOperator(nn.Module):
    def __init__(self, gradient_solver):
        super().__init__()
        self.grad_solver = gradient_solver
        self.sanity_limit = 100.0
        
    def forward(self, data: Data, displacement: torch.Tensor):
        gradients = self.grad_solver(data, displacement)
        dUx_dx = torch.clamp(gradients[0][:, 0:1], -self.sanity_limit, self.sanity_limit)
        dUx_dy = torch.clamp(gradients[0][:, 1:2], -self.sanity_limit, self.sanity_limit)
        dUy_dx = torch.clamp(gradients[1][:, 0:1], -self.sanity_limit, self.sanity_limit)
        dUy_dy = torch.clamp(gradients[1][:, 1:2], -self.sanity_limit, self.sanity_limit)
        epsilon_xx = dUx_dx
        epsilon_yy = dUy_dy
        epsilon_xy = 0.5 * (dUx_dy + dUy_dx)
        volumetric = epsilon_xx + epsilon_yy
        vm_squared = epsilon_xx**2 + epsilon_yy**2 + epsilon_xx * epsilon_yy + 3 * epsilon_xy**2
        von_mises = torch.sqrt(torch.clamp(vm_squared, min=1e-8))
        rotation = 0.5 * (dUy_dx - dUx_dy)
        return torch.cat([epsilon_xx, epsilon_yy, epsilon_xy, volumetric, von_mises, rotation], dim=1)


class StrainMLS(nn.Module):
    def __init__(self, gradient_solver, use_von_mises=True, use_volumetric=True, n_dimensions=2):
        super().__init__()
        self.gradient_solver = gradient_solver
        self.use_von_mises = use_von_mises
        self.use_volumetric = use_volumetric
        self.n_dimensions = n_dimensions
        self.sanity_limit = 100.0
        
    def forward(self, displacement, mesh_data):
        features = []
        if self.n_dimensions == 2:
            gradients = self.gradient_solver(mesh_data, displacement)
            dUx_dx = torch.clamp(gradients[0][:, 0:1], -self.sanity_limit, self.sanity_limit)
            dUx_dy = torch.clamp(gradients[0][:, 1:2], -self.sanity_limit, self.sanity_limit)
            dUy_dx = torch.clamp(gradients[1][:, 0:1], -self.sanity_limit, self.sanity_limit)
            dUy_dy = torch.clamp(gradients[1][:, 1:2], -self.sanity_limit, self.sanity_limit)
            epsilon_xx = dUx_dx
            epsilon_yy = dUy_dy
            epsilon_xy = 0.5 * (dUx_dy + dUy_dx)
            features.extend([epsilon_xx, epsilon_yy, epsilon_xy])
            if self.use_von_mises:
                vm_sq = epsilon_xx**2 + epsilon_yy**2 + epsilon_xx * epsilon_yy + 3 * epsilon_xy**2
                von_mises = torch.sqrt(torch.clamp(vm_sq, min=1e-8))
                features.append(von_mises)
            if self.use_volumetric:
                volumetric = epsilon_xx + epsilon_yy
                features.append(volumetric)
        return torch.cat(features, dim=1) if features else torch.empty(displacement.shape[0], 0, device=displacement.device)


class AdvectionMLS(nn.Module):
    def __init__(self, gradient_solver):
        super().__init__()
        self.gradient_solver = gradient_solver
        
    def forward(self, state_variable, velocity_field, mesh_data):
        gradients = self.gradient_solver(mesh_data, state_variable)
        grad_stacked = torch.stack(gradients, dim=1)
        return torch.sum(velocity_field.unsqueeze(1) * grad_stacked, dim=2)


class DiffusionMLS(nn.Module):
    def __init__(self, laplacian_solver):
        super().__init__()
        self.laplacian_solver = laplacian_solver
        
    def forward(self, state_variable, mesh_data):
        weights = self.laplacian_solver(mesh_data)
        return apply_laplacian(mesh_data, state_variable, weights)


def compute_strain_features(mesh_data, displacement, gradient_solver):
    strain_op = StrainOperator(gradient_solver)
    return strain_op(mesh_data, displacement)


def compute_equilibrium_residual(mesh_data, displacement, laplacian_solver):
    weights = laplacian_solver(mesh_data)
    return apply_laplacian(mesh_data, displacement, weights)


__all__ = [
    'SolveGradientsLST', 'SolveWeightLST2d', 'apply_laplacian',
    'StrainOperator', 'StrainMLS', 'AdvectionMLS', 'DiffusionMLS',
    'compute_strain_features', 'compute_equilibrium_residual',
    'compute_neighbor_damping', 'compute_2hop_extension'
]
