from __future__ import annotations
import torch
import torch.nn.functional as F



@torch.no_grad()
def dynamic_cells_from_parent_masks(
    masks_by_level: dict[int, torch.Tensor],
    H: int, W: int,
    xmin: float, xmax: float, ymin: float, ymax: float,
    build_edges: bool = False,
    refine_ratio: int = 2,
):
    """
    Convert hierarchical *parent* masks into the final leaf mesh.

    Input:
      masks_by_level[L]: bool tensor with shape
        (H*refine_ratio^(L-1), W*refine_ratio^(L-1))
        - L=1: parents at level-0 to be refined -> produce level-1 children
        - L=2: parents at level-1 to be refined -> produce level-2 children
        - ... (up to Lmax)

    Output:
      centers: (N,2) in bbox units
      levels:  (N,)  int64, level of each leaf cell (0..Lmax)
      parents: (N,)  int64, coarse (level-0) parent index in [0..H*W-1]
      edge_index: (2,E) (optional) 4-neighbor within-level connectivity
    """
    rr = int(refine_ratio)
    if rr < 2:
        raise ValueError(f"refine_ratio must be >=2, got {refine_ratio}")

    device = next(iter(masks_by_level.values())).device if masks_by_level else torch.device("cpu")
    Lmax = max(masks_by_level.keys(), default=0)

    # Normalize to bool with expected shapes
    refine = [None] * (Lmax + 1)   # refine[l] exists for l>=1
    for L, M in masks_by_level.items():
        Mh, Mw = H * (rr ** (L - 1)), W * (rr ** (L - 1))
        assert M.shape == (Mh, Mw), f"mask[{L}] has shape {tuple(M.shape)}; expected {(Mh, Mw)}"
        refine[L] = M.to(torch.bool)

    leaves_by_level: list[torch.Tensor] = []

    if Lmax == 0:
        # no refinement: all L0 are leaves
        leaves_by_level.append(torch.ones((H, W), dtype=torch.bool, device=device))
    else:
        # leaf at L0: those NOT refined at level-1
        leaves_by_level.append(~refine[1])

        # intermediate levels
        for l in range(1, Lmax):
            # children exist where parent grid asked for refinement at level l
            parent_refined = F.interpolate(
                refine[l].float().unsqueeze(0).unsqueeze(0),
                scale_factor=float(rr), mode="nearest"
            ).squeeze(0).squeeze(0).to(torch.bool)  # shape (H*rr^l, W*rr^l)

            leaves_l = parent_refined & ~refine[l+1]
            leaves_by_level.append(leaves_l)

        # deepest level: every child created by refine[Lmax] is a leaf
        leaf_Lmax = F.interpolate(
            refine[Lmax].float().unsqueeze(0).unsqueeze(0),
            scale_factor=float(rr), mode="nearest"
        ).squeeze(0).squeeze(0).to(torch.bool)     # (H*rr^Lmax, W*rr^Lmax)
        leaves_by_level.append(leaf_Lmax)

    # Assemble centers/levels/parents
    xs = xmax - xmin
    ys = ymax - ymin
    centers_list, levels_list, parents_list = [], [], []

    for l, leaf in enumerate(leaves_by_level):
        HH = H * (rr ** l)
        WW = W * (rr ** l)
        jj, ii = torch.nonzero(leaf, as_tuple=True)       # rows=j (y), cols=i (x)
        if jj.numel() == 0:
            continue

        # centers in bbox units (note division by WW/HH respectively)
        cx = xmin + (ii.to(torch.float32) + 0.5) * (xs / float(WW))
        cy = ymin + (jj.to(torch.float32) + 0.5) * (ys / float(HH))
        centers_list.append(torch.stack([cx, cy], dim=-1))

        # level ids
        levels_list.append(torch.full((jj.numel(),), l, dtype=torch.int64, device=leaf.device))

        # coarse (level-0) parent index (row-major; parent = j0*W + i0)
        i0 = (ii // (rr ** l)).to(torch.int64)
        j0 = (jj // (rr ** l)).to(torch.int64)
        parents_list.append(j0 * W + i0)

    if centers_list:
        centers = torch.cat(centers_list, dim=0)
        levels  = torch.cat(levels_list,  dim=0)
        parents = torch.cat(parents_list, dim=0)
    else:
        centers = torch.empty(0, 2, dtype=torch.float32, device=device)
        levels  = torch.empty(0,   dtype=torch.int64, device=device)
        parents = torch.empty(0,   dtype=torch.int64, device=device)

    # Optional: 4-neighbor edges within each level grid (no cross-level edges)
    if not build_edges or centers.numel() == 0:
        ei = torch.empty(2, 0, dtype=torch.int64, device=device)
        return centers, levels, parents, ei

    # Build within-level edges
    # Create a per-level linear index map -> global leaf index
    ei_src, ei_dst = [], []
    start = 0
    for l, leaf in enumerate(leaves_by_level):
        HH = H * (rr ** l)
        WW = W * (rr ** l)
        idx_map = -torch.ones((HH, WW), dtype=torch.int64, device=device)
        jj, ii = torch.nonzero(leaf, as_tuple=True)
        if jj.numel() == 0:
            continue
        count = jj.numel()
        idx_map[jj, ii] = torch.arange(start, start + count, device=device, dtype=torch.int64)
        start += count

        # right neighbors
        jj_r, ii_r = jj, ii + 1
        m_r = (ii_r < WW) & (idx_map[jj, ii] >= 0) & (idx_map[jj_r, ii_r] >= 0)
        if m_r.any():
            a = idx_map[jj[m_r], ii[m_r]]
            b = idx_map[jj_r[m_r], ii_r[m_r]]
            ei_src.append(a); ei_dst.append(b)

        # up neighbors
        jj_u, ii_u = jj + 1, ii
        m_u = (jj_u < HH) & (idx_map[jj, ii] >= 0) & (idx_map[jj_u, ii_u] >= 0)
        if m_u.any():
            a = idx_map[jj[m_u], ii[m_u]]
            b = idx_map[jj_u[m_u], ii_u[m_u]]
            ei_src.append(a); ei_dst.append(b)

    if ei_src:
        src = torch.cat(ei_src); dst = torch.cat(ei_dst)
        ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)  # undirected
    else:
        ei = torch.empty(2, 0, dtype=torch.int64, device=device)

    return centers, levels, parents, ei


def _as_2_by_E(edge_index: torch.Tensor) -> torch.Tensor:
    if isinstance(edge_index, (list, tuple)):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    if not torch.is_tensor(edge_index) or edge_index.dim() != 2:
        raise ValueError("edge_index must be 2D")
    if edge_index.size(0) == 2:
        return edge_index.contiguous()
    if edge_index.size(1) == 2:
        return edge_index.t().contiguous()
    raise ValueError("edge_index must be [2,E] or [E,2]")

def unique_undirected(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=edge_index.device)
    ei = _as_2_by_E(edge_index)
    # drop self-loops
    mask = ei[0] != ei[1]
    ei = ei[:, mask]
    # canonical ordering u<v
    u = torch.minimum(ei[0], ei[1])
    v = torch.maximum(ei[0], ei[1])
    ei = torch.stack([u, v], dim=0)
    # unique
    key = ei[0] * num_nodes + ei[1]
    perm = torch.argsort(key)
    ei = ei[:, perm]
    key = key[perm]
    keep = torch.ones(ei.size(1), dtype=torch.bool, device=ei.device)
    keep[1:] = key[1:] != key[:-1]
    return ei[:, keep]


@torch.no_grad()
def build_idw_map(
    dst_xy: torch.Tensor,   # (N, 2)
    src_xy: torch.Tensor,   # (M, 2)
    k: int = 8,
    chunk: int = 8192,
    eps: float = 1e-8,
    *,
    backend: str = "exact",
    faiss_nlist: int = 256,
    faiss_nprobe: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (idx, w) so that: dst_vals = sum_j w[:, j] * src_vals[idx[:, j]]
    Robust to exact coincidences and NaNs/Infs from distance calc.
    Shapes: idx -> (N, k), w -> (N, k)
    """
    backend_key = str(backend).strip().lower()
    if backend_key in ("exact", "torch", "cdist"):
        return _build_idw_map_exact(dst_xy, src_xy, k=k, chunk=chunk, eps=eps)
    if backend_key in ("faiss_flat", "flat"):
        return _build_idw_map_faiss(
            dst_xy=dst_xy,
            src_xy=src_xy,
            k=k,
            chunk=chunk,
            eps=eps,
            index_kind="flat",
            faiss_nlist=faiss_nlist,
            faiss_nprobe=faiss_nprobe,
        )
    if backend_key in ("faiss_ivf", "faiss", "ann", "approx"):
        return _build_idw_map_faiss(
            dst_xy=dst_xy,
            src_xy=src_xy,
            k=k,
            chunk=chunk,
            eps=eps,
            index_kind="ivf",
            faiss_nlist=faiss_nlist,
            faiss_nprobe=faiss_nprobe,
        )
    raise ValueError(
        f"Unknown IDW backend '{backend}'. Expected one of: exact, faiss_flat, faiss_ivf."
    )


def _idw_weights_from_dist(d_sel: torch.Tensor, eps: float) -> torch.Tensor:
    d_sel = torch.nan_to_num(d_sel, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
    w_sel = torch.empty_like(d_sel)
    exact = d_sel[:, 0] <= eps

    if exact.any():
        w_sel[exact] = 0.0
        w_sel[exact, 0] = 1.0

    if (~exact).any():
        nz = ~exact
        inv = 1.0 / torch.clamp(d_sel[nz], min=eps)
        inv_sum = inv.sum(dim=1, keepdim=True)
        w_norm = inv / (inv_sum + eps)
        w_sel[nz] = torch.nan_to_num(w_norm, nan=0.0, posinf=0.0, neginf=0.0)

    return w_sel


@torch.no_grad()
def _build_idw_map_exact(
    dst_xy: torch.Tensor,
    src_xy: torch.Tensor,
    k: int = 8,
    chunk: int = 8192,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    N = int(dst_xy.shape[0])
    M = int(src_xy.shape[0])
    k = min(k, M)
    if N == 0 or k <= 0:
        dev = dst_xy.device
        return (
            torch.empty((N, 0), dtype=torch.long, device=dev),
            torch.empty((N, 0), dtype=torch.float32, device=dev),
        )

    idx_all, w_all = [], []
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        q = dst_xy[start:end]  # (b, 2)

        # Pairwise distances; sanitize numerics before anything else
        d = torch.cdist(q, src_xy, p=2)
        d = torch.nan_to_num(d, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))

        # Pick neighbors by *smallest distance*
        tk = torch.topk(d, k=k, dim=1, largest=False)
        d_sel = tk.values.contiguous()   # (b, k)
        i_sel = tk.indices.contiguous()  # (b, k)

        w_sel = _idw_weights_from_dist(d_sel, eps=eps)

        idx_all.append(i_sel)
        w_all.append(w_sel)

    return torch.cat(idx_all, dim=0), torch.cat(w_all, dim=0)


@torch.no_grad()
def _build_idw_map_faiss(
    *,
    dst_xy: torch.Tensor,
    src_xy: torch.Tensor,
    k: int,
    chunk: int,
    eps: float,
    index_kind: str,
    faiss_nlist: int,
    faiss_nprobe: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "IDW backend requires faiss. Install faiss-gpu/faiss-cpu or switch backend to 'exact'."
        ) from e

    # Enables zero-copy torch tensor I/O when supported by the installed FAISS.
    try:
        import faiss.contrib.torch_utils  # type: ignore  # noqa: F401
    except Exception:
        pass

    dst_xy = dst_xy.contiguous().to(dtype=torch.float32)
    src_xy = src_xy.contiguous().to(dtype=torch.float32)
    dev = dst_xy.device
    if src_xy.device != dev:
        src_xy = src_xy.to(dev)

    N = int(dst_xy.shape[0])
    M = int(src_xy.shape[0])
    k = min(int(k), M)
    if N == 0 or k <= 0:
        return (
            torch.empty((N, 0), dtype=torch.long, device=dev),
            torch.empty((N, 0), dtype=torch.float32, device=dev),
        )

    dim = int(src_xy.shape[1])
    if dim <= 0:
        raise ValueError("src_xy must have shape (M, D) with D>0.")

    use_gpu = (dev.type == "cuda")
    faiss_index = None
    torch_io_ok = False

    if use_gpu:
        gpu_id = dev.index if dev.index is not None else 0
        res = faiss.StandardGpuResources()
        if index_kind == "flat":
            faiss_index = faiss.GpuIndexFlatL2(res, dim)
        else:
            if M < max(64, k * 8):
                faiss_index = faiss.GpuIndexFlatL2(res, dim)
            else:
                nlist_eff = max(1, min(int(faiss_nlist), M))
                quantizer = faiss.IndexFlatL2(dim)
                cpu_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist_eff, faiss.METRIC_L2)
                faiss_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_ivf)
                nprobe_eff = max(1, min(int(faiss_nprobe), nlist_eff))
                try:
                    faiss_index.nprobe = nprobe_eff
                except Exception:
                    try:
                        faiss.ParameterSpace().set_index_parameter(faiss_index, "nprobe", nprobe_eff)
                    except Exception:
                        pass
    else:
        if index_kind == "flat":
            faiss_index = faiss.IndexFlatL2(dim)
        else:
            if M < max(64, k * 8):
                faiss_index = faiss.IndexFlatL2(dim)
            else:
                nlist_eff = max(1, min(int(faiss_nlist), M))
                quantizer = faiss.IndexFlatL2(dim)
                faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist_eff, faiss.METRIC_L2)
                nprobe_eff = max(1, min(int(faiss_nprobe), nlist_eff))
                try:
                    faiss_index.nprobe = nprobe_eff
                except Exception:
                    try:
                        faiss.ParameterSpace().set_index_parameter(faiss_index, "nprobe", nprobe_eff)
                    except Exception:
                        pass

    assert faiss_index is not None

    # Train only for IVF indexes.
    if hasattr(faiss_index, "is_trained") and (not faiss_index.is_trained):
        trained = False
        try:
            faiss_index.train(src_xy)
            trained = True
            torch_io_ok = True
        except Exception:
            pass
        if not trained:
            src_np = src_xy.detach().cpu().numpy()
            faiss_index.train(src_np)
            torch_io_ok = False

    # Add source vectors.
    try:
        faiss_index.add(src_xy)
        torch_io_ok = True
    except Exception:
        src_np = src_xy.detach().cpu().numpy()
        faiss_index.add(src_np)
        torch_io_ok = False

    idx_all = []
    w_all = []
    for start in range(0, N, int(max(1, chunk))):
        end = min(N, start + int(max(1, chunk)))
        q = dst_xy[start:end]

        if torch_io_ok:
            try:
                d2, idx = faiss_index.search(q, k)
                if not torch.is_tensor(d2):
                    d2 = torch.from_numpy(d2).to(device=dev)
                if not torch.is_tensor(idx):
                    idx = torch.from_numpy(idx).to(device=dev)
            except Exception:
                q_np = q.detach().cpu().numpy()
                d2_np, idx_np = faiss_index.search(q_np, k)
                d2 = torch.from_numpy(d2_np).to(device=dev)
                idx = torch.from_numpy(idx_np).to(device=dev)
                torch_io_ok = False
        else:
            q_np = q.detach().cpu().numpy()
            d2_np, idx_np = faiss_index.search(q_np, k)
            d2 = torch.from_numpy(d2_np).to(device=dev)
            idx = torch.from_numpy(idx_np).to(device=dev)

        d2 = torch.nan_to_num(d2, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
        d_sel = torch.sqrt(torch.clamp_min(d2, 0.0))
        i_sel = idx.to(dtype=torch.long, device=dev)
        w_sel = _idw_weights_from_dist(d_sel, eps=eps)
        idx_all.append(i_sel)
        w_all.append(w_sel)

    return torch.cat(idx_all, dim=0), torch.cat(w_all, dim=0)


@torch.no_grad()
def apply_idw_map(idx: torch.Tensor, w: torch.Tensor, src_feat: torch.Tensor) -> torch.Tensor:
    # src_feat: (M, F) → returns (N, F)
    N, K = idx.shape
    F = src_feat.shape[1]
    gathered = src_feat.index_select(0, idx.reshape(-1)).view(N, K, F)  # (N, K, F)
    vals = (gathered * w.unsqueeze(-1)).sum(dim=1)
    return torch.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)


def _targeted_map_to_pred(
    # predicted (t+1) mesh
    pred_centers: torch.Tensor,        # [N_pred, 2]
    pred_levels : torch.Tensor,        # [N_pred]  (0 = coarse, 1 = child, ...)
    pred_parents: torch.Tensor,        # [N_pred]  parent index in [0..H*W-1]
    mask_pred   : torch.Tensor,        # [H, W] or [H*W] bool

    # source mesh positions+features (either t or GT(t+1))
    src_centers : torch.Tensor,        # [N_src, 2]
    src_feats   : torch.Tensor,        # [N_src, F]
    H: int, W: int,

    # optional parent-level source mask to skip IDW in unchanged coarse regions
    mask_src_parent: torch.Tensor | None,  # [H, W] or [H*W] bool; if None -> no identity copies

    # coarse aggregate on the source mesh (one vector per parent)
    src_parent_feats: torch.Tensor | None, # [H*W, F]; required if you pass mask_src_parent

    # idw controls
    knn_k: int = 8,
    chunk: int = 8192,
):
    """
    Map src_feats on src_centers -> pred_centers.

    If mask_src_parent and src_parent_feats are provided, we:
      - directly copy values for coarse predicted cells whose parent is marked
        active in mask_src_parent (no IDW needed there).
      - run IDW only for the remaining cells.

    Otherwise we do pure IDW for all predicted centers.
    """
    device = src_feats.device
    N_pred, F = pred_centers.size(0), src_feats.size(1)

    # Output buffer
    mapped = src_feats.new_zeros((N_pred, F), device=device)

    # Track which dst cells we still need to handle via IDW
    need_idw = torch.ones(N_pred, dtype=torch.bool, device=device)

    # make sure this always exists, regardless of branches
    unchanged = None

    # --------- optional: coarse "no-change" copy branch ----------
    if (mask_src_parent is not None) and (src_parent_feats is not None):
        mask_pred_flat = mask_pred.view(-1).bool()
        mask_src_flat  = mask_src_parent.view(-1).bool()

        parent_ix = pred_parents.long().clamp_(0, H * W - 1)

        # Only consider coarse predicted cells (level == 0)
        pred_is_coarse = (pred_levels == 0)

        # Parent active both in src and pred
        same_parent_active = mask_src_flat[parent_ix] & mask_pred_flat[parent_ix]

        # Cells where we can just copy the parent feature
        unchanged = pred_is_coarse & same_parent_active

        if unchanged.any():
            mapped[unchanged] = src_parent_feats[parent_ix[unchanged]]
            need_idw[unchanged] = False

    # --------- IDW for anything not handled by the coarse-copy branch ---------- 
    q_idx = need_idw.nonzero(as_tuple=True)[0]
    if q_idx.numel() > 0:
        q_pts = pred_centers[q_idx].to(device)  # (Q,2) queries

        idx_map, w_map = build_idw_map(q_pts, src_centers.to(device), k=knn_k, chunk=chunk)
        vals = apply_idw_map(idx_map, w_map, src_feats.to(device))   # (Q,F)
        mapped[q_idx] = vals

        idw_points = int(q_idx.numel())
    else:
        idw_points = 0

    stats = {
        "copied_same_coarse": int(unchanged.sum().item()) if unchanged is not None else 0,
        "idw_points": idw_points,
        "N_dst": N_pred,
    }
    return mapped, stats


# ---- Public unified API -----------------------------------------------------------

def build_coarse_n4(H: int, W: int, device=None) -> torch.Tensor:
    """
    Return [H*W, 4] neighbor indices (left, right, down, up) on the coarse grid,
    clamping at borders (i.e., border neighbors point to themselves).
    """
    j = torch.arange(H, device=device)
    i = torch.arange(W, device=device)
    JJ, II = torch.meshgrid(j, i, indexing="ij")  # JJ: [H,W] rows, II: [H,W] cols

    left_i  = torch.clamp(II - 1, 0, W - 1)
    right_i = torch.clamp(II + 1, 0, W - 1)
    down_j  = torch.clamp(JJ - 1, 0, H - 1)
    up_j    = torch.clamp(JJ + 1, 0, H - 1)

    left  = JJ * W + left_i
    right = JJ * W + right_i
    down  = down_j * W + II
    up    = up_j   * W + II

    n4 = torch.stack([left, right, down, up], dim=-1).reshape(H * W, 4)
    return n4.long()


def apply_precomputed_idw_map(
    idx_map: torch.Tensor,
    w_map: torch.Tensor,
    src_feats: torch.Tensor,
) -> torch.Tensor:
    """
    Cheap application of a precomputed IDW map.

    Args:
        idx_map: (N_dst, K) long tensor of source indices.
        w_map:   (N_dst, K) float tensor of normalized weights.
        src_feats: (N_src, F) float tensor of features defined on the
                   *source* mesh (step k).

    Returns:
        dst_feats: (N_dst, F) = IDW(src_feats) according to the map.

    This is just a thin wrapper around your existing _apply_idw_map,
    but it takes care of moving idx/w to the same device as src_feats.
    """
    if idx_map is None or w_map is None:
        raise ValueError("apply_precomputed_idw_map: idx_map and w_map must not be None")

    dev = src_feats.device
    idx_dev = idx_map.to(dev, non_blocking=True)
    w_dev   = w_map.to(dev, non_blocking=True)

    # apply_idw_map is your existing helper:
    #   out = apply_idw_map(idx, w, src_feat)
    dst_feats = apply_idw_map(idx_dev, w_dev, src_feats)
    return dst_feats


@torch.no_grad()
def knn_interpolate_cuda_cdist(
    src_xy: torch.Tensor,   # (Ns,2) float
    src_val: torch.Tensor,  # (Ns,F) float
    tgt_xy: torch.Tensor,   # (Nt,2) float
    *,
    k: int = 4,
    chunk: int = 65536,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    CUDA-optimized kNN/IDW using torch.cdist. Expects tensors on 'cuda'.
    """
    assert src_xy.device.type == "cuda" and tgt_xy.device.type == "cuda" and src_val.device.type == "cuda"
    Ns, F = src_val.shape
    Nt    = tgt_xy.size(0)
    out   = torch.empty((Nt, F), device=src_val.device, dtype=src_val.dtype)

    # Keep everything in float32 for numeric stability
    src_xy = src_xy.float(); tgt_xy = tgt_xy.float(); src_val = src_val.float()

    for s in range(0, Nt, chunk):
        e = min(s + chunk, Nt)
        D = torch.cdist(tgt_xy[s:e], src_xy)                 # (Ce, Ns)
        knnd, knni = torch.topk(D, k, dim=1, largest=False) # (Ce,k)
        w = 1.0 / knnd.clamp_min(eps)                       # IDW
        w = w / w.sum(dim=1, keepdim=True)

        vals = src_val.index_select(0, knni.reshape(-1)).reshape(e - s, k, F)
        out[s:e] = (w.unsqueeze(-1) * vals).sum(dim=1)

    return out


@torch.no_grad()
def knn_interpolate_matmul(
    src_xy: torch.Tensor,   # (Ns,2) float
    src_val: torch.Tensor,  # (Ns,F) float
    tgt_xy: torch.Tensor,   # (Nt,2) float
    *,
    k: int = 4,
    chunk: int = 65536,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    MPS/CPU-friendly kNN/IDW using the matmul distance trick:
      ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y^T
    Works well on 'mps' and 'cpu'.
    """
    dev = src_xy.device
    Ns, F = src_val.shape
    Nt    = tgt_xy.size(0)
    out   = torch.empty((Nt, F), device=dev, dtype=src_val.dtype)

    # Use float32 throughout for stability on MPS/CPU
    src_xy = src_xy.float(); tgt_xy = tgt_xy.float(); src_val = src_val.float()

    y2 = (src_xy**2).sum(dim=1)                       # (Ns,)
    for s in range(0, Nt, chunk):
        e  = min(s + chunk, Nt)
        X  = tgt_xy[s:e]                               # (Ce,2)
        x2 = (X**2).sum(dim=1, keepdim=True)          # (Ce,1)
        XY = X @ src_xy.T                              # (Ce,Ns)
        d2 = x2 + y2.unsqueeze(0) - 2.0 * XY          # (Ce,Ns)
        d2.clamp_min_(0.0)

        knnd2, knni = torch.topk(d2, k, dim=1, largest=False)  # (Ce,k)
        knnd = torch.sqrt(knnd2 + eps)
        w = 1.0 / knnd.clamp_min(eps)                        # IDW
        w = w / w.sum(dim=1, keepdim=True)

        vals = src_val.index_select(0, knni.reshape(-1)).reshape(e - s, k, F)
        out[s:e] = (w.unsqueeze(-1) * vals).sum(dim=1)

    return out


def parents_from_pos(centers: torch.Tensor,
                      H: int, W: int,
                      xmin: float, xmax: float, ymin: float, ymax: float) -> torch.Tensor:
    """
    Map centers in bbox units to coarse parent indices on HxW coarse grid.
    """
    x = centers[:, 0].to(torch.float32)
    y = centers[:, 1].to(torch.float32)
    dx = (xmax - xmin) / float(W)
    dy = (ymax - ymin) / float(H)

    col = torch.floor((x - xmin) / dx).long().clamp_(0, W - 1)
    row = torch.floor((y - ymin) / dy).long().clamp_(0, H - 1)
    return row * W + col
