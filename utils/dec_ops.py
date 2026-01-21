import torch
import torch.nn.functional as F

def _get_feature_name_list(cfg: dict) -> list[str] | None:
    feats = cfg.get("features", {}) or {}
    for k in ("dataset_order", "names", "use_columns"):
        v = feats.get(k, None)
        if isinstance(v, list) and len(v) > 0:
            return [str(x) for x in v]
    return None

def infer_feature_indices(cfg: dict, Fdim: int):
    """
    Returns indices for density, momx, momy, energy.
    Falls back to a reasonable guess if names are missing.
    """
    names = _get_feature_name_list(cfg)
    if not names:
        # fallback guess: [density, xmom, ymom, energy] OR your stated order [energy,xmom,ymom,density]
        # Try to keep robust: assume density is last if F=4 and user stated that order.
        if Fdim == 4:
            return {"rho": 3, "mx": 1, "my": 2, "E": 0}
        return {"rho": 0, "mx": 1 if Fdim > 1 else 0, "my": 2 if Fdim > 2 else 0, "E": 3 if Fdim > 3 else 0}

    lower = [n.lower() for n in names]
    def find_any(keys):
        for i, n in enumerate(lower):
            if any(k in n for k in keys):
                return i
        return None

    rho = find_any(["dens", "rho"])
    mx  = find_any(["x-mom", "xmom", "momx", "px", "mx"])
    my  = find_any(["y-mom", "ymom", "momy", "py", "my"])
    En  = find_any(["ener", "total e", "etot", "e " , " e_","energy"])

    # fallback by position if any missing
    if rho is None and Fdim == 4: rho = 3
    if mx  is None and Fdim == 4: mx  = 1
    if my  is None and Fdim == 4: my  = 2
    if En  is None and Fdim == 4: En  = 0

    rho = 0 if rho is None else int(rho)
    mx  = 0 if mx  is None else int(mx)
    my  = 0 if my  is None else int(my)
    En  = 0 if En  is None else int(En)
    return {"rho": rho, "mx": mx, "my": my, "E": En}

# -----------------------------
# PARC / Variant-B helpers
# -----------------------------

def dec_advdiff_terms_abs(
    x_abs: torch.Tensor,          # [N,F] absolute state at time t on pred mesh
    edge_index: torch.Tensor,     # [2,E]
    pred_ea: torch.Tensor,        # [E,5]
    levels: torch.Tensor,         # [N]
    *,
    dx0: float,
    dy0: float,
    cfg: dict,
    compute_adv: bool = True,
    compute_diff: bool = True,
):
    """
    Returns the *components* (absolute units):
      r_adv_abs  = -div(u*phi)      [N,F]
      r_diff_abs =  nu * Lap(phi)   [N,F]   (per-channel nu)
      area       = cell area        [N]
    These are unweighted by adv_weight/diff_weight; you apply weights outside.

    Uses the same geometry unpacking & conventions as dec_advdiff_rate.
    """
    loss = cfg.get("loss", {}) or {}
    scheme = str(loss.get("advection_scheme", "upwind")).lower()
    rho_eps = float(loss.get("rho_eps", 1e-8))

    nx, ny, face_len, _dual_len, tau = edge_attr_unpack(pred_ea)
    area = cell_area_from_levels(levels, dx0=dx0, dy0=dy0, dtype=x_abs.dtype, device=x_abs.device)  # [N]

    # ----- advection component -----
    if compute_adv:
        vel = compute_velocity_from_state(x_abs, cfg, eps=rho_eps)  # [N,2]
        div_adv = dec_divergence_advective_flux(
            phi=x_abs,
            vel=vel,
            edge_index=edge_index,
            nx=nx, ny=ny,
            face_len=face_len,
            area=area,
            scheme=scheme,
        )
        r_adv = -div_adv
    else:
        r_adv = torch.zeros_like(x_abs)

    # ----- diffusion component -----
    if compute_diff:
        nu = as_nu_tensor(loss.get("nu", 0.0), x_abs.size(1), device=x_abs.device, dtype=x_abs.dtype)  # [F]
        lap = dec_laplacian(x_abs, edge_index=edge_index, tau=tau, area=area)  # [N,F]
        r_diff = lap * nu.view(1, -1)
    else:
        r_diff = torch.zeros_like(x_abs)

    return r_adv, r_diff, area


def parc_select_feature_indices(cfg: dict, Fdim: int) -> list[int]:
    """
    Which state channels get operator-term inputs.
    Defaults to rho + E unless overridden.

    Priority:
      1) loss.parc_input_channels (list of strings)
      2) loss.channels (list of strings)
      3) default [rho, E]
    """
    loss = cfg.get("loss", {}) or {}
    names = loss.get("parc_input_channels", None)
    if not names:
        names = loss.get("channels", None)

    idx = infer_feature_indices(cfg, Fdim)

    if not names:
        return [idx["rho"], idx["E"]]

    out: list[int] = []
    for name in names:
        n = str(name).lower()
        if "dens" in n or n == "rho":
            out.append(idx["rho"])
        elif "ener" in n or n == "e":
            out.append(idx["E"])
        elif "x" in n and "mom" in n:
            out.append(idx["mx"])
        elif "y" in n and "mom" in n:
            out.append(idx["my"])

    # fallback if user provided something unusable
    if len(out) == 0:
        out = [idx["rho"], idx["E"]]
    return out


def parc_terms_to_node_inputs(
    r_adv_abs: torch.Tensor | None,     # [N,F] absolute or None
    r_diff_abs: torch.Tensor | None,    # [N,F] absolute or None
    *,
    dt_phys: torch.Tensor,              # scalar
    dt_ref: torch.Tensor | None,        # scalar or None
    sigma: torch.Tensor | None,         # [F] or None
    predict_type: str,                  # expects "rate" in your case
    cfg: dict,
    dtype,
    detach: bool = True,
):
    """
    Builds the operator-derived node feature block to concatenate onto _build_X(...).

    Config keys (under cfg["loss"]):
      parc_input_form: "rate" (default) or "delta"
      parc_include_adv: bool (default True)
      parc_include_diff: bool (default True)
      parc_input_weighted: bool (default False)
      parc_detach_inputs: bool (default True)

    Returns: node_inputs [N, Cextra]
    """
    loss = cfg.get("loss", {}) or {}
    form = str(loss.get("parc_input_form", "rate")).lower()
    include_adv = bool(loss.get("parc_include_adv", True))
    include_diff = bool(loss.get("parc_include_diff", True))
    weighted = bool(loss.get("parc_input_weighted", False))

    adv_w = float(loss.get("adv_weight", 1.0))
    diff_w = float(loss.get("diff_weight", 1.0))

    base = r_adv_abs if r_adv_abs is not None else r_diff_abs
    if base is None:
        raise ValueError("parc_terms_to_node_inputs: both r_adv_abs and r_diff_abs are None")

    Fdim = base.size(1)
    sel = parc_select_feature_indices(cfg, Fdim)

    # optional weighting (usually keep False; let NN learn mixing)
    adv_abs = None
    diff_abs = None
    if r_adv_abs is not None:
        adv_abs = (adv_w * r_adv_abs) if weighted else r_adv_abs
    if r_diff_abs is not None:
        diff_abs = (diff_w * r_diff_abs) if weighted else r_diff_abs

    # convert to desired units
    adv_u = None
    diff_u = None
    if form == "delta":
        if sigma is None:
            sigma_use = 1.0
        else:
            sigma_use = sigma.view(1, -1).clamp_min(1e-12)
        dt = dt_phys.clamp_min(1e-12)

        if adv_abs is not None:
            adv_u = (dt * adv_abs) / sigma_use
        if diff_abs is not None:
            diff_u = (dt * diff_abs) / sigma_use
    else:
        if adv_abs is not None:
            adv_u = physics_to_model_units(
                adv_abs, dt_phys=dt_phys, dt_ref=dt_ref, sigma=sigma, predict_type=predict_type
            )
        if diff_abs is not None:
            diff_u = physics_to_model_units(
                diff_abs, dt_phys=dt_phys, dt_ref=dt_ref, sigma=sigma, predict_type=predict_type
            )

    blocks = []
    if include_adv and (adv_u is not None):
        blocks.append(adv_u[:, sel])
    if include_diff and (diff_u is not None):
        blocks.append(diff_u[:, sel])

    if len(blocks) == 0:
        out = base.new_zeros((base.size(0), 0), dtype=dtype)
    else:
        out = torch.cat(blocks, dim=1).to(dtype=dtype)

    detach_cfg = bool(loss.get("parc_detach_inputs", True))
    if detach or detach_cfg:
        out = out.detach()
    return out


def parc_extra_in_channels(cfg: dict, Fdim: int) -> int:
    """
    Convenience: how many extra node channels are appended when PARC inputs are enabled.
    """
    loss = cfg.get("loss", {}) or {}
    include_adv = bool(loss.get("parc_include_adv", True))
    include_diff = bool(loss.get("parc_include_diff", True))
    sel = parc_select_feature_indices(cfg, Fdim)
    m = len(sel)
    extra = 0
    if include_adv:  extra += m
    if include_diff: extra += m
    return extra

def edge_attr_unpack(pred_ea: torch.Tensor):
    """
    Supports your current build_amr_face_adjacency_edges(return_edge_attr=True) layout:
      edge_attr = [nx, ny, face_len, dist, w_diff]
    where w_diff == face_len/dist (DEC tau).
    Returns: nx, ny, face_len, dual_len, tau
    """
    if pred_ea is None:
        raise RuntimeError("pred_ea is None (edge attributes missing).")
    if pred_ea.ndim != 2 or pred_ea.size(1) < 5:
        raise RuntimeError(f"pred_ea must be [E,>=5], got {tuple(pred_ea.shape)}")

    nx = pred_ea[:, 0]
    ny = pred_ea[:, 1]
    face_len = pred_ea[:, 2]
    dual_len = pred_ea[:, 3]
    tau = pred_ea[:, 4]  # face_len/dual_len
    return nx, ny, face_len, dual_len, tau

def cell_area_from_levels(levels: torch.Tensor, *, dx0: float, dy0: float, dtype, device):
    """
    axis-aligned dyadic quads: A = (dx0*dy0) / (4^L)
    """
    base_area = torch.tensor(float(dx0) * float(dy0), device=device, dtype=dtype)
    L = levels.to(device=device)
    if L.dtype not in (torch.int32, torch.int64):
        L = L.long()
    denom = torch.pow(torch.tensor(4.0, device=device, dtype=dtype), L.to(dtype=dtype))
    return base_area / denom

def dec_laplacian(phi: torch.Tensor, edge_index: torch.Tensor, tau: torch.Tensor, area: torch.Tensor):
    """
    phi: [N,F] (absolute units)
    edge_index: [2,E] directed
    tau: [E] (face_len/dual_len) for src->dst edge; ok if duplicated both ways
    area: [N] cell area
    Returns lap: [N,F] in absolute units / length^2 (consistent up to scaling).
    """
    N = phi.size(0)
    src = edge_index[0]
    dst = edge_index[1]

    # accumulate sum_j tau_ij (phi_j - phi_i)
    diff = (phi[dst] - phi[src]) * tau[:, None]  # [E,F]
    out = phi.new_zeros((N, phi.size(1)))
    out.index_add_(0, src, diff)

    area_safe = area.clamp_min(1e-12).unsqueeze(1)
    return out / area_safe

def dec_divergence_advective_flux(
    phi: torch.Tensor,          # [N,F] absolute
    vel: torch.Tensor,          # [N,2] absolute
    edge_index: torch.Tensor,   # [2,E] directed
    nx: torch.Tensor, ny: torch.Tensor,
    face_len: torch.Tensor,
    area: torch.Tensor,         # [N]
    scheme: str = "upwind",
):
    """
    Computes div(u * phi) using face-based fluxes stored on edges.
    For directed edge i->j, normal (nx,ny) points from i to j.

    Returns div: [N,F] (absolute units / time)
    """
    N, Fdim = phi.shape
    src = edge_index[0]
    dst = edge_index[1]

    # face-normal velocity: (u_face · n)
    u_face = 0.5 * (vel[src] + vel[dst])  # [E,2]
    un = u_face[:, 0] * nx + u_face[:, 1] * ny  # [E]

    if scheme.lower() == "central":
        phi_face = 0.5 * (phi[src] + phi[dst])  # [E,F]
    else:
        # upwind: if un>0 (flow from src to dst), take src; else take dst
        take_src = (un >= 0).unsqueeze(1)
        phi_face = torch.where(take_src, phi[src], phi[dst])  # [E,F]

    flux = (un * face_len)[:, None] * phi_face  # [E,F]
    div = phi.new_zeros((N, Fdim))
    div.index_add_(0, src, flux)

    area_safe = area.clamp_min(1e-12).unsqueeze(1)
    return div / area_safe

def compute_velocity_from_state(x_abs: torch.Tensor, cfg: dict, eps: float = 1e-8):
    """
    Derives velocity from conserved variables using u = (mx/rho, my/rho).
    x_abs: [N,F] absolute units
    """
    Fdim = x_abs.size(1)
    idx = infer_feature_indices(cfg, Fdim)
    rho = x_abs[:, idx["rho"]].clamp_min(eps)
    mx  = x_abs[:, idx["mx"]]
    my  = x_abs[:, idx["my"]]
    ux = mx / rho
    uy = my / rho
    return torch.stack([ux, uy], dim=1)  # [N,2]

def as_nu_tensor(nu, Fdim: int, *, device, dtype):
    """
    nu can be scalar or length-F list/tuple.
    """
    if nu is None:
        return torch.zeros((Fdim,), device=device, dtype=dtype)
    if isinstance(nu, (int, float)):
        return torch.full((Fdim,), float(nu), device=device, dtype=dtype)
    if isinstance(nu, (list, tuple)):
        if len(nu) == 1:
            return torch.full((Fdim,), float(nu[0]), device=device, dtype=dtype)
        if len(nu) != Fdim:
            raise RuntimeError(f"dec.nu has length {len(nu)} but Fdim={Fdim}")
        return torch.tensor([float(v) for v in nu], device=device, dtype=dtype)
    raise RuntimeError(f"Unsupported nu type: {type(nu)}")

def build_channel_mask_from_loss(cfg: dict, Fdim: int, *, device, dtype):
    loss = cfg.get("loss", {}) or {}
    chans = loss.get("channels", None)
    if not chans:
        return None
    idx = infer_feature_indices(cfg, Fdim)
    mask = torch.zeros((Fdim,), device=device, dtype=dtype)
    for name in chans:
        n = str(name).lower()
        if "dens" in n or n == "rho":
            mask[idx["rho"]] = 1.0
        elif "ener" in n or n == "e":
            mask[idx["E"]] = 1.0
        elif "x" in n and "mom" in n:
            mask[idx["mx"]] = 1.0
        elif "y" in n and "mom" in n:
            mask[idx["my"]] = 1.0
    return mask

def dec_advdiff_rate(
    x_abs: torch.Tensor,          # [N,F] absolute state (at time t on pred mesh)
    edge_index: torch.Tensor,     # [2,E]
    pred_ea: torch.Tensor,        # [E,5] edge_attr
    levels: torch.Tensor,         # [N]
    *,
    dx0: float,
    dy0: float,
    cfg: dict,
):
    """
    Returns r_phy_abs: [N,F] approximating dphi/dt = -div(u*phi) + nu*Lap(phi)
    Uses saved DEC/flux geometry from pred_ea.
    """
    #dec_cfg = cfg.get("dec", {}) or {}
    adv_w  = float(cfg["loss"].get("adv_weight", 1.0))
    diff_w = float(cfg["loss"].get("diff_weight", 1.0))
    scheme = str(cfg["loss"].get("advection_scheme", "upwind")).lower()
    rho_eps = float(cfg["loss"].get("rho_eps", 1e-8))
    nu = as_nu_tensor(cfg["loss"].get("nu", 0.0), x_abs.size(1), device=x_abs.device, dtype=x_abs.dtype)

    nx, ny, face_len, _dual_len, tau = edge_attr_unpack(pred_ea)

    area = cell_area_from_levels(levels, dx0=dx0, dy0=dy0, dtype=x_abs.dtype, device=x_abs.device)  # [N]
    vel = compute_velocity_from_state(x_abs, cfg, eps=rho_eps)  # [N,2]

    # div(u*phi)
    div_adv = 0.0
    if adv_w != 0.0:
        vel = compute_velocity_from_state(x_abs, cfg, eps=rho_eps)
        div_adv = dec_divergence_advective_flux(
            phi=x_abs,
            vel=vel,
            edge_index=edge_index,
            nx=nx, ny=ny,
            face_len=face_len,
            area=area,
            scheme=scheme,
        )
    else:
        div_adv = torch.zeros_like(x_abs)

    # Lap(phi)
    lap = dec_laplacian(x_abs, edge_index=edge_index, tau=tau, area=area)

    # r = -adv + nu*lap (per feature)
    r_phy = (-adv_w) * div_adv + diff_w * (lap * nu.view(1, -1))
    return r_phy, area

def physics_residual_loss_delta(
    y_pred_abs: torch.Tensor,   # [N,F]
    x_in_abs: torch.Tensor,     # [N,F]
    dt_phys: torch.Tensor,      # scalar tensor
    r_phy_abs: torch.Tensor,    # [N,F] absolute dphi/dt
    area: torch.Tensor,         # [N]
    *,
    sigma: torch.Tensor | None = None,       # [F] optional
    channel_mask: torch.Tensor | None = None # [F] optional float/bool
):
    """
    Delta-form residual:
        resid = (y_{t+1} - x_t) - dt * r_phy
    Area-weighted MSE. Avoids dividing by dt (stability).
    """
    dt = dt_phys.clamp_min(1e-12)
    resid = (y_pred_abs - x_in_abs) - (dt * r_phy_abs)  # [N,F]

    if sigma is not None:
        resid = resid / sigma.view(1, -1).clamp_min(1e-12)

    if channel_mask is not None:
        resid = resid * channel_mask.view(1, -1).to(resid.dtype)

    w = area.clamp_min(1e-12).unsqueeze(1)  # [N,1]
    return (w * resid.pow(2)).sum() / w.sum().clamp_min(1e-12)


def physics_to_model_units(
    r_phy_abs: torch.Tensor,     # [N,F] absolute dphi/dt
    *,
    dt_phys: torch.Tensor,       # scalar tensor
    dt_ref: torch.Tensor | None, # scalar tensor or None
    sigma: torch.Tensor | None,  # [F] or None
    predict_type: str,
):
    """
    Converts absolute physics rate into the same units as the model head output.
    - rate mode: model predicts (dt_ref * r_abs / sigma) if dt_ref provided else (r_abs / sigma)
    - delta mode: model predicts (dt_phys * r_abs / sigma)
    """
    if sigma is None:
        sigma_use = 1.0
    else:
        sigma_use = sigma.view(1, -1)

    if predict_type == "rate":
        dt_ref_use = (dt_ref if dt_ref is not None else None)
        if dt_ref_use is None:
            return r_phy_abs / sigma_use
        return (r_phy_abs * dt_ref_use) / sigma_use

    if predict_type == "delta":
        return (r_phy_abs * dt_phys) / sigma_use

    # absolute-predict mode: no clean mapping; return zeros so it’s inert
    return torch.zeros_like(r_phy_abs)

def physics_residual_loss(
    y_pred_abs: torch.Tensor,  # [N,F] absolute prediction at t+1
    x_in_abs: torch.Tensor,    # [N,F] absolute state at t
    dt_phys: torch.Tensor,     # scalar
    r_phy_abs: torch.Tensor,   # [N,F]
    area: torch.Tensor,        # [N]
):
    """
    Weighted MSE of (dphi/dt_pred - dphi/dt_phy), where dt_pred derived from absolute prediction.
    """
    r_pred_abs = (y_pred_abs - x_in_abs) / dt_phys.clamp_min(1e-12)
    diff = (r_pred_abs - r_phy_abs)
    w = area.clamp_min(1e-12).unsqueeze(1)  # [N,1]
    return (w * diff.pow(2)).sum() / w.sum().clamp_min(1e-12)
'''
def forward_main_head_with_edge_attr(model, x_in, edge_index, edge_attr=None):
    """
    Backward compatible: if your model ignores edge_attr, it still works.
    """
    if edge_attr is None:
        return forward_main_head(model, x_in, edge_index)  # your existing helper
    try:
        return forward_main_head(model, x_in, edge_index, edge_attr=edge_attr)
    except TypeError:
        # model/_forward_main_head doesn’t accept edge_attr yet
        return forward_main_head(model, x_in, edge_index)
'''
