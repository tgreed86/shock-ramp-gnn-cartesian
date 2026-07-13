import torch
import torch.nn.functional as F

def _get_refine_ratio(cfg: dict) -> int:
    pol = cfg.get("policy", {}) or {}
    mesh = cfg.get("mesh", {}) or {}
    rr = pol.get("refine_ratio", mesh.get("refine_ratio", 2))
    rr = int(rr)
    if rr < 2:
        raise ValueError(f"refine_ratio must be >=2, got {rr}")
    return rr

def _get_feature_name_list(cfg: dict) -> list[str] | None:
    feats = cfg.get("features", {}) or {}
    for k in ("dataset_order", "names", "use_columns"):
        v = feats.get(k, None)
        if isinstance(v, list) and len(v) > 0:
            return [str(x) for x in v]
    return None

def _tstats(name: str, t: torch.Tensor, max_elems: int = 0):
    if t is None:
        print(f"[NAN-DBG] {name}: None")
        return
    if not torch.is_tensor(t):
        print(f"[NAN-DBG] {name}: (non-tensor) {type(t)} = {t}")
        return

    tt = t.detach()
    finite = torch.isfinite(tt) if tt.dtype.is_floating_point or tt.dtype.is_complex else None

    n = tt.numel()
    msg = f"[NAN-DBG] {name}: shape={tuple(tt.shape)} dtype={tt.dtype} dev={tt.device} "

    # For float/complex: report finiteness and summary stats on finite values
    if tt.dtype.is_floating_point or tt.dtype.is_complex:
        nf = int(finite.sum().item())
        msg += f"finite={nf}/{n} "
        if nf > 0:
            v = tt[finite]
            msg += (
                f"min={v.min().item():.3e} max={v.max().item():.3e} "
                f"mean={v.mean().item():.3e} absmax={v.abs().max().item():.3e}"
            )
        else:
            msg += "ALL_NONFINITE"
        print(msg)

        if max_elems > 0 and nf < n:
            bad = torch.nonzero(~finite, as_tuple=False)
            bad = bad[:max_elems]
            print(f"[NAN-DBG] {name}: first nonfinite indices (up to {max_elems}): {bad.tolist()}")
        return

    # For non-float: no finiteness concept; report integer/bool stats
    # Always safe: min/max
    try:
        tmin = tt.min().item()
        tmax = tt.max().item()
        msg += f"min={tmin} max={tmax} "
    except Exception as e:
        msg += f"(min/max failed: {repr(e)}) "
        print(msg)
        return

    # If integer, report mean/absmax by casting to float FOR REPORTING ONLY
    if tt.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.long):
        v = tt.to(torch.float32)
        msg += f"mean={v.mean().item():.3e} absmax={v.abs().max().item():.3e}"
    elif tt.dtype == torch.bool:
        msg += f"true_count={int(tt.sum().item())}/{n}"
    else:
        # fallback
        pass

    print(msg)

def gas_gamma_from_cfg(cfg: dict, default: float = 1.4) -> float:
    """
    Resolve the ideal-gas gamma used by diagnostics/operators.
    Prefer loss.gamma to match the active training config, then physics/eos fallbacks.
    """
    loss = cfg.get("loss", {}) or {}
    physics = cfg.get("physics", {}) or {}
    eos = cfg.get("eos", {}) or {}
    for block in (loss, physics, eos):
        gamma = block.get("gamma", None)
        if gamma is not None:
            return float(gamma)
    return float(default)

def state_representation_from_cfg(cfg: dict, Fdim: int) -> str:
    """
    Infer whether state channels are conservative or primitive.
    Returns one of:
      - "conservative"
      - "primitive_uvrhope"  (U, V, RHO, P, E_internal)
    """
    names = _get_feature_name_list(cfg)
    if not names:
        return "primitive_uvrhope" if int(Fdim) >= 5 else "conservative"

    lower = [str(n).strip().lower() for n in names]

    def _has_exact_or_prefixed(keys: tuple[str, ...]) -> bool:
        for n in lower:
            if n in keys:
                return True
            for k in keys:
                if n.startswith(k + "_") or n.endswith("_" + k):
                    return True
        return False

    def _has_substr(keys: tuple[str, ...]) -> bool:
        return any(any(k in n for k in keys) for n in lower)

    has_u = _has_exact_or_prefixed(("u", "ux", "velx", "velocityx", "xvelocity")) or _has_substr(("x velocity", "velocity x"))
    has_v = _has_exact_or_prefixed(("v", "uy", "vely", "velocityy", "yvelocity")) or _has_substr(("y velocity", "velocity y"))
    has_rho = _has_substr(("rho", "dens"))
    has_p = _has_exact_or_prefixed(("p", "press", "pressure")) or _has_substr(("pressure",))
    has_mx = _has_substr(("xmom", "momx", "mx", "x_momentum", "momentum_x"))
    has_my = _has_substr(("ymom", "momy", "my", "y_momentum", "momentum_y"))

    if has_u and has_v and has_rho and has_p and not (has_mx or has_my):
        return "primitive_uvrhope"
    return "conservative"

def _state_views(
    x_abs: torch.Tensor,
    cfg: dict,
    *,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """
    Returns a canonical view dictionary with both conservative and primitive fields.
    Keys: rho,mx,my,E_tot,u,v,p,e_int
    """
    idx = infer_feature_indices(cfg, x_abs.size(1))
    rep = state_representation_from_cfg(cfg, x_abs.size(1))
    gamma = gas_gamma_from_cfg(cfg)

    if rep == "primitive_uvrhope":
        rho = x_abs[:, idx["rho"]]
        u = x_abs[:, idx["u"]]
        v = x_abs[:, idx["v"]]
        e_int = x_abs[:, idx["E"]]
        rho_safe = rho.abs().clamp_min(eps)
        mx = rho * u
        my = rho * v
        p_idx = idx.get("p", None)
        if p_idx is not None:
            p = x_abs[:, int(p_idx)]
        else:
            p = (gamma - 1.0) * rho_safe * e_int
        E_tot = rho_safe * e_int + 0.5 * rho_safe * (u * u + v * v)
        return {
            "rho": rho,
            "mx": mx,
            "my": my,
            "E_tot": E_tot,
            "u": u,
            "v": v,
            "p": p,
            "e_int": e_int,
        }

    rho = x_abs[:, idx["rho"]]
    mx = x_abs[:, idx["mx"]]
    my = x_abs[:, idx["my"]]
    E_tot = x_abs[:, idx["E"]]
    rho_safe = rho.abs().clamp_min(eps)
    u = mx / rho_safe
    v = my / rho_safe
    kinetic = 0.5 * (mx * mx + my * my) / rho_safe
    p = (gamma - 1.0) * (E_tot - kinetic)
    e_int = (E_tot / rho_safe) - 0.5 * (u * u + v * v)
    return {
        "rho": rho,
        "mx": mx,
        "my": my,
        "E_tot": E_tot,
        "u": u,
        "v": v,
        "p": p,
        "e_int": e_int,
    }

def state_views(
    x_abs: torch.Tensor,
    cfg: dict,
    *,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Public wrapper for canonical conservative/primitive state views."""
    return _state_views(x_abs, cfg, eps=eps)

def pressure_from_conservative_state(
    x_abs: torch.Tensor,  # [N,F] with channels rho,mx,my,E
    cfg: dict,
    *,
    eps: float = 1e-12,
    clamp_min: float = 0.0,
) -> torch.Tensor:
    """
    Ideal-gas pressure from conservative variables:
      p = (gamma - 1) * (E - 0.5*(mx^2+my^2)/rho)

    Assumes E is total energy density.
    """
    state = _state_views(x_abs, cfg, eps=eps)
    p = state["p"]

    if clamp_min is not None:
        p = p.clamp_min(float(clamp_min))
    return p

def specific_internal_energy_from_conservative_state(
    x_abs: torch.Tensor,  # [N,F] with channels rho,mx,my,E
    cfg: dict,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Specific internal energy from conservative variables:
      e_int = E/rho - 0.5 * |u|^2

    Assumes E is total energy density.
    """
    state = _state_views(x_abs, cfg, eps=eps)
    return state["e_int"]

def specific_entropy_from_conservative_state(
    x_abs: torch.Tensor,  # [N,F] with channels rho,mx,my,E
    cfg: dict,
    *,
    eps: float = 1e-12,
    p_floor: float | None = None,
) -> torch.Tensor:
    """
    Ideal-gas specific entropy proxy (up to an additive constant):
      s ~ log(p) - gamma*log(rho)
    """
    state = _state_views(x_abs, cfg, eps=eps)
    rho = state["rho"].abs().clamp_min(eps)
    p_min = max(float(eps), float(p_floor if p_floor is not None else eps))
    p = pressure_from_conservative_state(x_abs, cfg, eps=eps, clamp_min=p_min)
    gamma = gas_gamma_from_cfg(cfg)
    return torch.log(p) - gamma * torch.log(rho)

def _normalize_reconstruction_kind(name: str) -> str:
    n = str(name).strip().lower()
    if n in ("first_order", "first-order", "first", "none", "piecewise_constant", "piecewise-constant"):
        return "first_order"
    if n in ("muscl", "tvd", "second_order", "second-order", "2nd"):
        return "muscl"
    raise RuntimeError(f"Unsupported advection_reconstruction='{name}'. Use 'first_order' or 'muscl'.")

def _normalize_limiter_name(name: str) -> str:
    n = str(name).strip().lower()
    if n in ("minmod", "mc", "vanleer", "van_leer", "van-leer"):
        return "vanleer" if n in ("van_leer", "van-leer") else n
    raise RuntimeError(f"Unsupported muscl_limiter='{name}'. Use one of: minmod, mc, vanleer.")

def _minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    same_sign = (a * b) > 0
    return torch.where(same_sign, torch.sign(a) * torch.minimum(a.abs(), b.abs()), torch.zeros_like(a))

def _minmod3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return _minmod(_minmod(a, b), c)

def _apply_tvd_limiter(a: torch.Tensor, b: torch.Tensor, limiter: str) -> torch.Tensor:
    """
    Limit slope pair (a,b) with a TVD limiter.
      a: reconstructed directional slope
      b: one-sided edge slope
    """
    lim = _normalize_limiter_name(limiter)
    if lim == "minmod":
        return _minmod(a, b)
    if lim == "mc":
        return _minmod3(2.0 * a, 0.5 * (a + b), 2.0 * b)
    # van Leer
    prod = a * b
    denom = (a + b).abs().clamp_min(1e-12)
    out = 2.0 * prod / denom * torch.sign(a + b)
    return torch.where(prod > 0.0, out, torch.zeros_like(a))

def _compute_node_gradients_ls(
    phi: torch.Tensor,          # [N,F]
    edge_index: torch.Tensor,   # [2,E]
    nx: torch.Tensor,           # [E]
    ny: torch.Tensor,           # [E]
    edge_dist: torch.Tensor,    # [E]
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Least-squares nodal gradients from edge directional derivatives.
    Returns grad: [N,F,2] (x,y components).
    """
    N, Fdim = phi.shape
    src = edge_index[0].long()
    dst = edge_index[1].long()

    dist = edge_dist.to(device=phi.device, dtype=phi.dtype).clamp_min(eps)
    nx_ = nx.to(device=phi.device, dtype=phi.dtype)
    ny_ = ny.to(device=phi.device, dtype=phi.dtype)
    nrm = torch.sqrt((nx_ * nx_ + ny_ * ny_).clamp_min(eps))
    nxu = nx_ / nrm
    nyu = ny_ / nrm

    # directional slope along src->dst
    s = (phi[dst] - phi[src]) / dist[:, None]  # [E,F]

    # Assemble normal equations A_i g_i = b_i
    A00 = phi.new_zeros((N,))
    A01 = phi.new_zeros((N,))
    A11 = phi.new_zeros((N,))
    b0 = phi.new_zeros((N, Fdim))
    b1 = phi.new_zeros((N, Fdim))

    c00 = nxu * nxu
    c01 = nxu * nyu
    c11 = nyu * nyu
    rhs0 = nxu[:, None] * s
    rhs1 = nyu[:, None] * s

    # Add contributions symmetrically to both edge endpoints.
    A00.index_add_(0, src, c00); A00.index_add_(0, dst, c00)
    A01.index_add_(0, src, c01); A01.index_add_(0, dst, c01)
    A11.index_add_(0, src, c11); A11.index_add_(0, dst, c11)
    b0.index_add_(0, src, rhs0); b0.index_add_(0, dst, rhs0)
    b1.index_add_(0, src, rhs1); b1.index_add_(0, dst, rhs1)

    tr = A00 + A11
    reg = (1e-10 + 1e-8 * tr).to(phi.dtype)
    A00r = A00 + reg
    A11r = A11 + reg
    det = (A00r * A11r - A01 * A01).clamp_min(eps)

    gx = (A11r[:, None] * b0 - A01[:, None] * b1) / det[:, None]
    gy = (-A01[:, None] * b0 + A00r[:, None] * b1) / det[:, None]
    return torch.stack([gx, gy], dim=2)  # [N,F,2]

def _muscl_reconstruct_face_states(
    phi: torch.Tensor,          # [N,F]
    edge_index: torch.Tensor,   # [2,E]
    nx: torch.Tensor,           # [E]
    ny: torch.Tensor,           # [E]
    edge_dist: torch.Tensor,    # [E]
    *,
    limiter: str = "minmod",
    eps: float = 1e-12,
):
    """
    MUSCL/TVD face reconstruction for directed edge src->dst.
    Returns (phi_L, phi_R), both [E,F].
    """
    src = edge_index[0].long()
    dst = edge_index[1].long()

    dist = edge_dist.to(device=phi.device, dtype=phi.dtype).clamp_min(eps)
    nx_ = nx.to(device=phi.device, dtype=phi.dtype)
    ny_ = ny.to(device=phi.device, dtype=phi.dtype)
    nrm = torch.sqrt((nx_ * nx_ + ny_ * ny_).clamp_min(eps))
    nxu = nx_ / nrm
    nyu = ny_ / nrm

    grad = _compute_node_gradients_ls(phi, edge_index, nxu, nyu, dist, eps=eps)  # [N,F,2]

    proj_src = grad[src, :, 0] * nxu[:, None] + grad[src, :, 1] * nyu[:, None]    # [E,F]
    proj_dst_to_src = -(grad[dst, :, 0] * nxu[:, None] + grad[dst, :, 1] * nyu[:, None])  # [E,F]

    slope_edge = (phi[dst] - phi[src]) / dist[:, None]  # [E,F]
    lim_src = _apply_tvd_limiter(proj_src, slope_edge, limiter)
    lim_dst_to_src = _apply_tvd_limiter(proj_dst_to_src, -slope_edge, limiter)

    half_d = 0.5 * dist[:, None]
    phi_L = phi[src] + half_d * lim_src
    phi_R = phi[dst] + half_d * lim_dst_to_src
    return phi_L, phi_R

def dec_divergence_euler_flux(
    x_abs: torch.Tensor,       # [N,F] (expects conservative channels exist)
    edge_index: torch.Tensor,  # [2,E] (directed edges: src->dst)
    nx: torch.Tensor,          # [E]
    ny: torch.Tensor,          # [E]
    face_len: torch.Tensor,    # [E]
    edge_dist: torch.Tensor | None,  # [E] center distance (needed for MUSCL)
    area: torch.Tensor,        # [N]
    cfg: dict,
    *,
    scheme: str = "rusanov",
    reconstruction: str = "first_order",
    limiter: str = "minmod",
    eps: float = 1e-8,
):
    """
    Returns div(F) in conservative ordering [rho, mx, my, E] as [N,4],
    where F is the Euler flux dotted with the edge normal.
    Uses a Rusanov (local LF) numerical flux by default.

    IMPORTANT: Assumes edge normals (nx,ny) point outward from src toward dst,
    matching how your scalar div uses directed edges.
    """
    loss = cfg.get("loss", {}) or {}
    gamma = float(loss.get("gamma", 1.4))

    N, Fdim = x_abs.shape

    src = edge_index[0].long()
    dst = edge_index[1].long()
    rec_kind = _normalize_reconstruction_kind(reconstruction)

    state = _state_views(x_abs, cfg, eps=eps)
    rho_node = state["rho"].abs().clamp_min(eps)
    mx_node = state["mx"]
    my_node = state["my"]
    E_node = state["E_tot"]
    U_nodes = torch.stack([rho_node, mx_node, my_node, E_node], dim=1)  # [N,4]

    if rec_kind == "muscl":
        if edge_dist is None:
            raise RuntimeError("MUSCL reconstruction requires edge_dist in dec_divergence_euler_flux.")
        U_L, U_R = _muscl_reconstruct_face_states(
            U_nodes, edge_index, nx, ny, edge_dist, limiter=limiter, eps=eps
        )
    else:
        U_L = U_nodes[src]
        U_R = U_nodes[dst]

    rho_L = U_L[:, 0].clamp_min(eps)
    mx_L  = U_L[:, 1]
    my_L  = U_L[:, 2]
    E_L   = U_L[:, 3]
    rho_R = U_R[:, 0].clamp_min(eps)
    mx_R  = U_R[:, 1]
    my_R  = U_R[:, 2]
    E_R   = U_R[:, 3]

    u_clip = float(loss.get("u_clip", 1e3))
    u_L = mx_L / rho_L
    v_L = my_L / rho_L
    u_R = mx_R / rho_R
    v_R = my_R / rho_R
    if u_clip > 0:
        u_L = u_clip * torch.tanh(u_L / u_clip)
        v_L = u_clip * torch.tanh(v_L / u_clip)
        u_R = u_clip * torch.tanh(u_R / u_clip)
        v_R = u_clip * torch.tanh(v_R / u_clip)

    p_floor = float(loss.get("p_floor", 0.0))
    p_L = ((gamma - 1.0) * (E_L - 0.5 * (mx_L * mx_L + my_L * my_L) / rho_L)).clamp_min(p_floor)
    p_R = ((gamma - 1.0) * (E_R - 0.5 * (mx_R * mx_R + my_R * my_R) / rho_R)).clamp_min(p_floor)
    c_L = torch.sqrt((gamma * p_L / rho_L).clamp_min(0.0))
    c_R = torch.sqrt((gamma * p_R / rho_R).clamp_min(0.0))

    # normal velocity
    un_L = u_L * nx + v_L * ny
    un_R = u_R * nx + v_R * ny

    # physical flux dotted with n, per side: [E,4]
    # F_n(U) = [ rho*un,
    #            mx*un + p*nx,
    #            my*un + p*ny,
    #            (E+p)*un ]
    F_L = torch.stack(
        [
            rho_L * un_L,
            mx_L  * un_L + p_L * nx,
            my_L  * un_L + p_L * ny,
            (E_L + p_L) * un_L,
        ],
        dim=1,
    )
    F_R = torch.stack(
        [
            rho_R * un_R,
            mx_R  * un_R + p_R * nx,
            my_R  * un_R + p_R * ny,
            (E_R + p_R) * un_R,
        ],
        dim=1,
    )

    scheme = str(scheme).lower()
    if scheme in ("central", "avg", "average"):
        F_star = 0.5 * (F_L + F_R)
    else:
        # Rusanov / local LF
        smax = torch.maximum(un_L.abs() + c_L, un_R.abs() + c_R)  # [E]
        U_L = torch.stack([rho_L, mx_L, my_L, E_L], dim=1)
        U_R = torch.stack([rho_R, mx_R, my_R, E_R], dim=1)
        F_star = 0.5 * (F_L + F_R) - 0.5 * smax[:, None] * (U_R - U_L)

    # accumulate divergence on src (directed outward normals)
    div = x_abs.new_zeros((N, 4))
    div.index_add_(0, src, face_len[:, None] * F_star)
    div = div / area.clamp_min(1e-12)[:, None]

    return div

def _euler_primitive_rates_from_conservative_rates(
    *,
    x_abs: torch.Tensor,
    cfg: dict,
    qdot_cons: torch.Tensor,   # [N,4] in [rho,mx,my,E_tot] ordering
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """
    Convert conservative Euler rates to primitive rates for state
    representation [u, v, rho, p, e_int].
    """
    if qdot_cons.ndim != 2 or qdot_cons.size(1) != 4:
        raise RuntimeError(
            f"qdot_cons must have shape [N,4], got {tuple(qdot_cons.shape)}"
        )

    state = _state_views(x_abs, cfg, eps=eps)
    gamma = gas_gamma_from_cfg(cfg)

    rho = state["rho"]
    u = state["u"]
    v = state["v"]
    e_int = state["e_int"]
    rho_safe = rho.abs().clamp_min(eps)

    drho = qdot_cons[:, 0]
    dmx = qdot_cons[:, 1]
    dmy = qdot_cons[:, 2]
    dEt = qdot_cons[:, 3]

    du = (dmx - u * drho) / rho_safe
    dv = (dmy - v * drho) / rho_safe

    kinetic = 0.5 * (u * u + v * v)
    de_int = (dEt - drho * (e_int + kinetic) - rho_safe * (u * du + v * dv)) / rho_safe
    dp = (gamma - 1.0) * (drho * e_int + rho_safe * de_int)

    return {
        "rho": drho,
        "mx": dmx,
        "my": dmy,
        "E_tot": dEt,
        "u": du,
        "v": dv,
        "p": dp,
        "e_int": de_int,
    }

def infer_feature_indices(cfg: dict, Fdim: int):
    """
    Returns indices for density, momx, momy, energy.
    Falls back to a reasonable guess if names are missing.
    """
    names = _get_feature_name_list(cfg)
    if not names:
        if Fdim >= 5:
            # Default primitive order: [U,V,RHO,P,Eint]
            return {"rho": 2, "mx": 0, "my": 1, "E": 4, "u": 0, "v": 1, "p": 3}
        if Fdim == 4:
            return {"rho": 0, "mx": 1, "my": 2, "E": 3, "u": 1, "v": 2, "p": None}
        return {
            "rho": 0,
            "mx": 1 if Fdim > 1 else 0,
            "my": 2 if Fdim > 2 else 0,
            "E": min(Fdim - 1, 3),
            "u": 1 if Fdim > 1 else 0,
            "v": 2 if Fdim > 2 else 0,
            "p": None,
        }

    lower = [str(n).strip().lower() for n in names]

    def find_any(keys: tuple[str, ...]) -> int | None:
        for i, n in enumerate(lower):
            if any(k in n for k in keys):
                return i
        return None

    def find_exact_or_prefixed(keys: tuple[str, ...]) -> int | None:
        for i, n in enumerate(lower):
            if n in keys:
                return i
            for k in keys:
                if n.startswith(k + "_") or n.endswith("_" + k):
                    return i
        return None

    rho = find_any(("dens", "rho"))
    mx = find_any(("x-mom", "xmom", "momx", "px", "mx", "x_momentum", "momentum_x"))
    my = find_any(("y-mom", "ymom", "momy", "py", "my", "y_momentum", "momentum_y"))
    En = find_any(("ener", "total e", "etot", "energy", "internal_energy", "specific_energy", "eint"))
    if En is None:
        En = find_exact_or_prefixed(("e",))

    u = find_exact_or_prefixed(("u", "ux", "velx", "velocityx", "xvelocity"))
    v = find_exact_or_prefixed(("v", "uy", "vely", "velocityy", "yvelocity"))
    p = find_exact_or_prefixed(("p", "press", "pressure"))
    if p is None:
        p = find_any(("pressure",))

    rep = state_representation_from_cfg(cfg, Fdim)

    if rep == "primitive_uvrhope":
        if u is None:
            u = 0 if Fdim > 0 else None
        if v is None:
            v = 1 if Fdim > 1 else u
        if rho is None:
            rho = 2 if Fdim > 2 else 0
        if p is None:
            p = 3 if Fdim > 3 else None
        if En is None:
            En = 4 if Fdim > 4 else (Fdim - 1)
        return {
            "rho": int(rho),
            "mx": int(u),
            "my": int(v),
            "E": int(En),
            "u": int(u),
            "v": int(v),
            "p": (None if p is None else int(p)),
        }

    if rho is None and Fdim == 4:
        rho = 0
    if mx is None and Fdim == 4:
        mx = 1
    if my is None and Fdim == 4:
        my = 2
    if En is None and Fdim == 4:
        En = 3

    rho = 0 if rho is None else int(rho)
    mx = 1 if mx is None else int(mx)
    my = 2 if my is None else int(my)
    En = min(Fdim - 1, 3) if En is None else int(En)
    return {"rho": rho, "mx": mx, "my": my, "E": En, "u": mx, "v": my, "p": None}

# -----------------------------
# PARC / Variant-B helpers
# -----------------------------

def dec_advdiff_terms_abs(
    x_abs: torch.Tensor,         # [N,F] absolute state on the pred mesh at this step
    edge_index: torch.Tensor,    # [2,E]
    pred_ea: torch.Tensor,       # [E,>=5] includes tau in last col
    levels: torch.Tensor | None,        # [N] or None for uniform
    *,
    dx0: float,
    dy0: float,
    cfg: dict,
    compute_adv: bool = True,
    compute_diff: bool = True,
):
    """
    Compute operator terms in ABSOLUTE physical units:
      r_adv_abs  ≈ -div(F_adv)        (same units as dphi/dt)
      r_diff_abs ≈  nu * Laplacian(phi)

    - Advection can be:
        * "scalar" : -div(u * phi) using dec_divergence_advective_flux
        * "euler"  : conservative Euler flux divergence on [rho,mx,my,E]
    - Diffusion stays scalar Laplacian on selected channels (e.g. rho,E).
    """
    if x_abs.ndim != 2:
        raise ValueError(f"x_abs must be [N,F], got {tuple(x_abs.shape)}")
    N, Fdim = x_abs.shape

    loss = cfg.get("loss", {}) or {}
    rr = _get_refine_ratio(cfg)
    idx = infer_feature_indices(cfg, Fdim)

    # --- helpers ---
    def _names_to_indices(names, *, default, strict: bool = False, selection_name: str = "channels"):
        if not names:
            return list(default)
        out = []
        seen = set()
        unknown = []
        for name in names:
            n = str(name).lower()
            j = None
            if ("dens" in n) or (n == "rho"):
                j = idx["rho"]
            elif n in ("u", "ux", "x_velocity", "velocity_x", "velx"):
                j = idx.get("u", idx["mx"])
            elif n in ("v", "uy", "y_velocity", "velocity_y", "vely"):
                j = idx.get("v", idx["my"])
            elif ("press" in n) or (n == "p") or (n == "pressure"):
                p_idx = idx.get("p", None)
                if p_idx is None:
                    unknown.append(str(name))
                    continue
                j = int(p_idx)
            elif ("ener" in n) or (n == "e"):
                j = idx["E"]
            elif ("x" in n and "mom" in n) or (n in ("mx", "x_momentum", "mom_x")):
                j = idx["mx"]
            elif ("y" in n and "mom" in n) or (n in ("my", "y_momentum", "mom_y")):
                j = idx["my"]
            else:
                unknown.append(str(name))
                continue
            if j is None:
                unknown.append(str(name))
                continue
            if j not in seen:
                out.append(j)
                seen.add(j)
        if strict and len(unknown) > 0:
            raise RuntimeError(
                f"Strict Euler advection rejected unknown {selection_name}: {unknown}. "
                "Please fix loss.*_channels entries."
            )
        if len(out) == 0:
            if strict:
                raise RuntimeError(
                    f"Strict Euler advection could not map any {selection_name} from {names!r}."
                )
            return list(default)
        return out

    # --- geometry / weights ---
    if levels is None:
        area = x_abs.new_full((N,), float(dx0) * float(dy0))
    else:
        area = cell_area_from_levels(
            levels, dx0=dx0, dy0=dy0, dtype=x_abs.dtype, device=x_abs.device, refine_ratio=rr
        )  # [N]

    # --- edge geometry ---
    # pred_ea layout: [nx, ny, face_len, dual_len, tau]
    nx, ny, face_len, dual_len, tau = edge_attr_unpack(pred_ea)
    nx = nx.to(dtype=x_abs.dtype)
    ny = ny.to(dtype=x_abs.dtype)
    face_len = face_len.to(dtype=x_abs.dtype)
    dual_len = dual_len.to(dtype=x_abs.dtype)
    tau = tau.to(dtype=x_abs.dtype)

    # --- channel selections ---
    legacy = loss.get("channels", None)
    adv_names  = (loss.get("adv_channels", None)
                  or loss.get("dec_adv_channels", None)
                  or legacy)
    diff_names = (loss.get("diff_channels", None)
                  or loss.get("dec_diff_channels", None)
                  or legacy)

    advection_type = str(loss.get("advection_type", "scalar")).strip().lower()
    if advection_type not in ("scalar", "euler"):
        raise RuntimeError(
            f"Unsupported loss.advection_type={advection_type!r}. Use 'scalar' or 'euler'."
        )
    state_rep = state_representation_from_cfg(cfg, Fdim)
    advection_reconstruction = str(loss.get("advection_reconstruction", "first_order"))
    muscl_limiter = str(loss.get("muscl_limiter", "minmod"))

    # defaults:
    # - scalar advection default: rho + E (your prior behavior)
    # - euler advection default: full conservative state
    if state_rep == "primitive_uvrhope":
        p_idx = idx.get("p", None)
        default_adv_scalar = [idx["u"], idx["v"], idx["rho"]]
        if p_idx is not None:
            default_adv_scalar.append(int(p_idx))
        default_adv_scalar.append(idx["E"])
    else:
        default_adv_scalar = [idx["rho"], idx["E"]]
    default_adv_euler  = [idx["rho"], idx["mx"], idx["my"], idx["E"]]
    if state_rep == "primitive_uvrhope":
        p_idx = idx.get("p", None)
        default_diff = [idx["rho"]]
        if p_idx is not None:
            default_diff.append(int(p_idx))
        default_diff.append(idx["E"])
    else:
        default_diff = [idx["rho"], idx["E"]]

    sel_adv  = _names_to_indices(
        adv_names,
        default=(default_adv_euler if advection_type == "euler" else default_adv_scalar),
        strict=(advection_type == "euler"),
        selection_name="adv_channels",
    )
    sel_diff = _names_to_indices(diff_names, default=default_diff)

    # allocate full-sized outputs (fill selected columns only)
    r_adv  = x_abs.new_zeros((N, Fdim))
    r_diff = x_abs.new_zeros((N, Fdim))

    # ----- advection component -----
    if compute_adv and (len(sel_adv) > 0):
        if advection_type == "euler":
            scheme = str(loss.get("euler_flux_scheme", "rusanov")).lower()
            rho_eps = float(loss.get("rho_eps", 1e-8))

            div_euler_full = dec_divergence_euler_flux(
                x_abs=x_abs,
                edge_index=edge_index,
                nx=nx, ny=ny,
                face_len=face_len,
                edge_dist=dual_len,
                area=area,
                cfg=cfg,
                scheme=scheme,
                reconstruction=advection_reconstruction,
                limiter=muscl_limiter,
                eps=rho_eps,
            )  # [N,4] ordered [rho,mx,my,E]

            qdot_cons = -div_euler_full
            if state_rep == "primitive_uvrhope":
                prim_rates = _euler_primitive_rates_from_conservative_rates(
                    x_abs=x_abs,
                    cfg=cfg,
                    qdot_cons=qdot_cons,
                    eps=rho_eps,
                )
                p_idx = idx.get("p", None)
                rate_by_index: dict[int, torch.Tensor] = {
                    int(idx["rho"]): prim_rates["rho"],
                    int(idx.get("u", idx["mx"])): prim_rates["u"],
                    int(idx.get("v", idx["my"])): prim_rates["v"],
                    int(idx["E"]): prim_rates["e_int"],
                }
                if p_idx is not None:
                    rate_by_index[int(p_idx)] = prim_rates["p"]
            else:
                rate_by_index = {
                    int(idx["rho"]): qdot_cons[:, 0],
                    int(idx["mx"]): qdot_cons[:, 1],
                    int(idx["my"]): qdot_cons[:, 2],
                    int(idx["E"]): qdot_cons[:, 3],
                    int(idx.get("u", idx["mx"])): qdot_cons[:, 1],
                    int(idx.get("v", idx["my"])): qdot_cons[:, 2],
                }

            unsupported = [int(j) for j in sel_adv if int(j) not in rate_by_index]
            if len(unsupported) > 0:
                raise RuntimeError(
                    "Strict Euler advection cannot provide rates for selected channel indices "
                    f"{unsupported}. state_rep={state_rep}, sel_adv={sel_adv}, idx={idx}"
                )
            for j in sel_adv:
                r_adv[:, int(j)] = rate_by_index[int(j)]

        else:
            # scalar advection: -div(u * phi)
            rho_eps = float(loss.get("rho_eps", 1e-8))
            vel = compute_velocity_from_state(x_abs, cfg, eps=rho_eps)  # [N,2]
            phi_adv = x_abs[:, sel_adv]  # [N,Ca]
            scheme = str(loss.get("advection_scheme", "upwind")).lower()

            div_adv = dec_divergence_advective_flux(
                phi=phi_adv,
                vel=vel,
                edge_index=edge_index,
                nx=nx, ny=ny,
                face_len=face_len,
                edge_dist=dual_len,
                area=area,
                scheme=scheme,
                reconstruction=advection_reconstruction,
                limiter=muscl_limiter,
            )  # [N,Ca]

            r_adv[:, sel_adv] = -div_adv

    # ----- diffusion component -----
    if compute_diff and (len(sel_diff) > 0):
        nu_full = as_nu_tensor(loss.get("nu", 0.0), Fdim, device=x_abs.device, dtype=x_abs.dtype)  # [F]
        nu_sel  = nu_full[torch.as_tensor(sel_diff, device=x_abs.device)].view(1, -1)             # [1,Cd]

        phi_diff = x_abs[:, sel_diff]  # [N,Cd]
        lap = dec_laplacian(phi_diff, edge_index=edge_index, tau=tau, area=area)  # [N,Cd]
        r_diff[:, sel_diff] = lap * nu_sel

    return (r_adv if compute_adv else None), (r_diff if compute_diff else None), area



def _channels_to_indices(cfg: dict, Fdim: int, names, *, default: list[int]) -> list[int]:
    """Map a list of human-readable channel names to integer indices."""
    idx = infer_feature_indices(cfg, Fdim)
    if not names:
        return list(default)

    out: list[int] = []
    for name in names:
        n = str(name).lower()
        if ("dens" in n) or (n == "rho"):
            out.append(idx["rho"])
        elif n in ("u", "ux", "x_velocity", "velocity_x", "velx"):
            out.append(int(idx.get("u", idx["mx"])))
        elif n in ("v", "uy", "y_velocity", "velocity_y", "vely"):
            out.append(int(idx.get("v", idx["my"])))
        elif ("press" in n) or (n == "p") or (n == "pressure"):
            p_idx = idx.get("p", None)
            if p_idx is not None:
                out.append(int(p_idx))
        elif ("ener" in n) or (n == "e"):
            out.append(idx["E"])
        elif (("x" in n) and ("mom" in n)) or (n in ("mx", "momx", "mom_x", "xmom", "x_momentum")):
            out.append(idx["mx"])
        elif (("y" in n) and ("mom" in n)) or (n in ("my", "momy", "mom_y", "ymom", "y_momentum")):
            out.append(idx["my"])

    # fallback if user provided something unusable
    if len(out) == 0:
        out = list(default)

    # de-dup while preserving order
    seen = set()
    out2 = []
    for i in out:
        if i not in seen:
            out2.append(int(i))
            seen.add(i)
    return out2

def _names_to_indices(cfg: dict, Fdim: int, names, default: list[int]) -> list[int]:
    """
    Map channel-name strings to indices using infer_feature_indices().
    Preserves order and removes duplicates.
    """
    idx = infer_feature_indices(cfg, Fdim)
    if not names:
        return list(default)

    out = []
    for name in names:
        n = str(name).lower()
        if ("dens" in n) or (n == "rho") or (n == "density"):
            out.append(idx["rho"])
        elif n in ("u", "ux", "x_velocity", "velocity_x", "velx"):
            out.append(int(idx.get("u", idx["mx"])))
        elif n in ("v", "uy", "y_velocity", "velocity_y", "vely"):
            out.append(int(idx.get("v", idx["my"])))
        elif ("press" in n) or (n == "p") or (n == "pressure"):
            p_idx = idx.get("p", None)
            if p_idx is not None:
                out.append(int(p_idx))
        elif ("ener" in n) or (n == "e") or (n == "energy"):
            out.append(idx["E"])
        elif (("x" in n) and ("mom" in n)) or (n in ("mx", "mom_x", "x_momentum")):
            out.append(idx["mx"])
        elif (("y" in n) and ("mom" in n)) or (n in ("my", "mom_y", "y_momentum")):
            out.append(idx["my"])

    # de-dup, keep order
    seen = set()
    out2 = []
    for i in out:
        if i not in seen:
            seen.add(i)
            out2.append(i)

    return out2 if len(out2) > 0 else list(default)


def parc_select_feature_indices_adv(cfg: dict, Fdim: int) -> list[int]:
    """
    Which state channels get *advection* operator-term inputs.

    Priority:
      1) loss.parc_input_channels_adv (list[str])
      2) loss.adv_channels / loss.dec_adv_channels (list[str])
      3) loss.parc_input_channels (list[str])
      4) loss.channels (list[str])
      5) default [rho, E]
    """
    loss = cfg.get("loss", {}) or {}
    idx = infer_feature_indices(cfg, Fdim)
    if state_representation_from_cfg(cfg, Fdim) == "primitive_uvrhope":
        p_idx = idx.get("p", None)
        default = [idx["u"], idx["v"], idx["rho"]]
        if p_idx is not None:
            default.append(int(p_idx))
        default.append(idx["E"])
    else:
        default = [idx["rho"], idx["E"]]

    names = (loss.get("parc_input_channels_adv", None)
             or loss.get("adv_channels", None)
             or loss.get("dec_adv_channels", None)
             or loss.get("parc_input_channels", None)
             or loss.get("channels", None))

    return _channels_to_indices(cfg, Fdim, names, default=default)


def parc_select_feature_indices_diff(cfg: dict, Fdim: int) -> list[int]:
    """
    Which state channels get *diffusion* operator-term inputs.

    Priority:
      1) loss.parc_input_channels_diff (list[str])
      2) loss.diff_channels / loss.dec_diff_channels (list[str])
      3) loss.parc_input_channels (list[str])
      4) loss.channels (list[str])
      5) default [rho, E]
    """
    loss = cfg.get("loss", {}) or {}
    idx = infer_feature_indices(cfg, Fdim)
    if state_representation_from_cfg(cfg, Fdim) == "primitive_uvrhope":
        p_idx = idx.get("p", None)
        default = [idx["rho"]]
        if p_idx is not None:
            default.append(int(p_idx))
        default.append(idx["E"])
    else:
        default = [idx["rho"], idx["E"]]

    names = (loss.get("parc_input_channels_diff", None)
             or loss.get("diff_channels", None)
             or loss.get("dec_diff_channels", None)
             or loss.get("parc_input_channels", None)
             or loss.get("channels", None))

    return _channels_to_indices(cfg, Fdim, names, default=default)


def parc_select_feature_indices(cfg: dict, Fdim: int) -> list[int]:
    """
    Which state channels get operator-term inputs.

    Backwards compatible:
      - If you don't specify any split adv/diff channel keys, this behaves like the
        original implementation (defaults to rho + E unless overridden).

    If you *do* specify split keys (adv_channels/diff_channels or parc_input_channels_*),
    this returns the UNION of the adv+diff selections. This is useful for building masks
    that should cover all physics-touched channels (baseline/residual).
    """
    loss = cfg.get("loss", {}) or {}

    has_split = any(
        k in loss and loss.get(k, None)
        for k in ("parc_input_channels_adv", "parc_input_channels_diff",
                  "adv_channels", "diff_channels", "dec_adv_channels", "dec_diff_channels")
    )
    if has_split:
        sel_adv = parc_select_feature_indices_adv(cfg, Fdim)
        sel_diff = parc_select_feature_indices_diff(cfg, Fdim)
        return sorted(set(sel_adv + sel_diff))

    # ---- original behavior ----
    names = loss.get("parc_input_channels", None)
    if not names:
        names = loss.get("channels", None)

    idx = infer_feature_indices(cfg, Fdim)
    if state_representation_from_cfg(cfg, Fdim) == "primitive_uvrhope":
        p_idx = idx.get("p", None)
        default = [idx["rho"]]
        if p_idx is not None:
            default.append(int(p_idx))
        default.append(idx["E"])
    else:
        default = [idx["rho"], idx["E"]]
    if not names:
        return default

    return _channels_to_indices(cfg, Fdim, names, default=default)

def _operator_time_scale(
    *,
    cfg: dict,
    dt_phys: torch.Tensor,
    dt_ref: torch.Tensor | None,
    predict_type: str,
) -> torch.Tensor | float | None:
    """
    Time factor used when converting absolute operator rates to model inputs.

    Default "model" preserves the legacy behavior:
      rate model -> dt_ref * r / sigma, delta model -> dt_phys * r / sigma.

    For operator features it is often useful to decouple this from the model
    target and use a scale-only input such as r / sigma. Set
    loss.parc_input_time_scale="unit" for that behavior.
    """
    loss = cfg.get("loss", {}) or {}
    mode = str(loss.get("parc_input_time_scale", "model")).strip().lower()
    aliases = {
        "legacy": "model",
        "target": "model",
        "model_units": "model",
        "dt_ref": "dt_ref",
        "ref": "dt_ref",
        "reference": "dt_ref",
        "dt": "dt_phys",
        "dt_phys": "dt_phys",
        "physical_dt": "dt_phys",
        "unit": "unit",
        "none": "unit",
        "no_dt": "unit",
        "rate": "unit",
    }
    mode = aliases.get(mode, mode)

    if mode == "model":
        if str(predict_type).strip().lower() == "delta":
            return dt_phys
        if str(predict_type).strip().lower() == "rate":
            return dt_ref
        return None
    if mode == "dt_ref":
        return dt_ref
    if mode == "dt_phys":
        return dt_phys
    if mode == "unit":
        return 1.0
    try:
        return float(mode)
    except Exception as exc:
        raise ValueError(
            "loss.parc_input_time_scale must be one of "
            "{model, dt_ref, dt_phys, unit} or a numeric scalar; "
            f"got {loss.get('parc_input_time_scale')!r}."
        ) from exc


def operator_rate_to_input_units(
    r_abs: torch.Tensor,
    *,
    dt_phys: torch.Tensor,
    dt_ref: torch.Tensor | None,
    sigma: torch.Tensor | None,
    predict_type: str,
    cfg: dict,
) -> torch.Tensor:
    """Convert an absolute operator rate to PARC input units."""
    if sigma is None:
        sigma_use = 1.0
    else:
        sigma_use = sigma.view(1, -1).clamp_min(1e-12)

    time_scale = _operator_time_scale(
        cfg=cfg,
        dt_phys=dt_phys,
        dt_ref=dt_ref,
        predict_type=predict_type,
    )
    if time_scale is None:
        return torch.zeros_like(r_abs)
    if torch.is_tensor(time_scale):
        time_scale = time_scale.to(device=r_abs.device, dtype=r_abs.dtype)
    return (r_abs * time_scale) / sigma_use


def _feature_name_by_index(cfg: dict, Fdim: int) -> dict[int, str]:
    names = _get_feature_name_list(cfg)
    if not names:
        names = [str(i) for i in range(Fdim)]
    out: dict[int, str] = {}
    for i in range(Fdim):
        out[i] = str(names[i]) if i < len(names) else str(i)
    return out


def _scale_tensor_from_cfg_value(
    raw,
    *,
    cfg: dict,
    Fdim: int,
    selected_indices: list[int],
    device,
    dtype,
    key: str,
) -> torch.Tensor:
    n = len(selected_indices)
    if n == 0:
        return torch.empty((0,), device=device, dtype=dtype)
    if raw is None:
        return torch.ones((n,), device=device, dtype=dtype)
    if isinstance(raw, (int, float)):
        return torch.full((n,), float(raw), device=device, dtype=dtype)
    if isinstance(raw, (list, tuple)):
        if len(raw) == n:
            vals = [float(v) for v in raw]
        elif len(raw) == Fdim:
            vals = [float(raw[int(i)]) for i in selected_indices]
        elif len(raw) == 1:
            vals = [float(raw[0])] * n
        else:
            raise ValueError(
                f"loss.{key} must have length 1, selected-channel length {n}, "
                f"or full feature length {Fdim}; got length {len(raw)}."
            )
        return torch.tensor(vals, device=device, dtype=dtype)
    if isinstance(raw, dict):
        names = _feature_name_by_index(cfg, Fdim)
        lower_to_idx = {v.lower(): i for i, v in names.items()}
        vals = []
        for idx in selected_indices:
            idx_i = int(idx)
            name = names.get(idx_i, str(idx_i))
            candidates = (str(idx_i), name, name.lower())
            found = None
            for cand in candidates:
                if cand in raw:
                    found = raw[cand]
                    break
            if found is None and name.lower() in lower_to_idx and name.lower() in raw:
                found = raw[name.lower()]
            vals.append(1.0 if found is None else float(found))
        return torch.tensor(vals, device=device, dtype=dtype)
    raise ValueError(
        f"loss.{key} must be a scalar, list, or dict; got {type(raw).__name__}."
    )


def _parc_input_scale(
    cfg: dict,
    *,
    kind: str,
    Fdim: int,
    selected_indices: list[int],
    device,
    dtype,
) -> torch.Tensor:
    loss = cfg.get("loss", {}) or {}
    global_scale = _scale_tensor_from_cfg_value(
        loss.get("parc_input_scale", None),
        cfg=cfg,
        Fdim=Fdim,
        selected_indices=selected_indices,
        device=device,
        dtype=dtype,
        key="parc_input_scale",
    )
    explicit_key = f"parc_input_scale_{kind}"
    auto_key = f"parc_input_auto_scale_{kind}"
    explicit = _scale_tensor_from_cfg_value(
        loss.get(explicit_key, None),
        cfg=cfg,
        Fdim=Fdim,
        selected_indices=selected_indices,
        device=device,
        dtype=dtype,
        key=explicit_key,
    )
    auto = _scale_tensor_from_cfg_value(
        loss.get(auto_key, None),
        cfg=cfg,
        Fdim=Fdim,
        selected_indices=selected_indices,
        device=device,
        dtype=dtype,
        key=auto_key,
    )
    return global_scale * explicit * auto


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
    apply_input_scales: bool = True,
):
    """
    Builds the operator-derived node feature block to concatenate onto _build_X(...).

    Config keys (under cfg["loss"]):
      parc_input_form: "rate" (default) or "delta"
      parc_input_time_scale: "model" (legacy), "unit", "dt_ref", or "dt_phys"
      parc_input_scale / parc_input_scale_adv / parc_input_scale_diff
      parc_include_adv: bool (default True)
      parc_include_diff: bool (default True)
      parc_input_weighted: bool (default False)
      parc_detach_inputs: bool (default True)

      --- channel selection ---
      parc_input_channels_adv / parc_input_channels_diff (list[str])  # inputs-only
      adv_channels / diff_channels (list[str])                        # ops + inputs (default fallback)
      channels (list[str])                                            # legacy fallback

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

    # Channel subsets for adv/diff inputs (can be different)
    sel_adv = parc_select_feature_indices_adv(cfg, Fdim)
    sel_diff = parc_select_feature_indices_diff(cfg, Fdim)

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
            adv_u = operator_rate_to_input_units(
                adv_abs,
                dt_phys=dt_phys,
                dt_ref=dt_ref,
                sigma=sigma,
                predict_type=predict_type,
                cfg=cfg,
            )
        if diff_abs is not None:
            diff_u = operator_rate_to_input_units(
                diff_abs,
                dt_phys=dt_phys,
                dt_ref=dt_ref,
                sigma=sigma,
                predict_type=predict_type,
                cfg=cfg,
            )

    blocks = []
    if include_adv and (adv_u is not None):
        block = adv_u[:, sel_adv]
        if apply_input_scales:
            block = block * _parc_input_scale(
                cfg,
                kind="adv",
                Fdim=Fdim,
                selected_indices=sel_adv,
                device=block.device,
                dtype=block.dtype,
            ).view(1, -1)
        blocks.append(block)
    if include_diff and (diff_u is not None):
        block = diff_u[:, sel_diff]
        if apply_input_scales:
            block = block * _parc_input_scale(
                cfg,
                kind="diff",
                Fdim=Fdim,
                selected_indices=sel_diff,
                device=block.device,
                dtype=block.dtype,
            ).view(1, -1)
        blocks.append(block)

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
    How many PARC operator-derived channels are concatenated onto X.
    With adv/diff having independent channel sets:
      extra = (include_adv ? len(sel_adv) : 0) + (include_diff ? len(sel_diff) : 0)
    """
    loss = cfg.get("loss", {}) or {}
    include_adv = bool(loss.get("parc_include_adv", True))
    include_diff = bool(loss.get("parc_include_diff", True))

    sel_adv = parc_select_feature_indices_adv(cfg, Fdim)
    sel_diff = parc_select_feature_indices_diff(cfg, Fdim)

    extra = 0
    if include_adv:
        extra += len(sel_adv)
    if include_diff:
        extra += len(sel_diff)
    return int(extra)


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

def cell_area_from_levels(
    levels: torch.Tensor,
    *,
    dx0: float,
    dy0: float,
    dtype,
    device,
    refine_ratio: int = 2,
):
    """
    axis-aligned AMR quads: A = (dx0*dy0) / ((refine_ratio^2)^L)
    """
    rr = int(refine_ratio)
    if rr < 2:
        raise ValueError(f"refine_ratio must be >=2, got {refine_ratio}")

    base_area = torch.tensor(float(dx0) * float(dy0), device=device, dtype=dtype)
    L = levels.to(device=device)
    if L.dtype not in (torch.int32, torch.int64):
        L = L.long()
    denom = torch.pow(torch.tensor(float(rr * rr), device=device, dtype=dtype), L.to(dtype=dtype))
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
    edge_dist: torch.Tensor | None,  # [E] center distance (needed for MUSCL)
    area: torch.Tensor,         # [N]
    scheme: str = "upwind",
    reconstruction: str = "first_order",
    limiter: str = "minmod",
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

    rec_kind = _normalize_reconstruction_kind(reconstruction)
    if rec_kind == "muscl":
        if edge_dist is None:
            raise RuntimeError("MUSCL reconstruction requires edge_dist in dec_divergence_advective_flux.")
        phi_L, phi_R = _muscl_reconstruct_face_states(
            phi, edge_index, nx, ny, edge_dist, limiter=limiter
        )
    else:
        phi_L = phi[src]
        phi_R = phi[dst]

    if scheme.lower() == "central":
        phi_face = 0.5 * (phi_L + phi_R)  # [E,F]
    else:
        # upwind: if un>0 (flow from src to dst), take src; else take dst
        take_src = (un >= 0).unsqueeze(1)
        phi_face = torch.where(take_src, phi_L, phi_R)  # [E,F]

    flux = (un * face_len)[:, None] * phi_face  # [E,F]
    div = phi.new_zeros((N, Fdim))
    div.index_add_(0, src, flux)

    area_safe = area.clamp_min(1e-12).unsqueeze(1)
    return div / area_safe

def compute_velocity_from_state(x_abs: torch.Tensor, cfg: dict, eps: float = 1e-8):
    """
    Derives velocity from conserved variables using u = (mx/rho, my/rho),
    with robust clipping to prevent advection overflow when rho is floored.
    """
    loss = cfg.get("loss", {}) or {}

    # floors / clips (tune as needed)
    rho_floor = float(loss.get("rho_floor", 1e-6))
    u_clip    = float(loss.get("u_clip", 1e3))   # start generous; normal |u| is ~O(10-40)

    _ = float(rho_floor)  # retained for backward-compatible config surface
    state = _state_views(x_abs, cfg, eps=eps)
    ux = state["u"]
    uy = state["v"]

    # Smooth clip (preferred) to avoid hard kinks
    if u_clip > 0:
        ux = u_clip * torch.tanh(ux / u_clip)
        uy = u_clip * torch.tanh(uy / u_clip)

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
        elif n in ("u", "ux", "x_velocity", "velocity_x", "velx"):
            mask[int(idx.get("u", idx["mx"]))] = 1.0
        elif n in ("v", "uy", "y_velocity", "velocity_y", "vely"):
            mask[int(idx.get("v", idx["my"]))] = 1.0
        elif ("press" in n) or (n == "p"):
            p_idx = idx.get("p", None)
            if p_idx is not None:
                mask[int(p_idx)] = 1.0
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
    levels: torch.Tensor | None,         # [N] or None for uniform
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
    rr = _get_refine_ratio(cfg)

    nx, ny, face_len, dual_len, tau = edge_attr_unpack(pred_ea)

    if levels is None:
        area = x_abs.new_full((x_abs.size(0),), float(dx0) * float(dy0))
    else:
        area = cell_area_from_levels(
            levels, dx0=dx0, dy0=dy0, dtype=x_abs.dtype, device=x_abs.device, refine_ratio=rr
        )  # [N]
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
            edge_dist=dual_len,
            area=area,
            scheme=scheme,
            reconstruction=str(cfg["loss"].get("advection_reconstruction", "first_order")),
            limiter=str(cfg["loss"].get("muscl_limiter", "minmod")),
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
