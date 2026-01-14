# utils/uniform_mesh_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Any

import torch
import torch.nn.functional as F


# ---------- Types (inject your project-specific hooks) ----------
BuildXFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, dict], torch.Tensor]
ForwardFn = Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]
CoarseAggFn = Callable[[torch.Tensor, torch.Tensor, int, int], torch.Tensor]
LapFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TmpFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# ---------- Small utilities ----------
def require_list(batch: dict, key: str) -> list:
    v = batch.get(key, None)
    if v is None or not isinstance(v, (list, tuple)):
        raise RuntimeError(f"Missing required list '{key}' in batch.")
    return list(v)


def _to_tensor(x, *, device=None, dtype=None) -> torch.Tensor:
    if torch.is_tensor(x):
        t = x
    else:
        t = torch.as_tensor(x)
    if device is not None:
        t = t.to(device)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t


def maybe_norm(x: torch.Tensor, mu, sigma, eps: float = 1e-12) -> torch.Tensor:
    """
    mu, sigma may be None, list, numpy, or torch. Broadcastable to x.
    """
    if (mu is None) or (sigma is None):
        return x
    mu_t = _to_tensor(mu, device=x.device, dtype=x.dtype)
    sig_t = _to_tensor(sigma, device=x.device, dtype=x.dtype)
    sig_t = torch.clamp(sig_t, min=eps)
    return (x - mu_t) / sig_t


def maybe_denorm(x: torch.Tensor, mu, sigma) -> torch.Tensor:
    if (mu is None) or (sigma is None):
        return x
    mu_t = _to_tensor(mu, device=x.device, dtype=x.dtype)
    sig_t = _to_tensor(sigma, device=x.device, dtype=x.dtype)
    return x * sig_t + mu_t


def _zero_like_scalar(x: torch.Tensor) -> torch.Tensor:
    return x.new_zeros(())


# ---------- Uniform mesh construction ----------
def build_uniform_centers_levels_edges(
    *,
    Hu: int,
    Wu: int,
    bbox: tuple[float, float, float, float] | None,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    diag: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """
    Returns:
      centers_u: [Nu, 2] (physical or index coords)
      levels_u : [Nu] long zeros
      ei_u     : [2, E] long (bidirectional grid adjacency)
      dxu, dyu : uniform cell spacings (in same coord system as centers)
    """
    if bbox is None:
        # index-space: x in [0,Wu], y in [0,Hu]
        x0, x1, y0, y1 = 0.0, float(Wu), 0.0, float(Hu)
    else:
        x0, x1, y0, y1 = map(float, bbox)

    dxu = (x1 - x0) / float(Wu)
    dyu = (y1 - y0) / float(Hu)

    ys = (torch.arange(Hu, device=device, dtype=dtype) + 0.5) * dyu + x0 * 0.0 + y0
    xs = (torch.arange(Wu, device=device, dtype=dtype) + 0.5) * dxu + y0 * 0.0 + x0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    centers_u = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # [Nu,2]

    levels_u = torch.zeros((Hu * Wu,), device=device, dtype=torch.long)

    # Vectorized grid adjacency
    grid = torch.arange(Hu * Wu, device=device, dtype=torch.long).view(Hu, Wu)

    edges_src = []
    edges_dst = []

    # right neighbors
    if Wu > 1:
        u = grid[:, :-1].reshape(-1)
        v = grid[:, 1:].reshape(-1)
        edges_src += [u, v]
        edges_dst += [v, u]

    # down neighbors
    if Hu > 1:
        u = grid[:-1, :].reshape(-1)
        v = grid[1:, :].reshape(-1)
        edges_src += [u, v]
        edges_dst += [v, u]

    if diag:
        if (Hu > 1) and (Wu > 1):
            # down-right
            u = grid[:-1, :-1].reshape(-1)
            v = grid[1:, 1:].reshape(-1)
            edges_src += [u, v]
            edges_dst += [v, u]
            # down-left
            u = grid[:-1, 1:].reshape(-1)
            v = grid[1:, :-1].reshape(-1)
            edges_src += [u, v]
            edges_dst += [v, u]

    if len(edges_src) == 0:
        ei_u = torch.empty((2, 0), device=device, dtype=torch.long)
    else:
        src = torch.cat(edges_src, dim=0)
        dst = torch.cat(edges_dst, dim=0)
        ei_u = torch.stack([src, dst], dim=0).contiguous()

    return centers_u, levels_u, ei_u, float(dxu), float(dyu)


# ---------- Mapping GT dynamic AMR -> uniform grid ----------
def gt_dyn_to_uniform_via_coarse_upsample(
    feat_dyn: torch.Tensor,          # [N_dyn, F]
    parents_dyn: torch.Tensor,       # [N_dyn] parent id in [0, H0*W0)
    *,
    H0: int,
    W0: int,
    Hu: int,
    Wu: int,
    coarse_agg_fn: CoarseAggFn,
    mask0: torch.Tensor | None = None,   # optional [H0*W0] or [H0,W0] bool/0-1
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Dynamic features -> coarse H0xW0 (via coarse_agg_fn) -> bilinear upsample -> [Hu*Wu,F].

    Returns:
      x_u: [Hu*Wu, F]
      mask_u: [Hu*Wu] float {0,1} if mask0 provided, else None
    """
    # Xc: [H0*W0, F]
    Xc = coarse_agg_fn(feat_dyn, parents_dyn, H0, W0)

    # [1, F, H0, W0]
    Xc_img = Xc.view(H0, W0, -1).permute(2, 0, 1).unsqueeze(0)

    # [1, F, Hu, Wu]
    Xu_img = F.interpolate(Xc_img, size=(Hu, Wu), mode="bilinear", align_corners=False)

    # [Hu*Wu, F]
    x_u = Xu_img.squeeze(0).permute(1, 2, 0).reshape(Hu * Wu, -1)

    mask_u = None
    if mask0 is not None:
        if mask0.ndim == 1:
            m0 = mask0.view(H0, W0)
        else:
            m0 = mask0
        m0 = m0.to(device=x_u.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,H0,W0]
        mu_img = F.interpolate(m0, size=(Hu, Wu), mode="nearest")  # [1,1,Hu,Wu]
        mask_u = mu_img.reshape(-1)  # [Hu*Wu]
        x_u = x_u * mask_u.unsqueeze(1)

    return x_u, mask_u


# ---------- Engine ----------
@dataclass
class UniformMeshEngine:
    """
    Encapsulates uniform-mesh training + evaluation.

    You inject your project-specific hooks so this file can stay independent of your main training script:
      - build_X_fn(x_feat, centers, levels, cfg) -> node features for model
      - forward_fn(model, x_in, edge_index) -> prediction tensor
      - coarse_agg_fn(feat_dyn, parents_dyn, H, W) -> coarse grid features
      - lap_fn(pred, edge_index) -> scalar (optional)
      - tmp_fn(a,b) -> scalar (optional)

    Batch requirements:
      - feat_list: list length K, each [N_dyn, F] at that time
      - parents_list: list length K, each [N_dyn] mapping dyn->coarse parent id
      - dt_list: length K-1
      - optional mask_list: list length K, each [H*W] or [H,W] to mask outside domain
      - optional t_indices: length K (absolute times) for stats by absolute time
    """

    cfg: dict
    device: torch.device
    H0: int
    W0: int
    Hu: int = 256
    Wu: int = 256
    bbox: tuple[float, float, float, float] | None = None

    build_X_fn: BuildXFn = None
    forward_fn: ForwardFn = None
    coarse_agg_fn: CoarseAggFn = None
    lap_fn: Optional[LapFn] = None
    tmp_fn: Optional[TmpFn] = None

    teacher_forcing: bool = False
    diag_edges: bool = False

    def __post_init__(self):
        if self.build_X_fn is None:
            raise ValueError("UniformMeshEngine requires build_X_fn.")
        if self.forward_fn is None:
            raise ValueError("UniformMeshEngine requires forward_fn.")
        if self.coarse_agg_fn is None:
            raise ValueError("UniformMeshEngine requires coarse_agg_fn.")

        if self.bbox is None:
            self.bbox = tuple(self.cfg.get("data", {}).get("bbox", (0.0, 1.0, 0.0, 1.0)))

        centers_u, levels_u, ei_u, dxu, dyu = build_uniform_centers_levels_edges(
            Hu=self.Hu,
            Wu=self.Wu,
            bbox=self.bbox,
            device=self.device,
            dtype=torch.float32,
            diag=self.diag_edges,
        )

        self.centers_u = centers_u
        self.levels_u = levels_u
        self.ei_u = ei_u
        self.dxu = dxu
        self.dyu = dyu
        self.cell_area = float(dxu) * float(dyu)

    # ---------- Core: compute loss + pred_abs given one step ----------
    def _step_forward_and_loss(
        self,
        model: torch.nn.Module,
        *,
        x_in_abs: torch.Tensor,      # [Nu,F] in absolute units
        x_tgt_abs: torch.Tensor,     # [Nu,F] in absolute units
        dt_val: Any,
        mu=None,
        sigma=None,
        amp_ctx=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          loss (scalar tensor),
          pred_abs ([Nu,F] absolute units)
        """
        cfg = self.cfg

        huber_delta = float(cfg["loss"].get("huber_delta", 0.05))
        lap_w = float(cfg["loss"].get("laplacian_weight", 0.0))
        tmp_w = float(cfg["loss"].get("temporal_weight", 0.0))
        use_huber = bool(cfg["loss"].get("interp_use_huber", True))
        predict_type = cfg.get("model", {}).get("predict_type", "rate")

        norm_in = maybe_norm(x_in_abs, mu, sigma)
        norm_tgt = maybe_norm(x_tgt_abs, mu, sigma)

        if amp_ctx is None:
            amp_ctx = (lambda **kw: torch.autocast("cpu", enabled=False))

        with amp_ctx():
            x_in = self.build_X_fn(norm_in, self.centers_u, self.levels_u, cfg)
            y_pred = self.forward_fn(model, x_in, self.ei_u)

            # dt_hat
            dt = dt_val
            if torch.is_tensor(dt):
                dt = dt.to(device=self.device, dtype=norm_in.dtype)
            else:
                dt = torch.tensor(float(dt), device=self.device, dtype=norm_in.dtype)

            dt_ref = None
            if "dt_ref" in cfg.get("train", {}):
                # optional: allow cfg.train.dt_ref to define dt_ref globally
                dt_ref = cfg["train"]["dt_ref"]
            # also allow batch-provided dt_ref (handled by caller if desired)
            # if dt_ref is present:
            if dt_ref is not None:
                dt_ref = torch.tensor(float(dt_ref), device=self.device, dtype=norm_in.dtype)
                dt_hat = dt / dt_ref
            else:
                dt_hat = dt

            delta_target = norm_tgt - norm_in

            if predict_type == "rate":
                rate_target = delta_target / dt_hat
                center_loss = (F.huber_loss(y_pred, rate_target, delta=huber_delta)
                               if use_huber else F.l1_loss(y_pred, rate_target))
                pred_abs = maybe_denorm(norm_in + y_pred * dt_hat, mu, sigma)
            elif predict_type == "delta":
                center_loss = (F.huber_loss(y_pred, delta_target, delta=huber_delta)
                               if use_huber else F.l1_loss(y_pred, delta_target))
                pred_abs = maybe_denorm(norm_in + y_pred, mu, sigma)
            else:
                center_loss = (F.huber_loss(y_pred, norm_tgt, delta=huber_delta)
                               if use_huber else F.l1_loss(y_pred, norm_tgt))
                pred_abs = maybe_denorm(y_pred, mu, sigma)

            lap_loss = self.lap_fn(y_pred, self.ei_u) if (self.lap_fn is not None and lap_w > 0) else _zero_like_scalar(y_pred)
            tmp_loss = self.tmp_fn(norm_in, norm_tgt) if (self.tmp_fn is not None and tmp_w > 0) else _zero_like_scalar(y_pred)

            loss = center_loss + lap_w * lap_loss + tmp_w * tmp_loss

        return loss, pred_abs

    # ---------- Map whole window GT -> uniform ----------
    def map_window_gt_to_uniform(self, batch: dict) -> tuple[list[torch.Tensor], list[Optional[torch.Tensor]]]:
        feat_list = require_list(batch, "feat_list")
        parents_list = require_list(batch, "parents_list")
        mask_list = batch.get("mask_list", None)
        mask_list = list(mask_list) if isinstance(mask_list, (list, tuple)) else None

        K = len(feat_list)
        gt_u = []
        mask_u = []

        for k in range(K):
            feat_k = _to_tensor(feat_list[k], device=self.device)
            parents_k = _to_tensor(parents_list[k], device=self.device, dtype=torch.long)
            m0 = None
            if mask_list is not None:
                m0 = _to_tensor(mask_list[k], device=self.device)

            x_u, mu_u = gt_dyn_to_uniform_via_coarse_upsample(
                feat_k, parents_k,
                H0=self.H0, W0=self.W0,
                Hu=self.Hu, Wu=self.Wu,
                coarse_agg_fn=self.coarse_agg_fn,
                mask0=m0,
            )
            gt_u.append(x_u)
            mask_u.append(mu_u)

        return gt_u, mask_u

    # ---------- Training ----------
    def train_one_epoch(
        self,
        model: torch.nn.Module,
        loader,
        opt: torch.optim.Optimizer,
        *,
        mu=None,
        sigma=None,
        scaler=None,
    ):
        model.train()

        total_loss_accum = 0.0
        mae_accum = 0.0
        n_steps = 0

        speed = self.cfg.get("speed", {})
        use_amp = bool(speed.get("amp", True)) and self.device.type == "cuda"
        amp_ctx = (torch.cuda.amp.autocast if use_amp and self.device.type == "cuda"
                   else (lambda **kw: torch.autocast("cpu", enabled=False)))

        if scaler is None and use_amp and self.device.type == "cuda":
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()

        for batch in loader:
            dt_list = batch.get("dt_list", None)
            if dt_list is None:
                raise RuntimeError("Missing dt_list in batch.")
            if isinstance(dt_list, torch.Tensor):
                dt_list = list(dt_list)

            gt_u, _mask_u = self.map_window_gt_to_uniform(batch)
            K = len(gt_u)
            if K < 2:
                raise RuntimeError("window_size must be ≥ 2")

            if len(dt_list) < (K - 1):
                raise RuntimeError(f"dt_list length {len(dt_list)} is < K-1={K-1}")

            opt.zero_grad(set_to_none=True)

            window_loss_graph = None
            window_loss = 0.0
            window_mae = 0.0

            # step 0
            loss0, pred_abs = self._step_forward_and_loss(
                model,
                x_in_abs=gt_u[0],
                x_tgt_abs=gt_u[1],
                dt_val=dt_list[0],
                mu=mu,
                sigma=sigma,
                amp_ctx=amp_ctx,
            )
            window_loss += float(loss0.detach().cpu())
            window_mae += float(torch.mean(torch.abs(pred_abs.detach() - gt_u[1])).cpu())
            n_steps += 1

            window_loss_graph = loss0 if window_loss_graph is None else (window_loss_graph + loss0)

            pred_k = pred_abs  # absolute units on uniform grid

            # steps 1..K-2
            for k in range(1, K - 1):
                x_in_abs = gt_u[k] if self.teacher_forcing else pred_k
                loss_k, pred_abs = self._step_forward_and_loss(
                    model,
                    x_in_abs=x_in_abs,
                    x_tgt_abs=gt_u[k + 1],
                    dt_val=dt_list[k],
                    mu=mu,
                    sigma=sigma,
                    amp_ctx=amp_ctx,
                )
                window_loss += float(loss_k.detach().cpu())
                window_mae += float(torch.mean(torch.abs(pred_abs.detach() - gt_u[k + 1])).cpu())
                n_steps += 1

                window_loss_graph = window_loss_graph + loss_k
                pred_k = pred_abs

            if scaler is not None:
                scaler.scale(window_loss_graph).backward()
                scaler.step(opt)
                scaler.update()
            else:
                window_loss_graph.backward()
                opt.step()

            total_loss_accum += window_loss
            mae_accum += window_mae

        denom = max(n_steps, 1)
        return total_loss_accum / denom, mae_accum / denom, {
            "num_windows": len(loader),
            "num_steps": n_steps,
            "mesh_type": "uniform",
            "uniform_H": int(self.Hu),
            "uniform_W": int(self.Wu),
            "teacher_forcing": bool(self.teacher_forcing),
        }

    # ---------- Evaluation ----------
    def evaluate_one_epoch(
        self,
        model: torch.nn.Module,
        loader,
        *,
        mu=None,
        sigma=None,
        collect_examples: bool = False,
    ):
        model.eval()

        speed = self.cfg.get("speed", {})
        use_amp = bool(speed.get("amp", True)) and self.device.type == "cuda"
        amp_ctx = (torch.cuda.amp.autocast if use_amp and self.device.type == "cuda"
                   else (lambda **kw: torch.autocast("cpu", enabled=False)))

        # step-based accumulators
        step_wsum: list[float] = []
        step_mae_num: list[torch.Tensor] = []
        step_mse_num: list[torch.Tensor] = []
        step_gt2_num: list[torch.Tensor] = []

        # by absolute time (optional)
        by_t: dict[int, dict[str, Any]] = {}

        total_loss_accum = 0.0
        n_steps_total = 0

        examples = [] if collect_examples else None

        def ensure_step_capacity(k: int, Fdim: int):
            while len(step_wsum) <= k:
                step_wsum.append(0.0)
                step_mae_num.append(torch.zeros(Fdim, dtype=torch.float64))
                step_mse_num.append(torch.zeros(Fdim, dtype=torch.float64))
                step_gt2_num.append(torch.zeros(Fdim, dtype=torch.float64))

        def accumulate_metrics(
            *,
            k: int,
            t_abs: int | None,
            pred_abs: torch.Tensor,     # [Nu,F]
            gt_abs: torch.Tensor,       # [Nu,F]
            mask_u: Optional[torch.Tensor],  # [Nu] 0/1 or None
        ):
            if pred_abs.ndim == 1:
                pred_ = pred_abs[:, None]
            else:
                pred_ = pred_abs
            if gt_abs.ndim == 1:
                gt_ = gt_abs[:, None]
            else:
                gt_ = gt_abs

            if pred_.shape != gt_.shape:
                raise RuntimeError(f"Metric mismatch: pred {pred_.shape} vs gt {gt_.shape}")

            Nu, Fdim = pred_.shape
            ensure_step_capacity(k, Fdim)

            # uniform cell area weights (+ optional mask)
            w = torch.full((Nu,), self.cell_area, device=pred_.device, dtype=pred_.dtype)
            if mask_u is not None:
                w = w * mask_u.to(device=pred_.device, dtype=pred_.dtype)

            wsum_add = float(w.sum().detach().cpu())
            diff = pred_ - gt_

            mae_add = (w[:, None] * diff.abs()).sum(dim=0).detach().cpu().to(torch.float64)
            mse_add = (w[:, None] * diff.pow(2)).sum(dim=0).detach().cpu().to(torch.float64)
            gt2_add = (w[:, None] * gt_.pow(2)).sum(dim=0).detach().cpu().to(torch.float64)

            step_wsum[k] += wsum_add
            step_mae_num[k] += mae_add
            step_mse_num[k] += mse_add
            step_gt2_num[k] += gt2_add

            if t_abs is not None:
                rec = by_t.get(int(t_abs), None)
                if rec is None:
                    by_t[int(t_abs)] = {"wsum": wsum_add, "mae": mae_add.clone(), "mse": mse_add.clone(), "gt2": gt2_add.clone()}
                else:
                    rec["wsum"] += wsum_add
                    rec["mae"] += mae_add
                    rec["mse"] += mse_add
                    rec["gt2"] += gt2_add

        @torch.no_grad()
        def append_example_step(step_idx: int, gt_u: list[torch.Tensor], pred_u_tp1: torch.Tensor, batch: dict):
            if not collect_examples:
                return

            # Determine absolute time index (if present)
            t_indices = batch.get("t_indices", None)
            if t_indices is None:
                t_idx = int(step_idx + 1)
            else:
                if torch.is_tensor(t_indices):
                    t_idx = int(t_indices[step_idx + 1].item())
                else:
                    t_idx = int(t_indices[step_idx + 1])

            # These keys are what plot_qual_pdf expects (based on your stack trace and typical usage)
            bbox = tuple(self.bbox) if self.bbox is not None else tuple(self.cfg.get("data", {}).get("bbox", (0.0, 1.0, 0.0, 1.0)))

            # "parents" are not meaningful for a uniform mesh; include a dummy tensor to satisfy any downstream access
            pred_parents_dummy = torch.zeros((self.Hu * self.Wu,), dtype=torch.long, device=self.centers_u.device)

            examples.append({
                # ---- required by plot_qual_pdf ----
                "H": int(self.Hu),
                "W": int(self.Wu),
                "bbox": bbox,
                "t": int(t_idx),

                # ---- predicted mesh geometry (uniform) ----
                "pred_centers": self.centers_u.detach().cpu(),    # [Nu,2]
                "pred_levels":  self.levels_u.detach().cpu(),     # [Nu]
                "pred_parents": pred_parents_dummy.detach().cpu(),# [Nu]

                # ---- fields (already on the same uniform mesh) ----
                # These names match your predicted-mesh examples: gt_t, gt_tp1, pred_tp1
                "gt_t":     gt_u[step_idx].detach().cpu(),        # [Nu,F]
                "gt_tp1":   gt_u[step_idx + 1].detach().cpu(),    # [Nu,F]
                "pred_tp1": pred_u_tp1.detach().cpu(),            # [Nu,F]

                "centers_t":   self.centers_u.detach().cpu(),
                "centers_tp1": self.centers_u.detach().cpu(),
                "level_t":     self.levels_u.detach().cpu(),
                "level_tp1":   self.levels_u.detach().cpu(),
            })

        with torch.no_grad():
            for batch in loader:
                dt_list = batch.get("dt_list", None)
                if dt_list is None:
                    raise RuntimeError("Missing dt_list in batch.")
                if isinstance(dt_list, torch.Tensor):
                    dt_list = list(dt_list)

                gt_u, mask_u_list = self.map_window_gt_to_uniform(batch)
                K = len(gt_u)
                if K < 2:
                    raise RuntimeError("window_size must be ≥ 2")
                if len(dt_list) < (K - 1):
                    raise RuntimeError(f"dt_list length {len(dt_list)} is < K-1={K-1}")

                t_indices = batch.get("t_indices", None)

                # step 0
                loss0, pred_abs = self._step_forward_and_loss(
                    model,
                    x_in_abs=gt_u[0],
                    x_tgt_abs=gt_u[1],
                    dt_val=dt_list[0],
                    mu=mu,
                    sigma=sigma,
                    amp_ctx=amp_ctx,
                )

                total_loss_accum += float(loss0.detach().cpu())
                n_steps_total += 1

                t_abs0 = None
                if t_indices is not None:
                    t_abs0 = int(t_indices[1].item()) if torch.is_tensor(t_indices) else int(t_indices[1])

                accumulate_metrics(
                    k=0,
                    t_abs=t_abs0,
                    pred_abs=pred_abs,
                    gt_abs=gt_u[1],
                    mask_u=mask_u_list[1],
                )
                append_example_step(0, gt_u, pred_abs, batch)

                pred_k = pred_abs

                # steps 1..K-2
                for k in range(1, K - 1):
                    x_in_abs = gt_u[k] if self.teacher_forcing else pred_k
                    loss_k, pred_abs = self._step_forward_and_loss(
                        model,
                        x_in_abs=x_in_abs,
                        x_tgt_abs=gt_u[k + 1],
                        dt_val=dt_list[k],
                        mu=mu,
                        sigma=sigma,
                        amp_ctx=amp_ctx,
                    )

                    total_loss_accum += float(loss_k.detach().cpu())
                    n_steps_total += 1

                    t_absk = None
                    if t_indices is not None:
                        t_absk = int(t_indices[k + 1].item()) if torch.is_tensor(t_indices) else int(t_indices[k + 1])

                    accumulate_metrics(
                        k=k,
                        t_abs=t_absk,
                        pred_abs=pred_abs,
                        gt_abs=gt_u[k + 1],
                        mask_u=mask_u_list[k + 1],
                    )
                    append_example_step(k, gt_u, pred_abs, batch)

                    pred_k = pred_abs

        # finalize metrics
        eps = 1e-12
        S = len(step_wsum)
        if S == 0:
            raise RuntimeError("No steps accumulated in evaluation.")

        Fdim = int(step_mae_num[0].numel())
        maew_feat_by_step = torch.zeros((S, Fdim), dtype=torch.float64)
        rell2w_feat_by_step = torch.zeros((S, Fdim), dtype=torch.float64)
        maew_by_step = []
        rell2w_by_step = []

        for k in range(S):
            wsum = step_wsum[k]
            if wsum <= 0:
                maew_feat = torch.full((Fdim,), float("nan"), dtype=torch.float64)
                rell2w_feat = torch.full((Fdim,), float("nan"), dtype=torch.float64)
            else:
                maew_feat = step_mae_num[k] / wsum
                rell2w_feat = torch.sqrt(step_mse_num[k] / (step_gt2_num[k] + eps))

            maew_feat_by_step[k] = maew_feat
            rell2w_feat_by_step[k] = rell2w_feat
            maew_by_step.append(float(maew_feat.mean().item()))
            rell2w_by_step.append(float(rell2w_feat.mean().item()))

        t_values = sorted(by_t.keys())
        if len(t_values) > 0:
            maew_feat_by_t = torch.zeros((len(t_values), Fdim), dtype=torch.float64)
            rell2w_feat_by_t = torch.zeros((len(t_values), Fdim), dtype=torch.float64)
            maew_by_t = []
            rell2w_by_t = []

            for i, t_abs in enumerate(t_values):
                rec = by_t[t_abs]
                wsum = rec["wsum"]
                if wsum <= 0:
                    maew_feat = torch.full((Fdim,), float("nan"), dtype=torch.float64)
                    rell2w_feat = torch.full((Fdim,), float("nan"), dtype=torch.float64)
                else:
                    maew_feat = rec["mae"] / wsum
                    rell2w_feat = torch.sqrt(rec["mse"] / (rec["gt2"] + eps))

                maew_feat_by_t[i] = maew_feat
                rell2w_feat_by_t[i] = rell2w_feat
                maew_by_t.append(float(maew_feat.mean().item()))
                rell2w_by_t.append(float(rell2w_feat.mean().item()))
        else:
            maew_feat_by_t = None
            rell2w_feat_by_t = None
            maew_by_t = None
            rell2w_by_t = None

        avg_loss = total_loss_accum / max(n_steps_total, 1)

        stats = {
            "num_windows": len(loader),
            "num_steps": n_steps_total,
            "mesh_type": "uniform",
            "uniform_H": int(self.Hu),
            "uniform_W": int(self.Wu),
            "teacher_forcing": bool(self.teacher_forcing),

            "maew_by_rollout_step": maew_by_step,
            "rell2w_by_rollout_step": rell2w_by_step,
            "maew_feat_by_rollout_step": maew_feat_by_step,         # [S,F]
            "rell2w_feat_by_rollout_step": rell2w_feat_by_step,     # [S,F]

            "t_values": t_values,
            "maew_by_t": maew_by_t,
            "rell2w_by_t": rell2w_by_t,
            "maew_feat_by_t": maew_feat_by_t,
            "rell2w_feat_by_t": rell2w_feat_by_t,
        }

        if collect_examples:
            stats["examples"] = examples

        return avg_loss, stats
