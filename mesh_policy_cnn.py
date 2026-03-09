from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpFuseBlock(nn.Module):
    """
    Bilinear upsample + skip fusion block for decoder stages.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
            ConvBlock(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class MeshPolicyCNN(nn.Module):
    """
    Multi-head CNN for refine-only AMR policy.

    Inputs are defined on the coarse parent grid (H x W). For each level L in
    [1..max_level], the model predicts a binary refinement logit map on the
    parent grid of level (L-1), i.e. shape:
      H_L = H * refine_ratio ** (L-1)
      W_L = W * refine_ratio ** (L-1)
    """

    def __init__(
        self,
        in_channels: int,
        *,
        base_channels: int = 48,
        head_channels: int = 8,
        max_level: int = 3,
        refine_ratio: int = 4,
        model_type: str = "upsample_heads",
    ):
        super().__init__()
        if max_level < 1:
            raise ValueError(f"max_level must be >= 1, got {max_level}")
        if refine_ratio < 2:
            raise ValueError(f"refine_ratio must be >= 2, got {refine_ratio}")

        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.head_channels = int(head_channels)
        self.max_level = int(max_level)
        self.refine_ratio = int(refine_ratio)
        self.model_type = str(model_type).strip().lower()
        if self.model_type not in ("upsample_heads", "unet_hier"):
            raise ValueError(
                f"model_type must be one of ['upsample_heads', 'unet_hier'], got {model_type!r}"
            )

        C = self.base_channels
        HC = self.head_channels
        if self.model_type == "upsample_heads":
            # Legacy architecture: single coarse feature map + per-level upsampled heads.
            self.stem = nn.Sequential(
                nn.Conv2d(self.in_channels, C, kernel_size=3, padding=1),
                nn.GELU(),
                ConvBlock(C, C),
                ConvBlock(C, C),
            )
            self.head_proj = nn.Conv2d(C, HC, kernel_size=1)

            self.heads = nn.ModuleDict()
            for L in range(1, self.max_level + 1):
                self.heads[str(L)] = nn.Sequential(
                    nn.Conv2d(HC, HC, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(HC, 1, kernel_size=1),
                )
        else:
            # Hierarchical U-Net style backbone with level-conditional heads.
            self.stem = nn.Sequential(
                nn.Conv2d(self.in_channels, C, kernel_size=3, padding=1),
                nn.GELU(),
                ConvBlock(C, C),
            )
            self.enc1 = ConvBlock(C, C)
            self.down1 = nn.Sequential(
                nn.Conv2d(C, 2 * C, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                ConvBlock(2 * C, 2 * C),
            )
            self.down2 = nn.Sequential(
                nn.Conv2d(2 * C, 4 * C, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                ConvBlock(4 * C, 4 * C),
            )
            self.up2 = UpFuseBlock(4 * C + 2 * C, 2 * C)
            self.up1 = UpFuseBlock(2 * C + C, C)

            self.head_proj = nn.Conv2d(C, HC, kernel_size=1)
            self.level_pre = nn.ModuleDict()
            self.level_out = nn.ModuleDict()
            for L in range(1, self.max_level + 1):
                in_ch = HC if L == 1 else (HC + 1)
                self.level_pre[str(L)] = nn.Sequential(
                    nn.Conv2d(in_ch, HC, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(HC, HC, kernel_size=3, padding=1),
                    nn.GELU(),
                )
                self.level_out[str(L)] = nn.Conv2d(HC, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        x: (B, C_in, H, W)
        returns:
            logits_by_level[L] -> (B, 1, H*rr**(L-1), W*rr**(L-1))
        """
        if x.ndim != 4:
            raise ValueError(f"expected x to be 4D (B,C,H,W), got {tuple(x.shape)}")

        if self.model_type == "upsample_heads":
            feat0 = self.stem(x)
            head_base = self.head_proj(feat0)
            out: Dict[int, torch.Tensor] = {}
            for L in range(1, self.max_level + 1):
                scale = self.refine_ratio ** (L - 1)
                if scale == 1:
                    featL = head_base
                else:
                    featL = F.interpolate(
                        head_base,
                        scale_factor=float(scale),
                        mode="bilinear",
                        align_corners=False,
                    )
                out[L] = self.heads[str(L)](featL)
            return out

        # Hierarchical U-Net + level-conditional refinement heads.
        B, _C, H, W = x.shape
        e0 = self.stem(x)      # (B,C,H,W)
        e1 = self.enc1(e0)     # (B,C,H,W)
        d1 = self.down1(e1)    # (B,2C,H/2,W/2)
        d2 = self.down2(d1)    # (B,4C,H/4,W/4)
        u1 = self.up2(d2, d1)  # (B,2C,H/2,W/2)
        u0 = self.up1(u1, e1)  # (B,C,H,W)

        feat_base = self.head_proj(u0)  # (B,HC,H,W)
        out: Dict[int, torch.Tensor] = {}
        prev_feat: torch.Tensor | None = None
        prev_logit: torch.Tensor | None = None

        for L in range(1, self.max_level + 1):
            tgt_h = int(H * (self.refine_ratio ** (L - 1)))
            tgt_w = int(W * (self.refine_ratio ** (L - 1)))
            if L == 1:
                featL = feat_base
            else:
                assert prev_feat is not None
                featL = F.interpolate(
                    prev_feat,
                    size=(tgt_h, tgt_w),
                    mode="bilinear",
                    align_corners=False,
                )

            if prev_logit is None:
                head_in = featL
            else:
                cond = F.interpolate(
                    prev_logit,
                    size=(tgt_h, tgt_w),
                    mode="bilinear",
                    align_corners=False,
                )
                head_in = torch.cat([featL, cond], dim=1)

            hL = self.level_pre[str(L)](head_in)
            logitL = self.level_out[str(L)](hL)
            out[L] = logitL
            prev_feat = hL
            prev_logit = logitL

        return out


@dataclass
class MeshPolicyPostprocess:
    refine_ratio: int = 4
    threshold_by_level: Optional[Dict[int, float]] = None

    def _thr(self, L: int) -> float:
        if isinstance(self.threshold_by_level, dict) and L in self.threshold_by_level:
            return float(self.threshold_by_level[L])
        return 0.5

    @torch.no_grad()
    def logits_to_hierarchical_masks(
        self,
        logits_by_level: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """
        Convert per-level logits to refine-only masks with parent-child gating.
        Output masks have dtype bool and shape (B, H_L, W_L).
        """
        rr = int(self.refine_ratio)
        levels = sorted(int(k) for k in logits_by_level.keys())
        masks: Dict[int, torch.Tensor] = {}
        for L in levels:
            logits = logits_by_level[L]
            if logits.ndim != 4 or logits.size(1) != 1:
                raise ValueError(
                    f"logits_by_level[{L}] must have shape (B,1,H,W), got {tuple(logits.shape)}"
                )
            p = torch.sigmoid(logits)
            m = p >= self._thr(L)
            m = m[:, 0]  # (B,H,W)
            if L > 1:
                allow = F.interpolate(
                    masks[L - 1].float().unsqueeze(1),
                    scale_factor=float(rr),
                    mode="nearest",
                )[:, 0].bool()
                h, w = m.shape[-2], m.shape[-1]
                m = m & allow[:, :h, :w]
            masks[L] = m
        return masks


def level_weights_from_iterable(
    levels: Iterable[int],
    *,
    mode: str = "equal",
) -> Dict[int, float]:
    """
    Build scalar loss weights per level.

    mode:
      - "equal": all levels weighted 1.0
      - "coarse_priority": weight higher on coarse levels
      - "fine_priority": weight higher on fine levels
    """
    ls = sorted(int(L) for L in levels)
    if not ls:
        return {}
    if mode == "equal":
        return {L: 1.0 for L in ls}
    if mode == "coarse_priority":
        top = float(max(ls))
        return {L: (top - float(L) + 1.0) for L in ls}
    if mode == "fine_priority":
        low = float(min(ls))
        return {L: (float(L) - low + 1.0) for L in ls}
    raise ValueError(f"unknown mode: {mode}")
