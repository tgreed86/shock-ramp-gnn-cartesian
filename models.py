import torch
from torch import nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GATConv, GraphUNet
from torch_geometric.nn import SAGEConv, GINEConv, NNConv
from torch_geometric.utils import softmax as pyg_softmax
try:
    from torch_geometric.utils import scatter as pyg_scatter
except Exception:
    from torch_scatter import scatter as pyg_scatter
import inspect
from typing import Any, Dict


PREDICT_TYPE_STATE = "state"
PREDICT_TYPE_DELTA = "delta"
PREDICT_TYPE_RATE = "rate"
PREDICT_TYPES = {PREDICT_TYPE_STATE, PREDICT_TYPE_DELTA, PREDICT_TYPE_RATE}


def _make_activation(
    name: str,
    *,
    negative_slope: float = 0.01,
    elu_alpha: float = 1.0,
) -> nn.Module:
    key = str(name).strip().lower()
    if key in {"relu"}:
        return nn.ReLU()
    if key in {"leaky_relu", "leakyrelu", "lrelu", "leaky"}:
        return nn.LeakyReLU(negative_slope=float(negative_slope))
    if key in {"silu", "swish"}:
        return nn.SiLU()
    if key in {"gelu"}:
        return nn.GELU()
    if key in {"elu"}:
        return nn.ELU(alpha=float(elu_alpha))
    if key in {"none", "identity", "linear"}:
        return nn.Identity()
    raise ValueError(
        f"Unsupported activation '{name}'. "
        "Use one of: relu, leaky_relu, silu, gelu, elu, identity."
    )


def _make_mlp(
    *,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    activation: str,
    activation_negative_slope: float,
    activation_elu_alpha: float,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(int(in_dim), int(hidden_dim)),
        _make_activation(
            activation,
            negative_slope=float(activation_negative_slope),
            elu_alpha=float(activation_elu_alpha),
        ),
        nn.Dropout(p=float(dropout)),
        nn.Linear(int(hidden_dim), int(out_dim)),
    )


class LocalEdgeAttention(nn.Module):
    """
    Local (edge-index constrained) multi-head attention with optional edge-feature bias.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        heads: int = 4,
        dropout: float = 0.0,
        edge_dim: int | None = 5,
        use_edge_attr: bool = True,
        concat: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.concat = bool(concat)
        self.use_edge_attr = bool(use_edge_attr)

        if self.heads <= 0:
            raise ValueError(f"LocalEdgeAttention heads must be > 0, got {heads}.")
        if self.out_channels <= 0:
            raise ValueError(f"LocalEdgeAttention out_channels must be > 0, got {out_channels}.")

        hdim = self.heads * self.out_channels
        self.q_proj = nn.Linear(self.in_channels, hdim, bias=False)
        self.k_proj = nn.Linear(self.in_channels, hdim, bias=False)
        self.v_proj = nn.Linear(self.in_channels, hdim, bias=False)

        if self.use_edge_attr:
            if edge_dim is None:
                # Infer at first forward if caller does not want to hard-code edge dim.
                self.edge_proj = nn.LazyLinear(self.heads, bias=False)
            else:
                self.edge_proj = nn.Linear(int(edge_dim), self.heads, bias=False)
        else:
            self.edge_proj = None

        out_in = hdim if self.concat else self.out_channels
        self.out_proj = nn.Linear(out_in, self.out_channels, bias=True)
        self.score_scale = 1.0 / math.sqrt(float(self.out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"LocalEdgeAttention expects x with shape [N,F], got {tuple(x.shape)}.")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"LocalEdgeAttention expects edge_index with shape [2,E], got {tuple(edge_index.shape)}."
            )

        n_nodes = int(x.size(0))
        src = edge_index[0].long()
        dst = edge_index[1].long()

        q = self.q_proj(x).view(n_nodes, self.heads, self.out_channels)
        k = self.k_proj(x).view(n_nodes, self.heads, self.out_channels)
        v = self.v_proj(x).view(n_nodes, self.heads, self.out_channels)

        score = (q[dst] * k[src]).sum(dim=-1) * self.score_scale

        if self.use_edge_attr:
            if edge_attr is None:
                raise RuntimeError(
                    "LocalEdgeAttention is configured with use_edge_attr=True but edge_attr is None."
                )
            e = edge_attr.to(device=x.device, dtype=x.dtype)
            score = score + self.edge_proj(e)

        alpha = pyg_softmax(score, dst, num_nodes=n_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = v[src] * alpha.unsqueeze(-1)  # [E,H,D]
        out = pyg_scatter(msg, dst, dim=0, dim_size=n_nodes, reduce="sum")  # [N,H,D]

        if self.concat:
            out = out.reshape(n_nodes, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        return self.out_proj(out)


class SAGEConvModel(nn.Module):
    """GraphSAGE model matching the 1D advection/Burgers SAGEConv architecture."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = 1,
        state_channel: int = 0,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
        activation_negative_slope: float = 0.01,
        activation_elu_alpha: float = 1.0,
        use_skip: bool = True,
        use_layernorm: bool = False,
        layernorm_eps: float = 1e-6,
        predict_type: str = PREDICT_TYPE_STATE,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}.")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be > 0, got {out_channels}.")
        if state_channel < 0 or (state_channel + out_channels) > in_channels:
            raise ValueError(
                "state_channel/out_channels must select a valid state slice from the "
                f"input, got state_channel={state_channel}, out_channels={out_channels}, "
                f"in_channels={in_channels}."
            )
        if hidden <= 0:
            raise ValueError(f"hidden must be > 0, got {hidden}.")
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}.")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}.")
        predict_key = str(predict_type).strip().lower()
        if predict_key == "absolute":
            predict_key = PREDICT_TYPE_STATE
        if predict_key not in PREDICT_TYPES:
            raise ValueError(
                f"predict_type must be one of {sorted(PREDICT_TYPES)}, got '{predict_type}'."
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.state_channel = int(state_channel)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.predict_type = predict_key
        self.use_skip = bool(use_skip)
        self.use_layernorm = bool(use_layernorm)
        self.block_activation = _make_activation(
            activation,
            negative_slope=float(activation_negative_slope),
            elu_alpha=float(activation_elu_alpha),
        )

        dims = [self.in_channels] + [self.hidden] * self.layers
        self.convs = nn.ModuleList()
        self.skip_proj = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(len(dims) - 1):
            in_ch = int(dims[i])
            out_ch = int(dims[i + 1])
            self.convs.append(SAGEConv(in_ch, out_ch))
            if self.use_skip:
                if in_ch == out_ch:
                    self.skip_proj.append(nn.Identity())
                else:
                    self.skip_proj.append(nn.Linear(in_ch, out_ch, bias=False))
            else:
                self.skip_proj.append(nn.Identity())
            if self.use_layernorm:
                self.norms.append(nn.LayerNorm(out_ch, eps=float(layernorm_eps)))
            else:
                self.norms.append(nn.Identity())

        self.head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            _make_activation(
                activation,
                negative_slope=float(activation_negative_slope),
                elu_alpha=float(activation_elu_alpha),
            ),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden, self.out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_attr: torch.Tensor | None = None,
        dt: float | None = None,
    ) -> torch.Tensor:
        """Run the SAGEConv stack; edge_attr and dt are accepted for API parity."""
        del edge_attr
        del dt

        state = x[:, self.state_channel : self.state_channel + self.out_channels]
        h = x
        for li, conv in enumerate(self.convs):
            h_res = h
            h = conv(h, edge_index)
            if self.use_skip:
                h = h + self.skip_proj[li](h_res)
            h = self.norms[li](h)
            h = self.block_activation(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.head(h)
        if self.predict_type == PREDICT_TYPE_STATE:
            return state + out
        if self.predict_type == PREDICT_TYPE_DELTA:
            return out
        if self.predict_type == PREDICT_TYPE_RATE:
            return out
        raise RuntimeError(f"Unexpected predict_type='{self.predict_type}'.")

    def predict_state(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_attr: torch.Tensor | None = None,
        dt: float,
        state_override: torch.Tensor | None = None,
        state_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return next-state predictions regardless of model predict_type."""
        if state_override is not None and state_residual is not None:
            raise ValueError("Pass only one of state_override or state_residual.")

        state_in = x[:, self.state_channel : self.state_channel + self.out_channels]
        state_ref = state_override if state_override is not None else state_residual
        if state_ref is None:
            state_base = state_in
        else:
            if state_ref.ndim != 2 or state_ref.shape != state_in.shape:
                raise ValueError(
                    "state_override/state_residual must have shape matching the state "
                    f"slice {tuple(state_in.shape)}, got {tuple(state_ref.shape)}."
                )
            state_base = state_ref.to(device=x.device, dtype=x.dtype)

        y = self.forward(x, edge_index, edge_attr=edge_attr, dt=dt)
        if self.predict_type == PREDICT_TYPE_STATE:
            if state_ref is None:
                return y
            return y + (state_base - state_in)
        if self.predict_type == PREDICT_TYPE_DELTA:
            return state_base + y
        if self.predict_type == PREDICT_TYPE_RATE:
            return state_base + float(dt) * y
        raise RuntimeError(f"Unexpected predict_type='{self.predict_type}'.")


class MeshGraphNetProcessorBlock(nn.Module):
    """One edge-update + node-update MeshGraphNet processor block."""

    def __init__(
        self,
        *,
        hidden: int,
        dropout: float,
        activation: str,
        activation_negative_slope: float,
        activation_elu_alpha: float,
        use_layernorm: bool,
        layernorm_eps: float,
    ) -> None:
        super().__init__()
        self.edge_mlp = _make_mlp(
            in_dim=3 * int(hidden),
            hidden_dim=int(hidden),
            out_dim=int(hidden),
            activation=activation,
            activation_negative_slope=float(activation_negative_slope),
            activation_elu_alpha=float(activation_elu_alpha),
            dropout=float(dropout),
        )
        self.node_mlp = _make_mlp(
            in_dim=2 * int(hidden),
            hidden_dim=int(hidden),
            out_dim=int(hidden),
            activation=activation,
            activation_negative_slope=float(activation_negative_slope),
            activation_elu_alpha=float(activation_elu_alpha),
            dropout=float(dropout),
        )
        if use_layernorm:
            self.edge_norm = nn.LayerNorm(int(hidden), eps=float(layernorm_eps))
            self.node_norm = nn.LayerNorm(int(hidden), eps=float(layernorm_eps))
        else:
            self.edge_norm = nn.Identity()
            self.node_norm = nn.Identity()

    def forward(
        self,
        node_latent: torch.Tensor,
        edge_index: torch.Tensor,
        edge_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src = edge_index[0].long()
        dst = edge_index[1].long()

        edge_inputs = torch.cat(
            [node_latent[src], node_latent[dst], edge_latent],
            dim=-1,
        )
        edge_latent = edge_latent + self.edge_mlp(edge_inputs)
        edge_latent = self.edge_norm(edge_latent)

        agg = torch.zeros_like(node_latent)
        agg.index_add_(0, dst, edge_latent)
        node_inputs = torch.cat([node_latent, agg], dim=-1)
        node_latent = node_latent + self.node_mlp(node_inputs)
        node_latent = self.node_norm(node_latent)
        return node_latent, edge_latent


class MeshGraphNetModel(nn.Module):
    """MeshGraphNet-style model matching the 1D advection/Burgers processor stack."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = 1,
        state_channel: int = 0,
        edge_attr_channels: int = 5,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
        activation_negative_slope: float = 0.01,
        activation_elu_alpha: float = 1.0,
        use_layernorm: bool = False,
        layernorm_eps: float = 1e-6,
        predict_type: str = PREDICT_TYPE_STATE,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}.")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be > 0, got {out_channels}.")
        if state_channel < 0 or (state_channel + out_channels) > in_channels:
            raise ValueError(
                "state_channel/out_channels must select a valid state slice from the "
                f"input, got state_channel={state_channel}, out_channels={out_channels}, "
                f"in_channels={in_channels}."
            )
        if edge_attr_channels <= 0:
            raise ValueError(f"edge_attr_channels must be > 0, got {edge_attr_channels}.")
        if hidden <= 0:
            raise ValueError(f"hidden must be > 0, got {hidden}.")
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}.")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}.")
        predict_key = str(predict_type).strip().lower()
        if predict_key == "absolute":
            predict_key = PREDICT_TYPE_STATE
        if predict_key not in PREDICT_TYPES:
            raise ValueError(
                f"predict_type must be one of {sorted(PREDICT_TYPES)}, got '{predict_type}'."
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.state_channel = int(state_channel)
        self.edge_attr_channels = int(edge_attr_channels)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.predict_type = predict_key
        self.block_activation = _make_activation(
            activation,
            negative_slope=float(activation_negative_slope),
            elu_alpha=float(activation_elu_alpha),
        )

        self.node_encoder = _make_mlp(
            in_dim=self.in_channels,
            hidden_dim=self.hidden,
            out_dim=self.hidden,
            activation=activation,
            activation_negative_slope=float(activation_negative_slope),
            activation_elu_alpha=float(activation_elu_alpha),
            dropout=self.dropout,
        )
        self.edge_encoder = _make_mlp(
            in_dim=self.edge_attr_channels,
            hidden_dim=self.hidden,
            out_dim=self.hidden,
            activation=activation,
            activation_negative_slope=float(activation_negative_slope),
            activation_elu_alpha=float(activation_elu_alpha),
            dropout=self.dropout,
        )
        self.processor = nn.ModuleList(
            [
                MeshGraphNetProcessorBlock(
                    hidden=self.hidden,
                    dropout=self.dropout,
                    activation=activation,
                    activation_negative_slope=float(activation_negative_slope),
                    activation_elu_alpha=float(activation_elu_alpha),
                    use_layernorm=bool(use_layernorm),
                    layernorm_eps=float(layernorm_eps),
                )
                for _ in range(self.layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            _make_activation(
                activation,
                negative_slope=float(activation_negative_slope),
                elu_alpha=float(activation_elu_alpha),
            ),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden, self.out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_attr: torch.Tensor | None = None,
        dt: float | None = None,
    ) -> torch.Tensor:
        del dt

        state = x[:, self.state_channel : self.state_channel + self.out_channels]
        if edge_attr is None:
            raise ValueError("MeshGraphNetModel requires edge_attr in forward/predict_state.")
        if edge_attr.ndim != 2:
            raise ValueError(f"edge_attr must be rank-2 [E,D], got shape={tuple(edge_attr.shape)}.")
        if edge_attr.shape[1] != self.edge_attr_channels:
            raise ValueError(
                "edge_attr channel mismatch: "
                f"expected {self.edge_attr_channels}, got {edge_attr.shape[1]}."
            )
        if edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError(
                "edge_attr edge count mismatch: "
                f"edge_attr has {edge_attr.shape[0]} rows, edge_index has {edge_index.shape[1]} edges."
            )

        edge_attr = edge_attr.to(device=x.device, dtype=x.dtype)
        node_latent = self.node_encoder(x)
        edge_latent = self.edge_encoder(edge_attr)
        for block in self.processor:
            node_latent, edge_latent = block(node_latent, edge_index, edge_latent)
            node_latent = self.block_activation(node_latent)
            node_latent = F.dropout(node_latent, p=self.dropout, training=self.training)

        out = self.head(node_latent)
        if self.predict_type == PREDICT_TYPE_STATE:
            return state + out
        if self.predict_type == PREDICT_TYPE_DELTA:
            return out
        if self.predict_type == PREDICT_TYPE_RATE:
            return out
        raise RuntimeError(f"Unexpected predict_type='{self.predict_type}'.")

    def predict_state(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_attr: torch.Tensor | None = None,
        dt: float,
        state_override: torch.Tensor | None = None,
        state_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return next-state predictions regardless of model predict_type."""
        if state_override is not None and state_residual is not None:
            raise ValueError("Pass only one of state_override or state_residual.")

        state_in = x[:, self.state_channel : self.state_channel + self.out_channels]
        state_ref = state_override if state_override is not None else state_residual
        if state_ref is None:
            state_base = state_in
        else:
            if state_ref.ndim != 2 or state_ref.shape != state_in.shape:
                raise ValueError(
                    "state_override/state_residual must have shape matching the state "
                    f"slice {tuple(state_in.shape)}, got {tuple(state_ref.shape)}."
                )
            state_base = state_ref.to(device=x.device, dtype=x.dtype)

        y = self.forward(x, edge_index, edge_attr=edge_attr, dt=dt)
        if self.predict_type == PREDICT_TYPE_STATE:
            if state_ref is None:
                return y
            return y + (state_base - state_in)
        if self.predict_type == PREDICT_TYPE_DELTA:
            return state_base + y
        if self.predict_type == PREDICT_TYPE_RATE:
            return state_base + float(dt) * y
        raise RuntimeError(f"Unexpected predict_type='{self.predict_type}'.")


class FluxGraphNetModel(nn.Module):
    """
    FluxGraphNet-style conservative transport model for shock-ramp primitive fields.

    Public I/O follows the project convention: normalized state/delta/rate in the
    configured feature order. Internally, primitive [U,V,RHO,P,Eint] states are
    mapped to conservative [rho,rho*u,rho*v,E_tot], updated by anti-symmetric
    learned edge fluxes, then mapped back to primitive output channels.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = 5,
        state_channel: int = 0,
        edge_attr_channels: int = 5,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
        activation_negative_slope: float = 0.01,
        activation_elu_alpha: float = 1.0,
        use_layernorm: bool = False,
        layernorm_eps: float = 1e-6,
        predict_type: str = PREDICT_TYPE_RATE,
        state_representation: str = "primitive_uvrhope",
        u_index: int = 0,
        v_index: int = 1,
        rho_index: int = 2,
        p_index: int | None = 3,
        energy_index: int = 4,
        gamma: float = 1.4,
        rho_floor: float = 1e-6,
        e_floor: float = 1e-8,
        p_floor: float = 1e-8,
        velocity_clip: float = 0.0,
        signed_edge_channels: tuple[int, ...] | list[int] | None = (0, 1),
        fallback_directed_flux: bool = False,
        use_open_boundary_source: bool = True,
        use_ramp_boundary_source: bool = True,
        open_boundary_source_channels: tuple[int, ...] | list[int] | None = (0, 1, 2, 3),
        ramp_boundary_source_channels: tuple[int, ...] | list[int] | None = (1, 2),
        boundary_distance_start_col: int | None = None,
        boundary_distance_dim: int = 4,
        boundary_width: float = 0.02,
        ramp_signed_distance_col: int | None = None,
        ramp_boundary_width: float | None = None,
        ramp_normal: tuple[float, float] | list[float] | torch.Tensor | None = None,
        ramp_pressure_source_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}.")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be > 0, got {out_channels}.")
        if state_channel < 0 or (state_channel + out_channels) > in_channels:
            raise ValueError(
                "state_channel/out_channels must select a valid state slice from the "
                f"input, got state_channel={state_channel}, out_channels={out_channels}, "
                f"in_channels={in_channels}."
            )
        if edge_attr_channels <= 0:
            raise ValueError(f"edge_attr_channels must be > 0, got {edge_attr_channels}.")
        if hidden <= 0:
            raise ValueError(f"hidden must be > 0, got {hidden}.")
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}.")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}.")
        predict_key = str(predict_type).strip().lower()
        if predict_key == "absolute":
            predict_key = PREDICT_TYPE_STATE
        if predict_key not in PREDICT_TYPES:
            raise ValueError(
                f"predict_type must be one of {sorted(PREDICT_TYPES)}, got '{predict_type}'."
            )

        rep = str(state_representation).strip().lower()
        if rep in {"primitive", "primitive_uvrhope", "primitive_uv_rho_p_e", "uvrhope"}:
            rep = "primitive_uvrhope"
        elif rep in {"conservative", "conserved"}:
            rep = "conservative"
        else:
            raise ValueError(
                "state_representation must be one of {primitive_uvrhope, conservative}, "
                f"got {state_representation!r}."
            )
        if rep == "primitive_uvrhope" and out_channels < 5:
            raise ValueError("FluxGraphNetModel primitive mode requires at least 5 output channels.")
        if rep == "conservative" and out_channels < 4:
            raise ValueError("FluxGraphNetModel conservative mode requires at least 4 output channels.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.state_channel = int(state_channel)
        self.edge_attr_channels = int(edge_attr_channels)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.predict_type = predict_key
        self.state_representation = rep

        self.u_index = int(u_index)
        self.v_index = int(v_index)
        self.rho_index = int(rho_index)
        self.p_index = None if p_index is None else int(p_index)
        self.energy_index = int(energy_index)
        self.gamma = float(gamma)
        self.rho_floor = float(rho_floor)
        self.e_floor = float(e_floor)
        self.p_floor = float(p_floor)
        self.velocity_clip = float(velocity_clip)
        self.fallback_directed_flux = bool(fallback_directed_flux)
        self.use_open_boundary_source = bool(use_open_boundary_source)
        self.use_ramp_boundary_source = bool(use_ramp_boundary_source)
        self.boundary_distance_start_col = (
            None if boundary_distance_start_col is None else int(boundary_distance_start_col)
        )
        self.boundary_distance_dim = int(boundary_distance_dim)
        self.boundary_width = float(boundary_width)
        self.ramp_signed_distance_col = (
            None if ramp_signed_distance_col is None else int(ramp_signed_distance_col)
        )
        self.ramp_boundary_width = (
            float(ramp_boundary_width)
            if ramp_boundary_width is not None
            else float(boundary_width)
        )
        self.ramp_pressure_source_weight = float(ramp_pressure_source_weight)

        signed_channels = [] if signed_edge_channels is None else [int(c) for c in signed_edge_channels]
        for ch in signed_channels:
            if ch < 0 or ch >= self.edge_attr_channels:
                raise ValueError(
                    "signed_edge_channels entries must index edge_attr columns; "
                    f"got {ch} for edge_attr_channels={self.edge_attr_channels}."
                )
        self.signed_edge_channels = tuple(signed_channels)

        self.register_buffer("norm_mu", torch.zeros(self.out_channels, dtype=torch.float32), persistent=True)
        self.register_buffer("norm_sigma", torch.ones(self.out_channels, dtype=torch.float32), persistent=True)
        self.register_buffer("_has_norm_stats", torch.tensor(False), persistent=True)
        self.register_buffer(
            "open_boundary_source_mask",
            self._make_conserved_mask(open_boundary_source_channels),
            persistent=False,
        )
        self.register_buffer(
            "ramp_boundary_source_mask",
            self._make_conserved_mask(ramp_boundary_source_channels),
            persistent=False,
        )
        if ramp_normal is None:
            normal = torch.tensor([0.0, 1.0], dtype=torch.float32)
            has_normal = False
        else:
            normal = torch.as_tensor(ramp_normal, dtype=torch.float32).view(-1)
            if normal.numel() != 2:
                raise ValueError(f"ramp_normal must have two entries, got shape={tuple(normal.shape)}.")
            nrm = torch.linalg.norm(normal).clamp_min(1e-12)
            normal = normal / nrm
            has_normal = True
        self.register_buffer("ramp_normal", normal.to(torch.float32), persistent=False)
        self.register_buffer("_has_ramp_normal", torch.tensor(bool(has_normal)), persistent=False)

        self._flux_pair_cache: dict[
            tuple[int, int, int, str],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

        self.block_activation = _make_activation(
            activation,
            negative_slope=float(activation_negative_slope),
            elu_alpha=float(activation_elu_alpha),
        )

        self.node_encoder = _make_mlp(
            in_dim=self.in_channels,
            hidden_dim=self.hidden,
            out_dim=self.hidden,
            activation=activation,
            activation_negative_slope=float(activation_negative_slope),
            activation_elu_alpha=float(activation_elu_alpha),
            dropout=self.dropout,
        )
        self.edge_encoder = _make_mlp(
            in_dim=self.edge_attr_channels,
            hidden_dim=self.hidden,
            out_dim=self.hidden,
            activation=activation,
            activation_negative_slope=float(activation_negative_slope),
            activation_elu_alpha=float(activation_elu_alpha),
            dropout=self.dropout,
        )
        self.processor = nn.ModuleList(
            [
                MeshGraphNetProcessorBlock(
                    hidden=self.hidden,
                    dropout=self.dropout,
                    activation=activation,
                    activation_negative_slope=float(activation_negative_slope),
                    activation_elu_alpha=float(activation_elu_alpha),
                    use_layernorm=bool(use_layernorm),
                    layernorm_eps=float(layernorm_eps),
                )
                for _ in range(self.layers)
            ]
        )

        self.flux_head = _make_mlp(
            in_dim=(2 * self.hidden) + self.edge_attr_channels,
            hidden_dim=self.hidden,
            out_dim=4,
            activation=activation,
            activation_negative_slope=float(activation_negative_slope),
            activation_elu_alpha=float(activation_elu_alpha),
            dropout=self.dropout,
        )
        self.open_boundary_head = (
            _make_mlp(
                in_dim=self.hidden,
                hidden_dim=self.hidden,
                out_dim=4,
                activation=activation,
                activation_negative_slope=float(activation_negative_slope),
                activation_elu_alpha=float(activation_elu_alpha),
                dropout=self.dropout,
            )
            if self.use_open_boundary_source
            else None
        )
        self.ramp_boundary_head = (
            _make_mlp(
                in_dim=self.hidden,
                hidden_dim=self.hidden,
                out_dim=4,
                activation=activation,
                activation_negative_slope=float(activation_negative_slope),
                activation_elu_alpha=float(activation_elu_alpha),
                dropout=self.dropout,
            )
            if self.use_ramp_boundary_source
            else None
        )

    @staticmethod
    def _make_conserved_mask(channels: tuple[int, ...] | list[int] | None) -> torch.Tensor:
        mask = torch.zeros(4, dtype=torch.float32)
        if channels is None:
            return mask
        for ch in channels:
            ci = int(ch)
            if ci < 0 or ci >= 4:
                raise ValueError(f"Conserved source channel indices must be in [0,3], got {ci}.")
            mask[ci] = 1.0
        return mask

    def set_normalization_stats(self, mu, sigma) -> None:
        """Install state-channel normalization stats after the trainer computes them."""
        if mu is None or sigma is None:
            self.norm_mu.zero_()
            self.norm_sigma.fill_(1.0)
            self._has_norm_stats.fill_(False)
            return

        mu_t = torch.as_tensor(mu, dtype=self.norm_mu.dtype, device=self.norm_mu.device).view(-1)
        sig_t = torch.as_tensor(sigma, dtype=self.norm_sigma.dtype, device=self.norm_sigma.device).view(-1)
        if mu_t.numel() < self.out_channels or sig_t.numel() < self.out_channels:
            raise ValueError(
                "Normalization stats are shorter than the model state dimension: "
                f"mu={mu_t.numel()}, sigma={sig_t.numel()}, out_channels={self.out_channels}."
            )
        self.norm_mu.copy_(mu_t[: self.out_channels])
        self.norm_sigma.copy_(sig_t[: self.out_channels].clamp_min(1e-12))
        self._has_norm_stats.fill_(True)

    def _state_slice(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.state_channel : self.state_channel + self.out_channels]

    def _denorm_state(self, state: torch.Tensor) -> torch.Tensor:
        if not bool(self._has_norm_stats.item()):
            return state
        mu = self.norm_mu.to(device=state.device, dtype=state.dtype).view(1, -1)
        sigma = self.norm_sigma.to(device=state.device, dtype=state.dtype).view(1, -1).clamp_min(1e-12)
        return state * sigma + mu

    def _norm_state(self, state_abs: torch.Tensor) -> torch.Tensor:
        if not bool(self._has_norm_stats.item()):
            return state_abs
        mu = self.norm_mu.to(device=state_abs.device, dtype=state_abs.dtype).view(1, -1)
        sigma = self.norm_sigma.to(device=state_abs.device, dtype=state_abs.dtype).view(1, -1).clamp_min(1e-12)
        return (state_abs - mu) / sigma

    def _norm_rate(self, rate_abs: torch.Tensor) -> torch.Tensor:
        if not bool(self._has_norm_stats.item()):
            return rate_abs
        sigma = self.norm_sigma.to(device=rate_abs.device, dtype=rate_abs.dtype).view(1, -1).clamp_min(1e-12)
        return rate_abs / sigma

    def _primitive_abs_to_conservative(self, state_abs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.state_representation == "conservative":
            rho = state_abs[:, 0].clamp_min(self.rho_floor)
            mx = state_abs[:, 1]
            my = state_abs[:, 2]
            Et = state_abs[:, 3]
            rho_safe = rho.clamp_min(self.rho_floor)
            u = mx / rho_safe
            v = my / rho_safe
            if self.velocity_clip > 0.0:
                u = u.clamp(min=-self.velocity_clip, max=self.velocity_clip)
                v = v.clamp(min=-self.velocity_clip, max=self.velocity_clip)
            kinetic = 0.5 * (mx * mx + my * my) / rho_safe
            p = (self.gamma - 1.0) * (Et - kinetic)
            e_int = (Et / rho_safe) - 0.5 * (u * u + v * v)
            q = torch.stack([rho, mx, my, Et], dim=1)
            return q, {"rho": rho, "u": u, "v": v, "p": p, "e_int": e_int}

        rho = state_abs[:, self.rho_index].clamp_min(self.rho_floor)
        u = state_abs[:, self.u_index]
        v = state_abs[:, self.v_index]
        if self.velocity_clip > 0.0:
            u = u.clamp(min=-self.velocity_clip, max=self.velocity_clip)
            v = v.clamp(min=-self.velocity_clip, max=self.velocity_clip)
        e_int = state_abs[:, self.energy_index].clamp_min(self.e_floor)
        mx = rho * u
        my = rho * v
        Et = rho * (e_int + 0.5 * (u * u + v * v))
        if self.p_index is not None and 0 <= self.p_index < state_abs.size(1):
            p = state_abs[:, self.p_index].clamp_min(self.p_floor)
        else:
            p = ((self.gamma - 1.0) * rho * e_int).clamp_min(self.p_floor)
        q = torch.stack([rho, mx, my, Et], dim=1)
        return q, {"rho": rho, "u": u, "v": v, "p": p, "e_int": e_int}

    def _conservative_abs_to_primitive(self, q_abs: torch.Tensor, template_abs: torch.Tensor) -> torch.Tensor:
        if self.state_representation == "conservative":
            out = template_abs.clone()
            out[:, :4] = q_abs
            return out

        rho = q_abs[:, 0].clamp_min(self.rho_floor)
        mx = q_abs[:, 1]
        my = q_abs[:, 2]
        Et = q_abs[:, 3]
        u = mx / rho
        v = my / rho
        if self.velocity_clip > 0.0:
            u = u.clamp(min=-self.velocity_clip, max=self.velocity_clip)
            v = v.clamp(min=-self.velocity_clip, max=self.velocity_clip)
        e_int = ((Et / rho) - 0.5 * (u * u + v * v)).clamp_min(self.e_floor)
        p = ((self.gamma - 1.0) * rho * e_int).clamp_min(self.p_floor)

        out = template_abs.clone()
        out[:, self.u_index] = u
        out[:, self.v_index] = v
        out[:, self.rho_index] = rho
        out[:, self.energy_index] = e_int
        if self.p_index is not None and 0 <= self.p_index < out.size(1):
            out[:, self.p_index] = p
        return out

    def _conservative_rate_to_primitive_rate(
        self,
        qdot: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.state_representation == "conservative":
            out = qdot.new_zeros((qdot.size(0), self.out_channels))
            out[:, :4] = qdot
            return out

        rho = state["rho"].clamp_min(self.rho_floor)
        u = state["u"]
        v = state["v"]
        e_int = state["e_int"].clamp_min(self.e_floor)
        rho_dot = qdot[:, 0]
        mx_dot = qdot[:, 1]
        my_dot = qdot[:, 2]
        Et_dot = qdot[:, 3]

        u_dot = (mx_dot - u * rho_dot) / rho
        v_dot = (my_dot - v * rho_dot) / rho
        kinetic = 0.5 * (u * u + v * v)
        e_dot = (Et_dot - rho_dot * (e_int + kinetic) - rho * (u * u_dot + v * v_dot)) / rho
        p_dot = (self.gamma - 1.0) * (rho_dot * e_int + rho * e_dot)

        out = qdot.new_zeros((qdot.size(0), self.out_channels))
        out[:, self.u_index] = u_dot
        out[:, self.v_index] = v_dot
        out[:, self.rho_index] = rho_dot
        out[:, self.energy_index] = e_dot
        if self.p_index is not None and 0 <= self.p_index < out.size(1):
            out[:, self.p_index] = p_dot
        return out

    def _get_flux_pairing(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_key = (
            int(num_nodes),
            int(edge_index.shape[1]),
            int(edge_index.data_ptr()),
            str(edge_index.device),
        )
        cached = self._flux_pair_cache.get(cache_key, None)
        if cached is not None:
            return cached

        src = edge_index[0].long()
        dst = edge_index[1].long()
        if src.numel() == 0:
            empty = src.new_empty((0,))
            signs = torch.empty((0,), device=edge_index.device, dtype=torch.float32)
            cached = (empty, empty, empty, signs, empty)
            self._flux_pair_cache[cache_key] = cached
            return cached

        lo = torch.minimum(src, dst)
        hi = torch.maximum(src, dst)
        pair_key = lo * int(num_nodes) + hi
        _unique_keys, inverse, counts = torch.unique(
            pair_key,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        sort_perm = torch.argsort(pair_key)
        group_start = torch.cumsum(counts, dim=0) - counts
        rep_idx = sort_perm[group_start]

        orient = torch.where(
            src == lo,
            torch.ones_like(src, dtype=torch.float32),
            -torch.ones_like(src, dtype=torch.float32),
        )
        orient_sum = torch.zeros((int(counts.numel()),), device=edge_index.device, dtype=torch.float32)
        orient_sum.index_add_(0, inverse, orient)
        paired_groups = (counts == 2) & (lo[rep_idx] != hi[rep_idx]) & (orient_sum.abs() <= 1e-6)
        paired_edge_mask = paired_groups[inverse]

        rep_idx_paired = rep_idx[paired_groups]
        lo_paired = lo[rep_idx_paired]
        hi_paired = hi[rep_idx_paired]
        canonical_sign = torch.where(
            src[rep_idx_paired] == lo_paired,
            torch.ones_like(rep_idx_paired, dtype=torch.float32),
            -torch.ones_like(rep_idx_paired, dtype=torch.float32),
        )
        fallback_idx = (~paired_edge_mask).nonzero(as_tuple=False).view(-1)
        cached = (rep_idx_paired, lo_paired, hi_paired, canonical_sign, fallback_idx)
        self._flux_pair_cache[cache_key] = cached
        return cached

    def _canonical_edge_features(
        self,
        edge_attr: torch.Tensor,
        rep_idx: torch.Tensor,
        canonical_sign: torch.Tensor,
    ) -> torch.Tensor:
        edge_feat = edge_attr[rep_idx].abs()
        if len(self.signed_edge_channels) == 0 or rep_idx.numel() == 0:
            return edge_feat
        signed = edge_attr[rep_idx]
        s = canonical_sign.to(device=edge_attr.device, dtype=edge_attr.dtype).view(-1)
        edge_feat = edge_feat.clone()
        for ch in self.signed_edge_channels:
            edge_feat[:, int(ch)] = signed[:, int(ch)] * s
        return edge_feat

    def _compute_conservative_flux_update(
        self,
        *,
        node_latent: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = int(node_latent.size(0))
        src = edge_index[0].long()
        dst = edge_index[1].long()
        rep_idx, lo, hi, canonical_sign, fallback_idx = self._get_flux_pairing(edge_index, num_nodes)

        delta = torch.zeros((num_nodes, 4), device=node_latent.device, dtype=node_latent.dtype)

        if rep_idx.numel() > 0:
            h_lo = node_latent[lo]
            h_hi = node_latent[hi]
            pair_mean = 0.5 * (h_lo + h_hi)
            pair_absdiff = torch.abs(h_lo - h_hi)
            edge_feat = self._canonical_edge_features(edge_attr, rep_idx, canonical_sign)
            flux_inputs = torch.cat([pair_mean, pair_absdiff, edge_feat], dim=-1)
            paired_flux = self.flux_head(flux_inputs)
            delta.index_add_(0, lo, -paired_flux)
            delta.index_add_(0, hi, paired_flux)

        if self.fallback_directed_flux and fallback_idx.numel() > 0:
            h_src = node_latent[src[fallback_idx]]
            h_dst = node_latent[dst[fallback_idx]]
            pair_mean = 0.5 * (h_src + h_dst)
            pair_absdiff = torch.abs(h_src - h_dst)
            edge_feat = edge_attr[fallback_idx]
            flux_inputs = torch.cat([pair_mean, pair_absdiff, edge_feat], dim=-1)
            fallback_flux = self.flux_head(flux_inputs)
            delta.index_add_(0, src[fallback_idx], -fallback_flux)

        return delta

    def _linear_gate_from_distance(self, dist: torch.Tensor, width: float) -> torch.Tensor:
        width_t = max(float(width), 1e-12)
        return (1.0 - (dist.abs() / width_t)).clamp(min=0.0, max=1.0)

    def _boundary_gates(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = int(x.size(0))
        boundary_gate = x.new_zeros((n, 1))
        ramp_gate = x.new_zeros((n, 1))

        if self.boundary_distance_start_col is not None:
            start = int(self.boundary_distance_start_col)
            end = start + max(1, int(self.boundary_distance_dim))
            if 0 <= start < x.size(1) and end <= x.size(1):
                d = x[:, start:end]
                min_dist = d.abs().min(dim=1, keepdim=True).values
                boundary_gate = self._linear_gate_from_distance(min_dist, self.boundary_width)
                if d.size(1) >= 3:
                    bottom_dist = d[:, 2:3]
                    ramp_gate = self._linear_gate_from_distance(bottom_dist, self.ramp_boundary_width)

        if self.ramp_signed_distance_col is not None:
            col = int(self.ramp_signed_distance_col)
            if 0 <= col < x.size(1):
                signed = x[:, col : col + 1]
                ramp_gate = torch.maximum(
                    ramp_gate,
                    self._linear_gate_from_distance(signed, self.ramp_boundary_width),
                )

        open_gate = (boundary_gate * (1.0 - ramp_gate)).clamp(min=0.0, max=1.0)
        return open_gate, ramp_gate.clamp(min=0.0, max=1.0)

    def _boundary_source(
        self,
        *,
        x: torch.Tensor,
        node_latent: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        source = torch.zeros((x.size(0), 4), device=x.device, dtype=node_latent.dtype)
        open_gate, ramp_gate = self._boundary_gates(x)
        open_gate = open_gate.to(device=node_latent.device, dtype=node_latent.dtype)
        ramp_gate = ramp_gate.to(device=node_latent.device, dtype=node_latent.dtype)

        if self.open_boundary_head is not None and torch.any(open_gate > 0):
            mask = self.open_boundary_source_mask.to(device=node_latent.device, dtype=node_latent.dtype).view(1, 4)
            source = source + open_gate * self.open_boundary_head(node_latent) * mask

        if self.ramp_boundary_head is not None and torch.any(ramp_gate > 0):
            mask = self.ramp_boundary_source_mask.to(device=node_latent.device, dtype=node_latent.dtype).view(1, 4)
            source = source + ramp_gate * self.ramp_boundary_head(node_latent) * mask

        if (
            self.ramp_pressure_source_weight != 0.0
            and bool(self._has_ramp_normal.item())
            and torch.any(ramp_gate > 0)
        ):
            normal = self.ramp_normal.to(device=node_latent.device, dtype=node_latent.dtype).view(1, 2)
            p = state["p"].to(device=node_latent.device, dtype=node_latent.dtype).clamp_min(self.p_floor)
            wall = node_latent.new_zeros((x.size(0), 4))
            wall[:, 1] = p * normal[:, 0]
            wall[:, 2] = p * normal[:, 1]
            source = source + float(self.ramp_pressure_source_weight) * ramp_gate * wall

        return source

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_attr: torch.Tensor | None = None,
        dt: float | None = None,
    ) -> torch.Tensor:
        del dt

        state_norm = self._state_slice(x)
        state_abs = self._denorm_state(state_norm)
        q_abs, state = self._primitive_abs_to_conservative(state_abs)

        if edge_attr is None:
            raise ValueError("FluxGraphNetModel requires edge_attr in forward/predict_state.")
        if edge_attr.ndim != 2:
            raise ValueError(f"edge_attr must be rank-2 [E,D], got shape={tuple(edge_attr.shape)}.")
        if edge_attr.shape[1] != self.edge_attr_channels:
            raise ValueError(
                "edge_attr channel mismatch: "
                f"expected {self.edge_attr_channels}, got {edge_attr.shape[1]}."
            )
        if edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError(
                "edge_attr edge count mismatch: "
                f"edge_attr has {edge_attr.shape[0]} rows, edge_index has {edge_index.shape[1]} edges."
            )

        edge_attr = edge_attr.to(device=x.device, dtype=x.dtype)
        node_latent = self.node_encoder(x)
        edge_latent = self.edge_encoder(edge_attr)
        for block in self.processor:
            node_latent, edge_latent = block(node_latent, edge_index, edge_latent)
            node_latent = self.block_activation(node_latent)
            node_latent = F.dropout(node_latent, p=self.dropout, training=self.training)

        q_update = self._compute_conservative_flux_update(
            node_latent=node_latent,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        q_update = q_update + self._boundary_source(x=x, node_latent=node_latent, state=state)

        if self.predict_type == PREDICT_TYPE_RATE:
            prim_rate_abs = self._conservative_rate_to_primitive_rate(q_update, state)
            return self._norm_rate(prim_rate_abs)

        q_next_abs = q_abs + q_update
        prim_next_abs = self._conservative_abs_to_primitive(q_next_abs, state_abs)
        prim_next_norm = self._norm_state(prim_next_abs)
        if self.predict_type == PREDICT_TYPE_STATE:
            return prim_next_norm
        if self.predict_type == PREDICT_TYPE_DELTA:
            return prim_next_norm - state_norm
        raise RuntimeError(f"Unexpected predict_type='{self.predict_type}'.")

    def predict_state(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_attr: torch.Tensor | None = None,
        dt: float,
        state_override: torch.Tensor | None = None,
        state_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state_override is not None and state_residual is not None:
            raise ValueError("Pass only one of state_override or state_residual.")

        state_in = self._state_slice(x)
        state_ref = state_override if state_override is not None else state_residual
        if state_ref is None:
            state_base = state_in
        else:
            if state_ref.ndim != 2 or state_ref.shape != state_in.shape:
                raise ValueError(
                    "state_override/state_residual must have shape matching the state "
                    f"slice {tuple(state_in.shape)}, got {tuple(state_ref.shape)}."
                )
            state_base = state_ref.to(device=x.device, dtype=x.dtype)

        y = self.forward(x, edge_index, edge_attr=edge_attr, dt=dt)
        if self.predict_type == PREDICT_TYPE_STATE:
            if state_ref is None:
                return y
            return y + (state_base - state_in)
        if self.predict_type == PREDICT_TYPE_DELTA:
            return state_base + y
        if self.predict_type == PREDICT_TYPE_RATE:
            return state_base + float(dt) * y
        raise RuntimeError(f"Unexpected predict_type='{self.predict_type}'.")


class FeatureNet(nn.Module):
    """
    GraphSAGE encoder with a regression head that predicts features at dynamic cell centers.
    Optionally includes a refine-score head (for diagnostics), but geometry CE is *not* trained.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        hidden: int = 128,
        layers: int = 3,
        dropout: float = 0.1,
        make_score_head: bool = True,
        conv_type: str = "sage",
        edge_dim: int | None = None,
        nnconv_hidden: int = 64,
        use_skip: bool = False,
        skip_type: str = "block",
        use_layernorm: bool = False,
        layernorm_eps: float = 1e-6,
        use_attention: bool = False,
        attention_heads: int = 4,
        attention_dropout: float = 0.0,
        attention_edge_dim: int | None = 5,
        attention_use_edge_attr: bool = True,
        attention_replace_last: bool = True,
        activation: str = "relu",
        activation_negative_slope: float = 0.01,
        activation_elu_alpha: float = 1.0,
    ):
        super().__init__()
        if hidden <= 0:
            raise ValueError(f"hidden must be > 0, got {hidden}.")
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}.")
        self.dropout = dropout
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.conv_type = str(conv_type).strip().lower()
        if self.conv_type not in {"sage", "gine", "nnconv"}:
            raise ValueError(
                f"Unsupported FeatureNet conv_type '{conv_type}'. Use 'sage', 'gine', or 'nnconv'."
            )
        self.edge_dim = None if edge_dim is None else int(edge_dim)
        self.use_skip = bool(use_skip)
        self.skip_type = str(skip_type).strip().lower()
        if not self.use_skip:
            self.skip_type = "none"
        valid_skip_types = {"none", "block", "input", "both"}
        if self.skip_type not in valid_skip_types:
            raise ValueError(
                f"Unsupported FeatureNet skip_type '{skip_type}'. "
                f"Use one of {sorted(valid_skip_types)}."
            )
        self.use_layernorm = bool(use_layernorm)
        self.layernorm_eps = float(layernorm_eps)
        self.use_attention = bool(use_attention)
        self.attention_heads = int(attention_heads)
        self.attention_dropout = float(attention_dropout)
        self.attention_edge_dim = attention_edge_dim
        self.attention_use_edge_attr = bool(attention_use_edge_attr)
        self.attention_replace_last = bool(attention_replace_last)
        self.activation_name = str(activation).strip().lower()
        self.activation_negative_slope = float(activation_negative_slope)
        self.activation_elu_alpha = float(activation_elu_alpha)
        self.block_activation = _make_activation(
            self.activation_name,
            negative_slope=self.activation_negative_slope,
            elu_alpha=self.activation_elu_alpha,
        )

        dims = [in_channels] + [hidden] * self.layers
        self.convs = nn.ModuleList()
        self.layer_types = []
        self.block_skip_proj = nn.ModuleList()
        self.block_norms = nn.ModuleList()
        for i in range(len(dims) - 1):
            in_ch = int(dims[i])
            out_ch = int(dims[i + 1])
            replace_this_layer = (
                self.use_attention
                and self.attention_replace_last
                and (i == (len(dims) - 2))
            )
            if replace_this_layer:
                self.convs.append(
                    LocalEdgeAttention(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        heads=self.attention_heads,
                        dropout=self.attention_dropout,
                        edge_dim=self.attention_edge_dim,
                        use_edge_attr=self.attention_use_edge_attr,
                        concat=False,
                    )
                )
                self.layer_types.append("attention")
            else:
                if self.conv_type == "sage":
                    self.convs.append(SAGEConv(in_ch, out_ch))
                elif self.conv_type == "gine":
                    if self.edge_dim is None:
                        raise ValueError("FeatureNet conv_type='gine' requires edge_dim in config.")
                    mlp = nn.Sequential(
                        nn.Linear(in_ch, out_ch),
                        _make_activation(
                            self.activation_name,
                            negative_slope=self.activation_negative_slope,
                            elu_alpha=self.activation_elu_alpha,
                        ),
                        nn.Linear(out_ch, out_ch),
                    )
                    self.convs.append(GINEConv(mlp, edge_dim=self.edge_dim))
                else:  # nnconv
                    if self.edge_dim is None:
                        raise ValueError("FeatureNet conv_type='nnconv' requires edge_dim in config.")
                    edge_net = nn.Sequential(
                        nn.Linear(self.edge_dim, int(nnconv_hidden)),
                        _make_activation(
                            self.activation_name,
                            negative_slope=self.activation_negative_slope,
                            elu_alpha=self.activation_elu_alpha,
                        ),
                        nn.Linear(int(nnconv_hidden), in_ch * out_ch),
                    )
                    self.convs.append(NNConv(in_ch, out_ch, nn=edge_net, aggr="mean"))
                self.layer_types.append("conv")

            if self.skip_type in {"block", "both"}:
                if in_ch == out_ch:
                    self.block_skip_proj.append(nn.Identity())
                else:
                    self.block_skip_proj.append(nn.Linear(in_ch, out_ch, bias=False))
            else:
                self.block_skip_proj.append(nn.Identity())

            if self.use_layernorm:
                self.block_norms.append(nn.LayerNorm(out_ch, eps=self.layernorm_eps))
            else:
                self.block_norms.append(nn.Identity())

        final_hidden = int(dims[-1]) if len(dims) > 0 else int(in_channels)
        if self.skip_type in {"input", "both"}:
            if int(in_channels) == final_hidden:
                self.input_skip_proj = nn.Identity()
            else:
                self.input_skip_proj = nn.Linear(int(in_channels), final_hidden, bias=False)
        else:
            self.input_skip_proj = None

        self.feat_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            _make_activation(
                self.activation_name,
                negative_slope=self.activation_negative_slope,
                elu_alpha=self.activation_elu_alpha,
            ),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_channels)
        )
        self.score_head = None
        if make_score_head:
            self.score_head = nn.Sequential(
                nn.Linear(hidden, hidden//2),
                _make_activation(
                    self.activation_name,
                    negative_slope=self.activation_negative_slope,
                    elu_alpha=self.activation_elu_alpha,
                ),
                nn.Dropout(dropout),
                nn.Linear(hidden//2, 1)  # refine score (logit)
            )

    def forward(
        self,
        X,
        edge_index,
        edge_attr=None,
        *,
        return_score: bool = True,
        return_hidden: bool = True,
    ):
        h = X

        edge_attr_use = None
        need_edge_attr_base = (self.conv_type != "sage")
        need_edge_attr_attn = bool(self.use_attention and self.attention_use_edge_attr)
        if need_edge_attr_base or need_edge_attr_attn:
            if edge_attr is None:
                raise RuntimeError(
                    "FeatureNet requires edge_attr for this configuration "
                    f"(conv_type={self.conv_type}, use_attention={self.use_attention}, "
                    f"attention_use_edge_attr={self.attention_use_edge_attr}) but got None."
                )
            edge_attr_use = edge_attr.to(device=h.device, dtype=h.dtype)

        for li, conv in enumerate(self.convs):
            h_res = h
            if self.layer_types[li] == "attention":
                h = conv(h, edge_index, edge_attr=edge_attr_use)
            else:
                if self.conv_type == "sage":
                    h = conv(h, edge_index)
                else:
                    h = conv(h, edge_index, edge_attr_use)

            if self.skip_type in {"block", "both"}:
                h = h + self.block_skip_proj[li](h_res)

            if self.use_layernorm:
                h = self.block_norms[li](h)

            h = self.block_activation(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        if self.skip_type in {"input", "both"} and self.input_skip_proj is not None:
            h = h + self.input_skip_proj(X)

        y_feat = self.feat_head(h)
        y_score = self.score_head(h) if (return_score and self.score_head is not None) else None
        if return_score or return_hidden:
            return y_feat, y_score, (h if return_hidden else None)
        return y_feat

class FeatureExtractorGNN(nn.Module):
    """
    GraphUNet-based feature extractor for each node with attention.
    """
    def __init__(self, in_channels=2, hidden_channels=64, out_channels=128, 
                 depth=3, pool_ratios=0.5, heads=4, concat=True, dropout=0.6):
        super(FeatureExtractorGNN, self).__init__()
        self.unet = GraphUNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            depth=depth,
            pool_ratios=pool_ratios,
            act=F.relu
        )
        self.attention1 = GATConv(out_channels, out_channels, heads=heads, 
                                  concat=concat, dropout=dropout)
        self.attention2 = GATConv(out_channels * heads if concat else out_channels, 
                                  out_channels, heads=1, concat=False, dropout=dropout)
        # This linear layer for the residual connection needs to match the output of attention2
        self.residual_proj = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        # The original input is passed to the UNet
        unet_out = self.unet(x, edge_index)
        
        # The output of the UNet is used for the residual connection and the attention layers
        residual = self.residual_proj(unet_out)
        
        x = F.elu(self.attention1(unet_out, edge_index))
        x = self.attention2(x, edge_index)
        
        # Add the projected residual
        x += residual
        return x
    
class MPSFeatureExtractor(nn.Module):
    """
    MPS-safe alternative to FeatureExtractorGNN.
    Two GAT layers + residual; no GraphUNet, no CSR sparse ops.
    """
    def __init__(self, in_channels=2, hidden_channels=32, out_channels=32,
                 heads=2, concat=True, dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(in_channels, out_channels,
                            heads=heads, concat=concat, dropout=dropout)
        h1 = out_channels * heads if concat else out_channels
        self.gat2 = GATConv(h1, out_channels, heads=1, concat=False, dropout=dropout)
        self.residual_proj = nn.Linear(in_channels, out_channels)  # for skip from input

    def forward(self, x, edge_index):
        res = self.residual_proj(x)
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x + res
    
class DerivativeGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=3,
                 num_layers=3, heads=4, concat=True, dropout=0.2, use_residual=True):
        super(DerivativeGNN, self).__init__()
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.out_channels = out_channels

        if self.use_residual:
            self.residual_proj = nn.Linear(in_channels, out_channels)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            current_in = in_channels if i == 0 else hidden_channels * (heads if concat else 1)
            current_out = hidden_channels if i < num_layers - 1 else out_channels
            is_last_layer = (i == num_layers - 1)
            
            self.layers.append(nn.LayerNorm(current_in, eps=1e-6))# adding in , eps=1e-6
            self.layers.append(
                GATConv(current_in, current_out, 
                        heads=1 if is_last_layer else heads,
                        concat=False if is_last_layer else concat, 
                        dropout=dropout)
            )

    def forward(self, x, edge_index):
        residual = self.residual_proj(x) if self.use_residual else None

        for i in range(self.num_layers):
            ln = self.layers[2*i]
            gnn = self.layers[2*i + 1]
            
            x_res = x
            x = ln(x)
            x = gnn(x, edge_index)

            if i < self.num_layers - 1:
                x = F.gelu(x)
                if x.shape == x_res.shape: # Add skip connections between layers
                     x = x + x_res
        
        if self.use_residual and residual is not None:
            x = x + residual
            
        return x
    

class IntegralGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=3, 
                 num_layers=3, heads=4, concat=True, dropout=0.2, use_residual=True):
        super(IntegralGNN, self).__init__()
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.out_channels = out_channels

        if use_residual:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            current_in = in_channels if i == 0 else hidden_channels * (heads if concat else 1)
            current_out = hidden_channels if i < num_layers - 1 else out_channels
            is_last_layer = (i == num_layers - 1)

            self.layers.append(nn.LayerNorm(current_in, eps=1e-6))#, eps=1e-6
            self.layers.append(
                GATConv(current_in, current_out, 
                        heads=1 if is_last_layer else heads,
                        concat=False if is_last_layer else concat, 
                        dropout=dropout)
            )

    def forward(self, x, edge_index):
        residual = self.residual_proj(x) if self.use_residual else None
        
        for i in range(self.num_layers):
            ln = self.layers[2*i]
            gnn = self.layers[2*i + 1]
            
            x_res = x
            x = ln(x)
            x = gnn(x, edge_index)

            if i < self.num_layers - 1:
                x = F.gelu(x)
                if x.shape == x_res.shape: # Add skip connections between layers
                    x = x + x_res

        if self.use_residual and residual is not None:
            x = x + residual
            
        return x

# ---------------------------------------------------------------------
# GPARC-style wrapper keeping your modules unchanged
# ---------------------------------------------------------------------
class GPARCCompat(nn.Module):
    """
    Wraps FeatureExtractorGNN, DerivativeGNN, IntegralGNN into a single nn.Module.

    forward inputs:
      x_static : [N, S] (pos, level, etc.)
      x_dyn    : [N, D] (rho, px, E at time t on M_pred(t+1))
      edge_idx : [2, E]
      g        : [G] or [B,G] (optional global conditioning, currently ignored)

    forward outputs:
      x_pred   : [N, D] (features at t+1 on M_pred(t+1))
    """
    def __init__(
        self,
        in_static: int,
        in_dynamic: int,
        feature_out: int = 128,
        feat_hidden: int = 64,
        feat_depth: int = 2,
        feat_pool: float = 0.1,
        feat_heads: int = 4,
        feat_dropout: float = 0.2,
        deriv_hidden: int = 128,
        deriv_layers: int = 4,
        deriv_heads: int = 8,
        deriv_dropout: float = 0.3,
        deriv_residual: bool = True,
        integ_hidden: int = 128,
        integ_layers: int = 4,
        integ_heads: int = 8,
        integ_dropout: float = 0.3,
        integ_residual: bool = True,
        use_delta: bool = True,
        global_embed_dim: int = 0,  # set >0 if you later add a global embedding
    ):
        super().__init__()
        self.use_delta = bool(use_delta)
        self.in_dynamic = int(in_dynamic)
        self.global_embed_dim = int(global_embed_dim)

        # 1) per-node static encoder (GraphUNet + attention)
        self.feature_extractor = MPSFeatureExtractor(
            in_channels=in_static,
            hidden_channels=feat_hidden,
            out_channels=feature_out,
            heads=feat_heads,
            concat=True,
            dropout=feat_dropout,
        )

        # 2) time-derivative GNN (takes concat[feat_emb, x_dyn, (g_emb)])
        deriv_in = feature_out + in_dynamic + self.global_embed_dim
        self.derivative = DerivativeGNN(
            in_channels=deriv_in,
            hidden_channels=deriv_hidden,
            out_channels=in_dynamic,
            num_layers=deriv_layers,
            heads=deriv_heads,
            concat=True,
            dropout=deriv_dropout,
            use_residual=deriv_residual,
        )

        # 3) integrator GNN (maps derivative to a delta or absolute)
        self.integrator = IntegralGNN(
            in_channels=in_dynamic,
            hidden_channels=integ_hidden,
            out_channels=in_dynamic,
            num_layers=integ_layers,
            heads=integ_heads,
            concat=True,
            dropout=integ_dropout,
            use_residual=integ_residual,
        )

    def forward(
        self,
        x_static: torch.Tensor,
        x_dyn: torch.Tensor,
        edge_index: torch.Tensor,
        g: torch.Tensor | None = None,
    ):
        # 1) Feature embedding from static geometry
        z = self.feature_extractor(x_static, edge_index)  # [N, feature_out]

        # 2) Concatenate dynamic and optional global context
        if self.global_embed_dim > 0 and g is not None:
            if g.dim() == 1:
                g = g[None, :]                 # [1, G]
            g = g.expand(z.size(0), -1)        # naive broadcast
            h_in = torch.cat([z, x_dyn, g], dim=-1)
        else:
            h_in = torch.cat([z, x_dyn], dim=-1)

        # 3) Derivative -> Integral stacks
        dstate = self.derivative(h_in, edge_index)        # [N, D]
        delta  = self.integrator(dstate, edge_index)      # [N, D]

        # 4) Output contract:
        #    - use_delta=True  -> return Δ (your training/rollout expect this)
        #    - use_delta=False -> return absolute (x_dyn + Δ)
        return delta if self.use_delta else (x_dyn + delta)

def _filter_kwargs(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def build_model(cfg: Dict[str, Any], in_dim: int, out_dim: int):
    """
    Build a model from cfg. Expected cfg structure:
      cfg["model"] = {
        "name": "YourModelClassName",
        # ... hyperparameters specific to that class ...
      }

    We will pass (in_dim, out_dim, **filtered_kwargs) if the constructor
    accepts them; otherwise we pass only what the constructor supports.
    """
    if cfg is None:
        raise ValueError("cfg is None; need cfg['model']['name'] at minimum.")
    model_cfg = cfg.get("model", cfg)  # allow cfg itself to be the model dict
    name = model_cfg.get("name", None)
    if not name:
        raise ValueError("cfg['model']['name'] is required, e.g. 'MeshGNN'.")

    # Resolve the class by name from this module
    cls = globals().get(name, None)
    if cls is None:
        # Also allow lowercase alias
        for k, v in globals().items():
            if k.lower() == str(name).lower() and inspect.isclass(v):
                cls = v
                break
    if cls is None or not inspect.isclass(cls):
        raise ValueError(f"Model class '{name}' not found in models.py.")

    # Prepare kwargs: include in_dim/out_dim if the ctor supports them
    ctor = cls.__init__
    kwargs = dict(model_cfg)  # copy
    kwargs.pop("name", None)
    # Some configs put dims under different keys; keep a few aliases:
    if "input_dim" not in kwargs:  kwargs["input_dim"]  = in_dim
    if "in_dim" not in kwargs:     kwargs["in_dim"]     = in_dim
    if "out_dim" not in kwargs:    kwargs["out_dim"]    = out_dim
    if "output_dim" not in kwargs: kwargs["output_dim"] = out_dim

    kwargs = _filter_kwargs(ctor, kwargs)

    # Instantiate
    model = cls(**kwargs)

    # If the class exposes a 'reset_parameters' helper, call it
    if hasattr(model, "reset_parameters") and callable(getattr(model, "reset_parameters")):
        try:
            model.reset_parameters()
        except Exception:
            pass

    return model

# Back-compat alias some training scripts use
make_model = build_model


class ParcFeatureAdapter(nn.Module):
    """
    Applies:
      (1) pre-clip on PARC features
      (2) running normalization (train updates; eval uses frozen stats)
      (3) learnable gates (per-channel by default) initialized near 0 influence
      (4) optional post-clip
    """
    def __init__(
        self,
        dim_adv: int,
        dim_diff: int,
        *,
        use_norm: bool = True,
        clip_pre: float = 50.0,
        clip_post: float = 10.0,
        momentum: float = 0.02,
        eps: float = 1e-6,
        var_floor: float = 1e-6,
        per_channel_gates: bool = True,
        gate_init: float = -3.0,   # sigmoid(-5) ~ 0.0067 (starts near OFF)
    ):
        super().__init__()
        self.dim_adv = int(dim_adv)
        self.dim_diff = int(dim_diff)
        self.dim = self.dim_adv + self.dim_diff

        self.use_norm = bool(use_norm)
        self.clip_pre = float(clip_pre) if clip_pre is not None else None
        self.clip_post = float(clip_post) if clip_post is not None else None
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.var_floor = float(var_floor)

        self.per_channel_gates = bool(per_channel_gates)

        # Running stats in FP32 (stable across AMP / MPS / CUDA)
        self.register_buffer("running_mean", torch.zeros(self.dim, dtype=torch.float32))
        self.register_buffer("running_var",  torch.ones(self.dim,  dtype=torch.float32))
        self.register_buffer("num_updates",  torch.tensor(0, dtype=torch.long))

        # Gates
        if self.per_channel_gates:
            self.gate_logits = nn.Parameter(torch.full((self.dim,), float(gate_init), dtype=torch.float32))
        else:
            # one scalar for adv block, one for diff block
            self.gate_adv_logit  = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))
            self.gate_diff_logit = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    def _gate_vector(self, device, dtype):
        if self.dim == 0:
            return None
        if self.per_channel_gates:
            g = torch.sigmoid(self.gate_logits)  # [dim]
        else:
            g_adv  = torch.sigmoid(self.gate_adv_logit)
            g_diff = torch.sigmoid(self.gate_diff_logit)
            parts = []
            if self.dim_adv  > 0: parts.append(g_adv.expand(self.dim_adv))
            if self.dim_diff > 0: parts.append(g_diff.expand(self.dim_diff))
            g = torch.cat(parts, dim=0) if len(parts) else torch.zeros((0,), dtype=torch.float32)
        return g.to(device=device, dtype=dtype).view(1, -1)  # [1,dim]

    def forward(
        self,
        parc_extra: torch.Tensor,
        *,
        update_adv_stats: bool = True,
        update_diff_stats: bool = True,
    ) -> torch.Tensor:
        if parc_extra is None or parc_extra.numel() == 0:
            return parc_extra
        if parc_extra.ndim != 2 or parc_extra.size(1) != self.dim:
            raise RuntimeError(f"ParcFeatureAdapter expected [N,{self.dim}], got {tuple(parc_extra.shape)}")

        out_dtype = parc_extra.dtype
        x = parc_extra.to(dtype=torch.float32)

        # 1) pre-clip
        if (self.clip_pre is not None) and (self.clip_pre > 0):
            x = x.clamp(-self.clip_pre, self.clip_pre)

        # 2) normalization
        if self.use_norm:
            if self.training:
                # ---- batch stats for normalization ----
                m = x.mean(dim=0)
                v = (x - m).pow(2).mean(dim=0)  # population var

                denom = torch.sqrt(v.clamp_min(self.var_floor) + self.eps)
                x = (x - m) / denom

                # ---- selective EMA update for eval-time stability ----
                did_update = False
                if self.dim_adv > 0 and update_adv_stats:
                    sl = slice(0, self.dim_adv)
                    self.running_mean[sl].mul_(1.0 - self.momentum).add_(self.momentum * m[sl].detach())
                    self.running_var[sl].mul_(1.0 - self.momentum).add_(self.momentum * v[sl].detach())
                    did_update = True

                if self.dim_diff > 0 and update_diff_stats:
                    sl = slice(self.dim_adv, self.dim_adv + self.dim_diff)
                    self.running_mean[sl].mul_(1.0 - self.momentum).add_(self.momentum * m[sl].detach())
                    self.running_var[sl].mul_(1.0 - self.momentum).add_(self.momentum * v[sl].detach())
                    did_update = True

                if did_update:
                    self.num_updates.add_(1)

            else:
                # ---- eval: use running stats ----
                denom = torch.sqrt(self.running_var.clamp_min(self.var_floor) + self.eps)
                x = (x - self.running_mean) / denom

        # 3) gates
        g = self._gate_vector(device=x.device, dtype=x.dtype)  # [1,dim]
        if g is not None:
            x = x * g

        # 4) post-clip
        if (self.clip_post is not None) and (self.clip_post > 0):
            x = x.clamp(-self.clip_post, self.clip_post)

        return x.to(dtype=out_dtype)

    @torch.no_grad()
    def gate_values(self):
        """Convenience for debug prints."""
        if self.dim == 0:
            return {"adv": None, "diff": None}
        if self.per_channel_gates:
            g = torch.sigmoid(self.gate_logits).detach().cpu()
            ga = g[:self.dim_adv] if self.dim_adv > 0 else None
            gd = g[self.dim_adv:] if self.dim_diff > 0 else None
            return {"adv": ga, "diff": gd}
        else:
            return {
                "adv":  float(torch.sigmoid(self.gate_adv_logit).detach().cpu()),
                "diff": float(torch.sigmoid(self.gate_diff_logit).detach().cpu()),
            }
