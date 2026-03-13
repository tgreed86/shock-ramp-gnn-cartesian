import torch
from torch import nn
import torch.nn.functional as F
import time
from torch_geometric.nn import GATConv, GraphUNet
from torch_geometric.nn import SAGEConv, GINEConv, NNConv
import inspect
from typing import Any, Dict


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
    ):
        super().__init__()
        self.dropout = dropout
        self.conv_type = str(conv_type).strip().lower()
        if self.conv_type not in {"sage", "gine", "nnconv"}:
            raise ValueError(
                f"Unsupported FeatureNet conv_type '{conv_type}'. Use 'sage', 'gine', or 'nnconv'."
            )
        self.edge_dim = None if edge_dim is None else int(edge_dim)
        dims = [in_channels] + [hidden] * (layers - 1)
        self.convs = nn.ModuleList()
        for i in range(len(dims) - 1):
            in_ch = int(dims[i])
            out_ch = int(dims[i + 1])
            if self.conv_type == "sage":
                self.convs.append(SAGEConv(in_ch, out_ch))
            elif self.conv_type == "gine":
                if self.edge_dim is None:
                    raise ValueError("FeatureNet conv_type='gine' requires edge_dim in config.")
                mlp = nn.Sequential(
                    nn.Linear(in_ch, out_ch),
                    nn.ReLU(),
                    nn.Linear(out_ch, out_ch),
                )
                self.convs.append(GINEConv(mlp, edge_dim=self.edge_dim))
            else:  # nnconv
                if self.edge_dim is None:
                    raise ValueError("FeatureNet conv_type='nnconv' requires edge_dim in config.")
                edge_net = nn.Sequential(
                    nn.Linear(self.edge_dim, int(nnconv_hidden)),
                    nn.ReLU(),
                    nn.Linear(int(nnconv_hidden), in_ch * out_ch),
                )
                self.convs.append(NNConv(in_ch, out_ch, nn=edge_net, aggr="mean"))
        self.feat_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_channels)
        )
        self.score_head = None
        if make_score_head:
            self.score_head = nn.Sequential(
                nn.Linear(hidden, hidden//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden//2, 1)  # refine score (logit)
            )

    def forward(self, X, edge_index, edge_attr=None):
        h = X
        hang_dbg_once = not hasattr(self, "_hang_dbg_once")
        if hang_dbg_once:
            try:
                ei_shape = tuple(edge_index.shape) if torch.is_tensor(edge_index) else None
            except Exception:
                ei_shape = None
            try:
                ea_shape = tuple(edge_attr.shape) if torch.is_tensor(edge_attr) else None
            except Exception:
                ea_shape = None
            print(
                f"[HANG-DBG] FeatureNet.forward start: X={tuple(X.shape)} "
                f"dtype={X.dtype} dev={X.device} edge_index={ei_shape} "
                f"edge_attr={ea_shape} conv_type={self.conv_type}",
                flush=True,
            )

        edge_attr_use = None
        if self.conv_type != "sage":
            if edge_attr is None:
                raise RuntimeError(
                    f"FeatureNet conv_type='{self.conv_type}' requires edge_attr, but got None."
                )
            edge_attr_use = edge_attr.to(device=h.device, dtype=h.dtype)

        for li, conv in enumerate(self.convs):
            t0 = time.perf_counter() if hang_dbg_once else None
            if hang_dbg_once:
                print(f"[HANG-DBG] FeatureNet conv[{li}] begin", flush=True)
            if self.conv_type == "sage":
                h = conv(h, edge_index)
            else:
                h = conv(h, edge_index, edge_attr_use)
            if hang_dbg_once:
                print(
                    f"[HANG-DBG] FeatureNet conv[{li}] done in {time.perf_counter() - t0:.3f}s",
                    flush=True,
                )
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        y_feat = self.feat_head(h)
        y_score = self.score_head(h) if self.score_head is not None else None
        if hang_dbg_once:
            print(
                f"[HANG-DBG] FeatureNet heads done: y_feat={tuple(y_feat.shape)} "
                f"y_score={None if y_score is None else tuple(y_score.shape)}",
                flush=True,
            )
            self._hang_dbg_once = True
        return y_feat, y_score, h

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
