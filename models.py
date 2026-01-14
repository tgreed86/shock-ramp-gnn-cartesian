import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphUNet
from torch_geometric.nn import SAGEConv
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
    ):
        super().__init__()
        self.dropout = dropout
        dims = [in_channels] + [hidden] * (layers - 1)
        self.convs = nn.ModuleList([SAGEConv(dims[i], dims[i+1]) for i in range(len(dims)-1)])
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

    def forward(self, X, edge_index):
        h = X
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        y_feat = self.feat_head(h)
        y_score = self.score_head(h) if self.score_head is not None else None
        return y_feat, y_score, h

'''
class FeatureNet(nn.Module):
    def __init__(self, in_channels, out_channels=3, hidden=128, layers=3, dropout=0.1,
                 make_score_head=True, use_residual=True, use_norm=True):
        super().__init__()
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_norm = use_norm

        dims = [in_channels] + [hidden] * (layers - 1)
        self.convs = nn.ModuleList([SAGEConv(dims[i], dims[i+1]) for i in range(len(dims)-1)])

        # Projection for the first residual addition (only needed if in_channels != hidden)
        self.in_proj = nn.Linear(in_channels, hidden) if in_channels != hidden else nn.Identity()

        # LayerNorm per conv output (all convs output "hidden" except possibly the first,
        # but dims[i+1] is hidden for all convs in your construction when layers>=2)
        if use_norm:
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(len(self.convs))])
        else:
            self.norms = None

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
                nn.Linear(hidden//2, 1)
            )

    def forward(self, X, edge_index):
        h = X
        for i, conv in enumerate(self.convs):
            h_in = h

            h = conv(h, edge_index)

            # Normalize before nonlinearity (common “pre-activation” style)
            if self.use_norm:
                h = self.norms[i](h)

            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            if self.use_residual:
                # First residual needs projection if dimensions differ
                if i == 0:
                    h = h + self.in_proj(h_in)
                else:
                    h = h + h_in

        y_feat = self.feat_head(h)
        y_score = self.score_head(h) if self.score_head is not None else None
        return y_feat, y_score, h
'''
'''
def _make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    depth: int,
    dropout: float,
) -> nn.Sequential:
    """
    Builds an MLP with `depth` hidden layers:
      (Linear -> ReLU -> Dropout) * depth, then final Linear to out_dim.

    If depth == 0: returns a single Linear(in_dim, out_dim).
    """
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class FeatureNet(nn.Module):
    """
    GraphSAGE encoder with:
      - Residual connections + LayerNorm per message-passing layer
      - Deeper (pointwise) MLP head(s) to increase capacity without adding more message passing

    Notes on depth:
      - Message passing depth is controlled by `mp_steps` (preferred), or by legacy `layers`
        (where mp_steps defaults to max(layers - 1, 1) to preserve your current behavior).
      - Head depth is controlled by `feat_head_depth` / `score_head_depth`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        hidden: int = 128,
        layers: int = 3,                 # legacy: mp_steps defaults to layers-1
        dropout: float = 0.1,
        make_score_head: bool = True,
        mp_steps: int | None = None,     # preferred: number of SAGEConv layers
        feat_head_depth: int = 3,        # number of hidden layers in feat head MLP
        score_head_depth: int = 2,       # number of hidden layers in score head MLP
        score_head_hidden: int | None = None,  # internal width for score head; default hidden//2
    ):
        super().__init__()
        self.dropout = dropout

        # --- message passing depth ---
        self.mp_steps = int(mp_steps) if mp_steps is not None else max(int(layers) - 1, 1)

        # --- GraphSAGE conv stack ---
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First conv: in_channels -> hidden
        self.convs.append(SAGEConv(in_channels, hidden))
        self.norms.append(nn.LayerNorm(hidden))

        # Remaining convs: hidden -> hidden
        for _ in range(self.mp_steps - 1):
            self.convs.append(SAGEConv(hidden, hidden))
            self.norms.append(nn.LayerNorm(hidden))

        # Projection needed for the first residual (in_channels -> hidden)
        self.in_proj = nn.Linear(in_channels, hidden) if in_channels != hidden else nn.Identity()

        # --- Deeper MLP head(s) (pointwise; no neighbor mixing) ---
        self.feat_head = _make_mlp(
            in_dim=hidden,
            hidden_dim=hidden,
            out_dim=out_channels,
            depth=int(feat_head_depth),
            dropout=dropout,
        )

        self.score_head = None
        if make_score_head:
            sh = int(score_head_hidden) if score_head_hidden is not None else max(hidden // 2, 1)
            self.score_head = _make_mlp(
                in_dim=hidden,
                hidden_dim=sh,
                out_dim=1,
                depth=int(score_head_depth),
                dropout=dropout,
            )

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor):
        h = X

        for i, conv in enumerate(self.convs):
            h_in = h

            # Message passing
            h = conv(h, edge_index)

            # LayerNorm -> ReLU -> Dropout (pre-activation style)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # Residual / skip connection
            if i == 0:
                h = h + self.in_proj(h_in)
            else:
                h = h + h_in

        y_feat = self.feat_head(h)
        y_score = self.score_head(h) if self.score_head is not None else None
        return y_feat, y_score, h
'''

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
        '''
        self.feature_extractor = FeatureExtractorGNN(
            in_channels=in_static,
            hidden_channels=feat_hidden,
            out_channels=feature_out,
            depth=feat_depth,
            pool_ratios=feat_pool,
            heads=feat_heads,
            concat=True,
            dropout=feat_dropout,
        )
        '''
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
    '''
    def forward(self, x_static: torch.Tensor, x_dyn: torch.Tensor, edge_index: torch.Tensor, g: torch.Tensor | None = None):
        # Feature embedding from static geometry
        z = self.feature_extractor(x_static, edge_index)  # [N, feature_out]

        if self.global_embed_dim > 0 and g is not None:
            if g.dim() == 1: g = g[None, :]  # [1, G]
            g = g.expand(z.size(0), -1)      # naive broadcast
            h_in = torch.cat([z, x_dyn, g], dim=-1)
        else:
            h_in = torch.cat([z, x_dyn], dim=-1)

        dstate = self.derivative(h_in, edge_index)       # [N, D]
        delta  = self.integrator(dstate, edge_index)     # [N, D]

        if self.use_delta:
            x_pred = x_dyn + delta                       # semi-implicit
        else:
            x_pred = delta                               # absolute state

        return x_pred
    '''
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

'''
class GPARCCompat(nn.Module):
    """
    Wraps FeatureExtractorGNN, DerivativeGNN, IntegralGNN into a single nn.Module.

    forward(x_static [N,S], x_dyn [N,D], edge_index [2,E]) -> x_pred [N,D]
    If use_delta=True, returns x_dyn + Δ; else returns absolute state.
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
        global_embed_dim: int = 0,    # bump later if you add a global embedding
    ):
        super().__init__()
        self.use_delta = bool(use_delta)
        self.in_dynamic = int(in_dynamic)
        self.global_embed_dim = int(global_embed_dim)

        # 1) static geometry encoder
        self.feature_extractor = FeatureExtractorGNN(
            in_channels=in_static,
            hidden_channels=feat_hidden,
            out_channels=feature_out,
            depth=feat_depth,
            pool_ratios=feat_pool,
            heads=feat_heads,
            concat=True,
            dropout=feat_dropout,
        )

        # 2) derivative solver operates on [static_embed || x_dyn (|| g_embed)]
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

        # 3) integrator maps derivative to Δ (or absolute)
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

    def forward(self, x_static: torch.Tensor, x_dyn: torch.Tensor, edge_index: torch.Tensor, g: torch.Tensor | None = None):
        z = self.feature_extractor(x_static, edge_index)     # [N, feature_out]
        if self.global_embed_dim > 0 and g is not None:
            if g.dim() == 1: g = g[None, :]
            g = g.expand(z.size(0), -1)
            h = torch.cat([z, x_dyn, g], dim=-1)
        else:
            h = torch.cat([z, x_dyn], dim=-1)

        dstate = self.derivative(h, edge_index)              # [N, D]
        delta  = self.integrator(dstate, edge_index)         # [N, D]
        return x_dyn + delta if self.use_delta else delta
'''


