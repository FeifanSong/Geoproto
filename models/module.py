"""
ALPModule
Bin Prediction Head: predicts Scale Class Map from anatomy features.

Each foreground voxel predicts which distance bin it belongs to:
  - Bin 0: near boundary (small distance to surface)
  - Bin K-1: far from boundary (organ centre, large distance)
─────────────────────────────────────────────────────────────────────────────
Geometry-Aware Prototype Enhancement
  SDFBinEmbedding: a lightweight MLP that maps a scalar bin value
  (0 ~ K-1) to the same dimension as the feature vector, so every local
  prototype f_i can be enriched as:
      f_i' = f_i + e_i
  "Edge prototypes" and "centre prototypes" now live in different regions
  of feature space, enabling geometry-aware matching inside GridProto.
─────────────────────────────────────────────────────────────────────────────
"""
import torch
from torch import nn
from torch.nn import functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Bin Embedding
# ══════════════════════════════════════════════════════════════════════════════

class SDFBinEmbedding(nn.Module):
    """
    Maps a spatial SDF-bin map (float, 0 ~ K-1) to a feature-dimension
    additive offset that can be added to local prototypes.

    Architecture:
        scalar → Linear(1, hidden) → ReLU → Linear(hidden, feat_dim)

    This is applied only to foreground local prototypes (those kept by
    the thresh filter in gridconv / gridconv+) so background matching is
    unaffected.
    """

    def __init__(self, feat_dim: int = 256, K: int = 10, hidden: int = 64):
        """
        Args:
            feat_dim:   channel dimension of backbone features (e.g. 256)
            K:          number of SDF bins (must match SDFHead.K)
            hidden:     hidden size of the 2-layer MLP
        """
        super().__init__()
        self.K = K
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim),
        )
        # Initialize to near-zero so that at training start the enriched
        # prototypes are almost identical to the un-enriched ones, giving
        # the geometry embedding a chance to ramp up gradually.
        nn.init.normal_(self.mlp[0].weight, std=0.01)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=0.01)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, sdf_bin_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sdf_bin_values: (N_fg,) float tensor of raw bin indices in [0, K-1].
                            typically the spatially-pooled average bin value
                            for each local prototype region.
        Returns:
            offsets: (N, feat_dim) additive offset for each prototype.
        """
        normed = sdf_bin_values.unsqueeze(-1).float() / max(self.K - 1, 1)  # (N, 1)
        return self.mlp(normed)   # (N, feat_dim)


# ══════════════════════════════════════════════════════════════════════════════
# MultiProtoAsConv
# ══════════════════════════════════════════════════════════════════════════════

class MultiProtoAsConv(nn.Module):
    def __init__(self, proto_grid, feature_hw, upsample_mode='bilinear',
                 sdf_bin_embedding: SDFBinEmbedding = None):
        """
        ALPModule — extended with optional geometry-aware prototype enrichment.
        Originate from the SSL_ALPNet paper.

        Args:
            proto_grid:         grid size for multi-prototyping (e.g. [8, 8])
            feature_hw:         spatial size of input feature map (e.g. [32, 32])
            upsample_mode:      interpolation mode for output upsampling
            sdf_bin_embedding:  optional SDFBinEmbedding instance.
                                When provided and sdf_map is passed to forward(),
                                foreground local prototypes are enriched with a
                                geometry-aware additive offset before matching.
        """
        super(MultiProtoAsConv, self).__init__()
        self.proto_grid = proto_grid
        self.upsample_mode = upsample_mode
        kernel_size = [ft_l // grid_l for ft_l, grid_l in zip(feature_hw, proto_grid)]
        self.avg_pool_op = nn.AvgPool2d(kernel_size)
        self.sdf_bin_embedding = sdf_bin_embedding

    def forward(self, qry, sup_x, sup_y, mode, thresh,
                isval=False, val_wsize=None,
                sdf_map=None):
        """
        Args:
            mode:       'mask' / 'gridconv' / 'gridconv+'
            qry:        [way(1), nb(1), nc, h, w]
            sup_x:      [way(1), shot, nb(1), nc, h, w]
            sup_y:      [way(1), shot, nb(1), h, w]

        new kwarg:
            sdf_map:    (K_shots, h, w) float tensor of averaged SDF bin
                        values at feature resolution (0 ~ K-1), or None.

        Returns (unchanged):
            pred, [debug_info]
        """
        qry   = qry.squeeze(1)               # [way, nc, h, w]
        sup_x = sup_x.squeeze(0).squeeze(1)  # [nshot, nc, h, w]
        sup_y = sup_y.squeeze(0)             # [nshot, 1, h, w]

        def safe_norm(x, p=2, dim=1, eps=1e-4):
            x_norm = torch.norm(x, p=p, dim=dim)
            x_norm = torch.max(x_norm, torch.ones_like(x_norm) * eps)
            x = x.div(x_norm.unsqueeze(1).expand_as(x))
            return x

        if mode == 'mask':
            proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) \
                    / (sup_y.sum(dim=(-1, -2)) + 1e-5)
            proto = proto.mean(dim=0, keepdim=True)
            pred_mask = F.cosine_similarity(qry, proto[..., None, None], dim=1, eps=1e-4) * 20.0
            return pred_mask.unsqueeze(1), [pred_mask]

        # ── common setup for gridconv / gridconv+ ─────────────────────────────
        input_size = qry.shape
        nch        = input_size[1]
        sup_nshot  = sup_x.shape[0]

        n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op(sup_x)  # (nshot, nch, gh, gw)
        n_sup_x_flat = n_sup_x.view(sup_nshot, nch, -1).permute(0, 2, 1).unsqueeze(0)   # (1, nshot, n_cells, nch)
        n_sup_x_flat = n_sup_x_flat.reshape(1, -1, nch).unsqueeze(0)     # (1, 1, nshot*n_cells, nch)

        sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)
        sup_y_g_flat = sup_y_g.view(sup_nshot, 1, -1).permute(1, 0, 2).view(1, -1).unsqueeze(0)  # (1, 1, nshot*n_cells)

        # ── pool SDF map to prototype grid ─────────────────────────
        sdf_g_flat = None
        if sdf_map is not None and self.sdf_bin_embedding is not None:
            sdf_map_4d = sdf_map.unsqueeze(1).float()   # (nshot, 1, h, w)
            sdf_g = F.avg_pool2d(sdf_map_4d, val_wsize) if isval else self.avg_pool_op(sdf_map_4d) # (nshot, 1, gh, gw)
            sdf_g_flat = sdf_g.view(sup_nshot, -1).view(-1)  # (nshot*n_cells,)

        # ───Boolean mask: which (shot, cell) pairs are fg prototypes? ───────
        fg_mask_flat = sup_y_g_flat[0, 0] > thresh          # (nshot*n_cells,)
        protos_raw   = n_sup_x_flat[0, 0][fg_mask_flat, :]  # (n_fg, nch)
        cell_indices = torch.where(fg_mask_flat)[0]  # ← record cell ordinal number

        # ── enrich foreground prototypes with geometry offset ──────
        if sdf_g_flat is not None and protos_raw.shape[0] > 0:
            sdf_fg_vals = sdf_g_flat[fg_mask_flat]              # (n_fg,)
            geo_offset  = self.sdf_bin_embedding(sdf_fg_vals)   # (n_fg, nch)
            protos_raw  = protos_raw + geo_offset
        # ─────────────────────────────────────────────────────────────────────

        if mode == 'gridconv':
            protos   = protos_raw                               # (n_fg, nch)
            pro_n    = safe_norm(protos)
            qry_n    = safe_norm(qry)

            dists      = F.conv2d(qry_n, pro_n[..., None, None]) * 20
            pred_grid  = torch.sum(F.softmax(dists, dim=1) * dists, dim=1, keepdim=True)
            debug_assign = dists.argmax(dim=1).float().detach()
            return pred_grid, [debug_assign]

        elif mode == 'gridconv+':
            # global prototype (unaffected by geometry embedding to stay as anchor)
            glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) \
                        / (sup_y.sum(dim=(-1, -2)) + 1e-5)     # (nshot, nch)

            pro_n  = safe_norm(torch.cat([protos_raw, glb_proto], dim=0))
            qry_n  = safe_norm(qry)

            dists      = F.conv2d(qry_n, pro_n[..., None, None]) * 20
            pred_grid  = torch.sum(F.softmax(dists, dim=1) * dists, dim=1, keepdim=True)
            debug_assign = dists.argmax(dim=1).float()
            return pred_grid, [debug_assign]

        else:
            raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Bin Prediction Head
# ══════════════════════════════════════════════════════════════════════════════

class SDFHead(nn.Module):
    """
    For each organ class, predicts which of K distance bins each
    spatial location belongs to.
    """

    def __init__(self,
                 in_dim: int = 256,
                 K: int = 10):
        """
        Args:
            in_dim:     input feature channels (from SharedEncoder)
            K:          number of scale/distance bins
        """
        super().__init__()
        self.K = K

        self.head = nn.Sequential(
            nn.Conv2d(in_dim, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.K, 1)
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat:   (B, C, h, w) from SharedEncoder
        Returns:
            logits: (B, out_K, h, w)
        """
        out = self.head(feat)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Geometry Loss
# ══════════════════════════════════════════════════════════════════════════════

class SDFLoss(nn.Module):
    """
    Distance loss.

    Two terms:
    1. Softmax classification loss — penalises wrong bin prediction.
    2. Distance penalty term     — penalises by HOW FAR the prediction
       is from the GT bin, making the loss geometry-aware.

    Correct prediction → ω = 0, no extra penalty.
    5 bins off         → large extra penalty.
    """

    def __init__(self, K: int = 10, lambda_dist: float = 1.0):
        super().__init__()
        self.K = K
        self.lambda_dist = lambda_dist
        self.eps = 1e-6

    def forward(self,
                logits: torch.Tensor,
                sdf_gt: torch.Tensor,
                seg_mask: torch.Tensor) -> tuple:
        """
        Args:
            logits:    (B, K, h, w) predicted scale-class logits
            sdf_gt:    (B, H, W)    GT scale class map, int [0, K-1]
            seg_mask:  (B, H, W)    binary foreground mask

        Returns:
            loss:      scalar tensor
            loss_dict: dict with per-term losses for logging
        """
        B, K, h, w = logits.shape

        # Downsample GT to feature resolution
        sdf_gt_small = F.interpolate(
            sdf_gt.float(), size=(h, w), mode='nearest'
        ).squeeze(1).long()           # (B, h, w)

        seg_small = F.interpolate(
            seg_mask.unsqueeze(1).float(), size=(h, w), mode='nearest'
        ).squeeze(1)                  # (B, h, w)

        fg_mask = (seg_small > 0) & (sdf_gt_small > 0)

        if fg_mask.sum() == 0:
            zero = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return zero, {'sdf_cls': 0.0, 'sdf_dist': 0.0, 'sdf_total': 0.0}

        # Term 1: softmax classification loss
        log_probs    = F.log_softmax(logits, dim=1)
        z_v  = sdf_gt_small.clamp(0, K - 1)
        beta_p = 0.5 / (fg_mask.float().sum() + self.eps)

        l_cls = -log_probs.gather(1, z_v.unsqueeze(1)).squeeze(1)
        L_cls = beta_p * (l_cls * fg_mask.float()).sum()

        # Term 2: distance penalty loss
        probs      = F.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1)
        max_prob   = probs.max(dim=1).values
        omega        = (pred_class - z_v).abs().float() / self.K
        wrong_and_fg = fg_mask & (pred_class != z_v)

        if wrong_and_fg.sum() > 0:
            l_dist = -omega * torch.log(1.0 - max_prob + self.eps)
            L_dist = beta_p * (l_dist * wrong_and_fg.float()).sum()
        else:
            L_dist = torch.tensor(0.0, device=logits.device)

        total_loss = L_cls + self.lambda_dist * L_dist

        loss_dict = {
            'sdf_cls': L_cls.item(),
            'sdf_dist': float(L_dist.item() if isinstance(L_dist, torch.Tensor) else L_dist),
            'sdf_total': total_loss.item(),
        }

        return total_loss, loss_dict
