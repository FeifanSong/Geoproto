"""
Geoproto  —  geometry-aware framework for CD-FSMIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SDFBinEmbedding is registered inside cls_unit (MultiProtoAsConv).

During forward(), we compute a per-pixel bin map from _last_sdf_logits
and pool it to prototype-grid resolution.  Each local foreground prototype f_i
(shot × grid_cell) then receives an additive geometry offset:

    f_i' = f_i + MLP(d_i / (K-1))

where d_i is the average SDF bin value for that cell.  The enriched prototype
now encodes "what it looks like AND where it sits relative to the organ surface",
so edge-region prototypes will preferentially match query edge features and
centre-region prototypes will match query centre features.

The same sdf_map is passed through to cls_unit via the new `sdf_map` kwarg.
Everything else in cls_unit's interface is unchanged.

"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import MultiProtoAsConv, SDFBinEmbedding
from .backbone.torchvision_backbones import TVDeeplabRes101Encoder, TVDeeplabRes50Encoder
from .module import SDFHead

FG_PROT_MODE = 'gridconv+'
BG_PROT_MODE = 'gridconv'


FG_THRESH = 0.95
BG_THRESH = 0.95


class FewShotSeg(nn.Module):
    """
    ALPNet with geometry-aware prototype enrichment
    Input / output signature is identical to the original.
    """

    def __init__(self, in_channels=3, pretrained_path=None, cfg=None, sdf_criterion=None):
        super(FewShotSeg, self).__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        self.sdf_criterion = sdf_criterion
        self.get_encoder(in_channels)

        # ── Shape branch ───────────────────────────────────────────────────────
        shape_cfg = self.config.get('shape_branch', {})
        if shape_cfg.get('enabled', False):
            feat_dim   = shape_cfg.get('feat_dim', 256)
            K_bins     = shape_cfg.get('K_scale_classes', 10)
            self.sdf_head    = SDFHead(in_dim=feat_dim, K=K_bins)
            self._shape_enabled = True
            self._K_bins        = K_bins
            self._sdf_bin_emb = SDFBinEmbedding(
                feat_dim = feat_dim,
                K        = K_bins,
                hidden   = shape_cfg.get('geo_emb_hidden', 64),
            )

        else:
            self._shape_enabled = False
            self._sdf_bin_emb   = None

        proto_hw = self.config["proto_grid_size"]
        self.cls_unit = MultiProtoAsConv(
            proto_grid        = [proto_hw, proto_hw],
            feature_hw        = self.config["feature_hw"],
            sdf_bin_embedding = self._sdf_bin_emb,   # None when shape disabled self._sdf_bin_emb
        )

        # ── Load checkpoint ───────────────────────────────────────────────────
        if self.pretrained_path:
            ckpt = torch.load(self.pretrained_path, map_location='cpu')
            missing, unexpected = self.load_state_dict(ckpt, strict=False)
            if missing:
                print(f'[warn] missing keys: {missing}')
            if unexpected:
                print(f'[warn] unexpected keys: {unexpected}')
            print(f'###### Checkpoint loaded: {self.pretrained_path} ######')

    # ──────────────────────────────────────────────────────────────────────────
    def get_encoder(self, in_channels):
        if self.config['which_model'] == 'dlfcn_res101':
            use_coco_init = self.config['use_coco_init']
            self.encoder = TVDeeplabRes101Encoder(use_coco_init)
        elif self.config['which_model'] == 'dlfcn_res50':
            use_coco_init = self.config['use_coco_init']
            self.encoder = TVDeeplabRes50Encoder(use_coco_init)
        else:
            raise NotImplementedError(
                f'Backbone network {self.config["which_model"]} not implemented'
            )

    # ──────────────────────────────────────────────────────────────────────────
    def _build_sdf_map(self, sdf_logits: torch.Tensor) -> torch.Tensor:
        """
        Convert raw logits → per-pixel expected bin value at feature
        resolution, to be pooled inside MultiProtoAsConv.

        Args:
            sdf_logits: (nshots, out_K, h, w)
        Returns:
            sdf_map:    (nshots, h, w)  float, values in [0, K-1]
        """
        probs = F.softmax(sdf_logits, dim=1)      # (nshots, out_nshots, h, w)
        K_eff = probs.shape[1]
        bin_idx = torch.arange(K_eff, device=probs.device).float()
        expected_bin = (probs * bin_idx.view(1, K_eff, 1, 1)).sum(dim=1)  # (nshots, h, w) in [0, K_eff-1]
        return expected_bin

    # ──────────────────────────────────────────────────────────────────────────
    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs,
                isval, val_wsize, sdf_gt=None):
        """
        Identical signature to original FewShotSeg.forward().
        New kwargs:
            sdf_gt:  (1, nshots, H, W) int tensor;
                     None: skip shape loss
        Returns:
            output:      (N_queries × B, 1+n_ways, H, W)  segmentation logits
            align_loss:  scalar tensor
            sdf_loss:    scalar tensor
        """
        n_ways   = len(supp_imgs)
        n_shots  = len(supp_imgs[0])
        n_queries = len(qry_imgs)

        assert n_ways == 1, "Multi-way not yet implemented"
        assert n_queries == 1

        sup_bsize = supp_imgs[0][0].shape[0]
        img_size  = supp_imgs[0][0].shape[-2:]
        qry_bsize = qry_imgs[0].shape[0]
        assert sup_bsize == qry_bsize == 1

        # ── Encode all images in one forward pass ─────────────────────────────
        imgs_concat = torch.cat(
            [torch.cat(way, dim=0) for way in supp_imgs]
            + [torch.cat(qry_imgs, dim=0)],
            dim=0
        )
        img_fts  = self.encoder(imgs_concat, low_level=False)
        fts_size = img_fts.shape[-2:]

        supp_fts = img_fts[:n_ways * n_shots * sup_bsize].view(
            n_ways, n_shots, sup_bsize, -1, *fts_size)
        qry_fts  = img_fts[n_ways * n_shots * sup_bsize:].view(
            n_queries, qry_bsize, -1, *fts_size)

        self._last_query_feat = qry_fts[0,0].unsqueeze(0)  # (B, C, H, W)
        self._last_sup_feats = [supp_fts[0,s,0].unsqueeze(0) for s in range(n_shots)]  # list[(1,C,H,W)] len=K

        # ── bin prediction (shape branch) ─────────────────────────────────────
        if self._shape_enabled:
            supp_fts_flat        = img_fts[:n_ways * n_shots * sup_bsize]
            self._last_sdf_logits = self.sdf_head(supp_fts_flat)
            _sdf_map = self._build_sdf_map(self._last_sdf_logits)   # (nshots, h, w)
        else:
            _sdf_map = None

        # ── Mask preparation ──────────────────────────────────────────────────
        fore_mask = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0)
        fore_mask = torch.autograd.Variable(fore_mask, requires_grad=True)
        back_mask = torch.stack([torch.stack(way, dim=0) for way in back_mask], dim=0)

        align_loss = 0
        sdf_loss = torch.tensor(0.0, device=img_fts.device)
        outputs    = []

        for epi in range(1):
            res_fg_msk = torch.stack(
                [F.interpolate(fore_mask_w, size=fts_size, mode='bilinear')
                 for fore_mask_w in fore_mask], dim=0
            )
            res_bg_msk = torch.stack(
                [F.interpolate(back_mask_w, size=fts_size, mode='bilinear')
                 for back_mask_w in back_mask], dim=0
            )

            scores = []

            # ── Background prototype (gridconv, no geometry enrichment) ───────
            # because the SDF is defined only on foreground.
            _raw_bg, _ = self.cls_unit(
                qry_fts, supp_fts, res_bg_msk,
                mode=BG_PROT_MODE, thresh=BG_THRESH,
                isval=isval, val_wsize=val_wsize,
                sdf_map=None,
            )
            scores.append(_raw_bg)

            # ── Foreground prototype ───────────────────────────────────────────
            for way, _msk in enumerate(res_fg_msk):
                _use_mode = (FG_PROT_MODE
                             if F.avg_pool2d(_msk, 4).max() >= FG_THRESH
                             and FG_PROT_MODE != 'mask'
                             else 'mask')

                _raw_fg, _ = self.cls_unit(
                    qry_fts, supp_fts, _msk.unsqueeze(0),
                    mode=_use_mode, thresh=FG_THRESH,
                    isval=isval, val_wsize=val_wsize,
                    sdf_map=_sdf_map,
                )

                scores.append(_raw_fg)

            pred = torch.cat(scores, dim=1)              # (1, 1+n_ways, h, w)
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

            # ── Prototype alignment loss ───────────────────────────────────────
            if self.config['align'] and self.training:
                align_loss += self.alignLoss(
                    qry_fts[:, epi], pred,
                    supp_fts[:, :, epi],
                    fore_mask[:, :, epi],
                    back_mask[:, :, epi],
                )

            # ── Shape loss ────────────────────────────────────────────────────────
            if (self.sdf_criterion is not None
                    and self._shape_enabled
                    and sdf_gt is not None):
                sdf_gt_rs = sdf_gt[0].long()  # (nshots, H, W)
                seg_mask = fore_mask[0, :, 0].float()  # (nshots, H, W)

                sdf_logits_rs = F.interpolate(
                    self._last_sdf_logits.float(),  # (nshots, K_bins, h, w)
                    size=sdf_gt_rs.shape[-2:],
                    mode='bilinear', align_corners=False,
                )  # (nshots, K_bins, H, W)

                sdf_loss, _ = self.sdf_criterion(
                    logits=sdf_logits_rs,
                    sdf_gt=sdf_gt_rs,
                    seg_mask=seg_mask,
                )


        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])


        return output, align_loss / sup_bsize, sdf_loss

    # ──────────────────────────────────────────────────────────────────────────
    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Prototype alignment loss — identical to original.
        (Geometry enrichment is NOT applied here: the align loss should
         reflect pure appearance matching to avoid circular optimisation.)
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        pred_mask    = pred.argmax(dim=1).unsqueeze(0)
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways    = []

        qry_fts = qry_fts.unsqueeze(0).unsqueeze(2)

        loss = []
        for way in range(n_ways):
            if way in skip_ways:
                continue
            for shot in range(n_shots):
                img_fts = supp_fts[way: way + 1, shot: shot + 1]

                qry_pred_fg_msk = F.interpolate(
                    binary_masks[way + 1].float(), size=img_fts.shape[-2:],
                    mode='bilinear'
                )
                qry_pred_bg_msk = F.interpolate(
                    binary_masks[0].float(), size=img_fts.shape[-2:],
                    mode='bilinear'
                )

                scores = []

                # Background — no SDF enrichment in align loss
                _raw_score_bg, _ = self.cls_unit(
                    qry=img_fts, sup_x=qry_fts,
                    sup_y=qry_pred_bg_msk.unsqueeze(-3),
                    mode=BG_PROT_MODE, thresh=BG_THRESH,
                    sdf_map=None,
                )
                scores.append(_raw_score_bg)

                # Foreground — no SDF enrichment in align loss
                _use_mode = (FG_PROT_MODE
                             if F.avg_pool2d(qry_pred_fg_msk, 4).max() >= FG_THRESH
                             and FG_PROT_MODE != 'mask'
                             else 'mask')
                _raw_score_fg, _ = self.cls_unit(
                    qry=img_fts, sup_x=qry_fts,
                    sup_y=qry_pred_fg_msk.unsqueeze(-3),
                    mode=_use_mode, thresh=FG_THRESH,
                    sdf_map=None,
                )
                scores.append(_raw_score_fg)

                supp_pred = torch.cat(scores, dim=1)
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:],
                                          mode='bilinear')

                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0

                loss.append(
                    F.cross_entropy(supp_pred, supp_label[None, ...],
                                    ignore_index=255) / n_shots / n_ways
                )

        return torch.sum(torch.stack(loss))
