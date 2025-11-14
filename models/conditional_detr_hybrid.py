import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision.ops.misc import FrozenBatchNorm2d
import timm
from einops import rearrange
from models.custom_modules.cond_detr_decoder_layerV2_0 import ConditionalDecoderLayer


# ============================================================
# 🔹 Basic Utility Blocks
# ============================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DownBlock(nn.Module):
    """Concat doubled previous downsampled feature + current feature, then downsample"""
    def __init__(self, prev_ch, curr_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(prev_ch + curr_ch, out_ch, stride=1),
            ConvBNReLU(out_ch, out_ch)
        )

    def forward(self, prev_down, curr):
        prev_down_up = F.interpolate(prev_down, scale_factor=2.0, mode='bilinear', align_corners=False)
        prev_down_up = F.interpolate(prev_down_up, size=curr.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([prev_down_up, curr], dim=1)
        return self.block(x)


class UpBlock(nn.Module):
    """Concatenate x and skip, then downsample (or not)."""
    def __init__(self, in_ch, skip_ch, out_ch, mode="down"):
        super().__init__()
        if mode == "nodown":
            self.block = nn.Sequential(
                ConvBNReLU(in_ch + skip_ch, out_ch, stride=1),
                ConvBNReLU(out_ch, out_ch)
            )
        else:
            self.block = nn.Sequential(
                ConvBNReLU(in_ch + skip_ch, out_ch, stride=2),
                ConvBNReLU(out_ch, out_ch)
            )

    def forward(self, x, skip):
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        fused = torch.cat([x, skip], dim=1)
        return self.block(fused)


# ============================================================
# 🔹 Deformable Attention Utilities
# ============================================================

def create_grid_like(t, dim=0):
    h, w, device = *t.shape[-2:], t.device
    grid = torch.stack(torch.meshgrid(
        torch.arange(w, device=device),
        torch.arange(h, device=device),
        indexing='xy'
    ), dim=dim)
    grid.requires_grad = False
    return grid.type_as(t)


def normalize_grid(grid, dim=1, out_dim=-1):
    h, w = grid.shape[-2:]
    grid_h, grid_w = grid.unbind(dim=dim)
    grid_h = 2.0 * grid_h / max(h - 1, 1e-5) - 1.0
    grid_w = 2.0 * grid_w / max(w - 1, 1e-5) - 1.0
    return torch.stack((grid_h, grid_w), dim=out_dim)


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x): return x * self.scale


class CPB(nn.Module):
    """Continuous positional bias (SwinV2)."""
    def __init__(self, dim, *, heads, offset_groups, depth):
        super().__init__()
        self.heads = heads
        self.offset_groups = offset_groups
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Sequential(nn.Linear(2, dim), nn.ReLU()))
        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(nn.Linear(dim, dim), nn.ReLU()))
        self.mlp.append(nn.Linear(dim, heads // offset_groups))

    def forward(self, grid_q, grid_kv):
        grid_q = rearrange(grid_q, 'h w c -> 1 (h w) c')
        grid_kv = rearrange(grid_kv, 'b h w c -> b (h w) c')
        pos = rearrange(grid_q, 'b i c -> b i 1 c') - rearrange(grid_kv, 'b j c -> b 1 j c')
        bias = torch.sign(pos) * torch.log(pos.abs() + 1)
        for layer in self.mlp: bias = layer(bias)
        return rearrange(bias, '(b g) i j o -> b (g o) i j', g=self.offset_groups)


class DeformableLayer(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, dropout=0., downsample_factor=4,
                 offset_scale=None, offset_groups=None, offset_kernel_size=6):
        super().__init__()
        offset_scale = offset_scale or downsample_factor
        offset_groups = offset_groups or heads
        inner_dim = dim_head * heads
        offset_dims = inner_dim // offset_groups
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.offset_groups = offset_groups
        self.downsample_factor = downsample_factor

        self.to_offsets = nn.Sequential(
            nn.Conv2d(offset_dims, offset_dims, offset_kernel_size, groups=offset_dims,
                      stride=downsample_factor, padding=(offset_kernel_size - downsample_factor)//2),
            nn.GELU(), nn.Conv2d(offset_dims, 2, 1, bias=False), nn.Tanh(), Scale(offset_scale)
        )
        self.rel_pos_bias = CPB(dim // 4, offset_groups=offset_groups, heads=heads, depth=2)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Conv2d(dim, inner_dim, 1)
        self.to_k = nn.Conv2d(dim, inner_dim, 1)
        self.to_v = nn.Conv2d(dim, inner_dim, 1)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x):
        b, _, h, w = x.shape
        q = self.to_q(x)
        group = lambda t: rearrange(t, 'b (g d) ... -> (b g) d ...', g=self.offset_groups)
        grouped_q = group(q)
        offsets = self.to_offsets(grouped_q)
        grid = create_grid_like(offsets)
        vgrid_scaled = normalize_grid(grid + offsets)
        kv_feats = F.grid_sample(group(x), vgrid_scaled, mode='bilinear',
                                 padding_mode='zeros', align_corners=False)
        kv_feats = rearrange(kv_feats, '(b g) d ... -> b (g d) ...', b=b)
        k, v = self.to_k(kv_feats), self.to_v(kv_feats)
        q = (q * self.scale)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h=self.heads), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        rel_bias = self.rel_pos_bias(normalize_grid(create_grid_like(x), dim=0), vgrid_scaled)
        sim = sim + rel_bias - sim.amax(dim=-1, keepdim=True).detach()
        attn = self.dropout(sim.softmax(dim=-1))
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class DeformableEncoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = DeformableLayer(dim=d_model, heads=n_heads, dim_head=d_model // n_heads)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, src):
        B, N, C = src.shape
        H = W = int(math.sqrt(N))
        src2 = self.norm1(src).transpose(1, 2).reshape(B, C, H, W)
        attn_out = self.self_attn(src2).flatten(2).transpose(1, 2)
        src = src + self.dropout1(attn_out)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        return src + self.dropout2(src2)


# ============================================================
# 🔹 ReID + Temporal Memory
# ============================================================

class ReIDMemory:
    def __init__(self, dim=256, max_size=1000, momentum=0.8, device='cpu'):
        self.dim, self.max_size, self.momentum, self.device = dim, max_size, momentum, device
        self.features, self.boxes, self.ids, self.ts = [], [], [], []

    @torch.no_grad()
    def match_and_update(self, new_feats, new_boxes, new_ids, timestamp=None):
        smoothed = []
        for feat, box, pid in zip(new_feats, new_boxes, new_ids):
            pid = int(pid)
            feat, box = feat.detach(), box.detach()
            if pid in self.ids:
                idx = self.ids.index(pid)
                self.features[idx] = self.momentum * self.features[idx] + (1 - self.momentum) * feat
                self.boxes[idx], self.ts[idx] = box, timestamp
            else:
                self.ids.append(pid)
                self.features.append(feat)
                self.boxes.append(box)
                self.ts.append(timestamp)
            smoothed.append(self.features[-1])
            if len(self.features) > self.max_size:
                for arr in (self.features, self.boxes, self.ids, self.ts): arr.pop(0)
        return torch.stack(smoothed, dim=0)

    def clear(self): self.features, self.boxes, self.ids, self.ts = [], [], [], []


class TemporalQueryMemory:
    def __init__(self, max_persist=50):
        self.queries, self.ref_boxes, self.max_persist = None, None, max_persist

    @torch.no_grad()
    def get(self): return (self.queries, self.ref_boxes)

    @torch.no_grad()
    def update(self, queries, ref_boxes, temporal_k=None):
        if queries is None or ref_boxes is None: return
        if queries.dim() == 2: queries = queries.unsqueeze(0)
        if ref_boxes.dim() == 2: ref_boxes = ref_boxes.unsqueeze(0)
        q, b = queries.detach(), ref_boxes.detach()
        if self.queries is None:
            self.queries, self.ref_boxes = q, b
        else:
            self.queries = torch.cat([self.queries, q], dim=1)
            self.ref_boxes = torch.cat([self.ref_boxes, b], dim=1)
            if self.queries.shape[1] > self.max_persist:
                self.queries = self.queries[:, -self.max_persist:, :].detach()
                self.ref_boxes = self.ref_boxes[:, -self.max_persist:, :].detach()


# ============================================================
# 🔹 CONDITIONAL DETR (Hybrid MOTRv2)
# ============================================================

class ConditionalDETR(nn.Module):
    def __init__(self, d_model=256, n_classes=2,n_reid_classes=1000, n_tokens=225, n_layers=6, n_heads=8, n_queries=100,
                 use_deformable=True, use_reid=True, reid_dim=256, use_reid_classifier=False,
                 use_temporal=True, temporal_k=5, memory_max_size=1000, memory_momentum=0.8, device='cpu',use_frozen_bn=False):
        super().__init__()
        self.device, self.use_temporal, self.use_reid = device, use_temporal, use_reid
        self.use_deformable = use_deformable
        self.topk_spatial = n_queries // 2
        
        # Backbone
        self.backbone = timm.create_model("resnet101", pretrained=True, features_only=True, out_indices=(2,3,4),global_pool="")
        ch = self.backbone.feature_info.channels()
        self.conv2, self.conv3, self.conv4 = nn.Conv2d(ch[0], d_model, 1), nn.Conv2d(ch[1], d_model, 1), nn.Conv2d(ch[2], d_model, 1)
        self.pe_encoder = nn.Parameter(torch.rand((1, n_tokens, d_model)), requires_grad=True)
        
        # UNet aggregation
        self.down3, self.down4 = DownBlock(d_model, d_model, d_model), DownBlock(d_model, d_model, d_model)
        self.up4, self.up3, self.up2 = UpBlock(d_model, d_model, d_model), UpBlock(d_model, d_model, d_model), UpBlock(d_model, d_model, d_model, mode="nodown")
        self.fusion_proj = ConvBNReLU(3 * d_model, d_model)
        self.bottleneck = nn.Sequential(ConvBNReLU(d_model, d_model), ConvBNReLU(d_model, d_model))

        # Transformer
        self.transformer_encoder_ = nn.ModuleList([DeformableEncoderLayer(d_model, n_heads) for _ in range(n_layers)]) if use_deformable else nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, 0.1, batch_first=True),num_layers=n_layers)
        self.decoder_layers = nn.ModuleList([ConditionalDecoderLayer(d_model, n_heads) for _ in range(n_layers)])
        self.linear_class, self.linear_bbox = nn.Linear(d_model, n_classes), nn.Linear(d_model, 4)

        # Queries
        self.num_content_queries = n_queries // 2
        self.content_queries = nn.Parameter(torch.randn(n_queries // 2, d_model))
        self.spatial_score, self.content_ref_embed = nn.Linear(d_model, 1), nn.Embedding(self.num_content_queries, 4)
        self.content_ref_embed = nn.Embedding(self.num_content_queries, 4) 
        self.spatial_ref_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4))
        self.reid_dim = reid_dim
        
        # ReID + Memory
        if use_reid:
            self.reid_embed_head = nn.Sequential(nn.Linear(d_model, reid_dim), nn.BatchNorm1d(reid_dim), nn.ReLU(inplace=True))
            if use_reid_classifier: 
                self.reid_classifier = nn.Linear(reid_dim, max(1000, n_reid_classes))
        self.reid_memory = ReIDMemory(dim=reid_dim, max_size=memory_max_size, momentum=memory_momentum, device=device) if use_temporal else None

        # Temporal modules
        self.temporal_memory = TemporalQueryMemory()
        self.temporal_update = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.gate = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.memory_proj = nn.Sequential(nn.Linear(reid_dim + 6, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, d_model))
        self.motion_proj = nn.Sequential(nn.Linear(2, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, d_model))
        self.temporal_memory_bank = []
        self.temporal_k = temporal_k

    # ============================================================
    def forward(self, x, return_reid=False, memory_cond=False):
        B = x.shape[0]
        features = self.backbone(x)
        for i in range(0, 3): #1 to 4
            if features[i].shape[1] < features[i].shape[-1]:  # channels last
                features[i] = features[i].permute(0, 3, 1, 2).contiguous()
        p2, p3, p4 = self.conv2(features[0]), self.conv3(features[1]), self.conv4(features[2])
        d2, d3, d4 = p4, self.down3(p4, p3), self.down4(p3, p2)
        bn = self.bottleneck(d4)
        up4 = self.up4(bn, d4) 
        up3 = self.up3(up4, d3) 
        up2 = self.up2(up3, d2)
        H, W = up4.shape[-2:]
        up4_up = F.interpolate(up4, size=(H // 2, W // 2), mode='bilinear', align_corners=False)
        up4_up = F.interpolate(up4_up, size=up2.shape[-2:], mode='bilinear', align_corners=False)
        up3_up = F.interpolate(up3, size=up2.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.fusion_proj(torch.cat([up4_up, up3_up, up2], dim=1))
        tokens = fused.flatten(2).transpose(1, 2)
        tokens=tokens+self.pe_encoder
        if self.use_deformable:
            for layer in self.transformer_encoder_:
                memory_in = layer(tokens)
            memory = memory_in
        else:
            memory = self.transformer_encoder(tokens)

        # Queries
        content_q = self.content_queries.unsqueeze(0).expand(B, -1, -1) # [B, Qc, D]
        # Spatial queries: select top-k informative regions
        scores = self.spatial_score(memory).squeeze(-1) # [B, N]
        topk = min(self.topk_spatial, scores.shape[1])
        topk_vals, topk_idx = torch.topk(scores, k=topk, dim=1)
        batch_idx = torch.arange(B, device=tokens.device).unsqueeze(1).expand(-1, topk)
        spatial_q = memory[batch_idx, topk_idx] # [B, topk, D]
        #print(spatial_q.shape)
        #print(content_q.shape)
        queries = torch.cat([content_q, spatial_q], dim=1) # [B, Q_total, D]
        content_ref = torch.sigmoid(self.content_ref_embed.weight).unsqueeze(0).expand(B, -1, -1) #[B, Qc, 4]
        spatial_ref = torch.sigmoid(self.spatial_ref_head(spatial_q)) # [B, topk, 4]
        ref_boxes = torch.cat([content_ref, spatial_ref], dim=1) # [B, Q_total, 4]
        #print(queries.shape)
        # Temporal conditioning
        if self.use_temporal and memory_cond:
            pq, pb = self.temporal_memory.get()
            if pq is not None and pb is not None:
                if pq.dim() == 2: pq, pb = pq.unsqueeze(0), pb.unsqueeze(0)
                k_use = min(pq.shape[1], self.temporal_k)
                pq_use, pb_use = pq[:, -k_use:, :], pb[:, -k_use:, :]
                ref_slice = queries[:, :k_use, :] if queries.shape[1] >= k_use else F.pad(queries, (0,0,0,k_use-queries.shape[1]))
                gate = self.gate(pq_use)
                pq_new = (1 - gate) * pq_use + gate * self.temporal_update(torch.cat([pq_use, ref_slice], dim=-1))
                queries, ref_boxes = torch.cat([queries, pq_new], dim=1), torch.cat([ref_boxes, pb_use], dim=1)
                #print(queries.shape,ref_boxes.shape)
                if queries.shape[1] > 100:
                    queries, ref_boxes = queries[:, -100:, :], ref_boxes[:, -100:, :]

        # Decoder
        decoder_embeddings, class_preds, bbox_preds = queries, [], []
        cross_attn_weights = []
        for layer in self.decoder_layers:
            decoder_embeddings, ref_boxes, cross_attn_weight = layer(decoder_embeddings, ref_boxes, memory)
            cross_attn_weights.append(cross_attn_weight)
            class_preds.append(self.linear_class(decoder_embeddings))
            bbox_preds.append(self.linear_bbox(decoder_embeddings) + ref_boxes)

        # ReID
        reid_embeds = None
        if self.use_reid:
            #print(decoder_embeddings.shape)
            flat = decoder_embeddings.reshape(-1, decoder_embeddings.shape[-1])
            #print(flat.shape)
            reid_proj = self.reid_embed_head(flat).reshape(B, decoder_embeddings.shape[1], -1)
            reid_embeds = F.normalize(reid_proj, p=2, dim=-1)
            

        # Update temporal memory
        if self.use_temporal:
            n_persist = min(self.num_content_queries // 2, decoder_embeddings.shape[1])
            self.temporal_memory.update(decoder_embeddings[:, -n_persist:, :].detach(),
                                        ref_boxes[:, -n_persist:, :].detach(),
                                        temporal_k=self.temporal_k)

        if return_reid:
            return torch.stack(class_preds, dim=1), torch.stack(bbox_preds, dim=1), reid_embeds
        return torch.stack(class_preds, dim=1), torch.stack(bbox_preds, dim=1)

    # ============================================================
    @torch.no_grad()
    def update_memory(self, memory_feat):
        if memory_feat.dim() == 1:
            memory_feat = memory_feat.unsqueeze(0)
        elif memory_feat.dim() > 2:
            memory_feat = memory_feat.view(-1, memory_feat.shape[-1])
        projected = self.memory_proj(memory_feat.to(self.device))
        if len(self.temporal_memory_bank) > 50:
            self.temporal_memory_bank.pop(0)
        self.temporal_memory_bank.append(projected.mean(0, keepdim=True).detach())

    def get_temporal_memory(self, k=10):
        if not self.temporal_memory_bank:
            return torch.zeros(1, self.memory_proj[-1].out_features, device=self.device)
        return torch.cat(self.temporal_memory_bank[-k:], dim=0)

    def freeze_non_reid(self):
        for n, p in self.named_parameters(): p.requires_grad = ('reid' in n)
