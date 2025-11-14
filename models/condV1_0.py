import timm
import torch
from torch import einsum, nn
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.misc import FrozenBatchNorm2d
from einops import rearrange
#from models.custom_modules.cond_detr_decoder_layer import ConditionalDecoderLayer
from models.custom_modules.cond_detr_decoder_layerV2_0 import ConditionalDecoderLayer
import torch.nn.functional as F
from einops import rearrange, repeat

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
        # ✅ Double the previous downsampled feature (spatially upsample by 2×)
        prev_down_up = F.interpolate(prev_down, scale_factor=2.0, mode='bilinear', align_corners=False)
        
        # ✅ Match current feature spatial size
        prev_down_up = F.interpolate(prev_down_up, size=curr.shape[-2:], mode='bilinear', align_corners=False)
        
        # ✅ Concatenate and process
        x = torch.cat([prev_down_up, curr], dim=1)
        x = self.block(x)
        return x

class UpBlock(nn.Module):
    """
    Concatenate x and skip, then downsample by 2×.
    No addition.
    """
    def __init__(self, in_ch, skip_ch, out_ch, mode="down"):
        super().__init__()
        # After concatenation, total channels = in_ch + skip_ch
        if mode=="nodown":
            self.block = nn.Sequential(
                ConvBNReLU(in_ch + skip_ch, out_ch, stride=1),  # No downsample
                ConvBNReLU(out_ch, out_ch)
            )
        else:
            self.block = nn.Sequential(
            ConvBNReLU(in_ch + skip_ch, out_ch, stride=2),  # ↓ Downsample
            ConvBNReLU(out_ch, out_ch)
        )

    def forward(self, x, skip):
        # 🔹 Match spatial size
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        # 🔗 Concatenate (no addition)
        fused = torch.cat([x, skip], dim=1)

        # 🔽 Downsample by 2×
        out = self.block(fused)
        return out

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

# tensor helpers

def create_grid_like(t, dim = 0):
    h, w, device = *t.shape[-2:], t.device

    grid = torch.stack(torch.meshgrid(
        torch.arange(w, device = device),
        torch.arange(h, device = device),
    indexing = 'xy'), dim = dim)

    grid.requires_grad = False
    grid = grid.type_as(t)
    return grid

def normalize_grid(grid, dim = 1, out_dim = -1):
    # normalizes a grid to range from -1 to 1
    h, w = grid.shape[-2:]
    grid_h, grid_w = grid.unbind(dim = dim)

    grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
    grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0

    return torch.stack((grid_h, grid_w), dim = out_dim)

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

# continuous positional bias from SwinV2

class CPB(nn.Module):
    """ https://arxiv.org/abs/2111.09883v1 """

    def __init__(self, dim, *, heads, offset_groups, depth):
        super().__init__()
        self.heads = heads
        self.offset_groups = offset_groups

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU()
            ))

        self.mlp.append(nn.Linear(dim, heads // offset_groups))

    def forward(self, grid_q, grid_kv):
        device, dtype = grid_q.device, grid_kv.dtype

        grid_q = rearrange(grid_q, 'h w c -> 1 (h w) c')
        grid_kv = rearrange(grid_kv, 'b h w c -> b (h w) c')

        pos = rearrange(grid_q, 'b i c -> b i 1 c') - rearrange(grid_kv, 'b j c -> b 1 j c')
        bias = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            bias = layer(bias)

        bias = rearrange(bias, '(b g) i j o -> b (g o) i j', g = self.offset_groups)

        return bias

# main class
##DeformableEncoderLayer class
class DeformableLayer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        downsample_factor = 4,
        offset_scale = None,
        offset_groups = None,
        offset_kernel_size = 6,
        group_queries = True,
        group_key_values = True
    ):
        super().__init__()
        offset_scale = default(offset_scale, downsample_factor)
        assert offset_kernel_size >= downsample_factor, 'offset kernel size must be greater than or equal to the downsample factor'
        assert divisible_by(offset_kernel_size - downsample_factor, 2)

        offset_groups = default(offset_groups, heads)
        assert divisible_by(heads, offset_groups)

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.offset_groups = offset_groups

        offset_dims = inner_dim // offset_groups

        self.downsample_factor = downsample_factor

        self.to_offsets = nn.Sequential(
            nn.Conv2d(offset_dims, offset_dims, offset_kernel_size, groups = offset_dims, stride = downsample_factor, padding = (offset_kernel_size - downsample_factor) // 2),
            nn.GELU(),
            nn.Conv2d(offset_dims, 2, 1, bias = False),
            nn.Tanh(),
            Scale(offset_scale)
        )

        self.rel_pos_bias = CPB(dim // 4, offset_groups = offset_groups, heads = heads, depth = 2)

        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_queries else 1, bias = False)
        self.to_k = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        self.to_v = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x, return_vgrid = False):
        """
        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        g - offset groups
        """

        heads, b, h, w, downsample_factor, device = self.heads, x.shape[0], *x.shape[-2:], self.downsample_factor, x.device

        # queries

        q = self.to_q(x)

        # calculate offsets - offset MLP shared across all groups

        group = lambda t: rearrange(t, 'b (g d) ... -> (b g) d ...', g = self.offset_groups)

        grouped_queries = group(q)
        offsets = self.to_offsets(grouped_queries)

        # calculate grid + offsets

        grid =create_grid_like(offsets)
        vgrid = grid + offsets

        vgrid_scaled = normalize_grid(vgrid)

        kv_feats = F.grid_sample(
            group(x),
            vgrid_scaled,
        mode = 'bilinear', padding_mode = 'zeros', align_corners = False)

        kv_feats = rearrange(kv_feats, '(b g) d ... -> b (g d) ...', b = b)

        # derive key / values

        k, v = self.to_k(kv_feats), self.to_v(kv_feats)

        # scale queries

        q = q * self.scale

        # split out heads

        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = heads), (q, k, v))

        # query / key similarity

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # relative positional bias

        grid = create_grid_like(x)
        grid_scaled = normalize_grid(grid, dim = 0)
        rel_pos_bias = self.rel_pos_bias(grid_scaled, vgrid_scaled)
        sim = sim + rel_pos_bias

        # numerical stability

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate and combine heads

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        out = self.to_out(out)

        if return_vgrid:
            return out, vgrid

        return out

class DeformableEncoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        # ✅ Deformable self-attention replacing standard attention
        self.self_attn = DeformableLayer(
            dim=d_model,
            heads=n_heads,
            dim_head=d_model // n_heads,
            dropout=0.,
            downsample_factor=4,      # adjust depending on feature size
            offset_scale=4.0,
            offset_groups=None,  # more groups = more flexible offsets
            offset_kernel_size = 6,      # offset kernel size
        )

        # ✅ Feed-forward network (same as DETR)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # ✅ Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # ✅ Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # ✅ Activation function
        self.activation = nn.ReLU(inplace=True)

    def forward(self, src):
        """
        src: (B, N, C) — same as transformer encoder input
        but we reshape it to (B, C, H, W) for 2D deformable attention
        """
        B, N, C = src.shape
        H = W = int(N ** 0.5)  # assume square spatial grid

        # ---- 1. LayerNorm + Reshape ----
        src2 = self.norm1(src)
        src2 = src2.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)

        # ---- 2. Deformable Attention ----
        attn_out = self.self_attn(src2)  # (B, C, H, W) ##for encoder there is no QKV

        # ---- 3. Flatten and Residual ----
        attn_out = attn_out.flatten(2).transpose(1, 2)  # (B, N, C)
        src = src + self.dropout1(attn_out)

        # ---- 4. Feed-forward Network ----
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src

class ConditionalDETR(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_classes=92,
        n_tokens=225,
        n_layers=6,
        n_heads=8,
        n_queries=100,
        use_frozen_bn=False,
        use_deformable=True
    ):
        super().__init__()

        # self.backbone = create_feature_extractor(
        #     torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True),
        #     return_nodes={"layer4": "layer4"},
        # )
        self.backbone = timm.create_model(
                "resnet101",
                pretrained=True,
                features_only=True,      # only output feature maps
                out_indices=(2,3,4),   # choose which layers to extract
                global_pool=""           # disable global pooling
                )
        channels = self.backbone.feature_info.channels()
        if use_frozen_bn:
            self.replace_batchnorm(self.backbone)

        self.conv1x1 = nn.Conv2d(2048, d_model, kernel_size=1, stride=1)
        self.pe_encoder = nn.Parameter(
            torch.rand((1, n_tokens, d_model)), requires_grad=True
        )
        self.use_deformable = use_deformable
        if self.use_deformable:
            #self.transformer_encoder = nn.TransformerEncoder(
            #    nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, 0.1, batch_first=True),
            #    num_layers=3
            #)
            self.transformer_encoder_ = nn.ModuleList([
            DeformableEncoderLayer(d_model,n_heads,4 * d_model, 0.1) for _ in range(n_layers)
        ])
            
        else:
            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, 0.1, batch_first=True),
                num_layers=n_layers
            )
        self.queries = nn.Parameter(
            torch.rand((1, n_queries, d_model)), requires_grad=True
        )

        self.decoder_layers = nn.ModuleList(
            [ConditionalDecoderLayer(d_model, n_heads) for _ in range(n_layers)]
        )

        self.linear_class = nn.Linear(d_model, n_classes)
        self.linear_bbox = nn.Linear(d_model, 4)
        # Query generation (content + spatial top-k)
        self.num_content_queries = n_queries//2
        self.content_queries = nn.Parameter(torch.randn(n_queries//2, d_model))
        self.spatial_score = nn.Linear(d_model, 1)
        self.topk_spatial = n_queries//2
        self.box_init = nn.Linear(d_model, 4)
        self.pool2 = nn.Identity()  # same dimension
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)  # downsample ×2
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)  # downsample ×4
        
        # Project Swin stages to d_model
        self.conv2 = nn.Conv2d(channels[0], d_model, kernel_size=1)
        self.conv3 = nn.Conv2d(channels[1], d_model, kernel_size=1)
        self.conv4 = nn.Conv2d(channels[2], d_model, kernel_size=1)

        # Down path
        self.down3 = DownBlock(d_model, d_model, d_model)  # p3 + d2
        self.down4 = DownBlock(d_model, d_model, d_model)  # p4 + d3

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBNReLU(d_model, d_model),
            ConvBNReLU(d_model, d_model)
        )

        # Up path
        self.up4 = UpBlock(d_model, d_model, d_model)
        self.up3 = UpBlock(d_model, d_model, d_model)
        self.up2 = UpBlock(d_model, d_model, d_model, mode="nodown")

        # Final fusion of upsampled features
        self.fusion_proj = ConvBNReLU(3 * d_model, d_model)
        
        self.content_ref_embed = nn.Embedding(self.num_content_queries, 4)  # (cx, cy, w, h)
        self.spatial_ref_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4)
        )
    def forward(self, x):
        #tokens = self.backbone(x)["layer4"]
        #print(tokens.shape)
        #tokens = self.conv1x1(tokens)
        #tokens = rearrange(tokens, "b c h w -> b (h w) c")
        features = self.backbone(x)
        for i in range(0, 3): #1 to 4
            if features[i].shape[1] < features[i].shape[-1]:  # channels last
                features[i] = features[i].permute(0, 3, 1, 2).contiguous()
        # Feature projection using Unet
        p2 = self.conv2(features[0]) #60*60
        p3 = self.conv3(features[1]) #30*30
        p4 = self.conv4(features[2]) #15*15
        # Down path
        d2 = p4 
        d3 = self.down3(d2, p3) ##it should be up 30*30
        d4 = self.down4(d3, p2) ##it should be up 60*60
        # Bottleneck
        bn = self.bottleneck(d4)
        # Up path
        up4 = self.up4(bn, d4) ##it should be down 30*30
        up3 = self.up3(up4, d3) ##it should be down 15*15
        up2 = self.up2(up3, d2) ##it should be down 15*15
        # Multi-scale upsample fusion
        H, W = up4.shape[-2:]
        up4_up = F.interpolate(up4, size=(H // 2, W // 2), mode='bilinear', align_corners=False)
        up4_up = F.interpolate(up4_up, size=up2.shape[-2:], mode='bilinear', align_corners=False)
        up3_up = F.interpolate(up3, size=up2.shape[-2:], mode='bilinear', align_corners=False)
        cat = torch.cat([up4_up, up3_up, up2], dim=1)  # [B, 3*d_model, H, W] should be 15*15
        # Final projection
        out = self.fusion_proj(cat) ##size should be 15*15
        tokens = out.flatten(2).transpose(1, 2)        # [B, H*W, D]
        B,N,D = tokens.shape
        memory_in=tokens+self.pe_encoder
        if self.use_deformable:
            #print(N,N**0.5,N**0.5)
            #memory_in = memory_in.transpose(1, 2)  # [B, D, N]
            #memory_in = rearrange(memory_in, 'b q (h w) -> b q h w', h=int(N**0.5), w=int(N**0.5))
            #memory_in=self.transformer_encoder(memory_in) ##wait and watch
            for layer in self.transformer_encoder_:
                memory_in = layer(memory_in)
            memory = memory_in
        else:
            memory = self.transformer_encoder(memory_in)

        #object_queries = self.queries.repeat(memory.shape[0], 1, 1) ##original
        # ===== 2️⃣ Query generation =====
        # Content queries (learned embeddings)
        content_q = self.content_queries.unsqueeze(0).expand(B, -1, -1)   # [B, Qc, D]

        # Spatial queries: select top-k informative regions
        scores = self.spatial_score(memory).squeeze(-1)                   # [B, N]
        topk = min(self.topk_spatial, scores.shape[1])
        topk_vals, topk_idx = torch.topk(scores, k=topk, dim=1)
        batch_idx = torch.arange(B, device=tokens.device).unsqueeze(1).expand(-1, topk)
        spatial_feats = memory[batch_idx, topk_idx]                       # [B, topk, D]
        spatial_q = spatial_feats                                         # [B, topk, D]

        # ===== 3️⃣ Combine content + spatial queries =====
        queries = torch.cat([content_q, spatial_q], dim=1)                # [B, Q_total, D]
        # ====== 😈😎 Reference boxes ======
        # Content reference boxes: learned embeddings, shared across images
        content_ref_boxes = torch.sigmoid(self.content_ref_embed.weight)
        content_ref_boxes = content_ref_boxes.unsqueeze(0).expand(B, -1, -1) #[B, Qc, 4]
        spatial_ref_boxes = torch.sigmoid(self.spatial_ref_head(spatial_q))  # [B, topk, 4]
        ref_boxes = torch.cat([content_ref_boxes, spatial_ref_boxes], dim=1)  # [B, Q_total, 4]
        
        # Object queries are the same for the first decoder layer as decoder embeddings
        decoder_embeddings = queries
        #object_queries = queries
        class_preds, bbox_preds = [], []
        cross_attn_weights = []
        #decoder_embeddings = object_queries=tgt(target)
        #memory = src (source)
        #ref_boxes = ref
        ##here from feature map (memory)(fixed in all iteration) is maped to the decoder_embedding with the help of reference box 
        #both decoder_embedding and reference boxes are updated iteratively.
        for layer in self.decoder_layers:
            decoder_embeddings, ref_boxes, cross_attn_weight = layer(
                decoder_embeddings, ref_boxes, memory
            )
            cross_attn_weights.append(cross_attn_weight)
            class_preds.append(self.linear_class(decoder_embeddings))
            bbox_preds.append(self.linear_bbox(decoder_embeddings) + ref_boxes)

        return torch.stack(class_preds, dim=1), torch.stack(bbox_preds, dim=1), torch.stack(cross_attn_weights, dim=1)

    @staticmethod
    def replace_batchnorm(module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                frozen_bn = FrozenBatchNorm2d(child.num_features)
                frozen_bn.weight.data = child.weight.data
                frozen_bn.bias.data = child.bias.data
                frozen_bn.running_mean.data = child.running_mean.data
                frozen_bn.running_var.data = child.running_var.data
                setattr(module, name, frozen_bn)
            else:
                ConditionalDETR.replace_batchnorm(child)
