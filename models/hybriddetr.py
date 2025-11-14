import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from torchvision.models.feature_extraction import create_feature_extractor
class PositionEmbeddingSine(nn.Module):
    """
    Standard 2D sine-cosine positional encoding from DETR.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, x):
        """
        x: [B, C, H, W] feature map
        return: [B, H*W, 2*num_pos_feats]
        """
        B, C, H, W = x.shape
        mask = torch.zeros(B, H, W, device=x.device, dtype=torch.bool)

        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)  # [B, H, W, 2*num_pos_feats]

        return pos.flatten(1, 2)  # [B, H*W, D]
        
class PositionEmbeddingLearned(nn.Module):
    """
    Absolute learnable 2D positional embeddings.
    """
    def __init__(self, num_pos_feats=256, max_h=64, max_w=64):
        super().__init__()
        self.row_embed = nn.Embedding(max_h, num_pos_feats)
        self.col_embed = nn.Embedding(max_w, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, H*W, D]
        """
        B, C, H, W = x.shape
        i = torch.arange(W, device=x.device)
        j = torch.arange(H, device=x.device)
        x_emb = self.col_embed(i)  # [W, D]
        y_emb = self.row_embed(j)  # [H, D]
        pos = torch.cat([
            y_emb.unsqueeze(1).expand(H, W, -1),
            x_emb.unsqueeze(0).expand(H, W, -1)
        ], dim=-1)  # [H, W, 2*D]
        pos = pos.reshape(H * W, -1)
        pos = pos.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, 2*D]
        return pos

class SlotAttention(nn.Module):
    """
    Slot Attention module: aggregates N input features into K slot embeddings.
    Reference: Locatello et al., 2020
    """

    def __init__(self, num_slots=6, dim=256, iters=3):
        super().__init__()
        self.num_slots = num_slots      # number of slots
        self.iters = iters              # number of attention iterations
        self.scale = dim ** -0.5        # scaling factor for attention

        # Slot initialization parameters
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, num_slots, dim))

        # Linear maps for attention
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # Slot update GRU
        self.gru = nn.GRUCell(dim, dim)

        # Feed-forward module after GRU
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim)
        )

        # Layer norms
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: [B, N, D] input features (flattened spatial map)
        returns: [B, num_slots, D] aggregated slot embeddings
        """
        B, N, D = x.size()

        # Initialize slots
        mu = self.slots_mu.expand(B, -1, -1)                 # [B, num_slots, D]
        sigma = F.softplus(self.slots_sigma).expand(B, -1, -1)
        slots = mu + sigma * torch.randn_like(sigma)          # add noise

        # Normalize input features
        x = self.norm_input(x)

        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)

            # Compute attention
            q = self.to_q(slots_norm)     # [B, num_slots, D]
            k = self.to_k(x)              # [B, N, D]
            v = self.to_v(x)              # [B, N, D]

            # Attention logits and weights
            attn_logits = torch.matmul(k, q.transpose(-1, -2)) * self.scale  # [B, N, num_slots]
            attn = attn_logits.softmax(dim=-1)                                # softmax over slots

            # Weighted aggregation
            updates = torch.matmul(attn.transpose(-1, -2), v)  # [B, num_slots, D]

            # Slot update with GRU
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            ).reshape(B, -1, D)

            # Residual MLP
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots  # [B, num_slots, D]

class ReIDWithSlotAttention(nn.Module):
    def __init__(self, num_classes, input_dim=256, slot_dim=256, num_slots=1, emb_dim=512):
        super().__init__()
        self.slot_attention = SlotAttention(num_slots=num_slots, dim=slot_dim)
        self.proj = nn.Linear(input_dim, slot_dim)
        self.fc = nn.Linear(num_slots * slot_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, reid_input):
        """
        reid_input: [B, topk, D]
        returns:
          reid_emb   [B, topk, emb_dim]
          reid_logits [B, topk, num_classes]
        """
        B, T, D = reid_input.shape
        reid_input = self.proj(reid_input)  # [B, topk, slot_dim]

        # Treat each [B, topk] object as a separate batch item
        x = reid_input.reshape(B * T, 1, -1)  # [B*T, N=1, slot_dim]
        slots = self.slot_attention(x)        # [B*T, num_slots, slot_dim]
        #print("slots")
        #print(slots.shape)
        emb = self.fc(slots.view(B * T, -1))  # [B*T, emb_dim]
        emb = self.norm(emb)
        emb = F.normalize(emb, p=2, dim=-1)

        logits = self.classifier(emb)         # [B*T, num_classes]

        # reshape back
        emb = emb.view(B, T, -1)              # [B, topk, emb_dim]
        logits = logits.view(B, T, -1)        # [B, topk, num_classes]

        return emb, logits
class SwinUNetMultiUp(nn.Module):
    def __init__(self, swin_model_name="swin_large_patch4_window12_384",num_queries=200, topk_spatial=100,pos_type="normal", num_decoder_layers=6, pretrained=True, d_model=256):
        super().__init__()
        #self.backbone = timm.create_model(swin_model_name,pretrained=pretrained,features_only=True)
        self.backbone = timm.create_model(
                "resnet50",
                pretrained=True,
                features_only=True,      # only output feature maps
                out_indices=(2,3,4),   # choose which layers to extract
                global_pool=""           # disable global pooling
                )
        # --- Positional Encoding ---
        if pos_type == "sine":
            self.position_encoding = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=True)
        elif pos_type == "learned":
            self.position_encoding = PositionEmbeddingLearned(num_pos_feats=d_model // 2)
        else:
            self.position_encoding = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        channels = self.backbone.feature_info.channels()
        print(f"[INFO] channels: {channels}")
        
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
        #self.encoder = TransformerEncoder(d_model=d_model, nhead=8)
        self.encoder = nn.TransformerEncoder(
        encoder_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=1024),
        num_layers=6,   # default
        )
        # Query generation (content + spatial top-k)
        self.num_content_queries = num_queries
        self.content_queries = nn.Parameter(torch.randn(num_queries, d_model))
        self.spatial_score = nn.Linear(d_model, 1)
        self.topk_spatial = topk_spatial
        self.box_init = nn.Linear(d_model, 4)
        #self._init_box_init()

        # Decoder stack
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model=d_model, nhead=8) for _ in range(num_decoder_layers)])

        self.pe_encoder = nn.Parameter(
            torch.rand((1, 225, d_model)), requires_grad=True
        )
        # box refinement projection (optional)
        self.refine_proj = nn.Linear(d_model, 4) 
        self.reid_head = ReIDWithSlotAttention(num_classes=1000, slot_dim=256, num_slots=6)
    def _init_box_init(self):
        """Initialize box_init to produce small initial boxes."""
        nn.init.constant_(self.box_init.weight, 0.0)
        nn.init.constant_(self.box_init.bias, 0.0)
        with torch.no_grad():
            # sigmoid(-2.5) ≈ 0.08 → small width/height
            self.box_init.bias[2:].fill_(-2.5)
    def forward(self, x):
        
        # # ======== Vectorized TOP-K SELECTION FOR ReID ========
        # final_logits = all_logits[-1]      # [B, Q, num_classes_det]
        # final_boxes  = all_boxes[-1]       # [B, Q, 4]
        
        # scores = final_logits.softmax(-1)[..., 1]   # [B, Q], use class-1 prob or max prob
        # topk_vals, topk_idx = torch.topk(scores, k=self.topk_spatial, dim=1)
        
        # # Gather top-k embeddings
        # reid_input = q.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, q.size(-1)))
        
        # # Feed top-k queries into ReID head
        # reid_emb, reid_logits = self.reid_head(reid_input)  # [B, topk, embedding_dim], [B, topk, num_classes]
        
        #for i, b in enumerate(all_logits):
        #    print(f"Layer {i} boxes shape: {b.shape}")
        #for i, b in enumerate(all_boxes):
        #    print(f"Layer {i} boxes shape: {b.shape}")
        #print(reid_input.shape)
        #print(reid_emb.shape)
        #print(reid_logits.shape)
        # return {
        #     'per_layer_logits': all_logits,
        #     'per_layer_boxes': all_boxes,
        #     'final_logits': all_logits[-1],
        #     'final_boxes': all_boxes[-1],
        #     'init_boxes': ref_boxes,
        #     'topk_idx': topk_idx,
        # #    'reid_emb': reid_emd,
        # #    'reid_logits': reid_logits,
        # }
        # Stack and return
        class_preds = torch.stack(all_logits, dim=1)
        bbox_preds = torch.stack(all_boxes, dim=1)
        return class_preds, bbox_preds
