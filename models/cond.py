import timm
import torch
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.misc import FrozenBatchNorm2d
from einops import rearrange
#from models.custom_modules.cond_detr_decoder_layer import ConditionalDecoderLayer
from models.custom_modules.cond_detr_decoder_layerV1_0 import ConditionalDecoderLayer
import torch.nn.functional as F
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
    ):
        super().__init__()

        # self.backbone = create_feature_extractor(
        #     torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True),
        #     return_nodes={"layer4": "layer4"},
        # )
        self.backbone = timm.create_model(
                "resnet50",
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
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, n_heads, 4 * d_model, 0.1, batch_first=True
            ),
            num_layers=n_layers, #n_layers
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
        memory = self.transformer_encoder(tokens + self.pe_encoder)

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
        for layer in self.decoder_layers:
            decoder_embeddings, ref_boxes = layer(
                decoder_embeddings, ref_boxes, memory
            )
            class_preds.append(self.linear_class(decoder_embeddings))
            bbox_preds.append(self.linear_bbox(decoder_embeddings) + ref_boxes)

        return torch.stack(class_preds, dim=1), torch.stack(bbox_preds, dim=1)

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
