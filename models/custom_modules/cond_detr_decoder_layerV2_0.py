import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
# class MSDeformAttn_SingleLevel(nn.Module):
#     """
#     Simplified Deformable Attention for a single feature level.
#     (Fully PyTorch, slower but functionally identical)
#     """
#     def __init__(self, d_model=256, n_heads=8, n_points=4):
#         super().__init__()
#         if d_model % n_heads != 0:
#             raise ValueError("d_model must be divisible by n_heads")
        
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.n_points = n_points
#         self.head_dim = d_model // n_heads

#         # learnable projections
#         self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
#         self.attention_weights = nn.Linear(d_model, n_heads * n_points)
#         self.value_proj = nn.Linear(d_model, d_model)
#         self.output_proj = nn.Linear(d_model, d_model)
#         self.attn_dropout = nn.Dropout(p=0.1)
#         self._reset_parameters()

#     def _reset_parameters(self):
#         nn.init.constant_(self.sampling_offsets.weight, 0.)
#         nn.init.constant_(self.sampling_offsets.bias, 0.)
#         nn.init.constant_(self.attention_weights.weight, 0.)
#         nn.init.constant_(self.attention_weights.bias, 0.)
#         nn.init.xavier_uniform_(self.value_proj.weight)
#         nn.init.constant_(self.value_proj.bias, 0.)
#         nn.init.xavier_uniform_(self.output_proj.weight)
#         nn.init.constant_(self.output_proj.bias, 0.)

#     def forward(self, query, reference_points, input_flatten, spatial_shape, padding_mask=None):
#         """
#         Args:
#             query: (N, Lq, C)
#             reference_points: (N, Lq, 2), normalized [0, 1]
#             input_flatten: (N, H*W, C)
#             spatial_shape: (H, W)
#             padding_mask: (N, H*W) or None
#         """
#         N, Lq, C = query.shape
#         _,M,_=input_flatten.shape
#         H,W=int(M**0.5),int(M**0.5)

#         # project input features
#         value = self.value_proj(input_flatten)
#         if padding_mask is not None:
#             value = value.masked_fill(padding_mask[..., None], 0)

#         value = value.view(N, H * W, self.n_heads, self.head_dim).permute(0, 2, 3, 1)
#         value = value.contiguous().view(N * self.n_heads, self.head_dim, H, W)

#         # offsets and attention weights
#         offsets = self.sampling_offsets(query).view(N, Lq, self.n_heads, self.n_points, 2)
#         offsets = 0.1 * torch.tanh(offsets)
#         attn_weights = self.attention_weights(query).view(N, Lq, self.n_heads, self.n_points)
#         attn_weights = self.attn_dropout(F.softmax(attn_weights, -1))

#         # reference points normalized [0,1] → [-1,1]
#         ref_grid = 2 * reference_points[:, :, None, None, :] - 1  # (N, Lq, 1, 1, 2)
#         sampling_locations = ref_grid + offsets / torch.tensor([W, H], device=query.device)[None, None, None, None, :]

#         # to [-1,1] range for grid_sample
#         #sampling_grids = sampling_locations * 2 - 1
#         sampling_grids = sampling_locations.clamp(-1, 1)
#         sampling_grids = sampling_grids.view(N * self.n_heads, Lq * self.n_points, 1, 2)

#         # sample from feature map
#         sampled = F.grid_sample(
#             value, sampling_grids,
#             mode='bilinear', padding_mode='zeros', align_corners=False
#         )

#         sampled = sampled.view(N, self.n_heads, self.head_dim, Lq, self.n_points)
#         attn_weights = attn_weights.permute(0, 2, 1, 3).unsqueeze(2)
#         # sampled: [N, n_heads, head_dim, Lq, n_points]
#         # Weighted sum over points
#         #print(sampled.shape, attn_weights.shape)
#         output = (sampled * attn_weights).sum(-1).permute(0, 3, 1, 2).reshape(N, Lq, C) ##
#         #print(output.shape)
#         return self.output_proj(output), attn_weights.detach()
class ContinuousPosBias(nn.Module):
    """
    Continuous Positional Bias (from SwinV2)
    Adds geometry-aware bias to attention logits.
    """
    def __init__(self, dim, heads, depth=2):
        super().__init__()
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Sequential(nn.Linear(2, dim), nn.ReLU()))
        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(nn.Linear(dim, dim), nn.ReLU()))
        self.mlp.append(nn.Linear(dim, heads))

    def forward(self, ref_points, sample_points):
        """
        ref_points: (N, Lq, 2)
        sample_points: (N, Lq, n_points, 2)
        """
        # relative position: (N, Lq, n_points, 2)
        #print(ref_points.shape,sample_points.shape)
        if sample_points.shape[2] != ref_points.shape[1]:
            # (B, Lq, n_heads, n_points, 2) → (B, n_heads, Lq, n_points, 2)
            sample_points = sample_points.permute(0, 2, 1, 3, 4).contiguous()
        if sample_points.dim() == 4:
            # (B, Lq, n_points, 2)
            rel_pos = sample_points - ref_points.unsqueeze(-2)
        elif sample_points.dim() == 5:
            # (B, n_heads, Lq, n_points, 2)
            rel_pos = sample_points - ref_points.unsqueeze(1).unsqueeze(-2)
        else:
            raise ValueError(f"Unexpected sample_points shape: {sample_points.shape}")
        rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1.0)

        bias = rel_pos
        for layer in self.mlp:
            bias = layer(bias)
        # Output: (N, Lq, n_points, heads)
        #print(bias.shape)
        bias = bias.permute(0, 4, 2, 3, 1).mean(-1)  # → [B, heads, Lq, n_points]
        #print(bias.shape)
        return bias  # (N, Lq, heads, n_points)


# ------------------------
# Main class: Single-level Deformable Attention + CPB
# ------------------------
class MSDeformAttn_SingleLevel(nn.Module):
    """
    Simplified Deformable Attention + Continuous Positional Bias
    (PyTorch implementation, functional equivalent to Deformable DETR)
    """
    def __init__(self, d_model=256, n_heads=8, n_points=4, cpb_dim=64):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        # Learnable projections
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(0.1)

        # CPB module
        self.cpb = ContinuousPosBias(dim=cpb_dim, heads=n_heads, depth=2)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        nn.init.constant_(self.sampling_offsets.bias, 0.)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query, reference_points, input_flatten, padding_mask=None):
        """
        Args:
            query: (N, Lq, C)
            reference_points: (N, Lq, 2), normalized [0, 1]
            input_flatten: (N, H*W, C)
            spatial_shape: (H, W)
            padding_mask: (N, H*W) or None
        """
        N, Lq, C = query.shape
        _,M,_=input_flatten.shape
        H,W=int(M**0.5),int(M**0.5)

        # 1️⃣ Value projection
        value = self.value_proj(input_flatten)
        if padding_mask is not None:
            value = value.masked_fill(padding_mask[..., None], 0)
        value = value.view(N, H * W, self.n_heads, self.head_dim).permute(0, 2, 3, 1)
        value = value.contiguous().view(N * self.n_heads, self.head_dim, H, W)

        # 2️⃣ Predict offsets + attention weights
        offsets = self.sampling_offsets(query).view(N, Lq, self.n_heads, self.n_points, 2)
        offsets = 0.1 * torch.tanh(offsets)
        attn_weights = self.attention_weights(query).view(N, Lq, self.n_heads, self.n_points)
        attn_weights = F.softmax(attn_weights, -1)
        
        # 3️⃣ Compute sampling locations
        ref_grid = 2 * reference_points[:, :, None, None, :] - 1
        sampling_locations = ref_grid + offsets / torch.tensor([W, H], device=query.device)[None, None, None, None, :]
        sampling_grids = sampling_locations.clamp(-1, 1).view(N * self.n_heads, Lq * self.n_points, 1, 2)

        # 4️⃣ Deformable sampling via grid_sample
        sampled = F.grid_sample(
            value, sampling_grids, mode='bilinear', padding_mode='zeros', align_corners=False
        )
        sampled = sampled.view(N, self.n_heads, self.head_dim, Lq, self.n_points)

        # 5️⃣ Apply CPB bias to attention weights
        bias = self.cpb(reference_points, sampling_locations.squeeze(3))  # (N, Lq, heads, n_points)
        bias = torch.tanh(bias)  # smooth bias
        attn_weights = self.attn_dropout(F.softmax(attn_weights + bias.permute(0, 2, 1, 3), -1))

        # 6️⃣ Weighted sum over points
        attn_weights = attn_weights.permute(0, 2, 1, 3).unsqueeze(2)
        output = (sampled * attn_weights).sum(-1).permute(0, 3, 1, 2).reshape(N, Lq, C)

        return self.output_proj(output), attn_weights.detach()
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class ConditionalDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.1,n_points=4,group_detr=4):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = MSDeformAttn_SingleLevel(d_model, n_heads, n_points)
        #self.cross_attn = nn.MultiheadAttention(
        #    d_model, n_heads, dropout=dropout, batch_first=True
        #)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.group_detr=group_detr
        # Learnable anchor positions (x,y pairs)
        self.spatial_2d_coords = nn.Linear(d_model, 2)

        # Displacement FFN for spatial queries (calculated from the object queries)
        self.displacement_ffn = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )

        # Learnable diagonal transformation (λq)
        self.lambda_q = nn.Parameter(torch.ones(d_model))
        self.delta_box_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4)
        )
        self.learnedposition=MLP(d_model, d_model, d_model, 2)

    def positional_encoding(self, ref_points):
        """
        Applies sinusoidal encoding to reference points.
        ref_points: (batch_size, num_queries, 2) -> (batch_size, num_queries, d_model)
        """
        # Normalize using sigmoid
        ref_points = torch.sigmoid(ref_points)  # Normalize to [0,1]

        # Get half the dimension size for each axis..
        half_dim = self.lambda_q.shape[0] // 2

        # Compute sinusoidal embeddings (like Transformer positional encoding)
        dim_t = torch.arange(half_dim, dtype=torch.float32, device=ref_points.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / half_dim)

        pos_x = ref_points[..., 0, None] / dim_t
        pos_y = ref_points[..., 1, None] / dim_t

        pos_x = torch.cat(
            [torch.sin(pos_x[:, :, ::2]), torch.cos(pos_x[:, :, 1::2])], dim=-1
        )
        pos_y = torch.cat(
            [torch.sin(pos_y[:, :, ::2]), torch.cos(pos_y[:, :, 1::2])], dim=-1
        )
        #print(pos_x.shape,pos_y.shape)
        return torch.cat([pos_x, pos_y], dim=-1)  # (batch_size, num_queries, d_model) 
        ###uses learned positional encoding in original lwdetr implementation
        ###when adding with decoder embedding (it gives better result than sinusoidal encoding may be)
        ###when passing to cross attention as reference point sinusoidal is used.

    def forward(self, decoder_embed, ref_boxes, memory):
        # Add positional context based on ref_boxes
        pos_embed = self.positional_encoding(ref_boxes[..., :2])  # only (cx, cy)
        q = decoder_embed + self.learnedposition(pos_embed) ##batch,qeries,dim
        #print(f"pos_embed shape: {pos_embed.shape}")
        # Self-attention
        _,num_queries,_=decoder_embed.shape
        B,D,N=memory.shape
        H=W=int(D**0.5)
        spatial_shape = torch.as_tensor([[H, W]], device=memory.device, dtype=torch.long)
        # --- Grouped Self-Attention (memory-efficient) ---
        if self.training and num_queries % self.group_detr == 0 and self.group_detr > 1:
            #print("Using grouped self-attention with group size:", self.group_detr)
            q_group = torch.cat(q.split(num_queries // self.group_detr, dim=1), dim=0)
            k_group = torch.cat(q.split(num_queries // self.group_detr, dim=1), dim=0)
            v_group = torch.cat(decoder_embed.split(num_queries // self.group_detr, dim=1), dim=0)

            self_attn_out = self.self_attn(q_group, k_group, v_group)[0]
            self_attn_out = torch.cat(self_attn_out.split(B, dim=0), dim=1)
        else:
            #print("Using standard self-attention")
            self_attn_out = self.self_attn(q, q, decoder_embed)[0]
        
        #print(self_attn_out.shape)
        decoder_embed = self.norm1(decoder_embed + self.dropout(self_attn_out))

        # Cross-attention with encoder memory
        cross_out, cross_out_attain = self.cross_attn(decoder_embed + pos_embed, ref_boxes[..., :2] , memory) ##ref_boxes = memory #ref_boxes[..., :2]
        decoder_embed = self.norm2(decoder_embed + self.dropout(cross_out))

        # Feedforward
        ffn_out = self.ffn(decoder_embed)
        decoder_embed = self.norm3(decoder_embed + self.dropout(ffn_out))

        # Predict box deltas (Δcx, Δcy, Δw, Δh)
        delta_box = self.delta_box_head(decoder_embed)
        delta_box[..., :2] = torch.tanh(delta_box[..., :2]) * 0.05  # small center offset
        delta_box[..., 2:] = torch.tanh(delta_box[..., 2:]) * 0.1   # small scale offset

        # Refine boxes (iteratively)
        new_ref_boxes = ref_boxes.clone()
        new_ref_boxes[..., :2] = ref_boxes[..., :2] + delta_box[..., :2]  # refine center
        new_ref_boxes[..., 2:] = ref_boxes[..., 2:] * (1 + delta_box[..., 2:])  # refine size
        new_ref_boxes = new_ref_boxes.clamp(0, 1)

        return decoder_embed, new_ref_boxes, cross_out_attain
        