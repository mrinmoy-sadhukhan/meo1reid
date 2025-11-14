import torch
from torch import nn


class ConditionalDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

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

    def forward(self, decoder_embed, ref_boxes, memory):
        # Add positional context based on ref_boxes
        pos_embed = self.positional_encoding(ref_boxes[..., :2])  # only (cx, cy)
        q = decoder_embed + pos_embed

        # Self-attention
        self_attn_out = self.self_attn(q, q, decoder_embed)[0]
        decoder_embed = self.norm1(decoder_embed + self.dropout(self_attn_out))

        # Cross-attention with encoder memory
        cross_out = self.cross_attn(decoder_embed + pos_embed, memory, memory)[0]
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

        return decoder_embed, new_ref_boxes
        