import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
class DeformableCrossAttention(nn.Module):
    """
    Cross-attention that samples a small patch of keys/values from a 2D memory feature map
    around each reference point (cx, cy) using grid_sample, then applies multi-head attention
    between queries and those sampled keys/values.
    """
    def __init__(
        self,
        d_model,
        n_heads=8,
        n_points=7,            # number of sampling points along one axis (kernel size)
        sample_sigma=0.1,      # scale of learnable offsets relative to image size
        dropout=0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_points = n_points
        self.k = n_points * n_points

        # project decoder queries to Q
        self.q_proj = nn.Linear(d_model, d_model)
        # project sampled memory patches to K and V
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # optionally learn small offsets (dx,dy) per query + per sampling location
        # We'll predict offsets from the query embedding (per query)
        self.offset_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2 * self.k),  # dx, dy for each sampling location
            nn.Tanh()  # output in [-1,1] scaled later by sample_sigma
        )
        self.sample_sigma = nn.Parameter(torch.tensor(sample_sigma), requires_grad=True)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _make_local_grid(self, device):
        """Create a (k,2) grid of relative coordinates in normalized image coords [-1,1]."""
        p = self.n_points
        # coordinates centered at 0 in range [-r, r] before normalization
        coords = torch.linspace(-(p - 1) / 2.0, (p - 1) / 2.0, steps=p, device=device)
        gx, gy = torch.meshgrid(coords, coords, indexing='xy')
        grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (k,2)
        return grid  # in pixel-grid units (we'll normalize by H-1, W-1 when building absolute coords)

    def forward(self, decoder_embed, ref_boxes, memory):
        """
        decoder_embed: (B, N, d_model)
        ref_boxes: (B, N, 4) in normalized coords [0,1] (cx, cy, w, h)
        memory: (B, C_mem, H, W)  - feature map from encoder. C_mem should be == d_model (or we'll project)
        Returns:
            attended: (B, N, d_model)
            attn_weights: (B, n_heads, N, k)  # optional attention over sampled points
        """
        B, N, D = decoder_embed.shape
        _, C_mem, H, W = memory.shape
        device = decoder_embed.device

        # If memory channels != d_model, project it to d_model with a conv
        if C_mem != D:
            memory = F.conv2d(memory, weight=torch.eye(D, C_mem, device=device).unsqueeze(-1).unsqueeze(-1))
            # The above is a lightweight identity-like conv only if shapes match; for mismatched sizes it's better to use an explicit conv:
            # But simple and safe approach: use a 1x1 conv module instead.
            memory = memory  # (we will override properly below if necessary)

        # Use an explicit 1x1 conv to ensure memory channels -> d_model
        if C_mem != D:
            proj_mem = nn.Conv2d(C_mem, D, kernel_size=1).to(device)
            memory = proj_mem(memory)
            C_mem = D

        # 1) Predict Q
        q = self.q_proj(decoder_embed)  # (B, N, D)
        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, N, head_dim)

        # 2) Prepare sampling grids for each query
        # ref_boxes are normalized [0,1], centers at cx,cy. Convert to normalized grid coords (-1..1) expected by grid_sample
        # grid_sample uses (x, y) = (W, H) ordering; our normalized grid uses (x, y)
        cx = ref_boxes[..., 0]  # (B,N)
        cy = ref_boxes[..., 1]  # (B,N)

        # base local grid in pixel units centered at 0; we'll convert to normalized coords
        local_grid = self._make_local_grid(device)  # (k, 2) in "pixel steps"
        # Normalize local grid to [-1,1] by dividing by (max_dim-1); use H and W separately
        # We'll create per-query grid by converting base steps to normalized offsets:
        # dx_normalized = local_x * (2 / (W - 1))
        # dy_normalized = local_y * (2 / (H - 1))
        local_x = local_grid[:, 0].view(1, 1, self.k).to(device)  # (1,1,k)
        local_y = local_grid[:, 1].view(1, 1, self.k).to(device)

        x_scale = 2.0 / max(W - 1, 1)
        y_scale = 2.0 / max(H - 1, 1)
        local_x_norm = local_x * x_scale  # (1,1,k)
        local_y_norm = local_y * y_scale  # (1,1,k)
        base_local = torch.cat([local_x_norm, local_y_norm], dim=-1)  # (1,1,k*2) but we'll reshape later

        # 3) predict offsets per query and add to base grid
        offsets = self.offset_predictor(decoder_embed)  # (B, N, 2*k)
        offsets = offsets.view(B, N, self.k, 2)  # (B,N,k,2) in [-1,1] because of tanh
        # scale by sample_sigma (learnable) but also by small factor relative to image size
        offsets = offsets * (self.sample_sigma)

        # base center in normalized coords (-1..1) for grid_sample
        cx_norm = cx * 2.0 - 1.0  # (B,N)
        cy_norm = cy * 2.0 - 1.0  # (B,N)

        # Expand centers to match k
        cx_exp = cx_norm.unsqueeze(-1)  # (B,N,1)
        cy_exp = cy_norm.unsqueeze(-1)  # (B,N,1)

        # base_local per sampling location: create (B,N,k,2)
        # base_local currently has shape (1,1,k*2) if concatenated, instead compute per channel:
        base_offsets = torch.stack(
            [local_x_norm.squeeze(0).squeeze(0).unsqueeze(0).unsqueeze(0).repeat(B, N, 1),
             local_y_norm.squeeze(0).squeeze(0).unsqueeze(0).unsqueeze(0).repeat(B, N, 1)],
            dim=-1
        )  # (B,N,k,2)

        # total sampling locations = base_local (relative) + learned offsets
        sampling_locs = torch.zeros_like(base_offsets)
        sampling_locs[..., 0] = cx_exp + base_offsets[..., 0] + offsets[..., 0]  # x coordinates
        sampling_locs[..., 1] = cy_exp + base_offsets[..., 1] + offsets[..., 1]  # y coordinates
        # sampling_locs shape: (B,N,k,2) in normalized coords (-inf..inf), but expected near [-1,1]

        # clamp sampling coords to [-1,1] to avoid extreme values
        sampling_locs = sampling_locs.clamp(-1.0, 1.0)

        # 4) Use grid_sample to sample memory at these locations for keys and values
        # grid_sample requires grid shape (B_out, H_out, W_out, 2). We'll set H_out=1, W_out=k and reshape memory accordingly.
        # To vectorize: reshape memory to (B, C, H, W), and sample per-query by expanding memory to B*N
        # Build grids for grid_sample: (B*N, 1, k, 2)
        grid = sampling_locs.view(B * N, self.k, 2)
        grid = grid.unsqueeze(1)  # (B*N, 1, k, 2) -> grid_sample expects (N, H_out, W_out, 2) ; (B*N, 1, k, 2) is fine

        # prepare memory for sampling: repeat per query
        # memory: (B, C_mem, H, W) -> (B, C, H, W) repeat each B times N -> (B*N, C, H, W)
        mem_rep = memory.unsqueeze(1).repeat(1, N, 1, 1, 1).view(B * N, memory.shape[1], H, W)

        # Now sample: result (B*N, C, 1, k)
        sampled = F.grid_sample(mem_rep, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        # reshape to (B, N, C, k)
        sampled = sampled.view(B, N, memory.shape[1], 1, self.k).squeeze(3)  # (B,N,C,k)

        # transpose to (B,N,k,C)
        sampled = sampled.permute(0, 1, 3, 2).contiguous()  # (B,N,k,C)

        # 5) Project sampled features to K and V and apply attention per head
        sampled_flat = sampled.view(B * N * self.k, memory.shape[1])  # (B*N*k, C)
        k_all = self.k_proj(sampled_flat).view(B, N, self.k, D)  # (B,N,k,D)
        v_all = self.v_proj(sampled_flat).view(B, N, self.k, D)  # (B,N,k,D)

        # reshape for heads: (B, n_heads, N, k, head_dim)
        k_all = k_all.view(B, N, self.k, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v_all = v_all.view(B, N, self.k, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        # q shape: (B, n_heads, N, head_dim)
        # Compute attention: for each head, dot q with k over head_dim -> (B, n_heads, N, k)
        q = q  # already shaped
        attn_scores = torch.einsum('b h n d, b h n m d -> b h n m', q, k_all)  # (B, n_heads, N, k)
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values: (B, n_heads, N, head_dim)
        out_heads = torch.einsum('b h n m, b h n m d -> b h n d', attn, v_all)  # (B, n_heads, N, head_dim)
        # merge heads
        out = out_heads.permute(0, 2, 1, 3).contiguous().view(B, N, D)  # (B,N,D)
        out = self.out_proj(out)
        return out, attn  # attn for debugging/visualization (B, n_heads, N, k)