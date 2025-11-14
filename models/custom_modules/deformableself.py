import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

class DeformableSelfAttention(nn.Module):
    """
    Multi-head, multi-scale deformable attention (spatial deformable sampling).
    Input:  x -> (B, C, H, W)
    Output: (B, C, H, W)
    """

    def __init__(self, dim, n_heads=5, n_points=4, n_scales=1, offset_scale=1.5, dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.n_points = n_points
        self.n_scales = n_scales
        self.head_dim = dim // n_heads
        self.offset_scale = offset_scale

        # basic projections
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.kv_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)

        # offsets: for each head, scale, point -> 2 coords
        # output channels = n_heads * n_scales * n_points * 2
        self.offset_gen = nn.Conv2d(dim, n_heads * n_scales * n_points * 2, kernel_size=3, padding=1)

        # optional attention weight generator (per head, per scale, per point)
        # We'll compute attention via dot-product; providing an extra per-point scaling can help:
        self.attn_scale_gen = None
        # self.attn_scale_gen = nn.Conv2d(dim, n_heads * n_scales * n_points, kernel_size=3, padding=1)

        # final output projection
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _make_base_grid(B, H, W, device, dtype):
        # base normalized grid in [-1,1], shape (H, W, 2) where grid[...,0]=x (width), grid[...,1]=y (height)
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)
        base_grid = torch.stack([grid_x, grid_y], dim=-1)       # (H, W, 2)
        return base_grid  # caller will expand to needed shape

    def forward(self, x,H,W):
        """
        x: (B, C, H, W)
        returns: (B, C, H, W)
        """
        x = rearrange(x, 'b q (h w) -> b q h w', h=H, w=W)
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # ---- projections ----
        q = self.q_proj(x)                     # (B, C, H, W)
        kv = self.kv_proj(x)                   # (B, 2*C, H, W)
        k_all, v_all = kv.chunk(2, dim=1)      # each (B, C, H, W)

        # split into heads: shape -> (B, heads, head_dim, H, W)
        q_heads = rearrange(q, 'b (h d) H W -> b h d H W', h=self.n_heads)
        k_heads = rearrange(k_all, 'b (h d) H W -> b h d H W', h=self.n_heads)
        v_heads = rearrange(v_all, 'b (h d) H W -> b h d H W', h=self.n_heads)

        # ---- offsets generation ----
        offsets = self.offset_gen(x)  # (B, n_heads * n_scales * n_points * 2, H, W)
        # unpack to (B, heads, scales, points, 2, H, W)
        offsets = rearrange(
            offsets,
            'b (head scale point coord) H W -> b head scale point coord H W',
            head=self.n_heads, scale=self.n_scales, point=self.n_points, coord=2
        )  # (B, h, s, p, 2, H, W)
        # move last dims to friendly format: (B, h, s, p, H, W, 2)
        offsets = offsets.permute(0, 1, 2, 3, 5, 6, 4).contiguous()

        # scale offsets to reasonable normalized range
        offsets = torch.tanh(offsets) * float(self.offset_scale)  # small offsets in [-offset_scale,offset_scale]

        # ---- build deform grids = base_grid + offsets ----
        base_grid = self._make_base_grid(B, H, W, device, dtype)  # (H, W, 2)
        # expand base grid to (1,1,1,1,H,W,2) so we can add offsets broadcastably
        base_grid_exp = base_grid.view(1, 1, 1, 1, H, W, 2)

        # deform grid for each (b, head, scale, point): (B, h, s, p, H, W, 2)
        deform_grid = base_grid_exp + offsets
        deform_grid = deform_grid.clamp(-1.0, 1.0)

        # ---- flatten dims so grid_sample gets 4D input and 4D grid ----
        # grid_sample expects grid shape (N, H_out, W_out, 2), input (N, C, H, W).
        # We'll flatten (b, head, scale, point) into batch axis.
        b_h_s_p = B * self.n_heads * self.n_scales * self.n_points
        grid_flat = rearrange(deform_grid, 'b h s p H W c -> (b h s p) H W c')

        # prepare inputs for sampling: we need to sample k and v per head.
        # k_heads: (B, h, d, H, W) -> repeat for scales and points: (B*h*s*p, d, H, W)
        k_repeat = repeat(k_heads, 'b h d H W -> (b h s p) d H W', s=self.n_scales, p=self.n_points)
        v_repeat = repeat(v_heads, 'b h d H W -> (b h s p) d H W', s=self.n_scales, p=self.n_points)

        # ---- grid_sample to get sampled k/v per (b,h,s,p) ----
        # sampled_k: (B*h*s*p, d, H, W)
        sampled_k = F.grid_sample(k_repeat, grid_flat, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled_v = F.grid_sample(v_repeat, grid_flat, mode='bilinear', padding_mode='zeros', align_corners=True)

        # reshape back to (B, h, s, p, d, H, W)
        sampled_k = rearrange(sampled_k, '(b h s p) d H W -> b h s p d H W', b=B, h=self.n_heads, s=self.n_scales, p=self.n_points)
        sampled_v = rearrange(sampled_v, '(b h s p) d H W -> b h s p d H W', b=B, h=self.n_heads, s=self.n_scales, p=self.n_points)

        # collapse (s, p) -> K
        K = self.n_scales * self.n_points
        sampled_k = rearrange(sampled_k, 'b h s p d H W -> b h (s p) d H W')  # (B, h, K, d, H, W)
        sampled_v = rearrange(sampled_v, 'b h s p d H W -> b h (s p) d H W')  # (B, h, K, d, H, W)

        # ---- compute dot-product attention per spatial location ----
        # q_heads: (B, h, d, H, W)
        # we want scores: (B, h, K, H, W) = sum_d q * k_sampled over d
        # einsum with shapes: 'b h d H W, b h k d H W -> b h k H W'
        scores = einsum('b h d H W, b h k d H W -> b h k H W', q_heads, sampled_k)
        scores = scores / (self.head_dim ** 0.5)

        # softmax over K dimension
        attn = torch.softmax(scores, dim=2)  # (B, h, K, H, W)
        attn = self.dropout(attn)

        # weighted sum over sampled_v: out_heads (B, h, d, H, W)
        out_heads = einsum('b h k H W, b h k d H W -> b h d H W', attn, sampled_v)

        # ---- merge heads and project out ----
        out = rearrange(out_heads, 'b h d H W -> b (h d) H W')  # (B, dim, H, W)
        out = self.out_proj(out)  # (B, C, H, W)
        out = rearrange(x, 'b q h w -> b q (h w)')
        return out