import torch
from torch import nn
from torchsummary import summary

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, num_kv_heads = None, dropout = 0.):
        super().__init__()
        
        if num_kv_heads is None:
            num_kv_heads = max(1, heads // 2)
        assert heads % num_kv_heads == 0, 'Heads must be divisible by number of groups'

        self.dim = dim
        self.heads = heads
        self.num_kv_heads = num_kv_heads
        self.dim_head = dim_head
        self.heads_per_group = heads // num_kv_heads
        self.scale = dim_head ** -0.5
        
        inner_dim = dim_head * heads
        kv_dim = dim_head * num_kv_heads
        project_out = not (heads == 1 and dim_head == dim)

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # 分别定义Q、K、V的投影层
        self.q_proj = nn.Linear(dim, inner_dim, bias = False)
        self.k_proj = nn.Linear(dim, kv_dim, bias = False)
        self.v_proj = nn.Linear(dim, kv_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.heads)
        k = rearrange(k, 'b s (g d) -> b g s d', g=self.num_kv_heads)
        v = rearrange(v, 'b s (g d) -> b g s d', g=self.num_kv_heads)
        
        # 将查询头按组重新排列
        q = rearrange(q, 'b (g h_per_g) s d -> b g h_per_g s d', g=self.num_kv_heads)
        
        k = k.unsqueeze(2)
        v = v.unsqueeze(2)
        
        # 计算注意力分数
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # 应用注意力权重
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b g h_per_g s d -> b s (g h_per_g d)')
        
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., num_kv_heads = None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                GroupedQueryAttention(dim, heads = heads, dim_head = dim_head, num_kv_heads = num_kv_heads, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class GQAViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., num_kv_heads = None):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_kv_heads)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GQAViT(image_size=224, patch_size=16, num_classes=100, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1, emb_dropout=0.1, dim_head=64, pool='cls', channels=3, num_kv_heads=3).to(device)
    print(summary(model, (3, 224, 224)))