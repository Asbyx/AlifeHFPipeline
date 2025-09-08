from torchenhanced import ConfigModule, DevModule
import torch, torch.nn as nn


class MiniAttentionHead(ConfigModule):
    """
        Mini causal transformer head to be used on top of VJEPA representations.
    """

    def __init__(self, in_dim, num_tokens):
        """
            in_dim : int, dimension of the input
            num_tokens : int, number of tokens to output
        """
        super().__init__()
        
        self.attention = nn.MultiheadAttention(in_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.score_head = nn.Linear(in_dim, 1)
        self.pos_embedder = nn.Embedding(num_tokens, in_dim)

        self.register_buffer('cant_attend',torch.tril(torch.ones(num_tokens, num_tokens), diagonal=0))
        self.cant_attend = self.cant_attend==0

    def forward(self, x):
        """
            x : (B, T, in_dim) tensor
        """
        B, T, in_dim = x.shape
        x = x + self.pos_embedder(torch.arange(T).to(x.device)) # Pos embed

        out, _ = self.attention(x,x,x, is_causal=True, attn_mask=self.cant_attend[:T,:T])

        return out

class MiniBlock(DevModule):
    """
    One transformer block/layer, fast causal attention followed by a MLP.

    Args:
        embed_dim: number of embedding dimensions
        n_heads: number of attention heads
        attn_length: length of the attention window
        mlp_ratio: ratio of mlp hidden dim to embedding dim
        dropout: (optional) dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        num_tokens: int,
        device='cpu'
        ):
        super().__init__(device=device)

        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = MiniAttentionHead(
            in_dim=embed_dim,
            num_tokens=num_tokens,
        )

        self.ln_2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(embed_dim, int(2 * embed_dim)),
                act=nn.GELU(),
                c_proj=nn.Linear(int(2 * embed_dim), embed_dim),
                dropout=nn.Dropout(0.1),
            )
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp["dropout"](
            self.mlp["c_proj"](self.mlp["act"](self.mlp["c_fc"](self.ln_2(x))))
        )

        return x
