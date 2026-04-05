import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x * x, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

def block_attn_res(blocks: list[torch.Tensor], partial_block: torch.Tensor, proj: nn.Linear, norm: RMSNorm) -> torch.Tensor:
    """Inter-block attention: attend over block reps + partial sum."""
    
    v_list = blocks + ([partial_block] if partial_block is not None else [])
    V = torch.stack(v_list)  
    K = norm(V)
    
    w = proj.weight.squeeze() 
    
    logits = torch.einsum('d, n b t d -> n b t', w, K)
    h = torch.einsum('n b t, n b t d -> b t d', logits.softmax(dim=0), V)
    
    return h

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.size()
        
        Q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) 
        K = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5) 
        
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        out = attn @ V 
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, layer_number, block_size):
        super().__init__()
        self.layer_number = layer_number
        self.block_size = block_size
        
        self.attn = MultiHeadAttention(embed_dim, num_heads=4)
        self.attn_norm = RMSNorm(embed_dim)
        self.attn_res_proj = nn.Linear(embed_dim, 1, bias=False)
        self.attn_res_norm = RMSNorm(embed_dim)
        
        self.mlp = FeedForward(embed_dim)
        self.mlp_norm = RMSNorm(embed_dim)
        self.mlp_res_proj = nn.Linear(embed_dim, 1, bias=False)
        self.mlp_res_norm = RMSNorm(embed_dim)

        nn.init.zeros_(self.attn_res_proj.weight)
        nn.init.zeros_(self.mlp_res_proj.weight)
        
    def forward(self, blocks, hidden_states):
        partial_block = hidden_states
        
        h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)
        
        if self.layer_number % (self.block_size // 2) == 0:
            blocks.append(partial_block)
            partial_block = None
            
        attn_out = self.attn(self.attn_norm(h))
        partial_block = partial_block + attn_out if partial_block is not None else attn_out
        
        h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)
        
        mlp_out = self.mlp(self.mlp_norm(h))
        partial_block = partial_block + mlp_out
        
        return blocks, partial_block

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_layers=4, block_size=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(512, embed_dim)
        
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, layer_number=i, block_size=block_size) 
            for i in range(num_layers)
        ])
        
        self.final_norm = RMSNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape

        token_emb = self.embedding(x)
        pos = torch.arange(T, device=x.device)
        pos_emb = self.pos_embedding(pos)

        h = token_emb + pos_emb
        
        blocks = [h]
        partial_block = h
        
        for layer in self.layers:
            blocks, partial_block = layer(blocks, partial_block)

        final_out = self.final_norm(partial_block)
        logits = self.fc(partial_block)
        return logits