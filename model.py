import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Self Attention ---
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn = Q @ K.transpose(-2, -1) / (x.size(-1) ** 0.5)
        attn = F.softmax(attn, dim=-1)

        return attn @ V

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


class AttentionResidualBlock(nn.Module):
    def __init__(self, embed_dim, num_layers=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.attn = SelfAttention(embed_dim)
        self.ff = FeedForward(embed_dim)

        self.layer_query = nn.Linear(embed_dim, embed_dim)
        self.layer_key = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        states = [x]

        for _ in range(self.num_layers):
            current = states[-1]

            out = current + self.attn(current)
            out = out + self.ff(out)

            states.append(out)

        stacked = torch.stack(states, dim=1)  

        q = self.layer_query(states[-1]).unsqueeze(1)  
        k = self.layer_key(stacked)                    

        attn = (q * k).sum(-1) / (self.embed_dim ** 0.5)
        attn = torch.softmax(attn, dim=1)

        attn = attn.unsqueeze(-1)
        out = (attn * stacked).sum(1)

        return out

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(512, embed_dim)
        self.block = AttentionResidualBlock(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape

        token_emb = self.embedding(x)
        pos = torch.arange(T, device=x.device)
        pos_emb = self.pos_embedding(pos)

        x = token_emb + pos_emb
        x = self.block(x)
        logits = self.fc(x)
        return logits