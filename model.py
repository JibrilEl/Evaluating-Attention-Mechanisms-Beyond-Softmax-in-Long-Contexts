import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """Standard self-attention head with softmax"""

    def __init__(self, head_size, block_size, n_embd, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_hook=False):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        hook = wei
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        
        if return_hook:
            return out, hook
        return out


class SSMaxHead(nn.Module):
    """Self-attention head with SSMax normalization"""

    def __init__(self, head_size, block_size, n_embd, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        vector = torch.ones(size=(block_size,))
        n_col_vector = (torch.tril(torch.ones(block_size, block_size)) @ vector.T).T
        mask_ssmax = torch.stack([n_col_vector for _ in range(block_size)])
        self.register_buffer('tril_ssm', 0.1 * torch.log(mask_ssmax))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_hook=False):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = wei + self.tril_ssm[:T, :T]
        wei = F.softmax(wei, dim=-1)
        hook = wei
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        
        if return_hook:
            return out, hook
        return out


class LogisticHead(nn.Module):
    """Self-attention head with sigmoid activation"""

    def __init__(self, head_size, block_size, n_embd, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_hook=False):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.sigmoid(wei)
        hook = wei
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        
        if return_hook:
            return out, hook
        return out


class SoftmaxFreeHead(nn.Module):
    """Softmax-free self-attention head (SimA)"""

    def __init__(self, head_size, block_size, n_embd, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_hook=False):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        k = F.normalize(k)
        q = F.normalize(q)
        
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, 0)
        hook = wei
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        
        if return_hook:
            return out, hook
        return out


class MultiHeadAttention(nn.Module):
    """Multi-headed attention"""

    def __init__(self, num_heads, head_size, block_size, n_embd, dropout=0.0, head_type='softmax_free'):
        super().__init__()
        
        head_classes = {
            'softmax_free': SoftmaxFreeHead,
            'standard': Head,
            'ssmax': SSMaxHead,
            'logistic': LogisticHead
        }
        
        head_class = head_classes.get(head_type, SoftmaxFreeHead)
        self.heads = nn.ModuleList([
            head_class(head_size, block_size, n_embd, dropout) 
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_hook=False):
        if not return_hook:
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
            return out


class FeedForward(nn.Module):
    """Feed-forward network"""

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block"""

    def __init__(self, n_embd, n_head, block_size, dropout=0.0, head_type='softmax_free'):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, block_size, n_embd, dropout, head_type)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, latent_space=None):
        if isinstance(x, tuple):
            x, latent_space = x
        
        x = x + self.sa(self.ln1(x))
        latent_x_hook = x
        x = x + self.ffwd(self.ln2(x))
        return x, latent_x_hook


class SmallLanguageModel(nn.Module):
    """Small transformer language model"""

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.0, head_type='softmax_free'):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout, head_type) 
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x, latent = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, latent

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss, latent = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
