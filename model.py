import math
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
    
#VITs -----------------------

class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them"""
    
    def __init__(self, img_size=32, patch_size=4, in_channels=3, n_embd=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_embd,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)                        # (B, n_embd, H', W')
        x = x.flatten(2).transpose(1, 2)        # (B, num_patches, n_embd)
        return x


# ============================================
# ATTENTION HEADS (NO CAUSAL MASKING FOR VISION)
# ============================================

class VisionHead(nn.Module):
    """Standard self-attention head without causal masking (for vision)"""
    
    def __init__(self, head_size, n_embd, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, return_hook=False):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        attn = (q @ k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
        attn = F.softmax(attn, dim=-1)
        hook = attn
        attn = self.dropout(attn)
        
        v = self.value(x)
        out = attn @ v
        
        if return_hook:
            return out, hook
        return out


class VisionLogisticHead(nn.Module):
    """Logistic (sigmoid) attention head without causal masking"""
    
    def __init__(self, head_size, n_embd, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, return_hook=False):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = (q @ k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
        wei = F.sigmoid(wei)
        hook = wei
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        
        if return_hook:
            return out, hook
        return out


class VisionSoftmaxFreeHead(nn.Module):
    """Softmax-free attention head without causal masking (SimA for vision)"""
    
    def __init__(self, head_size, n_embd, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, return_hook=False):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        k = F.normalize(k)
        q = F.normalize(q)
        
        wei = (q @ k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
        hook = wei
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        
        if return_hook:
            return out, hook
        return out


# ============================================
# LANGUAGE MODEL ATTENTION HEADS (WITH CAUSAL MASKING)
# ============================================

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


class VisionMultiHeadAttention(nn.Module):
    """Multi-head attention for vision (no causal masking)"""
    
    def __init__(self, num_heads, head_size, n_embd, dropout=0.0, head_type='standard'):
        super().__init__()
        
        head_classes = {
            'standard': VisionHead,
            'logistic': VisionLogisticHead,
            'softmax_free': VisionSoftmaxFreeHead
        }
        
        head_class = head_classes.get(head_type, VisionHead)
        self.heads = nn.ModuleList([
            head_class(head_size, n_embd, dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, return_hook=False):
        if return_hook:
            outs, hooks = [], []
            for h in self.heads:
                o, hook = h(x, return_hook=True)
                outs.append(o)
                hooks.append(hook)
            
            out = torch.cat(outs, dim=-1)
            out = self.dropout(self.proj(out))
            hooks = torch.stack(hooks, dim=1)  # (B, num_heads, T, T)
            return out, hooks
        else:
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
            return out


class VisionBlock(nn.Module):
    """Transformer block for vision (no causal masking)"""
    
    def __init__(self, n_embd, n_head, dropout=0.0, head_type='standard'):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = VisionMultiHeadAttention(n_head, head_size, n_embd, dropout, head_type)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x, return_hook=False):
        if return_hook:
            sa_out, attn = self.sa(self.ln1(x), return_hook=True)
            x = x + sa_out
            x = x + self.ffwd(self.ln2(x))
            return x, attn
        else:
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x


class SmallVisionTransformer(nn.Module):
    """Vision Transformer for image classification"""
    
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 n_embd=64, n_head=8, n_layer=3, dropout=0.2, head_type='standard'):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, n_embd)
        num_patches = (img_size // patch_size) ** 2
        
        # Position embedding includes cls token
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, n_embd)
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))
        
        self.blocks = nn.ModuleList([
            VisionBlock(n_embd, n_head, dropout, head_type)
            for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.cls_head = nn.Linear(n_embd, num_classes)
    
    def forward(self, x, targets=None, return_hook=False):
        B = x.size(0)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, n_embd)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, n_embd)
        
        # Add position embedding
        x = x + self.position_embedding
        
        # Transformer blocks
        attentions = []
        for blk in self.blocks:
            if return_hook:
                x, attn = blk(x, return_hook=True)
                attentions.append(attn)
            else:
                x = blk(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Classification head (use cls token)
        cls_output = x[:, 0]
        logits = self.cls_head(cls_output)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        
        if return_hook:
            return logits, loss, attentions
        
        return logits, loss


# ============================================
# Char-level LM (text8): Task1 & Task2
# Reuses FeedForward from above.
# ============================================

def sinusoidal_pos_emb(T, C, device):
    """Sinusoidal position embedding for variable-length sequences."""
    position = torch.arange(T, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, C, 2, device=device) * (-math.log(10000.0) / C))
    pe = torch.zeros(T, C, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class HeadCharLM(nn.Module):
    """Vanilla self-attention head (softmax) for char-level LM."""

    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        scores = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(scores, dim=-1)
        w = self.dropout(w)
        v = self.value(x)
        return w @ v


class SSMaxHeadLearnable(nn.Module):
    """SSMax head with learnable s (q * s*log(n)) for char-level LM."""

    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.s_raw = nn.Parameter(torch.tensor(0.0))

    def s_value(self):
        return F.softplus(self.s_raw)

    def forward(self, x):
        B, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        n = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)
        logn = torch.log(n).view(1, T, 1)
        q_scaled = q * (self.s_value() * logn)
        scores = (q_scaled @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(scores, dim=-1)
        w = self.dropout(w)
        return w @ v


class MultiHeadAttentionChar(nn.Module):
    """Multi-head attention for char LM (softmax or ssmax learnable)."""

    def __init__(self, n_embd, n_head, block_size, dropout=0.0, attn_mode="softmax"):
        super().__init__()
        head_size = n_embd // n_head
        if attn_mode == "softmax":
            mk = lambda: HeadCharLM(n_embd, head_size, block_size, dropout)
        elif attn_mode == "ssmax":
            mk = lambda: SSMaxHeadLearnable(n_embd, head_size, block_size, dropout)
        else:
            raise ValueError(attn_mode)
        self.heads = nn.ModuleList([mk() for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class BlockChar(nn.Module):
    """Transformer block for char LM (returns x only). Reuses FeedForward."""

    def __init__(self, n_embd, n_head, block_size, dropout=0.0, attn_mode="softmax"):
        super().__init__()
        self.sa = MultiHeadAttentionChar(n_embd, n_head, block_size, dropout, attn_mode)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLMEmbedPos(nn.Module):
    """LM with learned position embedding (fixed block_size). For text8 task1."""

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.0, attn_mode="softmax"):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            BlockChar(n_embd, n_head, block_size, dropout, attn_mode) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


class BigramLMSinusoidal(nn.Module):
    """LM with sinusoidal position (variable T). For text8 task2."""

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.0, attn_mode="softmax"):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[
            BlockChar(n_embd, n_head, block_size, dropout, attn_mode) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = sinusoidal_pos_emb(T, self.n_embd, idx.device)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss
