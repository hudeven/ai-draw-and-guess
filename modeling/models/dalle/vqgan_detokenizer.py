import torch
from torch import nn
from math import sqrt


class ResnetBlock(nn.Module):
    def __init__(self, log2_dim_in: int, log2_dim_out: int):
        super().__init__()
        m, n = 2 ** log2_dim_in, 2 ** log2_dim_out
        self.is_middle = m == n
        self.norm1 = nn.GroupNorm(2 ** 5, m)
        self.conv1 = nn.Conv2d(m, n, 3, padding=1)
        self.norm2 = nn.GroupNorm(2 ** 5, n)
        self.conv2 = nn.Conv2d(n, n, 3, padding=1)
        self.nin_shortcut = nn.Conv2d(m, n, 1) if not self.is_middle else None

    def forward(self, x):
        h = x
        h = self.norm1.forward(h)
        h *= torch.sigmoid(h)
        h = self.conv1.forward(h)
        h = self.norm2.forward(h)
        h *= torch.sigmoid(h)
        h = self.conv2(h)
        if self.nin_shortcut is not None:
            x = self.nin_shortcut.forward(x)
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        n = 512
        self.norm = nn.GroupNorm(32, n)
        self.q = nn.Conv2d(n, n, 1)
        self.k = nn.Conv2d(n, n, 1)
        self.v = nn.Conv2d(n, n, 1)
        self.proj_out = nn.Conv2d(n, n, 1)

    def forward(self, x):
        n, m = 512, x.shape[0]
        h = x
        h = self.norm(h)
        k = self.k.forward(h)
        v = self.v.forward(h)
        q = self.q.forward(h)
        k = k.reshape(m, n, -1)
        v = v.reshape(m, n, -1)
        q = q.reshape(m, n, -1)
        q = q.permute(0, 2, 1)
        w = torch.bmm(q, k)
        w /= n ** 0.5
        w = torch.softmax(w, dim=2)
        w = w.permute(0, 2, 1)
        h = torch.bmm(v, w)
        token_size = int(sqrt(h.shape[-1]))
        h = h.reshape(m, n, token_size, token_size)
        h = self.proj_out.forward(h)
        return x + h


class MiddleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1 = ResnetBlock(9, 9)
        self.attn_1 = AttentionBlock()
        self.block_2 = ResnetBlock(9, 9)
    
    def forward(self, h):
        h = self.block_1.forward(h)
        h = self.attn_1.forward(h)
        h = self.block_2.forward(h)
        return h


class Upsample(nn.Module):
    def __init__(self, log2_dim):
        super().__init__()
        n = 2 ** log2_dim
        self.upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(n, n, 3, padding=1)

    def forward(self, x):
        x = self.upsample.forward(x.to(torch.float32))
        x = self.conv.forward(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(
        self, 
        log2_dim_in: int, 
        log2_dim_out: int, 
        has_attention: bool, 
        has_upsample: bool
    ):
        super().__init__()
        self.has_attention = has_attention
        self.has_upsample = has_upsample
        
        self.block = nn.ModuleList([
            ResnetBlock(log2_dim_in, log2_dim_out),
            ResnetBlock(log2_dim_out, log2_dim_out),
            ResnetBlock(log2_dim_out, log2_dim_out)
        ])

        self.attn = nn.ModuleList([
            AttentionBlock(),
            AttentionBlock(),
            AttentionBlock()
        ]) if has_attention else None

        self.upsample = Upsample(log2_dim_out) if has_upsample else None


    def forward(self, h):
        if self.attn is not None:
            for j, (block, attn) in enumerate(zip(self.block, self.attn)):
                h = block.forward(h)
                h = attn.forward(h)
        else:
            for block in self.block:
                h = block.forward(h)
        if self.upsample is not None:
            h = self.upsample.forward(h)
        return h


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_in = nn.Conv2d(256, 512, 3, padding=1)
        self.mid = MiddleLayer()

        self.up = nn.ModuleList([
            UpsampleBlock(7, 7, False, False),
            UpsampleBlock(8, 7, False, True),
            UpsampleBlock(8, 8, False, True),
            UpsampleBlock(9, 8, False, True),
            UpsampleBlock(9, 9, True, True)
        ])

        self.norm_out = nn.GroupNorm(32, 128)
        self.conv_out = nn.Conv2d(128, 3, 3, padding=1)

    def forward(self, z):
        z = self.conv_in.forward(z)
        z = self.mid.forward(z)

        for up in self.up[::-1]:
            z = up.forward(z)

        z = self.norm_out.forward(z)
        z *= torch.sigmoid(z)
        z = self.conv_out.forward(z)
        return z


class VQGanDetokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        vocab_size, embed_size = 16384, 256
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.post_quant_conv = nn.Conv2d(embed_size, embed_size, 1)
        self.decoder = Decoder()

    # def forward(self, is_seamless, z):
    #     z.clamp_(0, self.vocab_size - 1)
    #     grid_size = int(sqrt(z.shape[0]))
    #     token_size = grid_size * 16
        
    #     if is_seamless:
    #         z = z.view([grid_size, grid_size, 16, 16])
    #         z = z.flatten(1, 2).transpose(1, 0).flatten(1, 2)
    #         z = z.flatten().unsqueeze(1)
    #         z = self.embedding.forward(z)
    #         z = z.view((1, token_size, token_size, 256))
    #     else:
    #         z = self.embedding.forward(z)
    #         z = z.view((z.shape[0], 16, 16, 256))

    #     z = z.permute(0, 3, 1, 2).contiguous()
    #     z = self.post_quant_conv.forward(z)
    #     z = self.decoder.forward(z)
    #     z = z.permute(0, 2, 3, 1)
    #     z = z.clip(0.0, 1.0) * 255

    #     if is_seamless:
    #         z = z[0]
    #     else:
    #         z = z.view([grid_size, grid_size, 256, 256, 3])
    #         z = z.flatten(1, 2).transpose(1, 0).flatten(1, 2)

    #     return z

    def forward(self, z):
        z.clamp_(0, self.vocab_size - 1)
        grid_size = int(sqrt(z.shape[0]))
        
        z = self.embedding.forward(z)
        z = z.view((z.shape[0], 16, 16, 256))

        z = z.permute(0, 3, 1, 2).contiguous()
        z = self.post_quant_conv.forward(z)
        z = self.decoder.forward(z)
        z = z.permute(0, 2, 3, 1)
        z = z.clip(0.0, 1.0) * 255

        z = z.view([grid_size, grid_size, 256, 256, 3])
        z = z.flatten(1, 2).transpose(1, 0).flatten(1, 2)
        return z