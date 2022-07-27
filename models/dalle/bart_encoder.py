from typing import List
import torch
from torch import nn, BoolTensor, FloatTensor, LongTensor


class GLU(nn.Module):
    def __init__(self, input_dim: int, middle_dim: int):
        super().__init__()
        self.gelu = nn.GELU()
        self.ln0 = nn.LayerNorm(input_dim)
        self.ln1 = nn.LayerNorm(middle_dim)
        self.fc0 = nn.Linear(input_dim, middle_dim, bias=False)
        self.fc1 = nn.Linear(input_dim, middle_dim, bias=False)
        self.fc2 = nn.Linear(middle_dim, input_dim, bias=False)

    def forward(self, z: FloatTensor) -> FloatTensor:
        z = self.ln0.forward(z)
        w = self.fc0.forward(z)
        w = self.gelu.forward(w)
        v = self.fc1.forward(z)
        z = self.ln1.forward(w * v)
        z = self.fc2.forward(z)
        return z


class AttentionBase(nn.Module):
    def __init__(self, attention_heads: int, embed_size: int):
        super().__init__()
        self.attention_heads = attention_heads
        self.embed_size = embed_size

        self.k_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.v_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.q_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.out_proj = nn.Linear(embed_size, embed_size, bias=False)

    def forward(
        self,
        keys: FloatTensor,
        values: FloatTensor,
        queries: FloatTensor,
        attention_mask: BoolTensor,
    ) -> FloatTensor:
        keys = keys.reshape(keys.shape[:2] + (self.attention_heads, -1))
        values = values.reshape(values.shape[:2] + (self.attention_heads, -1))
        queries = queries.reshape(queries.shape[:2] + (self.attention_heads, -1))
        queries /= queries.shape[-1] ** 0.5

        attention_bias = (1 - attention_mask.to(torch.float32)) * -1e12
        attention_weights: FloatTensor = torch.einsum("bqhc,bkhc->bhqk", queries, keys)
        attention_weights += attention_bias[:, None, None, :]
        attention_weights = torch.softmax(attention_weights, -1)
        attention_output: FloatTensor = torch.einsum(
            "bhqk,bkhc->bqhc", attention_weights, values
        )
        shape = attention_output.shape[:2] + (self.embed_size,)
        attention_output = attention_output.reshape(shape)
        attention_output = self.out_proj.forward(attention_output)
        return attention_output


class EncoderSelfAttention(AttentionBase):
    def forward(
        self, encoder_state: FloatTensor, attention_mask: BoolTensor
    ) -> FloatTensor:
        keys = self.k_proj.forward(encoder_state)
        values = self.v_proj.forward(encoder_state)
        queries = self.q_proj.forward(encoder_state)
        return super().forward(keys, values, queries, attention_mask)


class EncoderLayer(nn.Module):
    def __init__(self, embed_size: int, attention_heads: int, glu_dim: int):
        super().__init__()
        self.pre_self_attn_layer_norm = nn.LayerNorm(embed_size)
        self.self_attn = EncoderSelfAttention(attention_heads, embed_size)
        self.self_attn_layer_norm = nn.LayerNorm(embed_size)
        self.glu = GLU(embed_size, glu_dim)

    def forward(
        self, encoder_state: FloatTensor, attention_mask: BoolTensor
    ) -> FloatTensor:
        residual = encoder_state
        encoder_state = self.pre_self_attn_layer_norm.forward(encoder_state)
        encoder_state = self.self_attn.forward(encoder_state, attention_mask)
        encoder_state = self.self_attn_layer_norm.forward(encoder_state)
        encoder_state = residual + encoder_state
        residual = encoder_state
        encoder_state = self.glu.forward(encoder_state)
        encoder_state = residual + encoder_state
        return encoder_state


class DalleBartEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_size: int,
        attention_heads: int,
        encoder_vocab_size: int,
        max_text_length: int,
        glu_dim: int,
        device: str,
    ):
        super().__init__()
        self.encoder_vocab_size = encoder_vocab_size
        self.embed_tokens = nn.Embedding(encoder_vocab_size, embed_size)
        self.embed_positions = nn.Embedding(max_text_length, embed_size)
        self.layers: List[EncoderLayer] = nn.ModuleList(
            [
                EncoderLayer(
                    embed_size=embed_size,
                    attention_heads=attention_heads,
                    glu_dim=glu_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.layernorm_embedding = nn.LayerNorm(embed_size)
        self.final_ln = nn.LayerNorm(embed_size)
        token_indices = torch.arange(max_text_length, device=device)
        self.pose_tokens = torch.stack([token_indices] * 2)

    def forward(self, text_tokens: LongTensor) -> FloatTensor:
        attention_mask = text_tokens.not_equal(1)
        encoder_state = self.embed_tokens.forward(
            text_tokens
        ) + self.embed_positions.forward(self.pose_tokens)
        encoder_state = self.layernorm_embedding.forward(encoder_state)
        for layer in self.layers:
            encoder_state = layer.forward(encoder_state, attention_mask)
        encoder_state = self.final_ln.forward(encoder_state)
        return encoder_state
