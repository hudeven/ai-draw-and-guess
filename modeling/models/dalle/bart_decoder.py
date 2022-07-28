from typing import Tuple, List
import torch
from torch import nn, LongTensor, FloatTensor, BoolTensor
from .bart_encoder import GLU, AttentionBase

IMAGE_TOKEN_LENGTH = 256


class DecoderCrossAttention(nn.Module):
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
        decoder_state: FloatTensor,
        encoder_state: FloatTensor,
        attention_mask: BoolTensor,
    ) -> FloatTensor:
        keys = self.k_proj.forward(encoder_state)
        values = self.v_proj.forward(encoder_state)
        queries = self.q_proj.forward(decoder_state)
        # return super().forward(keys, values, queries, attention_mask)

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


class DecoderSelfAttention(nn.Module):
    def __init__(self, head_dim: int, embed_size: int):
        super().__init__()
        self.attention_heads = head_dim
        self.embed_size = embed_size

        self.k_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.v_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.q_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.out_proj = nn.Linear(embed_size, embed_size, bias=False)

    def forward(
        self,
        decoder_state: FloatTensor,
        attention_state: FloatTensor,
        attn_mask: BoolTensor,
        token_index: LongTensor,
    ) -> Tuple[FloatTensor, FloatTensor]:
        keys = self.k_proj.forward(decoder_state)
        values = self.v_proj.forward(decoder_state)
        queries = self.q_proj.forward(decoder_state)
        attn_state_new = torch.cat([keys, values]).to(attention_state.dtype)
        attention_state[:, token_index] = attn_state_new
        batch_size = decoder_state.shape[0]
        keys = attention_state[:batch_size]
        values = attention_state[batch_size:]

        # decoder_state = super().forward(keys, values, queries, attn_mask)
        # We need to duplicate the code from AttentionBase for TorchScript to work
        keys = keys.reshape(keys.shape[:2] + (self.attention_heads, -1))
        values = values.reshape(values.shape[:2] + (self.attention_heads, -1))
        queries = queries.reshape(queries.shape[:2] + (self.attention_heads, -1))
        queries /= queries.shape[-1] ** 0.5

        attention_bias = (1 - attn_mask.to(torch.float32)) * -1e12
        attention_weights: FloatTensor = torch.einsum("bqhc,bkhc->bhqk", queries, keys)
        attention_weights += attention_bias[:, None, None, :]
        attention_weights = torch.softmax(attention_weights, -1)
        attention_output: FloatTensor = torch.einsum(
            "bhqk,bkhc->bqhc", attention_weights, values
        )
        shape = attention_output.shape[:2] + (self.embed_size,)
        attention_output = attention_output.reshape(shape)
        decoder_state = self.out_proj.forward(attention_output)        
        return decoder_state, attention_state


class DecoderLayer(nn.Module):
    def __init__(
        self, head_dim: int, embed_size: int, glu_dim: int, device: str
    ):
        super().__init__()
        self.pre_self_attn_layer_norm = nn.LayerNorm(embed_size)
        self.self_attn = DecoderSelfAttention(head_dim, embed_size)
        self.self_attn_layer_norm = nn.LayerNorm(embed_size)
        self.pre_encoder_attn_layer_norm = nn.LayerNorm(embed_size)
        self.encoder_attn = DecoderCrossAttention(head_dim, embed_size)
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_size)
        self.glu = GLU(embed_size, glu_dim)
        self.token_indices = torch.arange(IMAGE_TOKEN_LENGTH, device=device)

    def forward(
        self,
        decoder_state: FloatTensor,
        encoder_state: FloatTensor,
        attention_state: FloatTensor,
        attention_mask: BoolTensor,
        token_index: LongTensor,
    ) -> Tuple[FloatTensor, FloatTensor]:
        # Self Attention
        self_attn_mask = self.token_indices < token_index + 1
        self_attn_mask = self_attn_mask[None][[0] * decoder_state.shape[0]]
        residual = decoder_state
        decoder_state = self.pre_self_attn_layer_norm.forward(decoder_state)
        decoder_state, attention_state = self.self_attn.forward(
            decoder_state=decoder_state,
            attention_state=attention_state,
            attn_mask=self_attn_mask,
            token_index=token_index,
        )
        decoder_state = self.self_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state

        # Cross Attention
        residual = decoder_state
        decoder_state = self.pre_encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = self.encoder_attn.forward(
            decoder_state=decoder_state,
            encoder_state=encoder_state,
            attention_mask=attention_mask,
        )
        decoder_state = self.encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state

        # Feed forward
        residual = decoder_state
        decoder_state = self.glu.forward(decoder_state)
        decoder_state = residual + decoder_state

        return decoder_state, attention_state


class DalleBartDecoder(nn.Module):
    def __init__(
        self,
        image_vocab_size: int,
        embed_size: int,
        attention_heads: int,
        glu_dim: int,
        num_layers: int,
        device: str,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.image_vocab_size = image_vocab_size
        self.embed_tokens = nn.Embedding(image_vocab_size + 1, embed_size)
        self.embed_positions = nn.Embedding(IMAGE_TOKEN_LENGTH, embed_size)
        self.layers: List[DecoderLayer] = nn.ModuleList(
            [
                DecoderLayer(
                    head_dim=attention_heads,
                    embed_size=embed_size,
                    glu_dim=glu_dim,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )
        self.layernorm_embedding = nn.LayerNorm(embed_size)
        self.final_ln = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, image_vocab_size + 1, bias=False)
        self.token_indices = torch.arange(IMAGE_TOKEN_LENGTH, device=device)

    def forward(
        self,
        settings: FloatTensor,
        attention_mask: BoolTensor,
        encoder_state: FloatTensor,
        attention_state: FloatTensor,
        prev_tokens: LongTensor,
        token_index: LongTensor,
    ) -> Tuple[LongTensor, FloatTensor]:
        image_size = encoder_state.shape[0] // 2
        token_index_batched = token_index[[0] * image_size * 2]
        prev_tokens = prev_tokens[list(range(image_size)) * 2]
        prev_tokens.clamp_(0, self.image_vocab_size)
        decoder_state = self.embed_tokens.forward(prev_tokens)
        decoder_state += self.embed_positions.forward(token_index_batched)
        decoder_state = self.layernorm_embedding.forward(decoder_state)
        decoder_state = decoder_state[:, None]
        for i, layer in enumerate(self.layers):
            decoder_state, attention_state[i] = layer.forward(
                decoder_state,
                encoder_state,
                attention_state[i],
                attention_mask,
                token_index,
            )
        decoder_state = self.final_ln(decoder_state)
        logits = self.lm_head(decoder_state)
        temperature = settings[[0]]
        top_k = settings[[1]].to(torch.long)
        supercondition_factor = settings[[2]]
        logits = logits[:, -1, : 16384]
        logits: FloatTensor = (
            logits[:image_size] * (1 - supercondition_factor)
            + logits[image_size:] * supercondition_factor
        )
        logits_sorted, _ = logits.sort(descending=True)
        is_kept = logits >= logits_sorted[:, top_k - 1]
        logits -= logits_sorted[:, [0]]
        logits /= temperature
        logits.exp_()
        logits *= is_kept.to(torch.float32)
        image_tokens = torch.multinomial(logits, 1)[:, 0]
        return image_tokens, attention_state
