"""Sprint 9, A2: GQE policy models. Regime A (per-instance, Track B) needs no
conditioning -- a plain decoder-only autoregressive transformer over the A1
vocabulary, one instance = one freshly-initialized model. Regime B (Track C)
reuses the per-variable context encoder pattern from `pce/cloning_model.py`
(raw instance features, cross-attended by the decoder) -- structurally the
same encoder-decoder shape as the Sprint-3 cloning model, just over the much
larger full-gate vocabulary instead of the angle-only one.
"""
import torch
import torch.nn as nn


class GQEDecoderOnly(nn.Module):
    """Regime A: unconditional autoregressive decoder-only transformer.
    Small by design (per-instance, trained from scratch every time)."""

    def __init__(self, vocab_size: int, max_len: int, d_model: int = 96, nhead: int = 4,
                 n_layers: int = 3, ffn_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.max_len = max_len
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, ffn_dim, batch_first=True, dropout=dropout)
        self.decoder = nn.TransformerEncoder(layer, n_layers)  # causal-masked "encoder" = GPT-style
        self.out_proj = nn.Linear(d_model, vocab_size)

    def logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (batch, L). Causal self-attention. Returns (batch, L, vocab)."""
        batch, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(batch, -1)
        h = self.tok_embed(tokens) + self.pos_embed(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(L).to(tokens.device)
        h = self.decoder(h, mask=mask, is_causal=True)
        return self.out_proj(h)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def sample(self, batch: int, bos_id: int, eos_id: int, min_gates: int, temperature: float = 1.0,
               generator=None, device="cpu"):
        """Autoregressive sampling. Returns (batch, max_len) token ids incl.
        BOS/EOS/padding, and (batch, max_len) log-probs of the chosen tokens
        (0 for positions at/after that row's first EOS)."""
        self.eval()
        tokens = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
        logps = torch.zeros((batch, self.max_len), dtype=torch.float64, device=device)
        done = torch.zeros(batch, dtype=torch.bool, device=device)
        for t in range(self.max_len - 1):
            logits = self.logits(tokens)[:, -1, :]
            if t < min_gates:
                logits = logits.clone()
                logits[:, eos_id] = -1e9  # GQCO rule: EOS illegal before min_gates
            probs = torch.softmax(logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            next_tok = torch.where(done, torch.full_like(next_tok, eos_id), next_tok)
            logp = torch.log(probs.gather(1, next_tok.unsqueeze(-1)).squeeze(-1).clamp(min=1e-12))
            logps[:, t + 1] = torch.where(done, torch.zeros_like(logp), logp)
            done = done | (next_tok == eos_id)
            tokens = torch.cat([tokens, next_tok.unsqueeze(-1)], dim=1)
        return tokens, logps

    def sequence_logprob(self, tokens: torch.Tensor, eos_id: int) -> torch.Tensor:
        """Teacher-forced log p(sequence) under the CURRENT parameters, masked
        to stop contributing after each row's first EOS. tokens: (batch, L)."""
        tgt_in, tgt_out = tokens[:, :-1], tokens[:, 1:]
        logits = self.logits(tgt_in)
        logprobs = torch.log_softmax(logits, dim=-1)
        tok_lp = logprobs.gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1)  # (batch, L-1)
        eos_mask = (tgt_out == eos_id)
        first_eos = torch.cumsum(eos_mask.long(), dim=1)
        active = first_eos <= 1  # include the EOS token itself, exclude everything after
        return (tok_lp * active).sum(dim=1)


class GQEConditionalModel(nn.Module):
    """Regime B: per-variable context encoder (Sprint 3's cloning-model
    pattern, raw features, unchanged) cross-attended by a causal decoder over
    the A1 vocabulary."""

    def __init__(self, feat_dim: int, vocab_size: int, max_len: int, d_model: int = 128,
                 nhead: int = 4, enc_layers: int = 2, dec_layers: int = 3, ffn_dim: int = 384,
                 dropout: float = 0.1):
        super().__init__()
        self.max_len = max_len
        self.feat_proj = nn.Linear(feat_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ffn_dim, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, enc_layers, enable_nested_tensor=False)

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, ffn_dim, batch_first=True, dropout=dropout)
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def encode(self, feats, key_padding_mask):
        h = self.feat_proj(feats)
        return self.encoder(h, src_key_padding_mask=key_padding_mask)

    def decode_logits(self, memory, memory_padding_mask, tgt_tokens):
        batch, L = tgt_tokens.shape
        pos = torch.arange(L, device=tgt_tokens.device).unsqueeze(0).expand(batch, -1)
        h = self.tok_embed(tgt_tokens) + self.pos_embed(pos)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L).to(tgt_tokens.device)
        out = self.decoder(h, memory, tgt_mask=causal_mask, memory_key_padding_mask=memory_padding_mask)
        return self.out_proj(out)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def sample(self, feats, key_padding_mask, bos_id, eos_id, min_gates, temperature=1.0, generator=None):
        self.eval()
        batch = feats.shape[0]
        device = feats.device
        memory = self.encode(feats, key_padding_mask)
        tokens = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
        logps = torch.zeros((batch, self.max_len), dtype=torch.float64, device=device)
        done = torch.zeros(batch, dtype=torch.bool, device=device)
        for t in range(self.max_len - 1):
            logits = self.decode_logits(memory, key_padding_mask, tokens)[:, -1, :]
            if t < min_gates:
                logits = logits.clone()
                logits[:, eos_id] = -1e9
            probs = torch.softmax(logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            next_tok = torch.where(done, torch.full_like(next_tok, eos_id), next_tok)
            logp = torch.log(probs.gather(1, next_tok.unsqueeze(-1)).squeeze(-1).clamp(min=1e-12))
            logps[:, t + 1] = torch.where(done, torch.zeros_like(logp), logp)
            done = done | (next_tok == eos_id)
            tokens = torch.cat([tokens, next_tok.unsqueeze(-1)], dim=1)
        return tokens, logps

    def sequence_logprob(self, feats, key_padding_mask, tokens, eos_id):
        memory = self.encode(feats, key_padding_mask)
        tgt_in, tgt_out = tokens[:, :-1], tokens[:, 1:]
        logits = self.decode_logits(memory, key_padding_mask, tgt_in)
        logprobs = torch.log_softmax(logits, dim=-1)
        tok_lp = logprobs.gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1)
        eos_mask = (tgt_out == eos_id)
        first_eos = torch.cumsum(eos_mask.long(), dim=1)
        active = first_eos <= 1
        return (tok_lp * active).sum(dim=1)
