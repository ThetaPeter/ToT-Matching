"""N_{C,P}: Graph Transformer over GI + T5 decoder (GI → NL proof).

Phase-1 stand-in remains ``reasoner.prove_nl``. After training, wrap with
``NCPReasoner`` so the GI→Proof box no longer depends on Grok.
Lean stays a separate tool (do not train N_{P,L}).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tot.ir.schema import ClaimDAG, ClaimNode
from tot.prove.nl_proof import serialize_gi

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for tot.prove.ncp") from exc


def gi_text(gi: ClaimDAG | Mapping[str, Any]) -> str:
    """Serialize GI-only text for the encoder (ids, statements, deps)."""
    if isinstance(gi, ClaimDAG):
        payload = serialize_gi(gi)
    else:
        payload = dict(gi)
    lines = [f"problem {payload.get('problem_id', '')}".strip()]
    for node in payload.get("nodes", []):
        deps = ", ".join(node.get("dependencies") or [])
        lines.append(
            f"{node.get('id')}: {node.get('statement', '')} [deps: {deps}]"
        )
    return "\n".join(lines).strip()


class GraphTransformerEncoder(nn.Module):
    """Small GT encoder over GI nodes (features + dependency edges)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        node_feats: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.in_proj(node_feats)
        return self.encoder(h, mask=attn_mask)


@dataclass
class NCPConfig:
    t5_name: str = "t5-base"
    hidden_dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    max_source_len: int = 256
    max_target_len: int = 256
    lr: float = 5e-4
    stub_decoder: bool = False
    vocab_size: int = 128
    stub_embed_dim: int = 64


class _StubSeq2Seq(nn.Module):
    """Tiny decoder for pytest (no HuggingFace download)."""

    def __init__(self, embed_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.out = nn.Linear(embed_dim, vocab_size)
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

    def forward_loss(
        self,
        prefix: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        # prefix: (1, n_prefix, d); target_ids: (1, t)
        bos = torch.zeros_like(target_ids[:, :1])
        decoder_in = torch.cat([bos, target_ids[:, :-1]], dim=1)
        emb = self.embed(decoder_in)
        # mean-pool prefix as initial state
        h0 = prefix.mean(dim=1, keepdim=True).transpose(0, 1).contiguous()
        hidden, _ = self.rnn(emb, h0)
        logits = self.out(hidden)
        return nn.functional.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_ids.reshape(-1),
        )

    @torch.no_grad()
    def generate(self, prefix: torch.Tensor, max_new_tokens: int = 32) -> list[int]:
        h = prefix.mean(dim=1, keepdim=True).transpose(0, 1).contiguous()
        token = torch.zeros(1, 1, dtype=torch.long, device=prefix.device)
        outs: list[int] = []
        for _ in range(max_new_tokens):
            emb = self.embed(token)
            out, h = self.rnn(emb, h)
            logits = self.out(out[:, -1])
            token = torch.argmax(logits, dim=-1, keepdim=True)
            tid = int(token.item())
            if tid == 1:  # reserved EOS
                break
            outs.append(tid)
        return outs


def _tokenize_stub(text: str, vocab_size: int, max_len: int) -> list[int]:
    ids = [(ord(ch) % (vocab_size - 2)) + 2 for ch in text[:max_len]]
    if not ids:
        ids = [2]
    return ids


class NCPModel(nn.Module):
    """Graph Transformer soft-prefix + T5 (or stub) decoder for GI→proof."""

    def __init__(self, in_dim: int, config: NCPConfig | None = None) -> None:
        super().__init__()
        self.config = config or NCPConfig()
        self.in_dim = int(in_dim)
        self.gt = GraphTransformerEncoder(
            in_dim=self.in_dim,
            hidden_dim=self.config.hidden_dim,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
        )
        self.tokenizer = None
        self.t5 = None
        self.stub = None
        if self.config.stub_decoder:
            self.stub = _StubSeq2Seq(
                embed_dim=self.config.stub_embed_dim,
                vocab_size=self.config.vocab_size,
            )
            self.prefix_proj = nn.Linear(
                self.config.hidden_dim, self.config.stub_embed_dim
            )
        else:
            from transformers import T5ForConditionalGeneration, T5Tokenizer

            self.tokenizer = T5Tokenizer.from_pretrained(self.config.t5_name)
            self.t5 = T5ForConditionalGeneration.from_pretrained(self.config.t5_name)
            t5_dim = int(self.t5.config.d_model)
            self.prefix_proj = nn.Linear(self.config.hidden_dim, t5_dim)

    def _prefix(
        self,
        node_feats: torch.Tensor,
        adj_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self.gt(node_feats, attn_mask=adj_mask)
        return self.prefix_proj(encoded)

    def forward_loss(
        self,
        node_feats: torch.Tensor,
        source_text: str,
        target_text: str,
        adj_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prefix = self._prefix(node_feats, adj_mask=adj_mask)
        if self.stub is not None:
            ids = _tokenize_stub(
                target_text, self.config.vocab_size, self.config.max_target_len
            )
            target = torch.tensor([ids], dtype=torch.long, device=node_feats.device)
            return self.stub.forward_loss(prefix, target)

        assert self.tokenizer is not None and self.t5 is not None
        enc = self.tokenizer(
            source_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_source_len,
        )
        dec = self.tokenizer(
            text_target=target_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_target_len,
        )
        device = node_feats.device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels = dec["input_ids"].to(device)
        inputs_embeds = self.t5.encoder.embed_tokens(input_ids)
        prefix = prefix.to(device)
        inputs_embeds = torch.cat([prefix, inputs_embeds], dim=1)
        prefix_mask = torch.ones(
            prefix.shape[:2], dtype=attention_mask.dtype, device=device
        )
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        out = self.t5(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out.loss

    @torch.no_grad()
    def generate(
        self,
        node_feats: torch.Tensor,
        source_text: str,
        adj_mask: torch.Tensor | None = None,
        max_new_tokens: int = 256,
    ) -> str:
        self.eval()
        prefix = self._prefix(node_feats, adj_mask=adj_mask)
        if self.stub is not None:
            ids = self.stub.generate(prefix, max_new_tokens=max_new_tokens)
            return "".join(chr(2 + (i % 95)) for i in ids)

        assert self.tokenizer is not None and self.t5 is not None
        enc = self.tokenizer(
            source_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_source_len,
        )
        device = node_feats.device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        inputs_embeds = self.t5.encoder.embed_tokens(input_ids)
        prefix = prefix.to(device)
        inputs_embeds = torch.cat([prefix, inputs_embeds], dim=1)
        prefix_mask = torch.ones(
            prefix.shape[:2], dtype=attention_mask.dtype, device=device
        )
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        out_ids = self.t5.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)


def build_node_features(
    gi: ClaimDAG,
    embeddings: Mapping[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (1, n, d) features and bool attention mask (True = blocked)."""
    ids = [node.id for node in gi.nodes]
    feats = []
    for node in gi.nodes:
        emb = np.asarray(embeddings[node.id], dtype=np.float32).reshape(-1)
        extras = np.array(
            [
                0.0 if node.salience is None else float(node.salience),
                0.0 if node.support is None else float(node.support),
                float(len(node.dependencies)),
            ],
            dtype=np.float32,
        )
        feats.append(np.concatenate([emb, extras]))
    mat = torch.as_tensor(np.stack(feats, axis=0), dtype=torch.float32).unsqueeze(0)
    n = len(ids)
    index = {node_id: i for i, node_id in enumerate(ids)}
    allow = torch.eye(n, dtype=torch.bool)
    for node in gi.nodes:
        i = index[node.id]
        for dep in node.dependencies:
            if dep in index:
                j = index[dep]
                allow[i, j] = True
                allow[j, i] = True
    mask = ~allow
    return mat, mask


@dataclass
class NCPExample:
    gi: ClaimDAG
    human_proof: str
    embeddings: Mapping[str, np.ndarray] = field(default_factory=dict)


def train_ncp(
    examples: Sequence[NCPExample],
    *,
    epochs: int = 1,
    config: NCPConfig | None = None,
    device: str | None = None,
) -> NCPModel:
    if not examples:
        raise ValueError("train_ncp needs examples")
    cfg = config or NCPConfig()
    feats0, _ = build_node_features(examples[0].gi, examples[0].embeddings)
    model = NCPModel(in_dim=int(feats0.shape[-1]), config=cfg)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    model.train()
    for _ in range(int(epochs)):
        for example in examples:
            feats, mask = build_node_features(example.gi, example.embeddings)
            loss = model.forward_loss(
                feats.to(dev),
                gi_text(example.gi),
                example.human_proof,
                adj_mask=mask.to(dev),
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
    return model


class NCPReasoner:
    """Drop-in GI→NL reasoner using a trained NCP checkpoint."""

    def __init__(
        self,
        model: NCPModel,
        embeddings: Mapping[str, np.ndarray],
        *,
        base_reasoner: Any | None = None,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.embeddings = embeddings
        self.base_reasoner = base_reasoner
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()

    def expand(self, proof_state: str, last_claim: str, k: int) -> list[str]:
        if self.base_reasoner is None:
            raise RuntimeError("NCPReasoner.expand needs a base_reasoner")
        return self.base_reasoner.expand(proof_state, last_claim, k)

    def score(self, proof_state: str, claim: str) -> float:
        if self.base_reasoner is None:
            raise RuntimeError("NCPReasoner.score needs a base_reasoner")
        return float(self.base_reasoner.score(proof_state, claim))

    def prove_nl(self, gi_payload: Mapping[str, Any] | str) -> str:
        if isinstance(gi_payload, str):
            payload = json.loads(gi_payload)
        else:
            payload = dict(gi_payload)
        gi = ClaimDAG(
            problem_id=str(payload.get("problem_id") or "gi"),
            nodes=[
                ClaimNode(
                    id=str(node["id"]),
                    natural_language=str(node.get("statement") or node["id"]),
                    statement=str(node.get("statement") or ""),
                    dependencies=list(node.get("dependencies") or []),
                )
                for node in payload.get("nodes", [])
            ],
        )
        dim = None
        emb: dict[str, np.ndarray] = {}
        for node in gi.nodes:
            if node.id in self.embeddings:
                vec = np.asarray(self.embeddings[node.id], dtype=np.float32)
                emb[node.id] = vec
                dim = int(vec.shape[0])
        if dim is None:
            dim = 32
        for node in gi.nodes:
            emb.setdefault(node.id, np.zeros(dim, dtype=np.float32))
        feats, mask = build_node_features(gi, emb)
        return self.model.generate(
            feats.to(self.device),
            gi_text(gi),
            adj_mask=mask.to(self.device),
        )


def save_ncp(model: NCPModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "in_dim": model.in_dim,
            "config": model.config.__dict__,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_ncp(path: str | Path) -> NCPModel:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = NCPConfig(**payload["config"])
    model = NCPModel(in_dim=int(payload["in_dim"]), config=config)
    model.load_state_dict(payload["state_dict"])
    return model
