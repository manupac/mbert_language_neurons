"""Language-specific FFN neurons in mBERT: LAPE selection, causal ablation, alignment.

All functions are written to be imported by run_experiment.py.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_ID = "google-bert/bert-base-multilingual-cased"
REVISION = "3f076fdb1ab68d5b2880cb87a0886f315b8146f8"  # pinned checkpoint
LANGS = ["eng_Latn", "fra_Latn", "deu_Latn", "rus_Cyrl", "cmn_Hans", "yor_Latn"]
N_LAYERS, N_NEURONS = 12, 3072

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)  # inference only


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, revision=REVISION).eval().to(device)
    return tok, model


def load_flores(code, split):
    """One FLORES+ language as a DataFrame sorted by sentence id (gated dataset: needs an HF token)."""
    df = load_dataset("openlanguagedata/flores_plus", code, split=split).to_pandas()
    df["id"] = df["id"].astype(int)
    return df.sort_values("id").reset_index(drop=True)


def encode(tok, sentences, max_length=256):
    enc = tok(sentences, padding=True, truncation=True, max_length=max_length,
              return_tensors="pt", return_special_tokens_mask=True)
    special = enc.pop("special_tokens_mask").bool()
    keep = enc["attention_mask"].bool() & ~special  # (B, T): real word pieces only
    return enc.to(device), keep.to(device)


class FFNHooks:
    """Counts positive GELU activations and/or zeroes selected FFN neurons."""

    def __init__(self, model, ablate=None):
        self.layers = model.bert.encoder.layer
        self.counts = torch.zeros(len(self.layers), model.config.intermediate_size, device=device)
        self.n_tokens = 0
        self.keep = None  # (B, T) mask, set before each forward pass
        self.ablate = ablate or {}  # {layer index: LongTensor of neuron indices}

    def _hook(self, i):
        def hook(module, inputs, output):  # output: (B, T, 3072)
            if self.keep is not None:
                active = (output > 0) & self.keep.unsqueeze(-1)
                self.counts[i] += active.sum(dim=(0, 1))
            idx = self.ablate.get(i)
            if idx is not None and len(idx) > 0:
                output = output.clone()
                output[..., idx] = 0.0
                return output  # replaces the module output
        return hook

    def __enter__(self):
        self.handles = [layer.intermediate.register_forward_hook(self._hook(i))
                        for i, layer in enumerate(self.layers)]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


def activation_probabilities(model, tok, sentences, batch_size=32):
    """p[l, j] = P(neuron j of layer l is active | a token of these sentences)."""
    with FFNHooks(model) as hooks:
        for i in range(0, len(sentences), batch_size):
            enc, keep = encode(tok, sentences[i:i + batch_size])
            hooks.keep, hooks.n_tokens = keep, hooks.n_tokens + int(keep.sum())
            model.bert(**enc)
    return (hooks.counts / hooks.n_tokens).cpu().numpy()  # (12, 3072)


def lape_select(P, langs, top_frac=0.01, pct=95):
    """LAPE: keep the lowest-entropy `top_frac` of neurons, assign them to languages
    whose activation probability exceeds the `pct`-th percentile of all probabilities."""
    S = np.stack([P[c] for c in langs])  # (L, 12, 3072)
    alive = (S > 0).any(axis=0)
    tot = S.sum(axis=0)
    p_norm = S / np.where(tot > 0, tot, 1)  # each language's share of the neuron's activation
    logp = np.log(np.where(p_norm > 0, p_norm, 1))
    ent = -(p_norm * logp).sum(axis=0)
    ent[~alive] = np.inf  # dead neurons are never selected

    n_keep = int(top_frac * ent.size)
    kept = np.zeros(ent.size, dtype=bool)
    kept[np.argsort(ent.ravel())[:n_keep]] = True
    kept = kept.reshape(ent.shape)

    tau = np.percentile(S, pct)
    assign = (S > tau) & kept[None]
    return dict(langs=list(langs), ent=ent, kept=kept, tau=tau, assign=assign)


def neuron_set(res, lang):
    k = res["langs"].index(lang)
    return set(np.flatnonzero(res["assign"][k].ravel()))


def masked_batches(tok, sentences, p=0.15, seed=0, batch_size=32):
    """Masks are drawn once, so the same positions are masked in every condition."""
    g = torch.Generator().manual_seed(seed)
    batches = []
    for i in range(0, len(sentences), batch_size):
        enc, keep = encode(tok, sentences[i:i + batch_size])
        m = ((torch.rand(keep.shape, generator=g) < p).to(device)) & keep
        labels = enc["input_ids"].masked_fill(~m, -100)
        enc["input_ids"] = enc["input_ids"].masked_fill(m, tok.mask_token_id)
        batches.append((enc, labels))
    return batches


def mlm_loss(model, batches, ablate=None):
    """Mean cross-entropy (nats) on masked positions, optionally with neurons zeroed."""
    total, n = 0.0, 0
    with FFNHooks(model, ablate=ablate):
        for enc, labels in batches:
            h = model.bert(**enc).last_hidden_state
            m = labels != -100
            logits = model.cls(h[m])
            total += F.cross_entropy(logits, labels[m], reduction="sum").item()
            n += int(m.sum())
    return total / n


def ablation_for(res, lang):
    """{layer: LongTensor of neurons assigned to `lang`}."""
    k = res["langs"].index(lang)
    out = {}
    for l in range(N_LAYERS):
        idx = np.flatnonzero(res["assign"][k, l])
        if len(idx) > 0:
            out[l] = torch.as_tensor(idx, dtype=torch.long, device=device)
    return out


def random_ablation(res, k, rng):
    """Same number of neurons per layer as language k, drawn from neurons assigned to NO language."""
    assign = res["assign"]
    unassigned = ~assign.any(axis=0)
    out = {}
    for l in range(N_LAYERS):
        n_l = int(assign[k, l].sum())
        if n_l == 0:
            continue
        idx = rng.choice(np.flatnonzero(unassigned[l]), size=n_l, replace=False)
        out[l] = torch.as_tensor(idx, dtype=torch.long, device=device)
    return out


def alignment_r(model, tok, dev_sentences, res, langs, seed=0, n_dims=5):
    """r_j = ||Q^T w_j|| / ||w_j||: how much of a neuron's output weight vector lies in the
    subspace spanned by the centred per-language mean representations. Selected vs. random."""
    means = []
    for c in langs:
        reps = []
        for i in range(0, len(dev_sentences[c]), 32):
            enc, keep = encode(tok, dev_sentences[c][i:i + 32])
            hs = torch.stack(model.bert(**enc, output_hidden_states=True).hidden_states)  # (13,B,T,768)
            w = keep.unsqueeze(-1).float()
            reps.append(((hs * w).sum(2) / w.sum(1)).cpu().numpy())  # (13,B,768)
        means.append(np.concatenate(reps, axis=1).mean(axis=1))  # (13,768)
    means = np.stack(means)  # (L, 13, 768)

    Qs = []
    for l in range(N_LAYERS):
        M = means[:, l + 1, :]
        M = M - M.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        Qs.append(Vt[:n_dims].T)  # (768, 5)

    def r_values(l, idx):
        W = model.bert.encoder.layer[l].output.dense.weight.detach().cpu().numpy()  # (768, 3072)
        w = W[:, idx]
        return np.linalg.norm(Qs[l].T @ w, axis=0) / np.linalg.norm(w, axis=0)

    unassigned = ~res["assign"].any(axis=0)
    rng = np.random.default_rng(seed)
    sel, rnd = [], []
    for l in range(N_LAYERS):
        idx = np.flatnonzero(res["assign"][:, l].any(axis=0))
        if len(idx) == 0:
            continue
        sel.append(r_values(l, idx))
        pool = np.flatnonzero(unassigned[l])
        rnd.append(r_values(l, rng.choice(pool, size=len(idx), replace=False)))
    sel, rnd = np.concatenate(sel), np.concatenate(rnd)
    return sel.mean(), rnd.mean(), float(np.sqrt(n_dims / 768))
