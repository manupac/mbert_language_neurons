"""Run the full experiment: activation probabilities -> LAPE -> ablation -> controls -> alignment.

Usage:
    export HF_TOKEN=...            # FLORES+ is gated: accept its terms on Hugging Face first
    python run_experiment.py [--n-draws 3] [--mask-seed 0] [--n-sentences 300]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import lape as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-draws", type=int, default=3, help="random-control draws per language")
    ap.add_argument("--mask-seed", type=int, default=0, help="seed for the MLM masking pattern")
    ap.add_argument("--n-sentences", type=int, default=300, help="devtest sentences per language")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache", default="cache")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(f"{args.out}/figures", exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)

    langs = L.LANGS
    tok, model = L.load_model()
    dev = {c: L.load_flores(c, "dev") for c in langs}
    devtest = {c: L.load_flores(c, "devtest") for c in langs}
    dev_text = {c: dev[c]["text"].tolist() for c in langs}

    # 1) activation probabilities P[lang] in [0,1]^(12 x 3072), from the dev split (cached)
    cache_file = f"{args.cache}/P_dev.npz"
    if os.path.exists(cache_file):
        P = dict(np.load(cache_file))
    else:
        P = {c: L.activation_probabilities(model, tok, dev_text[c]) for c in langs}
        np.savez_compressed(cache_file, **P)

    # 2) LAPE selection (bottom 1% entropy, assign above the 95th percentile)
    res = L.lape_select(P, langs)
    assign = res["assign"]
    per_layer = pd.DataFrame(assign.sum(axis=2), index=langs)
    per_layer.to_csv(f"{args.out}/neurons_per_layer.csv")
    n_langs = assign.sum(axis=0)
    summary = {
        "tau": float(res["tau"]),
        "kept_neurons": int(res["kept"].sum()),
        "assigned_to_1_language": int((res["kept"] & (n_langs == 1)).sum()),
        "assigned_to_0_languages": int((res["kept"] & (n_langs == 0)).sum()),
        "assigned_to_more_than_1": int((res["kept"] & (n_langs > 1)).sum()),
        "neurons_per_language": {c: int(assign[i].sum()) for i, c in enumerate(langs)},
    }

    # stability: re-run without German, Jaccard on French neurons
    res_nodeu = L.lape_select(P, [c for c in langs if c != "deu_Latn"])
    a, b = L.neuron_set(res, "fra_Latn"), L.neuron_set(res_nodeu, "fra_Latn")
    summary["french_jaccard_without_german"] = len(a & b) / len(a | b)

    # 3) causal ablation on devtest (masks drawn once, shared by all conditions)
    batches = {c: L.masked_batches(tok, devtest[c]["text"].tolist()[:args.n_sentences],
                                   seed=args.mask_seed) for c in langs}
    baseline = {c: L.mlm_loss(model, batches[c]) for c in langs}
    assert abs(L.mlm_loss(model, batches["fra_Latn"], ablate={}) - baseline["fra_Latn"]) < 1e-9

    increase = pd.DataFrame(index=langs, columns=langs, dtype=float)  # rows: ablated, cols: evaluated
    for c in langs:
        ab = L.ablation_for(res, c)
        for e in langs:
            increase.loc[c, e] = L.mlm_loss(model, batches[e], ablate=ab) - baseline[e]

    # 4) random control: same per-layer count, drawn from neurons assigned to no language
    control_mean = pd.DataFrame(0.0, index=langs, columns=langs)
    for k_i, k in enumerate(langs):
        per_draw = []
        for d in range(args.n_draws):
            rng = np.random.default_rng(1000 * k_i + d)
            ab = L.random_ablation(res, k_i, rng)
            per_draw.append([L.mlm_loss(model, batches[e], ablate=ab) - baseline[e] for e in langs])
        control_mean.loc[k] = np.mean(per_draw, axis=0)

    # 5) output-direction alignment with the language-mean subspace
    r_sel, r_rnd, r_exp = L.alignment_r(model, tok, dev_text, res, langs)
    summary.update(r_selected=float(r_sel), r_random=float(r_rnd), r_expected_random_direction=r_exp)

    # outputs
    pd.Series(baseline).to_csv(f"{args.out}/baseline_loss.csv", header=["mlm_loss_nats"])
    increase.to_csv(f"{args.out}/loss_increase_selected.csv")
    control_mean.to_csv(f"{args.out}/loss_increase_random_control.csv")
    (increase - control_mean).to_csv(f"{args.out}/loss_increase_selected_minus_control.csv")
    pd.Series(summary).to_json(f"{args.out}/summary.json", indent=2)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c, row in per_layer.iterrows():
        ax.plot(range(L.N_LAYERS), row.values, marker="o", label=c)
    ax.set_xlabel("Layer"); ax.set_ylabel("Number of selected neurons"); ax.set_xticks(range(L.N_LAYERS))
    ax.set_title("Language-specific neurons per layer (LAPE, lowest 1% entropy)")
    ax.legend(); plt.tight_layout()
    plt.savefig(f"{args.out}/figures/neurons_per_layer.png", dpi=150)

    rel = {c: 100 * increase.loc[c, c] / baseline[c] for c in langs}
    print("Baseline MLM loss:", {c: round(v, 3) for c, v in baseline.items()})
    print("Relative loss increase on own language (%):", {c: round(v, 2) for c, v in rel.items()})
    print(summary)


if __name__ == "__main__":
    main()
