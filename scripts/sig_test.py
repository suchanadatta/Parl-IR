#!/usr/bin/env python3
"""
Paired significance tests for the ParliR results table, and the table body with markers.

Test: paired randomisation (sign-flip permutation) test on the per-topic scores, two-sided, comparing every
system with a reference system (default: Hybrid). p-values are Holm-corrected across the systems compared,
separately for each metric. A cell gets a marker (default a star) when the corrected p < alpha.

Input: the per_query_<name>.tsv files written by parlir_eval.py (same topics in each).

  python parlir_significance.py --dir eval/ --systems BM25 BGE-M3 Hybrid +CE Filter Boost Quota \
         --reference Hybrid --standard BM25 BGE-M3 Hybrid +CE --category Filter Boost Quota --out-dir sig/

Systems without a per_query file (e.g. +CE before it is run) are shown as '--'.
Outputs: pvalues.tsv (raw and Holm-corrected p, mean difference), table_body.tex (rows with bold best per half
and a marker for significant differences from the reference).
"""
import argparse
import csv
from pathlib import Path

import numpy as np

# (group, label, column, higher_is_better)
ROWS = [
    ("Relevance", "MRR", "MRR", True), ("Relevance", "nDCG@10", "nDCG@10", True),
    ("Relevance", "MAP@100", "MAP", True), ("Relevance", "Recall@10", "Recall@10", True),
    ("Composition", "CCS@10", "CCS@10", True), ("Composition", "CCS@20", "CCS@20", True),
    ("Composition", r"$\mathrm{CCS}_{\mathrm{rel}}@10$", "CCS_rel@10", True),
    ("Composition", r"$\mathrm{CCS}_{\mathrm{rel}}@20$", "CCS_rel@20", True),
    ("Category-weighted", "MCWAP", "MCWAP", True), ("Category-weighted", "cwnDCG@10", "cwnDCG@10", True),
    ("Misleading", r"MER@10 $\downarrow$", "MER@10", False), ("Misleading", r"MER@20 $\downarrow$", "MER@20", False),
]
NO_MARK = {("Filter", "MER@10"), ("Filter", "MER@20")}      # zero by construction, not a finding


def read_pq(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return {r["query_id"]: {k: (float(v) if v != "" else None) for k, v in r.items() if k != "query_id"} for r in rows}


def randomisation_p(d, perms, rng):
    """Two-sided paired sign-flip test of mean(d) = 0."""
    d = np.asarray(d, dtype=np.float64)
    if not d.any():
        return 1.0
    obs = abs(d.mean())
    count, done, chunk = 0, 0, 20000
    while done < perms:
        m = min(chunk, perms - done)
        signs = rng.integers(0, 2, size=(m, len(d)), dtype=np.int8) * 2 - 1
        count += int((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).sum())
        done += m
    return (count + 1) / (perms + 1)


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="folder with per_query_<name>.tsv files")
    ap.add_argument("--systems", nargs="+", required=True, help="table columns in order")
    ap.add_argument("--reference", default="Hybrid")
    ap.add_argument("--standard", nargs="+", required=True, help="systems of the standard half")
    ap.add_argument("--category", nargs="+", required=True, help="systems of the category-aware half")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--perms", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--marker", default="$^*$", help="LaTeX marker for significant cells (the table's dagger is used for the category-aware group)")
    ap.add_argument("--no-holm", action="store_true", help="use uncorrected p-values")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    pq = {}
    for s in a.systems:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)
        p = Path(a.dir) / f"per_query_{safe}.tsv"
        if p.exists():
            pq[s] = read_pq(p)
    ref = pq[a.reference]
    rng = np.random.default_rng(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    others = [s for s in a.systems if s in pq and s != a.reference]
    table = {}                               # (system, column) -> (mean, significant)
    lines = ["metric\tsystem\tmean\tref_mean\tmean_diff\tp_raw\tp_adjusted\tn_topics\tsignificant"]
    for g, label, col, higher in ROWS:
        raw, diffs, means, rmeans, ns = [], [], [], [], []
        for s in others:
            common = [q for q in ref if q in pq[s] and ref[q].get(col) is not None and pq[s][q].get(col) is not None]
            x = np.array([pq[s][q][col] for q in common]); y = np.array([ref[q][col] for q in common])
            raw.append(randomisation_p(x - y, a.perms, rng))
            diffs.append(float((x - y).mean())); means.append(float(x.mean())); rmeans.append(float(y.mean())); ns.append(len(common))
        adj = np.array(raw) if a.no_holm else holm(np.array(raw))
        for s, r, ad, df, m, rm, n in zip(others, raw, adj, diffs, means, rmeans, ns):
            sig = bool(ad < a.alpha) and (s, col) not in NO_MARK
            lines.append(f"{col}\t{s}\t{m:.4f}\t{rm:.4f}\t{df:+.4f}\t{r:.5f}\t{ad:.5f}\t{n}\t{'yes' if sig else ''}")
            table[(s, col)] = sig
        for s in a.systems:
            if s in pq:
                vals = [pq[s][q][col] for q in pq[s] if pq[s][q].get(col) is not None]
                table[(s, col, "mean")] = sum(vals) / len(vals)
    (out / "pvalues.tsv").write_text("\n".join(lines) + "\n")

    body, prev = [], None
    for g, label, col, higher in ROWS:
        best = {}
        for half in (a.standard, a.category):
            vals = {s: round(table[(s, col, "mean")], 3) for s in half if s in pq}
            if vals:
                b = max(vals.values()) if higher else min(vals.values())
                best.update({s: v == b for s, v in vals.items()})
        cells = []
        for s in a.systems:
            if s not in pq:
                cells.append("--")
                continue
            c = f"{table[(s, col, 'mean')]:.3f}"
            if best.get(s):
                c = f"\\textbf{{{c}}}"
            if table.get((s, col)):
                c += a.marker
            cells.append(c)
        if prev and g != prev:
            body.append("\\midrule")
        first = g != prev
        body.append((f"\\multirow{{{sum(1 for r in ROWS if r[0] == g)}}}{{*}}{{{g}}} " if first else "") +
                    f"& {label} & " + " & ".join(cells) + " \\\\")
        prev = g
    (out / "table_body.tex").write_text("\n".join(body) + "\n")
    n_sig = sum(1 for l in lines[1:] if l.endswith("yes"))
    print(f"{len(lines) - 1} comparisons against {a.reference}; {n_sig} significant at alpha={a.alpha} "
          f"({'uncorrected' if a.no_holm else 'Holm-corrected'}, {a.perms} permutations)")
    print(f"written to {out}/ (pvalues.tsv, table_body.tex)")


if __name__ == "__main__":
    main()