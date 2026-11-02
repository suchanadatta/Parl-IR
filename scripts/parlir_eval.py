#!/usr/bin/env python3
"""
ParliR evaluation in one file: every metric of Table 2 for any number of runs.

  Relevance          MRR, nDCG@10, MAP, Recall@10
  Composition        CCS@10, CCS@20, CCS_rel@10, CCS_rel@20
  Category-weighted  MCWAP, cwnDCG@10
  Misleading         MER@10, MER@20            (lower is better)
  Diagnostics        unjudged@10, CWAP coverage, number of evaluable queries

INPUTS
  --runs     TREC run files. Either 6 columns   qid Q0 doc rank score run_name
                          or 8 columns          qid Q0 doc rank score text category run_name
             (category = labels joined by '|' or ',').  Ordered by the rank column.
  --qrels    qrels with categories (preferred): qid<TAB>doc<TAB>rel<TAB>category   e.g.  12  A9-0068  1  position,proceedings
             or standard TREC qrels  qid 0 doc rel.  When categories are in the qrels they are used for every
             judged passage (they override the run file's category column), so all systems share one labelling.
  --queries  TSV or CSV with a header row; columns  id title description narrative decision position proceedings actors institution
             Weights may be on any scale (e.g. sum to 10); they are normalised per query.
  --docs     only for retrieved passages that are not judged and whose run has no category column:
             JSONL with  id (or doc_id) and category.  Usually not needed.
  --depth N  only the top-N of each run is scored (default 100), so MAP is comparable across runs of different lengths
  --fix-minutes  treat every passage whose id contains '-MIN-' (Minutes) as decision+proceedings (Table 1)

  python parlir_eval.py --runs bm25.run bge.run hybrid.run hybrid_ce.run filter.run boost.run quota.run \
         --names BM25 BGE-M3 Hybrid +CE Filter Boost Quota --qrels qrels.txt --queries queries.tsv \
         --out-dir eval/ [--exclude dev_ids.txt] [--extended] [--fix-minutes]

OUTPUTS (in --out-dir)  summary.tsv  results_table.tex (Table 2 body, best in bold)  per_query_<name>.tsv
Unjudged passages count as non-relevant. All scores are macro-averaged over the queries common to
every run, the qrels and the queries file.
"""
import argparse
import csv
import io
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

CATS = ["decision", "position", "proceedings", "actors", "institution"]


# ================================================================== readers
def split_cats(s):
    """'position,proceedings' or 'position|proceedings' -> {'position', 'proceedings'}."""
    return {c.strip() for c in re.split(r"[|,;]", s or "") if c.strip() in CATS}


def read_queries(path):
    """{qid: {category: weight}} normalised to sum 1. Skips the example row ('ex') and all-zero rows."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        text = f.read()
    if "\t" in text.split("\n", 1)[0]:                       # TSV (narratives may contain quotes)
        rows = list(csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE))
    else:                                                    # CSV, e.g. target_vectors.csv (quoted titles)
        rows = list(csv.reader(io.StringIO(text)))
    header = [h.strip().lower() for h in rows[0]]
    idx = {c: header.index(c) for c in CATS if c in header}
    if not idx:
        sys.exit("queries file: no weight columns found (expected: " + ", ".join(CATS) + ")")
    missing = [c for c in CATS if c not in idx]
    if missing:
        print(f"WARNING: queries file has no column for {missing}; their weight is taken as 0 "
              f"(inferred as the remainder to 10 for 'institution' only if the others sum below 10)", file=sys.stderr)
    out = {}
    for row in rows[1:]:
        if len(row) <= max(idx.values()) or not row[0].strip() or row[0].strip().lower() == "ex":
            continue
        try:
            w = {c: float(row[idx[c]]) if c in idx else 0.0 for c in CATS}
        except ValueError:
            continue
        if "institution" not in idx:
            w["institution"] = max(0.0, 10.0 - sum(w.values()))
        s = sum(w.values())
        if s <= 0:
            print(f"WARNING: query {row[0]} has all-zero weights, skipped", file=sys.stderr)
            continue
        out[row[0].strip()] = {c: v / s for c, v in w.items()}
    return out


def read_qrels(path):
    """Standard qrels 'qid 0 doc rel', or the 4-column file 'qid<TAB>doc<TAB>rel<TAB>category' (header skipped)."""
    q = defaultdict(dict)
    for line in open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t") if "\t" in line else line.split()
        if len(p) >= 4 and p[3].strip().lstrip("-").isdigit():          # qid iteration doc rel
            q[p[0]][p[2]] = int(p[3])
        elif len(p) >= 3 and p[2].strip().lstrip("-").isdigit():        # qid doc rel [category]
            q[p[0]][p[1]] = int(p[2])
    return q


def read_qrel_categories(path):
    """{doc_id: category_set} from qrels in the 'qid doc rel category' layout (empty for plain TREC qrels)."""
    cats = {}
    for line in open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t") if "\t" in line else line.split()
        if len(p) >= 4 and p[2] in ("0", "1") and p[3] not in ("0", "1"):
            cats[p[1]] = split_cats(p[3])
    return cats


def read_docs(path):
    """{doc_id: (text, category_set)} from a JSONL with id|doc_id, category, optional text|doc_text."""
    info = {}
    if not path:
        return info
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            d = r.get("id", r.get("doc_id"))
            c = r.get("category", [])
            cats = {x for x in c if x in CATS} if isinstance(c, list) else split_cats(c)
            info[d] = (r.get("text", r.get("doc_text", "")) or "", cats)
    return info


def read_run(path, docs=None, cat_map=None, fix_minutes=False):
    """Return (run_name, {qid: [dict(doc, score, text, cats, cats_str)] in rank order}).
    Category priority: cat_map (from the qrels) > the run's own category column > docs.
    fix_minutes: ids containing '-MIN-' become {decision, proceedings}."""
    docs, cat_map = docs or {}, cat_map or {}
    rows, name, no_cat, changed = defaultdict(list), Path(path).stem, 0, set()
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        p = line.rstrip("\n").split("\t") if "\t" in line else line.split()
        if len(p) != 6 and len(p) < 8:
            sys.exit(f"{path}:{n}: expected 6 or 8 columns, got {len(p)}")
        text, cats_str = (p[5], p[-2]) if len(p) >= 8 else ("", "")
        run_cats = split_cats(cats_str)
        cats = run_cats or docs.get(p[2], ("", set()))[1]
        if p[2] in cat_map:
            if run_cats and run_cats != cat_map[p[2]]:
                changed.add(p[2])
            cats = cat_map[p[2]]
        if fix_minutes and "-MIN-" in p[2]:
            cats = {"decision", "proceedings"}
        if not text:
            text = docs.get(p[2], ("", set()))[0]
        no_cat += 0 if cats else 1
        rows[p[0]].append((int(p[3]), dict(doc=p[2], score=float(p[4]), text=text, cats=cats,
                                           cats_str="|".join(sorted(cats)))))
        name = p[-1]
    if no_cat:
        print(f"WARNING: {path}: {no_cat} retrieved passages have no category (category metrics will ignore them; "
              f"use qrels with categories, an 8-column run, or --docs)", file=sys.stderr)
    if changed:
        print(f"NOTE: {path}: {len(changed)} passages have a different category in the run file than in the qrels; "
              f"the qrels labels were used", file=sys.stderr)
    return name, {q: [d for _, d in sorted(v, key=lambda x: x[0])] for q, v in rows.items()}


# ================================================================== metrics
def reciprocal_rank(rels):
    for i, r in enumerate(rels, 1):
        if r:
            return 1.0 / i
    return 0.0


def average_precision(rels, n_rel):
    hits, s = 0, 0.0
    for i, r in enumerate(rels, 1):
        if r:
            hits += 1
            s += hits / i
    return s / n_rel if n_rel else 0.0


def ndcg_at_k(rels, n_rel, k):
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rels[:k], 1))
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(k, n_rel) + 1))
    return dcg / idcg if idcg else 0.0


def recall_at_k(rels, n_rel, k):
    return sum(rels[:k]) / n_rel if n_rel else 0.0


def ccs_at_k(cats_list, w, k):
    """CCS@k = 1 - TVD(w, p@k); p = category counts / total assignments in the top-k. 0 if no labels."""
    counts = {c: 0 for c in CATS}
    for cats in cats_list[:k]:
        for c in cats:
            counts[c] += 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return 1.0 - 0.5 * sum(abs(w[c] - counts[c] / total) for c in CATS)


def ccs_rel_at_k(cats_list, rels, w, k, n_rel):
    """CCS over the relevant passages of the top-k. None if R_q empty; 0 if none retrieved."""
    if n_rel == 0:
        return None
    rc = [c for c, r in zip(cats_list[:k], rels[:k]) if r]
    return ccs_at_k(rc, w, len(rc)) if rc else 0.0


def cwap(ranked, qrel, w, doc_cats):
    """CWAP(q) and the share of positive target weight retained (C+). (None, coverage) if C+ is empty."""
    R = {c: 0 for c in CATS}
    for d, y in qrel.items():
        if y == 1:
            for c in doc_cats.get(d, ()):
                R[c] += 1
    pos = [c for c in CATS if w[c] > 0]
    plus = [c for c in pos if R[c] > 0]
    cov = sum(w[c] for c in plus) / sum(w[c] for c in pos) if pos else 0.0
    if not plus:
        return None, cov
    num = 0.0
    for c in plus:
        sub = [qrel.get(x["doc"], 0) for x in ranked if c in x["cats"]]
        num += w[c] * average_precision(sub, R[c])
    return num / sum(w[c] for c in plus), cov


def cw_ndcg_at_k(ranked, qrel, w, doc_cats, k):
    """gain(d) = y(d) * sum of target weights of d's categories; ideal = judged relevant sorted by gain."""
    g = [sum(w[c] for c in x["cats"]) if qrel.get(x["doc"], 0) == 1 else 0.0 for x in ranked[:k]]
    dcg = sum(v / math.log2(i + 1) for i, v in enumerate(g, 1))
    ideal = sorted((sum(w[c] for c in doc_cats.get(d, ())) for d, y in qrel.items() if y == 1), reverse=True)[:k]
    idcg = sum(v / math.log2(i + 1) for i, v in enumerate(ideal, 1))
    return dcg / idcg if idcg > 0 else None


def mer_at_k(cats_list, rels, w, k):
    """Share of the top-k that is relevant but only in categories with zero target weight (lower is better)."""
    return sum(1 for cats, r in zip(cats_list[:k], rels[:k]) if r and all(w[c] == 0 for c in cats)) / k


# ================================================================== per query / aggregation
# (group, table label, key, higher_is_better)
TABLE = [
    ("Relevance", "MRR", "MRR", True), ("Relevance", "nDCG@10", "nDCG@10", True),
    ("Relevance", "MAP", "MAP", True), ("Relevance", "Recall@10", "Recall@10", True),
    ("Composition", "CCS@10", "CCS@10", True), ("Composition", "CCS@20", "CCS@20", True),
    ("Composition", r"CCS$_{\mathrm{rel}}$@10", "CCS_rel@10", True),
    ("Composition", r"CCS$_{\mathrm{rel}}$@20", "CCS_rel@20", True),
    ("Category-weighted", "MCWAP", "MCWAP", True), ("Category-weighted", "cwnDCG@10", "cwnDCG@10", True),
    ("Misleading", r"MER@10$\downarrow$", "MER@10", False), ("Misleading", r"MER@20$\downarrow$", "MER@20", False),
]
EXTENDED = ["P@10", "Success@10", "Recall@100", "nDCG@20", "cwnDCG@20", "CCS@50"]


def evaluate_query(ranked, qrel, w, doc_cats):
    cl = [x["cats"] for x in ranked]
    rels = [qrel.get(x["doc"], 0) for x in ranked]       # unjudged = non-relevant
    n_rel = sum(1 for y in qrel.values() if y == 1)
    has = n_rel > 0
    o = {"n_rel": n_rel,
         "MRR": reciprocal_rank(rels) if has else None,
         "MAP": average_precision(rels, n_rel) if has else None}
    for k in (10, 20):
        o[f"nDCG@{k}"] = ndcg_at_k(rels, n_rel, k) if has else None
        o[f"cwnDCG@{k}"] = cw_ndcg_at_k(ranked, qrel, w, doc_cats, k)
        o[f"CCS@{k}"] = ccs_at_k(cl, w, k)
        o[f"CCS_rel@{k}"] = ccs_rel_at_k(cl, rels, w, k, n_rel)
        o[f"MER@{k}"] = mer_at_k(cl, rels, w, k) if has else None
    for k in (10, 100):
        o[f"Recall@{k}"] = recall_at_k(rels, n_rel, k) if has else None
    o["P@10"] = sum(rels[:10]) / 10 if has else None
    o["Success@10"] = (1.0 if any(rels[:10]) else 0.0) if has else None
    o["CCS@50"] = ccs_at_k(cl, w, 50)
    o["MCWAP"], o["CWAP_coverage"] = cwap(ranked, qrel, w, doc_cats)
    o["unjudged@10"] = sum(1 for x in ranked[:10] if x["doc"] not in qrel) / 10
    return o


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def bold_best(vals, higher):
    best = max(vals) if higher else min(vals)
    return [f"\\textbf{{{v:.3f}}}" if abs(v - best) < 5e-4 else f"{v:.3f}" for v in vals]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", help="table column names (default: run names)")
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--docs", help="JSONL with doc id and category (rarely needed, see above)")
    ap.add_argument("--fix-minutes", action="store_true", help="passages with '-MIN-' in the id = decision + proceedings")
    ap.add_argument("--depth", type=int, default=100,
                    help="score only the top-N of every run, so MAP and Recall use the same list length for all systems "
                         "(default 100; 0 = use each run as given)")
    ap.add_argument("--exclude", help="file of query ids to leave out (e.g. the Boost tuning set)")
    ap.add_argument("--extended", action="store_true", help="also write P@10, Success@10, Recall@100, nDCG@20, ...")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.names and len(a.names) != len(a.runs):
        sys.exit("--names needs one name per run")

    weights, qrels, docs = read_queries(a.queries), read_qrels(a.qrels), read_docs(a.docs)
    cat_map = read_qrel_categories(a.qrels)
    runs = [read_run(p, docs, cat_map, a.fix_minutes) for p in a.runs]
    if a.depth:
        runs = [(n, {q: r[:a.depth] for q, r in run.items()}) for n, run in runs]
    names = a.names or [n for n, _ in runs]
    excl = set(open(a.exclude).read().split()) if a.exclude else set()
    qids = [q for q in weights if q in qrels and q not in excl and all(q in r for _, r in runs)]
    lost = [q for q in weights if q not in qids and q not in excl]
    if lost:
        print(f"WARNING: {len(lost)} queries skipped (not in qrels or in some run): {lost[:8]}", file=sys.stderr)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cols = [k for _, _, k, _ in TABLE] + (EXTENDED if a.extended else [])
    diag = ["unjudged@10", "CWAP_coverage"]
    summ = []
    for name, (_, run) in zip(names, runs):
        doc_cats = {d: c for d, (_, c) in docs.items()}
        doc_cats.update(cat_map)
        for ranked in run.values():
            for x in ranked:
                doc_cats.setdefault(x["doc"], x["cats"])
        if a.fix_minutes:
            doc_cats.update({d: {"decision", "proceedings"} for d in doc_cats if "-MIN-" in d})
        per_q = {q: evaluate_query(run[q], qrels[q], weights[q], doc_cats) for q in qids}
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        with open(out / f"per_query_{safe}.tsv", "w", encoding="utf-8") as f:
            f.write("query_id\tn_rel\t" + "\t".join(cols + diag) + "\n")
            for q in qids:
                f.write(f"{q}\t{per_q[q]['n_rel']}\t" + "\t".join(
                    "" if per_q[q][c] is None else f"{per_q[q][c]:.6f}" for c in cols + diag) + "\n")
        s = {"system": name, "queries": len(qids),
             "with_relevant": sum(1 for q in qids if per_q[q]["n_rel"] > 0),
             "MCWAP_evaluable": sum(1 for q in qids if per_q[q]["MCWAP"] is not None)}
        for c in cols + diag:
            s[c] = mean(per_q[q][c] for q in qids)
        summ.append(s)

    keys = list(summ[0])
    with open(out / "summary.tsv", "w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for s in summ:
            f.write("\t".join(f"{s[k]:.6f}" if isinstance(s[k], float) else str(s[k]) for k in keys) + "\n")
    with open(out / "results_table.tex", "w", encoding="utf-8") as f:
        f.write("% Table 2 body. Columns: " + " & ".join(names) + "\n")
        prev = None
        for g, label, key, higher in TABLE:
            if prev is not None and g != prev:
                f.write("\\midrule\n")
            f.write(f"{g if g != prev else ''} & {label} & " + " & ".join(bold_best([s[key] for s in summ], higher)) + " \\\\\n")
            prev = g

    print(f"{len(qids)} queries, {len(summ)} systems  (queries with relevant passages: {summ[0]['with_relevant']})")
    print(f"{'':<13}" + "".join(f"{n[:9]:>10}" for n in names))
    for g, label, key, _ in TABLE + [(None, "", k, True) for k in (EXTENDED if a.extended else [])]:
        print(f"{key:<13}" + "".join(f"{s[key]:>10.3f}" for s in summ))
    print("\nunjudged@10  " + "".join(f"{s['unjudged@10']:>10.3f}" for s in summ))
    print("CWAP coverage" + "".join(f"{s['CWAP_coverage']:>9.3f}" for s in summ) +
          f"   (evaluable queries: {summ[0]['MCWAP_evaluable']})")
    print(f"\nwritten to {out}/  (summary.tsv, results_table.tex, per_query_*.tsv)")


if __name__ == "__main__":
    main()