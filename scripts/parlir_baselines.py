#!/usr/bin/env python3
"""
ParliR baselines in one file (BM25 comes from lucene_index_search/, BGE-M3 from bgem3_retrieve.py).

  hybrid   Reciprocal rank fusion of runs:  RRF(d) = sum_r 1/(60 + rank_r(d))
  ce       Hybrid+CE: cross-encoder rerank of the hybrid top-100 (default BAAI/bge-reranker-v2-m3)
  filter   keep only passages with a positively weighted category, in hybrid order
  boost    s'(d) = minmax(s(d)) + alpha * sum_{c in L(d)} w_{q,c}
  quota    fill the list so the category mix follows w_q at every rank
  tune     pick alpha for boost on a random development set of queries
  category run tune + filter + boost + quota in one command

Runs: 6-column TREC (qid Q0 doc rank score name) or 8-column (qid Q0 doc rank score text category name).
Category-aware methods need each passage's categories, and ce needs the text: use an 8-column base run, or
give --docs corpus.jsonl (id, text, category).  Output runs are always 8-column.
Pass --qrels qrels_category.tsv (qid doc rel category) so judged passages use the qrels' category labels, the same
labels parlir_eval.py scores with; --fix-minutes sets passages with '-MIN-' in the id to decision+proceedings.
Needs parlir_eval.py in the same folder.

  python parlir_baselines.py hybrid   --runs bm25.run bge.run --out hybrid.run --depth 500 --k 250
  python parlir_baselines.py ce       --base hybrid.run --topics topics.xml --out hybrid_ce.run --depth 100
  python parlir_baselines.py category --base hybrid.run --queries queries.tsv --qrels qrels.txt --out-dir runs/ --dev-n 20
  python parlir_baselines.py quota    --base hybrid.run --queries queries.tsv --out quota.run     (single methods too)
"""
import argparse
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from parlir_eval import (CATS, read_queries, read_qrels, read_qrel_categories, read_docs, read_run,
                         ndcg_at_k, ccs_at_k)


# ================================================================== I/O
def clean(s):
    return (s or "").replace("\t", " ").replace("\n", " ").replace("\r", " ")


def write_run(path, name, results, rescore=False):
    """results: {qid: [doc dicts]}. rescore=True writes scores that fall with rank (needed after reordering
    by something other than score, so trec_eval, which sorts by score, sees the same order)."""
    with open(path, "w", encoding="utf-8") as f:
        for q, docs in results.items():
            n = len(docs)
            for i, d in enumerate(docs, 1):
                score = (n - i + 1) if rescore else d["score"]
                f.write(f"{q}\tQ0\t{d['doc']}\t{i}\t{score:.8g}\t{clean(d['text'])}\t"
                        f"{'|'.join(sorted(d['cats']))}\t{name}\n")


def read_topics(path):
    t = {}
    for top in ET.parse(path).getroot().iter("top"):
        t[(top.findtext("num") or "").strip()] = {
            "title": (top.findtext("title") or "").strip(),
            "description": (top.findtext("desc") or "").strip(),
            "narrative": (top.findtext("narr") or "").strip()}
    return t


def query_text(topic, fields):
    return " ".join(topic[f] for f in fields.split(",") if topic[f])


# ================================================================== standard: hybrid and cross-encoder
def rrf(run_list, rrf_k=60, depth=500, out_k=250):
    out = {}
    for q in sorted(set().union(*[set(r) for r in run_list])):
        score, info = defaultdict(float), {}
        for r in run_list:
            for rank, d in enumerate(r.get(q, [])[:depth], 1):
                score[d["doc"]] += 1.0 / (rrf_k + rank)
                info.setdefault(d["doc"], d)
        order = sorted(score, key=lambda x: (-score[x], x))          # ties: by id, deterministic
        out[q] = [dict(info[x], score=score[x]) for x in order[:out_k]]
    return out


def cross_encode(base, topics, model, depth, batch, max_len, fields):
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(model, max_length=max_len)
    out = {}
    for q, docs in base.items():
        if q not in topics:
            continue
        docs = docs[:depth]
        if any(not d["text"] for d in docs):
            sys.exit("ce needs passage text: use an 8-column base run or pass --docs corpus.jsonl")
        sc = ce.predict([(query_text(topics[q], fields), d["text"]) for d in docs], batch_size=batch)
        order = sorted(range(len(docs)), key=lambda i: -float(sc[i]))
        out[q] = [dict(docs[i], score=float(sc[i])) for i in order]
    return out


# ================================================================== category-aware
def filter_(docs, w, k):
    return [d for d in docs if any(w[c] > 0 for c in d["cats"])][:k]


def boost(docs, w, k, alpha):
    if not docs:
        return []
    lo, hi = min(d["score"] for d in docs), max(d["score"] for d in docs)
    span = (hi - lo) or 1.0
    new = [dict(d, score=(d["score"] - lo) / span + alpha * sum(w[c] for c in d["cats"])) for d in docs]
    return sorted(new, key=lambda d: -d["score"])[:k]


def quota(docs, w, k):
    """Stride scheduling: at each rank take the prioritised category furthest below its target share
    (w_c * (n+1) - count_c) and insert its best-ranked unused passage. A multi-label passage counts for
    all its categories. When no prioritised category has passages left, keep the base order."""
    pos = [c for c in CATS if w[c] > 0]
    remaining, out, counts = list(docs), [], {c: 0 for c in CATS}
    while remaining and len(out) < k:
        avail = [c for c in pos if any(c in d["cats"] for d in remaining)]
        if not avail:
            break
        c = max(avail, key=lambda c: w[c] * (len(out) + 1) - counts[c])
        pick = next(d for d in remaining if c in d["cats"])
        remaining.remove(pick)
        out.append(pick)
        for cc in pick["cats"]:
            counts[cc] += 1
    out.extend(remaining[: max(0, k - len(out))])
    return out[:k]


def apply(method, base, weights, k, alpha=None):
    res = {}
    for q, docs in base.items():
        if q not in weights:
            continue
        w = weights[q]
        res[q] = (filter_(docs, w, k) if method == "filter" else
                  boost(docs, w, k, alpha) if method == "boost" else quota(docs, w, k))
    return res


def tune_alpha(base, weights, qrels, alphas, dev_n, seed, objective):
    usable = sorted(q for q in weights if q in base and q in qrels and any(qrels[q].values()))
    dev = sorted(random.Random(seed).sample(usable, min(dev_n, len(usable))))
    rows, best = [], None
    for a in alphas:
        nd, cc = [], []
        for q in dev:
            r = boost(base[q], weights[q], 100, a)
            rels = [qrels[q].get(d["doc"], 0) for d in r]
            nd.append(ndcg_at_k(rels, sum(1 for y in qrels[q].values() if y), 10))
            cc.append(ccs_at_k([d["cats"] for d in r], weights[q], 10))
        n, c = sum(nd) / len(nd), sum(cc) / len(cc)
        obj = {"mean": (n + c) / 2, "ndcg": n, "ccs": c}[objective]
        rows.append((a, n, c, obj))
        if best is None or obj > best[1]:
            best = (a, obj)
    return dev, rows, best[0]


# ================================================================== CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("hybrid")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--rrf-k", type=int, default=60)
    s.add_argument("--depth", type=int, default=500, help="passages of each input run that are fused")
    s.add_argument("--k", type=int, default=250, help="passages per query written")
    s.add_argument("--docs")
    s.add_argument("--qrels", help="qrels with categories (category labels for judged passages)")
    s.add_argument("--fix-minutes", action="store_true")
    s.add_argument("--name", default="hybrid")

    s = sp.add_parser("ce")
    s.add_argument("--base", required=True)
    s.add_argument("--topics", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    s.add_argument("--depth", type=int, default=100)
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--max-len", type=int, default=512)
    s.add_argument("--query-fields", default="title", help="title,description,narrative (same for all systems)")
    s.add_argument("--docs")
    s.add_argument("--qrels", help="qrels with categories (category labels for judged passages)")
    s.add_argument("--fix-minutes", action="store_true")
    s.add_argument("--name", default="hybrid_ce")

    for m in ("filter", "boost", "quota"):
        s = sp.add_parser(m)
        s.add_argument("--base", required=True)
        s.add_argument("--queries", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--k", type=int, default=100)
        s.add_argument("--docs")
        s.add_argument("--qrels", help="qrels with categories (category labels for judged passages)")
        s.add_argument("--fix-minutes", action="store_true")
        s.add_argument("--name", default=f"cat_{m}")
        if m == "boost":
            s.add_argument("--alpha", type=float, required=True)

    for m in ("tune", "category"):
        s = sp.add_parser(m)
        s.add_argument("--base", required=True)
        s.add_argument("--queries", required=True)
        s.add_argument("--qrels", required=True)
        s.add_argument("--out-dir", required=True)
        s.add_argument("--dev-n", type=int, default=20)
        s.add_argument("--seed", type=int, default=1)
        s.add_argument("--alphas", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2])
        s.add_argument("--objective", choices=["mean", "ndcg", "ccs"], default="mean",
                       help="mean = (nDCG@10 + CCS@10) / 2 on the dev queries")
        s.add_argument("--k", type=int, default=100)
        s.add_argument("--docs")
        s.add_argument("--fix-minutes", action="store_true")

    a = ap.parse_args()
    docs = read_docs(getattr(a, "docs", None))
    cmap = read_qrel_categories(a.qrels) if getattr(a, "qrels", None) else {}
    fixm = getattr(a, "fix_minutes", False)
    load = lambda p: read_run(p, docs, cmap, fixm)[1]

    if a.cmd == "hybrid":
        res = rrf([load(p) for p in a.runs], a.rrf_k, a.depth, a.k)
        write_run(a.out, a.name, res)
        print(f"{a.name}: {len(res)} queries -> {a.out}")
    elif a.cmd == "ce":
        res = cross_encode(load(a.base), read_topics(a.topics), a.model, a.depth, a.batch,
                           a.max_len, a.query_fields)
        write_run(a.out, a.name, res)
        print(f"{a.name}: {len(res)} queries reranked (top {a.depth}, {a.model}) -> {a.out}")
    elif a.cmd in ("filter", "boost", "quota"):
        res = apply(a.cmd, load(a.base), read_queries(a.queries), a.k, getattr(a, "alpha", None))
        write_run(a.out, a.name, res, rescore=(a.cmd == "quota"))
        short = sum(1 for d in res.values() if len(d) < a.k)
        print(f"{a.name}: {len(res)} queries -> {a.out}; {short} have fewer than {a.k} passages")
    else:                                              # tune / category
        base, weights, qrels = load(a.base), read_queries(a.queries), read_qrels(a.qrels)
        dev, rows, alpha = tune_alpha(base, weights, qrels, a.alphas, a.dev_n, a.seed, a.objective)
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "dev_ids.txt").write_text("\n".join(dev) + "\n")
        (out / "best_alpha.txt").write_text(f"{alpha}\n")
        print(f"dev queries: {len(dev)}   alpha   nDCG@10  CCS@10  objective")
        for al, n, c, o in rows:
            print(f"{'':>17}{al:<7} {n:.3f}    {c:.3f}   {o:.3f}{'   <- best' if al == alpha else ''}")
        if a.cmd == "category":
            for m in ("filter", "boost", "quota"):
                res = apply(m, base, weights, a.k, alpha)
                write_run(out / f"{m}.run", f"cat_{m}", res, rescore=(m == "quota"))
            print(f"wrote filter.run, boost.run (alpha={alpha}), quota.run, dev_ids.txt in {out}/")
            print(f"evaluate with:  --exclude {out}/dev_ids.txt   (the tuning queries are left out)")


if __name__ == "__main__":
    main()