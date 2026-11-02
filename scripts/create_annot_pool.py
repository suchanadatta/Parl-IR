#!/usr/bin/env python3
"""
Build a fixed-size judging pool per query from two or more run files.

Run files are tab-separated with 8 columns, as written by ParliViewSearcher:
  query_id  Q0  doc_id  rank  score  full_text  category  run_name

Method: round-robin interleaving. For each query the runs take turns contributing
their next-best unseen document (rank 1 of each run, then rank 2, ...) until the
pool holds --pool-size distinct documents. The run that goes first alternates
between queries, so neither run is favoured.

Guarantee: if the pool size is at least twice a depth d, every document in the
top-d of every run is in the pool. With two runs and a pool of 100, the top-50 of
both runs is always fully covered (checked by this script).

Usage:
  python3 make_pool_from_runs.py RUN1.tsv RUN2.tsv --pool-size 100 --out-dir pool/
"""
import argparse
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path


def read_run(path, max_rank):
    """Return (run_name, {qid: [(rank, doc_id, score, category, text), ...]}) for rank <= max_rank."""
    rows = defaultdict(list)
    run_name = None
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 8:
                sys.exit(f"{path}:{n}: expected 8 tab-separated columns, got {len(parts)}")
            qid, _, doc_id, rank, score, text, category, name = parts
            run_name = run_name or name
            rank = int(rank)
            if rank <= max_rank:
                rows[qid].append((rank, doc_id, float(score), category, text))
    for qid in rows:
        rows[qid].sort()
    return run_name, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run files (8-column TSV)")
    ap.add_argument("--pool-size", type=int, default=100)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    # Round-robin never needs to go deeper than pool-size in any run.
    runs = OrderedDict()
    for p in args.runs:
        name, rows = read_run(p, args.pool_size)
        if name in runs:
            sys.exit(f"two run files share the run name '{name}'")
        runs[name] = rows
    names = list(runs)

    qids = list(runs[names[0]])
    for name in names[1:]:
        if set(runs[name]) != set(qids):
            sys.exit("the run files do not contain the same set of queries")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    total = 0
    with open(out_dir / "pool.jsonl", "w", encoding="utf-8") as fj, \
         open(out_dir / "pool_ids.tsv", "w", encoding="utf-8") as ft:
        ft.write("query_id\tdoc_id\tcategory\t" + "\t".join(f"rank_{n}" for n in names) + "\n")

        for qi, qid in enumerate(qids):
            # per-run lookup and ordering; alternate which run goes first
            order = names[qi % len(names):] + names[:qi % len(names)]
            by_doc = {n: {r[1]: r for r in runs[n][qid]} for n in names}
            pool = OrderedDict()   # doc_id -> {category, text}
            depth_reached = {n: 0 for n in names}

            depth = 0
            while len(pool) < args.pool_size:
                depth += 1
                progressed = False
                for n in order:
                    if len(pool) >= args.pool_size:
                        break
                    rows = runs[n][qid]
                    if depth > len(rows):
                        continue
                    progressed = True
                    rank, doc_id, score, cat, text = rows[depth - 1]
                    depth_reached[n] = depth
                    if doc_id not in pool:
                        pool[doc_id] = {"category": cat, "text": text}
                if not progressed:
                    break

            # the same document must have the same text/category in every run
            for doc_id, info in pool.items():
                for n in names:
                    r = by_doc[n].get(doc_id)
                    if r and (r[4] != info["text"] or r[3] != info["category"]):
                        sys.exit(f"query {qid}, doc {doc_id}: text or category differs between runs")

            # guarantee check: top (pool_size // len(runs)) of every run is inside the pool
            d = args.pool_size // len(names)
            for n in names:
                missing = [r[1] for r in runs[n][qid][:d] if r[1] not in pool]
                if missing:
                    sys.exit(f"query {qid}: top-{d} of {n} not fully covered ({len(missing)} missing)")

            for doc_id, info in pool.items():
                ranks = {n: (by_doc[n][doc_id][0] if doc_id in by_doc[n] else None) for n in names}
                cats = [c for c in info["category"].split("|") if c]
                fj.write(json.dumps({
                    "query_id": qid,
                    "doc_id": doc_id,
                    "category": cats,
                    "doc_text": info["text"],
                    "pooled_from": {n: ranks[n] for n in names},
                }, ensure_ascii=False) + "\n")
                ft.write(f"{qid}\t{doc_id}\t{info['category']}\t" +
                         "\t".join("" if ranks[n] is None else str(ranks[n]) for n in names) + "\n")

            total += len(pool)
            summary.append((qid, len(pool), depth_reached))

    with open(out_dir / "pool_summary.tsv", "w", encoding="utf-8") as f:
        f.write("query_id\tpool_size\t" + "\t".join(f"depth_{n}" for n in names) + "\n")
        for qid, size, dr in summary:
            f.write(f"{qid}\t{size}\t" + "\t".join(str(dr[n]) for n in names) + "\n")

    print(f"{len(qids)} queries, {total} (query, document) pairs -> {out_dir / 'pool.jsonl'}")
    sizes = {s for _, s, _ in summary}
    print("pool sizes:", sorted(sizes))


if __name__ == "__main__":
    main()