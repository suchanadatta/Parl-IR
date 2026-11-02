#!/usr/bin/env python3
"""
Turn an 8-column run/TSV file into the pool.jsonl that the judging script reads.

  query_id  Q0  doc_id  rank  score  full_text  category  run_name   ->   {"query_id","doc_id","doc_text","category":[...]}

  python tsv_to_pool.py unjudged_bgem3_description_full_top250.tsv pool_unjudged.jsonl
"""
import json
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
seen, n = set(), 0
with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
    for ln, line in enumerate(f, 1):
        p = line.rstrip("\n").split("\t")
        if len(p) != 8:
            sys.exit(f"{src}:{ln}: expected 8 tab-separated columns, got {len(p)}")
        key = (p[0], p[2])
        if key in seen:
            continue
        seen.add(key)
        out.write(json.dumps({"query_id": p[0], "doc_id": p[2], "doc_text": p[5],
                              "category": [c for c in re.split(r"[|,]", p[6]) if c]},
                             ensure_ascii=False) + "\n")
        n += 1
print(f"{n} (query, passage) pairs written to {dst}")