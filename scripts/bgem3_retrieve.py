#!/usr/bin/env python3
"""
Dense retrieval over the whole ParlIR corpus with BGE-M3 (the BGE-M3 run of Table 2).

Every passage is embedded once and cached, then each topic is scored against the entire collection by
exhaustive cosine similarity. This is a first-stage retriever over all 297,279 passages, not a rerank of
a lexical run, so it can reach the non-English speech record that BM25 cannot match at all.

INPUTS
  --corpus   corpus.jsonl. Reads  passage_id (or id / doc_id), text, categories (or category)
  --topics   topics.json (topic_id, title, description, narrative) or the TREC topics.xml that
             make_topic_trec_format.py writes
  --query-fields  which topic fields form the query, comma-separated (default: description).
             The released run uses the description alone; pass 'title' to reproduce the title run.

OUTPUT  8-column run, the same layout as the BM25 run files:
  qid  Q0  doc  rank  score  text  category  run_name
Tabs and newlines are stripped from the text and category fields, so every line has exactly 8 columns.
Categories are comma-joined, as in the BM25 runs; parlir_eval.py accepts ',' or '|'.

The embeddings are cached as shards of --shard passages under --cache, and an interrupted run resumes from
them, which matters because encoding the full corpus takes hours on one GPU. Search itself takes seconds:
the matrix is 297k x 1024 floats (1.2 GB), so an exhaustive scan is cheaper than building an index.

  python bgem3_retrieve.py --corpus corpus.jsonl --topics topics.json --out bge.run
  python bgem3_retrieve.py --corpus corpus.jsonl --topics topics.xml --out bge_title.run \
         --query-fields title --name bgem3_title --depth 500
"""
import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

CATS = ["decision", "position", "proceedings", "actors", "institution"]
_WS = re.compile(r"[\t\r\n]+")


def clean(s):
    return _WS.sub(" ", s or "").strip()


# ================================================================== I/O
def read_corpus(path):
    """(ids, texts, categories) in file order. Lines that are not valid JSON are skipped with a warning."""
    ids, texts, cats, bad = [], [], [], 0
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"skip {path}:{n}: {e}", file=sys.stderr)
                bad += 1
                continue
            c = r.get("categories", r.get("category", []))
            if isinstance(c, str):
                c = [x.strip() for x in re.split(r"[|,;]", c)]
            ids.append(str(r.get("passage_id", r.get("id", r.get("doc_id")))))
            texts.append(r.get("text") or "")
            cats.append(",".join(sorted(x for x in c if x in CATS)))
    if bad:
        print(f"WARNING: {bad} unparseable lines skipped", file=sys.stderr)
    return ids, texts, cats


def read_topics(path):
    """{qid: {title, description, narrative}} from topics.json or the TREC topics.xml."""
    t = {}
    if Path(path).suffix.lower() == ".json":
        for r in json.load(open(path, encoding="utf-8")):
            t[str(r.get("topic_id", r.get("id")))] = {f: (r.get(f) or "").strip()
                                                      for f in ("title", "description", "narrative")}
    else:
        for top in ET.parse(path).getroot().iter("top"):
            t[(top.findtext("num") or "").strip()] = {
                "title": (top.findtext("title") or "").strip(),
                "description": (top.findtext("desc") or "").strip(),
                "narrative": (top.findtext("narr") or "").strip()}
    return t


def query_text(topic, fields):
    return " ".join(topic[f] for f in fields.split(",") if topic.get(f))


def qid_sort(q):
    return (0, int(q)) if q.isdigit() else (1, q)


# ================================================================== embedding
def load_model(name, max_len, device):
    from sentence_transformers import SentenceTransformer
    if device == "auto":
        import torch
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"loading {name} on {device}", flush=True)
    m = SentenceTransformer(name, device=device)
    m.max_seq_length = max_len
    return m


def embed_corpus(model, texts, cache, shard, batch, dim):
    """Encode every passage, caching shards of `shard` passages so an interrupted run resumes.
    Returns an L2-normalised (len(texts), dim) float32 matrix."""
    os.makedirs(cache, exist_ok=True)
    E = np.zeros((len(texts), dim), dtype=np.float32)
    nsh = (len(texts) + shard - 1) // shard
    done, t0 = 0, time.time()
    for s in range(nsh):
        lo, hi = s * shard, min((s + 1) * shard, len(texts))
        f = os.path.join(cache, f"shard_{s:05d}.npy")
        if os.path.exists(f):
            E[lo:hi] = np.load(f)
            continue
        v = model.encode(texts[lo:hi], batch_size=batch, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        np.save(f, v)
        E[lo:hi] = v
        done += 1
        el = time.time() - t0
        print(f"shard {s + 1}/{nsh}  {el / 60:.1f}m elapsed  "
              f"eta {el / done * (nsh - s - 1) / 60:.0f}m", flush=True)
    print(f"{len(texts):,} passages embedded ({nsh} shards in {cache})", flush=True)
    return E


# ================================================================== search
def search(E, Q, depth):
    """Exhaustive cosine search. Yields (query index, ranked passage indices, their scores).
    errstate: numpy 2.x on Apple Accelerate raises spurious FP warnings from float32 matmul (they fire
    on random arrays too, and the products are correct); it is suppressed here so the log stays readable."""
    for qi in range(len(Q)):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            s = E @ Q[qi]
        k = min(depth, len(s) - 1)
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        yield qi, top, s[top]


# ================================================================== CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--topics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--query-fields", default="description",
                    help="topic fields forming the query: title, description, narrative")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--depth", type=int, default=500, help="passages per query written")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024, help="passage tokens encoded")
    ap.add_argument("--dim", type=int, default=1024, help="embedding width of --model")
    ap.add_argument("--shard", type=int, default=2000, help="passages per cached shard")
    ap.add_argument("--cache", default=".bgem3_cache",
                    help="embedding cache; delete it when the corpus or --model changes")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--name", help="run name in column 8 (default: bgem3_<first query field>)")
    a = ap.parse_args()

    name = a.name or f"bgem3_{a.query_fields.split(',')[0]}"
    ids, texts, cats = read_corpus(a.corpus)
    topics = read_topics(a.topics)
    qids = sorted(topics, key=qid_sort)
    print(f"corpus: {len(ids):,} passages   topics: {len(qids)}   query: {a.query_fields}", flush=True)

    model = load_model(a.model, a.max_len, a.device)
    E = embed_corpus(model, texts, a.cache, a.shard, a.batch, a.dim)
    Q = model.encode([query_text(topics[q], a.query_fields) for q in qids], batch_size=16,
                     normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    t0 = time.time()
    with open(a.out, "w", encoding="utf-8") as f:
        for qi, top, scores in search(E, Q, a.depth):
            for rank, (i, sc) in enumerate(zip(top, scores), 1):
                f.write(f"{qids[qi]}\tQ0\t{ids[i]}\t{rank}\t{sc:.6f}\t"
                        f"{clean(texts[i])}\t{clean(cats[i])}\t{name}\n")
    print(f"{name}: {len(qids)} queries x top {a.depth} -> {a.out} "
          f"(search took {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
