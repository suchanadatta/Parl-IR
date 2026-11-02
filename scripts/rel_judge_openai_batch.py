#!/usr/bin/env python3
"""
Simple LLM relevance judging with the OpenAI Batch API.

The model sees ONLY: query title, description, narrative, and the document text.
It answers 0 (not relevant) or 1 (relevant).

Steps:
  python judge.py submit  --pool pool.jsonl --topics topics.xml --work-dir work
  python judge.py status  --work-dir work
  python judge.py collect --work-dir work --out-dir qrels

  pip install openai ;  export OPENAI_API_KEY=...

pool.jsonl : one JSON per line with  query_id, doc_id, doc_text
topics.xml : <topics><top><num/><title/><desc/><narr/></top>...</topics>
"""
import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

SYSTEM = (
    "You are a relevance assessor for a search system over European Parliament records. "
    "You get a search topic (title, description, narrative) and one document. "
    "The narrative defines what counts as relevant. "
    'Answer 1 if the document provides what the topic asks for (fully or partially), '
    "otherwise answer 0. Judge only from the document text. "
    'Reply with JSON only: {"judgment": 0} or {"judgment": 1}.'
)

USER = """Topic title: {title}

Topic description: {desc}

Topic narrative: {narr}

Document:
<<<
{text}
>>>

Is this document relevant to the topic? Answer with judgment 0 or 1."""

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "relevance",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"judgment": {"type": "integer", "enum": [0, 1]}},
            "required": ["judgment"],
            "additionalProperties": False,
        },
    },
}


def read_topics(path):
    topics = {}
    for top in ET.parse(path).getroot().iter("top"):
        qid = (top.findtext("num") or "").strip()
        topics[qid] = {
            "title": (top.findtext("title") or "").strip(),
            "desc": (top.findtext("desc") or "").strip(),
            "narr": (top.findtext("narr") or "").strip(),
        }
    return topics


def client():
    from openai import OpenAI
    return OpenAI()


# ------------------------------------------------------------------ submit
def cmd_submit(a):
    work = Path(a.work_dir)
    if (work / "batches.json").exists():
        sys.exit(f"{work}/batches.json exists - use a new --work-dir (or delete the old one)")
    work.mkdir(parents=True, exist_ok=True)

    topics = read_topics(a.topics)
    pool = [json.loads(l) for l in open(a.pool, encoding="utf-8") if l.strip()]

    missing = sorted({p["query_id"] for p in pool if p["query_id"] not in topics})
    if missing:
        sys.exit(f"no topic in {a.topics} for query ids: {missing[:10]}"
                 f"\ntopic ids in the file: {list(topics)[:10]}")

    # write mapping + batch input files
    files, n_in_file, fout = [], 0, None
    with open(work / "mapping.jsonl", "w", encoding="utf-8") as fmap:
        for i, p in enumerate(pool):
            if fout is None or n_in_file >= a.per_batch:
                if fout:
                    fout.close()
                path = work / f"batch_input_{len(files):03d}.jsonl"
                files.append(path)
                fout = open(path, "w", encoding="utf-8")
                n_in_file = 0
            t = topics[p["query_id"]]
            text = p["doc_text"]
            if a.max_doc_chars and len(text) > a.max_doc_chars:
                text = text[: a.max_doc_chars]
            body = {
                "model": a.model,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER.format(
                        title=t["title"], desc=t["desc"], narr=t["narr"], text=text)},
                ],
                "response_format": SCHEMA,
                "max_completion_tokens": a.max_completion_tokens,
            }
            if a.reasoning_effort != "none":
                body["reasoning_effort"] = a.reasoning_effort
            fout.write(json.dumps({"custom_id": str(i), "method": "POST",
                                   "url": "/v1/chat/completions", "body": body}) + "\n")
            fmap.write(json.dumps({"custom_id": str(i), "query_id": p["query_id"],
                                   "doc_id": p["doc_id"]}) + "\n")
            n_in_file += 1
    if fout:
        fout.close()
    print(f"{len(pool)} requests in {len(files)} batch file(s)")

    if a.dry_run:
        print("dry run: nothing uploaded")
        return

    c = client()
    batches = []
    for path in files:
        up = c.files.create(file=open(path, "rb"), purpose="batch")
        b = c.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                             completion_window="24h")
        batches.append({"file": path.name, "batch_id": b.id})
        print(f"submitted {path.name} -> {b.id}")
        (work / "batches.json").write_text(json.dumps(batches, indent=2))
        if a.wait:  # wait until finished before sending the next one (avoids queue limits)
            while True:
                s = c.batches.retrieve(b.id).status
                print(f"  {b.id}: {s}")
                if s in ("completed", "failed", "expired", "cancelled"):
                    break
                time.sleep(60)
    print("done. Check with:  python judge.py status --work-dir", work)


# ------------------------------------------------------------------ status
def cmd_status(a):
    c = client()
    for b in json.loads((Path(a.work_dir) / "batches.json").read_text()):
        r = c.batches.retrieve(b["batch_id"])
        rc = r.request_counts
        print(b["file"], b["batch_id"], r.status, f"{rc.completed}/{rc.total} done, {rc.failed} failed")


# ------------------------------------------------------------------ collect
def cmd_collect(a):
    work, out = Path(a.work_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    c = client()

    results = {}   # custom_id -> 0/1
    for b in json.loads((work / "batches.json").read_text()):
        r = c.batches.retrieve(b["batch_id"])
        if r.status != "completed":
            print(f"{b['file']}: status {r.status} - not finished yet, skipped")
            continue
        if r.output_file_id:
            for line in c.files.content(r.output_file_id).text.splitlines():
                if not line.strip():
                    continue
                o = json.loads(line)
                try:
                    resp = o["response"]
                    if resp["status_code"] != 200:
                        continue
                    j = json.loads(resp["body"]["choices"][0]["message"]["content"])["judgment"]
                    if j in (0, 1) and not isinstance(j, bool):
                        results[o["custom_id"]] = j
                except Exception:
                    pass  # counted as missing below

    mapping = [json.loads(l) for l in open(work / "mapping.jsonl", encoding="utf-8")]
    pool = {}
    if a.pool:
        for l in open(a.pool, encoding="utf-8"):
            if l.strip():
                p = json.loads(l)
                pool[(p["query_id"], p["doc_id"])] = p["doc_text"]
    topics = read_topics(a.topics) if a.topics else {}

    n_ok = n_missing = 0
    with open(out / "qrels.txt", "w", encoding="utf-8") as fq, \
         open(out / "qrels_with_text.tsv", "w", encoding="utf-8") as ft, \
         open(out / "missing.tsv", "w", encoding="utf-8") as fm:
        ft.write("query_id\tdoc_id\tjudgment\thuman_judgment\tquery_title\tdoc_text\n")
        fm.write("query_id\tdoc_id\n")
        for m in mapping:
            j = results.get(m["custom_id"])
            if j is None:
                fm.write(f"{m['query_id']}\t{m['doc_id']}\n")
                n_missing += 1
                continue
            fq.write(f"{m['query_id']} 0 {m['doc_id']} {j}\n")
            title = topics.get(m["query_id"], {}).get("title", "")
            text = pool.get((m["query_id"], m["doc_id"]), "").replace("\t", " ").replace("\n", " ")
            ft.write(f"{m['query_id']}\t{m['doc_id']}\t{j}\t\t{title}\t{text}\n")
            n_ok += 1
    print(f"judged: {n_ok}   missing/invalid: {n_missing}")
    print(f"wrote {out}/qrels.txt and {out}/qrels_with_text.tsv")
    if n_missing:
        print(f"see {out}/missing.tsv (re-run submit on just those pairs if needed)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("submit")
    s.add_argument("--pool", required=True)
    s.add_argument("--topics", required=True)
    s.add_argument("--work-dir", required=True)
    s.add_argument("--model", default="gpt-6-luna")
    s.add_argument("--reasoning-effort", default="low",
                   help="low/medium/high for reasoning models (gpt-5*, o*); 'none' for gpt-4.1 etc.")
    s.add_argument("--max-completion-tokens", type=int, default=2048)
    s.add_argument("--max-doc-chars", type=int, default=20000, help="0 = never cut the document")
    s.add_argument("--per-batch", type=int, default=5000, help="requests per batch file")
    s.add_argument("--wait", action="store_true", help="wait for each batch to finish before sending the next")
    s.add_argument("--dry-run", action="store_true", help="only write the batch files, do not upload")
    s.set_defaults(f=cmd_submit)

    s = sp.add_parser("status")
    s.add_argument("--work-dir", required=True)
    s.set_defaults(f=cmd_status)

    s = sp.add_parser("collect")
    s.add_argument("--work-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--pool", help="pool.jsonl, to put the document text in qrels_with_text.tsv")
    s.add_argument("--topics", help="topics.xml, to put the query title in qrels_with_text.tsv")
    s.set_defaults(f=cmd_collect)

    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()