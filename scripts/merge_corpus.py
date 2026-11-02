"""
Lucene field	                       Type	               From file 1	                      From file 2
id	            StringField, stored	   chunk_id	                          entry_id
category	    StringField, stored	   category (committee_memberships)	  set it yourself, e.g. factsheet (or derive it from doc_type)
text	        TextField, stored	   text	                              text
title	        TextField, stored	   none	                              title
doc_id	        StringField (multi-valued)	doc_ids[]	doc_id
mep_id, mep_name	StringField / TextField	yes	none
dossier_ref	StringField (multi-valued)	dossier_refs[]	none
active	StringField ("true"/"false")	yes	none
as_of_date, coverage_start, coverage_end	StringField (ISO dates sort and range-compare correctly as strings)	yes	none

"""

import json, sys

MEP_FILE = sys.argv[1]        # file with chunk_id / mep_id ...
FACT_FILE = sys.argv[2]       # file with entry_id / doc_type ...
OUT_FILE = sys.argv[3]        # merged.jsonl


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"skip {path}:{n}: {e}", file=sys.stderr)


n_mep = n_fact = 0

with open(OUT_FILE, "w", encoding="utf-8") as out:
    for r in read_jsonl(MEP_FILE):
        out.write(json.dumps({
            "id": r["chunk_id"],
            "category": r.get("category", "mep_record"),
            "doc_type": "MEP record",
            "title": None,
            "text": r["text"],
            "mep_id": r.get("mep_id"),
            "mep_name": r.get("mep_name"),
            "active": r.get("active"),
            "as_of_date": r.get("as_of_date"),
            "coverage_start": r.get("coverage_start"),
            "coverage_end": r.get("coverage_end"),
            "dossier_refs": r.get("dossier_refs", []),
            "doc_ids": r.get("doc_ids", []),
        }, ensure_ascii=False) + "\n")
        n_mep += 1

    for r in read_jsonl(FACT_FILE):
        out.write(json.dumps({
            "id": r["entry_id"],
            "category": "factsheet",
            "doc_type": r.get("doc_type", "Fact sheet"),
            "title": r.get("title"),
            "text": r["text"],
            "mep_id": None,
            "mep_name": None,
            "active": None,
            "as_of_date": None,
            "coverage_start": None,
            "coverage_end": None,
            "dossier_refs": [],
            "doc_ids": [r["doc_id"]] if r.get("doc_id") else [],
        }, ensure_ascii=False) + "\n")
        n_fact += 1

print(f"wrote {n_mep} MEP records + {n_fact} factsheets -> {OUT_FILE}")