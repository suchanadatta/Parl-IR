import sys
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

# Usage: python3 make_topics.py <queries.tsv> <topics.xml>
# Input columns (tab-separated): id, title, description, narrative
src, out_xml = sys.argv[1], sys.argv[2]

res = '<topics>\n'
n_topics = 0
with open(src, encoding="utf-8") as f:
    for n, line in enumerate(f, 1):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split("\t", 3)
        if len(parts) != 4:
            sys.exit(f"line {n}: expected 4 tab-separated columns, got {len(parts)}")
        q_id, title, desc, narr = (escape(p.strip()) for p in parts)
        res += ('<top>\n<num>' + q_id + '</num>\n'
                + '<title>' + title + '</title>\n'
                + '<desc>' + desc + '</desc>\n'
                + '<narr>' + narr + '</narr>\n</top>\n')
        n_topics += 1
res += '</topics>'

with open(out_xml, "w", encoding="utf-8") as trec_file:
    trec_file.write(res)

# Check the output is well-formed XML with the expected number of topics
root = ET.parse(out_xml).getroot()
assert len(root) == n_topics, "topic count mismatch"
print(f"{n_topics} topics written to {out_xml}; XML parses cleanly")