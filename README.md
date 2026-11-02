# ParlIR

**ParlIR** is a test collection for evaluating category-aware retrieval over the record of the European Parliament. It is designed for retrieval and retrieval-augmented generation (RAG) settings where documents that are equally "on topic" differ in what they can be used to establish. For example, a speech expresses a position of an individual or party, while an adopted text records a decision of Parliament.

## Overview

| Component | Size | Description |
|---|---|---|
| Corpus | 297,279 passages from 62,924 documents | 21 document types, 2 July 2019 to 16 July 2026 |
| Evidence categories | 5 | Each document type is mapped to one or more categories |
| Topics | 138 | Derived from real user questions, each with a target distribution over the categories |
| Relevance judgements | 59,232 topic–passage pairs | Binary labels generated with an LLM, with a sample checked by an expert annotator |

## Files and formats

| File | Format | Contents |
|---|---|---|
| `corpus.jsonl` | JSON Lines, one passage per line | 297,279 passages |
| `topics.json` | JSON array | 138 topics with their target distributions |
| `qrels.txt` | TREC qrels, space-separated | 59,232 topic–passage judgements |
| `doc_type_categories.csv` | CSV | mapping from document type to evidence categories |

### Corpus file

| Field | Description |
|---|---|
| `passage_id` | Unique passage identifier |
| `doc_id` | Identifier of the source document |
| `doc_type` | One of the 21 document types listed under [Corpus](#corpus) |
| `categories` | Evidence categories of the passage |
| `date` | Sitting or publication date, where available |
| `url` | URL of the source document |
| `text` | Passage text |

`date` is absent for procedure files, which record a span of activity rather
than a single date, and for the Member profiles, which are renderings of a
register rather than dated documents; 87% of passages carry one. Every passage
carries a `url`. For procedure files this is the Legislative Observatory entry
for the dossier, and for Member profiles the Member's page on Parliament's site.

### Topics file

| Field | Description |
|---|---|
| `topic_id` | Unique topic identifier |
| `title` | Short form of the question |
| `description` | Fuller statement of the information need |
| `narrative` | Which document types can and cannot answer the topic |
| `weights` | Target distribution over the five evidence categories (sums to 1) |

Identifiers run from 1 to 143 with five gaps. Topic 35 was withdrawn during
annotation. Topics 3, 125, 128 and 142 were withdrawn because the pool contains
no relevant passage for them: two ask which Members of a given nationality sit
on a named committee, which the register answers one Member at a time rather
than in any single passage, and two ask for a specific date or vote result that
no passage states directly.

### Judgements file

TREC qrels format, one judgement per line. Additionally, we keep the category/s 
of each respective passage.

```
topic_id  passage_id  relevance_score  doc_category
```

`label` is 1 for relevant and 0 for not relevant. The second column is the
unused iteration field retained for compatibility with standard evaluation
tools. Only judged pairs appear; a passage absent from the file is unjudged
rather than known to be irrelevant.

## Evidence categories

| Category | What it can be used to establish |
|---|---|
| Decision | What Parliament as a body decided or adopted, and what the adopted text says |
| Position | What a political actor (Member, political group, party, or delegation) stands for, argued, or asked |
| Proceedings | Where a file or meeting stands, what happened when, and what is under debate |
| Actors | Who Members and political groups are, and what offices they hold |
| Institution | How the EU and Parliament are constituted and how their rules and procedures work in general |

## Corpus

The corpus covers the ninth parliamentary term and the first two years of the tenth.

| Document type | Documents | Passages | Evidence categories |
|---|---:|---:|---|
| *Primary record* | | | |
| Debate intervention | 2,112 | 78,357 | position, proceedings |
| Verbatim report of proceedings | 4,549 | 16,026 | proceedings |
| Minutes | 10,942 | 10,942 | decision, proceedings |
| Text adopted | 2,969 | 12,772 | decision |
| Report | 1,800 | 74,359 | proceedings |
| Committee opinion | 1,400 | 22,575 | proceedings |
| Motion for resolution | 2,775 | 18,067 | position |
| Written question | 23,797 | 23,797 | position |
| Explanation of vote | 137 | 137 | position |
| *Secondary and reference material* | | | |
| Press release | 1,509 | 2,041 | decision, position, proceedings |
| Fact sheet | 177 | 814 | institution |
| Rule | 251 | 259 | institution |
| Annex | 8 | 24 | institution |
| *Structured data rendered as text* | | | |
| Procedure file | 3,177 | 15,955 | proceedings |
| MEP profile: authored documents | 1,258 | 14,770 | actors, proceedings |
| MEP profile: biography | 1,277 | 1,277 | actors |
| MEP profile: committee memberships | 1,269 | 1,293 | actors |
| MEP profile: delegation memberships | 1,259 | 1,264 | actors |
| MEP profile: dossier roles | 771 | 1,062 | actors, proceedings |
| MEP profile: other roles | 211 | 211 | actors |
| MEP profile: political affiliations | 1,277 | 1,277 | actors |
| **Total** | **62,924** | **297,279** | |

### Languages

The documentary record is in English throughout: adopted texts, minutes, committee reports and opinions, motions, written questions, procedure files, press releases, fact sheets, the Rules of Procedure and the Member profiles.

Plenary speeches are retained in the language in which they were delivered, and are not translated. The spoken record — debate interventions, verbatim reports of proceedings and explanations of vote — is 94,520 passages, 31.8% of the corpus, and 70% of it is in a language other than English. That is 66,239 passages, 22% of the collection, spanning 21 of the 24 official languages of the Union.

| Language | Passages | Language | Passages |
|---|---:|---|---:|
| German | 10,687 | Swedish | 1,833 |
| French | 9,106 | Czech | 1,690 |
| Spanish | 7,793 | Croatian | 1,668 |
| Italian | 6,352 | Slovak | 1,544 |
| Polish | 5,275 | Finnish | 1,278 |
| Portuguese | 3,920 | Bulgarian | 1,178 |
| Romanian | 2,976 | Danish | 948 |
| Dutch | 2,738 | Slovenian | 865 |
| Greek | 2,519 | Lithuanian | 781 |
| Hungarian | 1,986 | Latvian | 539 |
| | | Estonian | 486 |

Irish and Maltese do not appear. Languages were identified automatically from the passage text; a small residue of passages, around 0.1% of the non-English material, carries an implausible label and should be treated as detection error.

Users should note that the topics are in English while a substantial part of the speech record is not, so retrieval over this collection is in part a cross-lingual task.

Documents are split into passages following their natural structure where possible. For example, transcripts are split into one passage per speech, and written questions are kept together with their answers. Longer documents are split into passages of fixed length. Procedure files and MEP profiles are database records, which are rendered as text from fixed templates without the use of a language model.

## Topics

The 138 topics are based on questions asked by users of the ParliView platform. Each topic has a title, a description, a narrative, and a target distribution over the five evidence categories, giving the proportion of each kind of evidence a good answer would rely on.

The target distributions were assigned independently by three expert annotators familiar with the European Parliament and with this collection, each distributing ten points across the categories for every topic. The released distribution is their mean. Because each annotation is a distribution rather than a label, agreement is measured with Krippendorff's α under a difference function: over the 134 topics all three annotators covered, α = 0.75 taking total variation distance as the difference function, and 0.89 taking squared Euclidean distance. The dominant category is unanimous on 107 of those 134 topics.

## Relevance judgements

The collection includes binary relevance judgements for 59,232 topic–passage pairs, a mean of 429 per topic. Candidates were pooled to depth 250 from two systems: a BM25 run over the document titles, and a dense run with BGE-M3 over the whole corpus, searching with the topic description. Pooling from both matters here because BM25 cannot match the non-English speech record at all, so a single-system pool drawn from it under-represents the material that answers questions about what Members argued.

We constructed the relevance pool by combining the top 500 passages from three retrieval systems: (i) BM25 (ii) standalone dense retrieval using BGE-M3, and (iii) BM25–BGE-M3 fusion using reciprocal rank fusion (RRF). 

Labels were generated with an LLM (OpenAI `gpt-6-luna`) prompted with the topic title, description and narrative together with the passage text. 8,540 pairs (14.4%) are judged relevant, a mean of 62 and a median of 30 per topic, ranging from 1 to 318. Every released topic has at least one relevant passage.

The labels were validated against an expert annotator who independently judged the top 20 passages of a stratified sample of topics without sight of the model's labels. Over 160 such judgements, agreement is 85.0% with Cohen's κ = 0.67. Agreement varies by evidence category, and is lowest on topics whose answers rest on what Parliament decided, where a passage discussing a measure is readily mistaken for one that settles it.

## Original data sources

All distributed texts and records are taken from European Parliament sources:

- [European Parliament website](https://www.europarl.europa.eu/portal/en): verbatim reports, minutes, adopted texts, reports, opinions, motions, written questions, press releases, the [Fact Sheets on the European Union](https://www.europarl.europa.eu/factsheets/en/home), and the Rules of Procedure
- [European Parliament Open Data Portal](https://data.europarl.europa.eu/en/developer-corner/opendata-api)
- [Legislative Observatory (OEIL)](https://oeil.europarl.europa.eu)

[ParlTrack](https://parltrack.org) was used only to identify which documents exist and where they are published. No ParlTrack data is included in this release.

## Licence

The data is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) licence. Source texts are reused under the European Parliament's open data policy. Attribution: © European Union, 2019–2026 – Source: European Parliament. Each passage includes the URL of its source document.

The code is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
