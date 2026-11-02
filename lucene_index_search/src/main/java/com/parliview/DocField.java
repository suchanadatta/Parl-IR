package com.parliview;

/**
 * Names of the fields stored in the ParliView Lucene index.
 * Use these constants in both the indexer and the retriever so the names never drift apart.
 *
 * Source data: one JSON object per line (.jsonl), with the keys
 *   id, category, doc_type, title, text,
 *   mep_id, mep_name, active, as_of_date, coverage_start, coverage_end,
 *   dossier_refs, doc_ids
 *
 * Records from different collections share this schema. Keys that do not apply to a
 * record are null or empty (for example, a fact sheet has no mep_id) and are simply
 * not indexed for that document.
 *
 * Multi-valued fields: any key may hold a single value or a list. A list becomes one
 * Lucene field instance per value (for example "category": ["actors", "proceedings"]),
 * so read such fields with Document.getValues(...), not Document.get(...), which
 * returns only the first value.
 *
 * Other keys: a key that is not listed below is still indexed and stored under its own
 * name (a key that would clash with CONTENT, RAW_CONTENT or SOURCE is stored with the
 * prefix "extra_"). The complete original record is always available in SOURCE.
 */
public final class DocField {

    private DocField() { }

    // ------------------------------------------------------------------
    // Identity and category (present on every document)
    // ------------------------------------------------------------------

    /**
     * Unique id of the indexed entry (JSON key: "id").
     * Stored, not tokenized.
     */
    static final public String ID = "id";

    /**
     * Categories of the record (JSON key: "category"). A document can belong to several,
     * e.g. ["actors", "proceedings"].
     * Stored, not tokenized, multi-valued: one field instance per category.
     * A TermQuery matches if ANY of the document's categories matches, so it can be used
     * to restrict or route retrieval by category.
     * Read all values with Document.getValues(DocField.CATEGORY).
     */
    static final public String CATEGORY = "category";

    /**
     * Document type label, e.g. "MEP profile: authored documents", "Fact sheet"
     * (JSON key: "doc_type").
     * Stored, not tokenized. Use this when you need to show the type of a retrieved text.
     */
    static final public String DOC_TYPE = "doc_type";

    // ------------------------------------------------------------------
    // Text fields
    // ------------------------------------------------------------------

    /**
     * Title of the document (JSON key: "title"). Only some collections have one.
     * Stored and analyzed.
     */
    static final public String TITLE = "title";

    /**
     * Cleaned but unanalyzed body text (JSON key: "text").
     * Stored only: this is what you return to the user as the plain text of a hit.
     */
    static final public String RAW_CONTENT = "rawcontent";

    /**
     * Analyzed body text (same JSON key "text", run through the analyzer).
     * Searchable but not stored: this is the field to search against with BM25.
     * To show the text of a hit, read RAW_CONTENT instead.
     */
    static final public String CONTENT = "content";

    // ------------------------------------------------------------------
    // MEP-specific fields (only on MEP records)
    // ------------------------------------------------------------------

    /**
     * Numeric MEP identifier, kept as a string (JSON key: "mep_id").
     * Stored, not tokenized: match with a TermQuery.
     */
    static final public String MEP_ID = "mep_id";

    /**
     * Name of the MEP, e.g. "Paolo BORCHIA" (JSON key: "mep_name").
     * Stored and analyzed (lowercased, not stemmed), so partial names can be searched.
     */
    static final public String MEP_NAME = "mep_name";

    /**
     * Whether the MEP is currently serving (JSON key: "active").
     * Stored as the string "true" or "false", not tokenized.
     */
    static final public String ACTIVE = "active";

    // ------------------------------------------------------------------
    // Dates (ISO yyyy-MM-dd strings: they sort correctly as text)
    // ------------------------------------------------------------------

    /**
     * Date the data was observed (JSON key: "as_of_date").
     * Stored, not tokenized.
     */
    static final public String AS_OF_DATE = "as_of_date";

    /**
     * Start of the period the record covers (JSON key: "coverage_start").
     * Stored, not tokenized.
     */
    static final public String COVERAGE_START = "coverage_start";

    /**
     * End of the period the record covers (JSON key: "coverage_end").
     * Stored, not tokenized.
     */
    static final public String COVERAGE_END = "coverage_end";

    // ------------------------------------------------------------------
    // Multi-valued references (one Lucene field instance per value)
    // ------------------------------------------------------------------

    /**
     * Ids of related documents (JSON key: "doc_ids", an array).
     * Stored, not tokenized, multi-valued: a document can carry several.
     * Read all values with Document.getValues(DocField.DOC_ID).
     */
    static final public String DOC_ID = "doc_id";

    /**
     * Ids of related legislative dossiers (JSON key: "dossier_refs", an array).
     * Stored, not tokenized, multi-valued.
     * Read all values with Document.getValues(DocField.DOSSIER_REF).
     */
    static final public String DOSSIER_REF = "dossier_ref";

    // ------------------------------------------------------------------
    // Original record
    // ------------------------------------------------------------------

    /**
     * The full original JSON line.
     * Stored only, never searched. Parse it to read any field that is not stored separately.
     */
    static final public String SOURCE = "_source";
}