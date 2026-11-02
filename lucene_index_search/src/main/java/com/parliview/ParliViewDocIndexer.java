package com.parliview;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.BufferedReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

/**
 * Builds the ParliView Lucene index from a unified .jsonl collection.
 *
 * Usage: ParliViewDocIndexer <collection.jsonl> <indexDir>
 *
 * Field names come from {@link DocField}. The retriever must open the index with
 * the same analyzer: call {@link #createAnalyzer()} there.
 *
 * A record is one JSON object per line. Any key may hold a single value or a list;
 * lists become multi-valued Lucene fields (e.g. "category": ["actors", "proceedings"]).
 */
public class ParliViewDocIndexer {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** Keys that are mapped to a DocField explicitly in toDocument(). */
    private static final Set<String> KNOWN_KEYS = new HashSet<>(Arrays.asList(
            "id", "text", "category", "doc_type", "title",
            "mep_id", "mep_name", "active",
            "as_of_date", "coverage_start", "coverage_end",
            "doc_ids", "dossier_refs"));

    /** Index field names that an unknown JSON key must not overwrite. */
    private static final Set<String> RESERVED_FIELDS = new HashSet<>(Arrays.asList(
            DocField.CONTENT, DocField.RAW_CONTENT, DocField.SOURCE));

    /** Unknown string values longer than this are analysed as text instead of exact-match. */
    private static final int EXACT_MATCH_MAX_CHARS = 256;

    /**
     * Analyzer used for indexing AND for parsing queries.
     * - CONTENT, TITLE (and unknown long-text fields): EnglishAnalyzer (lowercase, stopwords removed, stemmed)
     * - MEP_NAME: StandardAnalyzer (lowercase only), so names are not stemmed
     */
    public static Analyzer createAnalyzer() {
        Map<String, Analyzer> perField = new HashMap<>();
        perField.put(DocField.MEP_NAME, new StandardAnalyzer());
        return new PerFieldAnalyzerWrapper(new EnglishAnalyzer(), perField);
    }

    public static void main(String[] args) throws Exception {
        
//        mvn clean compile exec:java -Dexec.mainClass=com.parliview.ParliViewDocIndexer \
//       -Dexec.args="/Users/suchana/Corpus/parliview_unified.jsonl /Users/suchana/Index/parliview_lucene8/"
        if (args.length != 2) {
            System.err.println("Usage: ParliViewDocIndexer <collection.jsonl> <indexDir>");
            System.exit(1);
        }
        Path jsonl = Paths.get(args[0]);
        Path indexDir = Paths.get(args[1]);

        IndexWriterConfig iwc = new IndexWriterConfig(createAnalyzer());
        iwc.setSimilarity(new BM25Similarity());              // BM25, k1=1.2, b=0.75
        iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE);   // always rebuild from scratch

        long indexed = 0, skipped = 0, lineNo = 0, multiCategory = 0;

        try (Directory dir = FSDirectory.open(indexDir);
             IndexWriter writer = new IndexWriter(dir, iwc);
             BufferedReader br = Files.newBufferedReader(jsonl, StandardCharsets.UTF_8)) {

            String line;
            while ((line = br.readLine()) != null) {
                lineNo++;
                if (line.trim().isEmpty()) continue;
                try {
                    Document d = toDocument(line);
                    writer.addDocument(d);
                    indexed++;
                    if (d.getValues(DocField.CATEGORY).length > 1) multiCategory++;
                } catch (Exception e) {
                    skipped++;
                    System.err.println("Skipping line " + lineNo + ": " + e.getMessage());
                }
                if (indexed > 0 && indexed % 10000 == 0) {
                    System.out.println("indexed " + indexed);
                }
            }
            writer.commit();
        }

        // Verify what is actually in the index
        try (Directory dir = FSDirectory.open(indexDir);
             DirectoryReader reader = DirectoryReader.open(dir)) {
            System.out.println("Indexed: " + indexed + ", skipped: " + skipped
                    + ", documents in index: " + reader.numDocs()
                    + ", documents with more than one category: " + multiCategory);
        }
    }

    /** One JSON line -> one Lucene Document. */
    static Document toDocument(String jsonLine) throws Exception {
        JsonNode rec = MAPPER.readTree(jsonLine);
        if (rec == null || !rec.isObject()) {
            throw new IllegalArgumentException("line is not a JSON object");
        }

        String id = str(rec, "id");
        String text = str(rec, "text");
        if (id == null || text == null) {
            throw new IllegalArgumentException("record is missing 'id' or 'text'");
        }

        Document d = new Document();

        // Identity and category: exact match, stored. category may hold several values.
        d.add(new StringField(DocField.ID, id, Field.Store.YES));
        addStrings(d, DocField.CATEGORY, rec.get("category"));
        addStrings(d, DocField.DOC_TYPE, rec.get("doc_type"));

        // Text: RAW_CONTENT is returned to the user, CONTENT is what BM25 searches
        d.add(new StoredField(DocField.RAW_CONTENT, text));
        d.add(new TextField(DocField.CONTENT, text, Field.Store.NO));
        addTexts(d, DocField.TITLE, rec.get("title"));

        // MEP fields (null on other collections, so skipped)
        addStrings(d, DocField.MEP_ID, rec.get("mep_id"));
        addTexts(d, DocField.MEP_NAME, rec.get("mep_name"));
        addStrings(d, DocField.ACTIVE, rec.get("active"));

        // Dates: ISO strings
        addStrings(d, DocField.AS_OF_DATE, rec.get("as_of_date"));
        addStrings(d, DocField.COVERAGE_START, rec.get("coverage_start"));
        addStrings(d, DocField.COVERAGE_END, rec.get("coverage_end"));

        // Multi-valued references: one field instance per value
        addStrings(d, DocField.DOC_ID, rec.get("doc_ids"));
        addStrings(d, DocField.DOSSIER_REF, rec.get("dossier_refs"));

        // Any other key in the record is kept too, under its own name
        Iterator<Map.Entry<String, JsonNode>> it = rec.fields();
        while (it.hasNext()) {
            Map.Entry<String, JsonNode> e = it.next();
            if (!KNOWN_KEYS.contains(e.getKey())) {
                addExtra(d, e.getKey(), e.getValue());
            }
        }

        // The full original record, so nothing is ever lost
        d.add(new StoredField(DocField.SOURCE, jsonLine));

        return d;
    }

    // ---- Helpers ---------------------------------------------------------------

    /** Value of a scalar JSON key as a string, or null if missing, null or empty. */
    private static String str(JsonNode rec, String key) {
        JsonNode n = rec.get(key);
        if (n == null || n.isNull() || n.isContainerNode()) return null;
        String s = n.asText();
        return s.trim().isEmpty() ? null : s;
    }

    /** Exact-match, stored. Accepts a single value or a list (one field instance per value). */
    private static void addStrings(Document d, String field, JsonNode node) {
        if (node == null || node.isNull()) return;
        if (node.isArray()) {
            for (JsonNode item : node) addOneString(d, field, item);
        } else {
            addOneString(d, field, node);
        }
    }

    private static void addOneString(Document d, String field, JsonNode item) {
        if (item == null || item.isNull() || item.isContainerNode()) return;
        String s = item.asText();
        if (!s.trim().isEmpty()) d.add(new StringField(field, s, Field.Store.YES));
    }

    /** Analyzed text, stored. Accepts a single value or a list. */
    private static void addTexts(Document d, String field, JsonNode node) {
        if (node == null || node.isNull()) return;
        if (node.isArray()) {
            for (JsonNode item : node) addOneText(d, field, item);
        } else {
            addOneText(d, field, node);
        }
    }

    private static void addOneText(Document d, String field, JsonNode item) {
        if (item == null || item.isNull() || item.isContainerNode()) return;
        String s = item.asText();
        if (!s.trim().isEmpty()) d.add(new TextField(field, s, Field.Store.YES));
    }

    /**
     * A key the indexer does not know. Short values become exact-match fields, long
     * values become analysed text, and nested objects are kept as stored JSON text.
     */
    private static void addExtra(Document d, String key, JsonNode node) {
        if (node == null || node.isNull()) return;
        String field = RESERVED_FIELDS.contains(key) ? "extra_" + key : key;

        if (node.isArray()) {
            for (JsonNode item : node) addExtra(d, key, item);
        } else if (node.isObject()) {
            d.add(new StoredField(field, node.toString()));
        } else {
            String s = node.asText();
            if (s.trim().isEmpty()) return;
            if (s.length() <= EXACT_MATCH_MAX_CHARS) {
                d.add(new StringField(field, s, Field.Store.YES));
            } else {
                d.add(new TextField(field, s, Field.Store.YES));
            }
        }
    }
}