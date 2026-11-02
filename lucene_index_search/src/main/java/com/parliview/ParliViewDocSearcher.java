package com.parliview;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilderFactory;
import java.io.BufferedWriter;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Runs the title of every topic against the ParliView index with BM25 and writes
 * the top-k results in a TREC-style run file.
 *
 * Usage: ParliViewSearcher <indexDir> <topics.xml> <output.run> <runName> [k=100]
 *
 * Topics file: <topics><top><num>..</num><title>..</title><desc>..</desc><narr>..</narr></top>...</topics>
 * Only <num> and <title> are used.
 *
 * Output: one line per result, tab-separated:
 *   query_id  Q0  document_id  rank  score  full_text  category  run_name
 * A document with several categories shows them joined with "|", e.g. actors|proceedings.
 */
public class ParliViewDocSearcher {

    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("Usage: ParliViewSearcher <indexDir> <topics.xml> <output.run> <runName> [k=100]");
            System.exit(1);
        }
        Path indexDir = Paths.get(args[0]);
        Path topicsFile = Paths.get(args[1]);
        Path outFile = Paths.get(args[2]);
        String runName = args[3];
        int k = args.length > 4 ? Integer.parseInt(args[4]) : 100;

        List<String[]> topics = readTopics(topicsFile);   // each entry: {num, title}
        System.out.println("Read " + topics.size() + " topics from " + topicsFile);

        try (Directory dir = FSDirectory.open(indexDir);
             DirectoryReader reader = DirectoryReader.open(dir);
             BufferedWriter out = Files.newBufferedWriter(outFile, StandardCharsets.UTF_8)) {

            IndexSearcher searcher = new IndexSearcher(reader);
            searcher.setSimilarity(new BM25Similarity());   // same BM25 settings as the indexer

            // Must be the same analyzer the indexer used, or stemmed terms will not match
            QueryParser parser = new QueryParser(DocField.CONTENT, ParliViewDocIndexer.createAnalyzer());

            for (String[] topic : topics) {
                String qid = topic[0];
                String title = topic[1];

                Query query = parser.parse(QueryParser.escape(title));
                TopDocs hits = searcher.search(query, k);

                int rank = 1;
                for (ScoreDoc sd : hits.scoreDocs) {
                    Document doc = searcher.doc(sd.doc);

                    String docId = nz(doc.get(DocField.ID));
                    String text = oneLine(doc.get(DocField.RAW_CONTENT));
                    String category = String.join("|", doc.getValues(DocField.CATEGORY));

                    out.write(qid + "\t" + "Q0" + "\t" + docId + "\t" + rank + "\t"
                            + String.format(Locale.ROOT, "%.6f", sd.score) + "\t"
                            + text + "\t" + category + "\t" + runName);
                    out.newLine();
                    rank++;
                }
                System.out.println("topic " + qid + ": " + hits.scoreDocs.length + " results");
            }
        }
        System.out.println("Run written to " + outFile);
    }

    /** Reads <top><num>..</num><title>..</title>..</top> entries from the topics file. */
    private static List<String[]> readTopics(Path file) throws Exception {
        DocumentBuilderFactory f = DocumentBuilderFactory.newInstance();
        f.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);  // no DTD/entities

        org.w3c.dom.Document xml;
        try (InputStream in = Files.newInputStream(file)) {
            xml = f.newDocumentBuilder().parse(in);
        }

        List<String[]> topics = new ArrayList<>();
        NodeList tops = xml.getElementsByTagName("top");
        for (int i = 0; i < tops.getLength(); i++) {
            Element top = (Element) tops.item(i);
            String num = text(top, "num");
            String title = text(top, "title");
            if (num.isEmpty() || title.isEmpty()) {
                throw new IllegalArgumentException("topic " + (i + 1) + " is missing <num> or <title>");
            }
            topics.add(new String[] { num, title });
        }
        return topics;
    }

    private static String text(Element parent, String tag) {
        NodeList nl = parent.getElementsByTagName(tag);
        return nl.getLength() == 0 ? "" : nl.item(0).getTextContent().trim();
    }

    /** Replaces tabs and line breaks so each result stays on one line of the run file. */
    private static String oneLine(String s) {
        return s == null ? "" : s.replaceAll("[\\t\\r\\n]+", " ").trim();
    }

    private static String nz(String s) {
        return s == null ? "" : s;
    }
}