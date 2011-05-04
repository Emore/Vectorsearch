package vectorsearch;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;

public class VectorSearch {
	// dir with all documents in .txt
	private static final String DOCS_DIR = "data/docs/"; 
	// norm doc lengths
	private static final String DOC_LENGTHS = "data/doc_lengths.txt";
	// inverted index
	private static final String INVERSE_INDEX = "data/index.txt"; 
	// relevant docs
	private static final String RELEVANT_DIR = "data/relevant.txt"; 
	// relevant/unrel docs
	private static final String FEEDBACK_DIR = "data/feedback.txt"; 
	// rel with feedback docs removed
	private static final String RELEVANT_NOF_DIR = "data/relevant_nofback.txt";
	
	private static final float alpha = 1f;
	private static final float beta  = 0.85f;
	private static final float gamma = 0.15f;
	
	// serialized data structures
	private Set<String> dictionary = new HashSet<String>();
	private Set<File> relevant = new HashSet<File>();
	private Set<File> relevantNoF = new HashSet<File>();
	private Map<Integer, List<File>> feedback = 
			new TreeMap<Integer, List<File>>();
	private Map<File, Map<String, Float>> documents = 
			new HashMap<File, Map<String, Float>>();
	private Map<File, Float> documentNorms = new HashMap<File, Float>();
	
	private float queryNorm;
	
	private Serializer serializer;
	
	/**
	 * Initialize data structures and continuously read stdin.
	 */
	@SuppressWarnings("unchecked")
	public VectorSearch() {
		serializer = new Serializer();
		
		System.out.println("Setup started...");
		long start = System.currentTimeMillis();
		
		// build data structures from files
		try {
			Scanner scan;
			String DOCUMENT_NORMS = "norms.serial";
			try {
				documentNorms = (Map<File, Float>) 
						serializer.deserialize(DOCUMENT_NORMS);
			} catch (IOException e1) {
				scan = new Scanner(new File(DOC_LENGTHS));
				while (scan.hasNext()) {
					File document = new File(DOCS_DIR + scan.next());
					float length = scan.nextFloat();
					documentNorms.put(document, length);
				}
				System.out.println("Built document norms index.");
				serializer.serialize((Object) documentNorms, DOCUMENT_NORMS);
			}
			
			String DICTIONARY = "dictionary.serial";
			try {
				dictionary = (Set<String>) serializer.deserialize(DICTIONARY);
			} catch (IOException e1) {
				scan = new Scanner(new File(INVERSE_INDEX));
				while (scan.hasNext()) {
					Scanner line = new Scanner(scan.nextLine());
					String term = line.next().toLowerCase();
					int docFreq = line.nextInt();
					if (docFreq >= 6 && docFreq <= 1600)
						dictionary.add(term);
				}
				System.out.println("Built dictionary of terms.");
				serializer.serialize((Object) dictionary, DICTIONARY);
			}
			
			String RELEVANT = "relevant.serial";
			try {
				relevant = (Set<File>) serializer.deserialize(RELEVANT);
			} catch (IOException e1) {
				scan = new Scanner(new File(RELEVANT_DIR));
				while (scan.hasNext()) {
					File file = new File(DOCS_DIR + scan.next());
					relevant.add(file);
				}
				System.out.println("Built relevant files index.");
				serializer.serialize((Object) relevant, RELEVANT);
			}
			
			String RELEVANT_NOF = "relevantNoF.serial";
			try {
				relevantNoF = (Set<File>) serializer.deserialize(RELEVANT_NOF);
			} catch (IOException e1) {
				scan = new Scanner(new File(RELEVANT_NOF_DIR));
				while (scan.hasNext()) {
					File file = new File(DOCS_DIR + scan.next());
					relevantNoF.add(file);
				}
				System.out.println("Built relevantNoF files index.");
				serializer.serialize((Object) relevantNoF, RELEVANT_NOF);
			}
			
			String FEEDBACK = "feedback.serial";
			try {
				feedback = (Map<Integer, List<File>>) 
						serializer.deserialize(FEEDBACK);
			} catch (IOException e1) {
				scan = new Scanner(new File(FEEDBACK_DIR));
				feedback.put(1, new LinkedList<File>());
				feedback.put(0, new LinkedList<File>());

				while (scan.hasNext()) {
					File file = new File(DOCS_DIR + scan.next());
					int i = scan.nextInt();
					if (i == 1)
						feedback.get(1).add(file);
					else if (i == 0)
						feedback.get(0).add(file);
				}
				System.out.println("Built feedback files index.");
				serializer.serialize((Object) feedback, FEEDBACK);
			}
			
			String DOCUMENTS_SERIAL = "index.serial";
			try {
				documents = (Map<File, Map<String, Float>>)
						serializer.deserialize(DOCUMENTS_SERIAL);
			} catch (IOException e1) {
				buildDocuments(DOCUMENTS_SERIAL);
			}
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		long stop = System.currentTimeMillis();
		
		System.out.println("Done: " + ((stop - start)) + "ms.\n");
		
		BufferedReader in = new BufferedReader(
				new InputStreamReader(System.in));
		System.out.println("Enter query: ");
		while (true) {
			try {
				System.out.print("> "); System.out.flush();
				String input = in.readLine(); // waits for input
				
				start = System.currentTimeMillis();
				
				List<String> args = Arrays.asList(input.split("\\s+"));
				List<String> queryTokens;
				File feedbackFile = null;
				File relevantFile = null;
				Map<String, Float> query = new HashMap<String, Float>();
				
				if (args.contains("-h")) {
					System.out.println(
							"java VectorSearch -q Search Terms " + 
							"[-f .txt-file with feedback]");
				} else if (args.contains("-q")) {
					int toIndex = args.subList(args.indexOf("-q"),
							args.size()).contains("-f") ? 
									args.indexOf("-f") : args.size();
					
					queryTokens = args.subList(args.indexOf("-q") + 1, toIndex);
					
					if (args.contains("-f"))
						feedbackFile = new File(
								args.get(args.indexOf("-f") + 1));
					
					if (args.contains("-r"))
						relevantFile = new File(
								args.get(args.indexOf("-r") + 1));
					else {
						System.out.println(
								"Please provide relevance judgements.");
						continue;
					}
					
					query = (feedbackFile != null) ? 
							buildQuery(queryTokens, feedback) : 
							buildQuery(queryTokens, null);
					
					Map<File, Float> results = search(query);
					stop = System.currentTimeMillis();
					
					if (relevantFile.getPath().equals(RELEVANT_NOF_DIR)) {
						print(results, (stop-start), relevantNoF);
					} else
						print(results, (stop-start), relevant);
				} else 
					System.out.println(
							"Please provide a query, preceded by -q.");
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	/**
	 * Performs a cosine measure, comparing the query to the documents.
	 * 
	 * @return a map of files and their corresponding similarity measure
	 */
	public Map<File, Float> search(Map<String, Float> query) {
		Map<File, Float> results = new HashMap<File, Float>();

		for (File file : documents.keySet()) {
			Map<String, Float> document = documents.get(file);
			float dot = 0;
			for (String term : query.keySet()) {
				if (document.get(term) != null) {
					dot += document.get(term) * query.get(term);
				}
			}
			if (dot > 0)
				results.put(file, 
						(dot / (documentNorms.get(file) * queryNorm)));
		}
		return results;
	}
	
	/**
	 * Prints a map of files and their corresponding similarity to stdout, 
	 * also calculating precision and recall at rank, precision at recall
	 * values as well as an average precision.
	 * 
	 * @param results, a map of files and their similarities (rank)
	 * @param time, the time it took to perform the search, in ms
	 */
	public void print(final Map<File, Float> results, long time, 
			Set<File> relevant) {
		List<File> filesByRank = new ArrayList<File>(results.keySet());
		
		// Sort files based on their similarity measure
		Collections.sort(filesByRank, new Comparator<File>() {
			@SuppressWarnings({ "rawtypes", "unchecked" })
			public int compare(File o1, File o2) {
				Object v1 = results.get(o1);
				Object v2 = results.get(o2);
				if (v1 == null) {
					return (v2 == null) ? 0 : 1;
				} else if (v1 instanceof Comparable) {
					return ((Comparable) v2).compareTo(v1);
				} else
					return 0;
			}
		});
									
		System.out.println(
				" Rank | Similarity | Document         | Precision  Recall ");
		System.out.println(
				"------+------------|------------------+-------------------");
		
		float relevantTotal = relevant.size();
		float relevantSeen = 0;
		
		TreeSet<Float> recallValues = new TreeSet<Float>();
		Map<Float, Float> precisionAtRecall = new HashMap<Float, Float>();
		
		float precision = 0.0f;
		float recall    = 0.0f;
		
		// print precision and recall at rank
		for (int i = 0; i < filesByRank.size(); i++) {
			
			if (relevant.contains(filesByRank.get(i))) {
				relevantSeen++;
				precision = relevantSeen/(float)(i+1);
				recall    = relevantSeen/relevantTotal;
				recallValues.add(recall);
				precisionAtRecall.put(recall, precision);
			}
			
			System.out.printf(
					"%1$4d  |  %2$4.4f    | %3$s |  %4$2.2f       %5$2.2f", 
					i+1, 
					results.get(filesByRank.get(i)),
					filesByRank.get(i).getName(),
					precision,
					recall);
			System.out.println("");
			if (relevantSeen/relevantTotal == 1.0)
				break;
		}
		
		// print precision-at-recall
		System.out.println("");
		System.out.println(" Recall | Precision ");
		System.out.println("--------+-----------");
		
		int recallLevels = 11; // standard in Baeza-Yates & Ribeiro-Neto
		float averagePrecision = 0.0f;

		for (int i = 0; i < recallLevels; i++) {
			
			recall = (float)(i)/(float)(recallLevels-1);
			
			// all intermediate recall levels for one full standard level
			float ceil = recall + ((float) 1 / (float)(recallLevels-1));
			SortedSet<Float> recallForInterval = 
					recallValues.subSet(recall, ceil);
			
			// find max precision level among all recall levels for stndrd level
			precision = 0.0f;
			for (Entry<Float, Float> entry : precisionAtRecall.entrySet()) {
				if (recallForInterval.contains(entry.getKey()) && 
						entry.getValue() > precision) {
					precision = entry.getValue();
				}
			}
			
			averagePrecision += precision;
			
			System.out.printf(" %1$1.1f    |  %2$1.2f", recall, precision);
			System.out.println("");
		}
		
		// finally print some key values
		System.out.println("");
		System.out.printf("%1$s: %2$1.2f\n", "Average precision", 
				(averagePrecision/(float) recallLevels));
		System.out.println(results.size() + " results in total (" + 
				time + "ms).");
	}
		
	/**
	 * Builds a map of weights for each document, using standard tf x idf
	 * weights.
	 */
	public void buildDocuments(String serialName) {
		// document -> (term, term frequency)
		Map<File, Map<String, Integer>> tf = 
				new HashMap<File, Map<String, Integer>>();
		// term -> document frequency ^-1
		Map<String, Float> idf = new HashMap<String, Float>();
		
		try {
			buildTFIDF(tf, idf);
		} catch (FileNotFoundException e2) {
			System.err.println("File not found: " + e2.getMessage());
		}
		
		//int N = tf.size();
		Set<File> files = tf.keySet();
		for (File file : files) {
			Map<String, Float> vector = new HashMap<String, Float>();
			Map<String, Integer> docTf = tf.get(file);
			
			for (String term : docTf.keySet()) {
				float weight = (float) docTf.get(term) * idf.get(term);
				vector.put(term, weight);
			}
				
			documents.put(file, vector);
		}
		
		System.out.println("Built document vectors.");
		serializer.serialize((Object) documents, serialName);

	}

	/**
	 * Builds a query vector from input terms. If relevance feedback given, the
	 * query vector applies the Rocchio feedback mechanism.
	 * 
	 * @param terms, a tokenized list of input strings
	 * @param feedback, a map with key 1 for a list of relevant docs; 0 for list 
	 * 		  of irrelevant docs. null if no feedback given.
	 * @return a map of strings and their query weights
	 */
	public Map<String, Float> buildQuery(List<String> terms, 
			Map<Integer, List<File>> feedback) {
		Map<String, Float> query = new HashMap<String, Float>();
		
		float defaultWeight = (feedback != null) ? alpha : 1f;
		
		for (String term : terms) {
			if (dictionary.contains(term.toLowerCase()))
				query.put(term.toLowerCase(), defaultWeight);
		}
		
		if (feedback != null) {
			int relevantSize = feedback.get(1).size();
			int irrelevantSize = feedback.get(0).size();
			
			// add weights of relevant documents
			for (File file : feedback.get(1)) {
				Map<String, Float> document = documents.get(file);
				for (String term : document.keySet()) {
					float incr = 
							beta * 1f/(float)relevantSize * document.get(term);
					if (query.containsKey(term))
						query.put(term, query.get(term)+incr);
					else
						query.put(term, incr);
				}
			}
			
			// subtract weights of irrelevant documents
			for (File file : feedback.get(0)) {
				Map<String, Float> document = documents.get(file);
				for (String term : document.keySet()) {
					if (query.containsKey(term)) {
						float decr = gamma * 1f/(float)irrelevantSize * 
								document.get(term);
						float newWeight = (query.get(term) > decr) ? 
								query.get(term)-decr : 0;
						if (newWeight != 0)
							query.put(term, newWeight);
						else
							query.remove(term); // remove query term if weight 0
					}
				}
			}
		}
	
		// update the norm of the query
		queryNorm = 0;
		for (float f : query.values()) {
			queryNorm += f*f;
		}
		queryNorm = (float) Math.sqrt(queryNorm);
		
		return query;
	}

	/**
	 * Builds the tf and idf data structures necessary to compute the document
	 * vectors.
	 * 
	 * @param tf, a map of files and their term frequencies
	 * @param idf, a map of terms and their inverse document frequencies
	 * @throws FileNotFoundException
	 */
	private void buildTFIDF(Map<File, Map<String, Integer>> tf, 
			Map<String, Float> idf) throws FileNotFoundException {
		Scanner fileScanner = new Scanner(new File(INVERSE_INDEX));
		
		while (fileScanner.hasNextLine()) {
			Scanner lineScanner = new Scanner(fileScanner.nextLine());
			
			String term = lineScanner.next().toLowerCase();
			float docFreq = lineScanner.nextInt();
			
			if (docFreq >= 6f && docFreq <= 1600f) {
				
				idf.put(term, 1 / docFreq);
				
				while (lineScanner.hasNext()) {
					File document = new File(DOCS_DIR + lineScanner.next());
					int termFreq = lineScanner.nextInt();
					
					Map<String, Integer> termFrequencies = tf.get(document);
					
					if (termFrequencies == null) {
						tf.put(document, new HashMap<String, Integer>());
						termFrequencies = tf.get(document);
					}
					
					termFrequencies.put(term, termFreq);
				}
			}
		}
		System.out.println("Built TF and IDF.");
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		new VectorSearch();
	}
}
