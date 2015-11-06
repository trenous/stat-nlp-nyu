package nlp.assignments;

import java.util.*;
import java.util.Map.Entry;
import java.io.*;
import java.nio.charset.StandardCharsets;

import nlp.io.IOUtils;
import nlp.util.*;

/**
 * Harness for testing word-level alignments. The code is hard-wired for the
 * alignment source to be English and the alignment target to be French (recall
 * that's the direction for translating INTO English in the noisy channel
 * model).
 * 
 * Your projects will implement several methods of word-to-word alignment.
 */
public class WordAlignmentTester {

	static final String ENGLISH_EXTENSION = "e";
	static final String FRENCH_EXTENSION = "f";

	/**
	 * A holder for a pair of sentences, each a list of strings. Sentences in
	 * the test sets have integer IDs, as well, which are used to retreive the
	 * gold standard alignments for those sentences.
	 */
	public static class SentencePair {
		int sentenceID;
		String sourceFile;
		List<String> englishWords;
		List<String> frenchWords;

		public int getSentenceID() {
			return sentenceID;
		}

		public String getSourceFile() {
			return sourceFile;
		}

		public List<String> getEnglishWords() {
			return englishWords;
		}

		public List<String> getFrenchWords() {
			return frenchWords;
		}

		public String toString() {
			StringBuilder sb = new StringBuilder();
			for (int englishPosition = 0; englishPosition < englishWords.size(); englishPosition++) {
				String englishWord = englishWords.get(englishPosition);
				sb.append(englishPosition);
				sb.append(":");
				sb.append(englishWord);
				sb.append(" ");
			}
			sb.append("\n");
			for (int frenchPosition = 0; frenchPosition < frenchWords.size(); frenchPosition++) {
				String frenchWord = frenchWords.get(frenchPosition);
				sb.append(frenchPosition);
				sb.append(":");
				sb.append(frenchWord);
				sb.append(" ");
			}
			sb.append("\n");
			return sb.toString();
		}

		public SentencePair(int sentenceID, String sourceFile,
				List<String> englishWords, List<String> frenchWords) {
			this.sentenceID = sentenceID;
			this.sourceFile = sourceFile;
			this.englishWords = englishWords;
			this.frenchWords = frenchWords;
		}
	}

	/**
	 * Alignments serve two purposes, both to indicate your system's guessed
	 * alignment, and to hold the gold standard alignments. Alignments map index
	 * pairs to one of three values, unaligned, possibly aligned, and surely
	 * aligned. Your alignment guesses should only contain sure and unaligned
	 * pairs, but the gold alignments contain possible pairs as well.
	 * 
	 * To build an alignment, start with an empty one and use
	 * addAlignment(i,j,true). To display one, use the render method.
	 */
	public static class Alignment {
		Set<Pair<Integer, Integer>> sureAlignments;
		Set<Pair<Integer, Integer>> possibleAlignments;

		public boolean containsSureAlignment(int englishPosition,
				int frenchPosition) {
			return sureAlignments.contains(new Pair<Integer, Integer>(
					englishPosition, frenchPosition));
		}

		public boolean containsPossibleAlignment(int englishPosition,
				int frenchPosition) {
			return possibleAlignments.contains(new Pair<Integer, Integer>(
					englishPosition, frenchPosition));
		}

		public void addAlignment(int englishPosition, int frenchPosition,
				boolean sure) {
			Pair<Integer, Integer> alignment = new Pair<Integer, Integer>(
					englishPosition, frenchPosition);
			if (sure)
				sureAlignments.add(alignment);
			possibleAlignments.add(alignment);
		}

		public Alignment() {
			sureAlignments = new HashSet<Pair<Integer, Integer>>();
			possibleAlignments = new HashSet<Pair<Integer, Integer>>();
		}

		public static String render(Alignment alignment,
				SentencePair sentencePair) {
			return render(alignment, alignment, sentencePair);
		}

		public static String render(Alignment reference, Alignment proposed,
				SentencePair sentencePair) {
			StringBuilder sb = new StringBuilder();
			for (int frenchPosition = 0; frenchPosition < sentencePair
					.getFrenchWords().size(); frenchPosition++) {
				for (int englishPosition = 0; englishPosition < sentencePair
						.getEnglishWords().size(); englishPosition++) {
					boolean sure = reference.containsSureAlignment(
							englishPosition, frenchPosition);
					boolean possible = reference.containsPossibleAlignment(
							englishPosition, frenchPosition);
					char proposedChar = ' ';
					if (proposed.containsSureAlignment(englishPosition,
							frenchPosition))
						proposedChar = '#';
					if (sure) {
						sb.append('[');
						sb.append(proposedChar);
						sb.append(']');
					} else {
						if (possible) {
							sb.append('(');
							sb.append(proposedChar);
							sb.append(')');
						} else {
							sb.append(' ');
							sb.append(proposedChar);
							sb.append(' ');
						}
					}
				}
				sb.append("| ");
				sb.append(sentencePair.getFrenchWords().get(frenchPosition));
				sb.append('\n');
			}
			for (int englishPosition = 0; englishPosition < sentencePair
					.getEnglishWords().size(); englishPosition++) {
				sb.append("---");
			}
			sb.append("'\n");
			boolean printed = true;
			int index = 0;
			while (printed) {
				printed = false;
				StringBuilder lineSB = new StringBuilder();
				for (int englishPosition = 0; englishPosition < sentencePair
						.getEnglishWords().size(); englishPosition++) {
					String englishWord = sentencePair.getEnglishWords().get(
							englishPosition);
					if (englishWord.length() > index) {
						printed = true;
						lineSB.append(' ');
						lineSB.append(englishWord.charAt(index));
						lineSB.append(' ');
					} else {
						lineSB.append("   ");
					}
				}
				index += 1;
				if (printed) {
					sb.append(lineSB);
					sb.append('\n');
				}
			}
			return sb.toString();
		}
	}

	/**
	 * WordAligners have one method: alignSentencePair, which takes a sentence
	 * pair and produces an alignment which specifies an english source for each
	 * french word which is not aligned to "null". Explicit alignment to
	 * position -1 is equivalent to alignment to "null".
	 */
	static interface WordAligner {
		Alignment alignSentencePair(SentencePair sentencePair);
	}

	/**
	 * Simple alignment baseline which maps french positions to english
	 * positions. If the french sentence is longer, all final word map to null.
	 */
	static class BaselineWordAligner implements WordAligner {
		public Alignment alignSentencePair(SentencePair sentencePair) {
			Alignment alignment = new Alignment();
			int numFrenchWords = sentencePair.getFrenchWords().size();
			int numEnglishWords = sentencePair.getEnglishWords().size();
			for (int frenchPosition = 0; frenchPosition < numFrenchWords; frenchPosition++) {
				int englishPosition = frenchPosition;
				if (englishPosition >= numEnglishWords)
					englishPosition = -1;
				alignment.addAlignment(englishPosition, frenchPosition, true);
			}
			return alignment;
		}
	}

	/**
	 * 
	 * Heuristic ALignere that scores alignments by c(e,f)/c(e)*c(f)
	 * 
	 * @author trenous
	 * 
	 */
	static class HeuristicAligner implements WordAligner {

		Counter<String> fWordCounter;

		public Counter<String> getfWordCounter() {
			return fWordCounter;
		}

		public void setfWordCounter(Counter<String> fWordCounter) {
			this.fWordCounter = fWordCounter;
		}

		public Counter<String> geteWordCounter() {
			return eWordCounter;
		}

		public void seteWordCounter(Counter<String> eWordCounter) {
			this.eWordCounter = eWordCounter;
		}

		public Counter<String> getPairWordCounter() {
			return pairWordCounter;
		}

		public void setPairWordCounter(Counter<String> pairWordCounter) {
			this.pairWordCounter = pairWordCounter;
		}

		Counter<String> eWordCounter;
		Counter<String> pairWordCounter;

		@Override
		public Alignment alignSentencePair(SentencePair sentencePair) {
			Alignment alignment = new Alignment();
			List<String> englishWords = sentencePair.getEnglishWords();
			List<String> frenchWords = sentencePair.getFrenchWords();
			for (int j = 0; j < frenchWords.size(); j++) {
				String french = frenchWords.get(j);
				Double fCount = fWordCounter.getCount(french);
				Counter<Integer> alignmentProbs = new Counter<Integer>();
				for (int i = 0; i < englishWords.size(); i++) {
					String english = englishWords.get(i);
					double prob = pairWordCounter.getCount(french + "//"
							+ english);
					prob /= (fCount * eWordCounter.getCount(english));
					alignmentProbs.setCount(i, prob);
				}
				alignmentProbs.setCount(-1,
						(alignmentProbs.totalCount() + 0.00001) / 4);
				alignment.addAlignment(alignmentProbs.argMax(), j, true);
			}
			return alignment;
		}

		public HeuristicAligner(List<SentencePair> training) {
			this.eWordCounter = new Counter<String>();
			this.fWordCounter = new Counter<String>();
			this.pairWordCounter = new Counter<String>();

			for (SentencePair pair : training) {
				Counter<String> frenchCounter = new Counter<String>();
				Counter<String> englishCounter = new Counter<String>();
				englishCounter.incrementCount("*NULL*", 1.0);
				eWordCounter.incrementCount("*NULL*", 1.0);
				for (String french : pair.getFrenchWords()) {
					frenchCounter.incrementCount(french, 1.0);
					fWordCounter.incrementCount(french, 1.0);
				}
				for (String english : pair.getEnglishWords()) {
					englishCounter.incrementCount(english, 1.0);
					eWordCounter.incrementCount(english, 1.0);
				}

				for (Entry<String, Double> french : frenchCounter.getEntrySet()) {
					for (Entry<String, Double> english : englishCounter
							.getEntrySet()) {
						String wordPair = french.getKey() + "//"
								+ english.getKey();
						pairWordCounter.incrementCount(wordPair,
								french.getValue() * english.getValue());
					}
				}

			}
		}
	}

	static class IBM1Aligner implements WordAligner {
		public CounterMap<Integer, Integer> getTranslationProbabilities() {
			return translationProbabilities;
		}

		public Map<String, Integer> getEnglishDict() {
			return englishDict;
		}

		public Map<String, Integer> getFrenchDict() {
			return frenchDict;
		}

		Double nullProb;

		public void setNullProb(Double nullProb) {
			this.nullProb = nullProb;
		}

		CounterMap<Integer, Integer> translationProbabilities;
		Map<String, Integer> englishDict;
		Map<String, Integer> frenchDict;

		public IBM1Aligner(List<SentencePair> train, int maxIter) {
			translationProbabilities = new CounterMap<Integer, Integer>();
			englishDict = new HashMap<String, Integer>();
			frenchDict = new HashMap<String, Integer>();
			int trainSize = train.size();
			Integer[][] englishSentences = new Integer[trainSize][];
			Integer[][] frenchSentences = new Integer[trainSize][];
			int frenchWordIndex = 0;
			int englishWordIndex = 0;
			long initTime = System.currentTimeMillis();
			// Create Mapping from Words to Integers
			for (SentencePair pair : train) {
				for (String french : pair.getFrenchWords()) {
					if (!frenchDict.containsKey(french)) {
						System.out.println(french + " key: " + frenchWordIndex);
						frenchDict.put(french, frenchWordIndex);
						frenchWordIndex++;
					}
				}

				for (String english : pair.getEnglishWords()) {
					if (!englishDict.containsKey(english)) {
						englishDict.put(english, englishWordIndex);
						englishWordIndex++;
					}
				}
			}
			//assert englishDict.containsKey("2");
			// Create Integer Array Representation of Training Data
			int sentenceIndex = 0;
			for (SentencePair pair : train) {
				englishSentences[sentenceIndex] = toIntArray(
						pair.getEnglishWords(), true);
				frenchSentences[sentenceIndex] = toIntArray(
						pair.getFrenchWords(), false);
				sentenceIndex++;
			}
			System.out.println("Encoding Done: "
					+ (System.currentTimeMillis() - initTime) + " MaxLength: "
					+ maxLength);
			// Initialize translation probabilities
			System.out.println("NumEn: " + englishDict.size() + " NumFr: "
					+ frenchDict.size());

			// First EM Round uses uniform initial probability

			Counter<Integer> lambda = new Counter<Integer>();
			for (sentenceIndex = 0; sentenceIndex < trainSize; sentenceIndex++) {
				int length = englishSentences[sentenceIndex].length;
				for (Integer frenchWord : frenchSentences[sentenceIndex]) {
					for (Integer englishWord : englishSentences[sentenceIndex]) {
						translationProbabilities.incrementCount(frenchWord,
								englishWord, 1.0 / length);
						/**
						 * System.out.println(1 / length + " fr " + frenchWord +
						 * " en " + englishWord + " transprob " +
						 * translationProbabilities.getCount(frenchWord,
						 * englishWord));
						 **/
						lambda.incrementCount(englishWord, 1.0 / length);
						assert (translationProbabilities.getCount(frenchWord,
								englishWord) != 0);
					}
				}
			}
			for (Integer frenchWord : translationProbabilities.keySet()) {
				for (Entry<Integer, Double> entry : translationProbabilities
						.getCounter(frenchWord).getEntrySet()) {
					translationProbabilities.setCount(frenchWord,
							entry.getKey(),
							entry.getValue() / lambda.getCount(entry.getKey()));
				}
			}

			boolean hasConverged = false;
			Double delta = 0.01;
			// E and M-Steps
			Counter<Integer> fTotalsCounter;
			CounterMap<Integer, Integer> fCounts;
			int iter = 0;
			while (!hasConverged && iter < maxIter) {
				long time = System.currentTimeMillis();
				// compute expected counts from data
				fCounts = new CounterMap<Integer, Integer>();
				lambda = new Counter<Integer>();
				for (sentenceIndex = 0; sentenceIndex < trainSize; sentenceIndex++) {
					fTotalsCounter = new Counter<Integer>();
					for (Integer frenchWord : frenchSentences[sentenceIndex]) {
						// if (fTotalsCounter.containsKey(frenchWord))
						// continue;
						for (Integer englishWord : englishSentences[sentenceIndex]) {
							fTotalsCounter.incrementCount(frenchWord,
									translationProbabilities.getCount(
											frenchWord, englishWord));
						}
					}
					for (Integer englishWord : englishSentences[sentenceIndex]) {
						for (Integer frenchWord : frenchSentences[sentenceIndex]) {
							lambda.incrementCount(
									englishWord,
									translationProbabilities.getCount(
											frenchWord, englishWord)
											/ fTotalsCounter
													.getCount(frenchWord));
							fCounts.incrementCount(
									frenchWord,
									englishWord,
									translationProbabilities.getCount(
											frenchWord, englishWord)
											/ fTotalsCounter
													.getCount(frenchWord));
						}
					}

				}

				System.out.println("E-Step done: " + iter + " Time taken: "
						+ ((System.currentTimeMillis() - time) / 1000) + "s"
						+ "SIze: " + fCounts.totalSize());
				time = System.currentTimeMillis();
				hasConverged = true;
				int numNotConverged = 0;
				// update translation probabilities to maximize likelihood

				for (Integer frenchWord : fCounts.keySet()) {
					for (Entry<Integer, Double> entry : fCounts.getCounter(
							frenchWord).getEntrySet()) {
						Integer englishWord = entry.getKey();
						Double oldValue = translationProbabilities.getCount(
								frenchWord, englishWord);
						Double newValue = entry.getValue()
								/ lambda.getCount(englishWord);
						fCounts.setCount(frenchWord, englishWord, newValue);
						boolean absChange = Math.abs(oldValue - newValue) < delta;
						numNotConverged += (!absChange) ? 1 : 0;
						hasConverged &= absChange;
					}
				}
				// hasConverged = (iter > 5 ) ?true : false;
				translationProbabilities = fCounts;
				System.out
						.println("M-Step Done: "
								+ iter
								+ " Time Taken: "
								+ ((Double) ((System.currentTimeMillis() - time) / 1000.0))
								+ "s");
				System.out.println("% not Converged: "
						+ (double) (numNotConverged * 100)
						/ translationProbabilities.totalSize());
				iter++;
			}

			System.out.println("Done "
					+ (System.currentTimeMillis() - initTime) / 1000.0);
		}

		private int maxLength = 0;

		private Integer[] toIntArray(List<String> sentence, boolean english) {
			Map<String, Integer> dict = frenchDict;
			if (english) {
				dict = englishDict;
			}
			int length = sentence.size();
			if (length > 256)
				maxLength += 1;
			Integer[] array = new Integer[length];
			for (int i = 0; i < length; i++) {
				if (sentence.get(i) == null) {
					array[i] = -1;
				} else
				
				array[i] = dict.get(sentence.get(i));
			}
			return array;
		}

		@Override
		public Alignment alignSentencePair(SentencePair sentencePair) {
			Integer[] englishSentence = toIntArray(
					sentencePair.getEnglishWords(), true);
			Integer[] frenchSentence = toIntArray(
					sentencePair.getFrenchWords(), false);
			Alignment alignment = new Alignment();
			Counter<Integer> wordTranslationProbs;
			for (int j = 0; j < frenchSentence.length; j++) {
				wordTranslationProbs = new Counter<Integer>();
				for (int i = 0; i < englishSentence.length; i++) {
					wordTranslationProbs.incrementCount(i,
							translationProbabilities.getCount(
									frenchSentence[j], englishSentence[i]));
					/**
					 * System.out.println(sentencePair.getFrenchWords().get(j) +
					 * " " + sentencePair.getEnglishWords().get(i) + " " +
					 * wordTranslationProbs.getCount(i) + " " +
					 * translationProbabilities.getCounter(
					 * frenchSentence[j]).containsKey( englishSentence[i]));
					 **/
				}
				wordTranslationProbs.incrementCount(-1,
						wordTranslationProbs.totalCount()
								* (nullProb / (1 - nullProb)));
				// System.out.println(j + " , " + wordTranslationProbs.argMax()
				// + " , " +
				// wordTranslationProbs.getCount(wordTranslationProbs.argMax()));
				alignment.addAlignment(wordTranslationProbs.argMax(), j, true);
			}

			return alignment;
		}

	}

	static class IBM2Aligner implements WordAligner {
		public CounterMap<Integer, Integer> getTranslationProbabilities() {
			return translationProbabilities;
		}

		public Map<String, Integer> getEnglishDict() {
			return englishDict;
		}

		public Map<String, Integer> getFrenchDict() {
			return frenchDict;
		}

		Double nullProb;

		public void setNullProb(Double nullProb) {
			this.nullProb = nullProb;
		}

		CounterMap<Integer, Integer> alignmentProbabilities;
		CounterMap<Integer, Integer> translationProbabilities;
		Map<String, Integer> englishDict;
		Map<String, Integer> frenchDict;

		public IBM2Aligner(List<SentencePair> train, int maxIter) {
			IBM1Aligner ibm1 = new IBM1Aligner(train, 10);
			translationProbabilities = ibm1.getTranslationProbabilities();
			alignmentProbabilities = new CounterMap<Integer, Integer>();
			englishDict = ibm1.getEnglishDict();
			frenchDict = ibm1.getFrenchDict();
			int trainSize = train.size();
			int[][] englishSentences = new int[trainSize][];
			int[][] frenchSentences = new int[trainSize][];

			long initTime = System.currentTimeMillis();

			// Create Integer Array Representation of Training Data
			int sentenceIndex = 0;
			for (SentencePair pair : train) {
				englishSentences[sentenceIndex] = toIntArray(
						pair.getEnglishWords(), true);
				frenchSentences[sentenceIndex] = toIntArray(
						pair.getFrenchWords(), false);
				sentenceIndex++;
			}

			// First EM Round uses uniform initial probabilities for alignment
			// positions

			Counter<Integer> lambda = new Counter<Integer>();
			Counter<Integer> mu = new Counter<Integer>();
			Counter<Integer> fTotalsCounter;
			CounterMap<Integer, Integer> fCounts = new CounterMap<Integer, Integer>();
			for (sentenceIndex = 0; sentenceIndex < trainSize; sentenceIndex++) {
				fTotalsCounter = new Counter<Integer>();
				int l = englishSentences[sentenceIndex].length;
				int m = frenchSentences[sentenceIndex].length;
				for (int j = 0; j < m; j++) {
					for (int i = 0; i < l; i++) {
						Double c = translationProbabilities.getCount(
								frenchSentences[sentenceIndex][j],
								englishSentences[sentenceIndex][i]) * (1.0/l);
						fTotalsCounter.incrementCount(j, c);
					}
				}
				for (int j = 0; j < frenchSentences[sentenceIndex].length; j++) {
					for (int i = 0; i < englishSentences[sentenceIndex].length; i++) {
						int frenchWord = frenchSentences[sentenceIndex][j];
						int englishWord = englishSentences[sentenceIndex][i];
						int alignmentIndex = createIndex(j, l, m);

						Double c = (translationProbabilities.getCount(
								frenchWord, englishWord) * (1.0 / l))
								/ fTotalsCounter.getCount(j);
						lambda.incrementCount(englishWord, c);
						mu.incrementCount(alignmentIndex, c);
						fCounts.incrementCount(frenchWord, englishWord, c);
						alignmentProbabilities.incrementCount(get(i),
								alignmentIndex, c);
					}
				}
			}
			for (Integer frenchWord : translationProbabilities.keySet()) {
				for (Entry<Integer, Double> entry : translationProbabilities
						.getCounter(frenchWord).getEntrySet()) {
					translationProbabilities.setCount(frenchWord,
							entry.getKey(),
							entry.getValue() / lambda.getCount(entry.getKey()));
				}
			}

			for (Integer i : alignmentProbabilities.keySet()) {
				for (Entry<Integer, Double> entry : alignmentProbabilities
						.getCounter(i).getEntrySet()) {
					alignmentProbabilities.setCount(i, entry.getKey(),
							entry.getValue() / mu.getCount(entry.getKey()));
				}
			}

			boolean hasConverged = false;
			Double delta = 0.05;
			// E and M-Steps
			CounterMap<Integer, Integer> aCounts = new CounterMap<Integer, Integer>();
			int iter = 0;
			while (!hasConverged && iter < maxIter) {
				long time = System.currentTimeMillis();
				// compute expected counts from data
				fCounts = new CounterMap<Integer, Integer>();
				lambda = new Counter<Integer>();
				mu = new Counter<Integer>();
				aCounts = new CounterMap<Integer, Integer>();
				for (sentenceIndex = 0; sentenceIndex < trainSize; sentenceIndex++) {
					int[] englishSentence = englishSentences[sentenceIndex];
					int[] frenchSentence = frenchSentences[sentenceIndex];
					int l = englishSentence.length;
					int m = frenchSentence.length;

					fTotalsCounter = new Counter<Integer>();
					for (int j = 0; j < m; j++) {

						for (int i = 0; i < l; i++) {
							double c = (translationProbabilities.getCount(
									frenchSentence[j], englishSentence[i]) * alignmentProbabilities
									.getCount(get(i), createIndex(j, l, m)));
							fTotalsCounter.incrementCount(j, c);

						}
					}
					for (int i = 0; i < l; i++) {
						for (int j = 0; j < m; j++) {
							int frenchWord = frenchSentences[sentenceIndex][j];
							int englishWord = englishSentences[sentenceIndex][i];
							int alignmentIndex = createIndex(j, l, m);
							Double c = (translationProbabilities.getCount(
									frenchWord, englishWord) * alignmentProbabilities
									.getCount(get(i), alignmentIndex))
									/ fTotalsCounter.getCount(j);
							lambda.incrementCount(englishWord, c);
							mu.incrementCount(alignmentIndex, c);
							fCounts.incrementCount(frenchWord, englishWord, c);
							aCounts.incrementCount(get(i), alignmentIndex, c);
						}
					}

				}

				System.out.println("E-Step done: " + iter + " Time taken: "
						+ ((System.currentTimeMillis() - time) / 1000) + "s");
				time = System.currentTimeMillis();
				hasConverged = true;
				int numNotConverged = 0;
				// update translation probabilities to maximize likelihood

				for (Integer frenchWord : fCounts.keySet()) {
					for (Entry<Integer, Double> entry : fCounts.getCounter(
							frenchWord).getEntrySet()) {
						Integer englishWord = entry.getKey();
						Double oldValue = translationProbabilities.getCount(
								frenchWord, englishWord);
						Double newValue = entry.getValue()
								/ lambda.getCount(englishWord);
						fCounts.setCount(frenchWord, englishWord, newValue);
						boolean absChange = Math.abs(oldValue - newValue) < delta;
						numNotConverged += (!absChange) ? 1 : 0;
						hasConverged &= absChange;
					}
				}

				for (int i : aCounts.keySet()) {
					for (Entry<Integer, Double> entry : aCounts.getCounter(i)
							.getEntrySet()) {
						int alignmentIndex = entry.getKey();
						double newValue = entry.getValue()
								/ mu.getCount(alignmentIndex);
						aCounts.setCount(i, alignmentIndex, newValue);
					}
				}
				// hasConverged = (iter > 5 ) ?true : false;
				alignmentProbabilities = aCounts;
				translationProbabilities = fCounts;
				System.out
						.println("M-Step Done: "
								+ iter
								+ " Time Taken: "
								+ ((Double) ((System.currentTimeMillis() - time) / 1000.0))
								+ "s");
				System.out.println("% not Converged: "
						+ (double) (numNotConverged * 100)
						/ translationProbabilities.totalSize());
				iter++;
			}

			System.out.println("Done "
					+ (System.currentTimeMillis() - initTime) / 1000.0);
		}

		private int createIndex(int j, int l, int m) {
			j = j/3;
			m = m/3;
			l = m/3;
			return (((j << 10) + m) << 10) + l;
		}

		private int[] toIntArray(List<String> sentence, boolean english) {
			if (sentence == null)
				System.out.println("Sentence Null");
			Map<String, Integer> dict = frenchDict;
			if (english) {
				dict = englishDict;
			}
			int length = sentence.size();
			int[] array = new int[length];
			for (int i = 0; i < length; i++) {
				if (dict.get(sentence.get(i)) == null)
					array[i] = -1;
				else
					array[i] = dict.get(sentence.get(i));
			}
			return array;
		}

		@Override
		public Alignment alignSentencePair(SentencePair sentencePair) {
			int[] englishSentence = toIntArray(sentencePair.getEnglishWords(),
					true);
			int[] frenchSentence = toIntArray(sentencePair.getFrenchWords(),
					false);
			int m = frenchSentence.length;
			int l = englishSentence.length;
			Alignment alignment = new Alignment();
			Counter<Integer> wordTranslationProbs;
			for (int j = 0; j < frenchSentence.length; j++) {
			
				wordTranslationProbs = new Counter<Integer>();
				if (frenchSentence[j] != -1)
				for (int i = 0; i < englishSentence.length; i++) {
					
					wordTranslationProbs.setCount(i,
							translationProbabilities.getCount(
									frenchSentence[j], englishSentence[i]) * alignmentProbabilities.getCount(get(i), createIndex(j,l,m)));
					/**
					 * System.out.println(sentencePair.getFrenchWords().get(j) +
					 * " " + sentencePair.getEnglishWords().get(i) + " " +
					 * wordTranslationProbs.getCount(i) + " " +
					 * translationProbabilities.getCounter(
					 * frenchSentence[j]).containsKey( englishSentence[i]));
					 **/
				} else {
					
				}
				
				wordTranslationProbs.incrementCount(-1,
						wordTranslationProbs.totalCount()
								* (nullProb / (1 - nullProb)));
				// System.out.println(j + " , " + wordTranslationProbs.argMax()
				// + " , " +
				// wordTranslationProbs.getCount(wordTranslationProbs.argMax()));
				alignment.addAlignment(wordTranslationProbs.argMax(), j, true);
			}

			return alignment;
		}

		private Integer get(int i) {
			// TODO Auto-generated method stub
			return i/3;
		}

	}

	public static void main(String[] args) throws IOException {
		// Parse command line flags and arguments
		Map<String, String> argMap = CommandLineUtils
				.simpleCommandLineParser(args);

		// Set up default parameters and settings
		String basePath = ".";
		int maxTrainingSentences = 0;
		boolean verbose = false;
		String dataset = "mini";
		String model = "baseline";

		// Update defaults using command line specifications
		if (argMap.containsKey("-path")) {
			basePath = argMap.get("-path");
			System.out.println("Using base path: " + basePath);
		}
		if (argMap.containsKey("-sentences")) {
			maxTrainingSentences = Integer.parseInt(argMap.get("-sentences"));
			System.out.println("Using an additional " + maxTrainingSentences
					+ " training sentences.");
		}
		if (argMap.containsKey("-data")) {
			dataset = argMap.get("-data");
			System.out.println("Running with data: " + dataset);
		} else {
			System.out
					.println("No data set specified.  Use -data [miniTest, validate, test].");
		}
		if (argMap.containsKey("-model")) {
			model = argMap.get("-model");
			System.out.println("Running with model: " + model);
		} else {
			System.out.println("No model specified.  Use -model modelname.");
		}
		if (argMap.containsKey("-verbose")) {
			verbose = true;
		}

		// Read appropriate training and testing sets.
		List<SentencePair> trainingSentencePairs = new ArrayList<SentencePair>();
		if (!dataset.equals("miniTest") && maxTrainingSentences > 0)
			trainingSentencePairs = readSentencePairs(basePath + "/training",
					maxTrainingSentences);
		List<SentencePair> testSentencePairs = new ArrayList<SentencePair>();
		Map<Integer, Alignment> testAlignments = new HashMap<Integer, Alignment>();
		if (dataset.equalsIgnoreCase("validate")) {
			testSentencePairs = readSentencePairs(basePath + "/trial",
					Integer.MAX_VALUE);
			testAlignments = readAlignments(basePath + "/trial/trial.wa");
		} else if (dataset.equalsIgnoreCase("miniTest")) {
			testSentencePairs = readSentencePairs(basePath + "/mini",
					Integer.MAX_VALUE);
			testAlignments = readAlignments(basePath + "/mini/mini.wa");
		} else {
			throw new RuntimeException("Bad data set mode: " + dataset
					+ ", use validate or miniTest.");
		}
		trainingSentencePairs.addAll(testSentencePairs);

		// Build model
		WordAligner wordAligner = null;
		if (model.equalsIgnoreCase("baseline")) {
			wordAligner = new BaselineWordAligner();
		}
		// TODO : build other alignment models
		if (model.equalsIgnoreCase("heuristic")) {
			wordAligner = new HeuristicAligner(trainingSentencePairs);
		}

		if (model.equalsIgnoreCase("ibm1")) {
			wordAligner = new IBM1Aligner(trainingSentencePairs, 15);
			// line search over nullProbability
			Double bestAER = Double.POSITIVE_INFINITY;
			Double bestEpsilon = 0.0;
			for (Double epsilon = 0.01; epsilon < 0.8; epsilon += 0.01) {
				((IBM1Aligner) wordAligner).setNullProb(epsilon);
				Double AER = test(wordAligner, testSentencePairs,
						testAlignments, false);
				boolean newBest = (AER < bestAER);
				bestEpsilon = newBest ? epsilon : bestEpsilon;
				bestAER = newBest ? AER : bestAER;
				System.out.println("");
			}
			((IBM1Aligner) wordAligner).setNullProb(bestEpsilon);
		}

		if (model.equalsIgnoreCase("ibm2")) {
			wordAligner = new IBM2Aligner(trainingSentencePairs, 17);
			// line search over nullProbability
			Double bestAER = Double.POSITIVE_INFINITY;
			Double bestEpsilon = 0.0;
			for (Double epsilon = 0.01; epsilon < 0.8; epsilon += 0.01) {
				((IBM2Aligner) wordAligner).setNullProb(epsilon);
				Double AER = test(wordAligner, testSentencePairs,
						testAlignments, false);
				boolean newBest = (AER < bestAER);
				bestEpsilon = newBest ? epsilon : bestEpsilon;
				bestAER = newBest ? AER : bestAER;
				System.out.println("");
			}
			((IBM2Aligner) wordAligner).setNullProb(bestEpsilon);
		}

		// Test model
		test(wordAligner, testSentencePairs, testAlignments, verbose);

		// Generate file for submission
		testSentencePairs = readSentencePairs(basePath + "/test",
				Integer.MAX_VALUE);
		predict(wordAligner, testSentencePairs, basePath + "/" + model + ".out");
	}

	private static Double test(WordAligner wordAligner,
			List<SentencePair> testSentencePairs,
			Map<Integer, Alignment> testAlignments, boolean verbose) {
		int proposedSureCount = 0;
		int proposedPossibleCount = 0;
		int sureCount = 0;
		int proposedCount = 0;
		for (SentencePair sentencePair : testSentencePairs) {
			assert (wordAligner != null);
			Alignment proposedAlignment = wordAligner
					.alignSentencePair(sentencePair);
			Alignment referenceAlignment = testAlignments.get(sentencePair
					.getSentenceID());
			if (referenceAlignment == null)
				throw new RuntimeException(
						"No reference alignment found for sentenceID "
								+ sentencePair.getSentenceID());
			if (verbose)
				System.out.println("Alignment:\n"
						+ Alignment.render(referenceAlignment,
								proposedAlignment, sentencePair));
			for (int frenchPosition = 0; frenchPosition < sentencePair
					.getFrenchWords().size(); frenchPosition++) {
				for (int englishPosition = 0; englishPosition < sentencePair
						.getEnglishWords().size(); englishPosition++) {
					boolean proposed = proposedAlignment.containsSureAlignment(
							englishPosition, frenchPosition);
					boolean sure = referenceAlignment.containsSureAlignment(
							englishPosition, frenchPosition);
					boolean possible = referenceAlignment
							.containsPossibleAlignment(englishPosition,
									frenchPosition);
					if (proposed && sure)
						proposedSureCount += 1;
					if (proposed && possible)
						proposedPossibleCount += 1;
					if (proposed)
						proposedCount += 1;
					if (sure)
						sureCount += 1;
				}
			}
		}
		System.out.println("Precision: " + proposedPossibleCount
				/ (double) proposedCount);
		System.out.println("Recall: " + proposedSureCount / (double) sureCount);
		Double AER = (1.0 - (proposedSureCount + proposedPossibleCount)
				/ (double) (sureCount + proposedCount));
		System.out.println("AER: " + AER);
		return AER;
	}

	private static void predict(WordAligner wordAligner,
			List<SentencePair> testSentencePairs, String path)
			throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(path));
		for (SentencePair sentencePair : testSentencePairs) {
			Alignment proposedAlignment = wordAligner
					.alignSentencePair(sentencePair);
			for (int frenchPosition = 0; frenchPosition < sentencePair
					.getFrenchWords().size(); frenchPosition++) {
				for (int englishPosition = 0; englishPosition < sentencePair
						.getEnglishWords().size(); englishPosition++) {
					if (proposedAlignment.containsSureAlignment(
							englishPosition, frenchPosition)) {
						writer.write(frenchPosition + "-" + englishPosition
								+ " ");
					}
				}
			}
			writer.write("\n");
		}
		writer.close();
	}

	// BELOW HERE IS IO CODE

	private static Map<Integer, Alignment> readAlignments(String fileName) {
		Map<Integer, Alignment> alignments = new HashMap<Integer, Alignment>();
		try {
			BufferedReader in = new BufferedReader(new FileReader(fileName));
			while (in.ready()) {
				String line = in.readLine();
				String[] words = line.split("\\s+");
				if (words.length != 4)
					throw new RuntimeException("Bad alignment file " + fileName
							+ ", bad line was " + line);
				Integer sentenceID = Integer.parseInt(words[0]);
				Integer englishPosition = Integer.parseInt(words[1]) - 1;
				Integer frenchPosition = Integer.parseInt(words[2]) - 1;
				String type = words[3];
				Alignment alignment = alignments.get(sentenceID);
				if (alignment == null) {
					alignment = new Alignment();
					alignments.put(sentenceID, alignment);
				}
				alignment.addAlignment(englishPosition, frenchPosition,
						type.equals("S"));
			}
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		return alignments;
	}

	private static List<SentencePair> readSentencePairs(String path,
			int maxSentencePairs) {
		List<SentencePair> sentencePairs = new ArrayList<SentencePair>();
		List<String> baseFileNames = getBaseFileNames(path);
		for (String baseFileName : baseFileNames) {
			if (sentencePairs.size() >= maxSentencePairs)
				continue;
			sentencePairs.addAll(readSentencePairs(baseFileName));
		}
		return sentencePairs;
	}

	private static List<SentencePair> readSentencePairs(String baseFileName) {
		List<SentencePair> sentencePairs = new ArrayList<SentencePair>();
		String englishFileName = baseFileName + "." + ENGLISH_EXTENSION;
		String frenchFileName = baseFileName + "." + FRENCH_EXTENSION;
		try {
			BufferedReader englishIn = new BufferedReader(new FileReader(
					englishFileName));
			BufferedReader frenchIn = new BufferedReader(new InputStreamReader(

			new FileInputStream(frenchFileName), StandardCharsets.ISO_8859_1));
			while (englishIn.ready() && frenchIn.ready()) {
				String englishLine = englishIn.readLine();
				String frenchLine = frenchIn.readLine();
				Pair<Integer, List<String>> englishSentenceAndID = readSentence(englishLine);
				Pair<Integer, List<String>> frenchSentenceAndID = readSentence(frenchLine);
				if (!englishSentenceAndID.getFirst().equals(
						frenchSentenceAndID.getFirst()))
					throw new RuntimeException("Sentence ID confusion in file "
							+ baseFileName + ", lines were:\n\t" + englishLine
							+ "\n\t" + frenchLine);
				sentencePairs.add(new SentencePair(englishSentenceAndID
						.getFirst(), baseFileName, englishSentenceAndID
						.getSecond(), frenchSentenceAndID.getSecond()));
			}
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		return sentencePairs;
	}

	private static Pair<Integer, List<String>> readSentence(String line) {
		int id = -1;
		List<String> words = new ArrayList<String>();
		String[] tokens = line.split("\\s+");
		for (int i = 0; i < tokens.length; i++) {
			String token = tokens[i];
			if (token.equals("<s"))
				continue;
			if (token.equals("</s>"))
				continue;
			if (token.startsWith("snum=")) {
				String idString = token.substring(5, token.length() - 1);
				id = Integer.parseInt(idString);
				continue;
			}
			words.add(token.intern());
		}
		return new Pair<Integer, List<String>>(id, words);
	}

	private static List<String> getBaseFileNames(String path) {
		List<File> englishFiles = IOUtils.getFilesUnder(path, new FileFilter() {
			public boolean accept(File pathname) {
				if (pathname.isDirectory())
					return true;
				String name = pathname.getName();
				return name.endsWith(ENGLISH_EXTENSION);
			}
		});
		List<String> baseFileNames = new ArrayList<String>();
		for (File englishFile : englishFiles) {
			String baseFileName = chop(englishFile.getAbsolutePath(), "."
					+ ENGLISH_EXTENSION);
			baseFileNames.add(baseFileName);
		}
		return baseFileNames;
	}

	private static String chop(String name, String extension) {
		if (!name.endsWith(extension))
			return name;
		return name.substring(0, name.length() - extension.length());
	}

}
