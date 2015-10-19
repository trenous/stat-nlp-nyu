package nlp.assignments;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.math3.linear.MatrixUtils;

import nlp.classify.FeatureExtractor;
import nlp.classify.LabeledInstance;
import nlp.classify.ProbabilisticClassifier;
import nlp.classify.ProbabilisticClassifierFactory;
import nlp.math.DoubleArrays;
import nlp.util.*;

/**
 * Harness for POS Tagger project.
 */
public class POSTaggerTester {

	static final String START_WORD = "<S>";
	static final String STOP_WORD = "</S>";
	static final String START_TAG = "<S>";
	static final String STOP_TAG = "</S>";

	/**
	 * Tagged sentences are a bundling of a list of words and a list of their
	 * tags.
	 */
	static class TaggedSentence {
		List<String> words;
		List<String> tags;

		public int size() {
			return words.size();
		}

		public List<String> getWords() {
			return words;
		}

		public List<String> getTags() {
			return tags;
		}

		public String toString() {
			StringBuilder sb = new StringBuilder();
			for (int position = 0; position < words.size(); position++) {
				String word = words.get(position);
				String tag = tags.get(position);
				sb.append(word);
				sb.append("_");
				sb.append(tag);
			}
			return sb.toString();
		}

		public boolean equals(Object o) {
			if (this == o)
				return true;
			if (!(o instanceof TaggedSentence))
				return false;

			final TaggedSentence taggedSentence = (TaggedSentence) o;

			if (tags != null ? !tags.equals(taggedSentence.tags)
					: taggedSentence.tags != null)
				return false;
			if (words != null ? !words.equals(taggedSentence.words)
					: taggedSentence.words != null)
				return false;

			return true;
		}

		public int hashCode() {
			int result;
			result = (words != null ? words.hashCode() : 0);
			result = 29 * result + (tags != null ? tags.hashCode() : 0);
			return result;
		}

		public TaggedSentence(List<String> words, List<String> tags) {
			this.words = words;
			this.tags = tags;
		}
	}

	/**
	 * States are pairs of tags along with a position index, representing the
	 * two tags preceding that position. So, the START state, which can be
	 * gotten by State.getStartState() is [START, START, 0]. To build an
	 * arbitrary state, for example [DT, NN, 2], use the static factory method
	 * State.buildState("DT", "NN", 2). There isnt' a single final state, since
	 * sentences lengths vary, so State.getEndState(i) takes a parameter for the
	 * length of the sentence.
	 */
	static class State {

		private static transient Interner<State> stateInterner = new Interner<State>(
				new Interner.CanonicalFactory<State>() {
					public State build(State state) {
						return new State(state);
					}
				});

		private static transient State tempState = new State();

		public static State getStartState() {
			return buildState(START_TAG, START_TAG, 0);
		}

		public static State getStopState(int position) {
			return buildState(STOP_TAG, STOP_TAG, position);
		}

		public static State buildState(String previousPreviousTag,
				String previousTag, int position) {
			tempState.setState(previousPreviousTag, previousTag, position);
			return stateInterner.intern(tempState);
		}

		public static List<String> toTagList(List<State> states) {
			List<String> tags = new ArrayList<String>();
			if (states.size() > 0) {
				tags.add(states.get(0).getPreviousPreviousTag());
				for (State state : states) {
					tags.add(state.getPreviousTag());
				}
			}
			return tags;
		}

		public int getPosition() {
			return position;
		}

		public String getPreviousTag() {
			return previousTag;
		}

		public String getPreviousPreviousTag() {
			return previousPreviousTag;
		}

		public State getNextState(String tag) {
			return State.buildState(getPreviousTag(), tag, getPosition() + 1);
		}

		public State getPreviousState(String tag) {
			return State.buildState(tag, getPreviousPreviousTag(),
					getPosition() - 1);
		}

		public boolean equals(Object o) {
			if (this == o)
				return true;
			if (!(o instanceof State))
				return false;

			final State state = (State) o;

			if (position != state.position)
				return false;
			if (previousPreviousTag != null ? !previousPreviousTag
					.equals(state.previousPreviousTag)
					: state.previousPreviousTag != null)
				return false;
			if (previousTag != null ? !previousTag.equals(state.previousTag)
					: state.previousTag != null)
				return false;

			return true;
		}

		public int hashCode() {
			int result;
			result = position;
			result = 29 * result
					+ (previousTag != null ? previousTag.hashCode() : 0);
			result = 29
					* result
					+ (previousPreviousTag != null ? previousPreviousTag
							.hashCode() : 0);
			return result;
		}

		public String toString() {
			return "[" + getPreviousPreviousTag() + ", " + getPreviousTag()
					+ ", " + getPosition() + "]";
		}

		int position;
		String previousTag;
		String previousPreviousTag;

		private void setState(String previousPreviousTag, String previousTag,
				int position) {
			this.previousPreviousTag = previousPreviousTag;
			this.previousTag = previousTag;
			this.position = position;
		}

		private State() {
		}

		private State(State state) {
			setState(state.getPreviousPreviousTag(), state.getPreviousTag(),
					state.getPosition());
		}
	}

	/**
	 * A Trellis is a graph with a start state an an end state, along with
	 * successor and predecessor functions.
	 */
	static class Trellis<S> {
		S startState;
		S endState;
		CounterMap<S, S> forwardTransitions;
		CounterMap<S, S> backwardTransitions;
		private int length;
		private HashMap<String, Integer> tagMap;
		private HashMap<Integer, String> indexMap;

		/**
		 * Get the unique start state for this trellis.
		 */
		public S getStartState() {
			return startState;
		}

		public void setStartState(S startState) {
			this.startState = startState;
		}

		/**
		 * Get the unique end state for this trellis.
		 */
		public S getEndState() {
			return endState;
		}

		public void setStopState(S endState) {
			this.endState = endState;
		}

		/**
		 * For a given state, returns a counter over what states can be next in
		 * the markov process, along with the cost of that transition. Caution:
		 * a state not in the counter is illegal, and should be considered to
		 * have cost Double.NEGATIVE_INFINITY, but Counters score items they
		 * don't contain as 0.
		 */
		public Counter<S> getForwardTransitions(S state) {
			return forwardTransitions.getCounter(state);

		}

		/**
		 * For a given state, returns a counter over what states can precede it
		 * in the markov process, along with the cost of that transition.
		 */
		public Counter<S> getBackwardTransitions(S state) {
			return backwardTransitions.getCounter(state);
		}

		public void setTransitionCount(S start, S end, double count) {
			forwardTransitions.setCount(start, end, count);
			backwardTransitions.setCount(end, start, count);
		}

		public Trellis() {
			forwardTransitions = new CounterMap<S, S>();
			backwardTransitions = new CounterMap<S, S>();
			tagMap = new HashMap<String, Integer>();
			indexMap = new HashMap<Integer, String>();

		}

		public int getLength() {
			return length;
		}

		public void setLength(int i) {
			this.length = i;

		}

		public void addTag(String tag) {
			if (!this.tagMap.containsKey(tag)) {
				tagMap.put(tag, tagMap.size());
				indexMap.put(tagMap.get(tag), tag);
			}
		}

		public int getNumTags() {
			return this.tagMap.keySet().size();
		}

		public int getTagIndex(String tag) {
			return this.tagMap.get(tag);
		}

		public String getTag(int index) {
			return this.indexMap.get(index);
		}
	}

	/**
	 * A TrellisDecoder takes a Trellis and returns a path through that trellis
	 * in which the first item is trellis.getStartState(), the last is
	 * trellis.getEndState(), and each pair of states is conntected in the
	 * trellis.
	 */
	static interface TrellisDecoder<S> {
		List<S> getBestPath(Trellis<S> trellis);
	}

	static class ViterbiDecoder<S> implements TrellisDecoder<S> {
		public List<S> getBestPath(Trellis<S> trellis) {
			S currentState = trellis.getStartState();
			HashMap<S, S> prevStates = new HashMap<S, S>();
			Counter<S> currentStates = new Counter<S>();
			currentStates.setCount(currentState, 0);
			int i = 0;
			while (!currentStates.containsKey(trellis.getEndState())) {
				if (i == 1000) 
					System.out.println(i);
				i++;
				Counter<S> nextStates = new Counter<S>();
				for (S state : currentStates.keySet()) {
					Counter<S> possibleNextStates = trellis
							.getForwardTransitions(state);
					double stateScore = currentStates.getCount(state);
					for (S nextState : possibleNextStates.keySet()) {
						double nextScore = stateScore
								+ possibleNextStates.getCount(nextState);
						if (nextStates.containsKey(nextState)) {
							if (nextStates.getCount(nextState) <= nextScore) {
								nextStates.setCount(nextState, nextScore);
								prevStates.put(nextState, state);
							}
						} else {
							nextStates.setCount(nextState, nextScore);
							prevStates.put(nextState, state);
						}
					}

				}
				currentStates = nextStates;
			}
			assert (prevStates.keySet().contains(trellis.getEndState()));
			assert (prevStates.containsValue(trellis.getStartState()));
			List<S> states = new ArrayList<S>();
			states.add(0, trellis.getEndState());

			while (states.get(0) != trellis.getStartState()) {
				states.add(0, prevStates.get(states.get(0)));
			}
			assert states.get(0) == trellis.getStartState();
			assert states.get(states.size() - 1) == trellis.getEndState();
			if (states.size() != trellis.getLength())
				;
			// System.out.println(states.size() - trellis.getLength()); else
			// System.out.println("OK");
			return states;
		}
	}

	static class POSTagger {

		LocalTrigramScorer localTrigramScorer;
		TrellisDecoder<State> trellisDecoder;

		// chop up the training instances into local contexts and pass them on
		// to the local scorer.
		public void train(List<TaggedSentence> taggedSentences) {
			localTrigramScorer
					.train(extractLabeledLocalTrigramContexts(taggedSentences));
		}

		// chop up the validation instances into local contexts and pass them on
		// to the local scorer.
		public void validate(List<TaggedSentence> taggedSentences) {
			localTrigramScorer.validate(
					extractLabeledLocalTrigramContexts(taggedSentences),
					taggedSentences);
		}

		private List<LabeledLocalTrigramContext> extractLabeledLocalTrigramContexts(
				List<TaggedSentence> taggedSentences) {
			List<LabeledLocalTrigramContext> localTrigramContexts = new ArrayList<LabeledLocalTrigramContext>();
			for (TaggedSentence taggedSentence : taggedSentences) {
				localTrigramContexts
						.addAll(extractLabeledLocalTrigramContexts(taggedSentence));
			}
			return localTrigramContexts;
		}

		private List<LabeledLocalTrigramContext> extractLabeledLocalTrigramContexts(
				TaggedSentence taggedSentence) {
			List<LabeledLocalTrigramContext> labeledLocalTrigramContexts = new ArrayList<LabeledLocalTrigramContext>();
			List<String> words = new BoundedList<String>(
					taggedSentence.getWords(), START_WORD, STOP_WORD);
			List<String> tags = new BoundedList<String>(
					taggedSentence.getTags(), START_TAG, STOP_TAG);
			for (int position = 0; position <= taggedSentence.size() + 1; position++) {
				labeledLocalTrigramContexts.add(new LabeledLocalTrigramContext(
						words, position, tags.get(position - 2), tags
								.get(position - 1), tags.get(position)));
			}
			return labeledLocalTrigramContexts;
		}

		/**
		 * Builds a Trellis over a sentence, by starting at the state State, and
		 * advancing through all legal extensions of each state already in the
		 * trellis. You should not have to modify this code (or even read it,
		 * really).
		 */
		private Trellis<State> buildTrellis(List<String> sentence) {
			Trellis<State> trellis = new Trellis<State>();
			trellis.setStartState(State.getStartState());
			State stopState = State.getStopState(sentence.size() + 2);
			trellis.setStopState(stopState);
			Set<State> states = Collections.singleton(State.getStartState());
			for (int position = 0; position <= sentence.size() + 1; position++) {
				Set<State> nextStates = new HashSet<State>();
				for (State state : states) {
					if (state.equals(stopState))
						continue;
					LocalTrigramContext localTrigramContext = new LocalTrigramContext(
							sentence, position, state.getPreviousPreviousTag(),
							state.getPreviousTag());
					Counter<String> tagScores = localTrigramScorer
							.getLogScoreCounter(localTrigramContext);
					for (String tag : tagScores.keySet()) {
						trellis.addTag(tag);
						double score = tagScores.getCount(tag);
						State nextState = state.getNextState(tag);
						trellis.setTransitionCount(state, nextState, score);
						nextStates.add(nextState);
					}
				}
				// System.out.println("States: "+nextStates);
				states = nextStates;
			}
			return trellis;
		}

		// to tag a sentence: build its trellis and find a path through that
		// trellis
		public List<String> tag(List<String> sentence) {
			Trellis<State> trellis = buildTrellis(sentence);
			List<State> states = trellisDecoder.getBestPath(trellis);
			assert states.size() == sentence.size() + 3;
			List<String> tags = State.toTagList(states);
			tags = stripBoundaryTags(tags);
			assert tags.size() == sentence.size();
			return tags;
		}

		/**
		 * Scores a tagging for a sentence. Note that a tag sequence not
		 * accepted by the markov process should receive a log score of
		 * Double.NEGATIVE_INFINITY.
		 */
		public double scoreTagging(TaggedSentence taggedSentence) {
			double logScore = 0.0;
			List<LabeledLocalTrigramContext> labeledLocalTrigramContexts = extractLabeledLocalTrigramContexts(taggedSentence);
			for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
				Counter<String> logScoreCounter = localTrigramScorer
						.getLogScoreCounter(labeledLocalTrigramContext);
				String currentTag = labeledLocalTrigramContext.getCurrentTag();
				if (logScoreCounter.containsKey(currentTag)) {
					logScore += logScoreCounter.getCount(currentTag);
				} else {
					logScore += Double.NEGATIVE_INFINITY;
				}
			}
			return logScore;
		}

		private List<String> stripBoundaryTags(List<String> tags) {
			return tags.subList(2, tags.size() - 2);
		}

		public POSTagger(LocalTrigramScorer localTrigramScorer,
				TrellisDecoder<State> trellisDecoder) {
			this.localTrigramScorer = localTrigramScorer;
			this.trellisDecoder = trellisDecoder;
		}
	}

	/**
	 * A LocalTrigramContext is a position in a sentence, along with the
	 * previous two tags -- basically a FeatureVector.
	 */
	static class LocalTrigramContext {
		List<String> words;
		int position;
		String previousTag;
		String previousPreviousTag;

		public List<String> getWords() {
			return words;
		}

		public String getCurrentWord() {
			return words.get(position);
		}

		public int getPosition() {
			return position;
		}

		public String getPreviousTag() {
			return previousTag;
		}

		public String getPreviousPreviousTag() {
			return previousPreviousTag;
		}

		public String toString() {
			return "[" + getPreviousPreviousTag() + ", " + getPreviousTag()
					+ ", " + getCurrentWord() + "]";
		}

		public LocalTrigramContext(List<String> words, int position,
				String previousPreviousTag, String previousTag) {
			this.words = words;
			this.position = position;
			this.previousTag = previousTag;
			this.previousPreviousTag = previousPreviousTag;
		}
	}

	/**
	 * A LabeledLocalTrigramContext is a context plus the correct tag for that
	 * position -- basically a LabeledFeatureVector
	 */
	static class LabeledLocalTrigramContext extends LocalTrigramContext {
		String currentTag;

		public String getCurrentTag() {
			return currentTag;
		}

		public String toString() {
			return "[" + getPreviousPreviousTag() + ", " + getPreviousTag()
					+ ", " + getCurrentWord() + "_" + getCurrentTag() + "]";
		}

		public LabeledLocalTrigramContext(List<String> words, int position,
				String previousPreviousTag, String previousTag,
				String currentTag) {
			super(words, position, previousPreviousTag, previousTag);
			this.currentTag = currentTag;
		}
	}

	/**
	 * LocalTrigramScorers assign scores to tags occuring in specific
	 * LocalTrigramContexts.
	 */
	static interface LocalTrigramScorer {
		/**
		 * The Counter returned should contain log probabilities, meaning if all
		 * values are exponentiated and summed, they should sum to one. For
		 * efficiency, the Counter can contain only the tags which occur in the
		 * given context with non-zero model probability.
		 */
		Counter<String> getLogScoreCounter(
				LocalTrigramContext localTrigramContext);

		void validate(List<LabeledLocalTrigramContext> localTrigramContexts,
				List<TaggedSentence> sentences);

		void train(List<LabeledLocalTrigramContext> localTrigramContexts);

		void validate(List<LabeledLocalTrigramContext> localTrigramContexts);
	}

	static class TrigramScorer implements LocalTrigramScorer {
		CounterMap<String, String> emissionCounter;
		Counter<Integer> lambdaCounter;
		Counter<String> unigramCounter;
		CounterMap<String, String> bigramCounter;
		CounterMap<String, String> trigramCounter;
		Counter<String> wordCounter;
		CounterMap<String, String> allowedTags;
		Set<String> unknownAllowedTags;
		List<LabeledInstance<String, String>> maxEntTrain;
		ProbabilisticClassifier<String, String> unknownWordEmissions;

		public TrigramScorer() {
			super();
			emissionCounter = new CounterMap<String, String>();
			lambdaCounter = new Counter<Integer>();
			unigramCounter = new Counter<String>();
			bigramCounter = new CounterMap<String, String>();
			trigramCounter = new CounterMap<String, String>();
			wordCounter = new Counter<String>();
			allowedTags = new CounterMap<String, String>();
			maxEntTrain = new ArrayList<LabeledInstance<String, String>>();
			unknownAllowedTags = new HashSet<String>();
		}

		@Override
		public Counter<String> getLogScoreCounter(
				LocalTrigramContext localTrigramContext) {
			int position = localTrigramContext.getPosition();
			String word = localTrigramContext.getWords().get(position);
			String prevTag = localTrigramContext.getPreviousTag();
			String prevPrevTag = localTrigramContext.getPreviousPreviousTag();
			Counter<String> logScoreCounter = new Counter<String>();
			/**if (!wordCounter.containsKey(word) && wordCounter.containsKey(Character.toUpperCase(word.charAt(0)) + word.substring(1)))
				word = Character.toUpperCase(word.charAt(0)) + word.substring(1); 
			else if (!wordCounter.containsKey(word) && wordCounter.containsKey(Character.toLowerCase(word.charAt(0)) + word.substring(1)))
				word = Character.toLowerCase(word.charAt(0)) + word.substring(1);		**/
			boolean isKnown = wordCounter.containsKey(word);
			//if (!wordCounter.containsKey(word))
			//	word = "UNK";
			//System.out.println(isKnown + " " + word + " " + prevTag + " " + prevPrevTag);
			Set<String> legalTags = isKnown ? allowedTags.getCounter(word).keySet() : unknownAllowedTags;
			for (String tag : legalTags) {
				double emProb;
				if (isKnown)
				emProb = emissionCounter.getCount(tag, word);
				else emProb = unknownWordEmissions.getProbabilities(word).getCount(tag) / unigramCounter.getCount(tag);
				double p1 = unigramCounter.getCount(tag);
				double p2 = bigramCounter.getCount(prevTag, tag);
				double p3 = trigramCounter.getCount(
						makeBigramString(prevPrevTag, prevTag), tag);
				double transProb = lambdaCounter.getCount(1) * p1
						+ lambdaCounter.getCount(2) * p2
						+ lambdaCounter.getCount(3) * p3;
				double logScore = Math.log(emProb * transProb);
				if (emProb * transProb != 0)
					logScoreCounter.incrementCount(tag, logScore);
			}

			return logScoreCounter;

		}

		private String makeBigramString(String prevTag, String tag) {
			return prevTag + " " + tag;
		}

		@Override
		public void train(List<LabeledLocalTrigramContext> localTrigramContexts) {
			// collect word-tag counts
			Counter<String> trigramOccurences = new Counter<String>();
			Counter<String> bigramOccurences = new Counter<String>();
			for (LabeledLocalTrigramContext context : localTrigramContexts) {
				
				String word = context.getCurrentWord();
				String tag = context.getCurrentTag();
				String prevTag = context.getPreviousTag();
				String prevPrevTag = context.getPreviousPreviousTag();
				if (wordCounter.getCount(word) == 0) {
					emissionCounter.incrementCount(tag, "UNK", 1.0);
					allowedTags.setCount("UNK", tag, 1);
				}
				wordCounter.incrementCount(word, 1.0);
				unigramCounter.incrementCount(tag, 1.0);
				bigramCounter.incrementCount(prevTag, tag, 1.0);
				trigramCounter.incrementCount(
						makeBigramString(prevPrevTag, prevTag), tag, 1.0);
				trigramOccurences.incrementCount(
						makeTrigramString(prevPrevTag, prevTag, tag), 1.0);
				bigramOccurences.incrementCount(makeBigramString(prevTag, tag),
						1.0);
				emissionCounter.incrementCount(tag, word, 1.0);
				allowedTags.setCount(word, tag, 1.0);

			}
			for (LabeledLocalTrigramContext context : localTrigramContexts) {
				String word = context.getCurrentWord();
				String tag = context.getCurrentTag();
				if (wordCounter.getCount(word) <= 10) {
				maxEntTrain.add(new LabeledInstance<String, String>(tag, word));
				unknownAllowedTags.add(tag);
				wordCounter.incrementCount("UNK", 1.0);
				}
			}
			
			System.out.println("TrainDone");
			ProbabilisticClassifierFactory<String, String> factory = new MaximumEntropyClassifier.Factory<String, String, String>(
					1, 1000, new POSFeatureExtractor());
			this.unknownWordEmissions = factory.trainClassifier(maxEntTrain);

			// Interpolation as in TnT tagger

			int N = localTrigramContexts.size();
			for (String trigram : trigramOccurences.keySet()) {
				String[] tags = trigram.split(" ");
				String prevPrevTag = tags[0];
				String prevTag = tags[1];
				String tag = tags[2];
				double f3 = (trigramOccurences.getCount(trigram) - 1)
						/ (bigramOccurences.getCount(makeBigramString(
								prevPrevTag, prevTag)) - 1);
				double f2 = (bigramOccurences.getCount(makeBigramString(
						prevTag, tag)) - 1)
						/ (unigramCounter.getCount(tag) - 1);
				double f1 = (unigramCounter.getCount(tag) - 1) / (N - 1);
				int index = (f1 >= f2 && f1 >= f3) ? 1
						: ((f2 > f1 && f2 >= f3) ? 2 : 3);
				lambdaCounter.incrementCount(index,
						trigramOccurences.getCount(trigram));
			}
			lambdaCounter.normalize();

			unigramCounter.normalize();
			wordCounter.normalize();
			emissionCounter = Counters.conditionalNormalize(emissionCounter);
			bigramCounter = Counters.conditionalNormalize(bigramCounter);
			trigramCounter = Counters.conditionalNormalize(trigramCounter);

		}

		@Override
		public void validate(
				List<LabeledLocalTrigramContext> localTrigramContexts,
				List<TaggedSentence> sentences) {
			// Line search over sigma
			TrellisDecoder<State> trellisDecoder = new ViterbiDecoder<State>();
			POSTagger posTagger = new POSTagger(this, trellisDecoder);
			List<LabeledInstance<String, String>> validationData = new ArrayList<LabeledInstance<String, String>>();
			for (LabeledLocalTrigramContext context : localTrigramContexts)
				validationData.add(new LabeledInstance<String, String>(context
						.getCurrentWord(), context.getCurrentTag()));
			int num = 1;
			ArrayList<ProbabilisticClassifier<String, String>> classifiers = new ArrayList<ProbabilisticClassifier<String, String>>();
			// Number of sigma values to search
			double[][] results = new double[num][2];
			for (int i = 0; i < num; i++) {
				System.out.println();
				System.out.println("ITERATION: " + i);
				System.out.println();
				ProbabilisticClassifierFactory<String, String> factory = new MaximumEntropyClassifier.Factory<String, String, String>(
						sigma(i), 1000, new POSFeatureExtractor());
				classifiers.add(factory.trainClassifier(maxEntTrain));

			}
			for (int i = 0; i < num; i++) {
				this.unknownWordEmissions = classifiers.get(i);
				Double result = evaluateTagger(posTagger, sentences,
						new HashSet<String>(), false);
				results[i][0] = sigma(i);
				results[i][1] = result;
			}
			int bestI = DoubleArrays.argMax(MatrixUtils.createRealMatrix(
					results).getColumn(1));
			Double bestSigma = results[bestI][0];
			System.out.println("Best Sigma: " + bestSigma);
			this.unknownWordEmissions = classifiers.get(bestI);
		}

		// Helper method to write data to CSV File
		public static void writeArrayToFile(double[][] results,
				CSVPrinter printer, int rows, int cols) throws IOException {
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++)
					printer.print(results[i][j]);
				printer.println();
			}

		}

		private double sigma(int i) {
			// TODO Auto-generated method stub
			return 0.1 + 0.1 * i;
		}

		private String makeTrigramString(String previousPreviousTag,
				String previousTag, String currentTag) {
			return previousPreviousTag + " " + previousTag + " " + currentTag;
		}

		@Override
		public void validate(
				List<LabeledLocalTrigramContext> localTrigramContexts) {
			// TODO Auto-generated method stub

		}

	}

	/**
	 * The MostFrequentTagScorer gives each test word the tag it was seen with
	 * most often in training (or the tag with the most seen word types if the
	 * test word is unseen in training. This scorer actually does a little more
	 * than its name claims -- if constructed with restrictTrigrams = true, it
	 * will forbid illegal tag trigrams, otherwise it makes no use of tag
	 * history information whatsoever.
	 */
	static class MostFrequentTagScorer implements LocalTrigramScorer {

		boolean restrictTrigrams; // if true, assign log score of
									// Double.NEGATIVE_INFINITY to illegal tag
									// trigrams.

		CounterMap<String, String> wordsToTags = new CounterMap<String, String>();
		Counter<String> unknownWordTags = new Counter<String>();
		Set<String> seenTagTrigrams = new HashSet<String>();

		public int getHistorySize() {
			return 2;
		}

		public Counter<String> getLogScoreCounter(
				LocalTrigramContext localTrigramContext) {
			int position = localTrigramContext.getPosition();
			String word = localTrigramContext.getWords().get(position);
			Counter<String> tagCounter = unknownWordTags;
			if (wordsToTags.keySet().contains(word)) {
				tagCounter = wordsToTags.getCounter(word);
			}
			Set<String> allowedFollowingTags = allowedFollowingTags(
					tagCounter.keySet(),
					localTrigramContext.getPreviousPreviousTag(),
					localTrigramContext.getPreviousTag());
			Counter<String> logScoreCounter = new Counter<String>();
			for (String tag : tagCounter.keySet()) {
				double logScore = Math.log(tagCounter.getCount(tag));
				if (!restrictTrigrams || allowedFollowingTags.isEmpty()
						|| allowedFollowingTags.contains(tag))
					logScoreCounter.setCount(tag, logScore);
			}
			return logScoreCounter;
		}

		private Set<String> allowedFollowingTags(Set<String> tags,
				String previousPreviousTag, String previousTag) {
			Set<String> allowedTags = new HashSet<String>();
			for (String tag : tags) {
				String trigramString = makeTrigramString(previousPreviousTag,
						previousTag, tag);
				if (seenTagTrigrams.contains((trigramString))) {
					allowedTags.add(tag);
				}
			}
			return allowedTags;
		}

		private String makeTrigramString(String previousPreviousTag,
				String previousTag, String currentTag) {
			return previousPreviousTag + " " + previousTag + " " + currentTag;
		}

		public void train(
				List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) {
			// collect word-tag counts
			for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
				String word = labeledLocalTrigramContext.getCurrentWord();
				String tag = labeledLocalTrigramContext.getCurrentTag();
				if (!wordsToTags.keySet().contains(word)) {
					// word is currently unknown, so tally its tag in the
					// unknown tag counter
					unknownWordTags.incrementCount(tag, 1.0);
				}
				wordsToTags.incrementCount(word, tag, 1.0);
				seenTagTrigrams.add(makeTrigramString(
						labeledLocalTrigramContext.getPreviousPreviousTag(),
						labeledLocalTrigramContext.getPreviousTag(),
						labeledLocalTrigramContext.getCurrentTag()));
			}
			wordsToTags = Counters.conditionalNormalize(wordsToTags);
			unknownWordTags = Counters.normalize(unknownWordTags);
		}

		public void validate(
				List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) {
			// no tuning for this dummy model!
		}

		public MostFrequentTagScorer(boolean restrictTrigrams) {
			this.restrictTrigrams = restrictTrigrams;
		}

		@Override
		public void validate(
				List<LabeledLocalTrigramContext> localTrigramContexts,
				List<TaggedSentence> sentences) {
			// TODO Auto-generated method stub

		}
	}
	
	
	public static class POSFeatureExtractor implements
			FeatureExtractor<String, String> {
		public Counter<String> extractFeatures(String name) {
			char[] characters = name.toCharArray();
			Counter<String> features = new Counter<String>();
			features.incrementCount("BIAS", 1.0);
			boolean isUpperCase = Character.isUpperCase(characters[0]);

			int i = 1;
			while (i < 9 && name.length() >= i) {
				if (isUpperCase)
					features.incrementCount(
							"UPPERCASE-" + name.substring(name.length() - i),
							1.0);
				else
					features.incrementCount(
							"LOWERCASE-" + name.substring(name.length() - i),
							1.0);
				i++;
			}

			return features;
		}
	}
	

	private static List<TaggedSentence> readTaggedSentences(String path,
			boolean hasTags) throws Exception {
		List<TaggedSentence> taggedSentences = new ArrayList<TaggedSentence>();
		BufferedReader reader = new BufferedReader(new FileReader(path));
		String line = "";
		List<String> words = new LinkedList<String>();
		List<String> tags = new LinkedList<String>();
		while ((line = reader.readLine()) != null) {
			if (line.equals("")) {
				taggedSentences.add(new TaggedSentence(new BoundedList<String>(
						words, START_WORD, STOP_WORD), new BoundedList<String>(
						tags, START_WORD, STOP_WORD)));
				words = new LinkedList<String>();
				tags = new LinkedList<String>();
			} else {
				String[] fields = line.split("\\s+");
				words.add(fields[0]);
				tags.add(hasTags ? fields[1] : "");
			}
		}
		reader.close();
		System.out.println("Read " + taggedSentences.size() + " sentences.");
		return taggedSentences;
	}

	private static void labelTestSet(POSTagger posTagger,
			List<TaggedSentence> testSentences, String path) throws Exception {
		BufferedWriter writer = new BufferedWriter(new FileWriter(path));
		for (TaggedSentence sentence : testSentences) {
			List<String> words = sentence.getWords();
			List<String> guessedTags = posTagger.tag(words);
			for (int i = 0; i < words.size(); i++) {
				writer.write(words.get(i) + "\t" + guessedTags.get(i) + "\n");
			}
			writer.write("\n");
		}
		writer.close();
	}

	private static Double evaluateTagger(POSTagger posTagger,
			List<TaggedSentence> taggedSentences,
			Set<String> trainingVocabulary, boolean verbose) {
		double numTags = 0.0;
		double numTagsCorrect = 0.0;
		double numUnknownWords = 0.0;
		double numUnknownWordsCorrect = 0.0;
		int numDecodingInversions = 0;
		for (TaggedSentence taggedSentence : taggedSentences) {
			List<String> words = taggedSentence.getWords();
			List<String> goldTags = taggedSentence.getTags();
			List<String> guessedTags = posTagger.tag(words);
			for (int position = 0; position < words.size() - 1; position++) {
				String word = words.get(position);
				String goldTag = goldTags.get(position);
				String guessedTag = guessedTags.get(position);
				if (guessedTag.equals(goldTag))
					numTagsCorrect += 1.0;
				numTags += 1.0;
				if (!trainingVocabulary.contains(word)) {
					if (guessedTag.equals(goldTag))
						numUnknownWordsCorrect += 1.0;
					numUnknownWords += 1.0;
				}
			}
			double scoreOfGoldTagging = posTagger.scoreTagging(taggedSentence);
			double scoreOfGuessedTagging = posTagger
					.scoreTagging(new TaggedSentence(words, guessedTags));
			if (scoreOfGoldTagging > scoreOfGuessedTagging) {
				numDecodingInversions++;
				if (verbose)
					System.out
							.println("WARNING: Decoder suboptimality detected.  Gold tagging has higher score than guessed tagging.");
			}
			if (verbose)
				System.out.println(alignedTaggings(words, goldTags,
						guessedTags, true) + "\n");
		}
		System.out.println("Tag Accuracy: " + (numTagsCorrect / numTags)
				+ " (Unknown Accuracy: "
				+ (numUnknownWordsCorrect / numUnknownWords)
				+ "Known Accuracy: "
				+ ((numTagsCorrect - numUnknownWordsCorrect)/(numTags-numUnknownWords))
				+ " Unknown Words Percentage: "
				+ (numUnknownWords/numTags)
				+ ")  Decoder Suboptimalities Detected: "
				+ numDecodingInversions);
		return (numTagsCorrect / numTags);
	}

	// pretty-print a pair of taggings for a sentence, possibly suppressing the
	// tags which correctly match
	private static String alignedTaggings(List<String> words,
			List<String> goldTags, List<String> guessedTags,
			boolean suppressCorrectTags) {
		StringBuilder goldSB = new StringBuilder("Gold Tags: ");
		StringBuilder guessedSB = new StringBuilder("Guessed Tags: ");
		StringBuilder wordSB = new StringBuilder("Words: ");
		for (int position = 0; position < words.size(); position++) {
			equalizeLengths(wordSB, goldSB, guessedSB);
			String word = words.get(position);
			String gold = goldTags.get(position);
			String guessed = guessedTags.get(position);
			wordSB.append(word);
			if (position < words.size() - 1)
				wordSB.append(' ');
			boolean correct = (gold.equals(guessed));
			if (correct && suppressCorrectTags)
				continue;
			guessedSB.append(guessed);
			goldSB.append(gold);
		}
		return goldSB + "\n" + guessedSB + "\n" + wordSB;
	}

	private static void equalizeLengths(StringBuilder sb1, StringBuilder sb2,
			StringBuilder sb3) {
		int maxLength = sb1.length();
		maxLength = Math.max(maxLength, sb2.length());
		maxLength = Math.max(maxLength, sb3.length());
		ensureLength(sb1, maxLength);
		ensureLength(sb2, maxLength);
		ensureLength(sb3, maxLength);
	}

	private static void ensureLength(StringBuilder sb, int length) {
		while (sb.length() < length) {
			sb.append(' ');
		}
	}

	private static Set<String> extractVocabulary(
			List<TaggedSentence> taggedSentences) {
		Set<String> vocabulary = new HashSet<String>();
		for (TaggedSentence taggedSentence : taggedSentences) {
			List<String> words = taggedSentence.getWords();
			vocabulary.addAll(words);
		}
		return vocabulary;
	}

	static class GreedyDecoder<S> implements TrellisDecoder<S> {
		public List<S> getBestPath(Trellis<S> trellis) {
			List<S> states = new ArrayList<S>();
			S currentState = trellis.getStartState();
			states.add(currentState);
			while (!currentState.equals(trellis.getEndState())) {
				Counter<S> transitions = trellis
						.getForwardTransitions(currentState);
				S nextState = transitions.argMax();
				states.add(nextState);
				currentState = nextState;
			}
			return states;
		}
	}
	public static void main(String[] args) throws Exception {
		// Parse command line flags and arguments
		Map<String, String> argMap = CommandLineUtils
				.simpleCommandLineParser(args);

		// Set up default parameters and settings
		String basePath = ".";
		boolean verbose = false;

		// Update defaults using command line specifications

		// The path to the assignment data
		if (argMap.containsKey("-path")) {
			basePath = argMap.get("-path");
		}
		System.out.println("Using base path: " + basePath);

		// Whether or not to print the individual errors.
		if (argMap.containsKey("-verbose")) {
			verbose = true;
		}

		// Read in data
		System.out.print("Loading training sentences...");
		List<TaggedSentence> trainTaggedSentences = readTaggedSentences(
				basePath + "/en-wsj-train.pos", true);
		Set<String> trainingVocabulary = extractVocabulary(trainTaggedSentences);
		System.out.println("done.");
		System.out.print("Loading in-domain dev sentences...");
		List<TaggedSentence> devInTaggedSentences = readTaggedSentences(
				basePath + "/en-wsj-dev.pos", true);
		System.out.println("done.");
		System.out.print("Loading out-of-domain dev sentences...");
		List<TaggedSentence> devOutTaggedSentences = readTaggedSentences(
				basePath + "/en-web-weblogs-dev.pos", true);
		System.out.println("done.");
		System.out.print("Loading out-of-domain blind test sentences...");
		List<TaggedSentence> testSentences = readTaggedSentences(basePath
				+ "/en-web-test.blind", false);
		System.out.println("done.");

		// Construct tagger components
		// TODO : improve on the MostFrequentTagScorer
		LocalTrigramScorer localTrigramScorer = new TrigramScorer();
		// TODO : improve on the GreedyDecoder
		TrellisDecoder<State> trellisDecoder = new ViterbiDecoder<State>();

		// Train tagger
		POSTagger posTagger = new POSTagger(localTrigramScorer, trellisDecoder);
		posTagger.train(trainTaggedSentences);

		// Optionally tune hyperparameters on dev data
		// posTagger.validate(devInTaggedSentences);

		// Test tagger
		System.out.println("Evaluating on in-domain data:.");
		evaluateTagger(posTagger, devInTaggedSentences, trainingVocabulary,
				verbose);
		System.out.println("Evaluating on out-of-domain data:.");
		evaluateTagger(posTagger, devOutTaggedSentences, trainingVocabulary,
				verbose);
		labelTestSet(posTagger, testSentences, basePath + "/en-web-test.tagged");
	}
}
