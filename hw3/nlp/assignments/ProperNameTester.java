package nlp.assignments;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.stat.correlation.*;
import org.apache.commons.csv.*;
import nlp.classify.*;
import nlp.math.DoubleArrays;
import nlp.util.CommandLineUtils;
import nlp.util.Counter;
import nlp.util.Pair;

/**
 * This is the main harness for assignment 2. To run this harness, use
 * <p/>
 * java nlp.assignments.ProperNameTester -path ASSIGNMENT_DATA_PATH -model
 * MODEL_DESCRIPTOR_STRING
 * <p/>
 * First verify that the data can be read on your system using the baseline
 * model. Second, find the point in the main method (near the bottom) where a
 * MostFrequentLabelClassifier is constructed. You will be writing new
 * implementations of the ProbabilisticClassifer interface and constructing them
 * there.
 */
public class ProperNameTester {

	public static class ProperNameFeatureExtractor implements
			FeatureExtractor<String, String> {

		/**
		 * This method takes the list of characters representing the proper name
		 * description, and produces a list of features which represent that
		 * description. The basic implementation is that only character-unigram
		 * features are extracted. An easy extension would be to also throw
		 * character bigrams into the feature list, but better features are also
		 * possible.
		 */
		public Counter<String> extractFeatures(String name) {
			char[] characters = name.toCharArray();
			Counter<String> features = new Counter<String>();

			String[] words = name.split(" ");

			
			if (words.length>1)
			for (String word : words) {
				features.incrementCount("WORD: " + word, 1.0);
			}
			//features.incrementCount(name,1.0);

			features.setCount("BIAS", 1.0);

			// Contains ' ' + 'X.' + ' ', X in {A,...,Z}
			for (int i = 1; i < words.length - 1; i++) {
				String currentWord = words[i];
				if (currentWord.matches("([A-Z])([\\.])")) {
					features.incrementCount("CONTAINS INITIAL", 1.0);
					break;
				}
			}

			// features.incrementCount("LENGTH-", name.length());

			int digits = 0;
			for (char c : characters) {
				if (Character.isDigit(c))
					digits++;
			}

			// Contains Number
			if (digits > 0) ;
				features.incrementCount("CONTAINS-NUMBER", 1.0);

			// Uni-, Bi-, Tri- and Quadgrams
			char start = 0;
			char end = 0;
			char lastLastLastChar = start;
			char lastLastChar = start;
			char lastChar = start; // 0 is start symbol
			char currentChar = start;
			for (char c : name.toCharArray()) {
				lastLastLastChar = lastLastChar;
				lastLastChar = lastChar;
				lastChar = currentChar;
				currentChar = c;
				features.setCount("UNIGRAM-" + currentChar, 1.0);
				features.incrementCount("BIGRAM-" + lastChar + currentChar, 1.0);
				features.setCount(
						"UNI-LO" + Character.toLowerCase(currentChar), 1.0);
				features.incrementCount(
						"TRIGRAM-" + Character.toLowerCase(lastLastChar)
								+ Character.toLowerCase(lastChar)
								+ Character.toLowerCase(currentChar), 1.0);
				features.incrementCount(
						"QUADGRAM-" + Character.toLowerCase(lastLastLastChar)
								+ Character.toLowerCase(lastLastChar)
								+ Character.toLowerCase(lastChar)
								+ Character.toLowerCase(currentChar), 1.0);
			}
			features.incrementCount("BIGRAM-" + currentChar + end, 1.0);
			features.incrementCount(
					"TRIGRAM-" + Character.toLowerCase(lastChar)
							+ Character.toLowerCase(currentChar) + end, 1.0);
			features.incrementCount(
					"QUADGRAM-" + Character.toLowerCase(lastLastChar)
							+ Character.toLowerCase(lastChar)
							+ Character.toLowerCase(currentChar) + end, 1.0);
			
			String endQuadgram = name.substring(Math.max(0, name.length() - 4),
					name.length());
			features.incrementCount(
					"LASTQUADGRAM-" + endQuadgram.toLowerCase(), 1.0); 
			
			  //Binary Length Features
			int bucket = 17;
			//features.incrementCount("LENGTH-" + (int)name.length()/bucket, 1.0);
			
			//features.incrementCount("LENGTH", name.length());
			
			return features;
		}
	}

	private static List<LabeledInstance<String, String>> loadData(
			String fileName) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(fileName));
		List<LabeledInstance<String, String>> labeledInstances = new ArrayList<LabeledInstance<String, String>>();
		while (reader.ready()) {
			String line = reader.readLine();
			String[] parts = line.split("\t");
			String label = parts[0];
			String name = parts[1];
			LabeledInstance<String, String> labeledInstance = new LabeledInstance<String, String>(
					label, name);
			labeledInstances.add(labeledInstance);
		}
		reader.close();
		return labeledInstances;
	}

	private static Pair<Double, Double> testClassifier(
			ProbabilisticClassifier<String, String> classifier,
			List<LabeledInstance<String, String>> testData, boolean verbose) {
		double numCorrect = 0.0;
		double numTotal = 0.0;
		double[] correct = new double[testData.size()];
		double[] confidences = new double[testData.size()];
		Double[][] array = new Double[10][2]; 
		for(int i = 0; i< 10 ; i++) 
			for (int j = 0 ; j<2 ; j++)
				array[i][j] = 0.0;
		int i = 0;
		for (LabeledInstance<String, String> testDatum : testData) {
			String name = testDatum.getInput();
			String label = classifier.getLabel(name);
			double confidence = classifier.getProbabilities(name).getCount(
					label);
			confidences[i] = confidence;
			if (label.equals(testDatum.getLabel())) {
				correct[i] = 1.0;
				numCorrect += 1.0;
			} else {
				correct[i] = 0.0;
				if (verbose) {
					// display an error
					System.err.println("Error: " + name + " guess=" + label
							+ " gold=" + testDatum.getLabel() + " confidence="
							+ confidence);
				}
			}
			array[(int)Math.min(9,Math.floor(confidence*10))][0] += correct[i];
			array[(int)Math.min(9,Math.floor(confidence*10))][1] += 1.0;
			
			numTotal += 1.0;
			i++;
		}
		
		for (int i1 = 0; i1<10 ; i1++) {
			System.out.println("Bracket [" + ((double)i1)/10 + "," + ((double)(i1+1))/10 + "]: " + array[i1][0]/array[i1][1]);
		}
		double accuracy = numCorrect / numTotal;
		System.out.println("Accuracy: " + accuracy);
		// Return The Accuracy and the correlation of accuracy and confidence
		return new Pair<Double, Double>(accuracy,
				(new PearsonsCorrelation()).correlation(correct, confidences));
	}

	public static void main(String[] args) throws IOException {
		// Parse command line flags and arguments
		Map<String, String> argMap = CommandLineUtils
				.simpleCommandLineParser(args);

		// Set up default parameters and settings
		String basePath = ".";
		String model = "baseline";
		boolean verbose = false;

		// The path to the assignment data

		if (argMap.containsKey("-path")) {
			basePath = argMap.get("-path");
		}
		System.out.println("Using base path: " + basePath);

		// A string descriptor of the model to use
		if (argMap.containsKey("-model")) {
			model = argMap.get("-model");
		}
		System.out.println("Using model: " + model);

		// A string descriptor of the testData to use

		System.out.println("Testing on: "
				+ (argMap.containsKey("-test") ? argMap.get("-test") : "test"));

		// Whether or not to print the individual speech errors.
		if (argMap.containsKey("-verbose")) {
			verbose = true;
		}

		// Load training, validation, and test data
		List<LabeledInstance<String, String>> trainingData = loadData(basePath
				+ "/pnp-train.txt");
		List<LabeledInstance<String, String>> validationData = loadData(basePath
				+ "/pnp-validate.txt");
		List<LabeledInstance<String, String>> testData = loadData(basePath
				+ "/pnp-test.txt");

		long trainingSpeed = 0;
		// Determine which data should be used for testing
		List<LabeledInstance<String, String>> testing = validationData;
		if (argMap.containsKey("-test")) {
			String test = argMap.get("-test");
			if (test.equalsIgnoreCase("test"))
				testing = testData;
			else if (test.equalsIgnoreCase("train"))
				testing = trainingData;
		}

		// Array of classifiers for Line search
		ArrayList<ProbabilisticClassifier<String, String>> classifiers = new ArrayList<ProbabilisticClassifier<String, String>>();
		// Number of sigma values to search
		int num = 1;

		// Array to save line search results
		double[][] results = new double[num][3];

		// Learn a classifier
		ProbabilisticClassifier<String, String> classifier = null;
		if (model.equalsIgnoreCase("baseline")) {
			classifier = new MostFrequentLabelClassifier.Factory<String, String>()
					.trainClassifier(trainingData);
		} else if (model.equalsIgnoreCase("perceptron")) {
			long tBefore = System.currentTimeMillis();
			classifiers.add(new Perceptron<String, String, String>(
					trainingData, 20, new ProperNameFeatureExtractor()));
			trainingSpeed = (System.currentTimeMillis() -tBefore);
		} else if (model.equalsIgnoreCase("maxent")) {

			// Line search over sigma
			for (int i = 0; i < num; i++) {
				System.out.println();
				System.out.println("ITERATION: " + i);
				System.out.println();
				ProbabilisticClassifierFactory<String, String> factory = new MaximumEntropyClassifier.Factory<String, String, String>(
						sigma(i), 1000, new ProperNameFeatureExtractor());
				long tBefore = System.currentTimeMillis();
				classifiers.add(factory.trainClassifier(trainingData));
				trainingSpeed = (System.currentTimeMillis() -tBefore);
			}
		} else {
			throw new RuntimeException("Unknown model descriptor: " + model);
		}

		// Test all classifiers
		for (int i = 0; i < num; i++) {
			Pair<Double, Double> result = testClassifier(classifiers.get(i),
					testing, verbose);
			results[i][0] = sigma(i);
			results[i][1] = result.getFirst();
			results[i][2] = result.getSecond();
		}

		// Write to CSV File for plotting
		FileWriter writer = new FileWriter("data.csv");
		CSVPrinter printer = new CSVPrinter(writer, CSVFormat.DEFAULT);
		printer.print("sigma");
		printer.print("acc");
		printer.print("corr");
		printer.println();
		writeArrayToFile(results, printer, num, 3);
		writer.close();
		printer.close();
		// Output the best Sigma we have seen
		int bestI = DoubleArrays.argMax(MatrixUtils.createRealMatrix(results)
				.getColumn(1));
		Double bestSigma = results[bestI][0];
		Double bestAcc = results[bestI][1];
		System.out.println("Best Accuracy: " + bestAcc + ", With sigma="
				+ bestSigma + " Speed:" + trainingSpeed);

	}

	// Helper method to write data to CSV File
	public static void writeArrayToFile(double[][] results, CSVPrinter printer,
			int rows, int cols) throws IOException {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++)
				printer.print(results[i][j]);
			printer.println();
		}

	}

	// Method for Line search over Sigma
	private static double sigma(int i) {

		return 1;
	}
}
