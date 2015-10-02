package nlp.assignments;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import nlp.assignments.MaximumEntropyClassifier.EncodedDatum;
import nlp.assignments.MaximumEntropyClassifier.Encoding;
import nlp.assignments.MaximumEntropyClassifier.IndexLinearizer;
import nlp.assignments.MaximumEntropyClassifier.ObjectiveFunction;
import nlp.assignments.ProperNameTester.ProperNameFeatureExtractor;
import nlp.classify.BasicFeatureVector;
import nlp.classify.BasicLabeledFeatureVector;
import nlp.classify.FeatureExtractor;
import nlp.classify.FeatureVector;
import nlp.classify.LabeledFeatureVector;
import nlp.classify.LabeledInstance;
import nlp.classify.ProbabilisticClassifier;
import nlp.classify.ProbabilisticClassifierFactory;
import nlp.math.DifferentiableFunction;
import nlp.math.DoubleArrays;
import nlp.math.GradientMinimizer;
import nlp.math.LBFGSMinimizer;
import nlp.util.Counter;
import nlp.util.Indexer;

public class Perceptron<I, F, L> implements ProbabilisticClassifier<I, L> {

	// fields
	private Encoding<F, L> encoding;
	private IndexLinearizer indexLinearizer;
	private FeatureExtractor<I, F> featureExtractor;
	double[] w;

	public Perceptron(List<LabeledInstance<I, L>> trainingData, int iterations,
			FeatureExtractor<I, F> featureExtractor) {
		this.featureExtractor = featureExtractor;
		this.encoding = buildEncoding(trainingData);
		this.indexLinearizer = buildIndexLinearizer(encoding);
		this.w = buildInitialWeights(indexLinearizer);
		EncodedDatum[] data = encodeData(trainingData, encoding);
		ArrayList<Integer> indices = new ArrayList<Integer>(data.length);
		for (int i = 0; i < data.length; i++) {
			indices.add(i);
		}
		Collections.shuffle(indices);

		for (int j = 0; j < iterations; j++) {

			for (int i : indices) {

				EncodedDatum datum = data[i];
				int label = datum.getLabelIndex();
				int predictedLabel = encoding.getLabelIndex(getProbabilities(
						datum).argMax());
				if (label != predictedLabel) {
					for (int f = 0; f < datum.getNumActiveFeatures(); f++) {

						int featureIndex = datum.getFeatureIndex(f);
						int predictedIndex = indexLinearizer.getLinearIndex(
								featureIndex, predictedLabel);
						int actualIndex = indexLinearizer.getLinearIndex(
								featureIndex, label); 
						w[predictedIndex] -=  datum.getFeatureCount(f);
						w[actualIndex] +=   datum.getFeatureCount(f);
					}
					
				}

			}
		}
		

	}

	
	

	




	

	private double[] buildInitialWeights(IndexLinearizer indexLinearizer) {
		return DoubleArrays.constantArray(0.0,
				indexLinearizer.getNumLinearIndexes());
	}

	private IndexLinearizer buildIndexLinearizer(Encoding<F, L> encoding) {
		return new IndexLinearizer(encoding.getNumFeatures(),
				encoding.getNumLabels());
	}

	private Encoding<F, L> buildEncoding(List<LabeledInstance<I, L>> data) {
		Indexer<F> featureIndexer = new Indexer<F>();
		Indexer<L> labelIndexer = new Indexer<L>();
		for (LabeledInstance<I, L> labeledInstance : data) {
			L label = labeledInstance.getLabel();
			Counter<F> features = featureExtractor
					.extractFeatures(labeledInstance.getInput());
			LabeledFeatureVector<F, L> labeledDatum = new BasicLabeledFeatureVector<F, L>(
					label, features);
			labelIndexer.add(labeledDatum.getLabel());
			for (F feature : labeledDatum.getFeatures().keySet()) {
				featureIndexer.add(feature);
			}
		}
		return new Encoding<F, L>(featureIndexer, labelIndexer);
	}

	private EncodedDatum[] encodeData(List<LabeledInstance<I, L>> data,
			Encoding<F, L> encoding) {

		EncodedDatum[] encodedData = new EncodedDatum[data.size()];
		for (int i = 0; i < data.size(); i++) {
			LabeledInstance<I, L> labeledInstance = data.get(i);
			L label = labeledInstance.getLabel();
			Counter<F> features = featureExtractor
					.extractFeatures(labeledInstance.getInput());
			LabeledFeatureVector<F, L> labeledFeatureVector = new BasicLabeledFeatureVector<F, L>(
					label, features);
			encodedData[i] = EncodedDatum.encodeLabeledDatum(
					labeledFeatureVector, encoding);
			
		}
		
		
		return encodedData;
	}



	@Override
	public L getLabel(I instance) {
		return getProbabilities(instance).argMax();
	}

	@Override
	public Counter<L> getProbabilities(I instance) {
		FeatureVector<F> featureVector = new BasicFeatureVector<F>(
				featureExtractor.extractFeatures(instance));
		EncodedDatum encodedDatum = EncodedDatum.encodeDatum(featureVector,
				encoding);
		return getProbabilities(encodedDatum);

	}

	private Counter<L> getProbabilities(EncodedDatum encodedDatum) {
		Counter<L> probabilityCounter = new Counter<L>();
		for (int label = 0 ; label < encoding.getNumLabels() ; label++) {
			double score = 0.0;
			for (int f = 0; f < encodedDatum.getNumActiveFeatures() ; f++) {
				int linearIndex = indexLinearizer.getLinearIndex(encodedDatum.getFeatureIndex(f), label);
				score += w[linearIndex] * encodedDatum.getFeatureCount(f);
				
			}
			probabilityCounter.setCount(encoding.getLabel(label), score);
		}

			//probabilityCounter.normalize();
		return probabilityCounter;
	}
}
