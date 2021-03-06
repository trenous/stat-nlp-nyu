package nlp.math;

public interface GradientLineSearcher {
	public double[] minimize(DifferentiableFunction function, double[] initial,
			double[] direction);
}
