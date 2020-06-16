package ginrummy.GinRummyEAAIorigin.GinRummyEAAI;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

public class PlayerModel {

	// Input value for curent state. Probability matrix of 52
	public ArrayList<Double> X;
	
	// The weights of X
	public ArrayList<Double> weights;
	
	// Eligibility_trace of one game
	public ArrayList<Double> eligibility_trace;

	// Randome seed
	int seed;
	
	// Learning rate
	double lr;
	
	// Number of training iterations
	double n_iter;

	// End game?
	public boolean end;
	
	
	/**
	 * Initialize the model with saved file
	 * @param filename (String): The path the saved model.
	 */
	@SuppressWarnings("unchecked")
	public PlayerModel (String filename) {
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename));
			this.weights = (ArrayList<Double>) in.readObject();
			in.close();
		} catch (FileNotFoundException e) {
			System.out.println("File not found: " + filename);
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Initialize a new model
	 */
	public PlayerModel () {
		// Have to call __init__ if model is new.
	}
	
	/**
	 * Initialize data for a new model
	 * @param weights (ArrayList<Double>): Probability vector of length 52.
	 * @param seed (int): Random seed.
	 * @param lr (double): Leanring rate.
	 * @param n_iter (int): Number of iterations.
	 */
	public void __init__(ArrayList<Double> weights, int seed, double lr, int n_iter, boolean end) {
		this.X = new ArrayList<Double>();
		this.weights = weights;
		this.seed = seed;
		this.lr = lr;
		this.n_iter = n_iter;
		this.end = end;
		this.eligibility_trace = new ArrayList<Double>();

		// Debugging
		assert this.eligibility_trace.size() == X.size();
	}
	
	/**
	 * Perform a sigmoid activation function
	 * @param x (double) param
	 * @return (double) the result of the sigmoid function.
	 */
	public static double sigmoid(double x) {
		return 1 / (1 + Math.exp(x));
	}
	
	/**
	 * Perform a dot product of 2 vectors with the same dimension.
	 * @param X (ArrayList<Double>): Param
	 * @param weights (ArrayList<Double>): Param
	 * @return (double) The dot product of the two vectors.
	 */
	public static double dot(ArrayList<Double> X, ArrayList<Double> weights) {
		assert X.size() == weights.size();
		
		double sum = 0;
		for (int i = 0; i < X.size(); i++) {
			sum += X.get(i) * weights.get(i);
		}
		return sum;
	}
	
	/**
	 * Normal feed-forward ANN computation for one node.
	 * @param inputs
	 * @param weights
	 * @return
	 */
	public static double compute_value(ArrayList<Double> inputs, ArrayList<Double> weights) {
		return sigmoid(dot(inputs, weights));
	}
	
	/**
	 * Perform custom back propagation according to https://www.aaai.org/Papers/ICML/2003/ICML03-050.pdf, update new weights.
	 * @param weights
	 * @param lr
	 * @param gamma
	 * @param reward_next_state
	 * @param value_current_state
	 * @param value_next_state
	 * @param eligibility_trace
	 * @return (ArrayList<Double>) new weights.
	 */
	public static ArrayList<Double> back_propagation(ArrayList<Double> weights, 
			double lr, 
			double gamma, 
			double reward_next_state,
			double value_current_state, 
			double value_next_state, 
			ArrayList<Double> eligibility_trace) {
		
		assert weights.size() == eligibility_trace.size();
		@SuppressWarnings("unchecked")
		ArrayList<Double> new_weights = (ArrayList<Double>) weights.clone();
		for (int i = 0; i < weights.size(); i++) {
			new_weights.set(i, new_weights.get(i) + lr * (reward_next_state + gamma * value_next_state - value_current_state) * eligibility_trace.get(i));
		}
		return new_weights;
	}
	
	/**
	 * compute_eligibility_trace based on paper https://www.aaai.org/Papers/ICML/2003/ICML03-050.pdf
	 * @param gamma
	 * @param delta
	 * @param prev_eligibility_trace
	 * @param value_current_state
	 * @return
	 */
	public static ArrayList<Double> compute_eligibility_trace(double gamma, double delta, ArrayList<Double> prev_eligibility_trace, double value_current_state) {
		@SuppressWarnings("unchecked")
		ArrayList<Double> eligibility_trace = (ArrayList<Double>) prev_eligibility_trace.clone();
		for (int i = 0; i < eligibility_trace.size(); i++) {
			eligibility_trace.set(i, eligibility_trace.get(i) * delta * gamma + value_current_state);
		}
		return eligibility_trace;
	}
	
	/**
	 * evaluate a full step of the network and save new weights.
	 * @param value_next_state (double): The value of the next state =))
	 * @param prev_eligibility_trace
	 * @param reward_next_state
	 */
	public void evaluate_step(double value_next_state, double reward_next_state, boolean end) {
		// Compute value
		double value_current_state = compute_value(this.X, this.weights);
		double gamma = 1.0;
		double delta = 1.0;

		// debugging
		assert this.eligibility_trace.size() == X.size();

		this.eligibility_trace = compute_eligibility_trace(gamma, delta, this.eligibility_trace, value_current_state);
		this.weights = back_propagation(this.weights, this.lr, gamma, reward_next_state, value_current_state, value_next_state, this.eligibility_trace);
		
		// If end game then reset eligibility trace.
		if (end) this.eligibility_trace = new ArrayList<>();
	}
	
	/**
	 * Save model, in this case, save weights.
	 * @param filename
	 */
	public void save(String filename) {
		try {
			ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename));
			out.writeObject(this.weights);
			out.close();
		} catch (FileNotFoundException e) {
			System.out.println("File not found: " + filename);
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	// Test
//	public static void main(String[] args) {
//		
//
//	}

}
