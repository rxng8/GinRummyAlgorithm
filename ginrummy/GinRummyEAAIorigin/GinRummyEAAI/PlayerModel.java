import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

public class PlayerModel {

	public ArrayList<Double> X;
	public ArrayList<Double> weights;
	int seed;
	double lr;
	double n_iter;
	
	
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
	
	public PlayerModel () {}
	
	public void __init__(ArrayList<Double> weights, int seed, double lr, int n_iter) {
		this.X = new ArrayList<Double>();
		this.weights = weights;
		this.seed = seed;
		this.lr = lr;
		this.n_iter = n_iter;
	}
	
	// Wikipedia
	public static double sigmoid(double x) {
		return 1 / (1 + Math.exp(x));
	}
	
	// Normal Math
	public static double dot(ArrayList<Double> X, ArrayList<Double> weights) {
		assert X.size() == weights.size();
		
		double sum = 0;
		for (int i = 0; i < X.size(); i++) {
			sum += X.get(i) * weights.get(i);
		}
		return sum;
	}
	
	// Normal feed-forward ANN computation
	public static double compute_value(ArrayList<Double> inputs, ArrayList<Double> weights) {
		return sigmoid(dot(inputs, weights));
	}
	
	// https://www.aaai.org/Papers/ICML/2003/ICML03-050.pdf
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
	
	// https://www.aaai.org/Papers/ICML/2003/ICML03-050.pdf
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
	 */
	public void evaluate_step(double value_next_state, ArrayList<Double> prev_eligibility_trace, double reward_next_state) {
		// Compute value
		double value_current_state = compute_value(this.X, this.weights);
		double gamma = 1.0;
		double delta = 1.0;
		
		ArrayList<Double> eligibility_trace = compute_eligibility_trace(gamma, delta, prev_eligibility_trace, value_current_state);		
		ArrayList<Double> new_weights = back_propagation(this.weights, this.lr, gamma, reward_next_state, value_current_state, value_next_state, eligibility_trace);
		this.weights = new_weights;
	}
	
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
