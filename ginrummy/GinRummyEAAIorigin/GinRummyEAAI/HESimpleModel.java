import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

public class HESimpleModel {
	
	/**
	 * This is a hand crafted simple fully connected neural network
	 * with fixed 2 layer. The first layer has input is the input and
	 * output 64 neurons. The second layer has input which is output of
	 * the first layer, and spit out a single value!
	 * 
	 * @param: X (float[][]): Shape (none, 52). With 52 cards represented by state.
	 * 		state -2: The opponent does discard the card.
	 * 		state -1: The opponent does not care about the card.
	 * 		state 0: This is the unrelated card.
	 * 		state 1: The opponent does care and pick up the card.
	 * @param: X (float[][]): Shape (none, 52). With 52 cards represented by state.
	 * 		state 0: The opponent does not have this card in hand.
	 * 		state 1: The opponent does have this card in hand.
	 * @param: seed (int): Random seed for nothing =))
	 * @param: lr (float): the learning rate!
	 * @param: n_iter (int): number of episode to be trained.
	 */
	
	// Input value for curent state. Probability matrix of 52
	// input.shape[0]: number of training data.
	// input.shape[1]: Number of features in one training data.
	public float[][] X;
	
	/**
	 * The label that we fit into the model!
	 */
	public float[][] Y;
	
	// Randome seed
	public int seed;
	
	// Learning rate
	public float lr;
	
	// Number of training iterations
	public float n_iter;
	
	// The weights of layer 1. Input: X, Output: Layer 1
	// weights.shape[0]: number of coordinates corresponding to number of input features.
	// weights.shape[1]: Number of weights combination.
	private float[][] weights1;
	
	// The weights of layer 2. Input: layer 1, Output: Layer 2
	// weights.shape[0]: number of coordinates corresponding to number of input features.
	// weights.shape[1]: Number of weights combination.
	private float[][] weights2;
	
	/**
	 * The computed, evaluated, or predicted value from weights.
	 */
	private float[] output;
	
	/**
	 * Initialize the model with saved file
	 * @param filename (String): The path the saved model.
	 */
	@SuppressWarnings("unchecked")
	public HESimpleModel (String filename) {
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename));
			this.weights1 = (float[][]) in.readObject();
			this.weights2 = (float[][]) in.readObject();
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
	public HESimpleModel () {
		// Have to call __init__ if model is new.
	}
	
	/**
	 * Initialize data for a new model
	 * @param weights (ArrayList<Double>): Probability vector of length 52.
	 * @param seed (int): Random seed.
	 * @param lr (double): Leanring rate.
	 * @param n_iter (int): Number of iterations.
	 */
	public void __init__(float[][] X, float[][] Y, int seed, float lr, int n_iter) {
		this.X = X;
		this.Y = Y;
		assert X.length == Y.length : "The number of training items from input and output must match!";
		this.weights1 = new float[X[0].length][64];
		this.weights2 = new float[weights1[0].length][1];
		this.seed = seed;
		this.lr = lr;
		this.n_iter = n_iter;
	}
	
	/**
	 * Perform a sigmoid activation function
	 * @param x (double) param
	 * @return (double) the result of the sigmoid function.
	 */
	public static float sigmoid(float x) {
		return (float) (1 / (1 + Math.exp(x)));
	}
	
	/**
	 * Perform a sigmoid activation function
	 * @param x (double) param
	 * @return (double) the result of the sigmoid function.
	 */
	public static float[] sigmoid(float[] x) {
		float[] y = new float[x.length];
		for (int i = 0; i < x.length; i++) {
			y[i] = sigmoid(x[i]);
		}
		return y;
	}
	
	/**
	 * Perform a scalar multiplication
	 * @param X (ArrayList<Double>): Param
	 * @param weights (ArrayList<Double>): Param
	 * @return (double) The dot product of the two vectors.
	 */
	public static float[] dot(float x1, float[] x2) {
		float[] y = new float[x2.length];
		for (int i = 0; i < x2.length; i++) y[i] = x1 * x2[i];
		return y;
	}
	
	/**
	 * Perform a dot product of 2 vectors with the same dimension.
	 * @param X (ArrayList<Double>): Param
	 * @param weights (ArrayList<Double>): Param
	 * @return (double) The dot product of the two vectors.
	 */
	public static double dot(float[] x1, float[] x2) {
		float sum = 0;
		for (int i = 0; i < x2.length; i++) sum += x1[i] * x2[i];
		return sum;
	}
	
	/**
	 * Perform a 2D matrix multiplication.
	 * @param X (ArrayList<Double>): Param
	 * @param weights (ArrayList<Double>): Param
	 * @return (double) The dot product of the two vectors.
	 */
	public static float[][] dot(float[][] x1, float[][] x2) {
		assert x1[0].length == x2.length : "The columns of previous matrix should match the rows of the next matrix";
		
		float[][] y = new float[x1.length][x2[0].length];
		for (int rowx1 = 0; rowx1 < x1.length; rowx1++) {
			for (int colx2 = 0; colx2 < x2[0].length; colx2++) {
				float sum = 0;
				for (int colx1 = 0; colx1 < x1[0].length; colx1++) {
					sum += x1[rowx1][colx1] * x2[colx1][colx2];
				}
				y[rowx1][colx2] = sum;
			}
		}
		return y;
	}
	
	/**
	 * Normal feed-forward ANN computation for one layer.
	 * @param inputs.
	 * @param weights. 
	 * @return
	 */
	public static float[] compute_value(float[] inputs, float[][] weights) {
		assert inputs.length == weights.length : "input length does not match weight feature length!";
		float[][] reshape_input = {inputs};
		float[][] y = dot(reshape_input, weights);
		
		assert y.length == 1 : "It must be the vector!";
		// Reshape
		return sigmoid(y[0]);
	}
	
	/**
	 * Normal feed-forward ANN computation for one layer.
	 * @param inputs.
	 * @param weights. 
	 * @return
	 */
	public static float categorical_crossentropy(float[] output, float[] label) {
		assert output.length == label.length : "output length and label length cannot be different!";
		
		
		
		return 0;
	}
	
	/**
	 * Perform custom back propagation.
	 */
	public void back_propagation() {
	
		
	}
	
	/**
	 * Perform one epoch over all input rows, feed-forward, loss, and back-propagation.
	 */
	public void evaluate_step() {
		for (float[] inputLayer : this.X) {
			
			float[] layer1 = compute_value(inputLayer, this.weights1);
			this.output = compute_value(layer1, this.weights2);
			
			
		}
	}
	
	public void train () {
		
	}
	
	/**
	 * Save model, in this case, save weights.
	 * @param filename
	 */
	public void save(String filename) {
		try {
			ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename));
			out.writeObject(this.weights1);
			out.writeObject(this.weights2);
			out.close();
		} catch (FileNotFoundException e) {
			System.out.println("File not found: " + filename);
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	// Debugging method
	@SuppressWarnings("unused")
	private static void printMatrix(float[][] mat) {
		System.out.print("[ ");
		for (int i = 0; i < mat.length; i++) {
			for (int j = 0; j < mat[i].length; j++) {
				System.out.print(mat[i][j] + " ");
			}
			if (i == mat[0].length - 1) {
				System.out.print("]\n");
			} else {
				System.out.println();
			}
		}
	}
	
//	 Test
	public static void main(String[] args) {
		// Test Matrix multiplication
		float[][] x1 = {{2, 3, 1}, {3, 6, 0}};
		float[][] x2 = {{2}, {3}, {0}};
		printMatrix(dot(x1, x2));
	}

}
