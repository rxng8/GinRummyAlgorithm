

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.function.Function;

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
	 * @param: Y (float[][]): Shape (none, 52). With 52 cards represented by state.
	 * 		state 0: The opponent does not have this card in hand.
	 * 		state 1: The opponent does have this card in hand.
	 * @param: seed (int): Random seed for nothing =))
	 * @param: lr (float): the learning rate!
	 * @param: n_iter (int): number of episode to be trained.
	 */
	
	/**
	 * TODO:
	 * 		1. Add bias variables for every weights.
	 * 		2. Working on back-propagation method:
	 * 			- Solve the dE/dW in the first layer where having to differentiate sigmoid function.
	 * 			- Fix bug or re-implement the whole function.
	 * 			- Methodize derivative function for better readable code.
	 * 				- Sigmoid_derivative
	 * 				- Softmax derivative
	 * 				- Categorical cross entropy derivative.
	 * 				- Linear Regression Sum derivative.
	 * 			- On back-propagating, research more on gradient descend and ascend, -= or +=.
	 * 			- Currently hard coding the number of layers. Need to write method that dynamically compute tye number of layers.
	 * 		3. Write Predicting method
	 * 		4. Test thoroughly
	 */

	/**
	 * ArrayList<ArrayList<ArrayList<ArrayList<short[][]>>>>: list of all games
	 * ArrayList<ArrayList<ArrayList<short[][]>>>: List of all players
	 * ArrayList<ArrayList<short[][]>>: List of all rounds.
	 * ArrayList<short[][]>: List of all turns.
	 * short[][]: One turn consisting of one input and one output.
	 */
	ArrayList<ArrayList<ArrayList<ArrayList<short[][]>>>> gamePlays;
	
	/**
	 * Input value for curent state. Probability matrix of 52
	 * input.shape[0]: number of training data.
	 * input.shape[1]: Number of features in one training data.
	 */
	public float[][] X;
	
	/**
	 * The label that we fit into the model!
	 */
	public float[][] Y;
	
	/**
	 * Random seed
	 */
	public int seed;
	
	/**
	 * Learning rate
	 */
	public float lr;
	
	/**
	 * Number of training iterations
	 */
	public float n_iter;
	
	/**
	 * The weights of layer 1. Input: X, Output: Layer 1
	 * weights.shape[0]: number of coordinates corresponding to number of input features.
	 * weights.shape[1]: Number of weights combination.
	 */
	private float[][] weights1;
	
	/**
	 * bias
	 */
	private float[] bias1;
	
	/**
	 * The weights of layer 2. Input: layer 1, Output: Layer 2
	 * weights.shape[0]: number of coordinates corresponding to number of input features.
	 * weights.shape[1]: Number of weights combination.
	 */
	private float[][] weights2;
	
	/**
	 * bias2
	 */
	private float[] bias2;
	
	/**
	 * The computed, evaluated, or predicted value from weights.
	 */
	private float[][] output;
	
	
	//Debugging params
	/**
	 * Verbose
	 */
	private boolean VERBOSE = true;
	
	/**
	 * Initialize the model with saved file
	 * @param filename (String): The path the saved model.
	 */
	@SuppressWarnings("unchecked")
	public HESimpleModel (String weights, String bias) {
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(weights));
			this.weights1 = (float[][]) in.readObject();
			this.weights2 = (float[][]) in.readObject();
			in.close();
			
			if (this.VERBOSE) {
				System.out.println("Read weights from file " + weights);
			}
			
		} catch (FileNotFoundException e) {
			System.out.println("File not found: " + weights);
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(bias));
			this.bias1 = (float[]) in.readObject();
			this.bias2 = (float[]) in.readObject();
			in.close();
			
			if (this.VERBOSE) {
				System.out.println("Read bias from file " + bias);
			}
			
		} catch (FileNotFoundException e) {
			System.out.println("File not found: " + bias);
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Save model, in this case, save weights.
	 * @param filename
	 */
	public void save(String weights, String bias) {
		try {
			ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(weights));
			out.writeObject(this.weights1);
			out.writeObject(this.weights2);
			out.close();
			if (this.VERBOSE) {
				System.out.println("Written weights to file " + weights);
			}
		} catch (FileNotFoundException e) {
			System.out.println("File not found: " + weights);
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		try {
			ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(bias));
			out.writeObject(this.bias1);
			out.writeObject(this.bias2);
			out.close();
			
			if (this.VERBOSE) {
				System.out.println("Written bias to file " + bias);
			}
		} catch (FileNotFoundException e) {
			System.out.println("File not found: " + bias);
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Initialize a new model
	 */
	public HESimpleModel () {
		// Have to call __init__ if model is new.
		if (this.VERBOSE) {
			System.out.println("Creating New Model...");
		}
	}
	
	/**
	 * Initialize data for a new model
	 * @param X (float[][]): One batch of input data. A batch of data with size X.length, and data length X[0].length
	 * @param Y (float[][]): One batch of output data. A batch of data with size Y.length, and data length Y[0].length
	 * @param seed (int): random seed for processing randomization.
	 * @param lr (float): learning rate.
	 * @param n_iter (int): Number of epochs.
	 */
	public void __init__(float[][] X, float[][] Y, int seed, float lr, int n_iter, boolean new_model) {
		this.X = X;
		this.Y = Y;
		assert X.length == Y.length : "The number of training items from input and output must match!";
		if (new_model) {
			this.weights1 = new float[X[0].length][8];
			this.bias1 = new float[this.weights1[0].length];
			this.weights2 = new float[weights1[0].length][Y[0].length];
			this.bias2 = new float[this.weights2[0].length];
		}
		this.seed = seed;
		this.lr = lr;
		this.n_iter = n_iter;
		
		if (this.VERBOSE) {
			System.out.println("Initializing Model...");
		}
	}
	
	public void __init__(int seed, float lr, int n_iter, boolean new_model) {
		preprocess_data();
		if (new_model) {
			this.weights1 = new float[X[0].length][8];
			this.bias1 = new float[this.weights1[0].length];
			this.weights2 = new float[weights1[0].length][Y[0].length];
			this.bias2 = new float[this.weights2[0].length];
		}
		this.seed = seed;
		this.lr = lr;
		this.n_iter = n_iter;
		
		if (this.VERBOSE) {
			System.out.println("Initializing Model...");
		}
	}
	
	/**
	 * 
	 * @param filename
	 */
	@SuppressWarnings("unchecked")
	public void __import_data__(String filename) {
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename));
			this.gamePlays = (ArrayList<ArrayList<ArrayList<ArrayList<short[][]>>>>) in.readObject();
			in.close();
			if (this.VERBOSE) {
				System.out.println("Imported data from file " + filename);
			}
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
	 *  Take the variable gameplays and preprocess to float[][] forms to put in X and Y
	 * 	ArrayList<ArrayList<ArrayList<ArrayList<short[][]>>>> gamePlays
	 */
	public void preprocess_data() {
		
		assert this.gamePlays != null : "You have to import the data first using __import_data__ method";
		
		if (this.VERBOSE) {
			System.out.println("Preprocessing data...");
		}
		
		int data_size = this.gamePlays.size()
				* this.gamePlays.get(0).size()
				* this.gamePlays.get(0).get(0).size()
				* this.gamePlays.get(0).get(0).get(0).size();
		int data_feature = this.gamePlays.get(0).get(0).get(0).get(0)[0].length;
		assert data_feature == this.gamePlays.get(0).get(0).get(0).get(0)[1].length : "Input and output does not match dimension";
		
		System.out.println("Data size: " + data_size);
		
		this.X = new float[data_size*10][data_feature];
		this.Y = new float[data_size*10][data_feature];
		
		Iterator<ArrayList<ArrayList<ArrayList<short[][]>>>> it_games = this.gamePlays.iterator();
		int i = 0;
		while(it_games.hasNext()) {
			Iterator<ArrayList<ArrayList<short[][]>>> it_players = it_games.next().iterator();
			while (it_players.hasNext()) {
				Iterator<ArrayList<short[][]>> it_rounds = it_players.next().iterator();
				while (it_rounds.hasNext()) {
					Iterator<short[][]> it_turns = it_rounds.next().iterator();
					while (it_turns.hasNext()) {
						short[][] turnData = it_turns.next();
						assert turnData.length == 2 : "Wrong data form!";
						for (int j = 0; j < turnData[0].length; j++) {
							X[i][j] = (float) turnData[0][j];
							Y[i][j] = (float) turnData[1][j];
						}
						i++;
					}
				}
			}
		}
		
	}
	
	
	/**
	 * Perform a sigmoid activation function
	 * @param x (float): param
	 * @return (float): the result of the sigmoid function.
	 */
	public static float sigmoid(float x) {
		return (float) (1 / (1 + Math.exp(x)));
	}
	
	/**
	 * Perform a sigmoid derivative function
	 * @param x (float): param
	 * @return (float): the result of the sigmoid derivative function.
	 */
	public static float sigmoid_derivative(float x) {
		return sigmoid(x) * (1 - sigmoid(x));
	}
	
	/**
	 * Perform a sigmoid activation function
	 * @param x (float[]): param vector
	 * @return (float[]): the result of the sigmoid function over a vector (element wise).
	 */
	public static float[] sigmoid(float[] x) {
		float[] y = new float[x.length];
		for (int i = 0; i < x.length; i++) {
			y[i] = sigmoid(x[i]);
		}
		return y;
	}
	
	/**
	 * Perform a softmax activation function
	 * @param x (float): param
	 * @return (float): the result of the softmax function.
	 */
	public static float[] softmax(float[] x) {
		float[] e_x = new float[x.length];
		
		float max_x = Float.MIN_VALUE;
		for (float f : x) {
			if (f > max_x) {
				max_x = f;
			}
		}
		
		float sum_x = 0;
		
		for (int i = 0; i < e_x.length; i++) {
			e_x[i] = (float) Math.exp(x[i] - max_x);
			sum_x += e_x[i];
		}
		
		// Normalize
		for (int i = 0; i < e_x.length; i++) {
			e_x[i] /= sum_x;
		}
		
		return e_x;
	}
	
	/**
	 * Perform a scalar multiplication
	 * @param x1 (float): scalar.
	 * @param x2 (float[]): vector.
	 * @return (float[]): The scaled vector.
	 */
	public static float[] dot(float x1, float[] x2) {
		float[] y = new float[x2.length];
		for (int i = 0; i < x2.length; i++) y[i] = x1 * x2[i];
		return y;
	}
	
	/**
	 * Perform a dot product of 2 vectors with the same dimension.
	 * @param x1 (float): param
	 * @param x2 (float): param
	 * @return (float): The dot product of the two vectors.
	 */
	public static double dot(float[] x1, float[] x2) {
		float sum = 0;
		for (int i = 0; i < x2.length; i++) sum += x1[i] * x2[i];
		return sum;
	}

	/**
	 * Perform a 2D matrix multiplication.
	 * @param x1 (float[][]): param
	 * @param x2 (float[][]): param
	 * @return (float[][]): The product of the two matrices.
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
	 * Normal feed-forward single-layer computation.
	 * @param input (float[]): input vector containing feature.
	 * @param weights (float[][]): the 2D weight matrix that map the input to an output vector of classes/nodes/neurons.
	 * @param bias (float[]): the bias that goes with the weight.
	 * @param activator (Function<float[], float[]>): The activation function to perform on the output element-wisely.
	 * @return (float[]): The output vector of classes/nodes/neurons.
	 */
	public static float[] compute_value(float[] input, float[][] weights, float[] bias, Function<float[], float[]> activator) {
		assert input.length == weights.length : "input length does not match weight feature length!";
		float[][] reshape_input = {input};
		float[][] y = dot(reshape_input, weights);
		
		assert y.length == 1 : "It must be a vector!";
		
		// Reshape
		float[] y_shaped = y[0];
		
		assert y_shaped.length == bias.length : "Bias length should be the same as number of neuron in a layer!";
		for (int i = 0; i < y_shaped.length; i++) {
			y_shaped[i] += bias[i];
		}
		
		return activator.apply(y_shaped);
	}
	
	/**
	 * Categorical cross entropy loss
	 * @param output (float[]): The predicted vector by the neural network.
	 * @param label (float[]): The actual label in the training set.
	 * @return (float): the amount of loss.
	 */
	public static float categorical_crossentropy(float[] output, float[] label) {
		assert output.length == label.length : "output length and label length cannot be different!";
		
		// Count non-zero label
		int non_zeros = 0;
		for (float l : label) {
			if (l != 0) non_zeros ++;
		}
		
		// Scale factor
		float scale_factor = 1f / non_zeros;
		
		// Compute log loss
		float log_loss = 0;
		for (int i = 0; i < label.length; i++) {
			if (label[i] != 0) { // Positive class.
				log_loss += -Math.log10(output[i]) * label[i] * scale_factor;
			}
		}
		
		return log_loss;
	}
	
	/**
	 * Perform custom back propagation and update weights.
	 * @param output
	 * @param label
	 */
//	public void back_propagation(float[] label, float[] output, float[]... layers) {
////		float loss = categorical_crossentropy(output, label);
//		
//		assert output.length == label.length : "";
//		assert output.length == this.weights2[0].length : "The length of output node in weights2 must match the number of ourput feature";
//		
//		float[][] dW2 = new float[this.weights2.length][this.weights2[0].length];
//		float d_sum = 0;
//		
//		for(int feature = 0; feature < this.weights2.length; feature++) {
//			for (int neuron = 0; neuron < this.weights2[0].length; neuron++) {
//				d_sum += dW2[feature][neuron] = this.lr * (output[neuron] - label[neuron]) * this.weights2[feature][neuron];
//				this.weights2[feature][neuron] += dW2[feature][neuron];
//				this.bias2[neuron] += dW2[feature][neuron];
//			}
//		}
//		
//		assert this.weights2.length == this.weights1[0].length : "";
//		
////		float[] l_layer1 = layers[0].clone();
//		
//		float[][] dW1 = new float[this.weights1.length][this.weights1[0].length];
//		
//		// BUG: Currently computing dE/dW by differentiating softmax, need to differentiate sigmoid.
//		for(int feature = 0; feature < this.weights1.length; feature++) {
//			for (int neuron = 0; neuron < this.weights1[0].length; neuron++) {
//				this.weights1[feature][neuron] += dW1[feature][neuron] = this.lr * d_sum * this.weights1[feature][neuron];
//				this.bias1[neuron] += dW1[feature][neuron];
//			}
//		}		
//	}
	
	/**
	 * Perform back-propagation and update weights and bias.
	 * @param label (float[]): The actual label in the training set.
	 * @param output (float[]): The predicted vector by the neural network.
	 * @param layers (float[]...): The computed hidden layers in the middle of the neural network in descending order.
	 * 		Eg: - Output layer L: output,
	 * 			- Hidden layer L-1: layers[0],
	 * 			- Hidden layer L-2: layers[1], etc.
	 */
	public void back_propagation(float[] label, float[] output, float[]... layers) {
//		float loss = categorical_crossentropy(output, label);
		
		assert output.length == label.length : "";
		assert output.length == this.weights2[0].length : "The length of output node in weights2 must match the number of ourput feature";
		
		float[][] dW2 = new float[this.weights2.length][this.weights2[0].length];
		
		
		// Compute dE/dW2 and update W2
		for (int neuron = 0; neuron < this.weights2[0].length; neuron++) {
			float dz_j = output[neuron] - label[neuron];
			for (int feature = 0; feature < this.weights2.length; feature++) {
				this.weights2[feature][neuron] -= dW2[feature][neuron] = this.lr * dz_j * layers[0][feature];
			}
		}
		
		// Compute dE/db2 and update b2
		for (int neuron = 0; neuron < this.weights2[0].length; neuron++) {
			float dz_j = output[neuron] - label[neuron];
			this.bias2[neuron] -= this.lr * dz_j;
		}
		
		// Compute dE/da^L-1_j
		float d_sum;
		float[] dA_L = new float[this.weights2.length];
		for (int feature = 0; feature < this.weights2.length; feature++) {
			d_sum = 0;
			for (int neuron = 0; neuron < this.weights2[0].length; neuron++) {
				float dz_j = output[neuron] - label[neuron];
				d_sum -= this.lr * dz_j * this.weights2[feature][neuron];
			}
			dA_L[feature] = d_sum;
		}
		
		assert this.weights2.length == this.weights1[0].length : "";
		
		float[][] dW1 = new float[this.weights1.length][this.weights1[0].length];
		
		// Compute dE/dW1 and update W1
		for (int neuron = 0; neuron < this.weights1[0].length; neuron++) {
			float d_a_j = dA_L[neuron];
			for (int feature = 0; feature < this.weights1.length; feature++) {
				this.weights1[feature][neuron] -= dW1[feature][neuron] = this.lr * sigmoid_derivative(d_a_j) * d_a_j * layers[1][feature];
			}
		}
		
		// Compute dE/db1 and update b1
		for (int neuron = 0; neuron < this.weights1[0].length; neuron++) {
			float d_a_j = dA_L[neuron];
			this.bias1[neuron] -= this.lr * sigmoid_derivative(d_a_j) * d_a_j * 1;
		}
	}
	
	/**
	 * Perform one epoch over all input rows, feed-forward, loss, and back-propagation.
	 */
	public void evaluate_step() {
		
		assert this.X.length == this.Y.length : "The length of input batches and label batches does not match!";
		
		for (int i = 0; i < X.length; i++) {
			
			float[] inputLayer = this.X[i];
			float[] label = this.Y[i];
			
			// Layer 1
			float[] layer1 = compute_value(inputLayer, this.weights1,this.bias1, HESimpleModel::sigmoid);
			
			// Layer 2 - Output
			float[] output = compute_value(layer1, this.weights2, this.bias2, HESimpleModel::softmax);
			
			back_propagation(label, output, layer1, inputLayer);
			
//			if (this.VERBOSE) {
//				this.print_weights();
//			}
			
			if (this.VERBOSE) {
				System.out.println("Model Architecture: ");
				System.out.print("Input: ");
				print_mat1D(inputLayer);
				System.out.print("Hidden Layer 1: ");
				print_mat1D(layer1);
				System.out.print("Output: ");
				print_mat1D(output);
				System.out.print("Label: ");
				print_mat1D(label);
				System.out.print("Loss: ");
				System.out.println(categorical_crossentropy(output, label) + "\n");	
			}
		}
	}
	
	/**
	 * Train through all epochs according to n_iter specified in the __init__() method
	 */
	public void train () {
		for (int _ = 0; _ < this.n_iter; _++) {
			this.evaluate_step();	
		}
		if (this.VERBOSE) {
			System.out.println("Weights: ");
			print_weights();
		}
	}
	
	/**
	 * Predict an input by using its current weights and bias matrix.
	 * @param input (float[]): input.
	 * @return (float[]): output.
	 */
	public float[] predict (float[] input) {
		assert input.length == X[0].length : "Wrong input size";
		
		float[] layer1 = compute_value(input, this.weights1, this.bias1, HESimpleModel::sigmoid);
		
		float[] output = compute_value(layer1, this.weights2, this.bias2, HESimpleModel::softmax);
		
		return output;
	}
	
	// Debugging method
	
	/**
	 * 
	 * @param mat
	 */
	@SuppressWarnings("unused")
	private static void print_mat2D(float[][] mat) {
		System.out.print("[");
		for (int i = 0; i < mat.length; i++) {
			System.out.print("[ ");
			for (int j = 0; j < mat[i].length; j++) {
				System.out.print(mat[i][j] + " ");
			}
			if (i == mat.length - 1) {
				System.out.print("]");
			} else {
				System.out.print("],\n");
			}
		}
		System.out.print("]\n");
	}
	
	/**
	 * 
	 * @param mat
	 */
	@SuppressWarnings("unused")
	private static void print_mat1D (float[] mat) {
		System.out.print("[");
		for (int i = 0; i < mat.length; i++) {
			System.out.print(mat[i] + " ");
		}
		System.out.print("]\n");
	}
	
	/**
	 * 
	 */
	@SuppressWarnings("unused")
	private void print_weights () {
		System.out.print("Weights 1: \n");
		print_mat2D(this.weights1);
		System.out.print("Bias1: \n");
		print_mat1D(this.bias1);
		System.out.print("\nWeights2: \n");
		print_mat2D(this.weights2);
		System.out.print("Bias2: \n");
		print_mat1D(this.bias2);
		System.out.println();
	}
	
	private static void print_mat1D_card(float[] mat, String name) {
		System.out.println();
		System.out.println(name + ": ");
		int a = 0;
		for (int i = 0; i < mat.length; i++) {
			// Debugging
			System.out.printf("%s: %.5f ",Card.getCard(i).toString(), mat[i]);
			a++;
			if(a == 13) {
				a = 0;
				System.out.println();
			}
		}
		System.out.println();
	}
	
	/**
	 * 
	 * @param v
	 */
	public void set_verbose(boolean v) {
		this.VERBOSE = v;
	}
	
	/**
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		// Test Matrix multiplication
//		float[][] x1 = {{2, 3, 1}, {3, 6, 0}};
//		float[][] x2 = {{2}, {3}, {0}};
//		printMatrix(dot(x1, x2));
		
//		float[][] X = {{-2, -1, 0, 0, 1},
//							{-2, 0, 1, 1, 0},
//							{-2, 1, 0, 1, 0},
//							{0, -2, -1, 0, 1}};
//		
//		float[][] Y = {{0, 0, 0, 1, 1},
//						{0, 1, 1, 1, 1},
//						{0, 1, 1, 1, 1},
//						{0, 0, 0, 1, 1}};
		
//		HESimpleModel model = new HESimpleModel();
//		model.__init__(X, Y, 10, 1e-2f, 300);
		
//		model.train();
//		print_mat1D(HESimpleModel.compute_value(X[0], model.weights1, model.bias1, HESimpleModel::sigmoid));
		
//		model.weights1[1][2] = 12.3f;
		
//		model.evaluate_step();
		
//		model.set_verbose(true);
//		model.train();
		
//		print_mat1D(HESimpleModel.compute_value(X[0], model.weights1, model.bias1, HESimpleModel::sigmoid));
//		System.out.println(model.weights1[0].length);
		
//		System.out.print("Prediction for :");
//		HESimpleModel.print_mat1D(X[1]);
//		System.out.print("\n");
//		HESimpleModel.print_mat1D(model.predict(X[1]));
		
		/**
		 * New model
		 */
//		HESimpleModel model = new HESimpleModel();
//		model.__import_data__("play_data_SimplePlayer_small.dat");
//		model.__init__(10, 10e-3f, 1, true);
//		model.train();
//		model.save("weights_100.dat", "bias_100.dat");
		
		/**
		 * Load model
		 */
		HESimpleModel model = new HESimpleModel("weights_100.dat", "bias_100.dat");
		model.__import_data__("play_data_SimplePlayer_small.dat");
		model.__init__(10, 10e-3f, 1, false);
		
		/**
		 * Predict hand.
		 */
		HESimpleModel.print_mat1D_card(model.X[2], "Input to be predicted");
		HESimpleModel.print_mat1D_card(model.predict(model.X[2]), "Predicted");
		HESimpleModel.print_mat1D_card(model.Y[2], "Actual Data");
	}
}
