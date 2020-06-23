import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.function.Function;

public class HandEstimationModel {

	/**
	 * This is a hand crafted Hand Estimation model with with 4 LSTM layer with 4 input, then concatenate those hidden layer to form a Dense
	 * network with dynamic options of layers and neurons.
	 * 
	 * @param: X (ArrayList<ArrayList<float[][]>>): Shape (none, 4, 52). With 52 one-hot encoded state.
	 * 		X.get(0): each round data has been  generated.
	 * 			X.get(0).get(0): each turn collected in each game.
	 * 				X.get(0).get(0)[0]: Vector of cards the opponent discarded this turn
	 * 				X.get(0).get(0)[1]: Vector of cards that the opponent does not care about (not picking up from discard pile) this turn.
	 * 				X.get(0).get(0)[2]: Vector of cards that the opponent does care and pick up from the discard pile.
	 * 				X.get(0).get(0)[3]: Cards that are on this player's hand and in the discard pile.
	 * @param: Y (ArrayList<ArrayList<float[]>>): Shape (none, 52). With 52 cards represented by state.
	 * 		Y.get(0): each round data has been  generated.
	 * 			Y.get(0).get(0): each turn collected in each game, containing an opponent hand.

	 * @param: seed (int): Random seed for nothing =))
	 * @param: lr (float): the learning rate!
	 * @param: n_iter (int): number of episode to be trained.
	 */
	
	/**
	 * TODO:
	 * 		1. Make the back-propagation function dynamic.
	 * 		2. Research more on back-propagation.
	 * 		3. Write training method.
	 * 		4. Finish the specific dense_layer methods.
	 */
	
	/**
	 * Constant
	 */
	public static final int DISCARD_VEC = 0;
	public static final int DONT_CARE_VEC = 1;
	public static final int CARE_VEC = 2;
	public static final int EXISTED_VEC = 3;
	
	public static final int ID_FORGET_GATE = 0;
	public static final int ID_INPUT_GATE = 1;
	public static final int ID_CANDIDATE_GATE = 2;
	public static final int ID_OUTPUT_GATE = 3;
	
	public static final int ID_MEMORY_VECTOR = 0;
	public static final int ID_LSTM_OUT_VECTOR = 1;
	
	public static final int CARD_FEATURE = 52;
	public static final int INPUT_TURN_LEN = 4;
	
	/**
	 * 
	 */
	ArrayList<ArrayList<ArrayList<ArrayList<short[][]>>>> gamePlays;
	
	/**
	 * one float[][] is one input datum for one turn.
	 * ArrayList<float[][]> represents sequence of turns.
	 * ArrayList<ArrayList<float[][]>> represents sequences of matches.
	 * 
	 * Future work: Take the data "Game Score" to process, too.
	 */
	public ArrayList<ArrayList<float[][]>> X;
	
	/**
	 * one float[][] is one input datum for one turn.
	 * ArrayList<float[][]> represents sequence of turns.
	 * ArrayList<ArrayList<float[][]>> represents sequences of matches.
	 * 
	 * Future work: Take the data "Game Score" to process, too.
	 */
	public ArrayList<ArrayList<float[]>> Y;
	
	/**
	 * Y_hat
	 */
	public ArrayList<ArrayList<float[]>> outputs;
	
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
	 * Array of length 4 => 4 state vector
	 */
	private int[] lstm_neurons;
	
	/**
	 * Array of length dynamic, corresponding to the number of feed forward layers.
	 */
	private int[] dense_neurons;
	
	/**
	 * Input dimension
	 */
	private int[] input_dim;
	
	/**
	 * 
	 */
	private int output_dim;
	
	/**
	 * equal lstm_neurons[i] + input_dim
	 */
	private int[] lstm_cell_neurons;
	
	/**
	 * lstm_weights[i].get(j)[q][k] is a coordinate.
	 * lstm_weights[i].get(j)[q] is a vector of coordinates that map input to one neuron.
	 * lstm_weights[i].get(j) is a particular kind of matrix that have actual function in an lstm cell.
	 * 		lstm_weights[0].get(0): weights for forgetting gate. sigmoid(Wf[h_t-1, x_t] + bf)
	 * 		lstm_weights[0].get(1): weights for input gate. sigmoid(Wi[h_t-1, x_t] + bi)
	 * 		lstm_weights[0].get(2): weights for candidate gate. tanh(Wc[h_t-1, x_t] + bc)
	 * 		lstm_weights[0].get(3): weights for output gate. sigmoid(Wo[h_t-1, x_t] + bo)
	 * lstm_weights[i] is the set of weights of each input
	 * 		lstm_weights[0] (ArrayList<float[][]>): the opponent discarded this turn
	 * 		lstm_weights[1] (ArrayList<float[][]>): the opponent does not care about (not picking up from discard pile) this turn.
	 * 		lstm_weights[2] (ArrayList<float[][]>): the opponent does care and pick up from the discard pile.
	 * 		lstm_weights[3] (ArrayList<float[][]>): Cards that are on this player's hand and in the discard pile.
	 */
	private ArrayList<float[][]>[] lstm_weights;
	
	private ArrayList<float[]>[] lstm_bias;
	
	private ArrayList<float[][]> dense_weights;
	
	private ArrayList<float[]> dense_bias;
	
	private boolean VERBOSE;
	
	/**
	 * Initialize a new model
	 */
	public HandEstimationModel () {
		// Have to call __init__ if model is new.
		if (this.VERBOSE) {
			System.out.println("Creating New Model...");
		}
	}
	
	/**
	 * Initialize the model with saved file
	 * @param filename (String): The path the saved model.
	 */
	@SuppressWarnings("unchecked")
	public HandEstimationModel (String filename) {
		if (this.VERBOSE) {
			System.out.println("Importing model...");
		}
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename));
			this.lstm_weights = (ArrayList<float[][]>[]) in.readObject();
			this.lstm_bias = (ArrayList<float[]>[]) in.readObject();
			this.dense_weights = (ArrayList<float[][]>) in.readObject();
			this.dense_bias = (ArrayList<float[]>) in.readObject();
			in.close();
			
			if (this.VERBOSE) {
				System.out.println("Read weights from file " + filename);
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
	 * Save model, in this case, save weights.
	 * @param filename
	 */
	public void save(String weights, String bias) {
		
		if (this.VERBOSE) {
			System.out.println("Saving model...");
		}
		
		try {
			ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(weights));
			out.writeObject(this.lstm_weights);
			out.writeObject(this.lstm_bias);
			out.writeObject(this.dense_weights);
			out.writeObject(this.dense_bias);
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
		
		if (this.VERBOSE) {
			System.out.println("Model Saved!");
		}
	}
	
	/**
	 * Initialize data for a new model, weights
	 * @param X (float[][]): One batch of input data. A batch of data with size X.length, and data length X[0].length
	 * @param Y (float[][]): One batch of output data. A batch of data with size Y.length, and data length Y[0].length
	 * @param seed (int): random seed for processing randomization.
	 * @param lr (float): learning rate.
	 * @param n_iter (int): Number of epochs.
	 */
	@SuppressWarnings("unchecked")
	public void __init__(int[] lstm_neurons,
			int[] dense_neurons,
			int[] input_dim,
			int output_dim, 
			int seed,
			float lr,
			int n_iter,
			boolean new_model) {
		
		preprocess_data();
		
		assert lstm_neurons.length == INPUT_TURN_LEN : "Number of lstm_neuron must exist in 4 input states";
		
		assert input_dim.length == INPUT_TURN_LEN : "Number of lstm_neuron must exist in 4 input states";
		
		if (new_model) {
			// Initialize weights.
			int total_lstm_neurons = 0;
			this.lstm_cell_neurons = new int[lstm_neurons.length];
			this.lstm_weights = (ArrayList<float[][]>[]) new Object[INPUT_TURN_LEN];
			this.lstm_bias = (ArrayList<float[]>[]) new Object[INPUT_TURN_LEN];
			for (int i = 0; i < INPUT_TURN_LEN; i++) {
				int lstm_neurons_this_cell = lstm_neurons[i];
				lstm_cell_neurons[i] = lstm_neurons[i] + input_dim[i];
				// Initialize weights
				float[][] forget_weights = new float[lstm_neurons_this_cell][lstm_cell_neurons[i]];
				float[][] input_weights = new float[lstm_neurons_this_cell][lstm_cell_neurons[i]];
				float[][] candidate_weights = new float[lstm_neurons_this_cell][lstm_cell_neurons[i]];
				float[][] output_weights = new float[lstm_neurons_this_cell][lstm_cell_neurons[i]];
				
				this.lstm_weights[i] = new ArrayList<float[][]>();
				this.lstm_weights[i].add(forget_weights);
				this.lstm_weights[i].add(input_weights);
				this.lstm_weights[i].add(candidate_weights);
				this.lstm_weights[i].add(output_weights);
				
				// Initialize bias
				float[] forget_bias = new float[lstm_neurons_this_cell];
				float[] input_bias = new float[lstm_neurons_this_cell];
				float[] candidate_bias = new float[lstm_neurons_this_cell];
				float[] output_bias = new float[lstm_neurons_this_cell];
				
				this.lstm_bias[i] = new ArrayList<float[]>();
//				this.lstm_bias[i].add(forget_bias);
//				this.lstm_bias[i].add(input_bias);
//				this.lstm_bias[i].add(candidate_bias);
//				this.lstm_bias[i].add(output_bias);
				this.lstm_bias[i].set(ID_FORGET_GATE, forget_bias);
				this.lstm_bias[i].set(ID_INPUT_GATE, input_bias);
				this.lstm_bias[i].set(ID_CANDIDATE_GATE, candidate_bias);
				this.lstm_bias[i].set(ID_OUTPUT_GATE, output_bias);
				
				total_lstm_neurons += lstm_neurons_this_cell;
			}
			
			// Initialize dense layer.
			this.dense_weights = new ArrayList<float[][]>();
			this.dense_bias = new ArrayList<float[]>();
			
			// For each layer, initialize with the specified neurons.
			for (int i = 0; i < dense_neurons.length; i++) {
				if (i == 0) {
					float[][] dense_weights_this_layer = new float[dense_neurons[i]][total_lstm_neurons];
					this.dense_weights.add(dense_weights_this_layer);
				} else {
					float[][] dense_weights_this_layer = new float[dense_neurons[i]][dense_neurons[i - 1]];
					this.dense_weights.add(dense_weights_this_layer);
				}
				
				this.dense_bias.add(new float[dense_neurons[i]]);
			}
			
			float[][] output_weights = new float[output_dim][dense_neurons[dense_neurons.length - 1]];
			this.dense_weights.add(output_weights);
			
			float[] output_bias = new float[output_dim];
			this.dense_bias.add(output_bias);
		}
		
		// Do every other things
		this.lstm_neurons = lstm_neurons.clone();
		this.dense_neurons = dense_neurons.clone();
		this.input_dim = input_dim.clone();
		this.output_dim = output_dim;
		this.seed = seed;
		this.lr = lr;
		this.n_iter = n_iter;
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
	@SuppressWarnings("unchecked")
	private void preprocess_data() {
		
		assert this.gamePlays != null : "You have to import the data first using __import_data__ method";
		
		if (this.VERBOSE) {
			System.out.println("Preprocessing data...");
		}
		
		int data_feature = this.gamePlays.get(0).get(0).get(0).get(0)[0].length;
		assert data_feature == this.gamePlays.get(0).get(0).get(0).get(0)[4].length : "Input and output does not match dimension";
		
		Iterator<ArrayList<ArrayList<ArrayList<short[][]>>>> it_games = this.gamePlays.iterator();
		while(it_games.hasNext()) {
			Iterator<ArrayList<ArrayList<short[][]>>> it_players = it_games.next().iterator();
			while (it_players.hasNext()) {
				Iterator<ArrayList<short[][]>> it_rounds = it_players.next().iterator();
				while (it_rounds.hasNext()) {
					Iterator<short[][]> it_turns = it_rounds.next().iterator();
					ArrayList<float[][]> inputs = new ArrayList<>();
					ArrayList<float[]> outputs = new ArrayList<>();
					while (it_turns.hasNext()) {
						short[][] turnData = it_turns.next();
						assert turnData.length == INPUT_TURN_LEN + 1 : "Wrong data form! There are 4 input vector, and 1 output vector.";
						assert turnData[0].length == CARD_FEATURE : "Wrong data form! One card vector must have 52 features";
						float[][] input = new float[INPUT_TURN_LEN][CARD_FEATURE];
						float[] output = new float[CARD_FEATURE];
						for (int j = 0; j < CARD_FEATURE; j++) {
							input[0][j] = (float) turnData[0][j];
							input[1][j] = (float) turnData[1][j];
							input[2][j] = (float) turnData[2][j];
							input[3][j] = (float) turnData[3][j];
							output[j] = (float) turnData[4][j];
						}
						inputs.add(input);
						outputs.add(output);
					}
					this.X.add(inputs);
					this.Y.add(outputs);
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
	 * Perform a sigmoid derivative function
	 * @param x (float): param
	 * @return (float): the result of the sigmoid derivative function.
	 */
	public static float sigmoid_derivative(float x) {
		return sigmoid(x) * (1 - sigmoid(x));
	}
	
	public static float tanh (float x) {
		return 2 * sigmoid(2 * x) - 1;
	}
	
	public static float[] tanh (float[] x) {
		float[] y = new float[x.length];
		for (int i = 0; i < y.length; i++) {
			y[i] = tanh(x[i]);
		}
		return y;
	}
	
	public static float tanh_derivative (float x) {
		return 1 - (float) Math.pow(tanh(x), 2);
	}
	
	public static float relu (float x) {
		return Math.max(0, x);
	}
	
	public static float[] relu (float[] x) {
		float[] y = new float[x.length];
		for (int i = 0; i < y.length; i++) {
			y[i] = relu(x[i]);
		}
		return y;
	}
	
	public static float relu_derivative (float x) {
		return x > 0 ? 1 : 0;
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

	public static float[] multiply(float[] x1, float[] x2) {
		assert x1.length == x2.length: "The dimension of two vector when multiplying element-wise must match";

		float[] y = new float[x1.length];
		for (int i = 0; i < y.length; i++) {
			y[i] = x1[i] * x2[i];
		}

		return y;
	}

	public static float[] add(float[] x1, float[] x2) {
		assert x1.length == x2.length: "The dimension of two vector when adding element-wise must match";

		float[] y = new float[x1.length];
		for (int i = 0; i < y.length; i++) {
			y[i] = x1[i] + x2[i];
		}

		return y;
	}
	
	/**
	 * Normal feed-forward single-layer computation.
	 * @param input (float[]): input vector containing feature.
	 * @param weights (float[][]): the 2D weight matrix that map the input to an output vector of classes/nodes/neurons.
	 * @param bias (float[]): the bias that goes with the weight.
	 * @param activator (Function<float[], float[]>): The activation function to perform on the output element-wisely.
	 * @return (float[]): The output vector of classes/nodes/neurons. delta ( W * x + b )
	 */
	public static float[] compute_value(float[][] weights, float[] input, float[] bias, Function<float[], float[]> activator) {
		assert input.length == weights[0].length : "input length does not match weight feature length!";
		float[][] reshaped_input = new float[input.length][1]; // Reshape vector
		
		for (int i = 0; i < input.length; i++) {
			reshaped_input[i][0] = input[i];
		}
		
		float[][] y = dot(weights, reshaped_input);
		
		assert y[0].length == 1 : "It must be a vector!";
		
		// Reshape
		float[] y_shaped = new float[y.length];
		
		for (int i = 0; i < y_shaped.length; i++) {
			y_shaped[i] = y[i][0];
		}
		
		assert y_shaped.length == bias.length : "Bias length should be the same as number of neuron in a layer!";
		
		for (int i = 0; i < y_shaped.length; i++) {
			y_shaped[i] += bias[i];
		}
		
		return activator.apply(y_shaped);
	}
	
	/**
	 * 
	 * @param vectors
	 * @return
	 */
	public static float[] concatenate_vector(float[]... vectors) {
		
		int length = 0;
		for (float[] vector : vectors) {
			length += vector.length;
		}
		
		float[] y = new float[length];
		int i = 0;
		for(float[] vector : vectors) {
			for (float x : vector) {
				y[i] = x;
				i++;
			}
		}
		
		return y;
	}
	
	/**
	 * LSTM Layer or LSTM Cell
	 * @param weights
	 * @param input
	 * @param bias
	 * @param state
	 * @return val (ArrayList<float[]>):
	 * 		val.get(0): memory
	 * 		val.get(1): output
	 */
	public static ArrayList<float[]> lstm_cell (ArrayList<float[][]> weights, float[] input, ArrayList<float[]> bias, ArrayList<float[]> state) {
		float[][] Wf = weights.get(ID_FORGET_GATE); // get 0
		float[][] Wi = weights.get(ID_INPUT_GATE); // get 1
		float[][] Wc = weights.get(ID_CANDIDATE_GATE); // get 2
		float[][] Wo = weights.get(ID_OUTPUT_GATE); // get 3
		
		float[] bf = bias.get(ID_FORGET_GATE); // get 0
		float[] bi = bias.get(ID_INPUT_GATE); // get 1
		float[] bc = bias.get(ID_CANDIDATE_GATE); // get 2
		float[] bo = bias.get(ID_OUTPUT_GATE); // get 3
		
		assert Wf.length == bf.length : "Bias dimension must match the number of neurons in weights";
		assert Wi.length == bi.length : "Bias dimension must match the number of neurons in weights";
		assert Wc.length == bc.length : "Bias dimension must match the number of neurons in weights";
		assert Wo.length == bo.length : "Bias dimension must match the number of neurons in weights";
		
		float[] prev_memory = state.get(ID_MEMORY_VECTOR);
		float[] prev_output = state.get(ID_LSTM_OUT_VECTOR);
		
		// Concatenate previous output vector with current input vector
		float[] concated_input = concatenate_vector(prev_output, input);
		
		// Assert, check, test
		assert Wf[0].length == concated_input.length : "Concatenated input length did not match forget gate weights";
		assert Wi[0].length == concated_input.length : "Concatenated input length did not match input gate weights";
		assert Wc[0].length == concated_input.length : "Concatenated input length did not match candidate gate weights";
		assert Wo[0].length == concated_input.length : "Concatenated input length did not match output gate weights";
		
		// Compute 4 outputs!
		
		float[] out_forget = dense_cell_sigmoid(Wf, concated_input, bf);
		float[] out_input = dense_cell_sigmoid(Wi, concated_input, bi);
		float[] out_candidate = dense_cell_tanh(Wc, concated_input, bc);
		float[] out_output = dense_cell_sigmoid(Wo, concated_input, bo);
		
		// Compute element-wise operations
		float[] new_memory = add(multiply(out_forget, prev_memory), multiply(out_input, out_candidate));
		float[] new_output = multiply(out_output, tanh(new_memory));
		
		ArrayList<float[]> new_state = new ArrayList<float[]>();
		new_state.set(ID_MEMORY_VECTOR, new_memory);
		new_state.set(ID_LSTM_OUT_VECTOR, new_output);
		
		return new_state;
	}
	
	public static float[] dense_cell_tanh (float[][] weights, float[] input, float[] bias) {
		assert weights[0].length == input.length: "The number of coordinates in weights must match the number of input features";
		assert weights.length == bias.length : "The number of neurons must match the number of bias";
		return compute_value(weights, input, bias, HandEstimationModel::tanh);
	}
	
	/**
	 * Dense layer or Dense cell using relu activation function
	 * @param weights
	 * @param input
	 * @param bias
	 * @return
	 */
	public static float[] dense_cell_relu (float[][] weights, float[] input, float[] bias) {
		assert weights[0].length == input.length: "The number of coordinates in weights must match the number of input features";
		assert weights.length == bias.length : "The number of neurons must match the number of bias";
		return compute_value(weights, input, bias, HandEstimationModel::relu);
	}
	
	/**
	 * Dense layer or Dense cell using sigmoid activation function
	 * @param weights
	 * @param input
	 * @param bias
	 * @return
	 */
	public static float[] dense_cell_sigmoid (float[][] weights, float[] input, float[] bias) {
		assert weights[0].length == input.length: "The number of coordinates in weights must match the number of input features";
		assert weights.length == bias.length : "The number of neurons must match the number of bias";
		return compute_value(weights, input, bias, HandEstimationModel::sigmoid);
	}
	
	/**
	 * Dense layer or Dense cell using softmax activation function
	 * @param weights
	 * @param input
	 * @param bias
	 * @return
	 */
	public static float[] dense_cell_softmax (float[][] weights, float[] input, float[] bias) {
		assert weights[0].length == input.length: "The number of coordinates in weights must match the number of input features";
		assert weights.length == bias.length : "The number of neurons must match the number of bias";
		return compute_value(weights, input, bias, HandEstimationModel::softmax);
	}
	
	/**
	 * Perform one step of feed forward network. Checkout model here and there!
	 * @param input
	 * @param states
	 * @return (Object[]):
	 * 		(ArrayList<float[]>[]): new_states
	 * 		(ArrayList<float[]>): layers
	 * 		(float[]): output
	 */
	@SuppressWarnings("unchecked")
	public Object[] evaluate_step(float[][] input, ArrayList<float[]>[] states) {
		
		assert states.length == INPUT_TURN_LEN : "list of number of states in input did not match the number of input!";
		
		// Create new state vector
		ArrayList<float[]>[] new_states = (ArrayList<float[]>[]) new Object[states.length];
		
		// For each input, we put into a LSTM cell. After that, concatenate.
		float[] concatenated_vector = new float[0];

		for (int i = 0; i < input.length; i++) {
			ArrayList<float[]> this_state = lstm_cell(this.lstm_weights[i], input[i], this.lstm_bias[i], states[i]);
			assert this_state.size() == 2 : "Wrong lstm cell behavior";
			float[] output = this_state.get(ID_LSTM_OUT_VECTOR);
			concatenated_vector = concatenate_vector(concatenated_vector, output);
			new_states[i] = this_state;
		}
		
		// Add the dense layer after the concatenation.
		ArrayList<float[]> dense_layers = new ArrayList<>();
		dense_layers.add(concatenated_vector);
		
		// Compute each layer of dense cell
		for (int i = 0; i < this.dense_neurons.length; i++) {
			float[] in;
			if (i == 0) {
				in = concatenated_vector;
			} else {
				in = dense_layers.get(dense_layers.size() - 1);
			}
			float[] out = dense_cell_relu(this.dense_weights.get(i), in, this.dense_bias.get(i));
			dense_layers.add(out);
		}
		
		// Compute output layer
		float[] y_pred = dense_cell_softmax(this.dense_weights.get(this.dense_weights.size() - 1), dense_layers.get(dense_layers.size() - 1), this.dense_bias.get(this.dense_bias.size() - 1));
		
		Object[] return_val = new Object[3];
		return_val[0] = new_states;
		return_val[1] = dense_layers;
		return_val[2] = y_pred;
		
		return return_val;
	}
	
	/**
	 * 
	 * @param output
	 * @param label
	 * @return
	 */
	public static float crossentropy (float[] output, float[] label) {
		return 0;
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
	 * 
	 * @param d_down_layer
	 * @param d_activator
	 * @param d_upper_layer
	 * @return
	 */
	public static float[][] compute_d_weights(float[] d_down_layer,
			float[] d_activator,
			float[] d_upper_layer) {
		
		assert d_activator.length == d_upper_layer.length;
		
		// Create derivative of weights matrix with neuron-rows and feature-columns
		float[][] dW = new float[d_upper_layer.length][d_down_layer.length];
		
		for (int neuron = 0; neuron < dW.length; neuron++) {
			for (int feature = 0; feature< dW[neuron].length; feature++) {
				dW[neuron][feature] = d_down_layer[feature] * d_activator[neuron] * d_upper_layer[neuron];
			}
		}
		
		return dW;
	}
	
	/**
	 * 
	 * @param d_activator
	 * @param d_upper_layer
	 * @return
	 */
	public static float[] compute_d_bias(float[] d_activator, float[] d_upper_layer) {
		
		assert d_activator.length == d_upper_layer.length;
		
		float[] d_bias = new float[d_activator.length];
		
		for (int i = 0; i < d_bias.length; i++) {
			d_bias[i] = d_activator[i] * d_upper_layer[i];
		}
		
		return d_bias;
	}
	
	/**
	 * 
	 * @param weights
	 * @param d_activator
	 * @param d_upper_layer
	 * @return
	 */
	public static float[] compute_d_layers(float[][] weights,
			float[] d_activator,
			float[] d_upper_layer) {
		
		assert d_upper_layer.length == d_activator.length: "";
		assert d_upper_layer.length == weights.length;
		
		float[] d_current_layer = new float[weights[0].length];
		
		for (int i = 0; i < d_current_layer.length; i++) {
			float sum = 0;
			for (int j = 0; j < d_upper_layer.length; j++) {
				sum += weights[j][i] * d_activator[j] * d_upper_layer[j];
			}
			d_current_layer[i] = sum;
		}
		return d_current_layer;
	}
	
	
	
	/**
	 * 
	 * The above have been coded !!!
	 * 
	 * The below not!!
	 * 
	 */
	
	
	
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
		float[] d_o = new float[output.length];
		
		// Compute dC / d_output
		
		
		
		// Compute dC / d_W dense layers backward.
		
		// Compute dC / d_b dense layers backward.
		
		// Compute dC / da_l dense layers backward.
		
				
	}
	
	/**
	 * Train through all epochs according to n_iter specified in the __init__() method
	 */
	public void train () {
		for (int _ = 0; _ < this.n_iter; _++) {
//			this.evaluate_step();
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
		assert input.length == CARD_FEATURE : "Wrong input size";
		
//		Object[] result = evaluate_step(input, states)
		
		return null;
	}
	
	/**
	 * 
	 * Debugging method
	 * 
	 */
	
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
	 */
	@SuppressWarnings("unused")
	private void print_weights () {
//		System.out.print("Weights 1: \n");
//		print_mat2D(this.weights1);
//		System.out.print("Bias1: \n");
//		print_mat1D(this.bias1);
//		System.out.print("\nWeights2: \n");
//		print_mat2D(this.weights2);
//		System.out.print("Bias2: \n");
//		print_mat1D(this.bias2);
//		System.out.println();
	}
	
	@SuppressWarnings("unused")
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
	
	public static void main(String[] args) {
		
	}

}
