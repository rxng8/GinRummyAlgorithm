package util;
import core.*;
import module.*;
/**
 * @author Alex Nguyen
 * Gettysburg College
 * 
 * Advisor: Professor Neller.
 */

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Iterator;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;



public class dl4jtest {

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
	public dl4jtest () {
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
	public dl4jtest (String filename) {
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
	public void __init__(
			int seed,
			float lr,
			int n_iter) {
		this.X = new ArrayList<>();
		this.Y = new ArrayList<>();
		preprocess_data();

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
	
	public ComputationGraph buildModel () {
		ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
			    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
			    .updater(new Sgd(this.lr))
			    .graphBuilder()
			    .addInputs("trainFeatures")
			    .setOutputs("predictMortality")
			    .addLayer("L1", new GravesLSTM.Builder()
			        .nIn(52)
			        .nOut(300)
			        .activation(Activation.SOFTSIGN)
			        .weightInit(WeightInit.DISTRIBUTION)
			        .build(), "trainFeatures")
			    .addLayer("predictMortality", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
			        .activation(Activation.SOFTMAX)
			        .weightInit(WeightInit.DISTRIBUTION)
			        .nIn(300).nOut(52).build(),"L1")
			    .pretrain(false).backprop(true)
			    .build();
		return new ComputationGraph(conf);
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
		dl4jtest obj = new dl4jtest();
		obj.__import_data__("dataset/validation_data.dat");
		obj.__init__(10, 1e-3f, 10);
		
		int max_match = obj.X.size();
		
		int match_from = 345;
		int match_to = 346;
		assert match_from < match_to : "Duhhhh!!";
		int length = match_to - match_from;
		assert match_to < max_match : "Invalid ending match index!";
	
		try {
//			String file_name = "lstm_200_200epoch";
			String file_name = "lstm_100_500epoch";
			String modelJson = new ClassPathResource("src/main/resources/model/" + file_name + "_config.json").getFile().getPath();
//			ComputationGraphConfiguration modelConfig = KerasModelImport.importKerasModelConfiguration(modelJson);
			
			String modelWeights = new ClassPathResource("src/main/resources/model/" + file_name + "_weights.h5").getFile().getPath();
			ComputationGraph network = KerasModelImport.importKerasModelAndWeights(modelJson, modelWeights);
			
			
			for (int match = 0; match < length; match++) {
				
				System.out.println("Match " + (match + match_from));
				
				ArrayList<float[][]> xdatum = obj.X.get(match_from + match);
				ArrayList<float[]> ydatum = obj.Y.get(match_from + match);
				
				assert xdatum.size() == ydatum.size() : "Number of turns in input and output must match!";
				
				int n_turns = xdatum.size();
				
				if (n_turns == 0 || n_turns == 1) {
					System.out.println("This match some player goes gin right from the start, not taking any turn, omitting, duhhh");
					continue;
				}
				
				float[] dx0 = new float[n_turns * 52];
				float[] dx1 = new float[n_turns * 52];
				float[] dx2 = new float[n_turns * 52];
				float[] dx3 = new float[n_turns * 52];
				float[][] dy = new float[n_turns][52];
				
				for (int turn = 0; turn < n_turns; turn++) {
					for (int feature = 0; feature < 52; feature ++) {
						dx0[feature + turn * 52] = xdatum.get(turn)[0][feature];
						dx1[feature + turn * 52] = xdatum.get(turn)[1][feature];
						dx2[feature + turn * 52] = xdatum.get(turn)[2][feature];
						dx3[feature + turn * 52] = xdatum.get(turn)[3][feature];
					}
					dy[turn] = ydatum.get(turn).clone();
				}
				
//				System.out.println("Get here");
				// Debug
//				System.out.println(x0.toString());
//				Iterator<float[][]> it = obj.X.get(390).iterator();
//				while(it.hasNext()) {
//					float[] features = it.next()[0];
//					print_mat1D_card(features, "test");
//				}
				
				INDArray x0 = Nd4j.create(dx0, new int[] {n_turns, 52}, 'c');
				INDArray x1 = Nd4j.create(dx1, new int[] {n_turns, 52}, 'c');
				INDArray x2 = Nd4j.create(dx2, new int[] {n_turns, 52}, 'c');
				INDArray x3 = Nd4j.create(dx3, new int[] {n_turns, 52}, 'c');

				INDArray[] X = {x0, x1, x2, x3};
				
				INDArray[] y_pred_ind = network.output(X);
				
				// Debug
//				System.out.println(y_pred_ind[0].shapeInfoToString());
				
				float[][] y_pred = y_pred_ind[0].toFloatMatrix();
				for (int turn = 0; turn < y_pred.length; turn++) {
					System.out.println("Turn " + turn);
					print_mat1D_card(y_pred[turn], "Predicted");
					print_mat1D_card(dy[turn], "Actual");
					System.out.println("Acc: " + HandEstimator2.cal_accuracy(dy[turn], y_pred[turn]));
				}
			}
			

		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException | InvalidKerasConfigurationException | UnsupportedKerasConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
