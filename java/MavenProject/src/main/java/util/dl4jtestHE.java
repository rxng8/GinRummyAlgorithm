package util;
import core.*;
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

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;



public class dl4jtestHE {
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
	public dl4jtestHE (String weights, String bias) {
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
	public dl4jtestHE () {
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
			this.weights1 = new float[X[0].length][64];
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
		
//		int data_size = this.gamePlays.size()
//				* this.gamePlays.get(0).size()
//				* this.gamePlays.get(0).get(0).size()
//				* this.gamePlays.get(0).get(0).get(0).size();
//		int data_feature = this.gamePlays.get(0).get(0).get(0).get(0)[0].length;
		int data_feature = 52;
//		assert data_feature == this.gamePlays.get(0).get(0).get(0).get(0)[1].length : "Input and output does not match dimension";
		
//		System.out.println("Data size: " + data_size);
		
		Iterator<ArrayList<ArrayList<ArrayList<short[][]>>>> it_games = this.gamePlays.iterator();
		int i = 0;
		while(it_games.hasNext()) {
			Iterator<ArrayList<ArrayList<short[][]>>> it_players = it_games.next().iterator();
			while (it_players.hasNext()) {
				Iterator<ArrayList<short[][]>> it_rounds = it_players.next().iterator();
				while (it_rounds.hasNext()) {
					ArrayList<short[][]> turns = it_rounds.next();
					i += turns.size();
				}
			}
		}
		
		this.X = new float[i][data_feature * 4];
		this.Y = new float[i][data_feature];
		
		it_games = this.gamePlays.iterator();
		i = 0;
		while(it_games.hasNext()) {
			Iterator<ArrayList<ArrayList<short[][]>>> it_players = it_games.next().iterator();
			while (it_players.hasNext()) {
				Iterator<ArrayList<short[][]>> it_rounds = it_players.next().iterator();
				while (it_rounds.hasNext()) {
					Iterator<short[][]> it_turns = it_rounds.next().iterator();
					while (it_turns.hasNext()) {
						short[][] turnData = it_turns.next();
						assert turnData.length == 5 : "Wrong data form!";
						for (int j = 0; j < turnData[0].length; j++) {
							for (int k = 0; k < 4; k++) {
								X[i][4 * k + j] = (float) turnData[k][j];
							}
							
							Y[i][j] = (float) turnData[4][j];
						}
						i++;
					}
				}
			}
		}
	}
	
	
	
	/**
	 * Predict an input by using its current weights and bias matrix.
	 * @param input (float[]): input.
	 * @return (float[]): output.
	 */
	public float[] predict (float[] input) {
		return null;
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
		for (int i = 0; i < 52; i++) {
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
		
		dl4jtestHE obj = new dl4jtestHE();
		obj.__import_data__("play_data_SimplePlayer.dat");
		obj.__init__(10, 1e-3f, 10, true);
		
		int from = 390;
		int to = 400;
		float[][] testX = new float[to - from][obj.X[0].length];
		for (int i = 0; i < to - from; i++) {
			for (int j = 0; j < obj.X[i].length; j++) {
				testX[i][j] = obj.X[i][j];
			}
		}
		
		float[][] testY = new float[to - from][obj.Y[0].length];
		for (int i = 0; i < to - from; i++) {
			for (int j = 0; j < obj.Y[i].length; j++) {
				testY[i][j] = obj.Y[i][j];
			}
		}
		
		try {
			String modelJson = new ClassPathResource("src/main/resources/model/model_config.json").getFile().getPath();
//			ComputationGraphConfiguration modelConfig = KerasModelImport.importKerasModelConfiguration(modelJson);
			
			String modelWeights = new ClassPathResource("src/main/resources/model/model_weights.h5").getFile().getPath();
			ComputationGraph network = KerasModelImport.importKerasModelAndWeights(modelJson, modelWeights);
			
			
			
			for (int i = 0; i < to - from; i++) {
				System.out.println("Predict: ");
//				print_mat1D_card(testX[i], "Input");
				INDArray X = Nd4j.create(testX[i]);
				INDArray[] predicted = network.output(X);
				print_mat1D_card(predicted[0].toFloatVector(), "Predicted");
				print_mat1D_card(testY[i], "Actual");
				
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
