import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

public class KnockerBot {
	
	public static final float THRESHOLD = 0.95f;
	
	static ComputationGraph network;
	static {
		try {
			String file_name = "knocking_100";
			String modelJson = new ClassPathResource("./model/" + file_name + "_config.json").getFile().getPath();
//			ComputationGraphConfiguration modelConfig = KerasModelImport.importKerasModelConfiguration(modelJson);
			
			String modelWeights = new ClassPathResource("./model/" + file_name + "_weights.h5").getFile().getPath();
			ComputationGraph network = KerasModelImport.importKerasModelAndWeights(modelJson, modelWeights);
			KnockerBot.network = network;
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException | InvalidKerasConfigurationException | UnsupportedKerasConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	public KnockerBot() {

	}
	
	/**
	 * Input : turn, deadwood, n_meld
	 * Output: Knock or not
	 * @param args
	 */
	public float predict(int[] X) {
		
		float[] arr = new float[X.length];
		for (int i = 0; i < X.length; i++) {
			arr[i] = (float) X[i];
		}
		
		INDArray input = Nd4j.create(arr, new int[] {1, arr.length}, 'c');
		
		INDArray[] out = network.output(input);
		
		float out_value = out[0].getFloat(0, 0);
		
		return out_value;
	}
	
	public static void main(String[] args) {
		KnockerBot k = new KnockerBot();
		int[] X = {25, 10, -1};
		System.out.println(k.predict(X));
	}
}
