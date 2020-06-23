/**
 * @author Alex Nguyen
 * Gettysburg College
 * 
 * Advisor: Professor Neller.
 */

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class dl4jtest {
	public static final double LEARNING_RATE = 0.05;
	public static final int lstmLayerSize = 300;
	public static final int NB_INPUTS = 86;
	
	public dl4jtest(int numLabelClasses) {
		ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
		    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		    .updater(new Sgd(LEARNING_RATE))
		    .graphBuilder()
		    .addInputs("trainFeatures")
		    .setOutputs("predictMortality")
		    .addLayer("L1", new GravesLSTM.Builder()
		        .nIn(NB_INPUTS)
		        .nOut(lstmLayerSize)
		        .activation(Activation.SOFTSIGN)
		        .weightInit(WeightInit.DISTRIBUTION)
		        .build(), "trainFeatures")
		    .addLayer("predictMortality", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
		        .activation(Activation.SOFTMAX)
		        .weightInit(WeightInit.DISTRIBUTION)
		        .nIn(lstmLayerSize).nOut(numLabelClasses).build(),"L1")
		    .pretrain(false).backprop(true)
		    .build();
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		dl4jtest model = new dl4jtest(10);
	}

}
