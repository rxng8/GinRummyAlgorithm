import ai.djl.*;
import ai.djl.nn.*;
import ai.djl.nn.core.*;
import ai.djl.training.*;
import java.nio.file.*;

import ai.djl.ndarray.types.*;
import ai.djl.training.dataset.*;
import ai.djl.training.initializer.*;
import ai.djl.training.loss.*;
import ai.djl.training.listener.*;
import ai.djl.training.evaluator.*;
import ai.djl.training.optimizer.*;
import ai.djl.training.util.*;

public class LinearNet {
	
	long inputSize = 52;
	long outputSize = 1;
	
	public LinearNet() {

		SequentialBlock block = new SequentialBlock();
		
		block.add(Blocks.batchFlattenBlock(inputSize));
		block.add(Linear.builder().setOutChannels(128).build());
		block.add(Activation::relu);
		block.add(Linear.builder().setOutChannels(64).build());
		block.add(Activation::relu);
		block.add(Linear.builder().setOutChannels(outputSize).build());

	}
	public static void main(String[] args) {
		
	}
}
