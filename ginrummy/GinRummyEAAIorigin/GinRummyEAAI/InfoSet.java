package ginrummy.GinRummyEAAIorigin.GinRummyEAAI;

import java.util.ArrayList;

public class InfoSet {
	/**
	 * The information sets are grouped by how a meld forms, for example:
	 * 		+ Collecting meld with the same rank, 1, 2, or 3 cards needed.
	 * 		+ Collecting meld by json, 1, 2, 3, or any number of cards needed.
	 */
	
	// Data Processing
	ArrayList<Double> strategy;
	ArrayList<Double> strategy_sum;
	ArrayList<Double> regret_sum;
	ArrayList<Double> cfvs;
	String repr;
	
	
	
}
