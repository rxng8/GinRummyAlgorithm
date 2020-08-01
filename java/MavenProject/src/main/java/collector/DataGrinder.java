package collector;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import core.*;
import module.*;
import player.*;
import util.*;

public class DataGrinder {
	/**
	 * Random number generator
	 */
	static final Random RANDOM = new Random();
	/**
	 * Hand size (before and after turn). After draw and before discard there is one extra card.
	 */
	static final int HAND_SIZE = 10;
	/**
	 * Whether or not to print information during game play
	 */
	static boolean playVerbose = false;
	/**
	 * Two Gin Rummy players numbered according to their array index.
	 */
	GinRummyPlayer[] players = new GinRummyPlayer[2];
	
	final int MAX_TURNS = 100; // TODO - have not determined maximum length legal gin rummy game; truncating if necessary 
	
//	public void to_CSV(String filename);

//	public void displayData();
}
