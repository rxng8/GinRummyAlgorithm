package collector;

/**
 * @author Alex Nguyen
 */

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import core.*;
import module.*;
import player.*;
import util.*;



public abstract class DataGrinder {
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
	
	/**
	 * Set whether or not there is to be printed output during gameplay.
	 * @param playVerbose whether or not there is to be printed output during gameplay
	 */
	public static void setPlayVerbose(boolean playVerbose) {
		DataGrinder.playVerbose = playVerbose;
	}
	
	// Needed methods
	public abstract void match(GinRummyPlayer p0, GinRummyPlayer p1, int numGames);
	public abstract int play(int startingPlayer);
	public abstract void to_CSV(String filename, boolean cont) throws IOException;
//	public abstract void collectData(int turnsTaken, int currentPlayer, Card faceUpCard, Card drawCard, Card discardCard, ArrayList<Card> hand, ArrayList<Card> opponentHand);
	public abstract void displayData();
	
}
