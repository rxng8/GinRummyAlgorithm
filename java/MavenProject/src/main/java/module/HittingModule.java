package module;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.util.HashSet;

import core.*;
import util.*;
/**
 * 
 * @author Alex Nguyen
 * Class for collecting data to process hitting operations
 */

/**
 * 
 * TODO: Bug: count hit meld need to be set a ceiling to limit the value of having more hit value
 *
 */

public class HittingModule extends Module {
	static ComputationGraph network;
	static {
		try {
			String file_name = "hit_100_v6";
			String modelJson = new ClassPathResource("./model/" + file_name + "_config.json").getFile().getPath();
//			ComputationGraphConfiguration modelConfig = KerasModelImport.importKerasModelConfiguration(modelJson);
			
			String modelWeights = new ClassPathResource("./model/" + file_name + "_weights.h5").getFile().getPath();
			ComputationGraph network = KerasModelImport.importKerasModelAndWeights(modelJson, modelWeights);
			HittingModule.network = network;
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException | InvalidKerasConfigurationException | UnsupportedKerasConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private static final boolean VERBOSE = false;
	
	HandEstimator2 estimator;
	
	// Known card in the opponent hand
	boolean[] op_known;
	
	// other known cards that are in the discard pile
	boolean[] discard_known;
	
	// other known cards that are in hand
	boolean[] hand;
	
	// current player hand
	// ArrayList<Card> cards;
	
	public HittingModule() {
		
	}
	
	public HittingModule(HandEstimator2 estimator) {
		this.estimator = estimator;
	}
	
	public void init() {
		op_known = new boolean[52];
		discard_known = new boolean[52];
	}
	
	/**
	 * Input : turn, rank, isHittingCard, expected turn before end
	 * Output: The card value.
	 * @param X
	 */
	public float predict(int[] X) {
		
		assert X.length == 4 : "Wrong features system";
		
		float[] arr = new float[X.length];
		for (int i = 0; i < X.length; i++) {
			arr[i] = (float) X[i];
		}
		
		INDArray input = Nd4j.create(arr, new int[] {1, arr.length}, 'c');
		
		INDArray[] out = network.output(input);
		
		float out_value = out[0].getFloat(0, 0);
		
		return out_value;
	}
	
	public boolean isHittingCard(Card c1, Card c2) {
		
		for (ArrayList<Card> meld : GinRummyUtil.cardsToAllMelds(get_availability())) {
			if (meld.size() < 4 && meld.contains(c1) && meld.contains(c2)) {
				return true;
			}
		}
		return false;
	}
	
	public boolean isHittingCard(ArrayList<Card> hand, Card c2) {
		ArrayList<ArrayList<Card>> melds = GinRummyUtil.cardsToAllMelds(get_availability());
		for (Card c1 : hand) {
			for (ArrayList<Card> meld : melds) {
				if (meld.size() < 4 && meld.contains(c1) && meld.contains(c2)) {
					return true;
				}
			}
		}
		return false;
	}
	
	/**
	 * Read the code
	 * @param hand
	 * @param c2
	 * @return
	 */
	public int countHitMeldType(ArrayList<Card> hand, Card c2) {
		int count = 0;
		
		// Everyone can just take maximum point for at most 1 suit and 1 json
		
		// checksuit wild card left
		boolean checkSuit = true;
		
		// Check JSON wild card left
		boolean checkJSON = true;
		
		ArrayList<ArrayList<Card>> melds = GinRummyUtil.cardsToAllMelds(get_availability());
		for (Card c1 : hand) {
			for (ArrayList<Card> meld : melds) {
				
				// Consider if it is a potential suit or json
				boolean suited = isSuit(meld);
				
				// If it's a potential suit and still have suit wildcard left
				if (meld.size() < 4 && meld.contains(c1) && meld.contains(c2) && suited && checkSuit) {
					checkSuit = false;
					count ++;
				} 
				
				// If it's a potential json and still have json wildcard left
				else if (meld.size() < 4 && meld.contains(c1) && meld.contains(c2) && !suited && checkJSON) {
					checkJSON = false;
					count ++;
				}
				
				// If no wildcard left, break for performance
				if (!checkSuit && !checkJSON) break;
				
			}
		}
//		System.out.println("Meld hit count : " + count);
		return count;
	}
	
	private boolean isSuit(ArrayList<Card> meld) {
		if (meld.get(0).rank == meld.get(1).rank) return true;
		return false;
	}
	
	
	/**
	 * return the number of possible hitting melds associated with this card considering melds in hand
	 * @param hand
	 * @param c2
	 * @return
	 */
	public int countHitMeld(ArrayList<Card> hand, Card c2) {
		int count = 0;
		
		ArrayList<ArrayList<Card>> melds = GinRummyUtil.cardsToAllMelds(get_availability());
		for (Card c1 : hand) {
			for (ArrayList<Card> meld : melds) {
				if (meld.size() < 4 && meld.contains(c1) && meld.contains(c2)) {
//					System.out.println("MELDLWELFLEF" + meld);
					count ++;
				}
			}
		}
//		System.out.println("Meld hit count : " + count);
		return count;
	}
	
	public int count_hitting(ArrayList<Card> hand) {
		
		return get_hitting(hand).size();
	}
	
	@SuppressWarnings("unchecked")
	public HashSet<Card> get_hitting(ArrayList<Card> hand) {
		
		// Get all meld sets
		ArrayList<ArrayList<ArrayList<Card>>> meldSet = GinRummyUtil.cardsToBestMeldSets(hand);
		
		// Construct remaining cards list
		ArrayList<Card> cards;
		
		if (meldSet.size() == 0) {
			if (VERBOSE)
			System.out.println("This turn the player does not have any meld, here is the hand: " + hand);
			cards = (ArrayList<Card>) hand.clone();
		} else {
			ArrayList<ArrayList<Card>> handmelds = meldSet.get(0);
			if (VERBOSE) {
				System.out.println("Hand: " + hand);
				System.out.println("Melds: " + handmelds);
			}
			
			cards = Util.get_unmelded_cards(handmelds, hand);
			
			if (VERBOSE)
			System.out.println("Unmelded Cards: " + cards);
		}
		
		ArrayList<ArrayList<Card>> melds = GinRummyUtil.cardsToAllMelds(get_availability());
		HashSet<Card> result = new HashSet<>();
		ArrayList<Card> rm_cards = (ArrayList<Card>) cards.clone();

		if (VERBOSE)
		System.out.println("count_hitting debug: unmelded cards size: " + rm_cards.size());

		for (Card c1 : rm_cards) {
			ArrayList<Card> tmp_cards = (ArrayList<Card>) rm_cards.clone();
			tmp_cards.remove(c1);
			for (Card c2 : tmp_cards) {
				
				for (ArrayList<Card> meld : melds) {
					if (meld.size() < 4 && meld.contains(c1) && meld.contains(c2)) {
						result.add(c1);
						result.add(c2);
					}
				}
			}
		}
		return result;
	}
	

	@SuppressWarnings("unchecked")
	public boolean isMeld(ArrayList<Card> hand, Card c) {
		ArrayList<Card> newCards = (ArrayList<Card>) hand.clone();
		newCards.add(c);
		for (ArrayList<Card> meld : GinRummyUtil.cardsToAllMelds(newCards)) {
			if (meld.contains(c)) {
				return true;
			}	
		}
		return false;
	}
	
	public void setOpKnown(Card card, boolean b) {
//		if (VERBOSE) if (b) System.out.println("The opponent has the card " + card.toString());
		op_known[card.getId()] = b;
	}

	public void setOpKnown(ArrayList<Card> hand, boolean b) {
		for (Card c : hand) {
			setOpKnown(c, b);
		}
	}
	
	public void setDiscardKnown(Card card, boolean b) {
//		if (VERBOSE) if (b) System.out.println("The opponent has the card " + card.toString());
		discard_known[card.getId()] = b;
	}
	
	public void setDiscardKnown(ArrayList<Card> hand, boolean b) {
		for (Card c : hand) {
			setDiscardKnown(c, b);
		}
	}
	
	public void reportDrawDiscard(Card faceUpCard, boolean drawn, Card discardedCard, int turn) {
//		if (VERBOSE) System.out.println("The opponent " + (b ? "" : "does not ") +"draw card " + faceUpCard.toString() + " and discard " + (discardedCard == null ? "null" : discardedCard.toString()));
		// Set known everything
		setOpKnown(faceUpCard, drawn);
		setDiscardKnown(faceUpCard, !drawn);
		setDiscardKnown(discardedCard, discardedCard != null);
	}
	
	public ArrayList<Card> get_availability() {
		ArrayList<Card> list = new ArrayList<Card>();
		
		for (int i = 0; i < this.op_known.length; i++) {
			 if (!this.op_known[i]) list.add(Card.getCard(i));
		}
		
		for (int i = 0; i < this.discard_known.length; i++) {
			 if (!this.discard_known[i]) list.add(Card.getCard(i));
		}
		
		return list;
	}
	
	public ArrayList<Card> get_op_hand_absolute() {
		ArrayList<Card> op = new ArrayList<>();
		for (int i = 0; i < this.op_known.length; i++) {
			 if (this.op_known[i]) op.add(Card.getCard(i));
		}
		return op;
	}
	
	public int get_n_op_pick() {
		int count = 0;
		for (int i = 0; i < this.op_known.length; i++) {
			 if (this.op_known[i]) count++;
		}
		return count;
	}
	

	public ArrayList<Card> get_availability_probs() {
		
		ArrayList<Card> list = new ArrayList<Card>();
		
		for (int i = 0; i < this.op_known.length; i++) {
			 if (!this.op_known[i]) list.add(Card.getCard(i));
		}
		
		for (int i = 0; i < this.discard_known.length; i++) {
			 if (!this.discard_known[i]) list.add(Card.getCard(i));
		}
		
		
		//Use some more data in estimaor here
		float[] probs_op = estimator.get_probs();
		
		// TODO: write algorithms
		
		

		
		return list;
	}
	
	public void print() {
		boolean[] wall = new boolean[52];
		for (int i = 0; i < wall.length; i++) {
			wall[i] = !(op_known[i] || discard_known[i]);
		}
		Util.print_mat(wall, "Card Availability (True value)");
	}
	
	public static void main(String[] args) {
		HittingModule h = new HittingModule();
		int[] X = {100, 7, 2};
		System.out.println(h.predict(X));
	}
	
}
