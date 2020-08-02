package module;

import java.util.ArrayList;
import org.nd4j.linalg.api.ops.impl.accum.CountZero;
import java.util.HashSet;
import collector.*;
import core.*;
import player.*;
import util.*;
/**
 * 
 * @author Alex Nguyen
 * Class for collecting data to process hitting operations
 */
public class HittingModule {
	
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
	
	public boolean isHittingCard(Card c1, Card c2) {
		
		for (ArrayList<Card> meld : GinRummyUtil.cardsToAllMelds(get_availability())) {
			if (meld.size() < 4 && meld.contains(c1) && meld.contains(c2)) {
				return true;
			}
		}
		return false;
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
	
}
