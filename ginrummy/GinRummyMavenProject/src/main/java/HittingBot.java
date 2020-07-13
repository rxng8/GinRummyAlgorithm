import java.util.ArrayList;

/**
 * 
 * @author alexv
 * Class for collecting data to process hitting operations
 */
public class HittingBot {
	
	private static final boolean VERBOSE = false;
	
	HandEstimator2 estimator;
	
	// Known card in the opponent hand
	boolean[] op_known;
	
	// other known cards that are in the discard pile
	boolean[] discard_known;
	
	// other known cards that are in hand
	boolean[] hand;
	
	// current player hand
	ArrayList<Card> cards;
	
	public HittingBot(HandEstimator2 estimator) {
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
