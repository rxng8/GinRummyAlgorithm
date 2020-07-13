import java.util.ArrayList;

public class PlayerGameState {
	// Known card in the opponent hand
	boolean[] op_known;
	
	// other known cards that are in the discard pile
	boolean[] discard_known;
	
	// other known cards that are in hand
	boolean[] hand;
	
	// current player hand
	ArrayList<Card> cards;
	
	public PlayerGameState() {}
	
	public void init () {
		op_known = new boolean[52];
		discard_known = new boolean[52];
		hand = new boolean[52];
		cards = new ArrayList<>();
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
}
