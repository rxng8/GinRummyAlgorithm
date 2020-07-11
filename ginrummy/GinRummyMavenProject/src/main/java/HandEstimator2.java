import java.util.ArrayList;

public class HandEstimator2 {

	private static final boolean VERBOSE = true;

	private static final float IGNORE_THRESHOLD_RATIO = 0.1f;
	
	// Known card in the opponent hand
	boolean[] op_known;
	
	// other known cards that are not in op hand
	boolean[] other_known;
	
	// prob in the opponent hand
	float[] probs;
	
	// Total unknown
	int n_unknown;
	
	public HandEstimator2() {
		
	}
	
	public void init() {
		op_known = new boolean[52];
		other_known = new boolean[52];
		probs = new float[52];
		
		// Initialize number of unknown card
		re_calculate_unknown();
		
		// Initialize probability
		for (int i = 0; i < probs.length; i++) {
			probs[i] = (float) (1.0 / n_unknown);
		}
		
		re_mask_prob();
		
	}
	
	
	
	
	/**
	 * Bayes thinsg/ algorithms
	 */
	
	
	
	
	@SuppressWarnings("unused")
	private static float prob_card (float prob_bayes_draw, float prob_bayes_discard) {
		return prob_bayes_draw * prob_bayes_discard;
	}
	
	@SuppressWarnings("unused")
	private static float[] prob_card (float[] prob_bayes_draw, float[] prob_bayes_discard) {
		float[] y_card = new float[52];
		for (int i = 0; i < y_card.length; i++) {
			y_card[i] = prob_card(prob_bayes_draw[i], prob_bayes_discard[i]);
		}
		return y_card;
	}
	
	
	

	private static float prob_bayes_discard(float meld_prop, float to_prop) {
		return meld_prop * to_prop * 11;
	}
	
	private static float[] prob_bayes_discard(float[] meld_prop, float[] to_prop) {
		assert meld_prop.length == to_prop.length;
		float[] y = new float[meld_prop.length];
		for (int i = 0; i < meld_prop.length; i++) {
			y[i] = prob_bayes_discard(meld_prop[i], to_prop[i]);
		}
		return y;
	}

	private static float prob_bayes_draw(float meld_prop, float to_prop) {
		return meld_prop * to_prop * 2;
	}
	
	private static float[] prob_bayes_draw(float[] meld_prop, float[] to_prop) {
		assert meld_prop.length == to_prop.length;
		float[] y = new float[meld_prop.length];
		for (int i = 0; i < meld_prop.length; i++) {
			y[i] = prob_bayes_draw(meld_prop[i], to_prop[i]);
		}
		return y;
	}
	
	@SuppressWarnings("unused")
	private float[] get_meld_draw(Card op_draw, boolean drawn) {
		float[] y = new float[52];
		if (drawn) {
			for (int i = 0; i < y.length; i++) {
				y[i] = (float) (1.0/n_unknown);
			}
			y[op_draw.getId()] = 1;
			// mask 1 around draw card
			int rank = op_draw.getRank();
			int suit = op_draw.getSuit();
			for (int r = rank - 2; r <= rank + 2; r++) {
				if (r < 0) continue;
				if (r > 12) continue;
				y[new Card(r, suit).getId()] = 1;
			}
			
			for (int s = suit - 3; s <= suit + 3; s++) {
				if (s < 0) continue;
				if (s > 3) continue;
				y[new Card(rank, s).getId()] = 1;
			}
		} else {
			for (int i = 0; i < y.length; i++) {
				y[i] = (float) (1.0/n_unknown);
			}
			y[op_draw.getId()] = (float) (1.0/n_unknown * IGNORE_THRESHOLD_RATIO);
			// mask 1 around draw card
			int rank = op_draw.getRank();
			int suit = op_draw.getSuit();
			
			for (int r = rank - 2; r <= rank + 2; r++) {
				if (r < 0) continue;
				if (r > 12) continue;
				y[new Card(r, suit).getId()] = (float) (1.0/n_unknown * IGNORE_THRESHOLD_RATIO);
			}
			
			for (int s = suit - 3; s <= suit + 3; s++) {
				if (s < 0) continue;
				if (s > 3) continue;
				y[new Card(rank, s).getId()] = (float) (1.0/n_unknown * IGNORE_THRESHOLD_RATIO);
			}
		}
		return y;
	}
	
	private float[] get_meld_discard(Card op_discard) {
		
		if (op_discard == null) {
			return new float[52];
		}
		
		float[] y = new float[52];
		
		for (int i = 0; i < y.length; i++) {
			y[i] = (float) (1.0/n_unknown);
		}
		
		y[op_discard.getId()] = (float) (1.0/n_unknown * IGNORE_THRESHOLD_RATIO);
		
		// mask 1 around discard card
		int rank = op_discard.getRank();
		int suit = op_discard.getSuit();
		
		for (int r = rank - 2; r <= rank + 2; r++) {
			if (r < 0) continue;
			if (r > 12) continue;
			y[new Card(r, suit).getId()] = (float) (1.0/n_unknown * IGNORE_THRESHOLD_RATIO);
		}
		
		for (int s = suit - 3; s <= suit + 3; s++) {
			if (s < 0) continue;
			if (s > 3) continue;
			y[new Card(rank, s).getId()] = (float) (1.0/n_unknown * IGNORE_THRESHOLD_RATIO);
		}
		
		return y;
	}
	
	
	/**
	 * https://en.wikipedia.org/wiki/Feature_scaling
	 * @param mat
	 * @return
	 */
	@SuppressWarnings("unused")
	private static float[] normalize(float[] mat) {
		
//		float[] y = new float[mat.length];
//		
//		float min = Float.MAX_VALUE;
//		float max = Float.MIN_VALUE;
//		
//		// Get max min
//		for(int i = 0; i < mat.length; i++) {
//			if (mat[i] > max) {
//				max = mat[i];
//			}
//			if (mat[i] < min) {
//				min = mat[i];
//			}			
//		}
//		
//		float distance = max - min;
//		
//		for(int i = 0; i < mat.length; i++) {
//			y[i] = (mat[i] - min) / distance;	
//		}
//		
//		return y;
		
		
		float normalizing_sum = 0;
		float[] y = new float[mat.length];
		
		for (int i = 0; i < y.length; i++) {
			double prob = mat[i];
			if (prob < 0) {
				prob = 0;
			}
			normalizing_sum += prob;
		}
		
		
		for (int i = 0; i < y.length; i++) {
			if (normalizing_sum != 0) {
				y[i] = (float) (Math.max(0.0, mat[i]) / normalizing_sum);
			} else {
				y[i] = (float) (1.0 / 42);
			}
		}
		
		return y;
//		float[] y = new float[mat.length];
//		for (int i = 0; i < y.length; i++) {
//			y[i] = mat[i] * 10000;
//		}
//		
//		return y;
		
//		return mat;
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
	
	public void setOtherKnown(Card card, boolean b) {
//		if (VERBOSE) if (b) System.out.println("The opponent has the card " + card.toString());
		other_known[card.getId()] = b;
	}
	
	public void setOtherKnown(ArrayList<Card> hand, boolean b) {
		for (Card c : hand) {
			setOtherKnown(c, b);
		}
	}

	public void reportDrawDiscard(Card faceUpCard, boolean drawn, Card discardedCard) {
//		if (VERBOSE) System.out.println("The opponent " + (b ? "" : "does not ") +"draw card " + faceUpCard.toString() + " and discard " + (discardedCard == null ? "null" : discardedCard.toString()));
		// Set known everything
		setOpKnown(faceUpCard, drawn);
		setOtherKnown(faceUpCard, !drawn);
		setOtherKnown(discardedCard, discardedCard != null);
		
		// Remask
		re_mask_prob();
		// Recalculate n_unknown card
		re_calculate_unknown();
		// normalize
		normalize(this.probs);
		
		// Get masking melding
		float[] meld_draw = get_meld_draw(faceUpCard, drawn);
		float[] meld_discard = get_meld_discard(discardedCard);
		
//		if (VERBOSE ) {
//			print_mat(meld_draw, "Masking drawing meld");
//			print_mat(meld_discard, "Masking discarding meld");
//		}
		
		float[] bayes_draw = prob_bayes_draw(meld_draw, this.probs);
		float[] bayes_discard = prob_bayes_discard(meld_discard, this.probs);
		
//		if (VERBOSE ) {
//			print_mat(bayes_draw, "Bayes drawing meld");
//			print_mat(bayes_discard, "Bayes discarding meld");
//		}
		
		this.probs = prob_card(bayes_draw, bayes_discard);
		re_mask_prob();
		this.probs = normalize(this.probs);
	}
	
	private void re_mask_prob() {
		for (int i = 0; i < this.probs.length; i++) {
			if (this.op_known[i]) {
				this.probs[i] = 1;
			}
			if (this.other_known[i]) {
				this.probs[i] = 0;
			}
		}
	}
	
	
	private void re_calculate_unknown() {
		this.n_unknown = 0;
		for (int i = 0; i < this.probs.length; i++) {
			if (!this.op_known[i] && !this.other_known[i]) {
				this.n_unknown++;
			}
		}
//		System.out.println("Number of unknown cards: " + n_unknown);
	}
	
	public void view() {
		print_mat(this.op_known, "Cards that are exactly in opponent's hand");
		print_mat(this.other_known, "Cards that are in my hand or discard pile");
		print_mat(this.probs, "Probability of opponent's hand");
	}
	
	public static void print_mat(float[] mat, String name) {
		System.out.println(name + ": ");
		System.out.print("Rank");
		for (int i = 0; i < Card.NUM_RANKS; i++)
			System.out.print("\t" + Card.rankNames[i]);
		for (int i = 0; i < Card.NUM_CARDS; i++) {
			if (i % Card.NUM_RANKS == 0)
				System.out.printf("\n%s", Card.suitNames[i / Card.NUM_RANKS]);
			System.out.print("\t");
			System.out.printf("%1.1e", mat[i]);
//			System.out.printf("%.5f", mat[i]);
		}
		System.out.println();
		System.out.println();
	}
	
	public static void print_mat(boolean[] mat, String name) {
		System.out.println(name + ": ");
		System.out.print("Rank");
		for (int i = 0; i < Card.NUM_RANKS; i++)
			System.out.print("\t" + Card.rankNames[i]);
		for (int i = 0; i < Card.NUM_CARDS; i++) {
			if (i % Card.NUM_RANKS == 0)
				System.out.printf("\n%s", Card.suitNames[i / Card.NUM_RANKS]);
			System.out.print("\t");
//			System.out.printf("%2.4f", mat[i]);
			System.out.printf(mat[i] ? "TRUE" : "_");
		}
		System.out.println();
		System.out.println();
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
