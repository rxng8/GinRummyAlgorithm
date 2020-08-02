import java.util.ArrayList;

public class HandEstimator3 extends HandEstimatorGeneral{

	private static final boolean VERBOSE = false;

	private static final float IGNORE_THRESHOLD_RATIO = 0.5f;
	
	private static final float PREV_TURN_IMPORTANT_POWER = 0.1f;
	
	// Known card in the opponent hand
	boolean[] op_known;
	
	// other known cards that are not in op hand
	boolean[] other_known;
	
	// prob in the opponent hand
	double[] probs;
	
	// Total unknown
	int n_unknown;
	
	public HandEstimator3() {}
	
	
	public void init() {
		op_known = new boolean[52];
		other_known = new boolean[52];
		probs = new double[52];
		
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
	private static double prob_card (double prob_bayes_draw, double prob_bayes_discard) {
		return prob_bayes_draw * prob_bayes_discard;
	}
	
	@SuppressWarnings("unused")
	private static double[] prob_card (double[] prob_bayes_draw, double[] prob_bayes_discard) {
		double[] y_card = new double[52];
		for (int i = 0; i < y_card.length; i++) {
			y_card[i] = prob_card(prob_bayes_draw[i], prob_bayes_discard[i]);
		}
		return y_card;
	}
	
	
	

	private static double prob_bayes_discard(double meld_prop, double to_prop) {
		return meld_prop * Math.pow(to_prop, PREV_TURN_IMPORTANT_POWER) * 11;
	}
	
	private static double[] prob_bayes_discard(double[] meld_prop, double[] to_prop) {
		assert meld_prop.length == to_prop.length;
		double[] y = new double[meld_prop.length];
		for (int i = 0; i < meld_prop.length; i++) {
			y[i] = prob_bayes_discard(meld_prop[i], to_prop[i]);
		}
		return y;
	}

	private static double prob_bayes_draw(double meld_prop, double to_prop) {
		return meld_prop * Math.pow(to_prop, PREV_TURN_IMPORTANT_POWER) * 2;
	}
	
	private static double[] prob_bayes_draw(double[] meld_prop, double[] to_prop) {
		assert meld_prop.length == to_prop.length;
		double[] y = new double[meld_prop.length];
		for (int i = 0; i < meld_prop.length; i++) {
			y[i] = prob_bayes_draw(meld_prop[i], to_prop[i]);
		}
		return y;
	}
	
	@SuppressWarnings("unused")
	private double[] get_meld_draw(Card op_draw, boolean drawn) {
		double[] y = new double[52];
		if (drawn) {
			for (int i = 0; i < y.length; i++) {
				y[i] = (1.0/n_unknown);
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
				y[i] = (1.0/n_unknown);
			}
			y[op_draw.getId()] = (1.0/n_unknown * IGNORE_THRESHOLD_RATIO);
			// mask 1 around draw card
			int rank = op_draw.getRank();
			int suit = op_draw.getSuit();
			
			for (int r = rank - 2; r <= rank + 2; r++) {
				if (r < 0) continue;
				if (r > 12) continue;
				y[new Card(r, suit).getId()] = (1.0/n_unknown * IGNORE_THRESHOLD_RATIO);
			}
			
			for (int s = suit - 3; s <= suit + 3; s++) {
				if (s < 0) continue;
				if (s > 3) continue;
				y[new Card(rank, s).getId()] = (1.0/n_unknown * IGNORE_THRESHOLD_RATIO);
			}
		}
		return y;
	}
	
	private double[] get_meld_discard(Card op_discard) {
		
		if (op_discard == null) {
			return new double[52];
		}
		
		double[] y = new double[52];
		
		for (int i = 0; i < y.length; i++) {
			y[i] = (1.0/n_unknown);
		}
		
		y[op_discard.getId()] = (1.0/n_unknown * IGNORE_THRESHOLD_RATIO);
		
		// mask 1 around discard card
		int rank = op_discard.getRank();
		int suit = op_discard.getSuit();
		
		for (int r = rank - 2; r <= rank + 2; r++) {
			if (r < 0) continue;
			if (r > 12) continue;
			y[new Card(r, suit).getId()] = (1.0/n_unknown * IGNORE_THRESHOLD_RATIO);
		}
		
		for (int s = suit - 3; s <= suit + 3; s++) {
			if (s < 0) continue;
			if (s > 3) continue;
			y[new Card(rank, s).getId()] = (1.0/n_unknown * IGNORE_THRESHOLD_RATIO);
		}
		
		return y;
	}
	
	
	/**
	 * https://en.wikipedia.org/wiki/Feature_scaling
	 * @param mat
	 * @return
	 */
	@SuppressWarnings("unused")
	private static double[] normalize(double[] mat, String method) {
		// 84.35 % and vary ( 60 - 80)
		if (method.equals("maxmin")) {
			double[] y = new double[mat.length];
			
			double min = Float.MAX_VALUE;
			double max = Float.MIN_VALUE;
			
			// Get max min
			for(int i = 0; i < mat.length; i++) {
				if (mat[i] > max) {
					max = mat[i];
				}
				if (mat[i] < min) {
					min = mat[i];
				}			
			}
			
			double distance = max - min;
			
			for(int i = 0; i < mat.length; i++) {
				y[i] = (mat[i] - min) / distance;	
			}
			
			return y;
		} 
		
		// 82.67 % acc
		else if (method.equals("probone")) {
			double normalizing_sum = 0;
			double[] y = new double[mat.length];
			
			for (int i = 0; i < y.length; i++) {
				double prob = mat[i];
				if (prob < 0) {
					prob = 0;
				}
				normalizing_sum += prob;
			}
			
			
			for (int i = 0; i < y.length; i++) {
				if (normalizing_sum != 0) {
					y[i] = (Math.max(0.0, mat[i]) / normalizing_sum);
				} else {
					y[i] = (1.0 / 42);
				}
			}
			
			return y;
		} 
		// 60 % acc due to out side 1
		else if (method.equals("standard_scale")) {
			double mean = Maths.mean(mat);
			double std = Maths.std(mat);
			
			double[] y = new double[mat.length];
			for(int i = 0; i < y.length; i++) {
				y[i] = (mat[i] - mean) / std;
			}
			return y;
		} else {
			return mat;
		}
	}
	
	/**
	 * Default normalize method min-max scaling
	 * https://en.wikipedia.org/wiki/Feature_scaling
	 * @param probs2
	 * @return
	 */
	@SuppressWarnings("unused")
	private static double[] normalize(double[] probs2) {
		double normalizing_sum = 0;
		double[] y = new double[probs2.length];
		
		for (int i = 0; i < y.length; i++) {
			double prob = probs2[i];
			if (prob < 0) {
				prob = 0;
			}
			normalizing_sum += prob;
		}
		
		
		for (int i = 0; i < y.length; i++) {
			if (normalizing_sum != 0) {
				y[i] = (Math.max(0.0, probs2[i]) / normalizing_sum);
			} else {
				y[i] = (1.0 / 42);
			}
		}
		
		return y;
	}
	
	/**
	 * Default normalize method
	 * https://en.wikipedia.org/wiki/Feature_scaling
	 * @param mat
	 * @return
	 */
	@SuppressWarnings("unused")
	private static double[] normalize(double[] mat, int turn) {
		if (turn == 0) {
			return mat;
		}
		float threshold = 1 - (1.0f / turn);
//		System.out.println("Turn : " + turn + " ; Threshold: " + threshold);
		double[] y = new double[52];
		double max = Float.MIN_VALUE;
		// Get max min
		for(int i = 0; i < mat.length; i++) {
			if (mat[i] > max) {
				max = mat[i];
			}		
		}
//		System.out.println(" ; ratio: " + threshold / max);
		for(int i = 0; i < mat.length; i++) {
			y[i] = mat[i] * (threshold / max);
//			System.out.println("Before: " + mat[i]);
//			System.out.println("After: " + mat[i] * (threshold / max));
		}

		return y;
	}
	
	
	/**
	 * Return the probability of opponent meld
	 * @return
	 */
	public int get_op_nmeld() {
		// Hard code
		int count = 1;
		for (boolean b : this.op_known) {
			if (b) count++;
		}
		return count;
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

	@Override
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
		this.probs = normalize(this.probs);
		
		// Get masking melding
		double[] meld_draw = get_meld_draw(faceUpCard, drawn);
		double[] meld_discard = get_meld_discard(discardedCard);
		
//		if (VERBOSE ) {
//			print_mat(meld_draw, "Masking drawing meld");
//			print_mat(meld_discard, "Masking discarding meld");
//		}
		
		double[] bayes_draw = prob_bayes_draw(meld_draw, this.probs);
		double[] bayes_discard = prob_bayes_discard(meld_discard, this.probs);
		
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
	
	/**
	 * Get set
	 * @return
	 */
	
	@Override
	public double[] getProb() {
		return this.probs;
	}

	@Override
	public boolean[] getKnown() {
		return this.op_known;
	}
	
	@Override
	public void setKnown(ArrayList<Card> cards, boolean held) {
		if(held) 
			for(Card c : cards)
				op_known[c.getId()] = true;
		else
			for(Card c : cards)
				other_known[c.getId()] = true;

		// Recalculate n_unknown card
		re_calculate_unknown();
		// Remask
		re_mask_prob();
		// normalize
		this.probs = normalize(this.probs);
	}
	
	
	public static float cal_accuracy(ArrayList<Card> op_hand, float[] pred) {
		
		float[] op_mat = new float[pred.length];
		
		for (Card c : op_hand) {
			op_mat[c.getId()] = 1;
		}
		
		float sum = 0;
		for (int i = 0; i < op_mat.length; i++) {
			sum += 1 - Math.abs(op_mat[i] - pred[i]);
		}
		
		return sum / 52;
	}
	
	public static float cal_accuracy(float[] op_mat, float[] pred) {
			
		float sum = 0;
		for (int i = 0; i < op_mat.length; i++) {
			sum += 1 - Math.abs(op_mat[i] - pred[i]);
		}
		
		return sum / 52;
	}
	
	
	public void view() {
		Util.print_mat(this.op_known, "Cards that are exactly in opponent's hand");
		Util.print_mat(this.other_known, "Cards that are in my hand or discard pile");
		Util.print_mat(this.probs, "Probability of opponent's hand");
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
	}
}
