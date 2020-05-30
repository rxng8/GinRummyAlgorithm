import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;
import java.util.Stack;
import java.io.FileWriter;
import java.io.IOException;


public class CFRPlayer implements GinRummyPlayer {
	private int playerNum;
	@SuppressWarnings("unused")
	private int startingPlayerNum;
	private ArrayList<Card> cards = new ArrayList<Card>();
	private Random random = new Random();
	private boolean opponentKnocked = false;
	Card faceUpCard, drawnCard; 
	ArrayList<Long> drawDiscardBitstrings = new ArrayList<Long>();
	
	
	int unknown_cards;
	HashMap<String, InfoSet> info_set_map;
	
	// Public State
	
	// Card Matrix
	// Discarded pile
//	Stack<Card> discardPile;
	// ArrayList of probabilities of unknown cards in draw pile
	ArrayList<Double> draw_pile;
	// ArrayList of probabilities expected cards in opponent hand.
	ArrayList<Double> op_cards;
	// Strategy of this player when discarding. Array probs of 52 
	ArrayList<Double> dis_strategy;
	
	@Override
	public void startGame(int playerNum, int startingPlayerNum, Card[] cards) {
		this.playerNum = playerNum;
		this.startingPlayerNum = startingPlayerNum;
		this.cards.clear();
		for (Card card : cards)
			this.cards.add(card);
		opponentKnocked = false;
		drawDiscardBitstrings.clear();
		
		unknown_cards = 42;
		
		info_set_map = new HashMap<String, InfoSet>();
		
//		discardPile = new Stack<>();
		draw_pile = new ArrayList<Double>();
		op_cards = new ArrayList<Double>();
		dis_strategy = new ArrayList<Double>();
		for (int i = 0; i < Card.NUM_CARDS; i++) {
			this.draw_pile.add(1.0/unknown_cards); // There is 1/42 chance of each card is in the draw pile.
			this.op_cards.add(1.0/unknown_cards); // There is 1/42 chance of each card is in the opponent hand.
			dis_strategy.add(1.0/52); // Discard strategy
		}
		
		// Set all the cards in hand for the possibility to exist in other piles is absolute 0.0
		for (Card c : this.cards) {
			op_cards.set(c.getId(), 0.0);
			draw_pile.set(c.getId(), 0.0);
		}
		
		
	}

	@Override
	public boolean willDrawFaceUpCard(Card card) {
		// Return true if max value function, false otherwise
		unknown_cards --;
//		discardPile.push(card);
		// Bait Draw Strategy
		
		// Collecting meld strategy
		
		this.faceUpCard = card;
		@SuppressWarnings("unchecked")
		ArrayList<Card> newCards = (ArrayList<Card>) cards.clone();
		newCards.add(card);
		
		// First, look at the card, and see how it rearrange the probability of my matrix
		// Base on those matrix, look to see how picking up the card rearrange the matrix
		// Lookahead tree, just look a head and pre-calculating things, don't have to evaluate actual matrix?
		// Only deciding, the data matrix evaluation is computed inside report() methods.
		// Or, simple, just use some vector as data to choose between 2 choices, it's easy com'on!
		
		op_cards.set(faceUpCard.getId(), 0.0);
		draw_pile.set(faceUpCard.getId(), 0.0);
		
		this.draw_pile = normalize(this.draw_pile);
		this.op_cards = normalize(this.op_cards);
		
		
		// Draft Function
		ArrayList<Card> allCards = new ArrayList<>();
		for (Card c : Card.allCards) {
			allCards.add(c);
		}
		for (Card c : this.cards) {
			for (ArrayList<Card> meld : GinRummyUtil.cardsToAllMelds(allCards)) {
				if (meld.size() < 4 && meld.contains(card) && meld.contains(c)) {
					return true;
				}
			}
		}
		
	
//		for (ArrayList<Card> meld : GinRummyUtil.cardsToAllMelds(newCards)) {
//			if (meld.contains(card)) {
//				return true;
//			}
//		}
			
		return false;
	}

	@Override
	public void reportDraw(int playerNum, Card drawnCard) {
		
		// Divide into 4 cases:
		//		+ Opponent draw face down
		//		+ opponent draw face up
		//		+ We draw face up
		// 		+ We draw face down
		
		if (drawnCard == null) {
			if (playerNum != this.playerNum) {
				// Opponent draw face down or pass in their first turn => We can conclude that it is not in any melds of opponent.
				// Therefore, lower the probability of the opponent having other cards in melds in hand.
				// Bayes theorem -> P (A|B) = P (B|A)P(A) / P(B) = P(B|A)
				// the probs of an opponent having an actual card A given that the opponent pass this card B is
				// the probs of an opponent having this card B given that he/she has pass the card A (minus the probs of the card A being in a meld) multiply with
				// the probs of the opponent having this card A (1 / unknown_cards) over the probs of the opponent having the discarded card (1).
				// The probs of that card being in a meld is the number of melds having those two card over the total number of meld.
				// Add the probability of each card to the total probability of opponent hands.
				
				//So,
				ArrayList<Card> totalCards = new ArrayList<Card>();
				for (int i = 0; i < Card.allCards.length; i++) {
					totalCards.add(Card.allCards[i]);
				}
				
				@SuppressWarnings("unchecked")
				ArrayList<Double> probs_op_card_this_turn = (ArrayList<Double>) this.op_cards.clone();
				
//				System.out.println("Op_cards vector:");
//				for (int i = 0; i <  op_cards.size(); i++) {
//					// Debugging
//					System.out.printf("%s: %.5f ",Card.getCard(i).toString(), op_cards.get(i));
//					System.out.printf("%.5f ", probs_op_card_this_turn.get(i));
//				}
				
				ArrayList<ArrayList<Card>> totalMelds = GinRummyUtil.cardsToAllMelds(totalCards);
				for (int i = 0; i < probs_op_card_this_turn.size(); i++) {
					int count_meld_containing_face_up = 0;
					for (ArrayList<Card> meld : totalMelds) {
						if (meld.contains(this.faceUpCard) && meld.contains(Card.getCard(i))) {
							count_meld_containing_face_up ++;
						}
					}
					
					double probs = (double) count_meld_containing_face_up / totalMelds.size();
//					System.out.print("Meld Faced Up: " + count_meld_containing_face_up + " ");
//					probs_op_card_this_turn.set(i, probs_op_card_this_turn.get(i) + probs * probs_op_card_this_turn.get(i) / 1);
					probs_op_card_this_turn.set(i, probs_op_card_this_turn.get(i) - probs / 1);
				}
//				System.out.println("Total meld size: " + totalMelds.size());
				
				this.op_cards = normalize(probs_op_card_this_turn);
				
				// Debugging
				if (GinRummyGame.playVerbose) {
					System.out.println("Op_cards vector:");
					for (int i = 0; i <  op_cards.size(); i++) {
						// Debugging
						System.out.printf("%s: %.5f ",Card.getCard(i).toString(), op_cards.get(i));
//						System.out.printf("%.5f ", probs_op_card_this_turn.get(i));
					}
					System.out.println();
				}

				
				return;
			}
			else return; // Nothing to do if we pass the first round
		}
		draw_pile.get(drawnCard.getId());
		if (playerNum == this.playerNum) {
			cards.add(drawnCard);
			this.drawnCard = drawnCard;
			op_cards.set(drawnCard.getId(), 0.0);
			// If we draw face up
			if (this.drawnCard.getId() == this.faceUpCard.getId()) {
				// Update
				// If the card we actually have drawn is actually the faceupcard, then we have to edit our strategy vector?
				// If not, the then we did draw from the pile, make change to draw_pile and op_matrix.
//				discardPile.push(drawnCard);
			}
			else {
				// If we draw face down
			}
		} else {
			// opponent draw face up
			op_cards.set(drawnCard.getId(), 1.0);
//			discardPile.pop();
			
			
			// Bayes theorem -> P (A|B) = P (B|A)P(A) / P(B) = P(B|A)
			// the probs of an opponent having an actual card A given that the opponent picked this card B is
			// the probs of an opponent having this card B given that he/she has picked the card A (probs of the card A being in a meld) multiply with
			// the probs of the opponent having this card A (1 / unknown_cards) over the probs of the opponent having the discarded card (1).
			// The probs of that card being in a meld is the number of melds having those two card over the total number of meld.
			// Add the probability of each card to the total probability of opponent hands.
			
			//So,
			ArrayList<Card> totalCards = new ArrayList<Card>();
			for (int i = 0; i < Card.allCards.length; i++) {
				totalCards.add(Card.allCards[i]);
			}
			
			@SuppressWarnings("unchecked")
			ArrayList<Double> probs_op_card_this_turn = (ArrayList<Double>) this.op_cards.clone();
			
//			System.out.println("Op_cards vector:");
//			for (int i = 0; i <  op_cards.size(); i++) {
//				// Debugging
//				System.out.printf("%s: %.5f ",Card.getCard(i).toString(), op_cards.get(i));
//				System.out.printf("%.5f ", probs_op_card_this_turn.get(i));
//			}
			
			ArrayList<ArrayList<Card>> totalMelds = GinRummyUtil.cardsToAllMelds(totalCards);
			for (int i = 0; i < probs_op_card_this_turn.size(); i++) {
				int count_meld_containing_face_up = 0;
				for (ArrayList<Card> meld : totalMelds) { 
					if (meld.contains(this.faceUpCard) && meld.contains(Card.getCard(i))) {
						count_meld_containing_face_up ++;
					}
				}
				
				double probs = (double) count_meld_containing_face_up / totalMelds.size();
//				System.out.print(probs + " ");
//				probs_op_card_this_turn.set(i, probs_op_card_this_turn.get(i) + probs * probs_op_card_this_turn.get(i) / 1);
				probs_op_card_this_turn.set(i, probs_op_card_this_turn.get(i) + probs / 1);
			}
			
			this.op_cards = normalize(probs_op_card_this_turn);
			
			
			// Debugging
			if (GinRummyGame.playVerbose) {
				System.out.println("Op_cards vector:");
				for (int i = 0; i <  op_cards.size(); i++) {
					// Debugging
					System.out.printf("%s: %.5f ",Card.getCard(i).toString(), op_cards.get(i));
//					System.out.printf("%.5f ", probs_op_card_this_turn.get(i));
				}
				System.out.println();
			}

			
			
		}
	}

	@SuppressWarnings("unchecked")
	@Override
	public Card getDiscard() {
		// Discard a random card (not just drawn face up) leaving minimal deadwood points.
		int minDeadwood = Integer.MAX_VALUE;
		
		for (Card card : cards) {
			// Cannot draw and discard face up card.
			if (card == drawnCard && drawnCard == faceUpCard)
				continue;
			// Disallow repeat of draw and discard.
			ArrayList<Card> drawDiscard = new ArrayList<Card>();
			drawDiscard.add(drawnCard);
			drawDiscard.add(card);
			if (drawDiscardBitstrings.contains(GinRummyUtil.cardsToBitstring(drawDiscard)))
				continue;
		}
//		Card discard = candidateCards.get(random.nextInt(candidateCards.size()));
		updateDiscardStrategy();
		
		ArrayList<Double> discard_strategy = (ArrayList<Double>) this.dis_strategy.clone();
		
		
		// Debugging
		if (GinRummyGame.playVerbose) {
			System.out.println("Discard strategy before masking!");
			for (int i = 0; i < discard_strategy.size(); i++) {
				System.out.printf("%s: %.5f ", Card.getCard(i), discard_strategy.get(i));
			}
			System.out.println();
		}

		// Masking
		for (int i = 0; i < discard_strategy.size(); i++) {
			boolean inHand = false;
			for (Card c : this.cards) {
				if (c.getId() == i) {
					inHand = true;
					break;
				}
			}
			if (!inHand) {
				discard_strategy.set(i, 0.0);
			}
		}
		discard_strategy.set(drawnCard.getId(), 0.0);

		// Creating meld set
		//Mask the cards in my hand which is not in melds out to form a discard candidate list.
		ArrayList<Card> candidateCards = new ArrayList<Card>(this.cards);
		ArrayList<ArrayList<Card>> melds = GinRummyUtil.cardsToAllMelds(candidateCards);
		for (Card c : this.cards) {
			for (ArrayList<Card> meld : melds) {
				if (meld.contains(c)) {
					candidateCards.remove(c);
				}
			}
		}
		
		
		// Mask the card not in candidate cards list.
		for (int i = 0; i < discard_strategy.size(); i++) {
			if (!candidateCards.contains(Card.getCard(i))) {
				discard_strategy.set(i, 0.0);
			}
		}
		
		
		// Debugging
//		Card discard = new Card(2,2);
		
		// Getting the max probability for the discarding strategy.
		double max_probs = Double.NEGATIVE_INFINITY;
		
		for (int i = 0; i < discard_strategy.size(); i++) {
			double probs = discard_strategy.get(i);
			if (probs >= max_probs) {
				max_probs = probs;
			}
		}
		
		// Debugging
		if (GinRummyGame.playVerbose) {
			System.out.println("Discard strategy vector after masking:");
			for (int i = 0; i < discard_strategy.size(); i++) {
				double probs = discard_strategy.get(i);
				System.out.printf("%s: %.5f ", Card.getCard(i).toString(), probs);
			}
			System.out.println();
		}
		
		// In the set of all maximum probability, pick the card that minimize deadwood point
		ArrayList<Card> finalCandidateCards = new ArrayList<Card>();
		for (int i = 0; i < discard_strategy.size(); i++) {
			double probs = discard_strategy.get(i);
			if (probs == max_probs) {
				finalCandidateCards.add(Card.getCard(i));
			}
		}
		
		// finalCandidateCards is guaranteed to have at least 1 card.
		Card discard = finalCandidateCards.get(0);
		for (Card c : finalCandidateCards) {
			if (c.getRank() > discard.getRank()) {
				discard = c;
			}
		}
		
		// Prevent future repeat of draw, discard pair.
		ArrayList<Card> drawDiscard = new ArrayList<Card>();
		drawDiscard.add(drawnCard);
		drawDiscard.add(discard);
		drawDiscardBitstrings.add(GinRummyUtil.cardsToBitstring(drawDiscard));
		return discard;
		
		
		// Only deciding, the data matrix evaluation is computed inside report() methods.
		// Base on the dis_strategy matrix to have the right discard solution.
		
	}

	@Override
	public void reportDiscard(int playerNum, Card discardedCard) {
//		discardPile.push(discardedCard);
		// Ignore other player discards.  Remove from cards if playerNum is this player.
		if (playerNum == this.playerNum) {
			cards.remove(discardedCard);
		} else {
			op_cards.set(discardedCard.getId(), 0.0);
			draw_pile.set(discardedCard.getId(), 0.0);
			
			op_cards = normalize(op_cards);
			draw_pile = normalize(draw_pile);
			
			// When an opponent discard a card:
			// What happened to op_cards?
			// What happened to draw_pile?
			// what happened to opponents strategy? melds? cfvs?
			// Does that option change our way of strategy?
			
		}
			
	}

	@Override
	public ArrayList<ArrayList<Card>> getFinalMelds() {
		// Check if deadwood of maximal meld is low enough to go out. 
		ArrayList<ArrayList<ArrayList<Card>>> bestMeldSets = GinRummyUtil.cardsToBestMeldSets(cards);
		if (!opponentKnocked && (bestMeldSets.isEmpty() || GinRummyUtil.getDeadwoodPoints(bestMeldSets.get(0), cards) > GinRummyUtil.MAX_DEADWOOD))
			return null;
		return bestMeldSets.isEmpty() ? new ArrayList<ArrayList<Card>>() : bestMeldSets.get(random.nextInt(bestMeldSets.size()));
		
		
	}

	@Override
	public void reportFinalMelds(int playerNum, ArrayList<ArrayList<Card>> melds) {
		// Melds ignored by simple player, but could affect which melds to make for complex player.
		if (playerNum != this.playerNum)
			opponentKnocked = true;
		
		// Check if opponent can have what meld, to push the card to!
	}

	@Override
	public void reportScores(int[] scores) {
		// Ignored by simple player, but could affect strategy of more complex player.
	}

	@Override
	public void reportLayoff(int playerNum, Card layoffCard, ArrayList<Card> opponentMeld) {
		// Ignored by simple player, but could affect strategy of more complex player.
		
	}

	@Override
	public void reportFinalHand(int playerNum, ArrayList<Card> hand) {
		// Ignored by simple player, but could affect strategy of more complex player.		
	}
	
	public int getPayoff () {
		return 0;
	}
	
	private ArrayList<Double> normalize(ArrayList<Double> vector) {
		
		double normalizing_sum = 0;
		ArrayList<Double> newVector = new ArrayList<Double>();
		Iterator<Double> it = vector.iterator();
		while (it.hasNext()) {
			double prob = it.next();
			if (prob < 0) {
				prob = 0;
			}
			normalizing_sum += prob;
		}
		
		it = vector.iterator();
		while (it.hasNext()) {
			if (normalizing_sum != 0) {
				newVector.add(Math.max(0.0, it.next()) / normalizing_sum);
			} else {
				newVector.add(1.0 / unknown_cards);
			}
		}
		
		return newVector;
	}
	
	@SuppressWarnings("unchecked")
	private void updateDiscardStrategy() {
		
		// From op_cards (expected card opponent has in hand, assess what card to discard! in the strategy vector!)
		
		// Get all value > 0 from opponent hand vector to form a candidate assessment of opponent hands.
		HashMap<Card, Double> map = new HashMap<Card, Double>();
		ArrayList<Double> op_expecting = (ArrayList<Double>) op_cards.clone();
		for (int i = 0; i < op_expecting.size(); i++) {
			double prob = op_expecting.get(i);
			if (prob > 0) {
				map.put(Card.getCard(i), prob);
			}
		}
		
		ArrayList<Card> opponentHand;
		
		for (int i = 0; i < op_expecting.size(); i++) {
			double prob = op_expecting.get(i);
			if (prob == 0.0) {
				Card thisCard = Card.getCard(i);
				opponentHand = new ArrayList<>();
				opponentHand.addAll(map.keySet());
				opponentHand.add(thisCard);
				ArrayList<ArrayList<Card>> melds = GinRummyUtil.cardsToAllMelds(opponentHand);
				if (melds.size() == 0) {
					// If there is no melds can be form, the opponent doesn't expect to form any meld from this card.
					continue;
				}
				int count_existence = 0;
				for (ArrayList<Card> meld : melds) {
					if (meld.contains(thisCard)) {
						count_existence ++;
					}
				}
				op_expecting.set(i, (double) count_existence / melds.size());
			}
		}
		
		// Flip
		for (int i = 0; i < this.dis_strategy.size(); i++) {
			double counter_op_value = 1.0 - op_expecting.get(i);
//			System.out.printf("%s: %.5f", Card.getCard(i).toString(), this.dis_strategy.get(i) + counter_op_value);
			this.dis_strategy.set(i, this.dis_strategy.get(i) + counter_op_value);
		}
//		System.out.println();
		this.dis_strategy = normalize (this.dis_strategy);		
	}
	
//	public static void main(String[] args) {
//		CFRPlayer c = new CFRPlayer();
//		Random RANDOM = new Random();
//		Stack<Card> deck = Card.getShuffle(RANDOM.nextInt());
//		Card[] cardsHand = new Card[52];
//
//		for (int i = 0; i < 52; i++) {
//			cardsHand[i] = deck.pop();
//			System.out.print(cardsHand[i].getId() + " ");
//		}
//		System.out.println();
//		
//		c.startGame(1, 1, cardsHand);
//		for (int i = 0; i < 10; i++) {
//			System.out.print(c.cards.get(i).getId() + " " );
//		}
//		System.out.println();
//		System.out.println(GinRummyUtil.cardsToBitstring(c.cards));
//		
//		
//		
//	}
	
}
