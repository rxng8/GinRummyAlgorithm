package player;

import java.util.ArrayList;
import java.util.Random;
import collector.*;
import core.*;
import module.*;
import util.*;


public class MediumPlayerKH implements GinRummyPlayer {
	protected int playerNum;
	@SuppressWarnings("unused")
	protected int startingPlayerNum;
	protected ArrayList<Card> cards = new ArrayList<Card>();
	protected Random random = new Random();
	protected boolean opponentKnocked = false;
	Card faceUpCard, drawnCard; 
	ArrayList<Long> drawDiscardBitstrings = new ArrayList<Long>();

	/**
	 * Debug engine verbose
	 */
	public boolean VERBOSE = false;
	
	/**
	 * Knocking threshold
	 */
	public static final float KNOCKING_THRESHOLD = 0.9f;
	
	/**
	 * Hitting threshold
	 */
	public static final float HIT_CARD_VALUE_THRESHOLD = 14f; // Min 0, Max > 15
	
	/**
	 * current turn
	 */
	int turn;
	
	/**
	 * HARD-CODED TURN WHEN THE GAME ENDS
	 */
	static int ENDGAME = 10;
	
	/**
	 * HARDCODED PLAYING MODE: if agressive mode is true, then dont discard hitting cards.
	 */
	static boolean aggressiveMode = true;
	
	/**
	 * Knocking Module
	 */
	KnockingModule knockEngine = new KnockingModule();
	
	/**
	 * Hittig module
	 */
	HittingModule hitEngine = new HittingModule();
	
	
	@Override
	public void startGame(int playerNum, int startingPlayerNum, Card[] cards) {
		this.playerNum = playerNum;
		this.startingPlayerNum = startingPlayerNum;
		this.cards.clear();
		for (Card card : cards)
			this.cards.add(card);
		opponentKnocked = false;
		drawDiscardBitstrings.clear();
		
		// Initialize hitting module
		hitEngine.init();
		
		// clone hand
		ArrayList<Card> hand = new ArrayList<Card>();
		for (Card c : cards) {
			hand.add(c);
		}
		
		// set every cards in hand to false in case it's not reset yet
		hitEngine.setOpKnown(hand, false);
		
		// Reset turn = 0;
		turn = 0;
		
	}

	@Override
	public boolean willDrawFaceUpCard(Card card) {
		
		// Set this card in opponent hand to false cuz it's face up on the table
		hitEngine.setOpKnown(card, false);
		
		// Return true if card would be a part of a meld, false otherwise.
		this.faceUpCard = card;
		
		// Put the new card to hand in a cloned hand to evaluate things.
		@SuppressWarnings("unchecked")
		ArrayList<Card> newCards = (ArrayList<Card>) cards.clone();
		newCards.add(card);
		
		
		// Return true if card would be a part of a meld
		for (ArrayList<Card> meld : GinRummyUtil.cardsToAllMelds(newCards)) {
			if (meld.contains(card)) {
				return true;
			}
		}
		
		// Return true if the value of the face up card is greater than the threshold
		
		// First, we evaluate hitting value of the face Up card by putting line data in pretrained model.
		// Check if it's hitting.
		int[] line = new int[3];
		line[0] = turn;
		line[1] = card.rank;
		line[2] = hitEngine.countHitMeld(cards, card);
//		line[3] = ENDGAME - turn;
		// Evaluate if we can pick the card or not based on the threshold
		boolean canPick = false;
		float cardValue = hitEngine.predict(line);
		if (cardValue > HIT_CARD_VALUE_THRESHOLD) canPick = true;
		return canPick;
	}

	@Override
	public void reportDraw(int playerNum, Card drawnCard) {

		this.drawnCard = drawnCard;
		
		// Add to cards if playerNum is this player.
		if (playerNum == this.playerNum) {
			cards.add(drawnCard);
			
			// Set this card in op hand to false cuz it's in my hand
			hitEngine.setOpKnown(drawnCard, false);
		}
	}

	@SuppressWarnings("unchecked")
	@Override
	public Card getDiscard() {
		// Discard a random card (not just drawn face up) leaving minimal deadwood points.
		int minDeadwood = Integer.MAX_VALUE;
		ArrayList<Card> candidateCards = new ArrayList<Card>();
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
			
			// If turn left < 5, disable agressive mode
			if (ENDGAME - turn < 5) aggressiveMode = false;
			
			// if in aggresive mode, omit discarding hitting card
			if (aggressiveMode) {
				if (hitEngine.isHittingCard(cards, card)) {
					continue;
				}
			}
			
			ArrayList<Card> remainingCards = (ArrayList<Card>) cards.clone();
			remainingCards.remove(card);
			ArrayList<ArrayList<ArrayList<Card>>> bestMeldSets = GinRummyUtil.cardsToBestMeldSets(remainingCards);
			int deadwood = bestMeldSets.isEmpty() ? GinRummyUtil.getDeadwoodPoints(remainingCards) : GinRummyUtil.getDeadwoodPoints(bestMeldSets.get(0), remainingCards);
			if (deadwood <= minDeadwood) {
				if (deadwood < minDeadwood) {
					minDeadwood = deadwood;
					candidateCards.clear();
				}
				candidateCards.add(card);
			}
		}
		
		// If everycard is hitting card, we discard the highest rank as usual
		if (candidateCards.isEmpty()) {
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
				
				ArrayList<Card> remainingCards = (ArrayList<Card>) cards.clone();
				remainingCards.remove(card);
				ArrayList<ArrayList<ArrayList<Card>>> bestMeldSets = GinRummyUtil.cardsToBestMeldSets(remainingCards);
				int deadwood = bestMeldSets.isEmpty() ? GinRummyUtil.getDeadwoodPoints(remainingCards) : GinRummyUtil.getDeadwoodPoints(bestMeldSets.get(0), remainingCards);
				if (deadwood <= minDeadwood) {
					if (deadwood < minDeadwood) {
						minDeadwood = deadwood;
						candidateCards.clear();
					}
					candidateCards.add(card);
				}
			}
		}
		
		Card discard = candidateCards.get(random.nextInt(candidateCards.size()));
		// Prevent future repeat of draw, discard pair.
		ArrayList<Card> drawDiscard = new ArrayList<Card>();
		drawDiscard.add(drawnCard);
		drawDiscard.add(discard);
		drawDiscardBitstrings.add(GinRummyUtil.cardsToBitstring(drawDiscard));
		return discard;
	}

	@Override
	public void reportDiscard(int playerNum, Card discardedCard) {
//		totalDiscarded++;
		if (playerNum == this.playerNum) {
			cards.remove(discardedCard);
			hitEngine.setDiscardKnown(discardedCard, true);
			
			// Count the number of hitting cards except for melds in hand

			if (VERBOSE) {
				System.out.println("Number of cards that are hitting card in the unmelds set: " + hitEngine.count_hitting(cards));
				System.out.println("Which is : " + hitEngine.get_hitting(cards));
			}
		}
		else {
			if (faceUpCard == null) {
				// the statement faceupcard == null ? drawnCard : faceupcard is to say that although the faceup card is always not null,
				// when the opponent draw the faceupcard in the first turn, it will be null. So we report the draw card instead
//				estimator.reportDrawDiscard(drawnCard, true, discardedCard, turn);
				hitEngine.reportDrawDiscard(drawnCard, true, discardedCard, turn);
			} else {
//				estimator.reportDrawDiscard(faceUpCard, faceUpCard == drawnCard, discardedCard, turn);
				hitEngine.reportDrawDiscard(faceUpCard, faceUpCard == drawnCard, discardedCard, turn);
			}
		}
		faceUpCard = discardedCard;
//		if (VERBOSE) estimator.view();
		if (VERBOSE) hitEngine.print();
		turn++;
	}

	@Override
	public ArrayList<ArrayList<Card>> getFinalMelds() {
		ArrayList<ArrayList<ArrayList<Card>>> bestMeldSets = GinRummyUtil.cardsToBestMeldSets(cards);
		float knock_prob = 0;
		if (!bestMeldSets.isEmpty() && GinRummyUtil.getDeadwoodPoints(bestMeldSets.get(0), cards) <= 10) {
			int deadwood, n_meld, n_hit, n_oppick;
			deadwood = GinRummyUtil.getDeadwoodPoints(bestMeldSets.get(0), cards);
			n_meld = bestMeldSets.get(0).size();
			n_hit = hitEngine.count_hitting(cards);
			n_oppick = hitEngine.get_n_op_pick();
			
			int[] X = {turn, deadwood, n_meld, n_hit, n_oppick};
			knock_prob = knockEngine.predict(X);
			if (VERBOSE) {
				System.out.printf("Current Deadwood: %d, Number of melds: %d, Number of hitting cards: %d, Opponent have picked %d card(s). So probs to knock is %.5f\n", deadwood, n_meld, n_hit, n_oppick, knock_prob);
			}
		}
		
		
		if (!opponentKnocked && (bestMeldSets.isEmpty() || knock_prob < KNOCKING_THRESHOLD))
			return null;
		return bestMeldSets.isEmpty() ? new ArrayList<ArrayList<Card>>() : bestMeldSets.get(random.nextInt(bestMeldSets.size()));
	}

	@Override
	public void reportFinalMelds(int playerNum, ArrayList<ArrayList<Card>> melds) {
		// Melds ignored by simple player, but could affect which melds to make for complex player.
		if (playerNum != this.playerNum)
			opponentKnocked = true;
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
	
}
