package player;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;
import collector.*;
import core.*;
import module.*;
import util.*;


public class MediumPlayerEK implements GinRummyPlayer {
	protected int playerNum;
	@SuppressWarnings("unused")
	protected int startingPlayerNum;
	protected ArrayList<Card> cards = new ArrayList<Card>();
	protected Random random = new Random();
	protected boolean opponentKnocked = false;
	Card faceUpCard, drawnCard; 
	ArrayList<Long> drawDiscardBitstrings = new ArrayList<Long>();
	
	protected HandEstimator3 estimator = new HandEstimator3();
	public ArrayList<Card> candidateCards = new ArrayList<>();
	public double[] cardDesirabilities;
	
	/**
	 * Knocking Module
	 */
	KnockingModule knockEngine = new KnockingModule();

	/**
	 * Hittig module
	 */
	HittingModule hitEngine = new HittingModule();
	
	public boolean VERBOSE = false;
	
	public static final float OPPO_CARD_PROB_WEIGHT = 1.0f;
	public static final float CARD_DEADWOOD_WEIGHT = 0.3f;
	
	public static final float KNOCKING_THRESHOLD = 0.9f;
	
	/**
	 * current turn
	 */
	int turn;
	
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
		estimator.init();
		
		ArrayList<Card> hand = new ArrayList<Card>();
		for (Card c : cards)
			hand.add(c);
		
		// set every cards in hand to false in case it's not reset yet
		hitEngine.setOpKnown(hand, false);
		estimator.setKnown(hand, false);
		
		turn = 0;
	}

	@Override
	public boolean willDrawFaceUpCard(Card card) {
		
		// Set this card in opponent hand to false cuz it's face up on the table
		hitEngine.setOpKnown(card, false);
		estimator.setKnown(card, false);
		
		// Return true if card would be a part of a meld, false otherwise.
		this.faceUpCard = card;
		@SuppressWarnings("unchecked")
		ArrayList<Card> newCards = (ArrayList<Card>) cards.clone();
		newCards.add(card);
		for (ArrayList<Card> meld : GinRummyUtil.cardsToAllMelds(newCards))
			if (meld.contains(card))
				return true;
		return false;
	}
	
	@Override
	public void reportDraw(int playerNum, Card drawnCard) {
		// Set known variables if the drawnCard is not null
		if(drawnCard != null) {
			if(playerNum == this.playerNum) {
				cards.add(drawnCard);
				hitEngine.setOpKnown(drawnCard, false);
				estimator.setKnown(drawnCard, false);
			}
			else 
				estimator.setKnown(drawnCard, true);
		}
		this.drawnCard = drawnCard;
	}
	
	
	@Override
	public Card getDiscard() {
		Card discard = getEstimatedCandidate();
		
		if (discard == null)
			discard = getSimpleCandidate();
		
		// Prevent future repeat of draw, discard pair.
		ArrayList<Card> drawDiscard = new ArrayList<Card>();
		drawDiscard.add(drawnCard);
		drawDiscard.add(discard);
		drawDiscardBitstrings.add(GinRummyUtil.cardsToBitstring(drawDiscard));
		return discard;
	}
	
	
	@SuppressWarnings("unchecked")
	public Card getEstimatedCandidate() {
		HashSet<Card> candidatesInSet = new HashSet<>();
		ArrayList<ArrayList<ArrayList<Card>>> bestMeldSets = GinRummyUtil.cardsToBestMeldSets(cards);
		if(bestMeldSets.size() > 0) {
			for(ArrayList<ArrayList<Card>> meldSet : bestMeldSets) {
				if(VERBOSE)
					System.out.println(meldSet);
				
				ArrayList<Card> remainingCards = (ArrayList<Card>) cards.clone();
				for(ArrayList<Card> meld : meldSet)
					for(Card c : meld)
						remainingCards.remove(c);
				for(Card card : remainingCards) {
					
					// Cannot draw and discard face up card.
					if (card == drawnCard && drawnCard == faceUpCard)
						continue;
					// Disallow repeat of draw and discard.
					ArrayList<Card> drawDiscard = new ArrayList<Card>();
					drawDiscard.add(drawnCard);
					drawDiscard.add(card);
					if (drawDiscardBitstrings.contains(GinRummyUtil.cardsToBitstring(drawDiscard)))
						continue;
					
					candidatesInSet.add(card);
				}
			}
			if(VERBOSE) 
				System.out.printf("estimating candidates: %s \n", candidatesInSet);
			
			candidateCards.addAll(candidatesInSet);
		}
		
		if(candidateCards.size() > 0) {
			double[] desirableRatio = getCardDesirability(candidateCards);
			
			if(VERBOSE) {
//				estimator.print();
//				for(int i = 0; i < desirableRatio.length; i++)
//					System.out.printf("%.4f ", desirableRatio[i]);
//				System.out.println();
				
				System.out.println(desirableRatio.length);
			}
			
			double max = 0;
			int maxIndex = 0;
			for(int i = 0; i < desirableRatio.length; i++) {
				
				double discardSafety = (1-desirableRatio[i]) * OPPO_CARD_PROB_WEIGHT + GinRummyUtil.getDeadwoodPoints(candidateCards.get(i)) * CARD_DEADWOOD_WEIGHT;
				
				if(discardSafety >= max) {
					max = discardSafety;
					maxIndex = i;
				}
				
				if(VERBOSE) { System.out.printf("%.4f ", discardSafety); }
			}
			if(VERBOSE) { System.out.println(); }
			return candidateCards.get(maxIndex);
		}
		
		return null;
	}
	
	
	@SuppressWarnings("unchecked")
	public Card getSimpleCandidate() {
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
		return candidateCards.get(random.nextInt(candidateCards.size()));
	}
	
	public double[] getCardDesirability(ArrayList<Card> candidates) {
		cardDesirabilities = new double[candidates.size()];
		
		for(int i = 0; i < candidates.size(); i++) {
			double desirability = 0;
			for(Long meldBitstring : GinRummyUtil.getAllMeldBitstrings()) {
				ArrayList<Card> cards = GinRummyUtil.bitstringToCards(meldBitstring);
				double probOfMeld = 0;
				if(cards.contains(candidates.get(i))) {
					probOfMeld = 1;
					cards.remove(candidates.get(i));
					for(Card card : cards) 
						probOfMeld *= estimator.getProb()[card.getId()];
				}
				desirability += probOfMeld;
			}
			cardDesirabilities[i] = desirability;
		}
		return cardDesirabilities;
	}
	
	public int getTurn() {
		return turn;
	}
	
	
	@Override
	public void reportDiscard(int playerNum, Card discardedCard) {
		// Record opponent's discard for the hand estimator.
		if (playerNum == this.playerNum) {
			cards.remove(discardedCard);
			candidateCards.clear();
			hitEngine.setDiscardKnown(discardedCard, true);
			
			if (VERBOSE) {
				System.out.println("Number of cards that are hitting card in the unmelds set: " + hitEngine.count_hitting(cards));
				System.out.println("Which is : " + hitEngine.get_hitting(cards));
			}
		}
		else {
			if (faceUpCard == null) {
				// the statement faceupcard == null ? drawnCard : faceupcard is to say that although the faceup card is always not null,
				// when the opponent draw the faceupcard in the first turn, it will be null. So we report the draw card instead
				estimator.reportDrawDiscard(drawnCard, true, discardedCard);
				hitEngine.reportDrawDiscard(drawnCard, true, discardedCard, turn);
			}
			else {
				estimator.reportDrawDiscard(faceUpCard, faceUpCard == drawnCard, discardedCard);
				hitEngine.reportDrawDiscard(faceUpCard, faceUpCard == drawnCard, discardedCard, turn);
			}
		}
		faceUpCard = discardedCard;
		if (VERBOSE) {
			estimator.print();
			hitEngine.print();
		}
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
