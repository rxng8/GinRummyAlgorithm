package player;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Random;
import core.*;
import collector.*;
import module.*;
import util.*;
/**
 * Reference: Professor Neller
 * @author alexv
 *	Gettysburg college
 */

/**
 * 
 * BUG: The player is not informed when in the first turn, the opponent draw the face up card (Solve)
 * Feature: Knocker Bot, Hand estimation
 */

public class DraftPlayer3 implements GinRummyPlayer {
	public static final boolean VERBOSE = false;
	public static final float KNOCKING_THRESHOLD = 0.1f;
	private int playerNum;
	@SuppressWarnings("unused")
	private int startingPlayerNum;
	private ArrayList<Card> cards = new ArrayList<Card>();
	private Random random = new Random();
	private boolean opponentKnocked = false;
	Card faceUpCard, drawnCard;
	ArrayList<Long> drawDiscardBitstrings = new ArrayList<Long>();
	HandEstimator2 estimator = new HandEstimator2();
	private int totalDiscarded = 0;
	
	int turn;
	KnockingModule kn_bot;
	
	@Override
	public void startGame(int playerNum, int startingPlayerNum, Card[] cards) {
		this.playerNum = playerNum;
		this.startingPlayerNum = startingPlayerNum;
		this.cards.clear();
		for (Card card : cards)
			this.cards.add(card);
		opponentKnocked = false;
		drawDiscardBitstrings.clear();
		estimator.init();
		ArrayList<Card> hand = new ArrayList<Card>();
		for (Card c : cards)
			hand.add(c);
		estimator.setOpKnown(hand, false);
		estimator.setOtherKnown(hand, true);
//		estimator.print();
		totalDiscarded = 0;
		kn_bot = new KnockingModule();
		turn = 0;
	}	

	@Override
	public boolean willDrawFaceUpCard(Card card) {
		// Return true if card would be a part of a meld, false otherwise.
		estimator.setOpKnown(card, false);
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
		this.drawnCard = drawnCard;
		if (playerNum == this.playerNum) {
			cards.add(drawnCard);
			estimator.setOpKnown(drawnCard, false);
			estimator.setOtherKnown(drawnCard, true);
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
//		if (candidateCards.size() > 1) {
//			System.out.print("Candidates: ");
//			for (Card c : candidateCards)
//				System.out.print(c.rank + " ");
//			System.out.println();
//		}
		if (candidateCards.size() > 1) {
			int maxRank = candidateCards.get(0).rank;
			ArrayList<Card> maxCandidateCards = new ArrayList<Card>();
			for (Card c : candidateCards) {
				if (c.rank > maxRank) {
					maxCandidateCards.clear();
					maxRank = c.rank;
				}
				if (c.rank == maxRank)
					maxCandidateCards.add(c);
			}
			candidateCards = maxCandidateCards;
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
		totalDiscarded++;
		if (playerNum == this.playerNum) {
			cards.remove(discardedCard);
		}
		else {
			if (faceUpCard == null) {
				// the statement faceupcard == null ? drawnCard : faceupcard is to say that although the faceup card is always not null,
				// when the opponent draw the faceupcard in the first turn, it will be null. So we report the draw card instead
				estimator.reportDrawDiscard(drawnCard, true, discardedCard, turn);
			} else {
				estimator.reportDrawDiscard(faceUpCard, faceUpCard == drawnCard, discardedCard, turn);
			}
		}
		faceUpCard = discardedCard;
		if (VERBOSE) estimator.view();
		turn++;
	}

	@Override
	public ArrayList<ArrayList<Card>> getFinalMelds() {
		// Check if deadwood of maximal meld is low enough to go out. 
//		ArrayList<ArrayList<ArrayList<Card>>> bestMeldSets = GinRummyUtil.cardsToBestMeldSets(cards);
//		if (!opponentKnocked && (bestMeldSets.isEmpty() || GinRummyUtil.getDeadwoodPoints(bestMeldSets.get(0), cards) > 0))
//			return null;
//		return bestMeldSets.isEmpty() ? new ArrayList<ArrayList<Card>>() : bestMeldSets.get(random.nextInt(bestMeldSets.size()));
		
		
		
		ArrayList<ArrayList<ArrayList<Card>>> bestMeldSets = GinRummyUtil.cardsToBestMeldSets(cards);
		float knock_prob = 0;
		if (!bestMeldSets.isEmpty() && GinRummyUtil.getDeadwoodPoints(bestMeldSets.get(0), cards) <= 10) {
			int deadwood, n_meld;
			deadwood = GinRummyUtil.getDeadwoodPoints(bestMeldSets.get(0), cards);
			n_meld = bestMeldSets.get(0).size() - 1;
			
			int[] X = {turn, deadwood, n_meld};
			knock_prob = kn_bot.predict(X);
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
		
	}

	@Override
	public void reportFinalHand(int playerNum, ArrayList<Card> hand) {
		if (playerNum != this.playerNum) { // opponent hand
//			// Record est. likelihood of actual opponent hand
//			int numCards = 0;
//			double estProb = 1;
//			for (Card card : hand) {
//				numCards++;
//				if (!estimator.known[card.getId()])
//					estProb *= estimator.prob[card.getId()];
//			}
//			// Record uniform likelihood of actual opponent hand
//			double uniformProb = 1;
//			// Compute the number of possible cards that may be those unknown in the opponent's hand
//			System.out.println("Number of opponent cards known: " + (hand.size() - estimator.numUnknownInHand));
//			System.out.println("Number discarded: " + totalDiscarded);
//			double numCandidates = Card.NUM_CARDS - totalDiscarded - hand.size() - (hand.size() - estimator.numUnknownInHand);
//			System.out.println("Number of candidates: " + numCandidates);
//			double singleCardProb = (double) estimator.numUnknownInHand / numCandidates;
//			for (int i = 0; i < estimator.numUnknownInHand; i++) 
//				uniformProb *= singleCardProb;
//
//			System.out.println(">>>> est. " + estProb + " unif. " + uniformProb + " ratio " + (estProb / uniformProb));
//			ratios.add((estProb / uniformProb));
//			System.out.println(ratios);
//			double sum = 0;
//			for (double ratio : ratios)
//				sum += ratio;
//			System.out.println("Average ratio: " + sum / ratios.size());
			
//			System.out.println("Accuracy: " + HandEstimator2.cal_accuracy(hand, estimator.probs));
			
		}
	}

	public static void main(String[] args) throws InstantiationException, IllegalAccessException, ClassNotFoundException, FileNotFoundException, UnsupportedEncodingException {
		
	}
}
