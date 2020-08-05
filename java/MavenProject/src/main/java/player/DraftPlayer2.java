package player;

/**
 * @author Alex Nguyen
 * Gettysburg College
 * 
 * Advisor: Professor Neller.
 */

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import core.*;
import collector.*;
import module.*;
import util.*;
/**
 * Implements a random dummy Gin Rummy player that has the following trivial, poor play policy: 
 * Ignore opponent actions and cards no longer in play.
 * Draw face up card only if it becomes part of a meld.  Draw face down card otherwise.
 * Discard a highest ranking unmelded card without regard to breaking up pairs, etc.
 * Knock as early as possible.
 * 
 * @author Todd W. Neller
 * @version 1.0

Copyright (C) 2020 Todd Neller

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Information about the GNU General Public License is available online at:
  http://www.gnu.org/licenses/
To receive a copy of the GNU General Public License, write to the Free
Software Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
02111-1307, USA.

 */


/**
 * 
 * Simple player with bayes estimation. And Knocker Bot (threshold 0.99)
 *
 */
public class DraftPlayer2 implements GinRummyPlayer {
	private int playerNum;
	@SuppressWarnings("unused")
	private int startingPlayerNum;
	private ArrayList<Card> cards = new ArrayList<Card>();
	private Random random = new Random();
	private boolean opponentKnocked = false;
	Card faceUpCard, drawnCard; 
	ArrayList<Long> drawDiscardBitstrings = new ArrayList<Long>();
	
	KnockingModule kn_bot;
	
	boolean VERBOSE = false;
	
	Card op_discard, op_draw, op_not_draw;
	ArrayList<Card> discardPile;
	float[] op_cards;
	int turn;
	int n_unknown;
	
	float knocking_threshhold = 0.99f;
	
	@Override
	public void startGame(int playerNum, int startingPlayerNum, Card[] cards) {
		this.playerNum = playerNum;
		this.startingPlayerNum = startingPlayerNum;
		this.cards.clear();
		for (Card card : cards)
			this.cards.add(card);
		opponentKnocked = false;
		drawDiscardBitstrings.clear();
		
		op_cards = new float[52];
		// Initialize to 1/42 for each unknown card
		for (int i = 0; i < op_cards.length; i++) {
			op_cards[i] = (float) (1.0 / 42);
		}
		for (Card card : cards) {
			op_cards[card.getId()] = 0;
		}
		
		
		discardPile = new ArrayList<>();
		turn = 0;
		n_unknown = 41;
		HashMap<Integer, ArrayList<Card>> known_cards;
		
		kn_bot = new KnockingModule();
		
	}

	@Override
	public boolean willDrawFaceUpCard(Card card) {
		
		
		
		
		// Return true if card would be a part of a meld, false otherwise.
		this.faceUpCard = card;
		

		
		// Add to discard pile if it  is this player's turn
		this.discardPile.add(card);
		
		
		
		@SuppressWarnings("unchecked")
		ArrayList<Card> newCards = (ArrayList<Card>) cards.clone();
		newCards.add(card);
		for (ArrayList<Card> meld : GinRummyUtil.cardsToAllMelds(newCards)) {
			if (meld.contains(card)) {
				return true;
			}
		}
		return false;
	}

	public void update_estimation() {
		if (VERBOSE) {
			float[] gmd = get_meld_draw();
			System.out.println("This turn The opponenet draw " + ((this.op_draw != null)? this.op_draw.toString(): "null"));
			print_mat1D_card(gmd, "Drawing meld matrix");
		}
		
		float[] drawn_probs = HandEstimator.prob_bayes_draw(get_meld_draw(), this.op_cards);
		if(VERBOSE) print_mat1D_card(drawn_probs, "Bayes draw");
		
		
		if (VERBOSE) {
			float[] gmdi = get_meld_discard();
			System.out.println("This turn The opponenet discard " + ((this.op_discard != null)? this.op_discard.toString(): "null"));
			print_mat1D_card(gmdi, "Discarding meld matrix");
		}
		
		
		float[] discard_probs = HandEstimator.prob_bayes_draw(get_meld_discard(), this.op_cards);
		if(VERBOSE) print_mat1D_card(drawn_probs, "Bayes discard");
		
		float[] new_op_cards = HandEstimator.prob_card(this.op_cards, turn, this.cards, discardPile, op_draw, op_discard, n_unknown, drawn_probs, discard_probs);

		this.op_cards = new_op_cards;
	}

	@Override
	public void reportDraw(int playerNum, Card drawnCard) {
		
		// Ignore other player draws.  Add to cards if playerNum is this player.
		if (playerNum == this.playerNum) {
			if (drawnCard == null) {
				this.drawnCard = null;
			} else {
				cards.add(drawnCard);
				this.drawnCard = drawnCard;
				discardPile.remove(discardPile.size() - 1);
			}
			n_unknown --;
		}
		else {
			if (drawnCard == null) {
				this.op_draw = null;
				this.op_not_draw = faceUpCard;
			} else {
				this.op_draw = drawnCard;
				this.op_not_draw = null;
			}
		}
	}

	public float[] get_meld_draw() {
		float[] y = new float[52];
		if (this.op_draw != null) {
			for (int i = 0; i < y.length; i++) {
				y[i] = (float) (1.0/n_unknown);
			}
			y[this.op_draw.getId()] = 1;
			// mask 1 around draw card
			int rank = this.op_draw.getRank();
			int suit = this.op_draw.getSuit();
//			System.out.println("Rank: " + rank + " Suit: " + suit);
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
		} else if(op_not_draw != null) {
			for (int i = 0; i < y.length; i++) {
				y[i] = 1;
			}
			y[this.op_not_draw.getId()] = (float) (1.0/n_unknown);
			// mask 1 around draw card
			int rank = this.op_not_draw.getRank();
			int suit = this.op_not_draw.getSuit();
			
			for (int r = rank - 2; r <= rank + 2; r++) {
				if (r < 0) continue;
				if (r > 12) continue;
				y[new Card(r, suit).getId()] = (float) (1.0/n_unknown);
			}
			
			for (int s = suit - 3; s <= suit + 3; s++) {
				if (s < 0) continue;
				if (s > 3) continue;
				y[new Card(rank, s).getId()] = (float) (1.0/n_unknown);
			}
		}
		return y;
	}
	
	public float[] get_meld_discard() {
		
		if (this.op_discard == null) {
			return new float[52];
		}
		
		float[] y = new float[52];
		
		for (int i = 0; i < y.length; i++) {
			y[i] = 1;
		}
		
		
		y[this.op_discard.getId()] = (float) (1.0/n_unknown);
		
		// mask 1 around discard card
		int rank = this.op_discard.getRank();
		int suit = this.op_discard.getSuit();
		
		for (int r = rank - 2; r <= rank + 2; r++) {
			if (r < 0) continue;
			if (r > 12) continue;
			y[new Card(r, suit).getId()] = (float) (1.0/n_unknown);
		}
		
		for (int s = suit - 3; s <= suit + 3; s++) {
			if (s < 0) continue;
			if (s > 3) continue;
			y[new Card(rank, s).getId()] = (float) (1.0/n_unknown);
		}
		
		return y;
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
			
			// Get candidate cards
			ArrayList<Card> remainingCards = (ArrayList<Card>) cards.clone();
			remainingCards.remove(card);
			ArrayList<ArrayList<ArrayList<Card>>> bestMeldSets = GinRummyUtil.cardsToBestMeldSets(remainingCards);
			int deadwood = bestMeldSets.isEmpty() ? GinRummyUtil.getDeadwoodPoints(remainingCards) : GinRummyUtil.getDeadwoodPoints(bestMeldSets.get(0), remainingCards);
			// If we have two or more card to discard with the same rank
			if (deadwood <= minDeadwood) {
				if (deadwood < minDeadwood) {
					minDeadwood = deadwood;
					candidateCards.clear();
				}
				candidateCards.add(card);
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
		assert discardedCard != null;
		// Ignore other player discards.  Remove from cards if playerNum is this player.
		if (playerNum == this.playerNum) {
			// When this player discard, dont add because we have added in the willdrawfaceup card
			cards.remove(discardedCard);
		} else {
			this.op_discard = discardedCard;
			discardPile.add(discardedCard);
		}
		
		if(VERBOSE)
		print_mat1D_card(op_cards, "Opponent Hand before update");
		// Update opponent card estimation for this turn
		update_estimation();
		
		if(VERBOSE)
		print_mat1D_card(op_cards, "Opponent Hand after update");
		
		turn ++;
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
		
		
		if (!opponentKnocked && (bestMeldSets.isEmpty() || knock_prob < knocking_threshhold))
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
	
	@SuppressWarnings("unused")
//	private static void print_mat1D_card(float[] mat, String name) {
//		System.out.println();
//		System.out.println(name + ": ");
//		int a = 0;
//		for (int i = 0; i < mat.length; i++) {
//			// Debugging
//			System.out.printf("%s: %.5f ",Card.getCard(i).toString(), mat[i]);
//			a++;
//			if(a == 13) {
//				a = 0;
//				System.out.println();
//			}
//		}
//		System.out.println();
//	}
	
	public static void print_mat1D_card(float[] mat, String name) {
		System.out.println(name + ": ");
		System.out.print("Rank");
		for (int i = 0; i < Card.NUM_RANKS; i++)
			System.out.print("\t" + Card.rankNames[i]);
		for (int i = 0; i < Card.NUM_CARDS; i++) {
			if (i % Card.NUM_RANKS == 0)
				System.out.printf("\n%s", Card.suitNames[i / Card.NUM_RANKS]);
			System.out.print("\t");
//			System.out.printf("%2.4f", mat[i]);
			System.out.printf("%1.1e", mat[i]);
		}
		System.out.println();
		System.out.println();
	}
	
	public static void main(String[] args) {
		GinRummyPlayer p = new DraftPlayer2();
		
	}
	
}
