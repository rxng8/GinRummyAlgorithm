package collector;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.Stack;
import java.util.regex.Pattern;

import core.Card;
import core.GinRummyPlayer;
import core.GinRummyUtil;
import module.HandEstimator3;

/*
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

public class GinRummyDataCollector {
	/**
	 * Random number generator
	 */
	private static final Random RANDOM = new Random();
	/**
	 * Hand size (before and after turn). After draw and before discard there is one extra card.
	 */
	private static final int HAND_SIZE = 10;
	/**
	 * Whether or not to print information during game play
	 */
	private static boolean playVerbose = false;
	/**
	 * Two Gin Rummy players numbered according to their array index.
	 */
	private GinRummyPlayer[] players = new GinRummyPlayer[2];
	
	final int MAX_TURNS = 100; // TODO - have not determined maximum length legal gin rummy game; truncating if necessary 
//	int[][] drawFaceUpCount = new int[MAX_TURNS][Card.NUM_RANKS]; // indexed by turns taken, face-up card rank
//	double[][][] drawFaceUpRankInSuitFreq = new double[MAX_TURNS][Card.NUM_RANKS][Card.NUM_RANKS]; // indexed by turns taken, face up rank, rank in suit
//	double[][][] drawFaceUpRankNotInSuitFreq = new double[MAX_TURNS][Card.NUM_RANKS][Card.NUM_RANKS]; // indexed by turns taken, face up rank, rank not in suit
//	int[][] drawNonFaceUpCount = new int[MAX_TURNS][Card.NUM_RANKS]; // indexed by turns taken, face-up card rank
//	double[][][] drawNonFaceUpRankInSuitFreq = new double[MAX_TURNS][Card.NUM_RANKS][Card.NUM_RANKS]; // indexed by turns taken, face up rank, rank in suit
//	double[][][] drawNonFaceUpRankNotInSuitFreq = new double[MAX_TURNS][Card.NUM_RANKS][Card.NUM_RANKS]; // indexed by turns taken, face up rank, rank not in suit
//	int[][] discardCount = new int[MAX_TURNS][Card.NUM_RANKS]; // indexed by turns taken, discard card rank
//	double[][][] discardRankInSuitFreq = new double[MAX_TURNS][Card.NUM_RANKS][Card.NUM_RANKS]; // indexed by turns taken, discard rank, rank in suit
//	double[][][] discardRankNotInSuitFreq = new double[MAX_TURNS][Card.NUM_RANKS][Card.NUM_RANKS]; // indexed by turns taken, discard rank, rank not in suit
	public int[] rankCounts = new int[Card.NUM_RANKS];
	public int[][][][] heldVisits = new int[2][2][Card.NUM_RANKS][Card.NUM_RANKS]; // indexed by drawFaceUp (0/1), discardSuited (0/1), faceUpRank, discardRank
	public int[][][][][][] heldCounts = new int[2][2][Card.NUM_RANKS][Card.NUM_RANKS][3][Card.NUM_RANKS]; // indexed by drawFaceUp (0/1), discardSuited (0/1), rank face up, rank discard, suited (0=unsuited with either, 1=suited with face-up, 2=suited with discard), rank 
	final int DRAW_FACE_UP = 1, DRAW_FACE_DOWN = 0, UNSUITED = 0, SUITED_WITH_FACE_UP = 1, SUITED_WITH_DISCARD = 2;
	public int faceDownDrawCount = 0;
	public int immediateDiscardCount = 0;
	
	public void collectData(int turnsTaken, int currentPlayer, Card faceUpCard, Card drawCard, Card discardCard, ArrayList<Card> hand, ArrayList<Card> opponentHand) {
		int faceUpRank = faceUpCard.rank;
		int discardRank = discardCard.rank;
		boolean drawFaceUp = faceUpCard == drawCard;
		int drawFaceUpIndex = drawFaceUp ? 1 : 0;
		boolean discardSuited = discardCard.suit == faceUpCard.suit;
		int discardSuitedIndex = discardSuited ? 1 : 0;
		heldVisits[drawFaceUpIndex][discardSuitedIndex][faceUpRank][discardRank]++;
		for (Card card : hand) {
			rankCounts[card.rank]++;
			int suitedIndex = UNSUITED;
			if (card.suit == faceUpCard.suit)
				suitedIndex = SUITED_WITH_FACE_UP;
			else if (card.suit == discardCard.suit)
				suitedIndex = SUITED_WITH_DISCARD;
			heldCounts[drawFaceUpIndex][discardSuitedIndex][faceUpRank][discardRank][suitedIndex][card.rank]++;
		}
		if (drawCard != faceUpCard) {
			faceDownDrawCount++;
			if (drawCard == discardCard)
				immediateDiscardCount++;
		}
	}

	public void displayData() {
		System.out.println("Probability of immediate discard of face-down draw: " + (double) immediateDiscardCount / faceDownDrawCount);
		for (int faceUp = 0; faceUp <= 1; faceUp++) {
			boolean drawFaceUp = faceUp == 1;
			for (int c1 = 0; c1 < Card.NUM_RANKS; c1++)
				for (int c2 = 0; c2 < 2 * Card.NUM_RANKS; c2++) {
					if (c1 == c2) continue;
					Card faceUpCard = Card.allCards[c1];
					Card discardCard = Card.allCards[c2];
					int faceUpRank = faceUpCard.rank;
					int discardRank = discardCard.rank;
					int drawFaceUpIndex = drawFaceUp ? 1 : 0;
					boolean discardSuited = discardCard.suit == faceUpCard.suit;
					int discardSuitedIndex = discardSuited ? 1 : 0;
					System.out.printf("Face-up card: %s  Drawn face-up? %s  Discard card: %s\n", faceUpCard, drawFaceUp, discardCard);
					System.out.println("Visits: " + heldVisits[drawFaceUpIndex][discardSuitedIndex][faceUpRank][discardRank]);
					System.out.print("Rank\t");
					for (int i = 0; i < Card.NUM_RANKS; i++)
						System.out.print("\t" + Card.rankNames[i]);
					System.out.println();
					String[] suitedLabels = {"Unsuited", "Face-up suited", "Discard suited"};
					for (int suitedIndex = 0; suitedIndex <= 2; suitedIndex++) {
						if (discardSuited && suitedIndex == 2) break;
						System.out.print(suitedLabels[suitedIndex]);
						for (int i = 0; i < Card.NUM_RANKS; i++)
							System.out.printf("\t%.3f", (double) heldCounts[drawFaceUpIndex][discardSuitedIndex][faceUpRank][discardRank][suitedIndex][i] / heldVisits[drawFaceUpIndex][discardSuitedIndex][faceUpRank][discardRank]);
						System.out.println();
					}
				}
		}
	}
	
	/**
	 * Set whether or not there is to be printed output during gameplay.
	 * @param playVerbose whether or not there is to be printed output during gameplay
	 */
	public static void setPlayVerbose(boolean playVerbose) {
		GinRummyDataCollector.playVerbose = playVerbose;
	}

	/**
	 * Play a game of Gin Rummy and return the winning player number 0 or 1.
	 * @return the winning player number 0 or 1
	 */
	@SuppressWarnings("unchecked")
	public int play(int startingPlayer) {
		int[] scores = new int[2];
		ArrayList<ArrayList<Card>> hands = new ArrayList<ArrayList<Card>>();
		hands.add(new ArrayList<Card>());
		hands.add(new ArrayList<Card>());
//		int startingPlayer = RANDOM.nextInt(2);
		
		while (scores[0] < GinRummyUtil.GOAL_SCORE && scores[1] <  GinRummyUtil.GOAL_SCORE) { // while game not over
			int currentPlayer = startingPlayer;
			int opponent = (currentPlayer == 0) ? 1 : 0;
			
			// get shuffled deck and deal cards
			Stack<Card> deck = Card.getShuffle(RANDOM.nextInt());
			hands.get(0).clear();
			hands.get(1).clear();
			for (int i = 0; i < 2 * HAND_SIZE; i++)
				hands.get(i % 2).add(deck.pop());
			for (int i = 0; i < 2; i++) {
				Card[] handArr = new Card[HAND_SIZE];
				hands.get(i).toArray(handArr);
				players[i].startGame(i, startingPlayer, handArr); 
				if (playVerbose)
					System.out.printf("Player %d is dealt %s.\n", i, hands.get(i));
			}
			if (playVerbose)
				System.out.printf("Player %d starts.\n", startingPlayer);
			Stack<Card> discards = new Stack<Card>();
			discards.push(deck.pop());
			if (playVerbose)
				System.out.printf("The initial face up card is %s.\n", discards.peek());
			Card firstFaceUpCard = discards.peek();
			int turnsTaken = 0;
			ArrayList<ArrayList<Card>> knockMelds = null;
			while (deck.size() > 2) { // while the deck has more than two cards remaining, play round
				// DRAW
				boolean drawFaceUp = false;
				Card faceUpCard = discards.peek();
				// offer draw face-up iff not 3rd turn with first face up card (decline automatically in that case) 
				if (!(turnsTaken == 3 && faceUpCard == firstFaceUpCard)) { // both players declined and 1st player must draw face down
					drawFaceUp = players[currentPlayer].willDrawFaceUpCard(faceUpCard);
					if (playVerbose && !drawFaceUp && faceUpCard == firstFaceUpCard && turnsTaken < 2)
						System.out.printf("Player %d declines %s.\n", currentPlayer, firstFaceUpCard);
				}
				if (!(!drawFaceUp && turnsTaken < 2 && faceUpCard == firstFaceUpCard)) { // continue with turn if not initial declined option
					Card drawCard = drawFaceUp ? discards.pop() : deck.pop();
					for (int i = 0; i < 2; i++) 
						players[i].reportDraw(currentPlayer, (i == currentPlayer || drawFaceUp) ? drawCard : null);
					if (playVerbose)
						System.out.printf("Player %d draws %s.\n", currentPlayer, drawCard);
					hands.get(currentPlayer).add(drawCard);
					
					// DISCARD
					Card discardCard = players[currentPlayer].getDiscard();
					if (!hands.get(currentPlayer).contains(discardCard) || discardCard == faceUpCard) {
						if (playVerbose)
							System.out.printf("Player %d discards %s illegally and forfeits.\n", currentPlayer, discardCard);
						return opponent;
					}
					hands.get(currentPlayer).remove(discardCard);
					for (int i = 0; i < 2; i++) 
						players[i].reportDiscard(currentPlayer, discardCard);
					if (playVerbose)
						System.out.printf("Player %d discards %s.\n", currentPlayer, discardCard);
					discards.push(discardCard);
					collectData(turnsTaken, currentPlayer, faceUpCard, drawCard, discardCard, hands.get(currentPlayer), hands.get(opponent));
					if (playVerbose) {
						ArrayList<Card> unmeldedCards = (ArrayList<Card>) hands.get(currentPlayer).clone();
						ArrayList<ArrayList<ArrayList<Card>>> bestMelds = GinRummyUtil.cardsToBestMeldSets(unmeldedCards);
						if (bestMelds.isEmpty()) 
							System.out.printf("Player %d has %s with %d deadwood.\n", currentPlayer, unmeldedCards, GinRummyUtil.getDeadwoodPoints(unmeldedCards));
						else {
							ArrayList<ArrayList<Card>> melds = bestMelds.get(0);
							for (ArrayList<Card> meld : melds)
								for (Card card : meld)
									unmeldedCards.remove(card);
							melds.add(unmeldedCards);
							System.out.printf("Player %d has %s with %d deadwood.\n", currentPlayer, melds, GinRummyUtil.getDeadwoodPoints(unmeldedCards));
						}
					}
						
					// CHECK FOR KNOCK 
					knockMelds = players[currentPlayer].getFinalMelds();
					if (knockMelds != null)
						break; // player knocked; end of round
				}

				turnsTaken++;
				currentPlayer = (currentPlayer == 0) ? 1 : 0;
				opponent = (currentPlayer == 0) ? 1 : 0;
			}
			
			if (knockMelds != null) { // round didn't end due to non-knocking and 2 cards remaining in draw pile
				// check legality of knocking meld
				long handBitstring = GinRummyUtil.cardsToBitstring(hands.get(currentPlayer));
				long unmelded = handBitstring;
				for (ArrayList<Card> meld : knockMelds) {
					long meldBitstring = GinRummyUtil.cardsToBitstring(meld);
					if (!GinRummyUtil.getAllMeldBitstrings().contains(meldBitstring) // non-meld ...
							|| (meldBitstring & unmelded) != meldBitstring) { // ... or meld not in hand
						if (playVerbose)
							System.out.printf("Player %d melds %s illegally and forfeits.\n", currentPlayer, knockMelds);
						return opponent;
					}
					unmelded &= ~meldBitstring; // remove successfully melded cards from 
				}
				// compute knocking deadwood
				int knockingDeadwood = GinRummyUtil.getDeadwoodPoints(knockMelds, hands.get(currentPlayer));
				if (knockingDeadwood > GinRummyUtil.MAX_DEADWOOD) {
					if (playVerbose)
						System.out.printf("Player %d melds %s with greater than %d deadwood and forfeits.\n", currentPlayer, knockMelds, knockingDeadwood);				
					return opponent;
				}
				
				ArrayList<ArrayList<Card>> meldsCopy = new ArrayList<ArrayList<Card>>();
				for (ArrayList<Card> meld : knockMelds)
					meldsCopy.add((ArrayList<Card>) meld.clone());
				for (int i = 0; i < 2; i++) 
					players[i].reportFinalMelds(currentPlayer, meldsCopy);
				if (playVerbose)
					if (knockingDeadwood > 0) 
						System.out.printf("Player %d melds %s with %d deadwood from %s.\n", currentPlayer, knockMelds, knockingDeadwood, GinRummyUtil.bitstringToCards(unmelded));
					else
						System.out.printf("Player %d goes gin with melds %s.\n", currentPlayer, knockMelds);

				// get opponent meld
				ArrayList<ArrayList<Card>> opponentMelds = players[opponent].getFinalMelds();
				for (ArrayList<Card> meld : opponentMelds)
					meldsCopy.add((ArrayList<Card>) meld.clone());
				meldsCopy = new ArrayList<ArrayList<Card>>();
				for (int i = 0; i < 2; i++) 
					players[i].reportFinalMelds(opponent, meldsCopy);
				
				// check legality of opponent meld
				long opponentHandBitstring = GinRummyUtil.cardsToBitstring(hands.get(opponent));
				long opponentUnmelded = opponentHandBitstring;
				for (ArrayList<Card> meld : opponentMelds) {
					long meldBitstring = GinRummyUtil.cardsToBitstring(meld);
					if (!GinRummyUtil.getAllMeldBitstrings().contains(meldBitstring) // non-meld ...
							|| (meldBitstring & opponentUnmelded) != meldBitstring) { // ... or meld not in hand
						if (playVerbose)
							System.out.printf("Player %d melds %s illegally and forfeits.\n", opponent, opponentMelds);
						return currentPlayer;
					}
					opponentUnmelded &= ~meldBitstring; // remove successfully melded cards from 
				}
				if (playVerbose)
					System.out.printf("Player %d melds %s.\n", opponent, opponentMelds);

				// lay off on knocking meld (if not gin)
				ArrayList<Card> unmeldedCards = GinRummyUtil.bitstringToCards(opponentUnmelded);
				if (knockingDeadwood > 0) { // knocking player didn't go gin
					boolean cardWasLaidOff;
					do { // attempt to lay each card off
						cardWasLaidOff = false;
						Card layOffCard = null;
						ArrayList<Card> layOffMeld = null;
						for (Card card : unmeldedCards) {
							for (ArrayList<Card> meld : knockMelds) {
								ArrayList<Card> newMeld = (ArrayList<Card>) meld.clone();
								newMeld.add(card);
								long newMeldBitstring = GinRummyUtil.cardsToBitstring(newMeld);
								if (GinRummyUtil.getAllMeldBitstrings().contains(newMeldBitstring)) {
									layOffCard = card;
									layOffMeld = meld;
									break;
								}
							}
							if (layOffCard != null) {
								if (playVerbose)
									System.out.printf("Player %d lays off %s on %s.\n", opponent, layOffCard, layOffMeld);
								unmeldedCards.remove(layOffCard);
								layOffMeld.add(layOffCard);
								cardWasLaidOff = true;
								break;
							}
								
						}
					} while (cardWasLaidOff);
				}
				int opponentDeadwood = 0;
				for (Card card : unmeldedCards)
					opponentDeadwood += GinRummyUtil.getDeadwoodPoints(card);
				if (playVerbose)
					System.out.printf("Player %d has %d deadwood with %s\n", opponent, opponentDeadwood, unmeldedCards); 

				// compare deadwood and compute new scores
				if (knockingDeadwood == 0) { // gin round win
					scores[currentPlayer] += GinRummyUtil.GIN_BONUS + opponentDeadwood;
					if (playVerbose)
						System.out.printf("Player %d scores the gin bonus of %d plus opponent deadwood %d for %d total points.\n", currentPlayer, GinRummyUtil.GIN_BONUS, opponentDeadwood, GinRummyUtil.GIN_BONUS + opponentDeadwood); 
				}
				else if (knockingDeadwood < opponentDeadwood) { // non-gin round win
					scores[currentPlayer] += opponentDeadwood - knockingDeadwood;
					if (playVerbose)
						System.out.printf("Player %d scores the deadwood difference of %d.\n", currentPlayer, opponentDeadwood - knockingDeadwood); 
				}
				else { // undercut win for opponent
					scores[opponent] += GinRummyUtil.UNDERCUT_BONUS + knockingDeadwood - opponentDeadwood;
					if (playVerbose)
						System.out.printf("Player %d undercuts and scores the undercut bonus of %d plus deadwood difference of %d for %d total points.\n", opponent, GinRummyUtil.UNDERCUT_BONUS, knockingDeadwood - opponentDeadwood, GinRummyUtil.UNDERCUT_BONUS + knockingDeadwood - opponentDeadwood); 
				}
				startingPlayer = (startingPlayer == 0) ? 1 : 0; // starting player alternates
			}
			else { // If the round ends due to a two card draw pile with no knocking, the round is cancelled.
				if (playVerbose)
					System.out.println("The draw pile was reduced to two cards without knocking, so the hand is cancelled.");
			}

			// score reporting
			if (playVerbose) 
				System.out.printf("Player\tScore\n0\t%d\n1\t%d\n", scores[0], scores[1]);
			for (int i = 0; i < 2; i++) 
				players[i].reportScores(scores.clone());
		}
		if (playVerbose)
			System.out.printf("Player %s wins.\n", scores[0] > scores[1] ? 0 : 1);
		return scores[0] >= GinRummyUtil.GOAL_SCORE ? 0 : 1;
	}
	
	private void match(String player1Name, String player2Name, int numGames) throws InstantiationException, IllegalAccessException, ClassNotFoundException, FileNotFoundException, UnsupportedEncodingException {
		String filename = "match-" + numGames + "-" + player1Name + "-" + player2Name + ".txt";
		PrintWriter writer = new PrintWriter(filename, "UTF-8");
		
		writer.printf("Match %s vs. %s (%d games) ... ", player1Name, player2Name, numGames);
		writer.flush();
		@SuppressWarnings("unchecked")
		Class<GinRummyPlayer> playerClass1 = (Class<GinRummyPlayer>) Class.forName(player1Name);
		players[0] = playerClass1.newInstance();
		@SuppressWarnings("unchecked")
		Class<GinRummyPlayer> playerClass2 = (Class<GinRummyPlayer>) Class.forName(player2Name);
		players[1] = playerClass2.newInstance();
		long startMs = System.currentTimeMillis();
		int numP1Wins = 0;
		for (int i = 0; i < numGames; i++) {
			numP1Wins += play(i % 2);
		}
		long totalMs = System.currentTimeMillis() - startMs;
		writer.printf("%d games played in %d ms.\n", numGames, totalMs);
		writer.printf("Games Won: %s:%d, %s:%d.\n", player1Name, numGames - numP1Wins, player2Name, numP1Wins);
		System.out.printf("%d games played in %d ms.\n", numGames, totalMs);
		System.out.printf("Games Won: %s:%d, %s:%d.\n", player1Name, numGames - numP1Wins, player2Name, numP1Wins);
		writer.close();
	}
	
	private static String getComputerName()
	{
		// From: http://stackoverflow.com/questions/7883542/getting-the-computer-name-in-java
	    Map<String, String> env = System.getenv();
	    if (env.containsKey("COMPUTERNAME"))
	        return env.get("COMPUTERNAME");
	    else if (env.containsKey("HOSTNAME"))
	        return env.get("HOSTNAME");
	    else {
	    	String hostname = "Unknown";
	    	try
	    	{
	    	    InetAddress addr;
	    	    addr = InetAddress.getLocalHost();
	    	    hostname = addr.getHostName();
	    	}
	    	catch (UnknownHostException ex)
	    	{
	    	    System.err.println("Hostname can not be resolved");
	    	}
	    	return hostname;
	    }
	}
	
	/**
	 * Test and demonstrate the use of the GinRummyGame class.
	 * @param args (unused)
	 * @throws ClassNotFoundException 
	 * @throws IllegalAccessException 
	 * @throws InstantiationException 
	 * @throws UnsupportedEncodingException 
	 * @throws FileNotFoundException 
	 */
	public static void main(String[] args) throws InstantiationException, IllegalAccessException, ClassNotFoundException, FileNotFoundException, UnsupportedEncodingException {
//		String[] playerNames = {"SimpleGinRummyPlayer", "EvenSimplerPlayer", "GinRummyPlayer2", "MyFinalGinRummyPlayer", "NewGinRummyPlayer", "OurGinRummyPlayer"};
		String[] playerNames = {"player.SimplePlayer", "player.SimplePlayer"};
		String computerName = getComputerName();
		Scanner in = new Scanner(computerName);
		int computerNum = Integer.parseInt(in.findInLine(Pattern.compile("[1-9][0-9]*")));
		in.close();
		System.out.println(computerNum);
		int minComputerNum = 1;
		int index = computerNum - minComputerNum;
		index = 0; // TODO: comment out to use computer names
		int index1 = 0, index2 = 0;
		for (int i = 0; i < playerNames.length && index >= 0; i++)
			for (int j = i + 1; j < playerNames.length && index >= 0; j++) {
				if (index == 0) {
					index1 = i;
					index2 = j;
				}
				index--;
			}
		setPlayVerbose(false);
		int numGames = 10000;
		GinRummyDataCollector collector = new GinRummyDataCollector();
		collector.match(playerNames[index1], playerNames[index2], numGames);
		collector.displayData();
		HandEstimator3 handEst;
		handEst = new HandEstimator3(collector);
		handEst.save("handEst1-" + numGames + ".dat");
		handEst = new HandEstimator3("handEst1-" + numGames + ".dat");
		handEst.test();
	}
}
