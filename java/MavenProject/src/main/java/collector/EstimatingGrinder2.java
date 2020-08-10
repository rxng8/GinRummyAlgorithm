package collector;

import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.Stack;
import java.util.regex.Pattern;

import org.bytedeco.javacpp.lept.alloc_fn;

import com.opencsv.CSVWriter;

import core.*;
import module.*;
import player.*;
import util.*;

public class EstimatingGrinder2 {
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

	final int DRAW_FACE_UP = 1, DRAW_FACE_DOWN = 0, UNSUITED = 0, SUITED_WITH_FACE_UP = 1, SUITED_WITH_DISCARD = 2;

	/**
	 * This arrayList contain lines of data
	 * Each line of data is an integer array:
	 * 		[0]: Rank of the faceup card.
	 * 		[1]: I pick or not?
	 * 		[2]: Rank of my discarded card.
	 * 		[3]: What is the estimated card rank
	 * 		[4]: If the estimating card suit match face up card suit
	 * 		[5]: If the estimating card suit match discard card suit
	 * 		[6]: LABEL: Whether the estimated card is in my hand or not!
	 */
	ArrayList<int[]> total_data;
	
	public EstimatingGrinder2() {
		total_data = new ArrayList<>();
	}
	
	public void collectData(int turnsTaken, int currentPlayer, Card faceUpCard, Card drawCard, Card discardCard, ArrayList<Card> hand, ArrayList<Card> opponentHand) {
		
		
		for (int i = 0; i < Card.NUM_CARDS; i++) {
			int[] line = new int[7];
			
			line[0] = faceUpCard.rank;
			line[1] = faceUpCard == drawCard ? 1 : 0;
			line[2] = discardCard.rank;
			
			// Estimated card rank
			// line[4]
			Card estimatingCard = Card.getCard(i);
			line[3] = estimatingCard.rank;
			line[4] = estimatingCard.suit == faceUpCard.suit ? 1 : 0;
			line[5] = estimatingCard.suit == discardCard.suit ? 1 : 0;
			
			// Check contain in hand?
			boolean contain = false;
			for (Card c : hand) {
				if (c.getId() == estimatingCard.getId()) {
					contain = true;
					break;
				}
			}
			
			line[6] = contain ? 1 : 0;
			
			total_data.add(line);
		}
	}

	public void displayData() {
		System.out.printf("Data:\n");
		System.out.printf("UpRank\tPicked?\tDisRank\tSRank\tSsUp\tSsDis\tLabel\n");
		for (int i = 0; i < total_data.size(); i++) {
			int[] datum = total_data.get(i);
			if (datum[1] == 0) continue;
			for (int j = 0; j < datum.length; j++) {
				System.out.printf("%d\t", datum[j]);
			}
			System.out.println();
		} 
	}
	
	/**
	 * Set whether or not there is to be printed output during gameplay.
	 * @param playVerbose whether or not there is to be printed output during gameplay
	 */
	public static void setPlayVerbose(boolean playVerbose) {
		EstimatingGrinder2.playVerbose = playVerbose;
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
		
		while (scores[0] < GinRummyUtil.GOAL_SCORE && scores[1] < GinRummyUtil.GOAL_SCORE) { // while game not over
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
	
	@SuppressWarnings("unused")
	private void match(GinRummyPlayer player1, GinRummyPlayer player2, int numGames) throws InstantiationException, IllegalAccessException, ClassNotFoundException, FileNotFoundException, UnsupportedEncodingException {
		
		
		System.out.printf("Match %s vs. %s (%d games) ... ", player1, player2, numGames);
		
		players[0] = player1;
		players[1] = player2;
		
		long startMs = System.currentTimeMillis();
		int numP1Wins = 0;
		for (int i = 0; i < numGames; i++) {
			numP1Wins += play(i % 2);
		}
		long totalMs = System.currentTimeMillis() - startMs;
		System.out.printf("%d games played in %d ms.\n", numGames, totalMs);
		System.out.printf("Games Won: %s:%d, %s:%d.\n", player1, numGames - numP1Wins, player2, numP1Wins);
		System.out.printf("%d games played in %d ms.\n", numGames, totalMs);
		System.out.printf("Games Won: %s:%d, %s:%d.\n", player1, numGames - numP1Wins, player2, numP1Wins);
	}
	
	/**
	 * 
	 * @param filename name of file
	 * @param cont Appending to the data file
	 */
	public void to_CSV(String filename, boolean cont) {
		//Instantiating the CSVWriter class
		
		try {
			CSVWriter writer;
			writer = new CSVWriter(new FileWriter(filename, cont));
		
			System.out.println("Writing data...");
			
			//Writing data to a csv file
			        
			// Header
			if (!cont) {
				String[] headers = new String[7];
				headers[0] = "Face Up Rank";
				headers[1] = "Picked?";
				headers[2] = "Discard Rank";
				headers[3] = "Estimating Card Rank";
				headers[4] = "Estimating suit is face Up suit";
				headers[5] = "Estimating suit is discard suit";
				headers[6] = "Label";
				writer.writeNext(headers);
			}
			
			for (int i = 0; i < total_data.size(); i++) {
				String[] line = new String[7];
				
				
				String strArray[] = Arrays.stream(total_data.get(i))
							.mapToObj(String::valueOf)
							.toArray(String[]::new);
				System.arraycopy(strArray, 0, line, 0, strArray.length);
				
				writer.writeNext(line);
			}
			
			writer.close();
		    System.out.println("Data entered!!!");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
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

		setPlayVerbose(false);
		int numGames = 10000;
		EstimatingGrinder2 collector = new EstimatingGrinder2();
		GinRummyPlayer p0 = new SimplePlayer();
		GinRummyPlayer p1 = new SimplePlayer();
		collector.match(p0, p1, numGames);
		
//		collector.displayData();
		
		// export data
		String filename = "estimating_10000_v1.csv";
		collector.to_CSV("dataset/" + filename, false);
	}
}
