
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.Stack;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import javax.xml.crypto.Data;

import com.opencsv.CSVWriter;

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

public class DataCollectorLinear implements DataCollector {
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

	@SuppressWarnings("unchecked")
	ArrayList<ArrayList<Integer>> labels = new ArrayList<>();
	
	
	ArrayList<ArrayList<ArrayList<int[]>>> hands;
	
	int turnsTaken = 0;
	
	
	
	public DataCollectorLinear () {
		labels.add(new ArrayList<Integer>());
		labels.add(new ArrayList<Integer>());
		
		hands = new ArrayList<ArrayList<ArrayList<int[]>>>();
		hands.add(new ArrayList<ArrayList<int[]>>());
		hands.add(new ArrayList<ArrayList<int[]>>()); 
	}
	
	
	public void collectLabel(int turnsTaken, int currentPlayer, boolean won) {
		int opponent = 1 - currentPlayer;
		labels.get(currentPlayer).add(won ? 1 : 0);
		labels.get(opponent).add(won? 0 : 1);
		
		// Add new arraylist of turns (match) to the hand list
		ArrayList<ArrayList<int[]>> player_game = hands.get(currentPlayer);
		player_game.add(new ArrayList<int[]>());
	}
	
	public void collectData(int turnsTaken, int currentPlayer, Card faceUpCard, Card drawCard, Card discardCard, ArrayList<Card> hand, ArrayList<Card> opponentHand) {

		int[] hand_arr = new int[52];
		for (Card c : hand) {
			hand_arr[c.getId()] = 1;
		}
		ArrayList<ArrayList<int[]>> player_game = hands.get(currentPlayer);
		ArrayList<int[]> player_match;
		if (player_game.size() != 0) {
			player_match = player_game.get(player_game.size() - 1);
			player_match.add(hand_arr);
		} else {
			player_match = new ArrayList<int[]>();
			player_match.add(hand_arr);
			player_game.add(player_match);
		}
		
		assert player_match != null: "duhhh";
	}
	
	@SuppressWarnings({ "rawtypes", "unchecked" })
	public void toCSV(String filename) throws IOException {

	      //Instantiating the CSVWriter class
	      CSVWriter writer = new CSVWriter(new FileWriter(filename));
	      //Writing data to a csv file
	            
	      // Header
//	      String[] headers = new String[105];
	      String[] headers = new String[53];
//	      for (int p = 0; p < 2; p++) {
	      for (int p = 0; p < 1; p++) {
	    	  for (int i = 0; i < 52; i++) {
		    	  String header = Card.getCard(i).toString();
		    	  headers[i] = header;
		      }
	      }
//	      headers[104] = "Label";
	      headers[52] = "Label";
	      writer.writeNext(headers);
	      
	      System.out.println("Writing to file ...");
	      
    	  ArrayList<ArrayList<int[]>> player_matches = hands.get(0);
    	  ArrayList<Integer> player_labels = labels.get(0);
    	  
    	  ArrayList<ArrayList<int[]>> opp_matches = hands.get(0);
//    	  ArrayList<Integer> opp_labels = labels.get(0);
    	  
    	  int n_matches = player_matches.size();
    	  assert n_matches == player_labels.size() : "Duh";
    	  

		  for (int match = 0; match < n_matches; match++) {
			  for (int turn_id = 0; turn_id < player_matches.get(match).size(); turn_id++) {
//				  String[] line = new String[105];
				  String[] line = new String[53];
				  String strArray_c[] = Arrays.stream(player_matches.get(match).get(turn_id))
							.mapToObj(String::valueOf)
							.toArray(String[]::new);
				  assert strArray_c.length == 52 : "Duhhh!";
				  System.arraycopy(strArray_c, 0, line, 0, strArray_c.length);
				  
				  
//				  String strArray_o[] = Arrays.stream(opp_matches.get(match).get(turn_id))
//							.mapToObj(String::valueOf)
//							.toArray(String[]::new);
//				  assert strArray_c.length == 52 : "Duhhh!";
//				  assert strArray_o.length == 52 : "Duhhh!";
//				  System.arraycopy(strArray_o, 0, line, 52, strArray_o.length);
				  
//				  line[104] = Integer.toString(player_labels.get(match));
				  line[52] = Integer.toString(player_labels.get(match));
				  writer.writeNext(line);
			  }
			  System.out.println("Written one match in match " + match);
		  }
		  System.out.println("Written one player!");
	      
	      
	      //Writing data to the csv file
//	      writer.writeAll(list);
//	      writer.flush();
	      writer.close();
	      System.out.println("Data entered!!!");
	}

	public void displayData() {
		
		for (ArrayList<ArrayList<int[]>> player : hands) {
			
			for (ArrayList<int[]> game: player) {
				for(int[] match : game) {
					
					Util.print_mat1D_card(match, "Turn");
				}
			}
		}
		for (ArrayList<Integer> label : labels) {
			Stream.of(label).forEach(win -> System.out.print(win + " "));
			System.out.println();
		}
	}
	
	/**
	 * Set whether or not there is to be printed output during gameplay.
	 * @param playVerbose whether or not there is to be printed output during gameplay
	 */
	public static void setPlayVerbose(boolean playVerbose) {
		DataCollectorLinear.playVerbose = playVerbose;
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
					collectLabel(turnsTaken, currentPlayer, true);
					
					
					if (playVerbose)
						System.out.printf("Player %d scores the gin bonus of %d plus opponent deadwood %d for %d total points.\n", currentPlayer, GinRummyUtil.GIN_BONUS, opponentDeadwood, GinRummyUtil.GIN_BONUS + opponentDeadwood); 
				}
				else if (knockingDeadwood < opponentDeadwood) { // non-gin round win
					scores[currentPlayer] += opponentDeadwood - knockingDeadwood;
					collectLabel(turnsTaken, currentPlayer, true);

					
					
					if (playVerbose)
						System.out.printf("Player %d scores the deadwood difference of %d.\n", currentPlayer, opponentDeadwood - knockingDeadwood); 
				}
				else { // undercut win for opponent
					scores[opponent] += GinRummyUtil.UNDERCUT_BONUS + knockingDeadwood - opponentDeadwood;
					collectLabel(turnsTaken, currentPlayer, false);
					
					
					
					if (playVerbose)
						System.out.printf("Player %d undercuts and scores the undercut bonus of %d plus deadwood difference of %d for %d total points.\n", opponent, GinRummyUtil.UNDERCUT_BONUS, knockingDeadwood - opponentDeadwood, GinRummyUtil.UNDERCUT_BONUS + knockingDeadwood - opponentDeadwood); 
				}
				
				
				collectData(turnsTaken, currentPlayer, null, null, null, hands.get(currentPlayer), hands.get(opponent));
				
				
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
			
			
//			break;
			
			
		}
		if (playVerbose)
			System.out.printf("Player %s wins.\n", scores[0] > scores[1] ? 0 : 1);
		return scores[0] >= GinRummyUtil.GOAL_SCORE ? 0 : 1;
	}
	
	public void match(GinRummyPlayer p0, GinRummyPlayer p1, int numGames) {
		
		this.players[0] = p0;
		this.players[1] = p1;

		long startMs = System.currentTimeMillis();
		int numP1Wins = 0;
		for (int i = 0; i < numGames; i++) {
			numP1Wins += play(i % 2);
		}
		long totalMs = System.currentTimeMillis() - startMs;

		System.out.printf("%d games played in %d ms.\n", numGames, totalMs);
		System.out.printf("Games Won: %s: %d, %s: %d.\n", "Player1", numGames - numP1Wins, "Player2", numP1Wins);

	}
	
	/**
	 * Test and demonstrate the use of the GinRummyGame class.
	 * @param args (unused)
	 * @throws ClassNotFoundException 
	 * @throws IllegalAccessException 
	 * @throws InstantiationException 
	 * @throws IOException 
	 */
	public static void main(String[] args) throws InstantiationException, IllegalAccessException, ClassNotFoundException, IOException {
		setPlayVerbose(false);
		System.out.println("Playing games...");
		int numGames = 50000;
		DataCollectorLinear collector = new DataCollectorLinear();
		
		GinRummyPlayer p0 = new SimplePlayer2();
		GinRummyPlayer p1 = new SimpleGinRummyPlayer();
		
		collector.match(p0, p1, numGames);
		

		collector.toCSV("data_linear.csv");
	}
}