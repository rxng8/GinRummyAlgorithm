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
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.Stack;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import javax.xml.crypto.Data;

import com.opencsv.CSVWriter;
import core.*;
import module.*;
import player.*;
import util.*;




public class BayesGrinder extends DataGrinder {

	int turnsTaken = 0;
	
	// Players / Matches / Turns / hand, pick/unpick, discard / cards
	ArrayList<int[][]> picking_data;
	
	// Players / Matches / Turns / Picked or not
	ArrayList<Integer> picking_labels;
	
	
	public BayesGrinder () {
		
		picking_data = new ArrayList<>();
		picking_labels = new ArrayList<>();

	}
	
	public void collectDataLabel(int turnsTaken, int currentPlayer, Card faceUpCard, Card drawCard, Card discardCard, ArrayList<Card> hand, ArrayList<Card> opponentHand) {
		//Data
		int[] hand_arr = new int[52];
		for (Card c : hand) {
			if (faceUpCard.getId() == c.getId()) continue;
			hand_arr[c.getId()] = 1;
		}
		int[] faceUp = new int[52];
		faceUp[faceUpCard.getId()] = 1;
		// Comment this part to take the draw card only
		int rank = faceUpCard.getRank();
		int suit = faceUpCard.getSuit();
		// Mask all suit
		for (int i = 0; i < 4; i++) {
			faceUp[i * 13 + rank] = 1;
		}
		// Mask all rank
		for (int i = 0; i < 13; i++) {
			faceUp[suit * 13 + 1] = 1;
		}
		// ----------^
		
		int[] disc = new int[52];
		
		// Comment this part to take the discard card only
		disc[discardCard.getId()] = 1;
		rank = discardCard.getRank();
		suit = discardCard.getSuit();
		// Mask all suit
		for (int i = 0; i < 4; i++) {
			disc[i * 13 + rank] = 1;
		}
		// Mask all rank
		for (int i = 0; i < 13; i++) {
			disc[suit * 13 + 1] = 1;
		}
		// ----^
		
		int[][] line = {hand_arr, faceUp, disc};
		
		picking_data.add(line);
		picking_labels.add(drawCard.getId() == faceUpCard.getId() ? 1 : 0);
		
	}
	
	@SuppressWarnings({ "rawtypes", "unchecked" })
	private void toCSV_picking(String filename, boolean cont) throws IOException {
		
		ArrayList<int[][]> data = this.picking_data;
		ArrayList<Integer> labels = this.picking_labels;
		
		//Instantiating the CSVWriter class
		CSVWriter writer = new CSVWriter(new FileWriter(filename, cont));
		//Writing data to a csv file
		        
		// Header
		String[] headers = new String[105];
		for (int p = 0; p < 2; p++) {
			for (int i = 0; i < 52; i++) {
		    	String header = Card.getCard(i).toString();
		    	headers[52*p + i] = header;
		    }
		}
		headers[104] = "Label";
		writer.writeNext(headers);
	    
		assert data.size() == labels.size() : "duh!!";
		
		for (int i = 0; i < data.size(); i++) {
			String[] line = new String[105];
			// Take only 2 first features
			for (int feature = 0; feature < data.get(i).length - 1; feature++) {
				  String strArray[] = Arrays.stream(data.get(i)[feature])
							.mapToObj(String::valueOf)
							.toArray(String[]::new);
				  System.arraycopy(strArray, 0, line, 52 * feature, strArray.length);
			}
			line[104] = String.valueOf(labels.get(i));
			writer.writeNext(line);
		}
		
		writer.close();
	    System.out.println("Data entered!!!");
	}
	
	private void toCSV_discard(String filename, boolean cont) throws IOException {
		
		ArrayList<int[][]> data = this.picking_data;
		ArrayList<Integer> labels = this.picking_labels;
		
		//Instantiating the CSVWriter class
		CSVWriter writer = new CSVWriter(new FileWriter(filename, cont));
		//Writing data to a csv file
		        
		// Header
		String[] headers = new String[105];
		for (int p = 0; p < 2; p++) {
			for (int i = 0; i < 52; i++) {
		    	String header = Card.getCard(i).toString();
		    	headers[52*p + i] = header;
		    }
		}
		headers[104] = "Label";
		writer.writeNext(headers);
	    
		assert data.size() == labels.size() : "duh!!";
		
		for (int i = 0; i < data.size(); i++) {
			String[] line = new String[105];
			// Take only feature 1 and feature 3 first features
			for (int feature = 0; feature < data.get(i).length - 1; feature++) {
				int tmp_feature = 0;
				if (feature == 1) tmp_feature = 2;
				String strArray[] = Arrays.stream(data.get(i)[tmp_feature])
							.mapToObj(String::valueOf)
							.toArray(String[]::new);
				System.arraycopy(strArray, 0, line, 52 * feature, strArray.length);
			}
			line[104] = String.valueOf(labels.get(i));
			writer.writeNext(line);
		}
		
		writer.close();
	    System.out.println("Data entered!!!");
	}
	
	public void displayData() {
		
//		for (ArrayList<ArrayList<int[]>> player : hands) {
//			
//			for (ArrayList<int[]> game: player) {
//				for(int[] match : game) {
//					
//					Util.print_mat1D_card(match, "Turn");
//				}
//			}
//		}
//		
//		for (int turn = 0; turn < picking_data.size(); turn++) {
//			int[][] turn_arr = picking_data.get(turn);
//			
//			Util.print_mat1D_card(turn_arr[0], "Hand");
//			Util.print_mat1D_card(turn_arr[1], "Face Up Card");
//			Util.print_mat1D_card(turn_arr[2], "Card to be discarded");
//			
//			int label = picking_labels.get(turn);
//			System.out.println("This player have " + (label == 1 ? "picked the card!" : "not picked the card!"));
//		}
		
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
					

					collectDataLabel(turnsTaken, currentPlayer, faceUpCard, drawCard, discardCard, hands.get(currentPlayer), hands.get(opponent));
					
					
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
		BayesGrinder collector = new BayesGrinder();
		
		GinRummyPlayer p0 = new DraftPlayer2();
		GinRummyPlayer p1 = new SimplePlayer();
		
		collector.match(p0, p1, numGames);
		
		
//		collector.displayData_picking();
		
		collector.toCSV_picking("data_picking.csv", false);
		collector.toCSV_discard("data_discard.csv", false);
	}

	private boolean drawMode = false;
	
	public void setDrawMode(boolean b) {
		this.drawMode = b;
	}
	
	@Override
	public void to_CSV(String filename, boolean cont) throws IOException {
		// TODO Auto-generated method stub
		if (drawMode) toCSV_picking(filename, cont);
		else toCSV_discard(filename, cont);
	}


}
