package collector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Stack;

import com.opencsv.CSVWriter;

import core.Card;
import core.GinRummyPlayer;
import core.GinRummyUtil;
import player.EstimatingPlayer;
import player.HittingPlayer1;

/**
 * @author Tom
 */


public class EPlayerWeightDataCollector {
	
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
	private GinRummyPlayer[] players = {new HittingPlayer1(), new HittingPlayer1()};
	
	private EstimatingPlayer estimator = new EstimatingPlayer();
	
	private ArrayList<double[]> estimatingData = new ArrayList<>();
	private double currDesirability, currDeadwood, currValue;
	private boolean isRecordingData = false;
	
	public EPlayerWeightDataCollector() {
		
	}
	
	@SuppressWarnings("unchecked")
	public int playWithEstimator() {
		int[] scores = new int[2];
		ArrayList<ArrayList<Card>> hands = new ArrayList<ArrayList<Card>>();
		hands.add(new ArrayList<Card>());
		hands.add(new ArrayList<Card>());
		int startingPlayer = RANDOM.nextInt(2);
		
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
				
				//setting up estimating player
				if(i == 0)
					estimator.startGame(0, startingPlayer, handArr);
				isRecordingData = false;
				
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
					
					estimator.willDrawFaceUpCard(faceUpCard);
					
					if (playVerbose && !drawFaceUp && faceUpCard == firstFaceUpCard && turnsTaken < 2)
						System.out.printf("Player %d declines %s.\n", currentPlayer, firstFaceUpCard);
				}
				if (!(!drawFaceUp && turnsTaken < 2 && faceUpCard == firstFaceUpCard)) { // continue with turn if not initial declined option
					Card drawCard = drawFaceUp ? discards.pop() : deck.pop();
					for (int i = 0; i < 2; i++) 
						players[i].reportDraw(currentPlayer, (i == currentPlayer || drawFaceUp) ? drawCard : null);
					
					//TODO
					estimator.reportDraw(currentPlayer, (currentPlayer == 0|| drawFaceUp) ? drawCard : null);
					if(currentPlayer == 1 && isRecordingData) {
						currValue = drawFaceUp ? 0 : GinRummyUtil.getDeadwoodPoints(faceUpCard);
						double[] currData = {currDesirability, currDeadwood, currValue};
						estimatingData.add(currData);
						
						if(playVerbose) {
							for(int i = 0; i < currData.length; i++)
								System.out.printf("%.4f ", currData[i]);
							System.out.println(); 
						}
					}
					
					if (playVerbose)
						System.out.printf("Player %d draws %s.\n", currentPlayer, drawCard);
					hands.get(currentPlayer).add(drawCard);

					// DISCARD
					Card discardCard = players[currentPlayer].getDiscard();
					
					ArrayList<Card> candidates = new ArrayList<>();
					double[] desirabilities = null;
					if(currentPlayer == 0) {
						estimator.getDiscard();
						if(!estimator.candidateCards.isEmpty() && estimator.getCardDesirability() != null)
						candidates.addAll(estimator.candidateCards);
						desirabilities = new double[estimator.getCardDesirability().length];
						
						if(playVerbose)
							System.out.println("desirability array length is " + desirabilities.length);
						
						for(int i = 0; i < desirabilities.length; i++) 
							desirabilities[i] = 1.0 / estimator.getCardDesirability()[i];
//						desirabilities.addAll(Arrays.asList(estimator.getCardDesirability()));
					
						if(playVerbose)
							System.out.println(candidates);
					}
					
					if (!hands.get(currentPlayer).contains(discardCard) || discardCard == faceUpCard) {
						if (playVerbose)
							System.out.printf("Player %d discards %s illegally and forfeits.\n", currentPlayer, discardCard);
						return opponent;
					}
					hands.get(currentPlayer).remove(discardCard);
					for (int i = 0; i < 2; i++) 
						players[i].reportDiscard(currentPlayer, discardCard);
					
					//TODO
					//record data for estimating weights
					estimator.reportDiscard(currentPlayer, discardCard);
					
					if(!candidates.isEmpty() && playVerbose) {
						System.out.println((candidates.size() + " and " + desirabilities.length));
					}
					
					if(currentPlayer == 0 && (estimator.getTurn() > 3) && !candidates.isEmpty() && (candidates.size() == desirabilities.length)) {
						if(playVerbose) {
							System.out.println(candidates);
							for(int i = 0; i < desirabilities.length; i++)
								System.out.printf("%.4f ", desirabilities[i]);
						}
						isRecordingData = true;
						int index;
						for(index = 0; index < candidates.size(); index++)
							if(candidates.get(index).getId() == discardCard.getId())
								break;
						if(index < candidates.size()) {
							currDesirability = desirabilities[index];
							currDeadwood = GinRummyUtil.getDeadwoodPoints(discardCard);
							
//							if(playVerbose) {
//								System.out.println(desirabilities[index]);
//								System.out.println(GinRummyUtil.getDeadwoodPoints(discardCard));
//							}
						}
					}
						else
							isRecordingData = false;
					
					if (playVerbose)
						System.out.printf("Player %d discards %s.\n", currentPlayer, discardCard);
					discards.push(discardCard);
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
	
	
	public void to_CSV(String filename, boolean cont) {
		//Instantiating the CSVWriter class
		
		try {
			CSVWriter writer;
			writer = new CSVWriter(new FileWriter(filename, cont));
		
			//Writing data to a csv file
			        
			// Header
			if (!cont) {
				String[] headers = new String[3];
				headers[0] = "DiscardSafety";
				headers[1] = "DeadwoodPoint";
				headers[2] = "Value";
				writer.writeNext(headers);
			}
			
			for (int i = 0; i < estimatingData.size(); i++) {
				
				String line[] = Arrays.stream(estimatingData.get(i)).mapToObj(String::valueOf).toArray(String[]::new);
				
				assert line.length == 3 : "Wrong system of features";
				
				writer.writeNext(line);
			}
			
			writer.close();
		    System.out.println("Data entered!!!");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
	
	public static void main(String[] args) {
		EPlayerWeightDataCollector collector = new EPlayerWeightDataCollector();
		
		playVerbose = true;
		collector.playWithEstimator();
		
//		playVerbose = false;
//		System.out.println("Playing games...");
//		int numGames = 100;
//		
//		for(int i = 0; i < numGames; i++)
//			collector.playWithEstimator();
		
		collector.to_CSV(".est_sp_100_v1.csv", false);
		
	}

}