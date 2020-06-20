
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Random;
import java.util.Stack;

import org.json.*;

import jdk.nashorn.internal.runtime.JSONListAdapter;

/**
 * a data generator meant for Alex's experiments
 * 
 * @author Tom Doan
 * Gettysburg College. Advisor: Todd W. Neller
 * @version 1.2.1
 */


public class TurnStatesDataCollector {
	
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
	
	/**
	 * The multi-dimentional array to store the game data in arrays of short[][], where
	 * playData[0] is the faceupCard's input
	 * playData[1] is the drawCard's input, if it appears that opponent pick up faceupCard
	 * playData[2] is the discardCard's input
	 * playData[1] is the model's expected result
	 */
	private static ArrayList<ArrayList<ArrayList<ArrayList<short[][]>>>> playData = new ArrayList<>();
	
	
	/**
	 * Play a game of Gin Rummy and return the winning player number 0 or 1. Add a new TurnState in line 132
	 * @return the winning player number 0 or 1
	 */
	public int play(int startingPlayer, GinRummyPlayer player0, GinRummyPlayer player1) {
		players = new GinRummyPlayer[] {player0, player1};
		int[] scores = new int[2];
		ArrayList<ArrayList<Card>> hands = new ArrayList<>();
		hands.add(new ArrayList<Card>());
		hands.add(new ArrayList<Card>());
//		int startingPlayer = RANDOM.nextInt(2);
		
		//game states with 2 vectors of turnStates to store one's own actions and hands		
		ArrayList<ArrayList<ArrayList<short[][]>>> gameData = new ArrayList<>();

		while (scores[0] < GinRummyUtil.GOAL_SCORE && scores[1] < GinRummyUtil.GOAL_SCORE) { // while game not over
			
			//create a variable to store turns' states			
			ArrayList<ArrayList<short[][]>> roundData = new ArrayList<>();
			roundData.add(new ArrayList<short[][]>());
			roundData.add(new ArrayList<short[][]>());
			
			//TODO remove magic numbers
			boolean[][] knownCards = new boolean[2][52];
			
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
				
				for(Card card: hands.get(i))
					knownCards[i][card.getId()] = true;
				
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
					
					//record a turn's data 
					knownCards[currentPlayer][faceUpCard.getId()] = true;
					roundData.get(currentPlayer).add(turnStateToArray(currentPlayer, faceUpCard, drawCard, discardCard, hands, knownCards));
					
					
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
			
			//record game data
			gameData.add(roundData);
		}
		if (playVerbose)
			System.out.printf("Player %s wins.\n", scores[0] > scores[1] ? 0 : 1);
		
		//record "big" game data
		playData.add(gameData);
		
		
		return scores[0] >= GinRummyUtil.GOAL_SCORE ? 0 : 1;
	}
	
	
	/**
	 * Given faceUp card, drawCard, and discarded card, return the corresponding short array
	 * @param Card objects representing faceUp card, drawCard, discarded card, and the player's hand
	 * @return the corresponding short[5][52] array
	 */	
	public short[][] turnStateToArray(int currentPlayer, Card faceUpCard, Card drawCard, Card discardCard, ArrayList<ArrayList<Card>> hands, boolean[][] knownCards) {
		int opponent = (currentPlayer == 0) ? 1 : 0;
		
		short[][] state = new short[4][52];
		state[0][faceUpCard.getId()] = 1;
		if(faceUpCard == drawCard)
			state[1][faceUpCard.getId()] = 1;
		state[2][discardCard.getId()] = 1;
		
		for(int id = 0; id < 52; id++) 
			if(knownCards[opponent][id] && !hands.get(currentPlayer).contains(Card.getCard(id)))
				state[3][id] = 1;
		
		for(Card card : hands.get(currentPlayer)) 
			state[4][card.getId()] = 1;
		
		return state;
	}
	
	
	public long[] turnStateToBitstring(int currentPlayer, Card faceUpCard, Card drawCard, Card discardCard, ArrayList<ArrayList<Card>> hands, boolean[][] knownCards) {
		int opponent = (currentPlayer == 0) ? 1 : 0;
		
		long[] stateBitstring = new long[4];
		
		ArrayList<Card> cards = new ArrayList<>();
		cards.add(faceUpCard);
		stateBitstring[0] = GinRummyUtil.cardsToBitstring(cards);
		
		cards.clear();
		cards.add(drawCard);
		stateBitstring[1] = GinRummyUtil.cardsToBitstring(cards);
		
		cards.clear();
		for(Card card : Card.allCards) 
			if(knownCards[opponent][card.getId()] && !hands.get(currentPlayer).contains(card))
				cards.add(card);
		stateBitstring[2] = GinRummyUtil.cardsToBitstring(cards);
		
		stateBitstring[3] = GinRummyUtil.cardsToBitstring(hands.get(opponent));
		
		return stateBitstring;
	}
	
	
//	public void saveGameBitstring (String filename, ArrayList<ArrayList<ArrayList<ArrayList<long[]>>>> gamePlaysInBitstring) {
//		
//		int size0 = gamePlaysInBitstring.size()
//		for(int i = 0; i < size0; i++) {
//			int size1 = gamePlaysInBitstring.get(i).size();
//			for(int j = 0; j < size1; j++) {
//				int size2 = gamePlaysInBitstring.get(i).get(j).size();
//				for(int k = 0; k < size2; k++) {
//					
//					
//					
//					int size3 = gamePlaysInBitstring.get(i).get(j).get(k).size();
//					for(int l = 0; l < size3; l++) {
//						
//					}
//				}
//			}
//		}
//		
//	}
	
	
	/**
	 * Save the gameplays as objects in dat file
	 */
	public void saveGame(String filename, ArrayList<ArrayList<ArrayList<ArrayList<short[][]>>>> gamePlays) {
		try {
			ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename));
			out.writeObject(gamePlays);
			out.flush();
			out.close();
		} catch (FileNotFoundException e) {
			System.out.println("File not found: " + filename);
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	
	/**
	 * Play a number of Gin Rummy games and record the data to file
	 */
	public static void main(String[] args) {
		
		TurnStatesDataCollector collector = new TurnStatesDataCollector();
		int numGameBig = 25000;
		for(int i = 0; i < numGameBig; i++) 
			collector.play(i%2, new SimpleGinRummyPlayer(), new SimpleGinRummyPlayer());
		
		long startMs = System.currentTimeMillis();
		collector.saveGame("play_data_SimplePlayer.dat", playData);
		System.out.println(System.currentTimeMillis() - startMs);
	}
}