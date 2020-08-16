package collector;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.Stack;

import com.opencsv.CSVWriter;
import core.*;
import module.*;
import player.*;


public class HittingGrinder3 extends DataGrinder {

    private static final int HITTING_REWARD_CONST = 2;
    private static final int CARD_RANK_FAVOR = 5;
//	private static final int MELD_REWARD_CONST = 10;

//	int turnsTaken = 0;

    /**
     * Verbose
     */
    boolean VERBOSE = true;


    /**
     * Stats
     */
    int[] total_hit_stat;

    /**
     * Module support
     */
    HittingModule hitEngine = new HittingModule();

    /**
     * each int[] is a line data. [0,1,2,3]: turn, card rank, n_hit, actual label
     */
    ArrayList<int[]> lines_data;


    /**
     * One line of data (turn)
     */
    int[] current_line;

    /**
     * current collection of line data in a match to be append to lines_data.
     */
    HashMap<int[], Card> current_match;

    /**
     * Current know cards. We why base on this to calculate the meld
     */
    HashSet<Card> current_hand;

    public HittingGrinder3 () {
        hitEngine.init();
        total_hit_stat = new int[2]; // [0]: total_drawn_hitting; [1]: total melds;
        lines_data = new ArrayList<>();
        current_match = new HashMap<>();
        current_hand = new HashSet<>();
    }


    /**
     * Onlly collect the player 0
     * update line[01234]. Turn, rank, is meld, hit melds, label
     * @param turnsTaken
     * @param hand
     * @param faceUp
     */
    @SuppressWarnings("unchecked")
    public void collectData(int turnsTaken, ArrayList<Card> hand, Card faceUp, Card discard, Card drawn) {
//		System.out.println("Turn: " + turnsTaken);
        if (faceUp != null) {
            // Reset new line
            current_line = new int[4];
            current_line[0] = turnsTaken;
            current_line[1] = faceUp.rank >= 9 ? 10 : faceUp.rank + 1;

            // Add every card to the current hand.
            for (Card c : hand) current_hand.add(c);
            current_hand.add(faceUp);
            if (drawn != null) current_hand.add(drawn);

//            boolean ismeld = hitEngine.isMeld(hand, faceUp);
//
//            // If is meld then we surely pick it
//            if (ismeld) {
//                current_line[2] = 1;
//            } else {
//                current_line[2] = 0;
//            }

            // Count hit melds
            int hitMelds = hitEngine.countHitMeld(hand, faceUp);
            current_line[2] = hitMelds;

            // Initialize value to 0
            current_line[3] = 0;

            // put to current match
            current_match.put(current_line, faceUp);

//			System.out.println("Currnet_mathc's size: " + current_match.size());
        }
    }

    /**
     * Only collect player 0
     * update line[3], edit line[4]. check paper
     *
     * @param turnsTaken
     * @param hand
     * @param melds
     */
    public void collectLabel(int turnsTaken, ArrayList<Card> hand, ArrayList<ArrayList<Card>> melds,
                             boolean won, int deadwoodsurpass) {

//		System.out.println("Turn: " + turnsTaken);

        for (Entry<int[], Card> e : current_match.entrySet()) {
            int[] line = e.getKey();
            Card card = e.getValue();

            // check if card is in a meld
//            boolean ismeld = false;
//            for (ArrayList<Card> meld : melds) {
//                if (meld.contains(card)) {
//                    ismeld = true;
//                    break;
//                }
//            }

            // Calculate turn value
            int turnValue = turnsTaken - line[0];

            if (turnValue < 0) {
                continue;
            }

            // Calculate Card Value
            int cardValue = 10 - (line[1]);

            // Calculate nmeld value
            int nMeldValue = 0;
            // convert a hashset to arraylist of card.
            ArrayList<Card> newHand = new ArrayList<Card>(current_hand);
            // See if the card can hit how many melds.
            for (ArrayList<Card> meld : GinRummyUtil.cardsToAllMelds(newHand)) {
                if (meld.contains(card)) {
                    nMeldValue ++;
                }
            }
            line[line.length - 1] = turnValue + cardValue + nMeldValue;

            if (VERBOSE) {
                System.out.println("This Match Data:");
                System.out.printf("Turn\tRank\tHitting\tTurnVal\tCardVal\tnMeldForm\tValue\n");
                for (int i = 0; i < line.length - 1; i++) {
                    System.out.printf("%s\t", line[i]);
                }
                System.out.printf("%s\t%s\t%s\t%s\n", turnValue, cardValue, nMeldValue, line[line.length - 1]);
            }

            // Concatenate and write all data
            int[] datum = new int[7];
            int i;
            for (i = 0; i < line.length - 1; i++) {
                datum[i] = line[i];
            }
            datum[i++] = turnValue;
            datum[i++] = cardValue;
            datum[i++] = nMeldValue;
            datum[i++] = line[line.length - 1];

            assert i == 7 : "Wrong expected i value!";

            lines_data.add(datum);

        }

        // reset current match
        current_match = new HashMap<>();
        // Reset currnet hand
        current_hand = new HashSet<>();

    }

    /**
     * Only collect player 0
     * @param hand
     * @param faceUp
     * @param melds
     */
    public void trackHit (ArrayList<Card> hand, Card faceUp, ArrayList<ArrayList<Card>> melds) {
        // check hitting card in hand

        if (faceUp != null) {
            boolean ishit = false;
            for (Card c : hand) {
                if (hitEngine.isHittingCard(c, faceUp)) ishit = true;
            }

            this.total_hit_stat[0] += ishit ? 1 : 0;
        }

        // If there is meld, plus into meld.
        if (melds != null) {
            total_hit_stat[1] += melds.size();
            hitEngine.init();
        }
    }

    public void displayTrackHit() {
        System.out.printf("Total number of hitting cards drawn: %d\nTotal of melds formed: %d\n", total_hit_stat[0], total_hit_stat[1]);
    }

    /**
     * Dis play the whole bunch of big data
     */
    public void display_lines_data() {
        System.out.printf("This is lines_data:\nTurn\tRank\tHitting\tTurnVal\tCardVal\tnMeldForm\tValue\n");
        for (int[] line : lines_data) {
            for (int v : line) {
                System.out.printf(v + "\t");
            }
            System.out.println();
        }
    }

    /**
     * Display current match data
     */
    public void display_current_match_data() {
        for (Entry<int[], Card> e : current_match.entrySet()) {
            System.out.printf("This is lines_data:\nTurn\tRank\tHitting\tEnd in\tValue\twith card %s\n", e.getValue());
            for (int v : e.getKey()) {
                System.out.printf(v + "\t");
            }
            System.out.println();
        }
    }

    public void to_CSV(String filename, boolean cont) {
        //Instantiating the CSVWriter class

        try {
            CSVWriter writer;
            writer = new CSVWriter(new FileWriter(filename, cont));

            //Writing data to a csv file

            // Header
            if(!cont) {
                String[] headers = new String[7];
                headers[0] = "Turn";
                headers[1] = "Rank";
                headers[2] = "nMeldHit";
                headers[3] = "TurnVal";
                headers[4] = "CardVal";
                headers[5] = "nMeldForm";
                headers[6] = "TotalVal";
                writer.writeNext(headers);
            }

            for (int i = 0; i < lines_data.size(); i++) {

                String[] line = new String[7];
                int[] datum = lines_data.get(i);
                line[0] = Integer.toString(datum[0]);
                line[1] = Integer.toString(datum[1]);
                line[2] = Integer.toString(datum[2]);
                line[3] = Integer.toString(datum[3]);
                line[4] = Integer.toString(datum[4]);
                line[5] = Integer.toString(datum[5]);
                line[6] = Integer.toString(datum[6]);
                writer.writeNext(line);
            }

            writer.close();
            System.out.println("Data entered!!!");
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public void displayData() {

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
                if (!(turnsTaken == 2 && faceUpCard == firstFaceUpCard)) { // both players declined and 1st player must draw face down
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


                    // Collect data after each turn
                    if (currentPlayer == 0) {
                        trackHit(hands.get(currentPlayer), drawFaceUp ? faceUpCard : null, null);
                    }

                    if (currentPlayer == 0) {
                        collectData(turnsTaken, hands.get(currentPlayer), faceUpCard, discardCard, drawCard);
                    }



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
                boolean won = false;
                // compare deadwood and compute new scores
                if (knockingDeadwood == 0) { // gin round win
                    scores[currentPlayer] += GinRummyUtil.GIN_BONUS + opponentDeadwood;

//					collectData(turnsTaken, knockingDeadwood, knockMelds.size() - 1, true);
//					collectData(turnsTaken, opponentDeadwood, opponentMelds.size() - 1, false);
                    won = true;

                    if (playVerbose)
                        System.out.printf("Player %d scores the gin bonus of %d plus opponent deadwood %d for %d total points.\n", currentPlayer, GinRummyUtil.GIN_BONUS, opponentDeadwood, GinRummyUtil.GIN_BONUS + opponentDeadwood);
                }
                else if (knockingDeadwood < opponentDeadwood) { // non-gin round win
                    scores[currentPlayer] += opponentDeadwood - knockingDeadwood;

//					collectData(turnsTaken, knockingDeadwood, knockMelds.size() - 1, true);
//					collectData(turnsTaken, opponentDeadwood, opponentMelds.size() - 1, false);
                    won = true;

                    if (playVerbose)
                        System.out.printf("Player %d scores the deadwood difference of %d.\n", currentPlayer, opponentDeadwood - knockingDeadwood);
                }
                else { // undercut win for opponent
                    scores[opponent] += GinRummyUtil.UNDERCUT_BONUS + knockingDeadwood - opponentDeadwood;

//					collectData(turnsTaken, knockingDeadwood, knockMelds.size() - 1, false);
//					collectData(turnsTaken, opponentDeadwood, opponentMelds.size() - 1, true);


                    if (playVerbose)
                        System.out.printf("Player %d undercuts and scores the undercut bonus of %d plus deadwood difference of %d for %d total points.\n", opponent, GinRummyUtil.UNDERCUT_BONUS, knockingDeadwood - opponentDeadwood, GinRummyUtil.UNDERCUT_BONUS + knockingDeadwood - opponentDeadwood);
                }

                startingPlayer = (startingPlayer == 0) ? 1 : 0; // starting player alternates





                // Collect data after each turn
                if (currentPlayer == 0) {
                    trackHit(hands.get(currentPlayer), null, knockMelds);
                } else {
                    trackHit(hands.get(opponent), null, opponentMelds);
                }

                // Collect data after each turn
                if (currentPlayer == 0) {
                    collectLabel(turnsTaken, hands.get(currentPlayer), knockMelds, won, knockingDeadwood - opponentDeadwood);
                } else {
                    collectLabel(turnsTaken, hands.get(opponent), opponentMelds, !won, opponentDeadwood - knockingDeadwood);
                }


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

    public void setVERBOSE (boolean b) {
        this.VERBOSE = b;
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

        HittingGrinder3 collector = new HittingGrinder3();

        collector.setVERBOSE(false);
        int numGames = 1000;
        GinRummyPlayer p0 = new HittingPlayer(21);
        GinRummyPlayer p1 = new HittingPlayer(21);
        collector.match(p0, p1, numGames);

//		collector.displayTrackHit();

//		collector.displayData_picking();

//		collector.display_lines_data();

        // v8 value = turn + card + meld
        // v9 value = turn * card * meld
        collector.to_CSV("./dataset/hit_sp_20000_v8.csv", false);

        collector = new HittingGrinder3();
        collector.setVERBOSE(false);
        GinRummyPlayer p2 = new HittingPlayer(1);
        GinRummyPlayer p3 = new HittingPlayer(1);
        collector.match(p0, p1, numGames);
        collector.to_CSV("./dataset/hit_sp_20000_v8.csv", true);

        collector = new HittingGrinder3();
        collector.setVERBOSE(false);
        GinRummyPlayer p4 = new SimplePlayer();
        GinRummyPlayer p5 = new SimplePlayer();
        collector.match(p0, p1, numGames);
        collector.to_CSV("./dataset/hit_sp_20000_v8.csv", true);
    }
}
