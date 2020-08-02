package module;

import java.util.ArrayList;
import core.*;
import collector.*;
import player.*;
import util.*;

public class PlayerGameState {
	
	Card faceUpCard, discard;
	Card drawnCard;
	// current player hand
	ArrayList<Card> cards;
	
	// turn
	int turn;
	
	int player;
	
	@SuppressWarnings("unchecked")
	public PlayerGameState(int player, int turn, ArrayList<Card> hand, Card faceUpCard, Card drawnCard, Card discardedCard) {
		this.player = player;
		this.turn = turn;
		this.faceUpCard = faceUpCard;
		this.discard = discardedCard;
		this.drawnCard = drawnCard;
		this.cards = (ArrayList<Card>) hand.clone();
	}
	
	@SuppressWarnings("unchecked")
	public String toString() {
		
		StringBuilder builder = new StringBuilder();
		builder.append(String.format("Turn %d, player %d move:\n", turn, player));
		
		if (drawnCard == faceUpCard) {
			builder.append(String.format("Player %d picked the face up card %s and discard %s\n", player, faceUpCard, discard));
		} else {
			builder.append(String.format("Player %d picked from the draw pile %s and discard %s\n", player, drawnCard,  discard));
		}
		
//		ArrayList<ArrayList<ArrayList<Card>>> set = GinRummyUtil.cardsToBestMeldSets(cards);
//		ArrayList<ArrayList<Card>> cardString;
//		if (!set.isEmpty()) {
//			cardString = set.get(0);
//		} else {
//			cardString = new ArrayList<>();
//			cardString.add(cards);
//		}
		
		ArrayList<Card> unmeldedCards = (ArrayList<Card>) cards.clone();
		ArrayList<ArrayList<ArrayList<Card>>> bestMelds = GinRummyUtil.cardsToBestMeldSets(unmeldedCards);
		if (bestMelds.isEmpty()) 
			builder.append(String.format("Player %d has %s with %d deadwood.\n", player, unmeldedCards, GinRummyUtil.getDeadwoodPoints(unmeldedCards)));
		else {
			ArrayList<ArrayList<Card>> melds = bestMelds.get(0);
			for (ArrayList<Card> meld : melds)
				for (Card card : meld)
					unmeldedCards.remove(card);
			melds.add(unmeldedCards);
			builder.append(String.format("Player %d has %s with %d deadwood.\n", player, melds, GinRummyUtil.getDeadwoodPoints(unmeldedCards)));
		}
		
//		builder.append(String.format("Resulting in the hand with meld:\n%s\n", cardString));
		
		return builder.toString();
	}
	
}
