package util;
import java.util.ArrayList;
import java.util.HashSet;
import collector.*;
import core.*;
import module.*;
import player.*;

public class Util {
	
	public static ArrayList<Card> get_unmelded_cards (ArrayList<ArrayList<Card>> melds, ArrayList<Card> hand) {
		
		if (melds == null) {
			return hand;
		}
		
		HashSet<Card> melded = new HashSet<Card>();
		for (ArrayList<Card> meld : melds) {
			for (Card card : meld) {
				melded.add(card);
			}
		}
		
		ArrayList<Card> unmelded = new ArrayList<>();
		for (Card card : hand)
			if (!melded.contains(card)) {
				unmelded.add(card);
			}
		return unmelded;
	}
	

	public static void print_mat(float[] mat, String name) {
		System.out.println(name + ": ");
		System.out.print("Rank");
		for (int i = 0; i < Card.NUM_RANKS; i++)
			System.out.print("\t" + Card.rankNames[i]);
		for (int i = 0; i < Card.NUM_CARDS; i++) {
			if (i % Card.NUM_RANKS == 0)
				System.out.printf("\n%s", Card.suitNames[i / Card.NUM_RANKS]);
			System.out.print("\t");
			System.out.printf("%1.1e", mat[i]);
//			System.out.printf("%.4f", mat[i]);
		}
		System.out.println();
		System.out.println();
	}
	
	public static void print_mat(boolean[] mat, String name) {
		System.out.println(name + ": ");
		System.out.print("Rank");
		for (int i = 0; i < Card.NUM_RANKS; i++)
			System.out.print("\t" + Card.rankNames[i]);
		for (int i = 0; i < Card.NUM_CARDS; i++) {
			if (i % Card.NUM_RANKS == 0)
				System.out.printf("\n%s", Card.suitNames[i / Card.NUM_RANKS]);
			System.out.print("\t");
//			System.out.printf("%2.4f", mat[i]);
			System.out.printf(mat[i] ? "TRUE" : "_");
		}
		System.out.println();
		System.out.println();
	}
	
	public static void print_mat1D_card(int[] mat, String name) {
		System.out.println();
		System.out.println(name + ": ");
		int a = 0;
		for (int i = 0; i < mat.length; i++) {
			// Debugging
			System.out.printf("%s: %d ",Card.getCard(i).toString(), mat[i]);
			a++;
			if(a == 13) {
				a = 0;
				System.out.println();
			}
		}
		System.out.println();
	}
}
