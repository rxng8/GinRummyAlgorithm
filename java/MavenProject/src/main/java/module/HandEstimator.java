package module;
/**
 * @author Alex Nguyen
 * Gettysburg College
 * 
 * Advisor: Professor Neller.
 */

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;
import java.util.Stack;

import ai.djl.modality.cv.transform.Normalize;

import java.io.FileWriter;
import java.io.IOException;
import collector.*;
import core.*;
import player.*;
import util.*;

public class HandEstimator {

	public static float prob_card (int turn, 
			ArrayList<Card> hand, 
			ArrayList<Card> discardPile, 
			Card to_prob, 
			Card op_drawnCard, 
			Card op_discardCard, 
			int n_unknown_card, 
			float prob_bayes_draw, 
			float prob_bayes_discard) {
		
		if (op_drawnCard != null && to_prob.getId() == op_drawnCard.getId()) {
			return 0.5f;
		} 
		
		// check either hand or discard pile contain the estimating card
		boolean zero = false;
		for (Card c : discardPile) {
			if (c.getId() == to_prob.getId()) {
				zero = true;
				break;
			}
		}
		for (Card c : hand) {
			if (c.getId() == to_prob.getId()) {
				zero = true;
				break;
			}
		}
		if (op_discardCard != null && to_prob.getId() == op_discardCard.getId() || zero) {
			return 0f;
		}
		
		if (turn == 0) {
			return 1f / n_unknown_card;
		}
		
		return prob_bayes_draw * prob_bayes_discard;
	}
	
	public static float[] prob_card (float[] card_vector, 
			int turn, 
			ArrayList<Card> hand, 
			ArrayList<Card> discardPile,
			Card op_drawnCard, 
			Card op_discardCard, 
			int n_unknown_card, 
			float[] prob_bayes_draw, 
			float[] prob_bayes_discard) {
		
		float[] y_card = new float[52];
		for (int i = 0; i < card_vector.length; i++) {
			y_card[i] = prob_card(turn, hand, discardPile, Card.getCard(i), op_drawnCard, op_discardCard, n_unknown_card, prob_bayes_draw[i], prob_bayes_discard[i]);
		}
		return normalize(y_card);
	}
	
	
//	private static float[] normalize(float[] y_card) {
//		for (int i = 0; i < y_card.length; i++) {
//			y_card[i] *= 10;
//		}
//		return y_card;
//	}

	public static float prob_bayes_discard(float meld_prop, float to_prop) {
		return meld_prop * to_prop * 11;
	}
	
	public static float[] prob_bayes_discard(float[] meld_prop, float[] to_prop) {
		assert meld_prop.length == to_prop.length;
		float[] y = new float[meld_prop.length];
		for (int i = 0; i < meld_prop.length; i++) {
			y[i] = prob_bayes_discard(meld_prop[i], to_prop[i]);
		}
		return y;
	}

	public static float prob_bayes_draw(float meld_prop, float to_prop) {
		return meld_prop * to_prop * 2;
	}
	
	public static float[] prob_bayes_draw(float[] meld_prop, float[] to_prop) {
		assert meld_prop.length == to_prop.length;
		float[] y = new float[meld_prop.length];
		for (int i = 0; i < meld_prop.length; i++) {
			y[i] = prob_bayes_draw(meld_prop[i], to_prop[i]);
		}
		return y;
	}
	
	public static float[] normalize(float[] mat) {
		
		double normalizing_sum = 0;
		float[] newVector = new float[mat.length];
		
		for (int i = 0; i < newVector.length; i++) {
			double prob = mat[i];
			if (prob < 0) {
				prob = 0;
			}
			normalizing_sum += prob;
		}
		
		
		for (int i = 0; i < newVector.length; i++) {
			if (normalizing_sum != 0) {
				newVector[i] = (float) (Math.max(0.0, mat[i]) / normalizing_sum);
			} else {
				newVector[i] = (float) (1.0 / 42);
			}
		}
		
		return newVector;
	}
	
	
	//Debug method
	public void print_vector(ArrayList<Double> list, String name) {
		System.out.println();
		System.out.println(name + ": ");
		int a = 0;
		for (int i = 0; i <  list.size(); i++) {
			// Debugging
			System.out.printf("%s: %.5f ",Card.getCard(i).toString(), list.get(i));
//			System.out.printf("%.5f ", probs_op_card_this_turn.get(i));
			a++;
			if(a == 13) {
				a = 0;
				System.out.println();
			}
		}
		System.out.println();
	}
	
	public static void main(String[] args) {

		ArrayList<Card> li = new ArrayList<>();
		li.add(new Card(0, 3));
		Card meow = new Card(0,3);
		System.out.print(li.contains(meow));
	}
	
}
