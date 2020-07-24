import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

public abstract class HandEstimatorGeneral{
	
	public abstract double[] getProb();
	
	public abstract boolean[] getKnown();
	
//	public abstract void setKnown(Card card, boolean held);
	
	public abstract void setKnown(ArrayList<Card> cards, boolean held);
	
	public abstract void reportDrawDiscard(Card faceUpCard, boolean drawn, Card discardedCard);
	
	
	public ArrayList<Card> getEstimatedHand() {
		
		ArrayList<Card> hand = new ArrayList<Card>();
		ArrayList<ArrayList<Double>> temp = new ArrayList<ArrayList<Double>>();
		double[] prob = getProb();
		boolean[] known = getKnown();
		
		for (int id = 0; id < 52; id++)
			if (known[id])
				hand.add(Card.strCardMap.get(Card.idStrMap.get(id)));
			else
				temp.add(new ArrayList<Double>(Arrays.asList(prob[id], (double) id)));
		
		int theRest = 10 - hand.size(); // the left number of unknown cards 
		Collections.sort(temp, new Comparator<ArrayList<Double>>()
				{
					@Override
					public int compare(ArrayList<Double> a, ArrayList<Double> b) {
						if (a.get(0) < b.get(0))
							return 1;
						else if (a.get(0) > b.get(0))
							return -1;
						return 0;
					}
				});
		
		ArrayList<Integer> uniqueID = new ArrayList<Integer>();
		for (int i = 0; i < temp.size(); i++)
		{
			if (i > 0 && temp.get(i).get(0) != temp.get(i-1).get(0))
			{
				uniqueID.add((int)Math.floor(temp.get(i-1).get(1)));
				theRest--;
				if (theRest == 0)
				{
					for (int id: uniqueID)
						hand.add(Card.strCardMap.get(Card.idStrMap.get(id)));
					break;
				}
			}
		}
//		estimatedHand = hand;
		return hand;
	}


	
}
