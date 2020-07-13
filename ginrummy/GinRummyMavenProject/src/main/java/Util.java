
public class Util {
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
