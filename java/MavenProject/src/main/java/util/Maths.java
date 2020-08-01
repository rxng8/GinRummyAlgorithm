package util;
import collector.*;
import core.*;
import module.*;
import player.*;
public class Maths {
	
	public static float mean (float[] x) {
		float sum = 0;
		for (float f : x) {
			sum += f;
		}
		return sum / x.length;
	}
	
	public static float std (float[] x) {
		float sum = 0;
		float mean = mean(x);
		for (float f : x) {
			sum += Math.pow((f - mean), 2);
		}
		return (float) Math.sqrt(sum / x.length);
	}
	
	public static void main(String[] args) {
		float[] a = {2, 4, 6};
		System.out.println("Mean: " + mean(a));
		System.out.println("STD: " + std(a));
	}

}
