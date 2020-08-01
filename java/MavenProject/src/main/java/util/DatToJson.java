package util;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.Iterator;

import org.json.*;

import com.google.gson.JsonObject;
import collector.*;
import core.*;
import module.*;
import player.*;


public class DatToJson {

	/**
	 * 
	 */
	ArrayList<ArrayList<ArrayList<ArrayList<short[][]>>>> gamePlays;
	
	/**
	 * one float[][] is one input datum for one turn.
	 * ArrayList<float[][]> represents sequence of turns in a match
	 * ArrayList<ArrayList<float[][]>> represents sequences of matches.
	 * 
	 * Future work: Take the data "Game Score" to process, too.
	 */
	public ArrayList<ArrayList<float[][]>> X;
	
	/**
	 * one float[][] is one input datum for one turn.
	 * ArrayList<float[][]> represents sequence of turns.
	 * ArrayList<ArrayList<float[][]>> represents sequences of matches.
	 * 
	 * Future work: Take the data "Game Score" to process, too.
	 */
	public ArrayList<ArrayList<float[]>> Y;
	
	public DatToJson() {
		this.X = new ArrayList<>();
		this.Y = new ArrayList<>();
	}
	
	/**
	 * 
	 * @param filename
	 */
	@SuppressWarnings("unchecked")
	public void __import_data__(String filename) {
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename));
			this.gamePlays = (ArrayList<ArrayList<ArrayList<ArrayList<short[][]>>>>) in.readObject();
			in.close();

		} catch (FileNotFoundException e) {
			System.out.println("File not found: " + filename);
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 *  Take the variable gameplays and preprocess to float[][] forms to put in X and Y
	 * 	ArrayList<ArrayList<ArrayList<ArrayList<short[][]>>>> gamePlays
	 */
	@SuppressWarnings({ "unused" })
	private void preprocess_data() {
		
		assert this.gamePlays != null : "You have to import the data first using __import_data__ method";
		
		int data_feature = this.gamePlays.get(0).get(0).get(0).get(0)[0].length;
		assert data_feature == this.gamePlays.get(0).get(0).get(0).get(0)[4].length : "Input and output does not match dimension";
		
		Iterator<ArrayList<ArrayList<ArrayList<short[][]>>>> it_games = this.gamePlays.iterator();
		while(it_games.hasNext()) {
			Iterator<ArrayList<ArrayList<short[][]>>> it_players = it_games.next().iterator();
			while (it_players.hasNext()) {
				Iterator<ArrayList<short[][]>> it_rounds = it_players.next().iterator();
				while (it_rounds.hasNext()) {
					Iterator<short[][]> it_turns = it_rounds.next().iterator();
					ArrayList<float[][]> inputs = new ArrayList<>();
					ArrayList<float[]> outputs = new ArrayList<>();
					while (it_turns.hasNext()) {
						short[][] turnData = it_turns.next();
						assert turnData.length == 4 + 1 : "Wrong data form! There are 4 input vector, and 1 output vector.";
						assert turnData[0].length == 52 : "Wrong data form! One card vector must have 52 features";
						float[][] input = new float[4][52];
						float[] output = new float[52];
						for (int j = 0; j < 52; j++) {
							input[0][j] = (float) turnData[0][j];
							input[1][j] = (float) turnData[1][j];
							input[2][j] = (float) turnData[2][j];
							input[3][j] = (float) turnData[3][j];
							output[j] = (float) turnData[4][j];
						}
						inputs.add(input);
						outputs.add(output);
					}
					this.X.add(inputs);
					this.Y.add(outputs);
				}
			}
		}
	}
	
	/**
	 * 
	 * @return
	 */
	public JSONObject toJSON () {
		JSONObject file = new JSONObject();
		JSONArray X = new JSONArray();
		JSONArray Y = new JSONArray();
		for (int match = 0; match < this.X.size(); match++) {
			JSONArray matchX = new JSONArray();
			JSONArray matchY = new JSONArray();
//			System.out.println("Get herer");
			for (int turn = 0; turn < this.X.get(match).size(); turn++) {
				JSONArray turnX = new JSONArray(); // 4 item
//				System.out.println("Get herer");
				for (int i = 0; i < 4; i++) {
					JSONArray turnX_feature = new JSONArray();
					for (int j = 0; j < 52; j++) {
						turnX_feature.put(this.X.get(match).get(turn)[i][j]);
					}
					turnX.put(turnX_feature);
				}
				JSONArray turnY = new JSONArray(); // 1 item
				for (int j = 0; j < 52; j++) {
					turnY.put(this.Y.get(match).get(turn)[j]);
				}
				matchX.put(turnX);
				matchY.put(turnY);
			}
			X.put(matchX);
			Y.put(matchY);
		}
		file.put("X", X);
		file.put("Y", Y);
		
		return file;
	}
	
	public static void writeFile (JSONObject json) {
		
		try {
	         FileWriter file = new FileWriter("output_500.json");
	         file.write(json.toString());
	         file.close();
	      } catch (IOException e) {
	         // TODO Auto-generated catch block
	         e.printStackTrace();
	      }
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		DatToJson fileWriter = new DatToJson();
		fileWriter.__import_data__("play_data_SimplePlayer.dat");
		fileWriter.preprocess_data();
		
//		System.out.println(fileWriter.X.get(2).get(2)[2][3]);
		
		JSONObject json = fileWriter.toJSON();
		DatToJson.writeFile(json);
	}

}
