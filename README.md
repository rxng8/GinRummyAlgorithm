# Algorithm of Reinforment Leanring for Imperfect Information Card Game: Gin Rummy

## Current Work and Code:

* Please navigate to `/ginrummy/GinRummyEAAIorigin/GinRummyEAAI/HESimpleModel.java` to see what I'm working on with hand estimation simple model.
    * This is the custom simple fully connected (feed-forward, back-propagation) artificial neural network model (2 layers) that I have written from scratch (no library use).
    
    * This model takes an input of observation of the opponent move, and output an estimated opponent hand. (For more information, please navigate to the documentation I have written in the file or navigate to report 4).

    * This file is being debugged and finished.

* Please navigate to `/ginrummy/GinRummyEAAIorigin/GinRummyEAAI/HElstm.java` to see what I'm working on with hand estimation lstm model.

    * This is the custom lstm model that I am about to research in.

    * (For more information, please navigate to the documentation I have written in the file or navigate to report 4)
s
-------------
## Past Work and Code:

* Please navigate to `/ginrummy/GinRummyEAAIorigin/GinRummyEAAI/CFRPlayer.java` to see what I'm working on.
    * This player use simple probability estimation algorithm stated from `week 2 report` including opponent hand estimation by Bayes Theorem. And Flipping and Masking to map that estimation to strategy.
    
    * Currently, this player has win rate of 30% matches over simple player in ONE game. And over the course of 1000 games, this player won 0 game over simple player.

* Please navigate to `/ginrummy/GinRummyEAAIorigin/GinRummyEAAI/PlayerModel.java` to see what I'm working on.

    * This is the model I write based on what I read about Temporal Difference technique in Reinforcement leanring in Gin Rummy according to [TD-Rummy paper](https://www.aaai.org/Papers/ICML/2003/ICML03-050.pdf).

    * This file includes a feed forward ANN with input is state vector in `Report 3, part 1, bullet point 1,` and the output is one simple value determine the value of that state.

    * Through the course of the game, the player will learn this value function and base on that to determine each action.

    * Read more about this on `Report 3, part 4.`

------------
# Report:

## Report 4, date: June 17, 2020
### 1. Observation Representation:

* TODO:

    * TODO:
    
    * TODO:

### 2. Hand Estimation using fully connected deep neural network.

* TODO:

    * TODO:
    
    * TODO:

### 3. Hand Estimation using LSTM:

* TODO:

    * TODO:
    
    * TODO:

### 4. Generalize a hand features/ traits using Auto Encoder

* TODO:

    * TODO:
    
    * TODO:

------------
## Report 3, date: June 9, 2020
### 1. Modeling the abstraction of the public state. State Representation.

* The state will be a vector of length 52. Each element is a card. Each element have 4 state, `in-player-hand`, `in-opponent-hand`, `in-discard-pile`, and `unknown`. Each card will be represented by 1 in these 4 states, with `integer` representation. So there are roughly 2x10^31 different states. NOTE: the card is only considered to be in opponent hand if it is picked by opponent from the discard pile. Other cards are unknown.

* One other way we can represent states is probability representation. The state can be presented as a vector of length 52 with each element is a card. Each element have 4 state, `in-player-hand`, `in-opponent-hand`, `in-discard-pile`, `in-draw-pile`, and `unknown`, and we will represent each state with some kind of a distance between states, like in K-Mean clustering. (Have not researched)


### 2. Hand Estimation using Bayes Theorem or Supervised Learning?:
* There are 2 approaches: 
    * Using Bayes Theorem to estimate opponent hand. P(A|B) = P(B|A) * P(A) / P(B). We can say that the probability of opponent having this card `X` in hand given that the opponent have picked an actual card `C` is the probability of the opponent should picked the card `C` if they has the card `X` in hand. Multiply with the probability of the card `X` exists in opponent hand, over 1 (The probability of opponent just pick the card.) => This estimation fails for phishing strategy.

    * Use Collected pick-up-and-discard data from playing games to do a supervised learning to estimate the probability of each card existing in opponent hand. => This estimation also fails for strategies of players other than SimplePlayer (Because we collect data from Simple Players' games).

### 3. Improving Policy evaluation function. \(Strategy about which to draw and which to discard\).
* Improve `updateStrategy()` method, not just by flipping.

* Currently, my player have win rate of 30% in one game vs the SimplePlayer.

* Navigate to #4.

### 4. Estimate the state value function? (According to [TD-Rummy](https://www.aaai.org/Papers/ICML/2003/ICML03-050.pdf) paper)
* The task is to learn the value function based only on the
results of self-play. The value function has only the input of
the game state and generates a single numeric evaluation
of it. There is no knowledge of the rules of the game built
into the value function. There is no notion of sequences,
sets or deadwood. \(TD-Rummy paper\).

* Currently, I am using the `updateStrategy()` using simple flipping method to map `opponent hand estimation` to `discarding strategy`. However I am expected to modify this method with a feed-forward ANN with custom back-propagation (According to TD-Rummy Paper):
    * The input of the ANN model is a vector of length 52. Each element in vector represent a value telling that it is in `1 in 4 states: in player hand, in opponent hand, in discard pile, and unknown card`

    * The vector is then passed through some combination of nodes with `sigmoid` activation function.

    * The last layer will be the expected value of that hand.

    * Now we perform action based on the value we got after considering every available actions.

    * We now have new state and new input, so we need to update weights by performing back-propagation. This will then evaluate the reward and value of the current state by the old weights and compute the new weights.

    * Therefore, we update weights every turns until the end of the game.

    * NOTE: Proving that using an ANN is possible to estimate the value of hand: (Unproved.)

----------
## Report 2, date: June 2, 2020
So far I have written a light version of the new bot, the current base of the bot depends on computing probability of strategy and the probability of which card is in the opponent's hand.

### 1. Player's variables:

* `unknown_cards (Integer)`: The number of cards this player doesn't know. Use to calculate probability.
* `op_cards (ArrayList<Double>)`: The probability of which card opponent has in hand. Array length of 52 corresponding to 52 cards in the deck. Initialized to `1 / unknown_cards` for each card when the game starts.
* `dis_strategy (ArrayList<Double>)`: The probability of which card this player should discard. Array length of 52 corresponding to 52 cards in the deck. Initialized to `1 / Card.NUM_CARDS` for each card when the game starts.

### 2. The process: At every decision point of both players, the variable vectors are updated as below:

* In the `willDrawFaceUpCard(Card card)`:
    * If the face up card is in the meld formed with every card in the deck and the meld size is less than 4, then draw. Draw face down otherwise. (Meld Collecting strategy). Notice that this code strategy differs from the original code that the original code only pick up card when this card + our hand form melds, on the other hand, this code will also pick the card that can potentially form meld.
    * Update op_cards at that spot by 0.0.
    * BUG: If the meld formed is larger than 4 and potential, the program may choose not to draw.
* In the `reportDraw(int playerNum, Card drawnCard)`: It is divided into 4 cases:
    * Case 1: Opponent draw face down: This means that the opponent does not have or is not collecting any meld relating to the face up card. We then lower the probability of the face up card and the card in meld with the face up card in the opponent hand vector `op_hands`.
    * Case 2: Opponent draw face up: This means that the opponent does have and is collecting some melds relating to the face up card. We then increase the probability of the face up card and the card in meld with the face up card in the opponent hand vector `op_hands`.
    * Case 3: This player draw face up: Have not implemented. TODO: Calculate the opponent regrets of having discarded that card. This will further help to implement the bait drawing strategy \(phishing).
    * Case 4: This player draw face down: Have not implemented. TODO: Calculate the opponent regrets of not having discarded some cards. This will further help to implement the bait drawing strategy \(phishing).
* Computing strategy: the strategy is computed from the opponent card vector `op_cards` in the `updateDiscardStrategy()` method:
    * First get the opponent card vector, get all non-zero value to form a possible opponent hand \(can be greater than 10 cards).
    * Base on the opponent hand card list, compute the meld probability of each of the zero value in `op_cards` to form a vector of opponent's expectation of card.
    * We then flip the opponent expectation vector to get our strategy vector \(what the opponent expect us to discard, we will not discard) and then normalize it.
* In the `getDiscard()`, the decision will be based on the strategy vector `dis_strategy` to compute the discard decision:
    * First, it clones the strategy vector of this player. Now the vector will contain the probability of discarding each card in each spot.
    * Perform masking processes so that we can just discard legally and discard the card not in melds.
    * After masking, choose the cards with highest probability as candidate cards.
    * Choose the highest rank card in the candidate cards list to minimize deadwood point.
* After that, we report discard in the `reportDiscard(int playerNum, Card discarded card)`: Have not implemented:
    * TODO: Divide into 2 cases: This player's discard, and opponent's discard

### 3. Question:
* I concern a lot about how I compute the probability of what card opponent have in hand. It is the probability of forming meld containing the picked card with the specific card that I want to assess, over the total number of melds formed by all deck.
* Can I approach this by another method:
    * I will write a payoff method that return the utility of hand after an action. (Reward method).
    * I will write an observation method that return some kind of public state generalization. (Observation method).
    * I will then let the computer player and continually save data as following:
        * We have the game data of one action step as an array [observation, action]
        * If the reward satisfy some conditions, then save the data.
        * Get all the observation as input, all the one-hot-encoded action as output.
        * Train by neural network or something.

-------------
### Consider the paper:
1. Neller. "An introduction to counterfactual regret minimalization." http://modelai.gettysburg.edu/2013/cfr/cfr.pdf
2. Deep Stack Poker. https://static1.squarespace.com/static/58a75073e6f2e1c1d5b36630/t/58b7a3dce3df28761dd25e54/1488430045412/DeepStack.pdf
3. Deep Stack Poker Supplementary. https://static1.squarespace.com/static/58a75073e6f2e1c1d5b36630/t/58bed28de3df287015e43277/1488900766618/DeepStackSupplement.pdf
4. Abstraction for Solving Large Incomplete-Information Games. https://www.cs.cmu.edu/~sandholm/game%20abstraction.aaai15SMT.pdf
5. Automated Action Abstraction of Imperfect Information Extensive-Form Games. https://poker.cs.ualberta.ca/publications/AAAI11.pdf
6. Evaluating State-Space Abstractions in Extensive-Form Games. https://poker.cs.ualberta.ca/publications/AAMAS13-abstraction.pdf
7. WAVEFORM CLUSTERING AS AN EXTENSIVE FORM ABSTRACTION
TECHNIQUE FOR POKER. https://www.digipen.edu/sites/default/files/public/docs/theses/kevin-kauth-digipen-master-of-science-in-computer-science-thesis-waveform-clustering-as-an-extensive-form-abstraction-technique-for-poker.pdf

-----------------

### Future Work:
1. [Deep Q-Learning](https://arxiv.org/abs/1312.5602)
2. [Deep Reinforcement Learning from Self-Play in Imperfect-Information Games](https://arxiv.org/abs/1603.01121)
3. [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164)

#### To knock or not: The game of pig. Play to maximize utility from hand, not for the whole game. Figure out whether to knock, to do undercut, or to go gin?

#### Comparing the to game of pig, maximize hand score.early game: maximize hand score, as the game goes by, player need to tkae greter risk, change play method to win game. => Non-linear policy of strategy.

* Tabular rl/ml approach.
* ml is not deep leanring, ml is about statistic, learn better function model, not just neural network. Leveraging the data to give better action.

* Combination of algorithms: 
    * Hand estimator function
    * Given estimation try to predict other card unknown?
    * Try to predict safety discard?

* Bayesian Network? Monte Carlo? Markup chain.

* Want to have some kind of a data-driven bot that takes the time off line to learn.

* collect a lot of playing data, learn from that, like supervised learning.

* Number of seed ? and dive deep down into the things.

-----------
## Author: Alex Nguyen
## Gettysburg College