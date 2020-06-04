# Algorithm of Reinforment Leanring for Imperfect Information Card Game: Gin Rummy

## Current Work:

Please navigate to `/ginrummy/GinRummyEAAIorigin/GinRummyEAAI/CFRPlayer.java` to see what I'm working on.

------------
# Report:

## Report 3, date: June 9, 2020
### 1. Modeling the abstraction of the public state.
* The public state is defined by a tuple S = (P, D, H) where P is a vector of player's hand, D is the vector of discarded pile, and H is the history of the game.
* Feature Analysis:
    * TODO
    * TODO
* Public state abstraction: 


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

-----------------

### Future Work:
1. [Deep Q-Learning](https://arxiv.org/abs/1312.5602)
2. [Deep Reinforcement Learning from Self-Play in Imperfect-Information Games](https://arxiv.org/abs/1603.01121)
3. [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164)

#### To knock or not: The game of pig. Play to maximize utility from hand, not for the whole game. Figure out whether to knock, to do undercut, or to go gin?

#### Comparing the to game of pig, maximize hand score.early game: maximize hand score, as the game goes by, player need to tkae greter risk, change play method to win game. => Non-linear policy of strategy.

-----------
## Author: Alex Nguyen
## Gettysburg College