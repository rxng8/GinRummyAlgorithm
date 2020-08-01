# %%

'''
Gin Rummy Game and Trainer
'''


# %%

from typing import Dict, List
import random
import numpy as np

# %%
# CONSTANT
DRAW_ACTIONS = ['U', 'D'] # U = Draw face up, D = draw face down
R = random.random()
rank_names = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
n_ranks = len(rank_names)
suit_names = ["C", "H", "S", "D"]
n_suit = len(suit_names)
n_cards = n_ranks * n_suit


# %%
class Card:
    def __init__(self, rank: int, suit: int):
        self.rank = rank
        self.suit = suit

    def toString(self) -> str:
        return rank_names[self.rank] + suit_names[self.suit]

    def getId(self) -> int:
        return suit * n_ranks + rank

# %%

# CONSTANT

all_cards: List[Card] = []
str_to_card: Dict[str, Card] = {}
str_to_id: Dict[str, int] = {}
id_to_str: Dict[int, str] = {}
i = 0
for suit in range(n_suit):
    for rank in range(n_ranks):
        c = Card(rank, suit)
        all_cards.append(c)
        str_to_card[c.toString()] = c
        str_to_id[c.toString()] = c.getId()
        id_to_str[c.getId()] = c.toString()
DISCARD_ACTIONS = [c.toString() for c in all_cards]

# Shuffle in place
def shuffle() -> List[Card]:
    deck = all_cards[:]
    random.shuffle(deck)
    return deck

shuffled_deck = shuffle()
# %%


# %%


'''
1. What determine our strategy to pick card? Either draw face down or draw face up?
    - Whether we see that the card in the discarded pile benefit us (Form meld)? How much benefit? Probabilities?
    - Whether the it increase the deadwood point or not ?

2. What determine our strategy to discard card? Either discard 1 card in our 11 cards?
    - Whether it decreases as much deadwood point as we want?
    - History of the discarded pile? So that opponent cannot form a meld when we discard? Probability? 
'''

''' Food for thoughts:
Should we use hands as information in each different hand we have a different information set?
    If says, we use hands as information set, then when we have 10 cards in hand, 
    the number of information set can be 52C10 = 15 billions different set. And similarly,
    when we draw an additional card, we have 11 cards in hand, so there will be a total of 52C11 = 60 billions set more.
    So we will have roughly 75 billions different information set?

In the Deepstack paper:
        "Suppose we have taken actions according to a particular solution strategy but then in some public state forget this strategy. 
    Can we reconstruct a solution strategy for the subtree without having to solve the entire game again? We can, 
    through the process of re-solving (17). We need to know both our range at the public state and a vector of expected
    values achieved by the opponent under the previous solution for each opponent hand (24). With
    these values, we can reconstruct a strategy for only the remainder of the game, which does not
    increase our overall exploitability. Each value in the opponent’s vector is a counterfactual value,
    a conditional “what-if” value that gives the expected value if the opponent reaches the public
    state with a particular hand. The CFR algorithm also uses counterfactual values, and if we use
    CFR as our solver, it is easy to compute the vector of opponent counterfactual values at any
    public state."

    I think this is where we dont actually need to use information Set to actually store all the combination of data case.
'''
'''
params:
    1. S: PublicState. S is a public state
'''
class InfoSet:
    def __init__(self, S):
        self.infoSet: str
        self.regret_sum: List[float]
        self.strategy: List[float]
        self.strategy_sum: List[float]
        self.op_cfr_values: List[float]
        self.n: int

        self.S = S

    def normalize(self, vector: List[float]) -> List[float]:
        normalized_vector = np.zeros(self.n)
        n_sum = sum(vector)
        for i in range(self.n):
            if n_sum > 0:
                normalized_vector[i] = vector[i] / n_sum
            else:
                normalized_vector[i] = 1.0 / self.n
        return normalized_vector

    def get_strategy (self, reach_probability : float) -> List[float] :
        for i in range (self.n):
            self.strategy[i] = self.regret_sum[i] if self.regret_sum[i] > 0 else 0
        self.strategy = self.normalize(self.strategy)
        for i in range(self.n):
            self.strategy_sum[i] += reach_probability * self.strategy[i]
        return self.strategy

    def get_average_strategy(self) -> List[float]:
        return self.normalize(self.strategy_sum)

    def toString (self) -> str :
        return "{}: {}".format(self.infoSet, self.get_average_strategy())
        
'''
1. What determine our strategy to pick card? Either draw face down or draw face up?
    - Whether we see that the card in the discarded pile benefit us (Form meld)? How much benefit? Probabilities?
    - Whether it increase the deadwood point or not ?

+ FOR THE ALGORITHM TO BE SIMPLE, WE WILL DRAW FACE UP IF THAT CARD FORM MELD, OTHERWISE DRAW FACE DOWN.
+ FOR FUTURE WORK, WE WILL LOOK INTO THE PROBABILITY OF FORMING MELDS OF EACH CARD, AND THE FACE UP CARD ALSO.
    AND ALSO CONCATENATE WITH THE PROBABILITY OF 
'''
class DrawInfoSet (InfoSet):
    def __init__(self):
        self.infoSet = ''
        self.n = len(DRAW_ACTIONS)
        self.regret_sum = np.zeros(self.n)
        self.strategy = np.zeros(self.n)
        self.strategy_sum = np.zeros(self.n)
        self.op_cfr_values: List[float] = np.zeros(self.n)

    def save_model(self):
        pass


'''
2. What determine our strategy to discard card? Either discard 1 card in our 11 cards?
    - Whether it decreases as much deadwood point as we want?
    - History of the discarded pile? So that opponent cannot form a meld when we discard? Probability? 
'''
class DiscardInfoSet (InfoSet):
    def __init__(self):
        self.infoSet = ''
        self.n = len(DISCARD_ACTIONS)
        self.regret_sum = np.zeros(self.n)
        self.strategy = np.zeros(self.n)
        self.strategy_sum = np.zeros(self.n)
        self.op_cfr_values: List[float] = np.zeros(self.n)

    def save_model(self):
        pass

'''
Data Structure of a public state.
If so, we can have 52C10 * (52!) different public states
Params:
    1. Cards: List[Card]: List of Cards in hands.
    2. Top Discards: List[Card]. The top card of the discard pile
    2. Discards: List[Card]. List of cards in the discarded pile but not the top card.
    4. Opponent known cards: List[Card]. cards picked up from discard pile, but not discarded
    5. The unknown cards: List[Card]. cards in stockpile or in opponent hand (but not known)
'''
class PublicState:
    def __init__(self, cards: List[Card], discards: List[Card]):
        self.cards = cards
        self.discards = discards

# %%

class GinTrainer:
    def __init__(self, n_iter: int=1):
        self.n_iter = n_iter

    '''
    INPUT: Public state S, player range r1 over our information sets in S, opponent counterfactual values
        v2 over their information sets in S, and player information set I ∈ S
    OUTPUT: Chosen action a, and updated representation after the action (S(a), r1(a), v2(a))
    (a range is the probability distribution over the player’s possible hands given that the public state is reached)
    '''
    def resolves (self, S, r1, v2, I):

        # Using arbitrary initial strategy profile for player

        # Using arbitrary opponent range

        # Initialize regret

        # For each timestep:
        #   Give counterfactual values of the player
        #   Update subtree strategy with regret matching
        #   Range Gadget (CFR-D? CFR+?)

        pass


    '''
    Given a public state, evaluate a current player's turn with a tree builder within a certain depth, 
    compute the cfvs, update regrets by regret matching, and re-compute strategy each turn.
    NOTE: This method is not taking into account the start of the game where each player decide to pass.
    NOTE: This method is not taking into account the end of the game where each player decide to knock or go gin.
    Params:
        1. cards: List[int]. The deck of cards in play. In this case, array of 52 cards.
        2. history: str. The string representation of the game state, including both public state and private state. 
            TODO: Need to find a more efficient way of storing state.
        3. reach_probability: List[float]. Reach probability of the actual current node of each player. 
            In this case the length of the array is 2 because there are 2 players.
        4. active_player: int. Current active player in the form 0 or 1.
        5. will_draw: bool. This is a boolean telling the recursive method which strategy to evaluate. 
            This is to ensure going through each state in order (Each player draw and then discard respectively).
        6. depth_limit: int. This is the depth limit of the tree, if a certain depth is reach, we will use a pretrained neural network 
            to predict the payoff right away so we don't have to find our way to the terminal state.
    Return: nodeUtil: float. This an actual node utility computed from traversing to terminal state or certain max tree depth.
    '''
    def evaluate_step(self, cards:List[int], history:str, reach_probability: List[float], active_player: int, will_draw: bool, depth_limit=10):
        # FOOD FOR THOUGHTS:
        # After drawing, re-evaluate opponent counterfactual values, strategy, regrets? - but by the given public state 
        #   and value function (Deep Stack Paper)? trained model CNN, RNN, LSTM, GRU? 
        #   (If terminal then get payoff, else if reach max depth limit and predict the payoff or utility with neural nets)
        #   - Base on the just-drawn card and the given data in the public state, determine what card opponent should have?
        #   - What melds opponent should have.
        #   - Since computing cfv and cfr, we have to traverse the entire game tree, so I'm thinking about each decision is a node, 
        #       and we traverse 2 consecutive time step for each player, and update 2 different vector of strategy, regret cfv for each player

        # If it is the terminal states, get payoff function (different in deadwood point! Or gin point!)
        if GinTrainer.isTerminal(history):
            return GinTrainer.get_payoff()

        # Else if, reach max depth, evaluate the payoff values with pretrained neural network model.
        elif depth_limit == 0:
            return self.neural_net_evaluate()
        # If not:

        # Initiate the node util value (delta(action)).
        nodeUtil = 0.0

        # then if will_draw is True, use strategy vector from information set to have specific DRAW action.
        if will_draw:
            
            # Get information set and re-compute strategy with current node reach probability and information set
            strategy: List[float]

            # Initiate the node util value (delta(action)).
            cfvs = np.zeros(len(DRAW_ACTIONS))

            # For each draw action, compute node utility by traversing
            for i in range(len(DRAW_ACTIONS)):
                action = DRAW_ACTIONS[i]
                
                # compute new reach probability vector with regard to that action
                new_reach_probability: List[float]

                # Traversal-y compute and update strategy, regrets, cfvs, and cfrs with respect to DISCARD action.
                cfvs[i] = self.evaluate_step(cards, history + action, new_reach_probability, active_player, not will_draw, depth_limit - 1)

                # Add the utility of each action to this node's utility
                nodeUtil += cfvs[i] * strategy[i]
                pass # End for
            

            # Evaluate node_actual_utility and regret. Regret = personal_counter_factual value[] - node actual utility


            # Update strategy and regret by regret matching algorithm


            # Return node_actual_utility.                

            pass # end first part of if

        # Else, use strategy vector from information set to have specific DISCARD action.
        else:
            # Get information set and re-compute strategy with current node reach probability and information set
            strategy: List[float]

            # Initiate the node util value (delta(action)).
            cfvs = np.zeros(len(DRAW_ACTIONS))
            
            # For each draw action, compute node utility by traversing
            for i in range(len(DISCARD_ACTIONS)):
                action = DRAW_ACTIONS[i]
                
                # compute new reach probability vector with regard to that action
                new_reach_probability: List[float]

                # Traversal-y compute and update strategy, regrets, cfvs, and cfrs with respect to DISCARD action.
                nodeUtil += self.evaluate_step(cards, history + action, new_reach_probability, 1 - active_player, not will_draw, depth_limit - 1)

                # Add the utility of each action to this node's utility
                nodeUtil += cfvs[i] * strategy[i]
                pass # End for
            
            # Evaluate node_actual_utility and regret. Regret = personal_counter_factual value[] - node actual utility


            # Update strategy and regret by regret matching algorithm


            # Return node_actual_utility. 
            pass # End else

        return nodeUtil


    '''
    Return an infoset
    '''
    def get_info_set(self, information: str) -> InformationSet:
        """[summary]

        Args:
            information (str): [description]

        Returns:
            InformationSet: [description]
        """
        pass


    '''
    Check if a state is a terminal state
    params:
    return: (bool) Whether it is a terminal state
    '''
    @staticmethod
    def isTerminal(history: str) -> bool :
        pass


    '''
    Reward function for the specified tree.
    params:
        1.
        2.
    return: 
    '''
    @staticmethod
    def get_payoff(history: str, cards: List[int], active_player: int) -> int:
        pass

    '''
    Predict utility for an actual node
    params:
        1.
        2.
    return:
    '''
    def neural_net_evaluate() -> float:
        pass

    '''
    Train
    '''
    def train(self):
        node_value = 0.0
        # for _ in range(self.n_iter):
            # node_value = self.cfr()
        
        # return node_value

# %%

class Util:
    def __init__(self) :
        pass

    @staticmethod
    def get_deadwood_point(cards: List[Card]):
        pass

    def get_best_meld(self):
        pass


# %%
'''
Main game environment, with data generation (State) process
'''
class GinGame:
    def __init__(self):
        pass

    