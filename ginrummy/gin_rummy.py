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
    Given a public state, evaluate a current player's turn with a tree builder within a certain depth.
    Params:
        1. depth_limit: int. This is the depth limit of the tree.
        2.
        3.
    Return:
    '''
    def evaluate_step(self, depth_limit=2):

        # If it is the terminal states, get payoff function (different in deadwood point! Or gin point!)

        # If not, then use strategy vector from information set to have specific draw action.

        # After drawing, re-evaluate opponent counterfactual values, strategy, regrets? - not by evaluating recursively - but by the given public state and value function (Deep Stack Paper)? trained model CNN, RNN, LSTM, GRU? 
        #   - Base on the just-drawn card and the given data in the public state, determine what card opponent should have?
        #   - What melds opponent should have.

        # After that, 

        # Given the public states, decide if 

        pass



    '''
    Get draw action base on Information Set's strategy vector
    params:
        1.
    return: 
    '''
    def get_draw_action(self):
        pass

    '''
    Get discard action base on Information Set's strategy vector
    params:
        1.
    return: 
    '''
    def get_discard_action(self):
        pass

    '''
    Params:
        1. 
        2.
    Return: Node actual value (utility)
    '''
    # def cfr(self) -> float:
    #     # Get player?
    #     opponent = 1 - active_player

    #     # Return payoff if state is terminal
    #     if self.isTerminal(S: PublicState):
    #         payoff = self.get_payoff(history, cards, active_player)
    #         return payoff
        
    #     # new information and convert to public state to put in information set?
    #     information = str(cards[active_player]) + history

    #     # Get information set base on the public state or create if non-existed
    #     infoSet = self.get_info_set(information)
    #     infoSet.infoSet = information

    #     # Get each strategy from the information set
    #     strategy = infoSet.get_strategy(reach_probability_p0 if active_player == 0 else reach_probability_p1)

    #     # For each action, call cfr with newer public state
        
    #     counterfactual_values = np.zeros(N_ACTIONS)
    #     node_actual_value = 0

    #     for i in range(N_ACTIONS):

    #         if len(history) == 3:
    #             return KuhnTrainer.get_payoff(history, cards, active_player)

    #         counterfactual_values[i] =\
    #             - self.cfr(cards, history + action, reach_probability_p0 * strategy[i], reach_probability_p1, opponent) if active_player == 0 else \
    #             - self.cfr(cards, history + action, reach_probability_p0, reach_probability_p1 * strategy[i], opponent)
    #         node_actual_value += strategy[i] * counterfactual_values[i]

    #     # For each action, compute and accumulate counterfactual regret. (Regret Matching)
    #     for i in range(N_ACTIONS):
    #         counterfactual_regret = counterfactual_values[i] - node_actual_value
    #         infoSet.regret_sum[i] += (reach_probability_p0 if active_player == 0 else reach_probability_p1) * counterfactual_regret
        
    #     return node_actual_value

    '''
    Check if a state is a terminal state
    params:
    return: (bool) Whether it is a terminal state
    '''
    @staticmethod
    def is_terminal():
        pass

    '''
    Reward function for the specified tree.
    params:
        1.
        2.
    return: 
    '''
    @staticmethod
    def get_payoff():
        pass

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

    