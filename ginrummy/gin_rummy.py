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

    # Shuffle in place
    @staticmethod
    def shuffle(seed: int) -> List[Card]:
        deck = all_cards[:]
        random.shuffle(deck, seed)
        return deck
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

In the Deepstack paper, can we limit the depth lookahead via intuition?
'''
class InformationSet:
    def __init__(self):
        self.infoSet = ''
        self.regret_sum = np.zeros(N_ACTIONS)
        self.strategy = np.zeros(N_ACTIONS)
        self.strategy_sum = np.zeros(N_ACTIONS)
        

    def normalize(self, vector: List[float]) -> List[float]:
        normalized_vector = np.zeros(N_ACTIONS)
        n_sum = sum(vector)
        for i in range(N_ACTIONS):
            if n_sum > 0:
                normalized_vector[i] = vector[i] / n_sum
            else:
                normalized_vector[i] = 1.0 / N_ACTIONS
        return normalized_vector

    def get_strategy (self, reach_probability : float) -> List[float] :
        for i in range (N_ACTIONS):
            self.strategy[i] = self.regret_sum[i] if self.regret_sum[i] > 0 else 0
        self.strategy = self.normalize(self.strategy)
        for i in range(N_ACTIONS):
            self.strategy_sum[i] += reach_probability * self.strategy[i]
        return self.strategy

    def get_average_strategy(self) -> List[float]:
        return self.normalize(self.strategy_sum)

    def toString (self) -> str :
        return "{}: {}".format(self.infoSet, self.get_average_strategy())
