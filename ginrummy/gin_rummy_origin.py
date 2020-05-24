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

class GinPlayer:
    def __init__(self):
        pass

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
