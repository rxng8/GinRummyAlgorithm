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

    # @staticmethod
    # def getCard(id: int):
    #     return all_cards[id]

    @staticmethod
    def getCard(rank: int, suit: int):
        return all_cards[suit * n_ranks + rank]

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

    GOAL_SCORE=100
    GIN_BONUS=25
    UNDERCUT_BONUS=25
    MAX_DEADWOOD=10
    DEADWOOD_POINTS=np.zeros(n_ranks)
    cardBitstrings = np.zeros(n_cards)
    meldBitstrings: List[List[int]]
    meldBitstringToCardsMap: Dict[int, List[Card]]

    # Initialize deadwood point
    for rank in range(n_ranks):
        DEADWOOD_POINTS[rank] = min(rank + 1, 10)

    # Initiate Card Bitstring
    bitstring = 1
    for i in range(n_cards):
        cardBitstrings[i] = bitstring
        bitstring <<= 1

    # build list of lists of meld bitstring where each subsequent meld 
    # bitstring in the list is a superset of previous meld bitstrings
    meldBitstrings: List[List[int]]
    meldBitstringToCardsMap: Dict[int, List[Card]]

    # build run meld lists
    for suit in range(n_suit):
        for runRankStart in range(n_ranks - 2):
            bitstringList: List[int] = []
            cards: List[int] = []
            c = Card.getCard(runRankStart, suit)
            cards.append(c)
            meldBitstring = cardBitstrings[c.getId()]
            c = Card.getCard(runRankStart + 1, suit)
            cards.append(c)
            meldBitstring |= cardBitstrings[c.getId()]
            for rank in range(runRankStart + 2, n_ranks, 1):
                c = Card.getCard(rank, suit)
                cards.append(c)
                meldBitstring |= cardBitstrings[c.getId()]
                bitstringList.append(meldBitstring)
                meldBitstringToCardsMap[meldBitstring] = cards.copy()
            meldBitstrings.append(bitstringList)

    for rank in range(n_ranks):
        cards: List[Card]
        for suit in range(n_suit):
            cards.append(Card.getCard(rank, suit))
        for suit in range(n_suit):
            cardSet = cards.copy()
            if suit < n_suit:
                cardSet.remove(Card.getCard(rank, suit))
            bitstringList: List[int] = []
            meldBitstring = 0
            for card in cardSet:
                meldBitstring |= cardBitstrings[card.getId()]
            bitstringList.append(meldBitstring)
            meldBitstringToCardsMap[meldBitstring] = cardSet
            meldBitstrings.append(bitstringList)


    # TODO: transfer more function from java:




    def __init__(self) :
        pass

    @staticmethod
    def cardsToBitstring(cards: List[Card]) -> int:
        pass

    @staticmethod
    def get_deadwood_point(cards: List[Card]):
        pass

    @staticmethod
    def get_best_meld(self):
        pass


# %%
'''
Main game environment, with data generation (State) process
'''
class GinGame:

    HAND_SIZE = 10
    VERBOSE = False

    def __init__(self, player0, player1):
        self.players: List[GinPlayer] = [player0, player1]

    def play(self) -> int:
        scores = [0, 0]
        

# %%



# %%
