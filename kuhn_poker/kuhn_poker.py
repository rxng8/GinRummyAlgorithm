# %%

'''
This is the full implementation of CRF training Kuhn Poker game:
'''

# %%

# Import Statement
from typing import Dict, List
import numpy as np
import random


PASS = 0
BET = 1
N_ACTIONS = 2
R = random.random()


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


class KuhnTrainer:
    def __init__(self, cards:List[int], n_iter=10):
        self.n_iter = n_iter
        self.cards = cards
        self.infoSet_map: Dict[str, InformationSet]  = {}
        # self.t = False

    def train(self) -> float:
        node_actual_value = 0.0
        for _ in range(self.n_iter):
            # Shuffle
            self.shuffle()
            node_actual_value += self.cfr(self.cards, '', 1, 1, 0)

        # self.shuffle()
        # node_actual_value += self.cfr(self.cards, '', 1, 1, 0)

        print("Average game value: {}".format(node_actual_value / self.n_iter))

        for _, v in self.infoSet_map.items():
            print(v.toString())

    def cfr(self, cards:List[int], history:str, reach_probability_p0:float, reach_probability_p1:float, active_player: int) -> float:
        
        # Get player?
        opponent = 1 - active_player
        # print("current player: " + str(active_player))

        # print(KuhnTrainer.isTerminal(history))

        # Return payoff if state is terminal
        # if self.t:
        #     print("Trueeeee")
        if KuhnTrainer.isTerminal(history):
            print("History: " + history + " -> True")
            payoff = KuhnTrainer.get_payoff(history, cards, active_player)
            return payoff

        print("History: " + history + " -> False")
        
        # String new information to put in information set?
        information = str(cards[active_player]) + history
        
        # print("Current history: " + history + "done!")

        # Get information set or create if non-existed
        infoSet = self.get_info_set(information)
        infoSet.infoSet = information
        # For each action, call cfr with additional history and probs
        strategy = infoSet.get_strategy(reach_probability_p0 if active_player == 0 else reach_probability_p1)
        counterfactual_values = np.zeros(N_ACTIONS)
        node_actual_value = 0

        for i in range(N_ACTIONS):
            action = "P" if i == 0 else "B"
            next_history = history + action

            # print(next_history)

            # CURRENT BUG
            if len(history) == 3:
                return KuhnTrainer.get_payoff(history, cards, active_player)


            counterfactual_values[i] =\
                - self.cfr(cards, history + action, reach_probability_p0 * strategy[i], reach_probability_p1, opponent) if active_player == 0 else \
                - self.cfr(cards, history + action, reach_probability_p0, reach_probability_p1 * strategy[i], opponent)
            node_actual_value += strategy[i] * counterfactual_values[i]

        # For each action, compute and accumulate counterfactual regret.
        for i in range(N_ACTIONS):
            counterfactual_regret = counterfactual_values[i] - node_actual_value
            infoSet.regret_sum[i] += (reach_probability_p0 if active_player == 0 else reach_probability_p1) * counterfactual_regret
        
        return node_actual_value

    # def cfr2(self, cards: List[str], history: str, reach_probabilities: np.array, active_player: int):
    #     if KuhnTrainer.isTerminal(history):
    #         return KuhnTrainer.get_payoff(history, cards, active_player)

    #     my_card = cards[active_player]
    #     info_set = self.get_info_set(str(my_card) + history)

    #     strategy = info_set.get_strategy(reach_probabilities[active_player])
    #     opponent = (active_player + 1) % 2
    #     counterfactual_values = np.zeros(N_ACTIONS)

    #     for ix in range(N_ACTIONS):
    #         action_probability = strategy[ix]
    #         action = "B" if ix == 0 else "C"
    #         # compute new reach probabilities after this action
    #         new_reach_probabilities = reach_probabilities.copy()
    #         new_reach_probabilities[active_player] *= action_probability

    #         # recursively call cfr method, next player to act is the opponent
    #         counterfactual_values[ix] = -self.cfr2(cards, history + action, new_reach_probabilities, opponent)

    #     # Value of the current game state is just counterfactual values weighted by action probabilities
    #     node_value = counterfactual_values.dot(strategy)
    #     for ix in range(N_ACTIONS):
    #         info_set.cumulative_regrets[ix] += reach_probabilities[opponent] * (counterfactual_values[ix] - node_value)

    #     return node_value

    # def train2(self) -> int:
    #     util = 0
    #     for _ in range(self.n_iter):
    #         cards = random.sample(self.cards, 2)
    #         history = ''
    #         reach_probabilities = np.ones(2)
    #         util += self.cfr2(cards, history, reach_probabilities, 0)
    #     return util


    def shuffle (self):
        # Durstenfeld version of the Fisher-Yates shuffle
        for i in range(len(self.cards) - 1, -1, -1):
            # Include both 0 and i
            j = random.randint(0, i)
            
            # Swap
            tmp = self.cards[i]
            self.cards[i] = self.cards[j]
            self.cards[j] = tmp


    def get_info_set(self, information: str) -> InformationSet:
        if information not in self.infoSet_map:
            self.infoSet_map[information] = InformationSet()
        return self.infoSet_map[information]

    @staticmethod
    def get_payoff(history: str, cards: List[int], active_player: int) -> int:
        payoff: int = 0
        if history in ['BP', 'PBP']:
            return 1
        else: 
            if history in ['PP']:
                payoff = 1
            elif history in ['PBB', 'BB']:
                payoff = 2
            player_card = cards[active_player]
            opponent_card = cards[1 - active_player]
            if player_card > opponent_card:
                return payoff
            elif player_card < opponent_card:
                return -payoff

    @staticmethod
    def isTerminal(history: str) -> bool :
        return history in ['BP', 'BB', 'PP', 'PBP', 'CPP']



# %%

CARDS = [1,2,3]

trainer = KuhnTrainer(CARDS, n_iter=10000)

trainer.train()

