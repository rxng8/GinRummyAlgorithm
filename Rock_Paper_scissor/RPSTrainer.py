# %%

### @Author: Alex ###


# %%

# Import

from typing import List

import random
import numpy as np

# %%

# CONSTANT

ACTIONS = [0, 1, 2]
ACTIONS_MAP = {0: "Rock", 1: "Paper", 2: "Scissor"}
N_ACTIONS = 3
opponent_strategy = [0.4, 0.3, 0.3]

# %%

# RPS Trainer class

class RPSTrainer:

    # Definition
    def __init__(self):
        self.regret_sum : List[float] = np.zeros(N_ACTIONS)
        self.strategy : List[float] = np.zeros(N_ACTIONS)
        self.strategy_sum : List[float] = np.zeros(N_ACTIONS)

    # Normalize a strategical vector
    def normalize (self, strategical_vector : List[float]) -> List[float] :
        normalized_vector = np.zeros(len(strategical_vector))
        normalizing_sum = sum(strategical_vector)
        # print("normalizing_sum: " + str(normalizing_sum))
        for i in range(N_ACTIONS):
            if normalizing_sum > 0 :
                normalized_vector[i] = strategical_vector[i] / normalizing_sum
            else :
                normalized_vector[i] = 1.0 / N_ACTIONS

            # print(normalized_vector[i])
        # print(normalized_vector)
        return normalized_vector

    # Get new strategy vector from regret_sum vector by using regret matching algorithm
    def get_strategy (self) -> List[float] :
        for i in range(N_ACTIONS):
            self.strategy[i] = self.regret_sum[i] if self.regret_sum[i] > 0 else 0

        # print("Strategy before copying from regret")
        # print(self.strategy)

        self.strategy = self.normalize(self.strategy)

        # print("Strategy after being normalized from regret")
        # print(self.strategy)

        for i in range (N_ACTIONS):
            self.strategy_sum[i] += self.strategy[i]
        return self.strategy

    # Get the actual action using the strategy vector by using random object
    def get_action (self, strategy_vector) -> int :
        # random_prob = random.random()
        random_prob = 0.8
        action = 0
        cum = 0
        for i in range(N_ACTIONS):
            cum += strategy_vector[i]
            if random_prob < strategy_vector[i]:
                action = i
                break
        
        return action

    # Compute action ultilities vector and accumulate the regrets vector, add that to the regret_sum vector.
    def accumulate_regret (self) -> List[float] :

        # Action ultilities vector 
        action_util = np.zeros(N_ACTIONS)
        
        opponent_action = self.get_action(opponent_strategy)
        player_action = self.get_action(self.strategy)

        print("You played " + ACTIONS_MAP[player_action])
        print("Your opponent played " + ACTIONS_MAP[opponent_action])


        # Action util of current player build
        action_util[opponent_action] = 0 # when two action are equal, util = 0.
        action_util[0 if opponent_action == N_ACTIONS - 1 else opponent_action + 1] = 1 # Current next action always win the previous action.
        action_util[N_ACTIONS - 1 if opponent_action == 0 else opponent_action - 1] = -1 # Likewise, current previous action always lose next action.

        # Calculate regret vectors: regret = u(delta(i, s'), delta(-i, s)) - u(action)
        for i in range(N_ACTIONS):
            self.regret_sum[i] += action_util[i] - action_util[player_action]

        print("Your current strategy matrix after this turn is: ")
        print(self.strategy)
        print("Your current regret matrix after this turn is: ")
        print(self.regret_sum)

        return self.regret_sum

    # Calculate Average strategy
    def get_average_strategy (self) -> List[float] :
        return self.normalize(self.strategy_sum)

    # Trainer
    def train(self, num_iter = 10) :
        
        if num_iter < 10:
            num_iter = 10
        self.strategy = self.get_strategy()
        print("Your initial strategy matrix is: ")
        print(self.strategy)
        print("Your initial regret matrix is: ")
        print(self.regret_sum)

        for _ in range(num_iter):
            print(f"\nNext Turn:\n")
            self.strategy = self.get_strategy()
            self.regret_sum = self.accumulate_regret()
        
        return self.get_average_strategy()

# %%

RPS = RPSTrainer()


# %%

RPS.train()

# %%

RPS.strategy_sum

# %%

# RPS Trainer Two player

class RPSTrainerTwoPlayer:

    # Definition
    def __init__(self, num_iter = 10):
        self.regret_sum : List[float] = np.zeros(N_ACTIONS)
        self.strategy : List[float] = np.zeros(N_ACTIONS)
        self.strategy_sum : List[float] = np.zeros(N_ACTIONS)

        self.regret_sum_op : List[float] = np.zeros(N_ACTIONS)
        self.strategy_op : List[float] = np.zeros(N_ACTIONS)
        self.strategy_sum_op : List[float] = np.zeros(N_ACTIONS)

        self.totalWin = 0
        self.totalWin_Op = 0
        self.draw = 0

        self.num_iter = num_iter

    # Normalize a strategical vector
    def normalize (self, strategical_vector : List[float]) -> List[float] :
        normalized_vector = np.zeros(len(strategical_vector))
        normalizing_sum = sum(strategical_vector)
        # print("normalizing_sum: " + str(normalizing_sum))
        for i in range(N_ACTIONS):
            if normalizing_sum > 0 :
                normalized_vector[i] = strategical_vector[i] / normalizing_sum
            else :
                normalized_vector[i] = 1.0 / N_ACTIONS

            # print(normalized_vector[i])
        # print(normalized_vector)
        return normalized_vector

    # Get new strategy vector from regret_sum vector by using regret matching algorithm
    def get_strategy (self) -> List[float] :

        for i in range(N_ACTIONS):
            self.strategy[i] = self.regret_sum[i] if self.regret_sum[i] > 0 else 0

        # print("Strategy before copying from regret")
        # print(self.strategy)

        self.strategy = self.normalize(self.strategy)

        # print("Strategy after being normalized from regret")
        # print(self.strategy)

        for i in range (N_ACTIONS):
            self.strategy_sum[i] += self.strategy[i]
        return self.strategy

    # Get new strategy vector from regret_sum vector by using regret matching algorithm
    def get_op_strategy (self) -> List[float] :

        for i in range(N_ACTIONS):
            self.strategy_op[i] = self.regret_sum_op[i] if self.regret_sum_op[i] > 0 else 0

        # print("Strategy before copying from regret")
        # print(self.strategy)

        self.strategy_op = self.normalize(self.strategy_op)

        # print("Strategy after being normalized from regret")
        # print(self.strategy)

        for i in range (N_ACTIONS):
            self.strategy_sum_op[i] += self.strategy_op[i]
        return self.strategy_op

    # Get the actual action using the strategy vector by using random object
    def get_action (self, strategy_vector) -> int :
        random_prob = random.random()
        # random_prob = 0.8
        action = 0
        cum = 0
        for i in range(N_ACTIONS):
            cum += strategy_vector[i]
            if random_prob < strategy_vector[i]:
                action = i
                break
        
        return action

    # Compute action ultilities vector and accumulate the regrets vector, add that to the regret_sum vector.
    def accumulate_regret (self) -> List[float] :

        # Action ultilities vector 
        action_util = np.zeros(N_ACTIONS)
        action_util_op = np.zeros(N_ACTIONS)
        
        opponent_action = self.get_action(self.strategy_op)
        player_action = self.get_action(self.strategy)

        print("You played " + ACTIONS_MAP[player_action])
        print("Your opponent played " + ACTIONS_MAP[opponent_action])

        if player_action == opponent_action:
            print("Draw!")
            self.draw += 1
        elif player_action - opponent_action == 1 or player_action - opponent_action == -3:
            print("You won!")
            self.totalWin += 1
        else:
            print("Your opponent won!")
            self.totalWin_Op += 1

        # Action util of current player build
        action_util[opponent_action] = 0 # when two action are equal, util = 0.
        action_util[0 if opponent_action == N_ACTIONS - 1 else opponent_action + 1] = 1 # Current next action always win the previous action.
        action_util[N_ACTIONS - 1 if opponent_action == 0 else opponent_action - 1] = -1 # Likewise, current previous action always lose next action.

        # Action util of current player's oppenent build
        action_util_op[player_action] = 0 # when two action are equal, util = 0.
        action_util_op[0 if player_action == N_ACTIONS - 1 else player_action + 1] = 1 # Current next action always win the previous action.
        action_util_op[N_ACTIONS - 1 if player_action == 0 else player_action - 1] = -1 # Likewise, current previous action always lose next action.

        # Calculate regret vectors: regret = u(delta(i, s'), delta(-i, s)) - u(action)
        for i in range(N_ACTIONS):
            self.regret_sum[i] += action_util[i] - action_util[player_action]
            self.regret_sum_op[i] += action_util_op[i] - action_util_op[opponent_action]

        # print("Your current strategy matrix after this turn is: ")
        # print(self.strategy)
        # print("Your current regret matrix after this turn is: ")
        # print(self.regret_sum)

        return self.regret_sum, self.regret_sum_op

    # Calculate Average strategy
    def get_average_strategy (self) -> List[float] :
        return self.normalize(self.strategy_sum), self.normalize(self.strategy_sum_op), self.totalWin, self.totalWin_Op, self.draw

    # Trainer
    def train(self) :
        
        if self.num_iter < 10:
            self.num_iter = 10
        self.strategy = self.get_strategy()
        print("Your initial strategy matrix is: ")
        print(self.strategy)
        print("Your initial regret matrix is: ")
        print(self.regret_sum)

        for _ in range(self.num_iter):
            print(f"\nNext Turn:\n")
            self.strategy = self.get_strategy()
            self.strategy_op = self.get_op_strategy()
            self.regret_sum, self.regret_sum_op = self.accumulate_regret()
        
        return self.get_average_strategy()


# %%


r = RPSTrainerTwoPlayer(1000000)
#%%

r.train()

# %%
'''
total: 1,000,000 rounds
 win : 332117,
 lose: 334561,
 draw: 333322)


'''

# %%
