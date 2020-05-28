# %%

import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(100):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

# %%

from rlcard.games.gin_rummy import Game
from rlcard.games.gin_rummy.utils import action_event

game = Game(True)


# %%

game.init_game()

# %%

game.step(action=game.decode_action(73))