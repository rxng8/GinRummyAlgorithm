# Algorithm of Reinforment Leanring for Imperfect Information Card Game: Gin Rummy

## For professor Neller:

Please navigate to [./ginrummy/gin_rummy.py](https://github.com/rxng8/GinRummyAlgorithm/blob/master/ginrummy/gin_rummy.py) if you want to know which feature I'm working on. I'm currently working on the evaluate_step() method that recursively traverse the game tree and update regrets by regret matching and return the actual node utility (delta(action)). This method is also very similar to the CFR Minimization Framework. 

When you get to the file please search for evaluate_step to find the method.

-------------
### Consider the paper:
1. Neller. "An introduction to counterfactual regret minimalization." http://modelai.gettysburg.edu/2013/cfr/cfr.pdf
2. Deep Stack Poker. https://static1.squarespace.com/static/58a75073e6f2e1c1d5b36630/t/58b7a3dce3df28761dd25e54/1488430045412/DeepStack.pdf
3. Deep Stack Poker Supplementary. https://static1.squarespace.com/static/58a75073e6f2e1c1d5b36630/t/58bed28de3df287015e43277/1488900766618/DeepStackSupplement.pdf
-----------------
## Author: Alex Nguyen
## Gettysburg College