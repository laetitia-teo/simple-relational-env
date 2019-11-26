"""
This file defines the validation/optimisation experiment proposed by PY.

This is a validation and illustration experiment, once we have a trained 
reward function. We use this reward function as an objective to optimize
for a (non-sequential) decision-making process.

We have a series of objects on the bottom of the image, below a certain
threshold. The idea would be to predict the actions to apply on these objects
(no collision, no physics -> no constraints on the order in which the decisions 
are made, no heavy RL machinery), to have them in the same configuration as the
given query. This could be in the form of a GNN that predicts a movement for
each of the sapes, for instance. We would then optimize the GNN to maximize the
pre-trained reward function, by gradient descent.
"""
import torch