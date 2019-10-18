import numpy as np
import torch

import baseline_models as bm
import graph_models as gm

from torch.utils.data import DataLoader

from env import *
from gen import SimpleTaskGen
from dataset import ObjectDataset

B_SIZE = 32

### Testing environment

# theta = np.pi / 4
# # theta = 0

env = Env(16, 20)
env.random_config(3)

# state = env.to_state_list()
# play = Playground(16, 20, state)
# play.interactive_run()

env.reset()

### Testing dataset generation

# gen = SimpleTaskGen(env, 3)
# gen.generate_mix(2, 5, 100)
# gen.save('data/simple_task/test.txt', 'images/simple_task/test')

# env2 = Env(16, 20)
# gen2 = SimpleTaskGen(env2, 3)
# gen2.load('data/simple_task/data.txt')
# gen2.save('data/simple_task/data2.txt')

### Testing dataset

ds = ObjectDataset('data/simple_task/test.txt', epsilon=1/10)
ds.process()
dl = DataLoader(ds, batch_size=B_SIZE, shuffle=True)

a = iter(dl)
t, i = next(a)

### Testing baselines

# naive_model = bm.NaiveMLP(3, 10, [32, 32])
# scene_model = bm.SceneMLP(3, 10, [16, 16], 16, [16, 16])

# objs = torch.reshape(t, (B_SIZE, 60))
# print(objs)
# res1 = naive_model(objs)
# print('res1 %s' % res1)

# objs = torch.reshape(t, (B_SIZE, 2, 30))
# obj1s, obj2s = objs[:, 0, :], objs[:, 1, :]
# res2 = scene_model(obj1s, obj2s)
# print('res2 %s ' % res2)

# ### Testing graph models

# f_e = 5
# f_u = 5
# f_x = 10
# graph1, graph2 = gm.tensor_to_graphs(t, 3, 5, 5)
# f_dict = {'f_e': f_e, 'f_u': f_u, 'f_x': f_x, 'f_out': 2}

# gembed = gm.GraphEmbedding([8, 8], 8, 10, f_dict)
# res3 = gembed(graph1, graph2)
# print('res3 %s' % (res3))

# gdiff = gm.GraphDifference([8, 8], 8, 10, f_dict, gm.identity_mapping)
# res4 = gdiff(graph1, graph2)
# print('res4 %s' % res4)

# galt = gm.Alternating([8, 8], 8, 10, f_dict)
# res5 = galt(graph1, graph2)
# preint('res5 %s' % res5)


### Testing batching for graph computations