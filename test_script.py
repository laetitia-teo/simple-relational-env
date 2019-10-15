import numpy as np

from env import *
from gen import SimpleTaskGen
from dataset import ObjectDataset

### Testing environment

# theta = np.pi / 4
# # theta = 0

# env = Env(16, 20)
# env.random_config(3)

# state = env.to_state_list()
# play = Playground(16, 20, state)
# play.interactive_run()

# env.reset()

### Testing dataset generation

# gen = SimpleTaskGen(env, 3)
# gen.generate_mix(10, 30, 30)
# gen.save('data/simple_task/data.txt', 'images/simple_task/')

# env2 = Env(16, 20)
# gen2 = SimpleTaskGen(env2, 3)
# gen2.load('data/simple_task/data.txt')
# gen2.save('data/simple_task/data2.txt')

### Testing dataset

ds = ObjectDataset('data/simple_task/data.txt')