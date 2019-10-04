import numpy as np

from env import *
from gen import SimpleTaskGen

theta = np.pi / 4
# theta = 0

env = Env(16, 20)
s = Square(2., np.array([255., 255., 255.]), np.array([2., 5.]), theta)
t = Triangle(2., np.array([155., 255., 155.]), np.array([10., 5.]), - theta)
c = Circle(0.5, np.array([055., 055., 155.]), np.array([3., 10.]), theta)
env.add_object(s)
print('added square')
env.add_object(t)
print('added triangle')
env.add_object(c)
print('added circle')

state = env.to_state_list()
play = Playground(16, 20, state)
play.interactive_run()

# env.reset()

# # Testing dataset generation

# gen = SimpleTaskGen(env, 3)
# gen.generate(30, 10)
# gen.save(
#     'data/simple_task/data.txt',
#     img_path='images/simple_task')

# env2 = Env(16, 20)
# gen2 = SimpleTaskGen(env2, 3)
# gen2.load('data/simple_task/data.txt')
# gen2.save('data/simple_task/data2.txt')