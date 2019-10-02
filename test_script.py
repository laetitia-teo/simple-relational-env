import numpy as np

from env import *

theta = np.pi / 6

env = Env(16, 20)
s = Square(1., np.array([255., 255., 255.]), np.array([2., 5.]), theta)
t = Triangle(2., np.array([155., 255., 155.]), np.array([10., 5.]), - theta)
c = Circle(0.5, np.array([055., 055., 155.]), np.array([3., 10.]), theta)
env.add_object(s)
print('added square')
env.add_object(t)
print('added triangle')
env.add_object(c)
print('added circle')
