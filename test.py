from env import Env, Collision

env = Env(16, 20)
env.random_config(4)
env.save_image('image.jpg')
center, phi = env.random_rotation()
env.rotate(phi, center)
env.save_image('image2.jpg')

a = input()
if a == 'a':
    colls = []
    phis = []
    for _ in range(1000):
        env.reset()
        env.random_config(4)
        collision_counter = 0
        for _ in range(20):
            center, phi = env.random_rotation()
            try:
                env.rotate(phi, center, True)
                phis.append(phi)
            except Collision:
                collision_counter += 1
        colls.append(collision_counter)
