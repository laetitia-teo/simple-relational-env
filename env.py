import os.path as op
import copy

import numpy as np
import cv2

# for interactive testing :
import pygame

SHAPE_NUMBER = 3

class CollisionError(Exception):
    """
    Raised when an action is performed that collides two shapes in the space.
    """
    def __init__(self, message):
        self.message = message

class AbstractEnv():

    def __init__(self):
        self.space = None # define this
        self.objects = []
        self.actions = [] # or find another way to represent actions

    def add_object(self, obj_dict):
        raise NotImplementedError

    def remove_object(self, obj_dict):
        raise NotImplementedError

    def get_objects(self):
        raise NotImplementedError

    def act(self, action, *args):
        raise NotImplementedError

class Shape():
    """
    Abstract implementation of an object and its methods.
    """

    def __init__(self, size, color, pos, ori):
        """
        An abstract shape representation. A shape has the following attributes:

            - size (float) : the radius of the shape. The radius is defined as
                the distance between the center of the shape and the point
                farthest from the shape. This is used to define a 
                bounding-circle centered on the position of the object.

            - color (RGB array) : the color of the shape.

            - pos (array of size 2) : the absolute poition of the object. 

            - ori (float) : the orientation of the object, in radians.
        """
        self.size = size
        self.color = color
        self.pos = pos
        self.ori = ori

        # concrete atributes
        self.cond = NotImplemented
        self.shape_index = NotImplemented

    def collides(self, object):
        raise NotImplementedError

    def collides_with_objects(self, list_of_objects):
        raise NotImplementedError

    def to_pixels(self, gridsize):
        """
        Returns a two dimensional array of 4D vectors (RGBa), of size the 
        object size times the grid size, in which the shape is encoded into
        pixels.

        For rendering purposes, and the pixel representation can also be used
        detect collisions.

        Arguments : 
            - gridsize (int) : the number of pixels in a unit.
        """
        size = int(self.size * gridsize)
        x, y = np.meshgrid(np.arange(2*size), np.arange(2*size))
        x = (x - size) / size
        y = (y - size) / size
        void = np.zeros(4)
        color = np.concatenate((self.color, [1.]))
        x = np.expand_dims(x, -1)
        y = np.expand_dims(y, -1)
        bbox = np.where(self.cond(x, y), color, void)
        return bbox

    def to_vector(self):
        """
        Returns an encoding of the object.
        """
        vec = np.zeros(SHAPE_NUMBER, dtype=float)
        vec[self.shape_index] = 1.
        return np.concatenate(
            (vec,
            np.array([self.size]),
            np.array(self.color),
            np.array(self.pos),
            np.array([self.ori])), 0)

class Square(Shape):

    def __init__(self, size, color, pos, ori):
        super(Square, self).__init__(
            size,
            color,
            pos,
            ori)
        self.shape_index = 0
        self.cond = self.cond_fn

    def cond_fn(self, x, y):
        theta = self.ori
        x_ = x * np.cos(theta) - y * np.sin(theta)
        y_ = x * np.sin(theta) + y * np.cos(theta)
        c =  np.less_equal(
            np.maximum(abs(x_), abs(y_)),
            1/np.sqrt(2))
        return c

class Circle(Shape):

    def __init__(self, size, color, pos, ori):
        super(Circle, self).__init__(
            size,
            color,
            pos,
            ori)
        self.shape_index = 1
        self.cond = lambda x, y : np.less_equal(x**2 + y**2, 1)

class Triangle(Shape):

    def __init__(self, size, color, pos, ori):
        super(Triangle, self).__init__(
            size,
            color,
            pos,
            ori)
        self.shape_index = 2
        self.cond = self.cond_fn

    def cond_fn(self, x, y):
        theta = self.ori
        x_ = x * np.cos(theta) - y * np.sin(theta)
        y_ = x * np.sin(theta) + y * np.cos(theta)
        a = np.sqrt(3)
        b = 1.
        c = np.greater_equal(y_, -1/2) * \
            np.less_equal(y_, a*x_ + b) * \
            np.less_equal(y_, (- a)*x_ + b)
        return c

def shape_from_vector(vec):
    """
    Takes in a vector encoding the shape and returns the corresponding Shape
    object.
    """
    shape = vec[0:SHAPE_NUMBER]
    size = vec[SHAPE_NUMBER]
    color = vec[SHAPE_NUMBER+1:SHAPE_NUMBER+4]
    pos = vec[SHAPE_NUMBER+4:SHAPE_NUMBER+6]
    ori = vec[SHAPE_NUMBER+6]
    if shape[0]:
        return Square(size, color, pos, ori)
    if shape[1]:
            return Circle(size, color, pos, ori)
    if shape[2]:
        return Triangle(size, color, pos, ori)


class Env(AbstractEnv):
    """
    Class for the implementation of the environment.
    """
    def __init__(self, gridsize, envsize):
        super(Env, self).__init__()
        self.gridsize = gridsize
        self.envsize = envsize
        self.L = int(envsize * gridsize)

        # matrix where all the rendering thakes place
        self.mat = np.zeros((self.L, self.L, 4))

    def l2_norm(self, pos1, pos2):
        """
        Euclidiean norm between pos1 and pos2.
        """
        x1, y1 = pos1
        x2, y2 = pos2
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def add_object(self, obj, idx=None):
        """
        Adds a Shape to the scene.

        Arguments :
            - obj (Shape) : the object to add
            - idx (int) : the index where to insert the object. This is used to
                keep the order of objects in actions.

        Raises ValueError if the given object collides with any other object
        or if it exceeds the environment range.
        """
        mat = np.zeros((self.L, self.L, 4))
        obj_mat = obj.to_pixels(self.gridsize)
        obj_mat = obj_mat[..., :] * np.expand_dims(obj_mat[..., 3], -1)
        s = len(obj_mat)
        ox, oy = ((self.gridsize * obj.pos) - int(s/2)).astype(int)
        # first perform check on env boundaries
        if (ox < 0 or oy < 0 or ox + s > self.L or oy + s > self.L):
            raise CollisionError('New shape out of environment range')
        mat[ox:ox + s, oy:oy + s] += obj_mat
        # then collision checks with environment and shape masks
        for obj2 in self.objects:
            if self.l2_norm(obj.pos, obj2.pos) <= obj.size + obj2.size:
                obj2_mat = obj2.to_pixels(self.gridsize)
                s2 = len(obj2_mat)
                ox2, oy2 = ((self.gridsize * obj2.pos) - int(s2/2)).astype(int)
                env_mask = mat[ox2:ox2+s2, oy2:oy2+s2, -1]
                obj2_mask = obj2_mat[:, :, -1]
                collides = np.logical_and(obj2_mask, env_mask)
                if collides.any():
                    raise CollisionError(
                        'New shape collides with existing shape')
        if idx is not None:
            self.objects.insert(idx, obj)
        else:
            self.objects.append(obj)

    def act(self, i_obj, a_vec):
        """
        Performs the action encoded by the vector a_vec on object indexed by
        i_obj.
        """
        obj = self.objects.pop(i_obj)
        o_vec = obj.to_vector()
        o_vec[SHAPE_NUMBER:] += a_vec
        obj2 = shape_from_vector(o_vec)
        try:
            self.add_object(obj2, i_obj)
        except CollisionError:
            print('Collision : invalid action')
            self.add_object(obj, i_obj)
        
    def render(self, show=True):
        """
        Renders the environment, returns a rasterized image as a numpy array.
        """
        mat = np.zeros((self.L, self.L, 4))
        for obj in self.objects:
            obj_mat = obj.to_pixels(self.gridsize)
            s = len(obj_mat)
            ox, oy = ((self.gridsize * obj.pos) - int(s/2)).astype(int)
            obj_mat = obj_mat[..., :] * np.expand_dims(obj_mat[..., 3], -1)
            mat[ox:ox + s, oy:oy + s] += obj_mat
        mat = np.flip(mat, axis=0)
        if show:
            pass
            # opencv stuff : windowing and stuff
            # or maybe use pyglet
        return mat

    def to_state_list(self):
        """
        Returns a list of all the objects in vector form.
        """
        return [obj.to_vector() for obj in self.objects]

    def from_state_list(self, state_list):
        """
        Adds the objects listed as vectors in state_list.

        Raises ValueError if objects are out of environment range or overlap 
        with other objects.
        """
        for vec in state_list:
            shape = shape_from_vector(vec)
            self.add_object(shape)

    def save(self, path, save_image=True, save_state=False):
        """
        Saves the current env image and the state description into the
        specified path.
        """
        if save_image:
            cv2.imwrite(path, self.render())
        if save_state:
            pass

class NActionSpace():
    """
    This class defines an action space for an environment with a variable
    number of objects, with objects that can enter or leave the environment
    during a single run. 

    The class provides a certain number of utilities, such as random sampling
    of an action, adding or removing objects, and the like.

    TODO : implements this a bit more seriously
    """
    def __init__(self, n_objects, n_args):
        """
        Init.

        Arguments :
            - n_objects : number of objects
            - args : action space for each of the objects
        """
        self.n_objects = n_objects
        self.n_args = n_args

    def sample(self):
        """
        Samples random action.
        """
        return (np.random.randint(self.n_objects), \
            np.random.randint(self.n_args))

class Playground():

    def __init__(self, gridsize, envsize, state):
        """
        Environment wrapper for use in a RL setting.
        """
        self._env = Env(gridsize, envsize)
        self._state = state
        self.reset()
        self.action_space = NActionSpace(len(self._env), 4)

    def reset(self):
        """
        Resets the environment to the state it was initialized with.
        """
        self._env.objects = []
        self._env.from_state_list(self._state)

    def move_shape(self, i_obj, direction):
        """
        Shape-moving api. Object is moved according to a fixed distance in one
        of the four cardinal directions.

        Arguments:
            - i_obj (int): index of the object to move.
            - dir (int between 0 and 4): direction to move.
        """
        a_vec = np.zeros(7)
        step_size = 0.5
        if direction == 0:
            a_vec[4] += step_size
        if direction == 1:
            a_vec[4] -= step_size
        if direction == 2:
            a_vec[5] += step_size
        if direction == 3:
            a_vec[5] -= step_size
        self._env.act(i_obj, a_vec)

    def render(self):
        return self._env.render()

    def step(self, action):
        i_obj, direction = action
        # no reward and no objective in this environment for now, we only
        # give the state as feedback.
        return self._env.to_state_list()

    def interactive_run(self, reset=True):
        """
        Launches a little terminal-based run of the game.
        """
        if reset:
            self.reset()
        pygame.init()
        done = False
        X = self._env.L
        Y = self._env.L
        framename = 'images/frame.jpg'
        self._env.save(framename)
        display = pygame.display.set_mode((X, Y))
        pygame.display.set_caption('Playground')
        frame = 0
        while not done:
            try:
                display.fill((0, 0, 0))
                display.blit(pygame.image.load(framename), (0, 0))
                pygame.display.update()
                i_obj = int(input('what shape to move ?'))
                if i_obj >= len(self._env.objects) or i_obj < 0:
                    raise ValueError('Invalid object index')
                direction = int(input('what direction ?'))
                if direction > 3 or direction < 0:
                    raise ValueError('Invalid direction')
                self.move_shape(i_obj, direction)
                self._env.save(framename)
                frame += 1
            except ValueError:
                done = True
                pygame.quit()