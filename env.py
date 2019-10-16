import os.path as op
import copy
import time

import numpy as np
import cv2

# for interactive testing :
import pygame

from random import shuffle

N_SH = 3 # number of shapes

class Collision(Exception):
    """
    Raised when an action is performed that collides two shapes in the space.
    """
    def __init__(self, message):
        self.message = message

class SamplingTimeout(Exception):
    """
    Raised when a sampling process times out, possibly because of a too high
    rejection rate.
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
        vec = np.zeros(N_SH, dtype=float)
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

    def copy(self):
        """
        Returns a Square with the same attributes as the current one.
        """
        size = self.size
        color = np.array(self.color)
        pos = np.array(self.pos)
        ori = self.ori
        return Square(size, color, pos, ori)

class Circle(Shape):

    def __init__(self, size, color, pos, ori):
        super(Circle, self).__init__(
            size,
            color,
            pos,
            ori)
        self.shape_index = 1
        self.cond = lambda x, y : np.less_equal(x**2 + y**2, 1)

    def copy(self):
        """
        Returns a Circle with the same attributes as the current one.
        """
        size = self.size
        color = np.array(self.color)
        pos = np.array(self.pos)
        ori = self.ori
        return Circle(size, color, pos, ori)

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

    def copy(self):
        """
        Returns a Triangle with the same attributes as the current one.
        """
        size = self.size
        color = np.array(self.color)
        pos = np.array(self.pos)
        ori = self.ori
        return Triangle(size, color, pos, ori)

def shape_from_vector(vec):
    """
    Takes in a vector encoding the shape and returns the corresponding Shape
    object.
    """
    shape = vec[0:N_SH]
    size = vec[N_SH]
    color = vec[N_SH+1:N_SH+4]
    pos = vec[N_SH+4:N_SH+6]
    ori = vec[N_SH+6]
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
        """
        Arguments :

            - gridsize : size of a unit in pixels.
            - envsize : size of the environment in units.
        """
        super(Env, self).__init__()
        self.gridsize = gridsize
        self.envsize = envsize
        self.L = int(envsize * gridsize)

        # matrix where all the rendering takes place
        self.mat = np.zeros((self.L, self.L, 4))

    def reset(self):
        self.objects = []

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
            raise Collision('New shape out of environment range')
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
                    raise Collision(
                        'New shape collides with existing shape')
        if idx is not None:
            self.objects.insert(idx, obj)
        else:
            self.objects.append(obj)

    def bounding_box(self):
        """
        Computes a bounding box for the set of objects currently in the 
        environnment, in real coordinates.

        The Bounding box is not strict, since it does not use pixel rendering
        of the shapes, but only individual shape rendering boxes.
        
        If no objects are present, returns (None, None), else returns the bottom
        left corner of the bbox and the upper right corner.
        """
        if not self.objects:
            return None, None
        aplus = np.array([obj.pos + obj.size for obj in self.objects])
        amin = np.array([obj.pos - obj.size for obj in self.objects])
        maxpos = np.max(aplus, 0)
        minpos = np.min(amin, 0)
        return minpos, maxpos

    def translate(self, amount, raise_collision=False):
        """
        Translates all the objects in the scene by amount, if there is no
        collision with the edge of the environment.

        Arguments :
            - amount : 2d array of floats
            - raise_collision (bool) : whether or not to raise a Collision
                exception when the translation fails
        """
        state_list = self.to_state_list()
        self.reset()
        tr_state_list = []
        for vec in state_list:
            tr_vec = np.array(vec)
            tr_vec[N_SH+4:N_SH+6] += amount
            tr_state_list.append(tr_vec)
        try:
            self.from_state_list(tr_state_list)
        except Collision:
            print('Collision : invalid translation')
            self.reset()
            self.from_state_list(state_list)
            if raise_collision:
                raise Collision('Failed translation')

    def scale(self, amount, center=None, raise_collision=False):
        """
        Scales all the scene by amount. If no center is given, the scene center
        is used.

        Arguments :
            - amount : float
        """
        if center is None:
            center = np.array([self.envsize/2, self.envsize/2])
        state_list = self.to_state_list()
        self.reset()
        sc_state_list = []
        for vec in state_list:
            sc_vec = np.array(vec)
            sc_vec[N_SH+4:N_SH+6] = amount * (sc_vec[N_SH+4:N_SH+6] - center) + center 
            sc_vec[N_SH] *= amount
            sc_state_list.append(sc_vec)
        try:
            self.from_state_list(sc_state_list)
        except Collision:
            print('Collision : invalid scaling')
            self.reset()
            self.from_state_list(state_list)
            if raise_collision:
                raise Collision('Failed scaling')

    def shuffle_objects(self):
        """
        Shuffles objects in the state list (the states are unchanged)
        This is for testing the models' robustness to permutation.
        """
        shuffle(self.objects)

    def act(self, i_obj, a_vec):
        """
        Performs the action encoded by the vector a_vec on object indexed by
        i_obj.

        If the action is invalid (Collision), the state is left unchanged.
        """
        obj = self.objects.pop(i_obj)
        o_vec = obj.to_vector()
        o_vec[N_SH:] += a_vec
        obj2 = shape_from_vector(o_vec)
        try:
            self.add_object(obj2, i_obj)
        except Collision:
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

    def from_state_list(self, state_list, reset=True):
        """
        Adds the objects listed as vectors in state.

        Raises Collision if objects are out of environment range or
        overlap with other objects.
        """
        if reset:
            self.reset()
        for vec in state_list:
            shape = shape_from_vector(vec)
            self.add_object(shape)

    def save_image(self, path):
        """
        Saves the current env image and the state description into the
        specified path.
        """
        cv2.imwrite(path, self.render())

    def random_mix(self, timeout=20):
        """
        Creates a scene configuration where the objects are the same, but the
        spatial configuration is randomly re-sampled.
        """
        count = 0
        while count < timeout:
            new_objects = []
            for obj in self.objects:
                new_pos = np.random.random(2)
                new_pos = (1 - new_pos) * obj.size + new_pos \
                    * (self.envsize - obj.size)
                new_obj = obj.copy()
                new_obj.pos = new_pos
                new_objects.append(new_obj)
            try:
                objects = self.objects
                self.reset()
                for obj in new_objects:
                    self.add_object(obj)
                return
            except Collision:
                self.objects = objects
            count += 1
        raise SamplingTimeout('Too many rejected samplings, try fewer objects')

    def add_random_object(self, timeout=20):
        """
        Adds a random object, with collision handling.

        The sampling algorithm is quite basic : uniformly sample the shape
        type, the shape color, the orientation and the shape size. Then (using
        size information), sample the position uniformly. If there is a
        collision with an aready-existing shape, resample. We allow up to
        timeout resamplings, after which, if all sampled positions were
        rejected, we throw an exception (the environment is probably too full
        by this point).

        Raises SamplingTimeout if more than timeout position samplings have
        given rise to an error.
        """
        count = 0
        minsize = 0.5
        maxsize = 2
        while count < timeout:
            shape = np.random.randint(N_SH)
            color = np.random.random(3) # U(0, 1) in 3d
            color = (255 * color).astype(int)
            size = np.random.random()
            size = (1 - size) * minsize + size * maxsize
            ori = np.random.random()
            ori = ori * 2 * np.pi # we allow up to 2pi rotations
            pos = np.random.random(2)
            pos = (1 - pos) * size + pos * (self.envsize - size)
            if shape == 0:
                obj = Square(size, color, pos, ori)
            elif shape == 1:
                obj = Circle(size, color, pos, ori)
            elif shape == 2:
                obj = Triangle(size, color, pos, ori)
            try:
                self.add_object(obj)
                return
            except Collision:
                pass # re-sample
            count += 1
        raise SamplingTimeout('Too many rejected samplings, check if the ' \
            + 'environment is not too full')

    def random_config(self, n_objects, timeout=20):
        """
        Returns a random configuration of the environment.
        Doesn't reset the environment to zero, this should be done manually
        if desired.

        Raises SamplingTimeout if we reject more than timeout position
        samplings on one of the random object generations.
        """
        for _ in range(n_objects):
            self.add_random_object(timeout)

    def random_translation_vector(self):
        """
        Performs a random translation, with the translation vector sampled
        uniformly in the space available as computed with the difference 
        between the environment size and the current object bounding box.
        """
        bboxmin, bboxmax = self.bounding_box()
        if bboxmin is None:
            return
        spacemin = np.array([0., 0.])
        spacemax = np.array([self.envsize, self.envsize])
        plusspace = spacemax - bboxmax
        minspace = spacemin - bboxmin
        # whether to sample in plusspace or minspace
        b = np.random.binomial(1, plusspace/(plusspace - minspace))
        # how much to move in x and y
        u = np.random.random(2)
        umax = plusspace * u
        umin = minspace * u
        U = np.array([umin, umax])
        tvec = U[b, np.arange(2)]
        return tvec

    def random_transformation(self, timeout=20):
        """
        Applies a random transformation on the state.

        This transformation can be a translation or a scaling of the current
        scene.

        Raises SamplingTimeout if more than ::timeout:: position samplings have
        been rejected.
        """
        count = 0
        while count < timeout:
            try:
                trans = np.random.randint(2)
                if trans == 0:
                    # maximum = self.envsize / 10
                    # minimum = - self.envsize / 10
                    # amount = np.random.random(2)
                    # amount = amount * (maximum - minimum) + minimum
                    amount = self.random_translation_vector()
                    self.translate(amount, raise_collision=True)
                    return
                elif trans == 1:
                    maximum = 2
                    minimum = 0.5
                    amount = np.random.random()
                    amount = amount * (maximum - minimum) + minimum
                    self.scale(amount, raise_collision=True)
                    return
            except Collision:
                pass # re-sample
            count += 1
        raise SamplingTimeout('Too many rejected samplings, check if the ' \
            + 'environment is not too full')


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

        This wrapper provides additionnal functionnalities allowing an agent
        or human to interact with the environment.

        1) Action discretization : in the original Env class, actions on an
        object are generic and modify an arbitrary number of an object's
        properties by an arbitrary float amount. Here the actions are 
        provided in a more structured way, such as movin a shape up, changing
        its size etc.

        2) Action-space description : the action_space attribute should give
        every information about the current implementation's action space. 

        3) RL-friendly api : the step() function is intended for use with an 
        RL algorithm.

        4) Interactive runs : one can run the environment in an interactive way
        as a game.
        """
        self._env = Env(gridsize, envsize)
        self._state = state
        self.reset()
        self.action_space = NActionSpace(len(self._env.objects), 4)

    def reset(self):
        """
        Resets the environment to the state it was initialized with.
        """
        self._env.reset()
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
        self._env.save_image(framename)
        display = pygame.display.set_mode((X, Y))
        pygame.display.set_caption('Playground')
        idx = 0
        while not done:
            display.fill((0, 0, 0))
            display.blit(pygame.image.load(framename), (0, 0))
            pygame.display.update()
            # i_obj = int(input('what shape to move ?'))
            # if i_obj >= len(self._env.objects) or i_obj < 0:
            #     raise ValueError('Invalid object index')
            # direction = int(input('what direction ?'))
            # if direction > 3 or direction < 0:
            #     raise ValueError('Invalid direction')
            # self.move_shape(i_obj, direction)
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.move_shape(idx, 3)
                    if event.key == pygame.K_RIGHT:
                        self.move_shape(idx, 2)
                    if event.key == pygame.K_UP:
                        self.move_shape(idx, 0)
                    if event.key == pygame.K_DOWN:
                        self.move_shape(idx, 1)
                    if event.key == pygame.K_SPACE:
                        idx = (idx + 1) % len(self._env.objects)
                        print(idx)
                    if event.key == pygame.K_ESCAPE:
                        done = True
            self._env.save_image(framename)
        pygame.quit()
            