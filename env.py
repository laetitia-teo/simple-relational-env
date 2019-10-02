import os.path as op

import numpy as np
import cv2

SHAPE_NUMBER = 3

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
        An abstract shape representation. A shape has the following attributes :

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
            vec,
            np.array(color),
            np.array(pos),
            np.array(ori), 0)

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
        x_ = x * np.cos(theta) + y * np.sin(theta)
        y_ = - x * np.sin(theta) + y * np.cos(theta)
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
        x_ = x * np.cos(theta) + y * np.sin(theta)
        y_ = - x * np.sin(theta) + y * np.cos(theta)
        a = np.sqrt(3)
        b = 1.
        c = np.greater_equal(y_, -1/2) * \
            np.less_equal(y_, a*x_ + b) * \
            np.less_equal(y_, (- a)*x_ + b)
        return c


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

    def add_object(self, obj):
        """
        Adds an Shape to the scene.

        Arguments :
            - obj (Shape) : the object to add

        Raises ValueError if the given object collides with any other object
        or if it exceeds the environment range.
        """
        # first perform check on env boundaries
        obj_mat = obj.to_pixels(self.gridsize)
        s = len(obj_mat)
        ox, oy = ((self.gridsize * obj.pos) - int(s/2)).astype(int)
        if (ox < 0 or oy < 0 or ox + s > self.L or oy + s > self.L):
            raise ValueError('New shape out of environment range')
        # then collision checks with environment and shape masks
        env_mask = self.mat[ox:ox+s, oy:oy+s, -1]
        obj_mask = obj_mat[:, :, -1]
        collides = np.logical_and(obj_mask, env_mask)
        if collides.any():
            raise ValueError('New shape collides with existing shapes')
        obj_mat = obj_mat[..., :] * np.expand_dims(obj_mat[..., 3], -1) # transparency
        self.mat[ox:ox + s, oy:oy + s] += obj_mat
        
    def render(self):
        """
        Renders the environment, returns a rasterized image as a numpy array.
        """
        pass

    def save(self, path, save_image=True, save_state=False):
        """
        Saves the current env image and the state description into the specified path.
        """
        if save_image:
            cv2.imwrite(path, self.mat)
        if save_state:
            pass