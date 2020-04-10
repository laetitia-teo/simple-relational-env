import os.path as op
import copy
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt

# for interactive testing :
import pygame

from random import shuffle

N_SH = 3 # number of shapes

class SamplingTimeout(Exception):
    def __init__(self, message):
        super(SamplingTimeout, self).__init__()
        print(message)

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

    def to_vector(self, norm=False):
        """
        Returns an encoding of the object.
        """
        vec = np.zeros(N_SH, dtype=float)
        vec[self.shape_index] = 1.
        if norm:
            color = self.color / 255
        else:
            color = self.color
        return np.concatenate(
            (vec,
            np.array([self.size]),
            np.array(color),
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

def shape_from_vector(vec, norm=False):
    """
    Takes in a vector encoding the shape and returns the corresponding Shape
    object.

    norm : whether or not to denormalize to the [0:255] integer range. 
    """
    shape = vec[0:N_SH]
    size = vec[N_SH]
    color = vec[N_SH+1:N_SH+4]
    if norm:
        color = (color * 255).astype(int)
    pos = vec[N_SH+4:N_SH+6]
    ori = vec[N_SH+6]
    if shape[0]:
        return Square(size, color, pos, ori)
    if shape[1]:
        return Circle(size, color, pos, ori)
    if shape[2]:
        return Triangle(size, color, pos, ori)

def overlay(mat1, mat2):
    """
    Overalays mat2 (last channel of last dimension is considered alpha channel)
    over mat1.
    Retruns the resulting matrix, with no alpha channel.
    """
    alphas = np.expand_dims(mat2[..., -1], -1)
    print(mat1.shape)
    print(alphas.shape)
    print(mat2.shape)
    return mat1 * (1 - alphas) \
        + mat2[..., :-1] * alphas

class Env(AbstractEnv):
    """
    Class for the implementation of the environment.
    """
    def __init__(self, gridsize=16, envsize=20):
        """
        Arguments :

            - gridsize : size of a unit in pixels.
            - envsize : size of the environment in units.

        Keep default sizes.
        """
        super(Env, self).__init__()
        self.gridsize = gridsize
        self.envsize = envsize
        self.L = int(envsize * gridsize)
        self.N_SH = N_SH

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

    def get_obj_dim(self):
        """
        Returns the number of features of an object in this envionment.
        """
        if not self.objects:
            self.add_random_object()
            f_x = len(self.to_state_list()[0])
            self.reset()
            return f_x
        f_x = len(self.to_state_list()[0])
        return f_x

    def add_object(self, obj, idx=None):
        """
        Adds a Shape to the scene.
        """
        if idx is not None:
            self.objects.insert(idx, obj)
        else:
            self.objects.append(obj)

    def add_object_from_specs(self, shape, size, color, pos, ori):
        """
        Raises Collision if specs correspond to an object that collides with 
        previous objects/outside of env.
        """
        if shape == 0:
            obj = Square(size, color, pos, ori)
        elif shape == 1:
            obj = Circle(size, color, pos, ori)
        elif shape == 2:
            obj = Triangle(size, color, pos, ori)
        self.add_object(obj)

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

    # def pbbox(self):
    #     """
    #     Computes bounding box of objects in pixel coordinates.
    #     """
    #     if not self.objects:
    #         return None, None
    #     obj_mat = obj.to_pixels(self.gridsize)
    #     s = len(obj_mat)
    #     ox, oy = ((self.gridsize * obj.pos) - int(s/2)).astype(int)
    #     aplus.append()

    def get_center(self):
        """
        Returns the center of the configuration in 2d space.
        """
        if not self.objects:
            return np.array([self.envsize, self.envsize])
        s = np.zeros(2)
        for obj in self.objects:
            s += obj.pos
        s /= len(self.objects)
        return s

    def translate(self, amount):
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
        self.from_state_list(tr_state_list)

    def scale(self, amount, center=None):
        """
        Scales all the scene by amount. If no center is given, the scene center
        is used.

        Arguments :
            - amount : float, the scale of the scaling.
        """
        if center is None:
            center = self.get_center()
        state_list = self.to_state_list()
        self.reset()
        sc_state_list = []
        for vec in state_list:
            sc_vec = np.array(vec)
            sc_vec[N_SH+4:N_SH+6] = amount * (sc_vec[N_SH+4:N_SH+6] - center) + center 
            sc_vec[N_SH] *= amount
            sc_state_list.append(sc_vec)
        self.from_state_list(sc_state_list)

    def rotate(self, theta, center=None):
        """
        Applies a rotation of angle theta on a scene. If no center is given,
        the scene center is used.

        Note, the rotation is performed clockwise.

        Arguments :
            - theta (float): angle of rotation, in radians
            - center (size 2 array): position of the rotation center. If None
                is given, the center is the center of the environment.
            - raise_collision (bool): whether to propagate the Collision
                exception if it happens.
        """
        if center is None:
            center = self.get_center()
        state_list = self.to_state_list()
        self.reset()
        rot_state_list = []
        for vec in state_list:
            rot_vec = np.array(vec)
            pos = rot_vec[N_SH+4:N_SH+6] - center
            rot_mx = np.array([[np.cos(theta), - np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
            rot_vec[N_SH+4:N_SH+6] = rot_mx.dot(pos) + center
            rot_vec[N_SH+6] += theta
            rot_state_list.append(rot_vec)
        self.from_state_list(rot_state_list)

    def shuffle_objects(self):
        """
        Shuffles objects in the state list (the states are unchanged)
        This is for testing the models' robustness to permutation.
        """
        shuffle(self.objects)

    def change_shape(self, i_obj, shape_index):
        """
        Changes the shape of the object at index i_obj to the shape specified
        by shape_index.
        """
        obj = self.objects.pop(i_obj)
        o_vec = obj.to_vector()
        o_vec[:N_SH] = np.array(
            [0. if i != shape_index else 1. for i in range(N_SH)])
        obj2 = shape_from_vector(o_vec)
        self.add_object(obj2, i_obj)

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
        self.add_object(obj2, i_obj)
        
    def render(self, show=True, mode='fixed'):
        """
        Renders the environment, returns a rasterized image as a numpy array.

        There are two modes for rendering : 

        First mode ('fixed') : we render the scene in a fixed size image
        (3 times the original environment size to account for all the possible
        translations and scalings that may have sent our objects outside the
        range in which they were created), this mode allows us to take the
        translation into account, since the coordinate-pixel mapping stays
        constant in this rendering mode.

        The second mode ('bbox') renders the scene as given by the bounding-box of the
        objects : this allows us to see the scalings, rotations and non-linear
        transformations, but we lose representation of translation.
        """
        if mode == 'fixed':
            L = self.L * 3
            mat = np.zeros((L, L, 3))
            l = self.L
            for obj in self.objects:
                obj_mat = obj.to_pixels(self.gridsize)
                s = len(obj_mat) # size of object in pixel space
                ox, oy = ((self.gridsize * obj.pos) - int(s/2)).astype(int)
                obj_mat = obj_mat[..., :] * np.expand_dims(obj_mat[..., 3], -1)
                # indices
                xmin = max(l + ox, 0)
                xmax = max(l + ox + s, 0)
                ymin = max(l + oy, 0)
                ymax = max(l + oy + s, 0)
                xminobj = max(-(l + ox), 0)
                xmaxobj = max(L - (l + ox), 0)
                yminobj = max(-(l + oy), 0)
                ymaxobj = max(L - (l + oy), 0)
                mat[xmin:xmax, ymin:ymax] = overlay(
                    mat[xmin:xmax, ymin:ymax],
                    obj_mat[xminobj:xmaxobj, yminobj:ymaxobj])
        if mode == 'bbox':
            bboxmin, bboxmax = self.bounding_box()
            Lx = int((bboxmax[0] - bboxmin[0]) * self.gridsize) + 1
            Ly = int((bboxmax[1] - bboxmin[1]) * self.gridsize) + 1
            # origin
            lx = int(bboxmin[0] * self.gridsize)
            ly = int(bboxmin[1] * self.gridsize)
            print('Ly %s, Lx %s' % (Lx, Ly))
            mat = np.zeros((Lx, Ly, 3))
            for obj in self.objects:
                obj_mat = obj.to_pixels(self.gridsize)
                s = len(obj_mat)
                ox, oy = ((self.gridsize * obj.pos) - int(s/2)).astype(int)
                obj_mat = obj_mat[..., :] * np.expand_dims(obj_mat[..., 3], -1)
                xmin = ox - lx
                xmax = ox + s - lx
                ymin = oy - ly
                ymax = oy + s - ly
                mat[xmin:xmax, ymin:ymax] = overlay(
                    mat[xmin:xmax, ymin:ymax],
                    obj_mat)
        if mode == 'envsize':
            L = self.L
            mat = np.zeros((L, L, 3))
            l = self.L
            for obj in self.objects:
                obj_mat = obj.to_pixels(self.gridsize)
                s = len(obj_mat) # size of object in pixel space
                ox, oy = ((self.gridsize * obj.pos) - int(s/2)).astype(int)
                obj_mat = obj_mat[..., :] * np.expand_dims(obj_mat[..., 3], -1)
                # indices
                xmin = max(ox, 0)
                xmax = max(ox + s, 0)
                ymin = max(oy, 0)
                ymax = max(oy + s, 0)
                xminobj = max(-ox, 0)
                xmaxobj = max(L - ox, 0)
                yminobj = max(-oy, 0)
                ymaxobj = max(L - oy, 0)
                mat[xmin:xmax, ymin:ymax] = overlay(
                    mat[xmin:xmax, ymin:ymax],
                    obj_mat[xminobj:xmaxobj, yminobj:ymaxobj])
        mat = np.flip(mat, axis=0)
        mat = mat.astype(int)
        if show:
            plt.imshow(mat)
            plt.show()
        return mat

    def to_state_list(self, norm=False):
        """
        Returns a list of all the objects in vector form.

        norm (bool): whether or not to normalize color to the [0:1] range.
        """
        return [obj.to_vector(norm) for obj in self.objects]

    def from_state_list(self, state_list, reset=True, norm=False):
        """
        Adds the objects listed as vectors in state.

        Raises Collision if objects are out of environment range or
        overlap with other objects.
        """
        if reset:
            self.reset()
        for vec in state_list:
            shape = shape_from_vector(vec, norm=norm)
            self.add_object(shape)

    def save_image(self, path):
        """
        Saves the current env image and the state description into the
        specified path.
        """
        cv2.imwrite(path, self.render(False))

    def random_mix(self):
        """
        Creates a scene configuration where the objects are the same, but the
        spatial configuration is randomly re-sampled.
        """
        new_objects = []
        for obj in self.objects:
            new_pos = np.random.random(2)
            new_pos = (1 - new_pos) * obj.size + new_pos \
                * (self.envsize - obj.size)
            new_obj = obj.copy()
            new_obj.pos = new_pos
            new_objects.append(new_obj)
        objects = self.objects
        self.reset()
        for obj in new_objects:
            self.add_object(obj)
        return

    def add_random_object(self,
                          color=None,
                          shape=None):
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

        Arguments :
            - timeout : number of failed samplings before raising
                SamplingTimeout;
            - color : the color of the sampled object, color is drawn uniformly
                in rgb space if unspecified;
            - shape : the shape of the sampled object. All shapes are drawn
                with equal probability if unspcified.
        """
        count = 0
        # maybe change this
        minsize = self.envsize / 40
        maxsize = self.envsize / 10
        if shape is None:
            shape = np.random.randint(N_SH)
        if color is None:
            color = np.random.random(3)
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
        self.add_object(obj)

    def random_config(self, n_objects):
        """
        Returns a random configuration of the environment.
        Doesn't reset the environment to zero, this should be done manually
        if desired.

        Raises SamplingTimeout if we reject more than timeout position
        samplings on one of the random object generations.
        """
        for _ in range(n_objects):
            self.add_random_object()

    def random_translation_vector(self, Rmin=None, Rmax=None):
        """
        Samples a random translation vector, with norm between 0 and envsize.
        """
        if Rmin is None:
            Rmin = 0
        if Rmax is None:
            Rmax = self.envsize
        R = np.random.random()
        R = Rmin * (1 - R) + Rmax * R
        t = np.random.random() * 2 * np.pi
        tvec = np.array([R * np.cos(t), R * np.sin(t)])
        return tvec

    def random_translation_vector_cartesian(self, ex_range=None):
        """
        Samples a random translation vector, the max for a single coordinate is
        th environment size. Is ex_range is provided, specifies the range from
        which to exclude sampling (normalized to ((0, 0), (1, 1))).

        Independent version, where the excluded ranges are excluded on the x
        and y dimension separately.
        """
        minvec = - self.envsize
        maxvec = self.envsize
        if ex_range is not None:
            # sample x
            x = np.random.random()
            if ex_range[0][0] == minvec:
                if ex_range[0][1] == maxvec:
                    x = 0.
                else:
                    # extrapolation
                    x = ex_range[0][1] * (1 - x) + maxvec * x
            elif ex_range[0][1] == maxvec:
                # extrapolation
                x = minvec * (1 - x) + ex_range[0][0] * x
            else:
                # interpolation
                L = ex_range[0][1] - ex_range[0][0]
                maxvecmodx = maxvec - L
                x = minvec * (1 - x) + maxvecmodx * x
                l = ex_range[0][0]
                if x > l:
                    x += L
            # sample y
            y = np.random.random()
            if ex_range[1][0] == minvec:
                if ex_range[1][1] == maxvec:
                    y = 1
                else:
                    # extrapolation
                    y = ex_range[1][1] * (1 - y) + maxvec * y
            elif ex_range[1][1] == maxvec:
                # extrapolation
                y = minvec * (1 - y) + ex_range[1][0] * y
            else:
                # interpolation
                L = ex_range[1][1] - ex_range[1][0]
                maxvec -= L
                y = minvec * (1 - y) + maxvec * y
                l = ex_range[1][0]
                if y > l:
                    y += L
            vec = np.array([x, y])
        else:
            vec = np.random.random(2)
            vec *= self.envsize
        return vec

    def random_translation_vector_cartesian_v2(
        self,
        ex_range=None,
        count=0, 
        timeout=100):
        """
        Samples a random translation vector, the max for a single coordinate is
        th environment size. Is ex_range is provided, specifies the range from
        which to exclude sampling (normalized to ((0, 0), (1, 1))).

        Dependent version, where the range is excluded in the x and y
        dimensions at the same time.
        """
        if count > timeout:
            raise SamplingTimeout('Sampling timed out in translation vector'\
                + ' generation')
        minvec = - self.envsize
        maxvec = self.envsize
        if ex_range is not None:
            # sample x and y
            x = np.random.random()
            y = np.random.random()
            xtest = minvec * (1 - x) + maxvec * x
            ytest = minvec * (1 - y) + maxvec * y
            if ytest <= ex_range[1][0] or ytest >= ex_range[1][1]:
                x = xtest
                y = ytest
                vec = np.array([x, y])
            elif xtest <= ex_range[0][0] or xtest >= ex_range[0][1]:
                x = xtest
                y = ytest
                vec = np.array([x, y])
            else:
                vec = self.random_translation_vector_cartesian_v2(
                    ex_range,
                    count+1)
        else:
            vec = np.random.random(2)
            vec = (1 - vec) * minvec + vec * maxvec
        return vec

    def random_scaling(self, minscale=None, maxscale=None, ex_range=None):
        """
        Randomly cooses a center and a scale for a scaling transformation,
        between authorized bounds.
        """
        center = self.get_center()
        if minscale is None:
            minscale = 0.5
        if maxscale is None:
            maxscale = 2.
        scale = np.random.random()
        if ex_range is not None:
            if ex_range[0] == minscale:
                if ex_range[1] == maxscale:
                    scale = 1
                else:
                    # extrapolation
                    scale = ex_range[1] * (1 - scale) + maxscale * scale
            elif ex_range[1] == maxscale:
                # extrapolation
                scale = minscale * (1 - scale) + ex_range[0] * scale
            else:
                # interpolation
                L = ex_range[1] - ex_range[0]
                maxscale -= L
                scale = minscale * (1 - scale) + maxscale * scale
                l = ex_range[0]
                if scale > l:
                    scale += L
        else:
            scale = minscale * (1 - scale) + maxscale * scale
        return center, scale

    def random_rotation(self, phi0=np.pi/2, ex_range=None):
        """
        Samples a random rotation center and vector.
        """
        center = self.get_center()
        minphi = 0
        maxphi = 2 * np.pi
        phi = np.random.random()
        if ex_range is not None:
            if ex_range[0] == minphi:
                if ex_range[1] == maxphi:
                    phi = 0.
                else:
                    # extrapolation
                    phi = ex_range[1] * (1 - phi) + maxphi * phi
            elif ex_range[1] == maxphi:
                # extrapolation
                phi = minphi * (1 - phi) + ex_range[0] * phi
            else:
                # interpolation
                L = ex_range[1] - ex_range[0]
                maxphi -= L
                phi = minphi * (1 - phi) + maxphi * phi
                l = ex_range[0]
                if phi > l:
                    phi += L
        else:
            phi = minphi * (1 - phi) + maxphi * phi
        # phi = 2 * np.pi * phi
        return center, phi

    def random_transformation(self,
                              rotations=True,
                              s_ex_range=None,
                              t_ex_range=None,
                              r_ex_range=None):
        """
        Applies a random transformation on the state.

        This transformation can be a translation or a scaling of the current
        scene.
        """
        amount = self.random_translation_vector_cartesian_v2(
            ex_range=t_ex_range)
        self.translate(amount)
        center, scale = self.random_scaling(ex_range=s_ex_range)
        self.scale(scale, center=center)
        if rotations:
            center, phi = self.random_rotation(ex_range=r_ex_range)
            self.rotate(phi, center)
        else:
            phi = 0
        return amount, scale, phi

    def small_perturbation(self, idx, eps):
        """
        Applies a small gaussian perturbation with mean 0 and variance
        proportional to eps, to the color, position and orientation of the 
        object at position idx.

        Returns the perturbation vector.
        """
        means = np.zeros(7)
        sigmas = np.array([2, 255, 255, 255, 20, 20, 2 * np.pi]) * eps
        amount = np.random.normal(means, sigmas)
        self.act(idx, amount)
        return amount

    def small_perturb_objects(self, eps):
        """
        Applies a small perturbation to all objects.
        """
        amounts = []
        for i in range(len(self.objects)):
            amounts.append(self.small_perturbation(i, eps))
        return amounts

    def perturb_one(self, idx, r=None, rotate=False):
        """
        Perturbs one object by sampling a radius and an angle at random, with 
        the radius larger than r, and translates the object there.
        """
        if r is None:
            r = self.envsize / 4 
        R_min = r # minimum translation length
        R_max = self.envsize
        R = np.random.random()
        R = R_min * (1 - R) + R_max * R
        theta = np.random.random()
        theta = 2 * np.pi * theta
        addpos = np.array([R * np.cos(theta), R * np.sin(theta)])
        amount = np.zeros(7)
        amount[4:6] = addpos
        self.act(idx, amount)
        if rotate:
            phi = theta = 2 * np.pi * np.random.random()
            self.objects[idx].ori += phi
        return R, theta

    def non_spatial_perturb_one(self, idx):
        """
        Perturbs one object by changing everything except its spatial position.
        """
        minsize = self.envsize / 40
        maxsize = self.envsize / 10
        pos = obj.pos
        si = self.objects[idx].shape_index
        shapelist = list(range(len(N_SH))).pop(si)
        si = np.random.choice(shapelist)
        color = (np.random.random(3) * 255).astype(int)
        size = np.random.random()
        size = (1 - size) * minsize + size * maxsize
        ori = np.random.random() * 2 * np.pi
        amount = np.concatenate([np.array([size]), color, pos, ori], 0)
        self.change_shape(idx, si)
        self.act(idx, amount)

    def perturb_objects(self, n_p):
        """
        Given the current state of the environment, perturbs n_p objects by
        applying random translations and changing their orientation randomly.
        """
        n = len(self.objects)
        n_p = min(n, n_p)
        indices = np.random.choice(n, n_p, replace=False)
        # data = []
        for idx in indices:
            R, theta = self.perturb_one(idx)
            # data.append(np.array([R, theta]))
        # return data

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

    def __init__(self, gridsize, envsize, state=None):
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
        if state is None:
            # re-write this
            self._env.random_config(3)
            state = self._env.to_state_list()
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