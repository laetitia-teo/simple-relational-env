import numpy as np

class AbstractEnv():

    def __init__(self):
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
