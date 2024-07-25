import random

import numpy as np

from environments.mujoco.rand_param_envs.gym.core import Env
from environments.mujoco.rand_param_envs.gym.envs.mujoco import MujocoEnv
from utils import helpers as utl

class MetaEnv(Env):
    def step(self, *args, **kwargs):
        return self._step(*args, **kwargs)

    def sample_tasks(self, n_tasks):
        """
        Samples task of the meta-environment

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        raise NotImplementedError

    def set_task(self, task):
        """
        Sets the specified task to the current environment

        Args:
            task: task of the meta-learning environment
        """
        raise NotImplementedError

    def get_task(self):
        """
        Gets the task that the agent is performing in the current environment

        Returns:
            task: task of the meta-learning environment
        """
        raise NotImplementedError

    def reset_task(self, task):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)

    def log_diagnostics(self, paths, prefix):
        """
        Logs env-specific diagnostic information

        Args:
            paths (list) : list of all paths collected with this env during this iteration
            prefix (str) : prefix for logger
        """
        pass


class RandomEnv(MetaEnv, MujocoEnv):
    """
    This class provides functionality for randomizing the physical parameters of a mujoco model
    The following parameters are changed:
        - body_mass
        - body_inertia
        - damping coeff at the joints
    """
    RAND_PARAMS = ['body_mass', 'dof_damping', 'body_inertia', 'geom_friction']
    RAND_PARAMS_EXTENDED = RAND_PARAMS + ['geom_size']

    def __init__(self, log_scale_limit, file_name, *args, rand_params=RAND_PARAMS, **kwargs):
        self.log_scale_limit = log_scale_limit
        self.rand_params = rand_params
        # print("file_name in base2.py", file_name)

        MujocoEnv.__init__(self, file_name, 4)
        assert set(rand_params) <= set(self.RAND_PARAMS_EXTENDED), \
            "rand_params must be a subset of " + str(self.RAND_PARAMS_EXTENDED)
        self.save_parameters()
        self.task_dim = self.rand_param_dim

    def sample_tasks(self, n_tasks):
        param_sets = []
        for _ in range(n_tasks):
            # body mass -> one multiplier for all body parts
            prob = random.random()
            if prob >= 0.5:
                body_mass_multiplyers_ = random.uniform(0, 0.5)
            else:
                body_mass_multiplyers_ = random.uniform(3.0, 3.5)
            new_params = self.set_mass_param(body_mass_multiplyers_)
            param_sets.append(new_params)
        return param_sets


    def set_mass_param(self, body_mass_multiplyers_):
        mass_size_ = np.prod(self.model.body_mass.shape)
        new_params = {}

        body_mass_multiplyers = np.array([body_mass_multiplyers_ for _ in range(mass_size_)])
        body_mass_multiplyers = np.array(1.5) ** body_mass_multiplyers
        body_mass_multiplyers = np.array(body_mass_multiplyers).reshape(self.model.body_mass.shape)

        new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers
        return new_params


    def set_task(self, task):

        # print("task in set_task", task)
        if type(task) == int:
            print("if type(task) == int:", task)
            task_idx = task
            task = self.set_mass_param(self.eval_task_list[task_idx])

        for param, param_val in task.items():
            param_variable = getattr(self.model, param)
            assert param_variable.shape == param_val.shape, 'shapes of new parameter value and old one must match'
            setattr(self.model, param, param_val)
        self.curr_params = task

    def get_task(self):
        if hasattr(self, 'curr_params'):
            task = self.curr_params
            task = np.concatenate([task[k].reshape(-1) for k in task.keys()])
        else:
            task = np.zeros(self.rand_param_dim)
        return task

    def save_parameters(self):
        self.init_params = {}
        if 'body_mass' in self.rand_params:
            self.init_params['body_mass'] = self.model.body_mass

        self.curr_params = self.init_params
