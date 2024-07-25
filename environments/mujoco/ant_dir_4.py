

import numpy as np
import random
# from gym.envs.mujoco import AntEnv as AntEnv_
from environments.mujoco.ant import AntEnv


class AntDir4Env(AntEnv):
    def __init__(self, max_episode_steps=200):
        self.set_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self.task_dim = 1
        self.step_count = 0  


        ##
        eval_for_train_task_list = [0 * 0.25 * np.pi,   2 * 0.25 * np.pi,  4 * 0.25 * np.pi,  6 * 0.25 * np.pi]  # [:4]
        eval_for_indis_task_list = [0 * 0.25 * np.pi,   2 * 0.25 * np.pi,  4 * 0.25 * np.pi,  6 * 0.25 * np.pi]  # [4:8]
        eval_for_test_task_list  = [1 * 0.25 * np.pi,   3 * 0.25 * np.pi,  5 * 0.25 * np.pi,  7 * 0.25 * np.pi]  # [8:12]

        train_tsne_tasks = [0 * 0.25 * np.pi,   2 * 0.25 * np.pi,  4 * 0.25 * np.pi,  6 * 0.25 * np.pi]
        test_tsne_tasks  = [1 * 0.25 * np.pi,   3 * 0.25 * np.pi,  5 * 0.25 * np.pi,  7 * 0.25 * np.pi]

        # 총
        self.eval_task_list = eval_for_train_task_list + \
                              eval_for_indis_task_list + \
                              eval_for_test_task_list + \
                              train_tsne_tasks + test_tsne_tasks  #
        ##

        # self.eval_task_list = [0 * 0.25 * np.pi,   2 * 0.25 * np.pi,  4 * 0.25 * np.pi,  6 * 0.25 * np.pi] + [1 * 0.25 * np.pi,   3 * 0.25 * np.pi,  5 * 0.25 * np.pi,  7 * 0.25 * np.pi]  # this list is for visualisation


        super(AntDir4Env, self).__init__()

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self.goal_direction), np.sin(self.goal_direction)) # angle

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2] / self.dt), direct)
        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0

        dense = True #dense reward
        if dense:
            #dense reward setting
            reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        else:
            #sparse reweard setting
            torso_speed = np.linalg.norm(torso_velocity[:2]/ self.dt)
            forward_angle = np.arccos(forward_reward/torso_speed)
            forward_condition = forward_angle< np.pi/8.0
            reward = forward_condition * forward_reward - ctrl_cost - contact_cost + survive_reward

        state = self.state_vector()

        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone

        ob = self._get_obs()

        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
            task=self.get_task()
        )

    def sample_tasks(self, n_tasks):
        return [random.choice([0 * 0.25 * np.pi,   2 * 0.25 * np.pi,  4 * 0.25 * np.pi,  6 * 0.25 * np.pi]) for _ in range(n_tasks, )]

    def set_task(self, task):
        self.goal_direction = task

    def get_task(self):
        return self.goal_direction

    def get_test_task_list(self):
        return self.eval_task_list











