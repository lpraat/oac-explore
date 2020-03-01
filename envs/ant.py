import numpy as np

from gym.envs.mujoco.ant_v3 import AntEnv

class Ant(AntEnv):

    def __init__(self, get_reward=None, *args, **kwargs):
        kwargs['exclude_current_positions_from_observation'] = False
        super().__init__(*args, **kwargs)
        if self._exclude_current_positions_from_observation == True:
            self.num_features = 27
        else:
            self.num_features = 29

        if get_reward is None:
            self.get_reward = lambda o, r, d, i : (r, d)
        else:
            self.get_reward = get_reward

    def _seed(self, seed=None):
        return self.seed(seed)

    def _step(self, action):
        observation, reward, done, info = self.step(action)

        # Get custom reward using provided lambda
        reward, done = self.get_reward(observation, reward, done, info)

        return observation[:self.num_features], reward, done, info

    def _reset(self):
        return self.reset()[:self.num_features]

    def _render(self, mode='human'):
        return self.render(mode)

if __name__ == '__main__':
    get_reward = lambda o, r, d, i : r
    env = Ant(get_reward, exclude_current_positions_from_observation=False)
    print(env.state_vector())
    for i in range(10000):
        s, r, done, _ = env._step(env.action_space.sample())
        np.testing.assert_almost_equal(s, env.state_vector())