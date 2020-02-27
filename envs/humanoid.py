import numpy as np

from gym.envs.mujoco.humanoid_v3 import HumanoidEnv

class Humanoid(HumanoidEnv):

    def __init__(self, get_reward=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_features = self.state_vector().shape[0]

        if get_reward is None:
            self.get_reward = lambda o, r, d, i : (r, d)
        else:
            self.get_reward = get_reward

        print("Adding non standup positions...")
        self.init_states = []
        while len(self.init_states) < 1000:
            s = self.reset()
            step = 0
            for _ in range(5000):
                s, r, d, i = self.step(self.action_space.sample())

                s = self.state_vector()

                if step > 100 and s[2] < 1.0:
                    self.init_states.append([s[0:24], s[24:47]])
                    break

                step += 1

    def _seed(self, seed=None):
        return self.seed(seed)

    def _step(self, action):
        observation, reward, done, info = self.step(action)
        observation = self.state_vector()

        # Get custom reward using provided lambda
        reward, done = self.get_reward(observation, reward, done, info)

        return self.state_vector(), reward, done, info

    def _reset(self):
        self.sim.reset()

        self.set_state(*self.init_states[np.random.randint(len(self.init_states))])

        return self.state_vector()

    def _render(self, mode='human'):
        return self.render(mode)


if __name__ == '__main__':
    env = Humanoid()
    print(env.__class__.__name__)

    for _ in range(10):
        s = env._reset()
        for _ in range(500):
            env.render()
            s, r, d, i = env._step(env.action_space.sample())

