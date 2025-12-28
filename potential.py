import gymnasium as gym

# Potential-based shaping: r' = r + alpha * (gamma*phi(s') - phi(s))
class PotentialShaping(gym.Wrapper):
    def __init__(self, env, gamma=0.99, alpha=1.0):
        super().__init__(env)
        self.gamma = gamma
        self.alpha = alpha
        self._prev_phi = None

    # obs[0] = hull angle
    # obs[2] = horizontal velocity
    def phi(self, obs):
        # encourage forward velocity (obs[2])
        # discourage large hull angle (obs[0])
        hull_angle = float(obs[0]) # IS THIS NOT ALREADY A FLOAT???
        x_vel = float(obs[2])
        return (1.0 * x_vel) - (0.5 * abs(hull_angle))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = self.phi(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        curr_phi = self.phi(obs)
        shaping = self.gamma * curr_phi - self._prev_phi
        self._prev_phi = curr_phi

        reward = reward + self.alpha * shaping
        return obs, reward, terminated, truncated, info
    
"""
@inproceedings{10.5555/645528.657613,
author = {Ng, Andrew Y. and Harada, Daishi and Russell, Stuart J.},
title = {Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping},
year = {1999},
isbn = {1558606122},
publisher = {Morgan Kaufmann Publishers Inc.},
address = {San Francisco, CA, USA},
booktitle = {Proceedings of the Sixteenth International Conference on Machine Learning},
pages = {278–287},
numpages = {10},
series = {ICML '99}
}
"""