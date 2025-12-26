import gymnasium as gym
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

ENV_ID = "BipedalWalker-v3"

def main():
    # VecEnv: run multiple envs in parallel
    venv = make_vec_env(ENV_ID, n_envs=4, seed=0)
    venv = VecMonitor(venv)

    # VecNormalize: normalize observations
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # PPO model
    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        device="cpu",
    )

    print("Start test")
    model.learn(total_timesteps = 10_000)
    print("Test complete")

    venv.close()

if __name__ == "__main__":
    main()
