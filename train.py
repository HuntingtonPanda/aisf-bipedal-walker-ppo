import gymnasium as gym
#import torch as th #why do i need this again???
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from potential import PotentialShaping

ENV_ID = "BipedalWalker-v3"

RUN_NAME = "bipedalwalker_potential_v6"
SEED = 0

TOTAL_TIMESTEPS = 500_000
N_ENVS = 4

def train():
    os.makedirs("runs", exist_ok=True)
    run_dir = os.path.join("runs", RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)
    
    # VecEnv: run multiple envs in parallel
    venv = make_vec_env(ENV_ID, n_envs=N_ENVS, seed=SEED, wrapper_class=lambda env: PotentialShaping(env, gamma=0.99, alpha=0.2))
    """
    idk man
    def make_env():
        env = gym.make(ENV_ID)
        env = PotentialShaping(env, gamma=0.99, alpha=0.2)
        return env
    """
    venv = VecMonitor(venv)

    # VecNormalize: normalize observations
    # clip_reward to avoid large shaping rewards damaging learning
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0, clip_reward=10.0)

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
        tensorboard_log=os.path.join("runs", "tb"),
    )

    print("Start training")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=RUN_NAME)

    model_path = os.path.join(run_dir, "model.zip")
    norm_path = os.path.join(run_dir, "vecnormalize.pkl")

    model.save(model_path)
    venv.save(norm_path)
    print("Training complete")

    venv.close()

def evaluate(n_eval_episodes: int = 10):
    run_dir = os.path.join("runs", RUN_NAME)
    model_path = os.path.join(run_dir, "model.zip")
    norm_path = os.path.join(run_dir, "vecnormalize.pkl")

    # Evaluation env
    def make_eval_env():
        env = Monitor(gym.make(ENV_ID))
        #env.reset(seed=SEED)
        return env

    eval_env = DummyVecEnv([make_eval_env]) # python will discard the env if not assigned
    eval_env.seed(SEED)
    eval_env = VecNormalize.load(norm_path, eval_env)

    # IMPORTANT!!! evaluation mode <-- set to False
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(model_path, env=eval_env, device="cpu")

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    print(f"Evaluation of {RUN_NAME} over {n_eval_episodes} episodes: mean={mean_reward:.2f} +- {std_reward:.2f}")
    eval_env.close()

if __name__ == "__main__":
    model_path = os.path.join("runs", RUN_NAME, "model.zip")
    if not os.path.exists(model_path):
        train()
    evaluate(n_eval_episodes=20)