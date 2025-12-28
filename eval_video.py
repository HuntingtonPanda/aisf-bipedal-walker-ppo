import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder

ENV_ID = "BipedalWalker-v3"
RUN_NAME = "bipedalwalker_potential_v1"

def main():
    run_dir = os.path.join("runs", RUN_NAME)
    model_path = os.path.join(run_dir, "model.zip")
    norm_path = os.path.join(run_dir, "vecnormalize.pkl")

    # record video in ./videos
    os.makedirs("videos", exist_ok=True)

    def make_env():
        return gym.make(ENV_ID, render_mode="rgb_array")

    venv = DummyVecEnv([make_env])
    venv = VecNormalize.load(norm_path, venv)
    venv.training = False
    venv.norm_reward = False
    
    venv = VecVideoRecorder(
        venv,
        video_folder="videos",
        record_video_trigger=lambda step: True,
        video_length=10000, # CHANGE TO DESIRED LENGTH <---------------- REMEMBER TO ADJUST
        name_prefix=RUN_NAME,
    )

    model = PPO.load(model_path, env=venv)

    obs = venv.reset()
    done = [False]
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)

    venv.close()
    print("Video recorded")

if __name__ == "__main__":
    main()
