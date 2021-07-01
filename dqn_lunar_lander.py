import sys

import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


def dqn_model(env, model_name, current_model=False):
    if not current_model:
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            exploration_final_eps=0.1,
            target_update_interval=250
        )
        model.learn(total_timesteps=int(1e5))
        model.save(model_name)

    model = DQN.load(model_name, env=env)
    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=10)
    return model


def main(new_model=False):
    rng = np.random.default_rng(2021)
    rng.random(4)

    env = gym.make("LunarLander-v2")
    model = dqn_model(env, "dqn_lunar", new_model)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


if __name__ == "__main__":
    new_model = True if sys.argv[1] == "new" else False
    main(new_model)
