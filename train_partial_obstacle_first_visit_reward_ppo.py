import os
import numpy as np
from stable_baselines3 import PPO
from partial_obstacle_first_visit_reward_env import PartialObstacleFirstVisitRewardEnv

MODEL_PATH = "models/partial_obstacle_first_visit_reward_ppo"
TOTAL_TIMESTEPS = 200_000
EVAL_EPISODES = 500
SEED = 42

def evaluate(model=None, random_policy=False):
    successes = 0
    success_steps, failure_steps = [], []
    for episode in range(EVAL_EPISODES):
        env = PartialObstacleFirstVisitRewardEnv()
        obs, _ = env.reset(seed=10000 + episode)
        rng = np.random.default_rng(20000 + episode)
        while True:
            if random_policy:
                action = int(rng.integers(0, env.action_space.n))
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                successes += 1
                success_steps.append(env.steps)
                break
            if truncated:
                failure_steps.append(env.steps)
                break
        env.close()
    return (
        successes / EVAL_EPISODES * 100.0,
        float(np.mean(success_steps)) if success_steps else 0.0,
        float(np.mean(failure_steps)) if failure_steps else 0.0,
    )

def show(label, result):
    rate, ss, fs = result
    print(f"{label:<10} | Success: {rate:6.2f}% | Success Steps: {ss:6.2f} | Failure Steps: {fs:6.2f}")

def main():
    os.makedirs("models", exist_ok=True)
    print("=== Experiment G: First Visit Reward ===")
    print("Observation      : 249")
    print("Step             : -1")
    print("Collision        : additional -2 (total -3)")
    print("First Visit      : +1 (once per tile)")
    print("New discovered   : +0.1 / cell")
    print("Goal             : +100")
    print("Training         : 200,000")
    print("Evaluation       : 500 fixed seeds")
    print()

    show("Random", evaluate(random_policy=True))

    env = PartialObstacleFirstVisitRewardEnv()
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=SEED,
        verbose=0,
    )

    show("0", evaluate(model=model))
    current = 0
    for target in [10_000, 20_000, 50_000, 100_000, 200_000]:
        model.learn(total_timesteps=target-current, reset_num_timesteps=False)
        current = target
        show(f"{target:,}", evaluate(model=model))

    model.save(MODEL_PATH)
    env.close()
    print()
    print("Model saved:", MODEL_PATH)

if __name__ == "__main__":
    main()
