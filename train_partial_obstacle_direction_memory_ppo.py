import os
import random
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from partial_obstacle_direction_memory_env import PartialObstacleDirectionMemoryEnv


SEED = 42
TOTAL_TIMESTEPS = 200_000
EVAL_EPISODES = 500
EVAL_SEED_BASE = 10000
MODEL_PATH = "models/partial_obstacle_direction_memory_ppo"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, episodes=EVAL_EPISODES, deterministic=True):
    successes = 0
    success_steps = []
    failure_steps = []

    for episode in range(episodes):
        env = PartialObstacleDirectionMemoryEnv()
        obs, _ = env.reset(seed=EVAL_SEED_BASE + episode)

        steps = 0
        success = False

        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            steps += 1

            if terminated:
                success = True
                break
            if truncated:
                break

        if success:
            successes += 1
            success_steps.append(steps)
        else:
            failure_steps.append(steps)

        env.close()

    return (
        successes / episodes * 100.0,
        float(np.mean(success_steps)) if success_steps else float("nan"),
        float(np.mean(failure_steps)) if failure_steps else float("nan"),
    )


def evaluate_random(episodes=EVAL_EPISODES):
    successes = 0
    success_steps = []
    failure_steps = []

    for episode in range(episodes):
        env = PartialObstacleDirectionMemoryEnv()
        obs, _ = env.reset(seed=EVAL_SEED_BASE + episode)
        rng = np.random.default_rng(SEED + episode)

        steps = 0
        success = False

        for _ in range(env.max_steps):
            action = int(rng.integers(0, env.action_space.n))
            obs, reward, terminated, truncated, _ = env.step(action)
            steps += 1

            if terminated:
                success = True
                break
            if truncated:
                break

        if success:
            successes += 1
            success_steps.append(steps)
        else:
            failure_steps.append(steps)

        env.close()

    return (
        successes / episodes * 100.0,
        float(np.mean(success_steps)) if success_steps else float("nan"),
        float(np.mean(failure_steps)) if failure_steps else float("nan"),
    )


class EvalCallback(BaseCallback):
    def __init__(self, eval_points):
        super().__init__()
        self.eval_points = list(eval_points)
        self.next_index = 0

    def _on_step(self):
        while (
            self.next_index < len(self.eval_points)
            and self.num_timesteps >= self.eval_points[self.next_index]
        ):
            target = self.eval_points[self.next_index]
            rate, succ_steps, fail_steps = evaluate(self.model)

            print(
                f"{target:>6,} steps | "
                f"Success {rate:6.2f}% | "
                f"Success Steps {succ_steps:6.2f} | "
                f"Failure Steps {fail_steps:6.2f}"
            )

            self.next_index += 1

        return True


def main():
    set_seed(SEED)
    os.makedirs("models", exist_ok=True)

    print("=== Experiment K: Compact Per-Cell Direction Memory ===")
    print("Observation: 21D")
    print("  local obstacles 9 + goal dx/dy 2 + previous action 4")
    print("  + previous move success 1 + collision 1 + current direction scores 4")
    print("Direction memory exists for every cell, but PPO sees only current cell's 4 values.")
    print("Reward: same as Experiment I")
    print("Evaluation: deterministic, 500 fixed maps")
    print()

    random_rate, random_succ_steps, random_fail_steps = evaluate_random()
    print(
        f"Random | Success {random_rate:6.2f}% | "
        f"Success Steps {random_succ_steps:6.2f} | "
        f"Failure Steps {random_fail_steps:6.2f}"
    )

    env = PartialObstacleDirectionMemoryEnv()

    model = PPO(
        "MlpPolicy",
        env,
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

    zero_rate, zero_succ_steps, zero_fail_steps = evaluate(model)
    print(
        f"{0:>6,} steps | "
        f"Success {zero_rate:6.2f}% | "
        f"Success Steps {zero_succ_steps:6.2f} | "
        f"Failure Steps {zero_fail_steps:6.2f}"
    )

    callback = EvalCallback(
        [10_000, 20_000, 50_000, 100_000, 200_000]
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
    )

    model.save(MODEL_PATH)
    env.close()

    print()
    print("Saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
