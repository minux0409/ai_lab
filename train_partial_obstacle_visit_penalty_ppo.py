import os
import random
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from partial_obstacle_visit_penalty_env import PartialObstacleVisitPenaltyEnv


SEED = 42
TOTAL_TIMESTEPS = 200_000
EVAL_EPISODES = 500
EVAL_SEED_BASE = 10000
MODEL_PATH = "models/partial_obstacle_visit_penalty_ppo"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, episodes=EVAL_EPISODES, deterministic=True):
    successes = 0
    success_steps = []
    failure_steps = []

    for episode in range(episodes):
        env = PartialObstacleVisitPenaltyEnv()
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

    success_rate = successes / episodes * 100.0
    avg_success_steps = (
        float(np.mean(success_steps)) if success_steps else float("nan")
    )
    avg_failure_steps = (
        float(np.mean(failure_steps)) if failure_steps else float("nan")
    )

    return success_rate, avg_success_steps, avg_failure_steps


def evaluate_random(episodes=EVAL_EPISODES):
    successes = 0
    success_steps = []
    failure_steps = []

    for episode in range(episodes):
        env = PartialObstacleVisitPenaltyEnv()
        obs, _ = env.reset(seed=EVAL_SEED_BASE + episode)

        # 평가 재현성을 위해 action RNG도 고정
        action_rng = np.random.default_rng(SEED + episode)

        steps = 0
        success = False

        for _ in range(env.max_steps):
            action = int(action_rng.integers(0, env.action_space.n))
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

    print("=== Experiment J: Compact Memory + Visit-Count Penalty ===")
    print("Observation: same 18D observation as Experiment I")
    print("Reward: Experiment I reward + destination previous visit count * -1")
    print("Evaluation: deterministic, 500 fixed maps")
    print()

    random_rate, random_succ_steps, random_fail_steps = evaluate_random()
    print(
        f"Random | Success {random_rate:6.2f}% | "
        f"Success Steps {random_succ_steps:6.2f} | "
        f"Failure Steps {random_fail_steps:6.2f}"
    )

    env = PartialObstacleVisitPenaltyEnv()

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

    # 학습 전 동일 deterministic 평가
    zero_rate, zero_succ_steps, zero_fail_steps = evaluate(model)
    print(
        f"{0:>6,} steps | "
        f"Success {zero_rate:6.2f}% | "
        f"Success Steps {zero_succ_steps:6.2f} | "
        f"Failure Steps {zero_fail_steps:6.2f}"
    )

    callback = EvalCallback(
        eval_points=[10_000, 20_000, 50_000, 100_000, 200_000]
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
