import os
import random
import numpy as np
import torch

from stable_baselines3 import PPO

from partial_obstacle_direction_memory_env import PartialObstacleDirectionMemoryEnv


TRAIN_SEEDS = [0, 1, 2, 3, 4]
TOTAL_TIMESTEPS = 200_000
EVAL_EPISODES = 500
EVAL_SEED_BASE = 10000
MODEL_DIR = "models"

# Experiment I의 기존 5-seed 수치와 섞지 않기 위해
# 이번 K 검증은 K 자체의 seed별 결과를 새로 출력한다.


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, episodes=EVAL_EPISODES):
    successes = 0
    success_steps = []
    failure_steps = []

    for episode in range(episodes):
        env = PartialObstacleDirectionMemoryEnv()
        obs, _ = env.reset(seed=EVAL_SEED_BASE + episode)

        steps = 0
        success = False

        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
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

    return {
        "success_rate": successes / episodes * 100.0,
        "successes": successes,
        "success_steps": (
            float(np.mean(success_steps))
            if success_steps else float("nan")
        ),
        "failure_steps": (
            float(np.mean(failure_steps))
            if failure_steps else float("nan")
        ),
    }


def train_one_seed(seed):
    set_seed(seed)

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
        seed=seed,
        verbose=0,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    path = os.path.join(
        MODEL_DIR,
        f"partial_obstacle_direction_memory_ppo_seed{seed}",
    )
    model.save(path)
    env.close()

    return model, path


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=== Experiment K Multi-Seed Validation ===")
    print(f"Training seeds: {TRAIN_SEEDS}")
    print(f"Training steps per seed: {TOTAL_TIMESTEPS:,}")
    print(f"Evaluation: deterministic, {EVAL_EPISODES} fixed maps")
    print(
        f"Evaluation seeds: "
        f"{EVAL_SEED_BASE}.."
        f"{EVAL_SEED_BASE + EVAL_EPISODES - 1}"
    )
    print()

    results = []

    for index, seed in enumerate(TRAIN_SEEDS, start=1):
        print("=" * 90)
        print(
            f"[{index}/{len(TRAIN_SEEDS)}] "
            f"Training seed {seed}"
        )
        print("=" * 90)

        model, path = train_one_seed(seed)
        result = evaluate(model)
        result["seed"] = seed
        result["model_path"] = path
        results.append(result)

        print(
            f"Seed {seed} | "
            f"Success {result['success_rate']:.2f}% "
            f"({result['successes']}/{EVAL_EPISODES}) | "
            f"Success Steps {result['success_steps']:.2f} | "
            f"Failure Steps {result['failure_steps']:.2f}"
        )
        print("Saved:", path)
        print()

    success_rates = np.array(
        [r["success_rate"] for r in results],
        dtype=np.float64,
    )
    success_steps = np.array(
        [r["success_steps"] for r in results],
        dtype=np.float64,
    )

    print("=" * 90)
    print("MULTI-SEED SUMMARY")
    print("=" * 90)

    for r in results:
        print(
            f"Seed {r['seed']} | "
            f"Success {r['success_rate']:6.2f}% | "
            f"Success Steps {r['success_steps']:6.2f}"
        )

    print()
    print(
        f"K Success Rate mean ± std : "
        f"{success_rates.mean():.2f}% ± "
        f"{success_rates.std(ddof=0):.2f}"
    )
    print(
        f"K Success Steps mean ± std: "
        f"{success_steps.mean():.2f} ± "
        f"{success_steps.std(ddof=0):.2f}"
    )
    print(
        f"K Success Rate range      : "
        f"{success_rates.min():.2f}% ~ "
        f"{success_rates.max():.2f}%"
    )

    print()
    print("Finished all 5 training seeds.")


if __name__ == "__main__":
    main()
