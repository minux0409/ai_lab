import numpy as np
from stable_baselines3 import PPO

from partial_obstacle_env import PartialObstacleGridEnv
from partial_obstacle_history_env import PartialObstacleHistoryGridEnv


SEEDS = [0, 1, 2, 3, 4]
TRAIN_TIMESTEPS = 200_000
EVAL_EPISODES = 500


def evaluate(model, env_class, episodes=EVAL_EPISODES):
    env = env_class()

    success = 0
    total_steps = 0

    for episode in range(episodes):
        # 모든 모델을 동일한 500개 평가 맵으로 비교
        obs, _ = env.reset(seed=10_000 + episode)

        steps = 0

        while True:
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, _ = env.step(action)

            steps += 1

            if terminated or truncated:
                if terminated:
                    success += 1

                total_steps += steps
                break

    env.close()

    return {
        "success_rate": success / episodes * 100,
        "avg_steps": total_steps / episodes,
    }


def train_and_evaluate(env_class, seed, model_name):
    env = env_class()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=0.0003,
        seed=seed,
    )

    print(
        f"{model_name} | seed={seed} | "
        f"{TRAIN_TIMESTEPS:,} timestep 학습..."
    )

    model.learn(
        total_timesteps=TRAIN_TIMESTEPS,
    )

    env.close()

    result = evaluate(
        model,
        env_class,
    )

    print(
        f"  Success {result['success_rate']:.1f}% | "
        f"Steps {result['avg_steps']:.2f}"
    )

    return result


def summarize(name, results):
    success_rates = np.array(
        [r["success_rate"] for r in results]
    )

    avg_steps = np.array(
        [r["avg_steps"] for r in results]
    )

    print(f"\n=== {name} Summary ===")

    for seed, result in zip(SEEDS, results):
        print(
            f"seed {seed}: "
            f"Success {result['success_rate']:.1f}% | "
            f"Steps {result['avg_steps']:.2f}"
        )

    print()
    print(
        f"Success Mean ± Std: "
        f"{success_rates.mean():.2f}% "
        f"± {success_rates.std(ddof=1):.2f}"
    )

    print(
        f"Steps Mean ± Std: "
        f"{avg_steps.mean():.2f} "
        f"± {avg_steps.std(ddof=1):.2f}"
    )

    return {
        "success_mean": success_rates.mean(),
        "success_std": success_rates.std(ddof=1),
        "steps_mean": avg_steps.mean(),
        "steps_std": avg_steps.std(ddof=1),
    }


if __name__ == "__main__":
    baseline_results = []
    history_results = []

    print("\n######## BASELINE PPO ########\n")

    for seed in SEEDS:
        result = train_and_evaluate(
            PartialObstacleGridEnv,
            seed,
            "Baseline",
        )

        baseline_results.append(result)

    print("\n######## HISTORY PPO ########\n")

    for seed in SEEDS:
        result = train_and_evaluate(
            PartialObstacleHistoryGridEnv,
            seed,
            "History",
        )

        history_results.append(result)

    baseline_summary = summarize(
        "Baseline PPO",
        baseline_results,
    )

    history_summary = summarize(
        "History Observation PPO",
        history_results,
    )

    print("\n=== 최종 비교 ===")

    print(
        "Baseline : "
        f"{baseline_summary['success_mean']:.2f}% "
        f"± {baseline_summary['success_std']:.2f}"
    )

    print(
        "History  : "
        f"{history_summary['success_mean']:.2f}% "
        f"± {history_summary['success_std']:.2f}"
    )

    improvement = (
        history_summary["success_mean"]
        - baseline_summary["success_mean"]
    )

    print(
        f"평균 성공률 차이: "
        f"{improvement:+.2f}%p"
    )