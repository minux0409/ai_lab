import numpy as np
from stable_baselines3 import PPO

from partial_obstacle_goal1000_visit_env import (
    PartialObstacleGoal1000VisitEnv,
)


CHECKPOINTS = [
    0,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
]

EVAL_EPISODES = 500
EVAL_SEED_BASE = 10_000


def evaluate_random(
    episodes=EVAL_EPISODES,
):
    env = (
        PartialObstacleGoal1000VisitEnv()
    )

    success = 0
    success_steps = []
    failure_steps = []

    for episode in range(episodes):
        obs, _ = env.reset(
            seed=(
                EVAL_SEED_BASE
                + episode
            )
        )

        # Random baseline 자체도 재현 가능하게
        rng = np.random.default_rng(
            20_000 + episode
        )

        steps = 0

        while True:
            action = int(
                rng.integers(
                    0,
                    env.action_space.n,
                )
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)

            steps += 1

            if terminated or truncated:
                if terminated:
                    success += 1
                    success_steps.append(
                        steps
                    )
                else:
                    failure_steps.append(
                        steps
                    )

                break

    env.close()

    success_rate = (
        success
        / episodes
        * 100.0
    )

    success_avg_steps = (
        np.mean(success_steps)
        if success_steps
        else 0.0
    )

    failure_avg_steps = (
        np.mean(failure_steps)
        if failure_steps
        else 0.0
    )

    return (
        success_rate,
        float(success_avg_steps),
        float(failure_avg_steps),
    )


def evaluate_model(
    model,
    episodes=EVAL_EPISODES,
):
    env = (
        PartialObstacleGoal1000VisitEnv()
    )

    success = 0
    success_steps = []
    failure_steps = []

    for episode in range(episodes):
        # 기존 실험과 동일한 500개 맵
        obs, _ = env.reset(
            seed=(
                EVAL_SEED_BASE
                + episode
            )
        )

        steps = 0

        while True:
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)

            steps += 1

            if terminated or truncated:
                if terminated:
                    success += 1

                    success_steps.append(
                        steps
                    )

                else:
                    failure_steps.append(
                        steps
                    )

                break

    env.close()

    success_rate = (
        success
        / episodes
        * 100.0
    )

    success_avg_steps = (
        np.mean(success_steps)
        if success_steps
        else 0.0
    )

    failure_avg_steps = (
        np.mean(failure_steps)
        if failure_steps
        else 0.0
    )

    return (
        success_rate,
        float(success_avg_steps),
        float(failure_avg_steps),
    )


if __name__ == "__main__":
    config_env = (
        PartialObstacleGoal1000VisitEnv()
    )

    print(
        "=== Goal 1000 + "
        "Visit Count Experiment ==="
    )

    print()
    print("Observation        : 188")
    print("  Previous History : 107")
    print("  Visit Count Map  : 81")

    print()
    print("Reward")
    print("  Step             : +0")
    print("  Discovery        : +0")
    print("  Collision        : -1")
    print("  Revisit          : +0")
    print("  Distance 1       : +15")
    print("  Distance 2       : +14")
    print("  ...")
    print("  Distance 15      : +1")
    print("  Distance 16      : +1")
    print("  Goal             : +1000")

    print()
    print(
        "Visit Count        : "
        "min(count, 10) / 10"
    )

    print(
        "Visit penalty      : NONE"
    )

    config_env.close()

    # -------------------------------------------------
    # Random baseline
    # -------------------------------------------------
    print()
    print("=== Random Agent ===")

    (
        random_success,
        random_success_steps,
        random_failure_steps,
    ) = evaluate_random()

    print(
        f"Success "
        f"{random_success:.1f}% | "
        f"Success Steps "
        f"{random_success_steps:.2f} | "
        f"Failure Steps "
        f"{random_failure_steps:.2f}"
    )

    # -------------------------------------------------
    # PPO
    # -------------------------------------------------
    train_env = (
        PartialObstacleGoal1000VisitEnv()
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        learning_rate=0.0003,
        seed=42,
    )

    previous_checkpoint = 0
    results = []

    for checkpoint in CHECKPOINTS:
        additional_steps = (
            checkpoint
            - previous_checkpoint
        )

        if additional_steps > 0:
            print()
            print(
                f"{previous_checkpoint:,} "
                f"-> "
                f"{checkpoint:,} "
                f"timestep 학습..."
            )

            model.learn(
                total_timesteps=(
                    additional_steps
                ),
                reset_num_timesteps=False,
            )

        (
            success_rate,
            success_avg_steps,
            failure_avg_steps,
        ) = evaluate_model(model)

        results.append(
            (
                checkpoint,
                success_rate,
                success_avg_steps,
                failure_avg_steps,
            )
        )

        print(
            f"{checkpoint:,} timestep | "
            f"Success "
            f"{success_rate:.1f}% | "
            f"Success Steps "
            f"{success_avg_steps:.2f} | "
            f"Failure Steps "
            f"{failure_avg_steps:.2f}"
        )

        previous_checkpoint = checkpoint

    # -------------------------------------------------
    # 새 모델로 별도 저장
    # -------------------------------------------------
    MODEL_PATH = (
        "models/"
        "partial_obstacle_"
        "goal1000_visit_ppo"
    )

    model.save(MODEL_PATH)

    train_env.close()

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------
    print()
    print("=== 전체 학습 결과 ===")

    print(
        f"Random  | "
        f"Success "
        f"{random_success:>5.1f}% | "
        f"Success Steps "
        f"{random_success_steps:>6.2f} | "
        f"Failure Steps "
        f"{random_failure_steps:>6.2f}"
    )

    for (
        checkpoint,
        success_rate,
        success_avg_steps,
        failure_avg_steps,
    ) in results:
        print(
            f"{checkpoint:>7,} | "
            f"Success "
            f"{success_rate:>5.1f}% | "
            f"Success Steps "
            f"{success_avg_steps:>6.2f} | "
            f"Failure Steps "
            f"{failure_avg_steps:>6.2f}"
        )

    print()
    print(
        "모델 저장:"
    )

    print(
        MODEL_PATH + ".zip"
    )