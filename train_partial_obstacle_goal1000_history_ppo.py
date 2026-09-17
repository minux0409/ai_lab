import numpy as np
from stable_baselines3 import PPO

from partial_obstacle_goal1000_history_env import (
    PartialObstacleGoal1000HistoryEnv,
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


def evaluate_model(
    model,
    episodes=EVAL_EPISODES,
):
    env = PartialObstacleGoal1000HistoryEnv()

    success = 0
    success_steps = []
    failure_steps = []

    for episode in range(episodes):
        obs, _ = env.reset(
            seed=EVAL_SEED_BASE + episode
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
        success / episodes * 100.0
    )

    success_avg_steps = (
        float(np.mean(success_steps))
        if success_steps
        else 0.0
    )

    failure_avg_steps = (
        float(np.mean(failure_steps))
        if failure_steps
        else 0.0
    )

    return (
        success_rate,
        success_avg_steps,
        failure_avg_steps,
    )


def evaluate_random(
    episodes=EVAL_EPISODES,
):
    env = PartialObstacleGoal1000HistoryEnv()

    success = 0
    success_steps = []
    failure_steps = []

    for episode in range(episodes):
        obs, _ = env.reset(
            seed=EVAL_SEED_BASE + episode
        )

        # Random action도 episode별 고정
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
        success / episodes * 100.0
    )

    success_avg_steps = (
        float(np.mean(success_steps))
        if success_steps
        else 0.0
    )

    failure_avg_steps = (
        float(np.mean(failure_steps))
        if failure_steps
        else 0.0
    )

    return (
        success_rate,
        success_avg_steps,
        failure_avg_steps,
    )


if __name__ == "__main__":
    test_env = (
        PartialObstacleGoal1000HistoryEnv()
    )

    print(
        "=== Goal1000 History "
        "Controlled Experiment ==="
    )

    print()
    print(
        "Observation:",
        test_env.observation_space.shape,
    )

    print()
    print("변경:")
    print("  Goal Reward : 100 -> 1000")
    print("  Tile Reward : distance 1=15")
    print()
    print("유지:")
    print("  History     : 기존 그대로")
    print("  Observation : 107")
    print("  Visit Count : 없음")
    print("  Collision   : -1")
    print("  Revisit     : 0")

    test_env.close()

    # ---------------------------------------------
    # Random
    # ---------------------------------------------
    print()
    print("Random 평가 중...")

    (
        random_success,
        random_success_steps,
        random_failure_steps,
    ) = evaluate_random()

    print(
        f"Random | "
        f"Success "
        f"{random_success:.1f}% | "
        f"Success Steps "
        f"{random_success_steps:.2f} | "
        f"Failure Steps "
        f"{random_failure_steps:.2f}"
    )

    # ---------------------------------------------
    # PPO
    # ---------------------------------------------
    train_env = (
        PartialObstacleGoal1000HistoryEnv()
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        learning_rate=0.0003,
        seed=42,
    )

    results = []
    previous_checkpoint = 0

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
                total_timesteps=additional_steps,
                reset_num_timesteps=False,
            )

        (
            success_rate,
            success_steps,
            failure_steps,
        ) = evaluate_model(model)

        results.append(
            (
                checkpoint,
                success_rate,
                success_steps,
                failure_steps,
            )
        )

        print(
            f"{checkpoint:,} timestep | "
            f"Success "
            f"{success_rate:.1f}% | "
            f"Success Steps "
            f"{success_steps:.2f} | "
            f"Failure Steps "
            f"{failure_steps:.2f}"
        )

        previous_checkpoint = checkpoint

    # ---------------------------------------------
    # Save
    # ---------------------------------------------
    MODEL_PATH = (
        "models/"
        "partial_obstacle_"
        "goal1000_history_ppo"
    )

    model.save(MODEL_PATH)

    train_env.close()

    # ---------------------------------------------
    # Summary
    # ---------------------------------------------
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
        success_steps,
        failure_steps,
    ) in results:
        print(
            f"{checkpoint:>7,} | "
            f"Success "
            f"{success_rate:>5.1f}% | "
            f"Success Steps "
            f"{success_steps:>6.2f} | "
            f"Failure Steps "
            f"{failure_steps:>6.2f}"
        )

    print()
    print(
        "모델 저장:",
        MODEL_PATH + ".zip",
    )