from stable_baselines3 import PPO

from partial_obstacle_map_visited_env import (
    PartialObstacleMapVisitedEnv,
)


CHECKPOINTS = [
    0,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
]


EVALUATION_EPISODES = 500


def evaluate_random(
    episodes=EVALUATION_EPISODES,
):
    env = PartialObstacleMapVisitedEnv()

    success_count = 0

    success_steps = []
    failure_steps = []

    for episode in range(episodes):
        obs, _ = env.reset(
            seed=10_000 + episode
        )

        steps = 0

        while True:
            action = env.action_space.sample()

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            steps += 1

            if terminated or truncated:
                if terminated:
                    success_count += 1
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
        success_count
        / episodes
        * 100.0
    )

    success_avg_steps = (
        sum(success_steps)
        / len(success_steps)
        if success_steps
        else 0.0
    )

    failure_avg_steps = (
        sum(failure_steps)
        / len(failure_steps)
        if failure_steps
        else 0.0
    )

    return (
        success_rate,
        success_avg_steps,
        failure_avg_steps,
    )


def evaluate(
    model,
    episodes=EVALUATION_EPISODES,
):
    env = PartialObstacleMapVisitedEnv()

    success_count = 0

    success_steps = []
    failure_steps = []

    for episode in range(episodes):
        # 모든 checkpoint를 동일한 500개 맵으로 평가
        obs, _ = env.reset(
            seed=10_000 + episode
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
                info,
            ) = env.step(action)

            steps += 1

            if terminated or truncated:
                if terminated:
                    success_count += 1
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
        success_count
        / episodes
        * 100.0
    )

    success_avg_steps = (
        sum(success_steps)
        / len(success_steps)
        if success_steps
        else 0.0
    )

    failure_avg_steps = (
        sum(failure_steps)
        / len(failure_steps)
        if failure_steps
        else 0.0
    )

    return (
        success_rate,
        success_avg_steps,
        failure_avg_steps,
    )


if __name__ == "__main__":
    print(
        "=== Map + Visited Memory PPO Experiment ==="
    )

    print()
    print("Observation       : 173")
    print("  Local 3x3       : 9")
    print("  Goal dx/dy      : 2")
    print("  Discovered map  : 81")
    print("  Visited map     : 81")

    print()
    print("Reward")
    print("  Step            : -1.0")
    print("  Discovery       : +0.2 / cell")
    print("  Goal            : +100")

    print()
    print(
        f"Evaluation maps   : "
        f"{EVALUATION_EPISODES}"
    )

    # -------------------------------------------------
    # Random Agent
    # -------------------------------------------------
    print("\n=== Random Agent ===")

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
        PartialObstacleMapVisitedEnv()
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
            print(
                f"\n"
                f"{previous_checkpoint:,}"
                f" → "
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
        ) = evaluate(
            model
        )

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
    # 모델 저장
    # -------------------------------------------------
    model.save(
        "models/"
        "partial_obstacle_map_visited_ppo"
    )

    train_env.close()

    # -------------------------------------------------
    # 결과 출력
    # -------------------------------------------------
    print(
        "\n=== 전체 학습 결과 ==="
    )

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