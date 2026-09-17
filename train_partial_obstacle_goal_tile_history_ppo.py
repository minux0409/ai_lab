from stable_baselines3 import PPO

from partial_obstacle_goal_tile_history_env import (
    PartialObstacleGoalTileHistoryEnv,
)


CHECKPOINTS = [
    0,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
]


def evaluate_random(
    episodes=500,
):
    env = (
        PartialObstacleGoalTileHistoryEnv()
    )

    success = 0
    success_steps = []
    failure_steps = []

    for episode in range(
        episodes
    ):
        obs, _ = env.reset(
            seed=10_000 + episode
        )

        steps = 0

        while True:
            action = (
                env.action_space.sample()
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)

            steps += 1

            if (
                terminated
                or truncated
            ):
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
        * 100
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
    episodes=500,
):
    env = (
        PartialObstacleGoalTileHistoryEnv()
    )

    success = 0
    success_steps = []
    failure_steps = []

    for episode in range(
        episodes
    ):
        # 이전 실험과 동일한
        # 500개 평가 맵
        obs, _ = env.reset(
            seed=10_000 + episode
        )

        steps = 0

        while True:
            action, _ = (
                model.predict(
                    obs,
                    deterministic=True,
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

            if (
                terminated
                or truncated
            ):
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
        * 100
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
    config_env = (
        PartialObstacleGoalTileHistoryEnv()
    )

    print(
        "=== Goal Tile + "
        "Short History Experiment ==="
    )

    print()
    print("Observation       : 107")
    print("  Local 3x3       : 9")
    print("  Goal dx/dy      : 2")
    print("  Discovered map  : 81")
    print("  Agent x/y       : 2")
    print("  Last Action     : 4")
    print("  Last Collision  : 1")
    print("  Position History: 8")

    print()
    print(
        f"Step penalty      : "
        f"{config_env.step_penalty:+.1f}"
    )

    print(
        f"Discovery reward  : "
        f"{config_env.discovery_reward:+.1f}"
    )

    print(
        f"Collision penalty : "
        f"{config_env.collision_penalty:+.1f}"
    )

    print(
        "Tile reward       : "
        "first visit only, "
        "max(1, 9-distance)"
    )

    print(
        f"Goal reward       : "
        f"{config_env.goal_reward:+.1f}"
    )

    config_env.close()

    # -------------------------------------------------
    # Random
    # -------------------------------------------------
    print(
        "\n=== Random Agent ==="
    )

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
        PartialObstacleGoalTileHistoryEnv()
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
                f"{previous_checkpoint:,} "
                f"→ "
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
        ) = evaluate(model)

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

        previous_checkpoint = (
            checkpoint
        )

    # -------------------------------------------------
    # 별도 모델 저장
    # -------------------------------------------------
    MODEL_PATH = (
        "models/"
        "partial_obstacle_"
        "goal_tile_history_ppo"
    )

    model.save(
        MODEL_PATH
    )

    train_env.close()

    # -------------------------------------------------
    # 결과
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

    print()
    print(
        "모델 저장:"
    )

    print(
        MODEL_PATH + ".zip"
    )