from stable_baselines3 import PPO

from partial_obstacle_map_reward_env import (
    PartialObstacleMapRewardEnv,
)


CHECKPOINTS = [
    0,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
]


def evaluate_random(episodes=500):
    env = PartialObstacleMapRewardEnv()

    success = 0

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
                _,
            ) = env.step(action)

            steps += 1

            if terminated or truncated:
                if terminated:
                    success += 1
                    success_steps.append(steps)
                else:
                    failure_steps.append(steps)

                break

    env.close()

    success_rate = (
        success / episodes * 100
    )

    success_avg_steps = (
        sum(success_steps) / len(success_steps)
        if success_steps
        else 0.0
    )

    failure_avg_steps = (
        sum(failure_steps) / len(failure_steps)
        if failure_steps
        else 0.0
    )

    return (
        success_rate,
        success_avg_steps,
        failure_avg_steps,
    )


def evaluate(model, episodes=500):
    env = PartialObstacleMapRewardEnv()

    success = 0

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
                _,
            ) = env.step(action)

            steps += 1

            if terminated or truncated:
                if terminated:
                    success += 1
                    success_steps.append(steps)
                else:
                    failure_steps.append(steps)

                break

    env.close()

    success_rate = (
        success / episodes * 100
    )

    success_avg_steps = (
        sum(success_steps) / len(success_steps)
        if success_steps
        else 0.0
    )

    failure_avg_steps = (
        sum(failure_steps) / len(failure_steps)
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
        "=== Map Memory + Position + Explore Reward Experiment ==="
    )

    print()
    print("Observation       : 94")
    print("  Local 3x3       : 9")
    print("  Goal dx/dy      : 2")
    print("  Discovered map  : 81")
    print("  Agent x/y       : 2")

    print()
    print("Step penalty      : -1.0")
    print("Discovery reward  : +0.2 / cell")
    print("Goal reward       : +100")

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
        f"Success {random_success:.1f}% | "
        f"Success Steps {random_success_steps:.2f} | "
        f"Failure Steps {random_failure_steps:.2f}"
    )

    # -------------------------------------------------
    # PPO
    # -------------------------------------------------
    train_env = PartialObstacleMapRewardEnv()

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
            checkpoint - previous_checkpoint
        )

        if additional_steps > 0:
            print(
                f"\n{previous_checkpoint:,} → "
                f"{checkpoint:,} timestep 학습..."
            )

            model.learn(
                total_timesteps=additional_steps,
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
            f"Success {success_rate:.1f}% | "
            f"Success Steps {success_avg_steps:.2f} | "
            f"Failure Steps {failure_avg_steps:.2f}"
        )

        previous_checkpoint = checkpoint

    # -------------------------------------------------
    # 94 Observation 모델은 별도 이름으로 저장
    # 92 Observation 실험 결과를 덮어쓰지 않는다.
    # -------------------------------------------------
    model.save(
        "models/"
        "partial_obstacle_map_position_reward_ppo"
    )

    train_env.close()

    # -------------------------------------------------
    # 최종 결과
    # -------------------------------------------------
    print("\n=== 전체 학습 결과 ===")

    print(
        f"Random  | "
        f"Success {random_success:>5.1f}% | "
        f"Success Steps {random_success_steps:>6.2f} | "
        f"Failure Steps {random_failure_steps:>6.2f}"
    )

    for (
        checkpoint,
        success_rate,
        success_avg_steps,
        failure_avg_steps,
    ) in results:
        print(
            f"{checkpoint:>7,} | "
            f"Success {success_rate:>5.1f}% | "
            f"Success Steps {success_avg_steps:>6.2f} | "
            f"Failure Steps {failure_avg_steps:>6.2f}"
        )