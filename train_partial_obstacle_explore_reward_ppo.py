from stable_baselines3 import PPO

from partial_obstacle_explore_reward_env import (
    PartialObstacleExploreRewardEnv,
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
    env = PartialObstacleExploreRewardEnv()

    success = 0
    total_steps = 0

    for episode in range(episodes):
        obs, _ = env.reset(seed=10_000 + episode)
        steps = 0

        while True:
            action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)
            steps += 1

            if terminated or truncated:
                if terminated:
                    success += 1

                total_steps += steps
                break

    env.close()

    return (
        success / episodes * 100,
        total_steps / episodes,
    )


def evaluate(model, episodes=500):
    env = PartialObstacleExploreRewardEnv()

    success = 0
    total_steps = 0

    for episode in range(episodes):
        # Baseline과 동일한 500개 seed 사용
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

    return (
        success / episodes * 100,
        total_steps / episodes,
    )


if __name__ == "__main__":
    print("=== Explore Reward Experiment ===")
    print("Step penalty       : -1.0")
    print("Discovery reward   : +0.2 / cell")
    print("Goal reward        : +100")
    print("Observation        : 11 (Baseline과 동일)")
    print()

    print("=== Random Agent ===")

    random_success, random_steps = evaluate_random()

    print(f"성공률: {random_success:.1f}%")
    print(f"평균 행동 수: {random_steps:.2f}")

    train_env = PartialObstacleExploreRewardEnv()

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
        additional_steps = checkpoint - previous_checkpoint

        if additional_steps > 0:
            print(
                f"\n{previous_checkpoint:,} → "
                f"{checkpoint:,} timestep 학습..."
            )

            model.learn(
                total_timesteps=additional_steps,
                reset_num_timesteps=False,
            )

        success_rate, avg_steps = evaluate(model)

        results.append(
            (
                checkpoint,
                success_rate,
                avg_steps,
            )
        )

        print(
            f"{checkpoint:,} timestep | "
            f"Success {success_rate:.1f}% | "
            f"Steps {avg_steps:.2f}"
        )

        previous_checkpoint = checkpoint

    model.save(
        "models/partial_obstacle_explore_reward_ppo"
    )

    train_env.close()

    print("\n=== 전체 학습 결과 ===")

    print(
        f"Random  | "
        f"Success {random_success:.1f}% | "
        f"Steps {random_steps:.2f}"
    )

    for checkpoint, success_rate, avg_steps in results:
        print(
            f"{checkpoint:>7,} | "
            f"Success {success_rate:>5.1f}% | "
            f"Steps {avg_steps:>6.2f}"
        )