from stable_baselines3 import PPO

from partial_obstacle_memory_env import PartialObstacleMemoryGridEnv


CHECKPOINTS = [
    0,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
]


def evaluate_random(episodes=500):
    env = PartialObstacleMemoryGridEnv()

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

    return success / episodes * 100, total_steps / episodes


def evaluate(model, episodes=500):
    env = PartialObstacleMemoryGridEnv()

    success = 0
    total_steps = 0

    for episode in range(episodes):
        # Baseline과 완전히 동일한 평가 seed 사용
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

    return success / episodes * 100, total_steps / episodes


if __name__ == "__main__":
    print("=== Experiment #02 - Memory PPO ===")
    print()
    print("Hypothesis:")
    print("Accumulate observed 3x3 areas and visited positions.")
    print("Primary metric: Success Rate")
    print("Pre-experiment prediction: approximately 75%")
    print()

    print("=== Random Agent Baseline ===")

    random_success, random_steps = evaluate_random()

    print(f"Success {random_success:.1f}%")
    print(f"Avg Steps {random_steps:.2f}")

    train_env = PartialObstacleMemoryGridEnv()

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
                f"\nTraining "
                f"{previous_checkpoint:,} -> "
                f"{checkpoint:,} timesteps..."
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
        "models/partial_obstacle_memory_ppo"
    )

    train_env.close()

    print("\n=== Experiment #02 Result ===")

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

    print(
        "\nModel saved: "
        "models/partial_obstacle_memory_ppo"
    )