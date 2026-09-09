from stable_baselines3 import PPO

from partial_obstacle_history_env import PartialObstacleHistoryGridEnv


CHECKPOINTS = [
    0,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
]


def evaluate(model, episodes=500):
    env = PartialObstacleHistoryGridEnv()

    success = 0
    total_steps = 0

    for episode in range(episodes):
        # 기존 실험과 동일한 평가 맵
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


if __name__ == "__main__":
    train_env = PartialObstacleHistoryGridEnv()

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

        result = evaluate(
            model,
            episodes=500,
        )

        results.append(
            {
                "timesteps": checkpoint,
                **result,
            }
        )

        print(
            f"{checkpoint:,} timestep | "
            f"Success {result['success_rate']:.1f}% | "
            f"Steps {result['avg_steps']:.2f}"
        )

        previous_checkpoint = checkpoint

    model.save(
        "models/partial_obstacle_history_ppo"
    )

    train_env.close()

    print("\n=== 전체 학습 결과 ===")

    for result in results:
        print(
            f"{result['timesteps']:>7,} | "
            f"Success {result['success_rate']:>5.1f}% | "
            f"Steps {result['avg_steps']:>6.2f}"
        )

    print("\n=== 기존 모델과 비교 ===")
    print("Baseline 200k : Success 84.4% | Steps 20.65")
    print("Revisit  200k : Success 81.4% | Steps 23.43")

    print(
        "History  200k : "
        f"Success {results[-1]['success_rate']:.1f}% | "
        f"Steps {results[-1]['avg_steps']:.2f}"
    )