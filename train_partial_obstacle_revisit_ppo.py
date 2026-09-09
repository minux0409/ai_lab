from stable_baselines3 import PPO

from partial_obstacle_revisit_env import PartialObstacleGridEnv


CHECKPOINTS = [
    0,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
]


def evaluate(model, episodes=500):
    env = PartialObstacleGridEnv()

    success = 0
    total_steps = 0
    total_reward = 0.0

    for episode in range(episodes):
        # 기존 실험과 동일한 평가 seed
        obs, _ = env.reset(seed=10_000 + episode)

        episode_reward = 0.0
        steps = 0

        while True:
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            steps += 1

            if terminated or truncated:
                if terminated:
                    success += 1

                total_steps += steps
                total_reward += episode_reward
                break

    env.close()

    return {
        "success_rate": success / episodes * 100,
        "avg_steps": total_steps / episodes,
        "avg_reward": total_reward / episodes,
    }


if __name__ == "__main__":
    train_env = PartialObstacleGridEnv()

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
            f"Steps {result['avg_steps']:.2f} | "
            f"Reward {result['avg_reward']:.2f}"
        )

        previous_checkpoint = checkpoint

    model.save(
        "models/partial_obstacle_revisit_ppo"
    )

    train_env.close()

    print("\n=== 전체 학습 결과 ===")

    for result in results:
        print(
            f"{result['timesteps']:>7,} | "
            f"Success {result['success_rate']:>5.1f}% | "
            f"Steps {result['avg_steps']:>6.2f} | "
            f"Reward {result['avg_reward']:>6.2f}"
        )

    print("\n=== 기존 Baseline과 비교 ===")
    print("기존 PPO 200k : Success 84.4% | Steps 20.65")
    print(
        "Revisit 200k  : "
        f"Success {results[-1]['success_rate']:.1f}% | "
        f"Steps {results[-1]['avg_steps']:.2f}"
    )