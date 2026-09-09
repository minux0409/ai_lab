from stable_baselines3 import PPO

from random_sparse_gridworld_env import RandomSparseGridWorldEnv


CHECKPOINTS = [
    0,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
]


def evaluate(model, episodes=500):
    env = RandomSparseGridWorldEnv()

    success = 0
    timeout = 0
    total_reward = 0.0
    total_steps = 0

    for episode in range(episodes):
        # 평가 조건을 checkpoint마다 동일하게 재현하기 위해
        # episode 번호를 seed로 사용
        obs, _ = env.reset(seed=episode)

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
                else:
                    timeout += 1

                break

        total_reward += episode_reward
        total_steps += steps

    env.close()

    return {
        "success_rate": success / episodes * 100,
        "timeout_rate": timeout / episodes * 100,
        "avg_reward": total_reward / episodes,
        "avg_steps": total_steps / episodes,
    }


def print_result(timesteps, result):
    print(f"\n=== {timesteps:,} timestep ===")
    print(f"성공률: {result['success_rate']:.1f}%")
    print(f"타임아웃률: {result['timeout_rate']:.1f}%")
    print(f"평균 보상: {result['avg_reward']:.2f}")
    print(f"평균 행동 수: {result['avg_steps']:.2f}")


if __name__ == "__main__":
    train_env = RandomSparseGridWorldEnv()

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
        additional_steps = checkpoint - previous_checkpoint

        if additional_steps > 0:
            print(
                f"\n{previous_checkpoint:,} "
                f"→ {checkpoint:,} timestep 학습..."
            )

            model.learn(
                total_timesteps=additional_steps,
                reset_num_timesteps=False,
            )

        result = evaluate(
            model,
            episodes=500,
        )

        print_result(
            checkpoint,
            result,
        )

        results.append(
            {
                "timesteps": checkpoint,
                **result,
            }
        )

        previous_checkpoint = checkpoint

    model.save(
        "models/random_sparse_gridworld_ppo"
    )

    train_env.close()

    print("\n=== 전체 학습 결과 ===")

    for result in results:
        print(
            f"{result['timesteps']:>7,} | "
            f"Success {result['success_rate']:>5.1f}% | "
            f"Reward {result['avg_reward']:>5.2f} | "
            f"Steps {result['avg_steps']:>5.2f}"
        )