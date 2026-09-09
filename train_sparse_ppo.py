from stable_baselines3 import PPO

from sparse_gridworld_env import SparseGridWorldEnv


def evaluate_random(episodes=500):
    env = SparseGridWorldEnv()

    success = 0
    timeout = 0
    total_reward = 0.0
    total_steps = 0

    for _ in range(episodes):
        obs, _ = env.reset()

        ep_reward = 0.0
        steps = 0

        while True:
            action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)

            ep_reward += reward
            steps += 1

            if terminated or truncated:
                if terminated:
                    success += 1
                else:
                    timeout += 1
                break

        total_reward += ep_reward
        total_steps += steps

    env.close()

    return {
        "success_rate": success / episodes * 100,
        "timeout_rate": timeout / episodes * 100,
        "avg_reward": total_reward / episodes,
        "avg_steps": total_steps / episodes,
    }


def evaluate_ppo(model, episodes=500):
    env = SparseGridWorldEnv()

    success = 0
    timeout = 0
    total_reward = 0.0
    total_steps = 0

    for _ in range(episodes):
        obs, _ = env.reset()

        ep_reward = 0.0
        steps = 0

        while True:
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, _ = env.step(action)

            ep_reward += reward
            steps += 1

            if terminated or truncated:
                if terminated:
                    success += 1
                else:
                    timeout += 1
                break

        total_reward += ep_reward
        total_steps += steps

    env.close()

    return {
        "success_rate": success / episodes * 100,
        "timeout_rate": timeout / episodes * 100,
        "avg_reward": total_reward / episodes,
        "avg_steps": total_steps / episodes,
    }


def print_result(title, result):
    print(f"\n=== {title} ===")
    print(f"성공률: {result['success_rate']:.1f}%")
    print(f"타임아웃률: {result['timeout_rate']:.1f}%")
    print(f"평균 보상: {result['avg_reward']:.2f}")
    print(f"평균 행동 수: {result['avg_steps']:.2f}")


if __name__ == "__main__":
    print("Random Agent 평가 시작...")

    random_result = evaluate_random()
    print_result(
        "Random Agent",
        random_result,
    )

    env = SparseGridWorldEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
    )

    print("\nSparse Reward PPO 학습 시작...")

    model.learn(
        total_timesteps=100_000,
    )

    model.save(
        "models/sparse_gridworld_ppo"
    )

    env.close()

    print("\n학습 완료!")

    ppo_result = evaluate_ppo(
        model,
        episodes=500,
    )

    print_result(
        "Sparse Reward PPO",
        ppo_result,
    )