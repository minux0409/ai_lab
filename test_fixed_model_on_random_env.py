from stable_baselines3 import PPO

from gridworld_random_env import RandomGridWorldEnv


def evaluate_model(model, episodes=100):
    env = RandomGridWorldEnv()

    success_count = 0
    fail_count = 0
    timeout_count = 0
    total_reward = 0.0
    total_steps = 0

    for _ in range(episodes):
        observation, _ = env.reset()

        episode_reward = 0.0
        episode_steps = 0

        while True:
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            episode_steps += 1

            if terminated or truncated:
                if reward == 10.0:
                    success_count += 1
                elif reward == -10.0:
                    fail_count += 1
                else:
                    timeout_count += 1

                break

        total_reward += episode_reward
        total_steps += episode_steps

    env.close()

    print("\n=== 고정맵 PPO → 랜덤맵 평가 ===")
    print(f"성공률: {success_count / episodes * 100:.1f}%")
    print(f"적 충돌률: {fail_count / episodes * 100:.1f}%")
    print(f"타임아웃률: {timeout_count / episodes * 100:.1f}%")
    print(f"평균 보상: {total_reward / episodes:.2f}")
    print(f"평균 행동 수: {total_steps / episodes:.2f}")


if __name__ == "__main__":
    model = PPO.load("models/gridworld_ppo")

    evaluate_model(
        model,
        episodes=1000,
    )