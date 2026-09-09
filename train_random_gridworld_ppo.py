from stable_baselines3 import PPO

from gridworld_random_env import RandomGridWorldEnv


def evaluate(model, episodes=1000):
    env = RandomGridWorldEnv()

    success = 0
    collision = 0
    timeout = 0

    total_reward = 0.0
    total_steps = 0

    for _ in range(episodes):
        observation, _ = env.reset()

        episode_reward = 0.0
        steps = 0

        while True:
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            steps += 1

            if terminated or truncated:
                if reward == 10.0:
                    success += 1
                elif reward == -10.0:
                    collision += 1
                else:
                    timeout += 1

                break

        total_reward += episode_reward
        total_steps += steps

    env.close()

    print("\n=== 랜덤맵 학습 PPO 평가 ===")
    print(f"성공률: {success / episodes * 100:.1f}%")
    print(f"적 충돌률: {collision / episodes * 100:.1f}%")
    print(f"타임아웃률: {timeout / episodes * 100:.1f}%")
    print(f"평균 보상: {total_reward / episodes:.2f}")
    print(f"평균 행동 수: {total_steps / episodes:.2f}")


if __name__ == "__main__":
    env = RandomGridWorldEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
    )

    print("랜덤맵 PPO 학습 시작...")

    model.learn(
        total_timesteps=200_000,
    )

    model.save("models/gridworld_random_ppo")

    print("\n모델 저장 완료!")

    env.close()

    evaluate(
        model,
        episodes=1000,
    )