import numpy as np
from stable_baselines3 import PPO

from gridworld_env import GridWorldEnv


def evaluate_random_agent(episodes=100):
    env = GridWorldEnv()

    success_count = 0
    fail_count = 0
    total_reward = 0.0
    total_steps = 0

    for _ in range(episodes):
        observation, _ = env.reset()

        episode_reward = 0.0
        episode_steps = 0

        while True:
            action = env.action_space.sample()

            observation, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            episode_steps += 1

            if terminated or truncated:
                if reward == 10.0:
                    success_count += 1
                elif reward == -10.0:
                    fail_count += 1

                break

        total_reward += episode_reward
        total_steps += episode_steps

    env.close()

    return {
        "success_rate": success_count / episodes,
        "fail_rate": fail_count / episodes,
        "avg_reward": total_reward / episodes,
        "avg_steps": total_steps / episodes,
    }


def evaluate_ppo_agent(model, episodes=100):
    env = GridWorldEnv()

    success_count = 0
    fail_count = 0
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

                break

        total_reward += episode_reward
        total_steps += episode_steps

    env.close()

    return {
        "success_rate": success_count / episodes,
        "fail_rate": fail_count / episodes,
        "avg_reward": total_reward / episodes,
        "avg_steps": total_steps / episodes,
    }


def print_result(title, result):
    print(f"\n=== {title} ===")
    print(f"성공률: {result['success_rate'] * 100:.1f}%")
    print(f"적 충돌률: {result['fail_rate'] * 100:.1f}%")
    print(f"평균 보상: {result['avg_reward']:.2f}")
    print(f"평균 행동 수: {result['avg_steps']:.2f}")


def show_ppo_play(model):
    env = GridWorldEnv()

    observation, _ = env.reset()

    print("\n=== 학습된 PPO Agent 실제 플레이 ===")
    env.render()

    step = 0

    while True:
        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        observation, reward, terminated, truncated, _ = env.step(action)

        step += 1

        print(
            f"Step {step} | "
            f"Action: {int(action)} | "
            f"Reward: {reward}"
        )

        env.render()

        if terminated or truncated:
            if reward == 10.0:
                print("Goal 도착 성공!")
            elif reward == -10.0:
                print("Enemy 충돌 실패!")
            else:
                print("최대 Step 초과")

            break

    env.close()


if __name__ == "__main__":
    print("Random Agent 평가 중...")

    random_result = evaluate_random_agent(episodes=100)
    print_result("Random Agent", random_result)

    print("\nPPO 학습 시작...")

    env = GridWorldEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
    )

    model.learn(total_timesteps=50_000)

    model.save("models/gridworld_ppo")

    env.close()

    print("\nPPO 학습 완료!")

    ppo_result = evaluate_ppo_agent(
        model,
        episodes=100,
    )

    print_result("PPO Agent", ppo_result)

    show_ppo_play(model)