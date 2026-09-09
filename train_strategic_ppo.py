from stable_baselines3 import PPO

from strategic_gridworld_env import StrategicGridWorldEnv


def evaluate_random(episodes=500):
    env = StrategicGridWorldEnv()

    success = 0
    timeout = 0
    total_reward = 0
    total_steps = 0
    total_danger = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0
        danger_count = 0
        steps = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)

            ep_reward += reward
            steps += 1

            if tuple(env.agent_pos) in env.danger_zones:
                danger_count += 1

            if terminated or truncated:
                if terminated:
                    success += 1
                else:
                    timeout += 1
                break

        total_reward += ep_reward
        total_steps += steps
        total_danger += danger_count

    env.close()

    return {
        "success": success / episodes * 100,
        "timeout": timeout / episodes * 100,
        "reward": total_reward / episodes,
        "steps": total_steps / episodes,
        "danger": total_danger / episodes,
    }


def evaluate_ppo(model, episodes=500):
    env = StrategicGridWorldEnv()

    success = 0
    timeout = 0
    total_reward = 0
    total_steps = 0
    total_danger = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0
        danger_count = 0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            ep_reward += reward
            steps += 1

            if tuple(env.agent_pos) in env.danger_zones:
                danger_count += 1

            if terminated or truncated:
                if terminated:
                    success += 1
                else:
                    timeout += 1
                break

        total_reward += ep_reward
        total_steps += steps
        total_danger += danger_count

    env.close()

    return {
        "success": success / episodes * 100,
        "timeout": timeout / episodes * 100,
        "reward": total_reward / episodes,
        "steps": total_steps / episodes,
        "danger": total_danger / episodes,
    }


def print_result(name, result):
    print(f"\n=== {name} ===")
    print(f"성공률: {result['success']:.1f}%")
    print(f"타임아웃률: {result['timeout']:.1f}%")
    print(f"평균 보상: {result['reward']:.2f}")
    print(f"평균 행동 수: {result['steps']:.2f}")
    print(f"평균 위험지역 진입: {result['danger']:.2f}")


def show_play(model):
    env = StrategicGridWorldEnv()

    obs, _ = env.reset()

    path = [tuple(env.agent_pos)]
    total_reward = 0
    danger_count = 0

    print("\n=== PPO 실제 플레이 ===")
    env.render()

    while True:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)

        position = tuple(env.agent_pos)
        path.append(position)

        total_reward += reward

        if position in env.danger_zones:
            danger_count += 1

        if terminated or truncated:
            break

    print("이동 경로:")
    print(" -> ".join(str(p) for p in path))

    print(f"\n총 행동 수: {len(path) - 1}")
    print(f"위험지역 진입 횟수: {danger_count}")
    print(f"총 Reward: {total_reward:.2f}")

    if terminated:
        print("Goal 도착 성공!")
    else:
        print("시간 초과!")

    print("\n최종 상태:")
    env.render()

    env.close()


if __name__ == "__main__":
    # 1. Baseline
    print("Random Agent 평가...")
    random_result = evaluate_random()
    print_result("Random Agent", random_result)

    # 2. PPO
    env = StrategicGridWorldEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
    )

    print("\nPPO 학습 시작...")

    model.learn(
        total_timesteps=200_000,
    )

    model.save("models/strategic_gridworld_ppo")
    env.close()

    print("\nPPO 학습 완료!")

    # 3. Evaluation
    ppo_result = evaluate_ppo(model)
    print_result("PPO Agent", ppo_result)

    # 4. 실제 행동 확인
    show_play(model)