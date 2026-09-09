from stable_baselines3 import PPO

from risk_tradeoff_env import RiskTradeoffEnv


def train_model(danger_penalty, model_name):
    env = RiskTradeoffEnv(danger_penalty=danger_penalty)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=0.0003,
    )

    print(f"\n학습 시작: danger_penalty = {danger_penalty}")

    model.learn(
        total_timesteps=100_000,
    )

    model.save(model_name)

    env.close()

    print("학습 완료")

    return model


def evaluate_model(model, danger_penalty):
    env = RiskTradeoffEnv(danger_penalty=danger_penalty)

    obs, _ = env.reset()

    path = [tuple(env.agent_pos)]
    total_reward = 0.0
    danger_count = 0
    steps = 0

    while True:
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, _ = env.step(action)

        pos = tuple(env.agent_pos)

        path.append(pos)
        total_reward += reward
        steps += 1

        if pos in env.danger_zones:
            danger_count += 1

        if terminated or truncated:
            break

    env.close()

    return {
        "path": path,
        "steps": steps,
        "danger_count": danger_count,
        "reward": total_reward,
        "success": terminated,
    }


def print_result(title, result):
    print(f"\n=== {title} ===")
    print("경로:")
    print(" -> ".join(str(p) for p in result["path"]))
    print(f"총 행동 수: {result['steps']}")
    print(f"위험지역 진입 횟수: {result['danger_count']}")
    print(f"총 Reward: {result['reward']:.2f}")
    print(
        "결과:",
        "Goal 도착 성공" if result["success"] else "실패"
    )


if __name__ == "__main__":
    # 위험을 크게 싫어하는 Agent
    safe_model = train_model(
        danger_penalty=-1.0,
        model_name="models/risk_safe_ppo",
    )

    safe_result = evaluate_model(
        safe_model,
        danger_penalty=-1.0,
    )

    print_result(
        "위험 패널티 -1.0",
        safe_result,
    )

    # 위험을 거의 신경쓰지 않는 Agent
    fast_model = train_model(
        danger_penalty=-0.05,
        model_name="models/risk_fast_ppo",
    )

    fast_result = evaluate_model(
        fast_model,
        danger_penalty=-0.05,
    )

    print_result(
        "위험 패널티 -0.05",
        fast_result,
    )