import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


# 1. 환경 생성
env = gym.make("CartPole-v1")

# 2. 학습 전 모델 생성
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
)

# 3. 학습 전 평가
mean_reward_before, std_reward_before = evaluate_policy(
    model,
    env,
    n_eval_episodes=10,
    deterministic=True,
)

print("\n=== 학습 전 평가 ===")
print(f"평균 보상: {mean_reward_before:.2f}")
print(f"표준편차: {std_reward_before:.2f}")

# 4. PPO 학습
print("\n=== 학습 시작 ===")
model.learn(total_timesteps=30_000)

# 5. 학습 후 평가
mean_reward_after, std_reward_after = evaluate_policy(
    model,
    env,
    n_eval_episodes=10,
    deterministic=True,
)

print("\n=== 학습 후 평가 ===")
print(f"평균 보상: {mean_reward_after:.2f}")
print(f"표준편차: {std_reward_after:.2f}")

# 6. 모델 저장
model.save("models/cartpole_ppo")

print("\n모델 저장 완료: models/cartpole_ppo.zip")

env.close()