import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from partial_obstacle_cnn_env import PartialObstacleCnnEnv


SEED = 42
TOTAL_TIMESTEPS = 200_000
EVAL_EPISODES = 500
MODEL_PATH = "models/partial_obstacle_cnn_ppo"


class GridCNN(BaseFeaturesExtractor):
    """
    9x9 grid 전용 CNN.
    SB3 기본 NatureCNN은 Atari 크기 영상용이라 9x9에 맞지 않으므로
    작은 kernel을 사용하는 feature extractor를 명시적으로 사용한다.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(
                observation_space.sample()[None]
            ).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def make_env():
    return PartialObstacleCnnEnv()


def evaluate(model, episodes=EVAL_EPISODES):
    successes = 0
    success_steps = []
    failure_steps = []

    for episode in range(episodes):
        env = make_env()
        obs, _ = env.reset(seed=10_000 + episode)

        for step in range(1, env.max_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))

            if terminated:
                successes += 1
                success_steps.append(step)
                break

            if truncated:
                failure_steps.append(step)
                break

        env.close()

    return (
        successes / episodes * 100.0,
        float(np.mean(success_steps)) if success_steps else 0.0,
        float(np.mean(failure_steps)) if failure_steps else 0.0,
    )


def evaluate_random(episodes=EVAL_EPISODES):
    rng = np.random.default_rng(SEED)
    successes = 0
    success_steps = []
    failure_steps = []

    for episode in range(episodes):
        env = make_env()
        _, _ = env.reset(seed=10_000 + episode)

        for step in range(1, env.max_steps + 1):
            action = int(rng.integers(0, env.action_space.n))
            _, _, terminated, truncated, _ = env.step(action)

            if terminated:
                successes += 1
                success_steps.append(step)
                break

            if truncated:
                failure_steps.append(step)
                break

        env.close()

    return (
        successes / episodes * 100.0,
        float(np.mean(success_steps)) if success_steps else 0.0,
        float(np.mean(failure_steps)) if failure_steps else 0.0,
    )


def print_result(label, result):
    success, success_steps, failure_steps = result
    print(
        f"{label:<9} | Success: {success:6.2f}%"
        f" | Success Steps: {success_steps:6.2f}"
        f" | Failure Steps: {failure_steps:6.2f}"
    )


def main():
    print("=== Experiment H: Spatial Map + CNN ===")
    print("Observation      : 5 x 9 x 9 spatial channels")
    print("Channels         : unknown / obstacle / visits / agent / goal")
    print("Policy           : PPO + custom small CNN")
    print("Step             : -1")
    print("Collision        : additional -2 (total -3)")
    print("First Visit      : +1 (once per tile)")
    print("New discovered   : +0.1 / cell")
    print("Goal             : +100")
    print("Training         : 200,000")
    print("Evaluation       : 500 fixed seeds")
    print()

    print_result("Random", evaluate_random())

    policy_kwargs = dict(
        features_extractor_class=GridCNN,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    env = make_env()
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        seed=SEED,
        verbose=0,
    )

    print_result("0", evaluate(model))

    checkpoints = [10_000, 20_000, 50_000, 100_000, 200_000]
    trained = 0

    for checkpoint in checkpoints:
        amount = checkpoint - trained
        model.learn(
            total_timesteps=amount,
            reset_num_timesteps=False,
        )
        trained = checkpoint
        print_result(f"{checkpoint:,}", evaluate(model))

    model.save(MODEL_PATH)
    env.close()
    print()
    print("Model saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
