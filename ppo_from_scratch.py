import numpy as np
import torch
import torch.nn as nn

from torch.distributions import Categorical

from partial_obstacle_env import PartialObstacleGridEnv


SEED = 42
ROLLOUT_STEPS = 128

GAMMA = 0.99
GAE_LAMBDA = 0.95

CLIP_RANGE = 0.2

LEARNING_RATE = 3e-4
TRAIN_EPOCHS = 10

VALUE_COEF = 0.5
ENTROPY_COEF = 0.01


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_size,
        action_size,
    ):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_size),
        )

        self.critic = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        observation,
    ):
        action_logits = self.actor(
            observation
        )

        value = self.critic(
            observation
        )

        return (
            action_logits,
            value,
        )


class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probabilities = []
        self.values = []
        self.dones = []

        self.advantages = []
        self.returns = []

    def add(
        self,
        observation,
        action,
        reward,
        log_probability,
        value,
        done,
    ):
        self.observations.append(
            observation.copy()
        )

        self.actions.append(
            action
        )

        self.rewards.append(
            reward
        )

        self.log_probabilities.append(
            log_probability
        )

        self.values.append(
            value
        )

        self.dones.append(
            done
        )

    def compute_gae(
        self,
        last_value,
    ):
        advantages = np.zeros(
            len(self.rewards),
            dtype=np.float32,
        )

        gae = 0.0

        for step in reversed(
            range(len(self.rewards))
        ):
            if step == len(self.rewards) - 1:
                next_value = last_value
            else:
                next_value = self.values[
                    step + 1
                ]

            next_non_terminal = (
                1.0
                - float(self.dones[step])
            )

            delta = (
                self.rewards[step]
                + GAMMA
                * next_value
                * next_non_terminal
                - self.values[step]
            )

            gae = (
                delta
                + GAMMA
                * GAE_LAMBDA
                * next_non_terminal
                * gae
            )

            advantages[step] = gae

        values = np.asarray(
            self.values,
            dtype=np.float32,
        )

        self.advantages = advantages
        self.returns = (
            advantages
            + values
        )


def select_action(
    model,
    observation,
):
    observation_tensor = torch.tensor(
        observation,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        logits, value = model(
            observation_tensor
        )

        distribution = Categorical(
            logits=logits
        )

        action_tensor = (
            distribution.sample()
        )

        log_probability = (
            distribution.log_prob(
                action_tensor
            )
        )

    return (
        action_tensor.item(),
        log_probability.item(),
        value.item(),
    )


def get_value(
    model,
    observation,
):
    observation_tensor = torch.tensor(
        observation,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        _, value = model(
            observation_tensor
        )

    return value.item()


def collect_rollout(
    env,
    model,
    observation,
):
    buffer = RolloutBuffer()

    episode_rewards = []
    current_episode_reward = 0.0

    for _ in range(
        ROLLOUT_STEPS
    ):
        (
            action,
            log_probability,
            value,
        ) = select_action(
            model,
            observation,
        )

        (
            next_observation,
            reward,
            terminated,
            truncated,
            _,
        ) = env.step(
            action
        )

        done = (
            terminated
            or truncated
        )

        buffer.add(
            observation=observation,
            action=action,
            reward=reward,
            log_probability=log_probability,
            value=value,
            done=done,
        )

        current_episode_reward += reward

        observation = (
            next_observation
        )

        if done:
            episode_rewards.append(
                current_episode_reward
            )

            current_episode_reward = 0.0

            observation, _ = env.reset()

    last_value = get_value(
        model,
        observation,
    )

    buffer.compute_gae(
        last_value
    )

    return (
        buffer,
        observation,
        episode_rewards,
    )


def train_ppo(
    model,
    optimizer,
    buffer,
):
    observations = torch.tensor(
        np.asarray(
            buffer.observations
        ),
        dtype=torch.float32,
    )

    actions = torch.tensor(
        buffer.actions,
        dtype=torch.long,
    )

    old_log_probabilities = torch.tensor(
        buffer.log_probabilities,
        dtype=torch.float32,
    )

    returns = torch.tensor(
        buffer.returns,
        dtype=torch.float32,
    )

    advantages = torch.tensor(
        buffer.advantages,
        dtype=torch.float32,
    )

    advantages = (
        advantages
        - advantages.mean()
    ) / (
        advantages.std()
        + 1e-8
    )

    print()
    print(
        "=== PPO Training ==="
    )

    for epoch in range(
        1,
        TRAIN_EPOCHS + 1
    ):
        logits, values = model(
            observations
        )

        values = (
            values.squeeze(-1)
        )

        distribution = Categorical(
            logits=logits
        )

        new_log_probabilities = (
            distribution.log_prob(
                actions
            )
        )

        entropy = (
            distribution.entropy()
            .mean()
        )

        # ------------------------------------------
        # PPO Ratio
        # ------------------------------------------

        ratios = torch.exp(
            new_log_probabilities
            - old_log_probabilities
        )

        # ------------------------------------------
        # PPO Clipped Objective
        # ------------------------------------------

        objective_1 = (
            ratios
            * advantages
        )

        clipped_ratios = torch.clamp(
            ratios,
            1.0 - CLIP_RANGE,
            1.0 + CLIP_RANGE,
        )

        objective_2 = (
            clipped_ratios
            * advantages
        )

        actor_loss = -torch.min(
            objective_1,
            objective_2,
        ).mean()

        # ------------------------------------------
        # Critic Loss
        # ------------------------------------------

        critic_loss = (
            (
                values
                - returns
            ) ** 2
        ).mean()

        # ------------------------------------------
        # Total Loss
        #
        # Actor:
        # 정책 개선
        #
        # Critic:
        # Value 예측 개선
        #
        # Entropy:
        # 너무 빨리 한 Action만 선택하는 것 방지
        # ------------------------------------------

        total_loss = (
            actor_loss
            + VALUE_COEF
            * critic_loss
            - ENTROPY_COEF
            * entropy
        )

        optimizer.zero_grad()

        total_loss.backward()

        optimizer.step()

        # ------------------------------------------
        # 실제로 clipping 대상이 된 비율
        # ------------------------------------------

        with torch.no_grad():
            clip_fraction = (
                (
                    torch.abs(
                        ratios - 1.0
                    )
                    > CLIP_RANGE
                )
                .float()
                .mean()
                .item()
            )

            ratio_min = (
                ratios.min().item()
            )

            ratio_max = (
                ratios.max().item()
            )

        print(
            f"Epoch {epoch:02d} | "
            f"Actor {actor_loss.item():8.4f} | "
            f"Critic {critic_loss.item():8.4f} | "
            f"Entropy {entropy.item():7.4f} | "
            f"Ratio "
            f"{ratio_min:6.3f}~{ratio_max:6.3f} | "
            f"ClipFrac "
            f"{clip_fraction * 100:6.2f}%"
        )


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = PartialObstacleGridEnv()

    observation, _ = env.reset(
        seed=SEED
    )

    observation_size = (
        env.observation_space.shape[0]
    )

    action_size = (
        env.action_space.n
    )

    model = ActorCritic(
        observation_size,
        action_size,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print()
    print(
        "=== PPO From Scratch ==="
    )

    print(
        f"Observation Size: "
        f"{observation_size}"
    )

    print(
        f"Action Size: "
        f"{action_size}"
    )

    buffer, observation, episode_rewards = (
        collect_rollout(
            env,
            model,
            observation,
        )
    )

    print()
    print(
        f"Collected Rollout Steps: "
        f"{len(buffer.rewards)}"
    )

    print(
        f"Completed Episodes: "
        f"{len(episode_rewards)}"
    )

    if episode_rewards:
        print(
            f"Average Episode Reward: "
            f"{np.mean(episode_rewards):.2f}"
        )

    train_ppo(
        model,
        optimizer,
        buffer,
    )

    env.close()


if __name__ == "__main__":
    main()