import numpy as np
import torch
import torch.nn as nn

from torch.distributions import Categorical

from partial_obstacle_env import PartialObstacleGridEnv


SEED = 42

TOTAL_TIMESTEPS = 100_000
ROLLOUT_STEPS = 512

GAMMA = 0.99
GAE_LAMBDA = 0.95

CLIP_RANGE = 0.2

LEARNING_RATE = 3e-4
TRAIN_EPOCHS = 10

VALUE_COEF = 0.5
ENTROPY_COEF = 0.01

EVAL_EPISODES = 500
EVAL_INTERVAL = 10_000


class ActorCritic(nn.Module):

    def __init__(
        self,
        observation_size,
        action_size,
    ):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(
                observation_size,
                128,
            ),
            nn.Tanh(),

            nn.Linear(
                128,
                128,
            ),
            nn.Tanh(),

            nn.Linear(
                128,
                action_size,
            ),
        )

        self.critic = nn.Sequential(
            nn.Linear(
                observation_size,
                128,
            ),
            nn.Tanh(),

            nn.Linear(
                128,
                128,
            ),
            nn.Tanh(),

            nn.Linear(
                128,
                1,
            ),
        )

    def forward(
        self,
        observation,
    ):

        logits = self.actor(
            observation
        )

        value = self.critic(
            observation
        )

        return (
            logits,
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

        self.advantages = None
        self.returns = None


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
            range(
                len(self.rewards)
            )
        ):

            if step == (
                len(self.rewards) - 1
            ):
                next_value = (
                    last_value
                )
            else:
                next_value = (
                    self.values[
                        step + 1
                    ]
                )

            next_non_terminal = (
                1.0
                - float(
                    self.dones[
                        step
                    ]
                )
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

            advantages[
                step
            ] = gae

        values = np.asarray(
            self.values,
            dtype=np.float32,
        )

        self.advantages = (
            advantages
        )

        self.returns = (
            advantages
            + values
        )


def select_action(
    model,
    observation,
    deterministic=False,
):

    observation_tensor = (
        torch.tensor(
            observation,
            dtype=torch.float32,
        )
        .unsqueeze(0)
    )

    with torch.no_grad():

        logits, value = model(
            observation_tensor
        )

        distribution = Categorical(
            logits=logits
        )

        if deterministic:

            action_tensor = (
                torch.argmax(
                    logits,
                    dim=-1,
                )
            )

        else:

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

    observation_tensor = (
        torch.tensor(
            observation,
            dtype=torch.float32,
        )
        .unsqueeze(0)
    )

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
            observation,
            action,
            reward,
            log_probability,
            value,
            done,
        )

        observation = (
            next_observation
        )

        if done:

            observation, _ = (
                env.reset()
            )

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

    old_log_probabilities = (
        torch.tensor(
            buffer.log_probabilities,
            dtype=torch.float32,
        )
    )

    advantages = torch.tensor(
        buffer.advantages,
        dtype=torch.float32,
    )

    returns = torch.tensor(
        buffer.returns,
        dtype=torch.float32,
    )

    advantages = (
        advantages
        - advantages.mean()
    ) / (
        advantages.std()
        + 1e-8
    )

    final_actor_loss = 0.0
    final_critic_loss = 0.0
    final_entropy = 0.0
    final_clip_fraction = 0.0

    for _ in range(
        TRAIN_EPOCHS
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

        ratios = torch.exp(
            new_log_probabilities
            - old_log_probabilities
        )

        objective_1 = (
            ratios
            * advantages
        )

        clipped_ratios = (
            torch.clamp(
                ratios,
                1.0
                - CLIP_RANGE,
                1.0
                + CLIP_RANGE,
            )
        )

        objective_2 = (
            clipped_ratios
            * advantages
        )

        actor_loss = (
            -torch.min(
                objective_1,
                objective_2,
            )
            .mean()
        )

        critic_loss = (
            (
                values
                - returns
            ) ** 2
        ).mean()

        total_loss = (
            actor_loss
            + VALUE_COEF
            * critic_loss
            - ENTROPY_COEF
            * entropy
        )

        optimizer.zero_grad()

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=0.5,
        )

        optimizer.step()

        with torch.no_grad():

            clip_fraction = (
                (
                    torch.abs(
                        ratios
                        - 1.0
                    )
                    > CLIP_RANGE
                )
                .float()
                .mean()
                .item()
            )

        final_actor_loss = (
            actor_loss.item()
        )

        final_critic_loss = (
            critic_loss.item()
        )

        final_entropy = (
            entropy.item()
        )

        final_clip_fraction = (
            clip_fraction
        )

    return (
        final_actor_loss,
        final_critic_loss,
        final_entropy,
        final_clip_fraction,
    )


def evaluate(
    model,
):

    env = PartialObstacleGridEnv()

    successes = 0
    total_steps = 0

    for episode in range(
        EVAL_EPISODES
    ):

        observation, _ = env.reset(
            seed=10_000 + episode
        )

        episode_steps = 0

        while True:

            (
                action,
                _,
                _,
            ) = select_action(
                model,
                observation,
                deterministic=True,
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(
                action
            )

            episode_steps += 1

            if terminated:

                successes += 1

            if (
                terminated
                or truncated
            ):

                break

        total_steps += (
            episode_steps
        )

    env.close()

    success_rate = (
        successes
        / EVAL_EPISODES
        * 100.0
    )

    average_steps = (
        total_steps
        / EVAL_EPISODES
    )

    return (
        success_rate,
        average_steps,
    )


def main():

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    env = (
        PartialObstacleGridEnv()
    )

    observation, _ = env.reset(
        seed=SEED
    )

    observation_size = (
        env.observation_space
        .shape[0]
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
        "=== PPO From Scratch Training ==="
    )

    print()

    print(
        f"Total Timesteps: "
        f"{TOTAL_TIMESTEPS}"
    )

    print(
        f"Rollout Steps: "
        f"{ROLLOUT_STEPS}"
    )

    print(
        f"Training Epochs: "
        f"{TRAIN_EPOCHS}"
    )

    print()

    # -----------------------------------------
    # 학습 전 성능
    # -----------------------------------------

    (
        success_rate,
        average_steps,
    ) = evaluate(
        model
    )

    print(
        f"{0:>7} steps | "
        f"Success "
        f"{success_rate:6.2f}% | "
        f"Avg Steps "
        f"{average_steps:6.2f}"
    )

    total_steps = 0
    next_evaluation = (
        EVAL_INTERVAL
    )

    while total_steps < (
        TOTAL_TIMESTEPS
    ):

        (
            buffer,
            observation,
        ) = collect_rollout(
            env,
            model,
            observation,
        )

        (
            actor_loss,
            critic_loss,
            entropy,
            clip_fraction,
        ) = train_ppo(
            model,
            optimizer,
            buffer,
        )

        total_steps += (
            ROLLOUT_STEPS
        )

        if (
            total_steps
            >= next_evaluation
        ):

            (
                success_rate,
                average_steps,
            ) = evaluate(
                model
            )

            print(
                f"{total_steps:>7} steps | "
                f"Success "
                f"{success_rate:6.2f}% | "
                f"Avg Steps "
                f"{average_steps:6.2f} | "
                f"Actor "
                f"{actor_loss:7.3f} | "
                f"Critic "
                f"{critic_loss:7.3f} | "
                f"Entropy "
                f"{entropy:6.3f} | "
                f"Clip "
                f"{clip_fraction * 100:5.1f}%"
            )

            next_evaluation += (
                EVAL_INTERVAL
            )

    env.close()

    torch.save(
        model.state_dict(),
        "models/ppo_from_scratch.pt",
    )

    print()
    print(
        "Model saved: "
        "models/ppo_from_scratch.pt"
    )


if __name__ == "__main__":
    main()