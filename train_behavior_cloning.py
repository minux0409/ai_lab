import os
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from partial_obstacle_env import PartialObstacleGridEnv


DATASET_PATH = "datasets/bc_expert_dataset.npz"
MODEL_PATH = "models/behavior_cloning.pt"

SEED = 42
EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

EVAL_EPISODES = 500
EVAL_SEED_START = 10_000


# ---------------------------------------------------------
# Seed
# ---------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------
# Behavior Cloning Policy
# ---------------------------------------------------------

class BCPolicy(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_size),
        )

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

def load_dataset():
    data = np.load(DATASET_PATH)

    observations = data["observations"]
    actions = data["actions"]

    observations = torch.tensor(
        observations,
        dtype=torch.float32,
    )

    actions = torch.tensor(
        actions,
        dtype=torch.long,
    )

    dataset = TensorDataset(
        observations,
        actions,
    )

    return dataset, observations.shape[1]


# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

def train():
    set_seed(SEED)

    dataset, observation_size = load_dataset()

    action_size = 4

    train_size = int(
        len(dataset) * 0.9
    )

    validation_size = (
        len(dataset) - train_size
    )

    generator = torch.Generator().manual_seed(
        SEED
    )

    train_dataset, validation_dataset = random_split(
        dataset,
        [
            train_size,
            validation_size,
        ],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = BCPolicy(
        observation_size,
        action_size,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.CrossEntropyLoss()

    print()
    print("=== Behavior Cloning Training ===")
    print(f"Dataset: {len(dataset)}")
    print(f"Train: {train_size}")
    print(f"Validation: {validation_size}")
    print(f"Observation Size: {observation_size}")
    print(f"Action Size: {action_size}")
    print()

    for epoch in range(1, EPOCHS + 1):

        # -----------------------------
        # Train
        # -----------------------------

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for observations, actions in train_loader:

            optimizer.zero_grad()

            logits = model(
                observations
            )

            loss = criterion(
                logits,
                actions,
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item()
                * observations.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            train_correct += (
                predictions == actions
            ).sum().item()

            train_total += actions.size(0)

        train_loss /= train_total

        train_accuracy = (
            train_correct
            / train_total
            * 100.0
        )

        # -----------------------------
        # Validation
        # -----------------------------

        model.eval()

        validation_loss = 0.0
        validation_correct = 0
        validation_total = 0

        with torch.no_grad():

            for observations, actions in validation_loader:

                logits = model(
                    observations
                )

                loss = criterion(
                    logits,
                    actions,
                )

                validation_loss += (
                    loss.item()
                    * observations.size(0)
                )

                predictions = torch.argmax(
                    logits,
                    dim=1,
                )

                validation_correct += (
                    predictions == actions
                ).sum().item()

                validation_total += (
                    actions.size(0)
                )

        validation_loss /= (
            validation_total
        )

        validation_accuracy = (
            validation_correct
            / validation_total
            * 100.0
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_accuracy:6.2f}% | "
            f"Val Loss {validation_loss:.4f} | "
            f"Val Acc {validation_accuracy:6.2f}%"
        )

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "observation_size":
                observation_size,

            "action_size":
                action_size,
        },
        MODEL_PATH,
    )

    print()
    print(
        f"Model saved: {MODEL_PATH}"
    )

    return model


# ---------------------------------------------------------
# Environment Evaluation
# ---------------------------------------------------------

def evaluate(model):

    env = PartialObstacleGridEnv()

    model.eval()

    successes = 0
    timeouts = 0

    total_steps = 0

    with torch.no_grad():

        for episode in range(
            EVAL_EPISODES
        ):

            seed = (
                EVAL_SEED_START
                + episode
            )

            observation, _ = env.reset(
                seed=seed
            )

            terminated = False
            truncated = False

            steps = 0

            while (
                not terminated
                and not truncated
            ):

                obs_tensor = torch.tensor(
                    observation,
                    dtype=torch.float32,
                ).unsqueeze(0)

                logits = model(
                    obs_tensor
                )

                action = torch.argmax(
                    logits,
                    dim=1,
                ).item()

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = env.step(action)

                steps += 1

            if terminated:
                successes += 1
            else:
                timeouts += 1

            total_steps += steps

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

    print()
    print("=== Behavior Cloning Evaluation ===")

    print(
        f"Episodes: {EVAL_EPISODES}"
    )

    print(
        f"Success: {successes}"
    )

    print(
        f"Timeout: {timeouts}"
    )

    print(
        f"Success Rate: "
        f"{success_rate:.2f}%"
    )

    print(
        f"Average Steps: "
        f"{average_steps:.2f}"
    )

    return (
        success_rate,
        average_steps,
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    model = train()

    evaluate(model)