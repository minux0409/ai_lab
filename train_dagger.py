from collections import deque
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from partial_obstacle_env import PartialObstacleGridEnv


BASE_DATASET_PATH = "datasets/bc_expert_dataset.npz"
BC_MODEL_PATH = "models/behavior_cloning.pt"
DAGGER_MODEL_PATH = "models/dagger_policy.pt"

SEED = 42

DAGGER_ITERATIONS = 5
COLLECT_EPISODES = 500

EPOCHS_PER_ITERATION = 10
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

EVAL_EPISODES = 500
EVAL_SEED_START = 10_000

ACTION_DELTAS = {
    0: (0, -1),   # UP
    1: (0, 1),    # DOWN
    2: (-1, 0),   # LEFT
    3: (1, 0),    # RIGHT
}


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_bc_model():
    checkpoint = torch.load(
        BC_MODEL_PATH,
        map_location="cpu",
    )

    model = BCPolicy(
        checkpoint["observation_size"],
        checkpoint["action_size"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def get_neighbors(env, position):
    x, y = position

    result = []

    for action, (dx, dy) in ACTION_DELTAS.items():
        nx = x + dx
        ny = y + dy

        if not (
            0 <= nx < env.grid_size
            and 0 <= ny < env.grid_size
        ):
            continue

        if env.grid[ny, nx] == 1:
            continue

        result.append(
            ((nx, ny), action)
        )

    return result


def expert_action_from_current_state(env):
    """
    현재 BC가 도달한 위치에서 Goal까지
    BFS 최단경로의 첫 번째 Action을 반환한다.
    """

    start = tuple(env.agent_pos)
    goal = tuple(env.goal_pos)

    if start == goal:
        return None

    queue = deque([start])

    parent = {
        start: None
    }

    parent_action = {}

    while queue:
        current = queue.popleft()

        if current == goal:
            break

        for next_pos, action in get_neighbors(
            env,
            current,
        ):
            if next_pos in parent:
                continue

            parent[next_pos] = current
            parent_action[next_pos] = action

            queue.append(next_pos)

    if goal not in parent:
        return None

    current = goal

    actions = []

    while current != start:
        actions.append(
            parent_action[current]
        )

        current = parent[current]

    actions.reverse()

    return actions[0]


def predict_action(model, observation):
    obs_tensor = torch.tensor(
        observation,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model(obs_tensor)

        return torch.argmax(
            logits,
            dim=1,
        ).item()


def collect_dagger_data(
    model,
    iteration,
):
    """
    중요한 부분.

    환경에서는 현재 학습된 Agent 행동을 실행한다.
    하지만 그 Agent가 방문한 상태마다
    BFS Expert에게 정답 Action을 질문한다.
    """

    env = PartialObstacleGridEnv()

    observations = []
    expert_actions = []

    agent_successes = 0

    seed_start = (
        30_000
        + iteration * COLLECT_EPISODES
    )

    for episode in range(
        COLLECT_EPISODES
    ):
        observation, _ = env.reset(
            seed=seed_start + episode
        )

        terminated = False
        truncated = False

        while (
            not terminated
            and not truncated
        ):
            expert_action = (
                expert_action_from_current_state(
                    env
                )
            )

            if expert_action is None:
                break

            # Agent가 실제 방문한 Observation에
            # Expert label을 붙인다.
            observations.append(
                observation.copy()
            )

            expert_actions.append(
                expert_action
            )

            # 실제 환경에서는 Agent 행동 실행
            agent_action = predict_action(
                model,
                observation,
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(agent_action)

        if terminated:
            agent_successes += 1

    env.close()

    observations = np.asarray(
        observations,
        dtype=np.float32,
    )

    expert_actions = np.asarray(
        expert_actions,
        dtype=np.int64,
    )

    success_rate = (
        agent_successes
        / COLLECT_EPISODES
        * 100.0
    )

    print(
        f"Collected Samples: "
        f"{len(expert_actions)}"
    )

    print(
        f"Collection Agent Success: "
        f"{success_rate:.2f}%"
    )

    return (
        observations,
        expert_actions,
    )


def train_model(
    model,
    observations,
    actions,
):
    dataset = TensorDataset(
        torch.tensor(
            observations,
            dtype=torch.float32,
        ),
        torch.tensor(
            actions,
            dtype=torch.long,
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(
        1,
        EPOCHS_PER_ITERATION + 1
    ):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_obs, batch_actions in loader:
            optimizer.zero_grad()

            logits = model(batch_obs)

            loss = criterion(
                logits,
                batch_actions,
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item()
                * batch_obs.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct += (
                predictions == batch_actions
            ).sum().item()

            total += batch_actions.size(0)

        print(
            f"  Epoch {epoch:02d} | "
            f"Loss {total_loss / total:.4f} | "
            f"Accuracy "
            f"{correct / total * 100:.2f}%"
        )


def evaluate(model):
    env = PartialObstacleGridEnv()

    successes = 0
    total_steps = 0

    model.eval()

    for episode in range(
        EVAL_EPISODES
    ):
        observation, _ = env.reset(
            seed=EVAL_SEED_START + episode
        )

        terminated = False
        truncated = False
        steps = 0

        while (
            not terminated
            and not truncated
        ):
            action = predict_action(
                model,
                observation,
            )

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

    return (
        success_rate,
        average_steps,
    )


def main():
    set_seed(SEED)

    base_data = np.load(
        BASE_DATASET_PATH
    )

    all_observations = (
        base_data["observations"].copy()
    )

    all_actions = (
        base_data["actions"].copy()
    )

    model = load_bc_model()

    print()
    print("=== Initial Behavior Cloning ===")

    success_rate, average_steps = evaluate(
        model
    )

    print(
        f"Success Rate: "
        f"{success_rate:.2f}%"
    )

    print(
        f"Average Steps: "
        f"{average_steps:.2f}"
    )

    print(
        f"Initial Dataset Size: "
        f"{len(all_actions)}"
    )

    results = [
        (
            0,
            success_rate,
            average_steps,
            len(all_actions),
        )
    ]

    for iteration in range(
        1,
        DAGGER_ITERATIONS + 1
    ):
        print()
        print("=" * 60)

        print(
            f"DAgger Iteration "
            f"{iteration}/{DAGGER_ITERATIONS}"
        )

        print("=" * 60)

        new_observations, new_actions = (
            collect_dagger_data(
                model,
                iteration,
            )
        )

        all_observations = np.concatenate(
            [
                all_observations,
                new_observations,
            ],
            axis=0,
        )

        all_actions = np.concatenate(
            [
                all_actions,
                new_actions,
            ],
            axis=0,
        )

        print(
            f"Aggregated Dataset Size: "
            f"{len(all_actions)}"
        )

        train_model(
            model,
            all_observations,
            all_actions,
        )

        success_rate, average_steps = (
            evaluate(model)
        )

        print()
        print(
            f"DAgger {iteration} Evaluation"
        )

        print(
            f"Success Rate: "
            f"{success_rate:.2f}%"
        )

        print(
            f"Average Steps: "
            f"{average_steps:.2f}"
        )

        results.append(
            (
                iteration,
                success_rate,
                average_steps,
                len(all_actions),
            )
        )

    os.makedirs(
        "models",
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "observation_size":
                all_observations.shape[1],

            "action_size":
                4,
        },
        DAGGER_MODEL_PATH,
    )

    print()
    print("=" * 60)
    print("=== DAgger Final Results ===")
    print("=" * 60)

    print(
        f"{'Iteration':<12}"
        f"{'Success':<14}"
        f"{'Avg Steps':<14}"
        f"{'Dataset':<12}"
    )

    for (
        iteration,
        success_rate,
        average_steps,
        dataset_size,
    ) in results:

        print(
            f"{iteration:<12}"
            f"{success_rate:<14.2f}"
            f"{average_steps:<14.2f}"
            f"{dataset_size:<12}"
        )

    print()
    print(
        f"Model saved: "
        f"{DAGGER_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()