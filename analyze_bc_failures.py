from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from partial_obstacle_env import PartialObstacleGridEnv


MODEL_PATH = "models/behavior_cloning.pt"

EVAL_EPISODES = 500
EVAL_SEED_START = 10_000

MAX_FAILURES_TO_PRINT = 5


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


def load_model():
    checkpoint = torch.load(
        MODEL_PATH,
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


def get_agent_position(env):
    return tuple(env.agent_pos)


def count_ab_a_oscillations(path):
    count = 0

    for i in range(2, len(path)):
        if path[i] == path[i - 2] and path[i] != path[i - 1]:
            count += 1

    return count


def manhattan_distance(a, b):
    return (
        abs(a[0] - b[0])
        + abs(a[1] - b[1])
    )


def print_grid(env, path):
    path_counts = Counter(path)

    agent = tuple(env.agent_pos)
    goal = tuple(env.goal_pos)

    print()

    for y in range(env.grid_size):
        row = []

        for x in range(env.grid_size):
            pos = (x, y)

            if pos == agent:
                cell = "A"

            elif pos == goal:
                cell = "G"

            elif env.grid[y, x] == 1:
                cell = "#"

            elif pos in path_counts:
                count = path_counts[pos]

                if count >= 10:
                    cell = "*"
                else:
                    cell = str(count)

            else:
                cell = "."

            row.append(cell)

        print(" ".join(row))

    print()


def analyze_failure(
    env,
    seed,
    path,
    actions,
):
    goal = tuple(env.goal_pos)
    start = path[0]

    position_counts = Counter(path)

    repeated_positions = sum(
        1
        for count in position_counts.values()
        if count > 1
    )

    oscillations = count_ab_a_oscillations(
        path
    )

    min_goal_distance = min(
        manhattan_distance(
            position,
            goal,
        )
        for position in path
    )

    most_common = position_counts.most_common(
        5
    )

    action_counts = Counter(actions)

    print("=" * 60)
    print(f"Failure Seed: {seed}")
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    print(f"Steps: {len(actions)}")
    print(
        f"Repeated Positions: "
        f"{repeated_positions}"
    )
    print(
        f"A -> B -> A Oscillations: "
        f"{oscillations}"
    )
    print(
        f"Minimum Goal Distance: "
        f"{min_goal_distance}"
    )

    print()
    print("Most Visited Positions:")

    for position, count in most_common:
        print(
            f"  {position}: "
            f"{count} times"
        )

    print()
    print("Action Counts:")

    action_names = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT",
    }

    for action in range(4):
        print(
            f"  {action_names[action]}: "
            f"{action_counts[action]}"
        )

    print()
    print("Path Visualization")
    print(
        "* = visited 10+ times"
    )

    print_grid(
        env,
        path,
    )


def main():
    model = load_model()

    env = PartialObstacleGridEnv()

    successes = 0
    failures = 0

    failure_data = []

    total_oscillations = 0
    total_repeated_positions = 0

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

            path = [
                get_agent_position(env)
            ]

            actions = []

            while (
                not terminated
                and not truncated
            ):

                observation_tensor = (
                    torch.tensor(
                        observation,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                )

                logits = model(
                    observation_tensor
                )

                action = torch.argmax(
                    logits,
                    dim=1,
                ).item()

                actions.append(action)

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = env.step(action)

                path.append(
                    get_agent_position(env)
                )

            if terminated:
                successes += 1

            else:
                failures += 1

                oscillations = (
                    count_ab_a_oscillations(
                        path
                    )
                )

                repeated_positions = sum(
                    1
                    for count
                    in Counter(path).values()
                    if count > 1
                )

                total_oscillations += (
                    oscillations
                )

                total_repeated_positions += (
                    repeated_positions
                )

                failure_data.append(
                    {
                        "seed": seed,
                        "path": path,
                        "actions": actions,
                        "oscillations":
                            oscillations,
                        "repeated_positions":
                            repeated_positions,
                    }
                )

    print()
    print(
        "=== Behavior Cloning "
        "Failure Analysis ==="
    )

    print(
        f"Episodes: {EVAL_EPISODES}"
    )

    print(
        f"Successes: {successes}"
    )

    print(
        f"Failures: {failures}"
    )

    print(
        f"Success Rate: "
        f"{successes / EVAL_EPISODES * 100:.2f}%"
    )

    if failures > 0:

        print(
            f"Average Oscillations "
            f"per Failure: "
            f"{total_oscillations / failures:.2f}"
        )

        print(
            f"Average Repeated Positions "
            f"per Failure: "
            f"{total_repeated_positions / failures:.2f}"
        )

    failure_data.sort(
        key=lambda x: (
            x["oscillations"],
            x["repeated_positions"],
        ),
        reverse=True,
    )

    print()
    print(
        f"=== Top "
        f"{min(MAX_FAILURES_TO_PRINT, len(failure_data))} "
        f"Failure Cases ==="
    )

    for data in failure_data[
        :MAX_FAILURES_TO_PRINT
    ]:
        seed = data["seed"]

        env.reset(seed=seed)

        # 동일 경로 재생해서 마지막 상태까지 이동
        for action in data["actions"]:
            _, _, terminated, truncated, _ = env.step(
                action
            )

            if terminated or truncated:
                break

        analyze_failure(
            env=env,
            seed=seed,
            path=data["path"],
            actions=data["actions"],
        )

    env.close()


if __name__ == "__main__":
    main()