from collections import defaultdict, Counter, deque

import numpy as np
import torch
import torch.nn as nn

from partial_obstacle_env import PartialObstacleGridEnv


MODEL_PATH = "models/behavior_cloning.pt"

EPISODES = 2000
SEED_START = 40_000

ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
}

ACTION_DELTAS = {
    0: (0, -1),
    1: (0, 1),
    2: (-1, 0),
    3: (1, 0),
}


class BCPolicy(nn.Module):
    def __init__(
        self,
        observation_size,
        action_size,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                observation_size,
                128,
            ),
            nn.ReLU(),

            nn.Linear(
                128,
                128,
            ),
            nn.ReLU(),

            nn.Linear(
                128,
                action_size,
            ),
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


def predict_action(
    model,
    observation,
):
    obs_tensor = torch.tensor(
        observation,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model(
            obs_tensor
        )

    return torch.argmax(
        logits,
        dim=1,
    ).item()


def get_neighbors(
    env,
    position,
):
    x, y = position

    result = []

    for action, (
        dx,
        dy,
    ) in ACTION_DELTAS.items():

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
            (
                (nx, ny),
                action,
            )
        )

    return result


def expert_action_from_current_state(
    env,
):
    start = tuple(
        env.agent_pos
    )

    goal = tuple(
        env.goal_pos
    )

    if start == goal:
        return None

    queue = deque(
        [start]
    )

    parent = {
        start: None
    }

    parent_action = {}

    while queue:
        current = queue.popleft()

        if current == goal:
            break

        for (
            next_position,
            action,
        ) in get_neighbors(
            env,
            current,
        ):
            if (
                next_position
                in parent
            ):
                continue

            parent[
                next_position
            ] = current

            parent_action[
                next_position
            ] = action

            queue.append(
                next_position
            )

    if goal not in parent:
        return None

    current = goal

    actions = []

    while current != start:
        actions.append(
            parent_action[
                current
            ]
        )

        current = parent[
            current
        ]

    actions.reverse()

    return actions[0]


def observation_key(
    observation,
):
    """
    Observation이 float32이므로
    비교를 안정적으로 하기 위해
    소수점 6자리로 round 후 tuple 변환.
    """

    rounded = np.round(
        observation,
        decimals=6,
    )

    return tuple(
        rounded.tolist()
    )


def main():
    model = load_model()

    env = PartialObstacleGridEnv()

    # Observation →
    # 해당 상태에서 BFS Expert가 준 Action 목록
    labels_by_observation = defaultdict(
        list
    )

    total_samples = 0
    successes = 0

    for episode in range(
        EPISODES
    ):
        observation, _ = env.reset(
            seed=SEED_START + episode
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

            key = observation_key(
                observation
            )

            labels_by_observation[
                key
            ].append(
                expert_action
            )

            total_samples += 1

            # 실제로는 BC Agent 행동을 실행.
            # 즉 Expert의 정상 경로뿐 아니라
            # BC가 실제 방문하는 상태를 분석한다.
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
            ) = env.step(
                agent_action
            )

        if terminated:
            successes += 1

    env.close()

    unique_observations = len(
        labels_by_observation
    )

    repeated_observations = 0
    conflicting_observations = 0

    repeated_samples = 0
    conflicting_samples = 0

    conflict_examples = []

    for (
        observation,
        labels,
    ) in labels_by_observation.items():

        if len(labels) > 1:
            repeated_observations += 1
            repeated_samples += len(
                labels
            )

        unique_labels = set(
            labels
        )

        if len(unique_labels) > 1:
            conflicting_observations += 1
            conflicting_samples += len(
                labels
            )

            counts = Counter(
                labels
            )

            conflict_examples.append(
                (
                    len(labels),
                    observation,
                    counts,
                )
            )

    print()
    print(
        "=== Expert Label Conflict Analysis ==="
    )

    print(
        f"Episodes: {EPISODES}"
    )

    print(
        f"BC Success Rate: "
        f"{successes / EPISODES * 100:.2f}%"
    )

    print(
        f"Total Collected Samples: "
        f"{total_samples}"
    )

    print(
        f"Unique Observations: "
        f"{unique_observations}"
    )

    print()

    print(
        f"Repeated Observations: "
        f"{repeated_observations}"
    )

    print(
        f"Conflicting Observations: "
        f"{conflicting_observations}"
    )

    if repeated_observations > 0:
        conflict_rate = (
            conflicting_observations
            / repeated_observations
            * 100.0
        )
    else:
        conflict_rate = 0.0

    print(
        "Conflict Rate among "
        f"Repeated Observations: "
        f"{conflict_rate:.2f}%"
    )

    print()

    print(
        f"Samples belonging to "
        f"Repeated Observations: "
        f"{repeated_samples}"
    )

    print(
        f"Samples belonging to "
        f"Conflicting Observations: "
        f"{conflicting_samples}"
    )

    if total_samples > 0:
        sample_conflict_rate = (
            conflicting_samples
            / total_samples
            * 100.0
        )
    else:
        sample_conflict_rate = 0.0

    print(
        f"Conflict Sample Rate: "
        f"{sample_conflict_rate:.2f}%"
    )

    # 가장 많이 등장한 conflict부터
    # 출력
    conflict_examples.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    print()
    print(
        "=== Top Conflicting Observations ==="
    )

    for index, (
        count,
        observation,
        action_counts,
    ) in enumerate(
        conflict_examples[:10],
        start=1,
    ):
        print()
        print(
            f"[Conflict {index}]"
        )

        print(
            f"Occurrences: {count}"
        )

        print(
            "Observation:"
        )

        print(
            np.array(
                observation
            )
        )

        print(
            "Expert Labels:"
        )

        for (
            action,
            action_count,
        ) in action_counts.items():

            print(
                f"  "
                f"{ACTION_NAMES[action]}: "
                f"{action_count}"
            )

    print()
    print("=" * 60)

    if (
        conflicting_observations
        > 0
    ):
        print(
            "RESULT:"
        )

        print(
            "Identical agent observations "
            "received different BFS expert actions."
        )

        print(
            "This provides evidence of "
            "observation aliasing under "
            "partial observability."
        )

    else:
        print(
            "RESULT:"
        )

        print(
            "No exact observation-label "
            "conflicts were detected."
        )

        print(
            "The DAgger degradation requires "
            "another explanation."
        )


if __name__ == "__main__":
    main()