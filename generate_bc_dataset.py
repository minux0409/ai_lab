from collections import deque

import numpy as np

from partial_obstacle_env import PartialObstacleGridEnv


DATASET_PATH = "datasets/bc_expert_dataset.npz"
EPISODES = 3000


def get_neighbors(env, position):
    x, y = position

    candidates = [
        ((x, y - 1), 0),  # up
        ((x, y + 1), 1),  # down
        ((x - 1, y), 2),  # left
        ((x + 1, y), 3),  # right
    ]

    result = []

    for (nx, ny), action in candidates:
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


def find_bfs_actions(env):
    start = tuple(env.agent_pos)
    goal = tuple(env.goal_pos)

    queue = deque([start])

    # 각 위치가 어디서 왔는지 기록
    parent = {
        start: None
    }

    # 해당 위치로 들어올 때 사용한 action
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

    # Goal → Start 역추적
    actions = []
    current = goal

    while current != start:
        actions.append(
            parent_action[current]
        )

        current = parent[current]

    actions.reverse()

    return actions


def generate_dataset():
    env = PartialObstacleGridEnv()

    observations = []
    actions = []

    success_episodes = 0

    for episode in range(EPISODES):
        obs, _ = env.reset(
            seed=20_000 + episode
        )

        expert_actions = find_bfs_actions(env)

        if expert_actions is None:
            continue

        success_episodes += 1

        for action in expert_actions:
            # 현재 상태에서 전문가가 어떤 행동을 했는지 저장
            observations.append(
                obs.copy()
            )

            actions.append(action)

            obs, reward, terminated, truncated, _ = env.step(
                action
            )

            if terminated:
                break

    env.close()

    observations = np.array(
        observations,
        dtype=np.float32,
    )

    actions = np.array(
        actions,
        dtype=np.int64,
    )

    print("\n=== Expert Dataset ===")
    print(f"사용한 Episode: {success_episodes}")
    print(f"총 Demonstration Step: {len(actions)}")
    print(f"Observation Shape: {observations.shape}")
    print(f"Action Shape: {actions.shape}")

    np.savez_compressed(
        DATASET_PATH,
        observations=observations,
        actions=actions,
    )

    print(
        f"\n저장 완료: {DATASET_PATH}"
    )


if __name__ == "__main__":
    generate_dataset()