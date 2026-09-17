from collections import Counter

import numpy as np
from stable_baselines3 import PPO

from partial_obstacle_goal_tile_history_env import (
    PartialObstacleGoalTileHistoryEnv,
)


MODEL_PATH = (
    "models/"
    "partial_obstacle_goal_tile_history_ppo"
)

EPISODES = 500

ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
}


def count_cycle(path, cycle_length):
    """
    특정 길이의 이동 패턴이 바로 반복된 횟수.

    예:
    cycle_length=2
    A B A B

    cycle_length=3
    A B C A B C
    """

    count = 0

    if len(path) < cycle_length * 2:
        return 0

    for i in range(
        cycle_length * 2,
        len(path) + 1,
    ):
        first = path[
            i - cycle_length * 2:
            i - cycle_length
        ]

        second = path[
            i - cycle_length:
            i
        ]

        if first == second:
            count += 1

    return count


def longest_same_position_run(path):
    """
    같은 위치에 연속으로 머문 최대 횟수.

    벽에 계속 박는 경우 크게 나온다.
    """

    if not path:
        return 0

    longest = 1
    current = 1

    for i in range(1, len(path)):
        if path[i] == path[i - 1]:
            current += 1
            longest = max(
                longest,
                current,
            )
        else:
            current = 1

    # 이동 횟수 기준으로 보기 위해 -1
    return max(0, longest - 1)


def analyze():
    print("=== History 모델 실패 분석 ===")
    print()
    print("Model:")
    print(MODEL_PATH)

    model = PPO.load(MODEL_PATH)

    env = (
        PartialObstacleGoalTileHistoryEnv()
    )

    successes = 0
    success_steps = []

    failures = []

    for episode in range(EPISODES):
        obs, _ = env.reset(
            seed=10_000 + episode
        )

        start = tuple(
            int(v)
            for v in env.agent_pos
        )

        goal = tuple(
            int(v)
            for v in env.goal_pos
        )

        path = [start]

        actions = []
        rewards = []
        tile_rewards = []
        collisions = []
        first_visits = []

        total_reward = 0.0

        for step in range(
            1,
            env.max_steps + 1,
        ):
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            action = int(action)

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            current_position = (
                int(env.agent_pos[0]),
                int(env.agent_pos[1]),
            )

            path.append(
                current_position
            )

            actions.append(action)

            rewards.append(
                float(reward)
            )

            tile_rewards.append(
                float(
                    info["tile_reward"]
                )
            )

            collisions.append(
                bool(
                    info["collision"]
                )
            )

            first_visits.append(
                bool(
                    info["first_visit"]
                )
            )

            total_reward += float(
                reward
            )

            if terminated:
                successes += 1

                success_steps.append(
                    step
                )

                break

            if truncated:
                unique_positions = len(
                    set(path)
                )

                collision_count = sum(
                    collisions
                )

                unchanged_moves = sum(
                    1
                    for i in range(
                        1,
                        len(path),
                    )
                    if path[i]
                    == path[i - 1]
                )

                first_visit_count = sum(
                    first_visits
                )

                min_goal_distance = min(
                    abs(
                        goal[0]
                        - position[0]
                    )
                    + abs(
                        goal[1]
                        - position[1]
                    )
                    for position in path
                )

                position_counts = Counter(
                    path
                )

                most_common_positions = (
                    position_counts
                    .most_common(5)
                )

                cycle_2 = count_cycle(
                    path,
                    2,
                )

                cycle_3 = count_cycle(
                    path,
                    3,
                )

                cycle_4 = count_cycle(
                    path,
                    4,
                )

                max_stuck = (
                    longest_same_position_run(
                        path
                    )
                )

                failures.append(
                    {
                        "episode":
                            episode,

                        "start":
                            start,

                        "goal":
                            goal,

                        "path":
                            path,

                        "actions":
                            actions,

                        "rewards":
                            rewards,

                        "tile_rewards":
                            tile_rewards,

                        "collisions":
                            collisions,

                        "first_visits":
                            first_visits,

                        "total_reward":
                            total_reward,

                        "unique_positions":
                            unique_positions,

                        "collision_count":
                            collision_count,

                        "unchanged_moves":
                            unchanged_moves,

                        "first_visit_count":
                            first_visit_count,

                        "min_goal_distance":
                            min_goal_distance,

                        "discovered_cells":
                            info[
                                "discovered_cells"
                            ],

                        "cycle_2":
                            cycle_2,

                        "cycle_3":
                            cycle_3,

                        "cycle_4":
                            cycle_4,

                        "max_stuck":
                            max_stuck,

                        "most_common":
                            most_common_positions,
                    }
                )

                break

    env.close()

    print()
    print("=== 전체 결과 ===")

    print(
        f"Success : "
        f"{successes}/{EPISODES} "
        f"({successes / EPISODES * 100:.1f}%)"
    )

    print(
        f"Failure : "
        f"{len(failures)}/{EPISODES}"
    )

    if success_steps:
        print(
            "Success Avg Steps : "
            f"{np.mean(success_steps):.2f}"
        )

    if not failures:
        return

    print()
    print("=== 실패 평균 ===")

    metrics = [
        (
            "Unique Positions",
            "unique_positions",
        ),
        (
            "Collisions",
            "collision_count",
        ),
        (
            "Unchanged Moves",
            "unchanged_moves",
        ),
        (
            "First Visits",
            "first_visit_count",
        ),
        (
            "Discovered Cells",
            "discovered_cells",
        ),
        (
            "Min Goal Distance",
            "min_goal_distance",
        ),
        (
            "2-Step Cycle",
            "cycle_2",
        ),
        (
            "3-Step Cycle",
            "cycle_3",
        ),
        (
            "4-Step Cycle",
            "cycle_4",
        ),
        (
            "Max Same-Pos Run",
            "max_stuck",
        ),
        (
            "Total Reward",
            "total_reward",
        ),
    ]

    for name, key in metrics:
        value = np.mean(
            [
                failure[key]
                for failure in failures
            ]
        )

        if key == "discovered_cells":
            print(
                f"{name:20s}: "
                f"{value:.2f}/81"
            )
        else:
            print(
                f"{name:20s}: "
                f"{value:.2f}"
            )

    # ---------------------------------------------
    # 실패 유형 대략적인 개수
    # ---------------------------------------------
    heavy_collision = sum(
        1
        for f in failures
        if f["collision_count"] >= 20
    )

    heavy_two_cycle = sum(
        1
        for f in failures
        if f["cycle_2"] >= 20
    )

    near_goal = sum(
        1
        for f in failures
        if f["min_goal_distance"] <= 1
    )

    explored_many = sum(
        1
        for f in failures
        if f["unique_positions"] >= 20
    )

    print()
    print("=== 실패 유형 ===")

    print(
        "Collision >= 20 : "
        f"{heavy_collision}"
        f"/{len(failures)}"
    )

    print(
        "2-Step Cycle >= 20 : "
        f"{heavy_two_cycle}"
        f"/{len(failures)}"
    )

    print(
        "Goal 거리 <= 1 실패 : "
        f"{near_goal}"
        f"/{len(failures)}"
    )

    print(
        "20칸 이상 이동 후 실패 : "
        f"{explored_many}"
        f"/{len(failures)}"
    )

    # ---------------------------------------------
    # 실패 Episode 상세
    # ---------------------------------------------
    print()
    print(
        "=== 실패 Episode 상세 "
        "(앞 10개) ==="
    )

    for failure in failures[:10]:
        print()
        print("=" * 75)

        print(
            f"Episode "
            f"{failure['episode']}"
        )

        print(
            f"Start : "
            f"{failure['start']}"
        )

        print(
            f"Goal  : "
            f"{failure['goal']}"
        )

        print(
            f"Unique Positions : "
            f"{failure['unique_positions']}"
        )

        print(
            f"Collisions       : "
            f"{failure['collision_count']}"
        )

        print(
            f"Unchanged Moves  : "
            f"{failure['unchanged_moves']}"
        )

        print(
            f"First Visits     : "
            f"{failure['first_visit_count']}"
        )

        print(
            f"Discovered       : "
            f"{failure['discovered_cells']}"
            f"/81"
        )

        print(
            f"Min Goal Distance: "
            f"{failure['min_goal_distance']}"
        )

        print(
            f"2-Step Cycle     : "
            f"{failure['cycle_2']}"
        )

        print(
            f"3-Step Cycle     : "
            f"{failure['cycle_3']}"
        )

        print(
            f"4-Step Cycle     : "
            f"{failure['cycle_4']}"
        )

        print(
            f"Max Same-Pos Run : "
            f"{failure['max_stuck']}"
        )

        print(
            f"Total Reward     : "
            f"{failure['total_reward']:.1f}"
        )

        print()
        print(
            "Most visited positions:"
        )

        for (
            position,
            count,
        ) in failure["most_common"]:
            print(
                f"  {position} x {count}"
            )

        # -----------------------------------------
        # 처음 25 step
        # -----------------------------------------
        print()
        print("--- 처음 25 Step ---")

        first_count = min(
            25,
            len(
                failure["actions"]
            ),
        )

        for i in range(first_count):
            print_step(
                failure,
                i,
            )

        # -----------------------------------------
        # 마지막 25 step
        # -----------------------------------------
        print()
        print("--- 마지막 25 Step ---")

        start_index = max(
            0,
            len(
                failure["actions"]
            ) - 25,
        )

        for i in range(
            start_index,
            len(
                failure["actions"]
            ),
        ):
            print_step(
                failure,
                i,
            )


def print_step(
    failure,
    index,
):
    before = (
        failure["path"][index]
    )

    after = (
        failure["path"][index + 1]
    )

    action = (
        failure["actions"][index]
    )

    reward = (
        failure["rewards"][index]
    )

    tile_reward = (
        failure[
            "tile_rewards"
        ][index]
    )

    collision = (
        failure[
            "collisions"
        ][index]
    )

    first_visit = (
        failure[
            "first_visits"
        ][index]
    )

    history_start = max(
        0,
        index - 3,
    )

    recent_positions = (
        failure["path"][
            history_start:
            index + 1
        ]
    )

    print(
        f"{index + 1:3d} | "
        f"{before} "
        f"--{ACTION_NAMES[action]:5s}--> "
        f"{after} | "
        f"R {reward:5.1f} | "
        f"Tile {tile_reward:3.1f} | "
        f"First {str(first_visit):5s} | "
        f"Collision {str(collision):5s} | "
        f"H {recent_positions}"
    )


if __name__ == "__main__":
    analyze()