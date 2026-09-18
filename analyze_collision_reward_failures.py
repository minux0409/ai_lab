from collections import Counter

import numpy as np
from stable_baselines3 import PPO

from partial_obstacle_collision_reward_env import (
    PartialObstacleCollisionRewardEnv,
)


MODEL_PATH = "models/partial_obstacle_collision_reward_ppo"

EPISODES = 500

ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
}


def longest_same_position_run(positions):
    if not positions:
        return 0

    longest = 1
    current = 1

    for i in range(1, len(positions)):
        if positions[i] == positions[i - 1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 1

    return longest


def count_ab_oscillations(positions):
    """
    A -> B -> A 형태가 발생한 횟수.
    """

    count = 0

    for i in range(2, len(positions)):
        if (
            positions[i] == positions[i - 2]
            and positions[i] != positions[i - 1]
        ):
            count += 1

    return count


def longest_ab_oscillation(positions):
    """
    연속적인 A <-> B 왕복 길이.
    """

    longest = 0
    current = 0

    for i in range(2, len(positions)):
        if (
            positions[i] == positions[i - 2]
            and positions[i] != positions[i - 1]
        ):
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def classify_failure(
    unchanged_moves,
    max_same_position_run,
    longest_ab_loop,
):
    # 기존 Full Memory 실패 분석과 동일한 목적:
    # 장시간 같은 위치에 머무는 실패를 STUCK으로 분류.
    if (
        unchanged_moves >= 50
        or max_same_position_run >= 20
    ):
        return "STUCK"

    # 두 위치를 반복적으로 왕복하는 경우.
    if longest_ab_loop >= 10:
        return "A_B_LOOP"

    return "OTHER"


def analyze_episode(
    model,
    seed,
):
    env = PartialObstacleCollisionRewardEnv()

    observation, _ = env.reset(
        seed=seed
    )

    start = tuple(
        int(v)
        for v in env.agent_pos
    )

    goal = tuple(
        int(v)
        for v in env.goal_pos
    )

    positions = [start]
    actions = []

    unchanged_moves = 0
    minimum_goal_distance = (
        abs(start[0] - goal[0])
        + abs(start[1] - goal[1])
    )

    total_reward = 0.0

    success = False

    for step in range(1, env.max_steps + 1):

        previous_position = tuple(
            int(v)
            for v in env.agent_pos
        )

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        action = int(action)

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        current_position = tuple(
            int(v)
            for v in env.agent_pos
        )

        positions.append(
            current_position
        )

        actions.append(
            action
        )

        total_reward += float(
            reward
        )

        if current_position == previous_position:
            unchanged_moves += 1

        distance = (
            abs(current_position[0] - goal[0])
            + abs(current_position[1] - goal[1])
        )

        minimum_goal_distance = min(
            minimum_goal_distance,
            distance,
        )

        if terminated:
            success = True
            break

        if truncated:
            break

    final_position = positions[-1]

    final_goal_distance = (
        abs(final_position[0] - goal[0])
        + abs(final_position[1] - goal[1])
    )

    unique_positions = len(
        set(positions)
    )

    max_same_run = (
        longest_same_position_run(
            positions
        )
    )

    ab_count = (
        count_ab_oscillations(
            positions
        )
    )

    longest_ab = (
        longest_ab_oscillation(
            positions
        )
    )

    known_cells = int(
        np.sum(
            env.discovered_map != -1.0
        )
    )

    failure_type = None

    if not success:
        failure_type = classify_failure(
            unchanged_moves,
            max_same_run,
            longest_ab,
        )

    result = {
        "seed": seed,
        "success": success,
        "steps": len(actions),

        "start": start,
        "goal": goal,
        "final": final_position,

        "unique_positions":
            unique_positions,

        "unchanged_moves":
            unchanged_moves,

        "max_same_position_run":
            max_same_run,

        "ab_oscillations":
            ab_count,

        "longest_ab_loop":
            longest_ab,

        "minimum_goal_distance":
            minimum_goal_distance,

        "final_goal_distance":
            final_goal_distance,

        "known_cells":
            known_cells,

        "total_reward":
            total_reward,

        "actions":
            actions,

        "positions":
            positions,

        "failure_type":
            failure_type,
    }

    env.close()

    return result


def print_failure(
    result,
):
    action_counter = Counter(
        result["actions"]
    )

    print()
    print(
        f'[{result["failure_type"]}] '
        f'Seed {result["seed"]}'
    )

    print(
        "Start:",
        result["start"],
        "| Goal:",
        result["goal"],
        "| Final:",
        result["final"],
    )

    print(
        "Unique Positions       :",
        result["unique_positions"],
    )

    print(
        "Unchanged Moves        :",
        result["unchanged_moves"],
    )

    print(
        "Max Same Position Run  :",
        result["max_same_position_run"],
    )

    print(
        "A-B Oscillations       :",
        result["ab_oscillations"],
    )

    print(
        "Longest A-B Loop       :",
        result["longest_ab_loop"],
    )

    print(
        "Minimum Goal Distance  :",
        result["minimum_goal_distance"],
    )

    print(
        "Final Goal Distance    :",
        result["final_goal_distance"],
    )

    print(
        "Known Cells            :",
        f'{result["known_cells"]}/81',
    )

    print(
        "Total Reward           :",
        f'{result["total_reward"]:.2f}',
    )

    print(
        "Actions:"
    )

    for action in range(4):
        print(
            f"  {ACTION_NAMES[action]:5}: "
            f"{action_counter[action]}"
        )

    print(
        "Last 20 positions:"
    )

    for position in result[
        "positions"
    ][-20:]:
        print(
            position,
            end=" ",
        )

    print()


def main():

    print()
    print(
        "=== Experiment F Failure Analysis ==="
    )

    print(
        "Model:",
        MODEL_PATH,
    )

    print(
        "Episodes:",
        EPISODES,
    )

    print()

    model = PPO.load(
        MODEL_PATH
    )

    results = []

    for episode in range(
        EPISODES
    ):
        seed = (
            10_000
            + episode
        )

        result = analyze_episode(
            model,
            seed,
        )

        results.append(
            result
        )

    successes = [
        r
        for r in results
        if r["success"]
    ]

    failures = [
        r
        for r in results
        if not r["success"]
    ]

    print(
        "=============================================="
    )

    print(
        "OVERALL"
    )

    print(
        "=============================================="
    )

    print(
        "Episodes :",
        len(results),
    )

    print(
        "Success  :",
        len(successes),
        f"({len(successes) / EPISODES * 100:.2f}%)",
    )

    print(
        "Failure  :",
        len(failures),
        f"({len(failures) / EPISODES * 100:.2f}%)",
    )

    failure_counter = Counter(
        r["failure_type"]
        for r in failures
    )

    print()
    print(
        "=============================================="
    )

    print(
        "FAILURE PATTERNS"
    )

    print(
        "=============================================="
    )

    for failure_type in [
        "STUCK",
        "A_B_LOOP",
        "OTHER",
    ]:

        count = failure_counter[
            failure_type
        ]

        percentage = (
            count
            / len(failures)
            * 100.0
            if failures
            else 0.0
        )

        print(
            f"{failure_type:<10}: "
            f"{count:3} "
            f"({percentage:6.2f}%)"
        )

    if failures:

        print()
        print(
            "=============================================="
        )

        print(
            "FAILURE RAW STATISTICS"
        )

        print(
            "=============================================="
        )

        def average(key):
            return float(
                np.mean(
                    [
                        r[key]
                        for r in failures
                    ]
                )
            )

        print(
            "Avg Unique Positions       :",
            f'{average("unique_positions"):.2f}',
        )

        print(
            "Avg Unchanged Moves        :",
            f'{average("unchanged_moves"):.2f}',
        )

        print(
            "Avg Max Same Position Run  :",
            f'{average("max_same_position_run"):.2f}',
        )

        print(
            "Avg Longest A-B Loop       :",
            f'{average("longest_ab_loop"):.2f}',
        )

        print(
            "Avg Minimum Goal Distance  :",
            f'{average("minimum_goal_distance"):.2f}',
        )

        print(
            "Avg Final Goal Distance    :",
            f'{average("final_goal_distance"):.2f}',
        )

        print(
            "Avg Known Cells            :",
            f'{average("known_cells"):.2f}/81',
        )

        near_goal_failures = [
            r
            for r in failures
            if r[
                "minimum_goal_distance"
            ] <= 1
        ]

        print()
        print(
            "Goal distance <= 1 then failed:",
            len(near_goal_failures),
            "/",
            len(failures),
            (
                f"("
                f"{len(near_goal_failures) / len(failures) * 100:.2f}%"
                f")"
            ),
        )

    print()
    print(
        "=============================================="
    )

    print(
        "REPRESENTATIVE FAILURES"
    )

    print(
        "=============================================="
    )

    for failure_type in [
        "STUCK",
        "A_B_LOOP",
        "OTHER",
    ]:

        examples = [
            r
            for r in failures
            if r[
                "failure_type"
            ] == failure_type
        ][:3]

        for result in examples:
            print_failure(
                result
            )


if __name__ == "__main__":
    main()