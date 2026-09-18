from collections import Counter

import numpy as np
from stable_baselines3 import PPO

from partial_obstacle_full_memory_env import (
    PartialObstacleFullMemoryEnv,
)


MODEL_PATH = "models/partial_obstacle_full_memory_ppo"

START_SEED = 10_000
EPISODES = 500

ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
}


def manhattan(a, b):
    return (
        abs(int(a[0]) - int(b[0]))
        + abs(int(a[1]) - int(b[1]))
    )


def run_episode(model, seed):
    env = PartialObstacleFullMemoryEnv()

    observation, _ = env.reset(seed=seed)

    start_pos = tuple(env.agent_pos)
    goal_pos = tuple(env.goal_pos)

    positions = [start_pos]
    actions = []

    moved_flags = []

    total_reward = 0.0

    min_goal_distance = manhattan(
        env.agent_pos,
        env.goal_pos,
    )

    while True:
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

        position = tuple(env.agent_pos)

        positions.append(position)
        actions.append(action)

        moved_flags.append(
            bool(info["moved"])
        )

        total_reward += float(reward)

        min_goal_distance = min(
            min_goal_distance,
            manhattan(
                env.agent_pos,
                env.goal_pos,
            ),
        )

        if terminated or truncated:
            success = bool(terminated)
            break

    discovered_cells = int(
        np.sum(
            env.discovered_map != -1
        )
    )

    visit_count_map = (
        env.visit_count_map.copy()
    )

    env.close()

    return {
        "seed": seed,
        "success": success,
        "steps": len(actions),

        "start": start_pos,
        "goal": goal_pos,
        "final": positions[-1],

        "positions": positions,
        "actions": actions,
        "moved_flags": moved_flags,

        "total_reward": total_reward,

        "unique_positions": len(
            set(positions)
        ),

        "unchanged_moves": sum(
            not moved
            for moved in moved_flags
        ),

        "min_goal_distance":
            min_goal_distance,

        "final_goal_distance":
            manhattan(
                positions[-1],
                goal_pos,
            ),

        "discovered_cells":
            discovered_cells,

        "visit_count_map":
            visit_count_map,
    }


def longest_same_position_run(
    positions,
):
    if not positions:
        return 0

    longest = 1
    current = 1

    for i in range(
        1,
        len(positions)
    ):
        if (
            positions[i]
            == positions[i - 1]
        ):
            current += 1

            longest = max(
                longest,
                current,
            )
        else:
            current = 1

    return longest


def count_two_position_oscillation(
    positions,
):
    """
    A -> B -> A -> B 형태가 몇 번 반복됐는지 센다.
    """

    count = 0

    for i in range(
        2,
        len(positions)
    ):
        if (
            positions[i]
            == positions[i - 2]
            and
            positions[i]
            != positions[i - 1]
        ):
            count += 1

    return count


def longest_two_position_oscillation(
    positions,
):
    """
    연속적인 A-B-A-B 패턴 길이를 센다.

    반환값은 패턴 판정 횟수 기준.
    """

    longest = 0
    current = 0

    for i in range(
        2,
        len(positions)
    ):
        is_oscillation = (
            positions[i]
            == positions[i - 2]
            and
            positions[i]
            != positions[i - 1]
        )

        if is_oscillation:
            current += 1

            longest = max(
                longest,
                current,
            )
        else:
            current = 0

    return longest


def analyze_failure(episode):
    positions = episode["positions"]

    same_run = (
        longest_same_position_run(
            positions
        )
    )

    oscillations = (
        count_two_position_oscillation(
            positions
        )
    )

    longest_oscillation = (
        longest_two_position_oscillation(
            positions
        )
    )

    unchanged = (
        episode["unchanged_moves"]
    )

    # --------------------------------------------
    # 행동 유형 분류
    #
    # 이건 "원인" 확정이 아니라
    # 경로에서 관측된 행동 패턴 분류다.
    # --------------------------------------------

    if same_run >= 20:
        category = "STUCK"

    elif longest_oscillation >= 20:
        category = "A_B_LOOP"

    else:
        category = "OTHER"

    return {
        **episode,

        "same_position_max_run":
            same_run,

        "two_position_oscillations":
            oscillations,

        "longest_two_position_loop":
            longest_oscillation,

        "category":
            category,

        "unchanged_ratio":
            (
                unchanged
                / episode["steps"]
                if episode["steps"] > 0
                else 0.0
            ),
    }


def print_example(
    failure,
):
    print()
    print(
        "----------------------------------------"
    )

    print(
        f"Seed: {failure['seed']}"
    )

    print(
        f"Category: "
        f"{failure['category']}"
    )

    print(
        f"Start: {failure['start']}"
    )

    print(
        f"Goal: {failure['goal']}"
    )

    print(
        f"Final: {failure['final']}"
    )

    print(
        f"Unique Positions: "
        f"{failure['unique_positions']}"
    )

    print(
        f"Unchanged Moves: "
        f"{failure['unchanged_moves']}"
        f"/{failure['steps']}"
        f" "
        f"({failure['unchanged_ratio'] * 100:.1f}%)"
    )

    print(
        f"Max Same Position Run: "
        f"{failure['same_position_max_run']}"
    )

    print(
        f"A-B Oscillations: "
        f"{failure['two_position_oscillations']}"
    )

    print(
        f"Longest A-B Loop: "
        f"{failure['longest_two_position_loop']}"
    )

    print(
        f"Minimum Goal Distance: "
        f"{failure['min_goal_distance']}"
    )

    print(
        f"Final Goal Distance: "
        f"{failure['final_goal_distance']}"
    )

    print(
        f"Known Cells: "
        f"{failure['discovered_cells']}/81"
    )

    print(
        f"Total Reward: "
        f"{failure['total_reward']:.2f}"
    )

    action_counts = Counter(
        failure["actions"]
    )

    print(
        "Actions:",
        {
            ACTION_NAMES[action]:
                count
            for action, count
            in action_counts.items()
        },
    )

    print(
        "Last 20 Positions:"
    )

    print(
        failure["positions"][-20:]
    )


def main():
    print()
    print(
        "=== Full Memory PPO Failure Analysis ==="
    )

    print()
    print(
        f"Model: {MODEL_PATH}"
    )

    model = PPO.load(
        MODEL_PATH
    )

    successes = []
    failures = []

    print()
    print(
        f"{EPISODES}개 Episode 분석 중..."
    )

    for episode_index in range(
        EPISODES
    ):
        seed = (
            START_SEED
            + episode_index
        )

        result = run_episode(
            model,
            seed,
        )

        if result["success"]:
            successes.append(
                result
            )
        else:
            failures.append(
                analyze_failure(
                    result
                )
            )

    # ============================================
    # 전체 결과
    # ============================================

    print()
    print(
        "========================================"
    )

    print(
        "OVERALL"
    )

    print(
        "========================================"
    )

    print(
        f"Episodes : {EPISODES}"
    )

    print(
        f"Success  : "
        f"{len(successes)} "
        f"({len(successes) / EPISODES * 100:.2f}%)"
    )

    print(
        f"Failure  : "
        f"{len(failures)} "
        f"({len(failures) / EPISODES * 100:.2f}%)"
    )

    if not failures:
        print(
            "실패 Episode가 없습니다."
        )

        return

    # ============================================
    # 실패 패턴
    # ============================================

    categories = Counter(
        failure["category"]
        for failure in failures
    )

    print()
    print(
        "========================================"
    )

    print(
        "FAILURE PATTERNS"
    )

    print(
        "========================================"
    )

    for category in [
        "STUCK",
        "A_B_LOOP",
        "OTHER",
    ]:
        count = categories[
            category
        ]

        percentage = (
            count
            / len(failures)
            * 100.0
        )

        print(
            f"{category:<10} : "
            f"{count:>3} "
            f"({percentage:>6.2f}%)"
        )

    # ============================================
    # Raw 통계
    # ============================================

    avg_unique = np.mean(
        [
            failure[
                "unique_positions"
            ]
            for failure in failures
        ]
    )

    avg_unchanged = np.mean(
        [
            failure[
                "unchanged_moves"
            ]
            for failure in failures
        ]
    )

    avg_same_run = np.mean(
        [
            failure[
                "same_position_max_run"
            ]
            for failure in failures
        ]
    )

    avg_ab_loop = np.mean(
        [
            failure[
                "longest_two_position_loop"
            ]
            for failure in failures
        ]
    )

    avg_min_distance = np.mean(
        [
            failure[
                "min_goal_distance"
            ]
            for failure in failures
        ]
    )

    avg_final_distance = np.mean(
        [
            failure[
                "final_goal_distance"
            ]
            for failure in failures
        ]
    )

    avg_known = np.mean(
        [
            failure[
                "discovered_cells"
            ]
            for failure in failures
        ]
    )

    print()
    print(
        "========================================"
    )

    print(
        "FAILURE RAW STATISTICS"
    )

    print(
        "========================================"
    )

    print(
        f"Avg Unique Positions       : "
        f"{avg_unique:.2f}"
    )

    print(
        f"Avg Unchanged Moves        : "
        f"{avg_unchanged:.2f}"
    )

    print(
        f"Avg Max Same Position Run  : "
        f"{avg_same_run:.2f}"
    )

    print(
        f"Avg Longest A-B Loop       : "
        f"{avg_ab_loop:.2f}"
    )

    print(
        f"Avg Minimum Goal Distance  : "
        f"{avg_min_distance:.2f}"
    )

    print(
        f"Avg Final Goal Distance    : "
        f"{avg_final_distance:.2f}"
    )

    print(
        f"Avg Known Cells            : "
        f"{avg_known:.2f}/81"
    )

    # ============================================
    # Goal 근처까지 갔다 실패
    # ============================================

    near_goal = [
        failure
        for failure in failures
        if failure[
            "min_goal_distance"
        ] <= 1
    ]

    print()
    print(
        "========================================"
    )

    print(
        "NEAR GOAL FAILURES"
    )

    print(
        "========================================"
    )

    print(
        f"Goal 거리 1 이하까지 갔다 실패: "
        f"{len(near_goal)}"
        f"/{len(failures)} "
        f"({len(near_goal) / len(failures) * 100:.2f}%)"
    )

    # ============================================
    # 대표 실패 출력
    # ============================================

    print()
    print(
        "========================================"
    )

    print(
        "EXAMPLES"
    )

    print(
        "========================================"
    )

    for category in [
        "STUCK",
        "A_B_LOOP",
        "OTHER",
    ]:

        examples = [
            failure
            for failure in failures
            if failure[
                "category"
            ] == category
        ]

        if not examples:
            continue

        print()
        print(
            f"### {category}"
        )

        # 유형별 최대 3개
        for failure in (
            examples[:3]
        ):
            print_example(
                failure
            )


if __name__ == "__main__":
    main()