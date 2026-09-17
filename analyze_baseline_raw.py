from collections import Counter

import numpy as np
from stable_baselines3 import PPO

from partial_obstacle_env import PartialObstacleGridEnv


# ============================================================
# Experiment #01
# Baseline PPO Raw Failure Analysis
#
# 목적:
#   실패 원인을 미리 분류하지 않고 관측 가능한 값만 수집한다.
# ============================================================

MODEL_PATH = "models/partial_obstacle_ppo"
NUM_EPISODES = 500

ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
}


def get_position(env):
    """현재 agent 위치를 tuple 형태로 반환."""
    return tuple(int(v) for v in env.agent_pos)


def get_goal_position(env):
    """현재 goal 위치를 tuple 형태로 반환."""
    return tuple(int(v) for v in env.goal_pos)


def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def max_consecutive_same_position(positions):
    """동일 위치가 연속으로 몇 번까지 반복됐는지 계산."""
    if not positions:
        return 0

    max_count = 1
    current_count = 1

    for i in range(1, len(positions)):
        if positions[i] == positions[i - 1]:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 1

    return max_count


def compact_trajectory(positions, max_items=30):
    """
    trajectory가 너무 길 경우 앞/뒤만 출력.
    """
    if len(positions) <= max_items:
        return positions

    half = max_items // 2

    return (
        positions[:half]
        + [("...", "...")]
        + positions[-half:]
    )


def main():

    print("=" * 70)
    print("Experiment #01 - Baseline PPO Raw Evaluation")
    print("=" * 70)

    print(f"Model    : {MODEL_PATH}")
    print(f"Episodes : {NUM_EPISODES}")
    print()

    model = PPO.load(MODEL_PATH)

    env = PartialObstacleGridEnv()

    success_count = 0
    failure_count = 0

    all_steps = []
    all_rewards = []

    failure_records = []

    for episode in range(NUM_EPISODES):

        obs, info = env.reset()

        start_pos = get_position(env)
        goal_pos = get_goal_position(env)

        positions = [start_pos]
        actions = []

        episode_reward = 0.0
        step_count = 0
        position_unchanged_count = 0

        min_goal_distance = manhattan_distance(
            start_pos,
            goal_pos,
        )

        terminated = False
        truncated = False

        while not (terminated or truncated):

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            action = int(action)

            before_pos = get_position(env)

            obs, reward, terminated, truncated, info = env.step(action)

            after_pos = get_position(env)

            if after_pos == before_pos:
                position_unchanged_count += 1

            positions.append(after_pos)
            actions.append(action)

            episode_reward += float(reward)
            step_count += 1

            distance = manhattan_distance(
                after_pos,
                goal_pos,
            )

            min_goal_distance = min(
                min_goal_distance,
                distance,
            )

        # Goal 도달 여부.
        #
        # 환경 내부 성공 판정 방식과 최대한 독립적으로 보기 위해
        # 최종 위치가 goal과 같은지도 확인한다.
        success = get_position(env) == goal_pos

        if success:
            success_count += 1

        else:
            failure_count += 1

            position_counter = Counter(positions)
            action_counter = Counter(actions)

            most_common_position, most_common_position_count = (
                position_counter.most_common(1)[0]
            )

            if action_counter:
                most_common_action, most_common_action_count = (
                    action_counter.most_common(1)[0]
                )
            else:
                most_common_action = -1
                most_common_action_count = 0

            failure_records.append(
                {
                    "episode": episode + 1,
                    "start": start_pos,
                    "goal": goal_pos,
                    "steps": step_count,
                    "reward": episode_reward,
                    "unique_positions": len(set(positions)),
                    "unchanged_moves": position_unchanged_count,
                    "max_same_position": max_consecutive_same_position(
                        positions
                    ),
                    "most_common_position": most_common_position,
                    "most_common_position_count": (
                        most_common_position_count
                    ),
                    "most_common_action": most_common_action,
                    "most_common_action_count": (
                        most_common_action_count
                    ),
                    "min_goal_distance": min_goal_distance,
                    "trajectory": positions,
                }
            )

        all_steps.append(step_count)
        all_rewards.append(episode_reward)

    # ========================================================
    # 전체 결과
    # ========================================================

    print()
    print("=" * 70)
    print("OVERALL RESULT")
    print("=" * 70)

    print(f"Total Episodes : {NUM_EPISODES}")
    print(
        f"Success        : {success_count} "
        f"({success_count / NUM_EPISODES * 100:.2f}%)"
    )
    print(
        f"Failure        : {failure_count} "
        f"({failure_count / NUM_EPISODES * 100:.2f}%)"
    )

    print(f"Average Steps  : {np.mean(all_steps):.2f}")
    print(f"Average Reward : {np.mean(all_rewards):.4f}")

    # ========================================================
    # 실패 전체에 대한 raw 통계
    # ========================================================

    if failure_records:

        print()
        print("=" * 70)
        print("FAILURE RAW STATISTICS")
        print("=" * 70)

        print(
            "Avg Unique Positions       : "
            f"{np.mean([r['unique_positions'] for r in failure_records]):.2f}"
        )

        print(
            "Avg Unchanged Moves        : "
            f"{np.mean([r['unchanged_moves'] for r in failure_records]):.2f}"
        )

        print(
            "Avg Max Same Position Run  : "
            f"{np.mean([r['max_same_position'] for r in failure_records]):.2f}"
        )

        print(
            "Avg Minimum Goal Distance  : "
            f"{np.mean([r['min_goal_distance'] for r in failure_records]):.2f}"
        )

        # ====================================================
        # 실패 사례
        # ====================================================

        print()
        print("=" * 70)
        print("FAILURE EXAMPLES")
        print("=" * 70)

        for record in failure_records[:10]:

            action_id = record["most_common_action"]

            action_name = ACTION_NAMES.get(
                action_id,
                f"UNKNOWN({action_id})",
            )

            print()
            print("-" * 70)

            print(f"Episode                 : {record['episode']}")
            print(f"Start                    : {record['start']}")
            print(f"Goal                     : {record['goal']}")

            print(f"Steps                    : {record['steps']}")
            print(f"Reward                   : {record['reward']:.4f}")

            print(
                f"Unique Positions         : "
                f"{record['unique_positions']}"
            )

            print(
                f"Position Unchanged Moves : "
                f"{record['unchanged_moves']}"
            )

            print(
                f"Max Same Position Run    : "
                f"{record['max_same_position']}"
            )

            print(
                f"Most Visited Position    : "
                f"{record['most_common_position']} "
                f"({record['most_common_position_count']} visits)"
            )

            print(
                f"Most Selected Action     : "
                f"{action_name} "
                f"({record['most_common_action_count']} times)"
            )

            print(
                f"Minimum Goal Distance    : "
                f"{record['min_goal_distance']}"
            )

            print("Trajectory:")

            print(
                compact_trajectory(
                    record["trajectory"]
                )
            )

    env.close()

    print()
    print("=" * 70)
    print("Evaluation finished.")
    print("=" * 70)


if __name__ == "__main__":
    main()