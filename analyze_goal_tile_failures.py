from collections import Counter

import numpy as np
from stable_baselines3 import PPO

from partial_obstacle_map_reward_env import (
    PartialObstacleMapRewardEnv,
)


MODEL_PATH = (
    "models/"
    "partial_obstacle_goal_tile_reward_ppo"
)

EPISODES = 500

ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
}


def position_text(pos):
    return f"({pos[0]},{pos[1]})"


def analyze():
    print("모델 로드:")
    print(MODEL_PATH)

    model = PPO.load(MODEL_PATH)

    env = PartialObstacleMapRewardEnv()

    successes = 0
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

            current = (
                int(env.agent_pos[0]),
                int(env.agent_pos[1]),
            )

            path.append(current)
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

            total_reward += reward

            if terminated:
                successes += 1
                break

            if truncated:
                unique_positions = (
                    len(set(path))
                )

                collision_count = sum(
                    collisions
                )

                first_visit_count = sum(
                    first_visits
                )

                # -----------------------------
                # 제자리 반복 횟수
                # -----------------------------
                unchanged_moves = 0

                for i in range(
                    1,
                    len(path),
                ):
                    if (
                        path[i]
                        == path[i - 1]
                    ):
                        unchanged_moves += 1

                # -----------------------------
                # A -> B -> A 진동
                # -----------------------------
                aba_count = 0

                for i in range(
                    2,
                    len(path),
                ):
                    if (
                        path[i]
                        == path[i - 2]
                        and path[i]
                        != path[i - 1]
                    ):
                        aba_count += 1

                # -----------------------------
                # 가장 많이 방문한 위치
                # -----------------------------
                position_counts = Counter(
                    path
                )

                most_common_positions = (
                    position_counts
                    .most_common(5)
                )

                # -----------------------------
                # Goal에 가장 가까이 갔던 거리
                # -----------------------------
                min_goal_distance = min(
                    abs(goal[0] - p[0])
                    + abs(goal[1] - p[1])
                    for p in path
                )

                failures.append(
                    {
                        "episode": episode,
                        "start": start,
                        "goal": goal,
                        "path": path,
                        "actions": actions,
                        "rewards": rewards,
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
                        "aba_count":
                            aba_count,
                        "first_visit_count":
                            first_visit_count,
                        "most_common":
                            most_common_positions,
                        "min_goal_distance":
                            min_goal_distance,
                        "discovered_cells":
                            info[
                                "discovered_cells"
                            ],
                    }
                )

                break

    env.close()

    # =========================================
    # 전체 결과
    # =========================================
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

    if not failures:
        return

    print()
    print("=== 실패 평균 ===")

    print(
        "Unique Positions     : "
        f"{np.mean([
            f['unique_positions']
            for f in failures
        ]):.2f}"
    )

    print(
        "Collisions           : "
        f"{np.mean([
            f['collision_count']
            for f in failures
        ]):.2f}"
    )

    print(
        "Unchanged Moves      : "
        f"{np.mean([
            f['unchanged_moves']
            for f in failures
        ]):.2f}"
    )

    print(
        "A-B-A Repeats        : "
        f"{np.mean([
            f['aba_count']
            for f in failures
        ]):.2f}"
    )

    print(
        "First Visits         : "
        f"{np.mean([
            f['first_visit_count']
            for f in failures
        ]):.2f}"
    )

    print(
        "Discovered Cells     : "
        f"{np.mean([
            f['discovered_cells']
            for f in failures
        ]):.2f}/81"
    )

    print(
        "Min Goal Distance    : "
        f"{np.mean([
            f['min_goal_distance']
            for f in failures
        ]):.2f}"
    )

    print(
        "Total Reward         : "
        f"{np.mean([
            f['total_reward']
            for f in failures
        ]):.2f}"
    )

    # =========================================
    # 실패 예시
    # =========================================
    print()
    print(
        "=== 실패 Episode 상세 "
        "(앞 10개) ==="
    )

    for failure in failures[:10]:
        print()
        print(
            "=" * 70
        )

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
            f"A-B-A Repeats    : "
            f"{failure['aba_count']}"
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
                f"  {position} "
                f"x {count}"
            )

        print()
        print(
            "--- 처음 30 Step ---"
        )

        max_print_steps = min(
            30,
            len(
                failure["actions"]
            ),
        )

        for i in range(
            max_print_steps
        ):
            before = (
                failure["path"][i]
            )

            after = (
                failure["path"][i + 1]
            )

            action = (
                failure["actions"][i]
            )

            reward = (
                failure["rewards"][i]
            )

            tile_reward = (
                failure[
                    "tile_rewards"
                ][i]
            )

            collision = (
                failure[
                    "collisions"
                ][i]
            )

            first_visit = (
                failure[
                    "first_visits"
                ][i]
            )

            print(
                f"{i + 1:3d} | "
                f"{position_text(before)} "
                f"--{ACTION_NAMES[action]:5s}--> "
                f"{position_text(after)} | "
                f"R {reward:5.1f} | "
                f"Tile {tile_reward:3.1f} | "
                f"First "
                f"{str(first_visit):5s} | "
                f"Collision "
                f"{collision}"
            )

        # 마지막 부분도 보여준다.
        print()
        print(
            "--- 마지막 20 Step ---"
        )

        start_index = max(
            0,
            len(
                failure["actions"]
            ) - 20,
        )

        for i in range(
            start_index,
            len(
                failure["actions"]
            ),
        ):
            before = (
                failure["path"][i]
            )

            after = (
                failure["path"][i + 1]
            )

            action = (
                failure["actions"][i]
            )

            reward = (
                failure["rewards"][i]
            )

            tile_reward = (
                failure[
                    "tile_rewards"
                ][i]
            )

            collision = (
                failure[
                    "collisions"
                ][i]
            )

            first_visit = (
                failure[
                    "first_visits"
                ][i]
            )

            print(
                f"{i + 1:3d} | "
                f"{position_text(before)} "
                f"--{ACTION_NAMES[action]:5s}--> "
                f"{position_text(after)} | "
                f"R {reward:5.1f} | "
                f"Tile {tile_reward:3.1f} | "
                f"First "
                f"{str(first_visit):5s} | "
                f"Collision "
                f"{collision}"
            )


if __name__ == "__main__":
    analyze()