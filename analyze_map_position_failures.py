from collections import Counter

import numpy as np
from stable_baselines3 import PPO

from partial_obstacle_map_reward_env import (
    PartialObstacleMapRewardEnv,
)


MODEL_PATH = "models/partial_obstacle_map_position_reward_ppo"

EPISODES = 500
MAX_FAILURE_EXAMPLES = 10


def manhattan_distance(a, b):
    return (
        abs(int(a[0]) - int(b[0]))
        + abs(int(a[1]) - int(b[1]))
    )


def print_discovered_map(discovered_map, agent_pos, goal_pos):
    print("마지막 누적 지도:")

    for y in range(discovered_map.shape[0]):
        row = []

        for x in range(discovered_map.shape[1]):
            if (
                x == agent_pos[0]
                and y == agent_pos[1]
            ):
                row.append("A")

            elif (
                x == goal_pos[0]
                and y == goal_pos[1]
                and discovered_map[y, x] != -1
            ):
                row.append("G")

            elif discovered_map[y, x] == -1:
                row.append("?")

            elif discovered_map[y, x] == 1:
                row.append("#")

            else:
                row.append(".")

        print(" ".join(row))


def analyze_episode(model, env, seed):
    obs, _ = env.reset(seed=seed)

    start = tuple(int(v) for v in env.agent_pos)
    goal = tuple(int(v) for v in env.goal_pos)

    path = [start]
    actions = []

    unchanged_moves = 0
    discovered_per_step = []

    min_goal_distance = manhattan_distance(
        env.agent_pos,
        env.goal_pos,
    )

    while True:
        before = tuple(
            int(v) for v in env.agent_pos
        )

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        action = int(action)

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        after = tuple(
            int(v) for v in env.agent_pos
        )

        actions.append(action)
        path.append(after)

        if before == after:
            unchanged_moves += 1

        discovered_per_step.append(
            info["newly_discovered"]
        )

        distance = manhattan_distance(
            env.agent_pos,
            env.goal_pos,
        )

        min_goal_distance = min(
            min_goal_distance,
            distance,
        )

        if terminated or truncated:
            break

    position_counts = Counter(path)

    unique_positions = len(position_counts)

    most_common_position, most_common_count = (
        position_counts.most_common(1)[0]
    )

    # A -> B -> A 형태가 얼마나 반복됐는지
    two_position_loops = 0

    for i in range(2, len(path)):
        if (
            path[i] == path[i - 2]
            and path[i] != path[i - 1]
        ):
            two_position_loops += 1

    total_discovered = int(
        np.sum(env.discovered_map != -1.0)
    )

    return {
        "success": terminated,
        "start": start,
        "goal": goal,
        "path": path,
        "actions": actions,
        "steps": len(actions),
        "unique_positions": unique_positions,
        "unchanged_moves": unchanged_moves,
        "most_common_position": most_common_position,
        "most_common_count": most_common_count,
        "two_position_loops": two_position_loops,
        "total_discovered": total_discovered,
        "min_goal_distance": min_goal_distance,
        "discovered_map": env.discovered_map.copy(),
        "final_position": tuple(
            int(v) for v in env.agent_pos
        ),
    }


def format_path(path, max_items=30):
    if len(path) <= max_items:
        return " -> ".join(
            str(pos) for pos in path
        )

    first = path[:15]
    last = path[-10:]

    return (
        " -> ".join(str(pos) for pos in first)
        + " -> ... -> "
        + " -> ".join(str(pos) for pos in last)
    )


def main():
    print("모델 로딩...")

    model = PPO.load(MODEL_PATH)

    env = PartialObstacleMapRewardEnv()

    failures = []
    success_count = 0

    for episode in range(EPISODES):
        seed = 10_000 + episode

        result = analyze_episode(
            model,
            env,
            seed,
        )

        if result["success"]:
            success_count += 1
        else:
            result["episode"] = episode
            result["seed"] = seed
            failures.append(result)

    print()
    print("====================================")
    print("전체 결과")
    print("====================================")

    print(f"평가 Episode : {EPISODES}")
    print(f"성공          : {success_count}")
    print(f"실패          : {len(failures)}")
    print(
        f"성공률        : "
        f"{success_count / EPISODES * 100:.1f}%"
    )

    if not failures:
        print("실패 Episode가 없습니다.")
        env.close()
        return

    print()
    print("====================================")
    print("실패 전체 평균")
    print("====================================")

    print(
        "평균 Unique Positions : "
        f"{np.mean([x['unique_positions'] for x in failures]):.2f}"
    )

    print(
        "평균 제자리 행동      : "
        f"{np.mean([x['unchanged_moves'] for x in failures]):.2f}"
    )

    print(
        "평균 A-B-A 반복       : "
        f"{np.mean([x['two_position_loops'] for x in failures]):.2f}"
    )

    print(
        "평균 발견 Map 칸      : "
        f"{np.mean([x['total_discovered'] for x in failures]):.2f} / 81"
    )

    print(
        "평균 최소 Goal 거리   : "
        f"{np.mean([x['min_goal_distance'] for x in failures]):.2f}"
    )

    print()
    print("====================================")
    print(
        f"실패 사례 {min(MAX_FAILURE_EXAMPLES, len(failures))}개"
    )
    print("====================================")

    action_names = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT",
    }

    for failure in failures[:MAX_FAILURE_EXAMPLES]:
        print()
        print("------------------------------------")
        print(
            f"Episode {failure['episode']} "
            f"(seed={failure['seed']})"
        )
        print("------------------------------------")

        print("Start :", failure["start"])
        print("Goal  :", failure["goal"])
        print("Final :", failure["final_position"])

        print("Steps :", failure["steps"])

        print(
            "Unique positions :",
            failure["unique_positions"],
        )

        print(
            "제자리 행동 :",
            failure["unchanged_moves"],
        )

        print(
            "A-B-A 반복 :",
            failure["two_position_loops"],
        )

        print(
            "가장 많이 방문 :",
            failure["most_common_position"],
            f"({failure['most_common_count']}회)",
        )

        print(
            "밝힌 Map :",
            f"{failure['total_discovered']} / 81",
        )

        print(
            "Goal 최소 거리 :",
            failure["min_goal_distance"],
        )

        print()
        print("이동 경로:")
        print(
            format_path(
                failure["path"]
            )
        )

        print()
        print("마지막 20개 Action:")

        last_actions = failure["actions"][-20:]

        print(
            " -> ".join(
                action_names[a]
                for a in last_actions
            )
        )

        print()

        print_discovered_map(
            failure["discovered_map"],
            failure["final_position"],
            failure["goal"],
        )

    env.close()


if __name__ == "__main__":
    main()