from collections import Counter

from stable_baselines3 import PPO

from partial_obstacle_env import PartialObstacleGridEnv


MODEL_PATH = "models/partial_obstacle_ppo"

MAX_SUCCESS_CASES = 5
MAX_FAILURE_CASES = 5
EVAL_EPISODES = 500


def analyze_episode(model, seed):
    env = PartialObstacleGridEnv()

    obs, _ = env.reset(seed=seed)

    path = [tuple(env.agent_pos)]
    actions = []
    rewards = []

    while True:
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        action = int(action)

        obs, reward, terminated, truncated, _ = env.step(action)

        actions.append(action)
        rewards.append(reward)
        path.append(tuple(env.agent_pos))

        if terminated or truncated:
            break

    result = {
        "seed": seed,
        "success": terminated,
        "steps": len(actions),
        "path": path,
        "actions": actions,
        "rewards": rewards,
        "agent_start": path[0],
        "goal": tuple(env.goal_pos),
        "grid": env.grid.copy(),
    }

    env.close()

    return result


def count_repeated_positions(path):
    counts = Counter(path)

    repeated = {
        pos: count
        for pos, count in counts.items()
        if count > 1
    }

    return repeated


def count_back_and_forth(path):
    """
    A -> B -> A 같은 왕복 패턴 횟수
    """
    count = 0

    for i in range(len(path) - 2):
        if path[i] == path[i + 2]:
            count += 1

    return count


def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def min_goal_distance(path, goal):
    return min(
        manhattan_distance(pos, goal)
        for pos in path
    )


def print_case(title, case):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(f"Seed: {case['seed']}")
    print(f"Start: {case['agent_start']}")
    print(f"Goal: {case['goal']}")
    print(f"Steps: {case['steps']}")
    print(
        "Result:",
        "SUCCESS" if case["success"] else "FAILURE"
    )

    repeated = count_repeated_positions(case["path"])
    back_and_forth = count_back_and_forth(case["path"])
    min_distance = min_goal_distance(
        case["path"],
        case["goal"],
    )

    print(f"반복 방문 위치 수: {len(repeated)}")
    print(f"A→B→A 왕복 횟수: {back_and_forth}")
    print(f"Goal 최소 Manhattan 거리: {min_distance}")

    if repeated:
        print("\n가장 많이 방문한 위치:")

        sorted_positions = sorted(
            repeated.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for pos, count in sorted_positions[:5]:
            print(
                f"  {pos}: {count}회"
            )

    print("\n이동 경로:")

    print(
        " -> ".join(
            str(pos)
            for pos in case["path"]
        )
    )


def render_case(case):
    grid = case["grid"]

    height, width = grid.shape

    output = []

    for y in range(height):
        row = []

        for x in range(width):
            pos = (x, y)

            if pos == case["agent_start"]:
                row.append("S")

            elif pos == case["goal"]:
                row.append("G")

            elif grid[y, x] == 1:
                row.append("#")

            elif pos in case["path"]:
                row.append("*")

            else:
                row.append(".")

        output.append(
            " ".join(row)
        )

    print("\n경로 시각화:")
    print("S = Start")
    print("G = Goal")
    print("# = 장애물")
    print("* = Agent가 지나간 위치")
    print()

    for row in output:
        print(row)


if __name__ == "__main__":
    model = PPO.load(MODEL_PATH)

    success_cases = []
    failure_cases = []

    print(
        f"{EVAL_EPISODES}개 고정 평가 맵 분석 시작..."
    )

    for episode in range(EVAL_EPISODES):
        seed = 10_000 + episode

        case = analyze_episode(
            model,
            seed,
        )

        if case["success"]:
            if len(success_cases) < MAX_SUCCESS_CASES:
                success_cases.append(case)

        else:
            if len(failure_cases) < MAX_FAILURE_CASES:
                failure_cases.append(case)

        if (
            len(success_cases) >= MAX_SUCCESS_CASES
            and len(failure_cases) >= MAX_FAILURE_CASES
        ):
            break

    print(
        f"\n성공 사례 확보: {len(success_cases)}"
    )

    print(
        f"실패 사례 확보: {len(failure_cases)}"
    )

    print("\n\n######## SUCCESS CASES ########")

    for i, case in enumerate(
        success_cases,
        start=1,
    ):
        print_case(
            f"SUCCESS CASE {i}",
            case,
        )

        render_case(case)

    print("\n\n######## FAILURE CASES ########")

    for i, case in enumerate(
        failure_cases,
        start=1,
    ):
        print_case(
            f"FAILURE CASE {i}",
            case,
        )

        render_case(case)