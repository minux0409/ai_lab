from collections import Counter
import numpy as np
from stable_baselines3 import PPO

from partial_obstacle_compact_memory_env import PartialObstacleCompactMemoryEnv

MODEL_PATH = "models/partial_obstacle_compact_memory_ppo"
EPISODES = 500

ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}


def longest_same_position_run(positions):
    if not positions:
        return 0
    longest = current = 1
    for i in range(1, len(positions)):
        if positions[i] == positions[i - 1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def longest_ab_oscillation(positions):
    longest = current = 0
    for i in range(2, len(positions)):
        if positions[i] == positions[i - 2] and positions[i] != positions[i - 1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def classify_failure(unchanged_moves, max_same_run, longest_ab):
    # Experiment F 분석과 같은 분류 기준
    if unchanged_moves >= 50 or max_same_run >= 20:
        return "STUCK"
    if longest_ab >= 10:
        return "A_B_LOOP"
    return "OTHER"


def analyze_episode(model, seed):
    env = PartialObstacleCompactMemoryEnv()
    obs, _ = env.reset(seed=seed)

    start = tuple(int(v) for v in env.agent_pos)
    goal = tuple(int(v) for v in env.goal_pos)
    positions = [start]
    actions = []
    unchanged_moves = 0
    min_goal_distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    total_reward = 0.0
    success = False

    for step in range(1, env.max_steps + 1):
        prev = tuple(int(v) for v in env.agent_pos)
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        cur = tuple(int(v) for v in env.agent_pos)

        positions.append(cur)
        actions.append(action)
        total_reward += float(reward)

        if cur == prev:
            unchanged_moves += 1

        dist = abs(cur[0] - goal[0]) + abs(cur[1] - goal[1])
        min_goal_distance = min(min_goal_distance, dist)

        if terminated:
            success = True
            break
        if truncated:
            break

    final = positions[-1]
    final_goal_distance = abs(final[0] - goal[0]) + abs(final[1] - goal[1])
    unique_positions = len(set(positions))
    max_same_run = longest_same_position_run(positions)
    longest_ab = longest_ab_oscillation(positions)

    result = {
        "seed": seed,
        "success": success,
        "steps": len(actions),
        "start": start,
        "goal": goal,
        "final": final,
        "unique_positions": unique_positions,
        "unchanged_moves": unchanged_moves,
        "max_same_position_run": max_same_run,
        "longest_ab_loop": longest_ab,
        "minimum_goal_distance": min_goal_distance,
        "final_goal_distance": final_goal_distance,
        "total_reward": total_reward,
        "actions": actions,
        "positions": positions,
        "failure_type": None,
    }

    if not success:
        result["failure_type"] = classify_failure(
            unchanged_moves, max_same_run, longest_ab
        )

    env.close()
    return result


def print_failure(r):
    counts = Counter(r["actions"])
    print(f'\n[{r["failure_type"]}] Seed {r["seed"]}')
    print("Start:", r["start"], "| Goal:", r["goal"], "| Final:", r["final"])
    print("Unique Positions       :", r["unique_positions"])
    print("Unchanged Moves        :", r["unchanged_moves"])
    print("Max Same Position Run  :", r["max_same_position_run"])
    print("Longest A-B Loop       :", r["longest_ab_loop"])
    print("Minimum Goal Distance  :", r["minimum_goal_distance"])
    print("Final Goal Distance    :", r["final_goal_distance"])
    print("Total Reward           :", f'{r["total_reward"]:.2f}')
    print("Actions:")
    for a in range(4):
        print(f"  {ACTION_NAMES[a]:5}: {counts[a]}")
    print("Last 20 positions:")
    print(" ".join(str(x) for x in r["positions"][-20:]))


def main():
    print("\n=== Experiment I Failure Analysis ===")
    print("Model:", MODEL_PATH)
    print("Episodes:", EPISODES)

    model = PPO.load(MODEL_PATH)
    results = [
        analyze_episode(model, 10_000 + episode)
        for episode in range(EPISODES)
    ]

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    print("\n==============================================")
    print("OVERALL")
    print("==============================================")
    print("Episodes :", len(results))
    print("Success  :", len(successes), f"({len(successes)/EPISODES*100:.2f}%)")
    print("Failure  :", len(failures), f"({len(failures)/EPISODES*100:.2f}%)")

    fc = Counter(r["failure_type"] for r in failures)
    print("\n==============================================")
    print("FAILURE PATTERNS")
    print("==============================================")
    for kind in ("STUCK", "A_B_LOOP", "OTHER"):
        n = fc[kind]
        pct = n / len(failures) * 100 if failures else 0
        print(f"{kind:<10}: {n:3} ({pct:6.2f}%)")

    if failures:
        def avg(key):
            return float(np.mean([r[key] for r in failures]))

        print("\n==============================================")
        print("FAILURE RAW STATISTICS")
        print("==============================================")
        print("Avg Unique Positions       :", f'{avg("unique_positions"):.2f}')
        print("Avg Unchanged Moves        :", f'{avg("unchanged_moves"):.2f}')
        print("Avg Max Same Position Run  :", f'{avg("max_same_position_run"):.2f}')
        print("Avg Longest A-B Loop       :", f'{avg("longest_ab_loop"):.2f}')
        print("Avg Minimum Goal Distance  :", f'{avg("minimum_goal_distance"):.2f}')
        print("Avg Final Goal Distance    :", f'{avg("final_goal_distance"):.2f}')

        near = [r for r in failures if r["minimum_goal_distance"] <= 1]
        print(
            "\nGoal distance <= 1 then failed:",
            len(near), "/", len(failures),
            f"({len(near)/len(failures)*100:.2f}%)"
        )

        print("\n==============================================")
        print("ALL FAILURES")
        print("==============================================")
        # 실패가 18개 정도라 전부 출력해도 분석하기 좋다.
        for r in failures:
            print_failure(r)


if __name__ == "__main__":
    main()
