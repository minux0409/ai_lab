from collections import Counter
import numpy as np
from stable_baselines3 import PPO

from partial_obstacle_visit_penalty_env import PartialObstacleVisitPenaltyEnv

MODEL_PATH = "models/partial_obstacle_visit_penalty_ppo"
NUM_EPISODES = 500
SEED_BASE = 10000


def longest_same_position_run(positions):
    best = 1
    cur = 1
    for i in range(1, len(positions)):
        if positions[i] == positions[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def longest_periodic_cycle(positions, min_period=2, max_period=8):
    """
    연속 위치열에서 period 2~8의 반복 cycle을 찾는다.
    반환: (반복된 transition 수에 가까운 길이, period)
    """
    best_len = 0
    best_period = 0

    n = len(positions)
    for p in range(min_period, max_period + 1):
        for start in range(0, n - 2 * p):
            length = 0
            i = start + p
            while i < n and positions[i] == positions[i - p]:
                length += 1
                i += 1
            if length > best_len:
                best_len = length
                best_period = p

    return best_len, best_period


def run_episode(model, seed):
    env = PartialObstacleVisitPenaltyEnv()
    obs, _ = env.reset(seed=seed)

    start = tuple(int(v) for v in env.agent_pos)
    goal = tuple(int(v) for v in env.goal_pos)

    positions = [start]
    rewards = []
    revisit_penalties = []
    unchanged = 0
    success = False
    min_goal_distance = (
        abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    )

    for _ in range(env.max_steps):
        prev = tuple(int(v) for v in env.agent_pos)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        cur = tuple(int(v) for v in env.agent_pos)
        positions.append(cur)
        rewards.append(float(reward))
        revisit_penalties.append(float(info.get("revisit_penalty", 0.0)))

        if cur == prev:
            unchanged += 1

        dist = abs(cur[0] - goal[0]) + abs(cur[1] - goal[1])
        min_goal_distance = min(min_goal_distance, dist)

        if terminated:
            success = True
            break
        if truncated:
            break

    cycle_len, cycle_period = longest_periodic_cycle(positions)

    result = {
        "seed": seed,
        "success": success,
        "steps": len(positions) - 1,
        "start": start,
        "goal": goal,
        "final": positions[-1],
        "unique": len(set(positions)),
        "unchanged": unchanged,
        "same_run": longest_same_position_run(positions),
        "cycle_len": cycle_len,
        "cycle_period": cycle_period,
        "min_goal_dist": min_goal_distance,
        "total_reward": sum(rewards),
        "total_revisit_penalty": sum(revisit_penalties),
        "positions": positions,
    }

    env.close()
    return result


def classify(r):
    # 거의 한 자리에서 끝까지 버틴 경우
    if r["same_run"] >= 20 or r["unchanged"] >= 50:
        return "STUCK"

    # period 2~8 반복이 20 step 이상 이어진 경우
    if r["cycle_len"] >= 20:
        return f"CYCLE_P{r['cycle_period']}"

    return "OTHER"


def main():
    print("=== Experiment J Failure Analysis ===")
    print("Model:", MODEL_PATH)
    print("Evaluation: deterministic, 500 fixed maps")
    print()

    model = PPO.load(MODEL_PATH)

    results = [
        run_episode(model, SEED_BASE + i)
        for i in range(NUM_EPISODES)
    ]

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    print("=" * 100)
    print("OVERALL")
    print("=" * 100)
    print(
        f"Success: {len(successes)}/{NUM_EPISODES} "
        f"({len(successes) / NUM_EPISODES * 100:.2f}%)"
    )
    print(
        f"Failure: {len(failures)}/{NUM_EPISODES} "
        f"({len(failures) / NUM_EPISODES * 100:.2f}%)"
    )

    counts = Counter(classify(r) for r in failures)

    print("\nFailure patterns:")
    for name, count in counts.most_common():
        print(
            f"  {name:10s}: {count:3d}/{len(failures)} "
            f"({count / len(failures) * 100:6.2f}%)"
        )

    if failures:
        print("\nFailure raw:")
        print(f"  Avg Unique Positions       : {np.mean([r['unique'] for r in failures]):.2f}")
        print(f"  Avg Unchanged Moves        : {np.mean([r['unchanged'] for r in failures]):.2f}")
        print(f"  Avg Max Same Position Run  : {np.mean([r['same_run'] for r in failures]):.2f}")
        print(f"  Avg Periodic Cycle Length  : {np.mean([r['cycle_len'] for r in failures]):.2f}")
        print(f"  Avg Minimum Goal Distance  : {np.mean([r['min_goal_dist'] for r in failures]):.2f}")
        print(
            f"  Avg Total Revisit Penalty  : "
            f"{np.mean([r['total_revisit_penalty'] for r in failures]):.2f}"
        )

    print("\n" + "=" * 100)
    print("COMPARISON WITH EXPERIMENT I'S 18 KNOWN FAILURE SEEDS")
    print("=" * 100)

    i_failure_seeds = {
        10020, 10026, 10040, 10042, 10044, 10059,
        10076, 10132, 10187, 10232, 10237, 10312,
        10321, 10368, 10369, 10442, 10447, 10489,
    }

    j_by_seed = {r["seed"]: r for r in results}

    old_fixed = []
    old_still_failed = []

    for seed in sorted(i_failure_seeds):
        r = j_by_seed[seed]
        if r["success"]:
            old_fixed.append(seed)
            status = f"SUCCESS in {r['steps']} steps"
        else:
            old_still_failed.append(seed)
            status = (
                f"FAIL {classify(r)} | unique={r['unique']} "
                f"| unchanged={r['unchanged']} "
                f"| cycle=P{r['cycle_period']}/{r['cycle_len']} "
                f"| minGoal={r['min_goal_dist']}"
            )
        print(f"  Seed {seed}: {status}")

    print(
        f"\nOld I failures fixed by J: "
        f"{len(old_fixed)}/{len(i_failure_seeds)}"
    )
    print(
        f"Old I failures still failing: "
        f"{len(old_still_failed)}/{len(i_failure_seeds)}"
    )

    print("\n" + "=" * 100)
    print("NEW FAILURES INTRODUCED BY J")
    print("=" * 100)

    new_failures = [
        r for r in failures
        if r["seed"] not in i_failure_seeds
    ]

    print(
        f"J failures that were NOT among I's 18 failures: "
        f"{len(new_failures)}"
    )

    for r in new_failures:
        print(
            f"  Seed {r['seed']}: {classify(r):10s} "
            f"| unique={r['unique']:2d} "
            f"| unchanged={r['unchanged']:3d} "
            f"| cycle=P{r['cycle_period']}/{r['cycle_len']:2d} "
            f"| minGoal={r['min_goal_dist']:2d} "
            f"| revisitPenalty={r['total_revisit_penalty']:7.1f}"
        )

    print("\n" + "=" * 100)
    print("REPRESENTATIVE FAILURE PATHS")
    print("=" * 100)

    # 최대 15개만 상세 경로 출력
    for r in failures[:15]:
        print(
            f"\nSeed {r['seed']} | {classify(r)} | "
            f"Start {r['start']} -> Goal {r['goal']} | "
            f"Unique {r['unique']} | MinGoal {r['min_goal_dist']} | "
            f"RevisitPenalty {r['total_revisit_penalty']:.1f}"
        )

        pos = r["positions"]
        if len(pos) <= 35:
            print("  Path:", pos)
        else:
            print("  First 15:", pos[:15])
            print("  Last  20:", pos[-20:])


if __name__ == "__main__":
    main()
