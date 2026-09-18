from collections import Counter
import numpy as np
from stable_baselines3 import PPO

from partial_obstacle_direction_memory_env import PartialObstacleDirectionMemoryEnv


MODEL_PATH = "models/partial_obstacle_direction_memory_ppo"
NUM_EPISODES = 500
SEED_BASE = 10000

# Experiment I deterministic failure seeds
I_FAILURE_SEEDS = {
    10020, 10026, 10040, 10042, 10044, 10059,
    10076, 10132, 10187, 10232, 10237, 10312,
    10321, 10368, 10369, 10442, 10447, 10489,
}


def longest_same_position_run(positions):
    best = 1
    current = 1

    for i in range(1, len(positions)):
        if positions[i] == positions[i - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1

    return best


def longest_periodic_cycle(positions, min_period=2, max_period=10):
    """
    period 2~10의 반복 위치 패턴을 찾는다.
    반환값: (반복 길이, period)
    """
    best_length = 0
    best_period = 0
    n = len(positions)

    for period in range(min_period, max_period + 1):
        for start in range(0, n - period):
            length = 0
            i = start + period

            while (
                i < n
                and positions[i] == positions[i - period]
            ):
                length += 1
                i += 1

            if length > best_length:
                best_length = length
                best_period = period

    return best_length, best_period


def run_episode(model, seed):
    env = PartialObstacleDirectionMemoryEnv()
    obs, _ = env.reset(seed=seed)

    start = tuple(int(v) for v in env.agent_pos)
    goal = tuple(int(v) for v in env.goal_pos)

    positions = [start]
    actions = []
    direction_snapshots = []

    success = False
    unchanged = 0

    min_goal_distance = (
        abs(start[0] - goal[0])
        + abs(start[1] - goal[1])
    )

    for _ in range(env.max_steps):
        x = int(env.agent_pos[0])
        y = int(env.agent_pos[1])

        # 행동 직전 현재 칸의 실제 raw direction score 기록
        before_scores = env.direction_scores[y, x].copy()

        action, _ = model.predict(
            obs,
            deterministic=True,
        )
        action = int(action)

        previous_position = tuple(
            int(v) for v in env.agent_pos
        )

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        current_position = tuple(
            int(v) for v in env.agent_pos
        )

        actions.append(action)
        direction_snapshots.append(
            (
                previous_position,
                before_scores,
                action,
            )
        )
        positions.append(current_position)

        if current_position == previous_position:
            unchanged += 1

        distance = (
            abs(current_position[0] - goal[0])
            + abs(current_position[1] - goal[1])
        )
        min_goal_distance = min(
            min_goal_distance,
            distance,
        )

        if terminated:
            success = True
            break

        if truncated:
            break

    cycle_length, cycle_period = (
        longest_periodic_cycle(positions)
    )

    result = {
        "seed": seed,
        "success": success,
        "steps": len(actions),
        "start": start,
        "goal": goal,
        "final": positions[-1],
        "positions": positions,
        "actions": actions,
        "direction_snapshots": direction_snapshots,
        "unique": len(set(positions)),
        "unchanged": unchanged,
        "same_run": longest_same_position_run(
            positions
        ),
        "cycle_length": cycle_length,
        "cycle_period": cycle_period,
        "min_goal_distance": min_goal_distance,
    }

    env.close()
    return result


def classify(result):
    # 한 자리에서 장시간 행동을 반복
    if (
        result["same_run"] >= 20
        or result["unchanged"] >= 50
    ):
        return "STUCK"

    # period 2~10 반복 cycle
    if result["cycle_length"] >= 20:
        return f"CYCLE_P{result['cycle_period']}"

    return "OTHER"


def action_name(action):
    return ["UP", "DOWN", "LEFT", "RIGHT"][action]


def main():
    print("=== Experiment K Failure Analysis ===")
    print("Model:", MODEL_PATH)
    print("Evaluation: deterministic, 500 fixed maps")
    print()

    model = PPO.load(MODEL_PATH)

    results = [
        run_episode(
            model,
            SEED_BASE + episode,
        )
        for episode in range(NUM_EPISODES)
    ]

    successes = [
        r for r in results
        if r["success"]
    ]
    failures = [
        r for r in results
        if not r["success"]
    ]

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

    counts = Counter(
        classify(r)
        for r in failures
    )

    print("\nFailure patterns:")
    for name, count in counts.most_common():
        print(
            f"  {name:10s}: "
            f"{count:2d}/{len(failures)} "
            f"({count / len(failures) * 100:6.2f}%)"
        )

    if failures:
        print("\nFailure raw:")
        print(
            f"  Avg Unique Positions      : "
            f"{np.mean([r['unique'] for r in failures]):.2f}"
        )
        print(
            f"  Avg Unchanged Moves       : "
            f"{np.mean([r['unchanged'] for r in failures]):.2f}"
        )
        print(
            f"  Avg Max Same Position Run : "
            f"{np.mean([r['same_run'] for r in failures]):.2f}"
        )
        print(
            f"  Avg Periodic Cycle Length : "
            f"{np.mean([r['cycle_length'] for r in failures]):.2f}"
        )
        print(
            f"  Avg Minimum Goal Distance : "
            f"{np.mean([r['min_goal_distance'] for r in failures]):.2f}"
        )

    by_seed = {
        r["seed"]: r
        for r in results
    }

    print("\n" + "=" * 100)
    print("EXPERIMENT I'S 18 FAILURE SEEDS -> EXPERIMENT K")
    print("=" * 100)

    fixed = []
    still_failed = []

    for seed in sorted(I_FAILURE_SEEDS):
        result = by_seed[seed]

        if result["success"]:
            fixed.append(seed)
            print(
                f"  Seed {seed}: "
                f"SUCCESS in {result['steps']} steps"
            )
        else:
            still_failed.append(seed)
            print(
                f"  Seed {seed}: "
                f"FAIL {classify(result)} | "
                f"unique={result['unique']} | "
                f"unchanged={result['unchanged']} | "
                f"cycle=P{result['cycle_period']}/"
                f"{result['cycle_length']} | "
                f"minGoal={result['min_goal_distance']}"
            )

    print()
    print(
        f"Old I failures fixed by K : "
        f"{len(fixed)}/{len(I_FAILURE_SEEDS)}"
    )
    print(
        f"Old I failures still fail : "
        f"{len(still_failed)}/{len(I_FAILURE_SEEDS)}"
    )

    print("\n" + "=" * 100)
    print("NEW FAILURES INTRODUCED BY K")
    print("=" * 100)

    new_failures = [
        r for r in failures
        if r["seed"] not in I_FAILURE_SEEDS
    ]

    print(
        f"K failures not present in I's 18 failures: "
        f"{len(new_failures)}"
    )

    for result in new_failures:
        print(
            f"  Seed {result['seed']}: "
            f"{classify(result):10s} | "
            f"unique={result['unique']:2d} | "
            f"unchanged={result['unchanged']:3d} | "
            f"cycle=P{result['cycle_period']}/"
            f"{result['cycle_length']:2d} | "
            f"minGoal={result['min_goal_distance']:2d}"
        )

    print("\n" + "=" * 100)
    print("ALL K FAILURES")
    print("=" * 100)

    for result in failures:
        print(
            f"\nSeed {result['seed']} | "
            f"{classify(result)} | "
            f"Start {result['start']} -> "
            f"Goal {result['goal']} | "
            f"Unique {result['unique']} | "
            f"MinGoal {result['min_goal_distance']}"
        )

        positions = result["positions"]

        if len(positions) <= 35:
            print("  Path:", positions)
        else:
            print(
                "  First 15:",
                positions[:15],
            )
            print(
                "  Last 20 :",
                positions[-20:],
            )

        # 마지막 12회 행동에서 PPO가 실제로 어떤 방향 점수를
        # 보고도 같은 행동을 선택했는지 확인.
        print("  Last direction decisions:")

        for (
            position,
            scores,
            action,
        ) in result["direction_snapshots"][-12:]:
            print(
                f"    pos={position} "
                f"scores="
                f"[U:{scores[0]:5.1f}, "
                f"D:{scores[1]:5.1f}, "
                f"L:{scores[2]:5.1f}, "
                f"R:{scores[3]:5.1f}] "
                f"-> {action_name(action)}"
            )


if __name__ == "__main__":
    main()
