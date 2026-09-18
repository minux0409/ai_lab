from collections import Counter
import numpy as np
from stable_baselines3 import PPO

from partial_obstacle_compact_memory_env import PartialObstacleCompactMemoryEnv

MODEL_PATH = "models/partial_obstacle_compact_memory_ppo"

# Experiment I에서 확인된 실패 seed들
FAILURE_SEEDS = [
    10020, 10026, 10040, 10042, 10044, 10059,
    10076, 10132, 10187, 10232, 10237, 10312,
    10321, 10368, 10369, 10442, 10447, 10489,
]

STOCHASTIC_RUNS_PER_SEED = 100


def run_episode(model, env_seed, deterministic, action_seed=None):
    # env_seed는 동일하게 유지하여 같은 맵/Start/Goal을 사용한다.
    env = PartialObstacleCompactMemoryEnv()
    obs, _ = env.reset(seed=env_seed)

    # stochastic predict에서 PyTorch RNG를 사용하므로 반복 실행마다 seed를 바꾼다.
    if action_seed is not None:
        model.set_random_seed(action_seed)

    start = tuple(int(v) for v in env.agent_pos)
    goal = tuple(int(v) for v in env.goal_pos)

    positions = [start]
    unchanged = 0
    total_reward = 0.0
    success = False

    for _ in range(env.max_steps):
        prev = tuple(int(v) for v in env.agent_pos)

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(int(action))

        cur = tuple(int(v) for v in env.agent_pos)
        positions.append(cur)
        total_reward += float(reward)

        if cur == prev:
            unchanged += 1

        if terminated:
            success = True
            break
        if truncated:
            break

    result = {
        "success": success,
        "steps": len(positions) - 1,
        "reward": total_reward,
        "unique": len(set(positions)),
        "unchanged": unchanged,
        "start": start,
        "goal": goal,
        "final": positions[-1],
    }

    env.close()
    return result


def main():
    print("=== Experiment I: Deterministic vs Stochastic on Failure Seeds ===")
    print("Model:", MODEL_PATH)
    print("Failure seeds:", len(FAILURE_SEEDS))
    print("Stochastic runs per seed:", STOCHASTIC_RUNS_PER_SEED)

    model = PPO.load(MODEL_PATH)

    all_stochastic = []

    print("\n" + "=" * 100)
    print(
        f"{'Seed':>6} | {'Det':>4} | {'Stochastic Success':>20} | "
        f"{'Succ Steps':>10} | {'Fail Steps':>10} | {'Avg Unique':>10}"
    )
    print("-" * 100)

    for seed in FAILURE_SEEDS:
        det = run_episode(model, seed, deterministic=True)

        runs = []
        for i in range(STOCHASTIC_RUNS_PER_SEED):
            # 같은 환경 seed + 서로 다른 action sampling seed
            action_seed = seed * 1000 + i
            r = run_episode(
                model,
                env_seed=seed,
                deterministic=False,
                action_seed=action_seed,
            )
            runs.append(r)
            all_stochastic.append((seed, r))

        successes = [r for r in runs if r["success"]]
        failures = [r for r in runs if not r["success"]]

        success_rate = len(successes) / STOCHASTIC_RUNS_PER_SEED * 100
        succ_steps = np.mean([r["steps"] for r in successes]) if successes else np.nan
        fail_steps = np.mean([r["steps"] for r in failures]) if failures else np.nan
        avg_unique = np.mean([r["unique"] for r in runs])

        det_text = "OK" if det["success"] else "FAIL"

        print(
            f"{seed:6d} | {det_text:>4} | "
            f"{len(successes):3d}/{STOCHASTIC_RUNS_PER_SEED:<3d} "
            f"({success_rate:6.2f}%) | "
            f"{succ_steps:10.2f} | {fail_steps:10.2f} | {avg_unique:10.2f}"
        )

    total = len(all_stochastic)
    total_success = sum(1 for _, r in all_stochastic if r["success"])
    successful_steps = [
        r["steps"] for _, r in all_stochastic if r["success"]
    ]

    per_seed_rates = []
    for seed in FAILURE_SEEDS:
        seed_runs = [r for s, r in all_stochastic if s == seed]
        rate = np.mean([1.0 if r["success"] else 0.0 for r in seed_runs])
        per_seed_rates.append((seed, rate))

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(
        f"Deterministic: 0/{len(FAILURE_SEEDS)} success "
        "(these are the known Experiment I failure seeds)"
    )
    print(
        f"Stochastic   : {total_success}/{total} success "
        f"({total_success / total * 100:.2f}%)"
    )

    if successful_steps:
        print(
            "Stochastic avg steps on successful runs:",
            f"{np.mean(successful_steps):.2f}"
        )

    print("\nPer-seed stochastic success ranking:")
    for seed, rate in sorted(per_seed_rates, key=lambda x: x[1], reverse=True):
        print(f"  Seed {seed}: {rate * 100:6.2f}%")

    escaped_seeds = sum(1 for _, rate in per_seed_rates if rate > 0)
    mostly_solved = sum(1 for _, rate in per_seed_rates if rate >= 0.80)

    print("\nInterpretation helpers:")
    print(
        f"  Seeds with at least one stochastic success: "
        f"{escaped_seeds}/{len(FAILURE_SEEDS)}"
    )
    print(
        f"  Seeds with >=80% stochastic success: "
        f"{mostly_solved}/{len(FAILURE_SEEDS)}"
    )
    print(
        "\nIf stochastic succeeds frequently on seeds that deterministic always fails, "
        "the policy likely contains viable escape actions but deterministic argmax "
        "locks it into a repeated local choice."
    )


if __name__ == "__main__":
    main()
