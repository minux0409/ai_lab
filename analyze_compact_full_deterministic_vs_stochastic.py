import numpy as np
from stable_baselines3 import PPO

from partial_obstacle_compact_memory_env import PartialObstacleCompactMemoryEnv

MODEL_PATH = "models/partial_obstacle_compact_memory_ppo"
NUM_MAPS = 500
STOCHASTIC_RUNS_PER_MAP = 20
EVAL_SEED_BASE = 10000


def run_episode(model, env_seed, deterministic, action_seed=None):
    env = PartialObstacleCompactMemoryEnv()
    obs, _ = env.reset(seed=env_seed)

    if action_seed is not None:
        model.set_random_seed(action_seed)

    steps = 0
    total_reward = 0.0
    success = False

    for _ in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        steps += 1
        total_reward += float(reward)

        if terminated:
            success = True
            break
        if truncated:
            break

    env.close()
    return success, steps, total_reward


def main():
    print("=== Experiment I: Full 500 Maps Deterministic vs Stochastic ===")
    print("Model:", MODEL_PATH)
    print("Maps:", NUM_MAPS)
    print("Stochastic runs per map:", STOCHASTIC_RUNS_PER_MAP)

    model = PPO.load(MODEL_PATH)

    det_results = {}
    stochastic_results = {}

    print("\nRunning deterministic evaluation...")
    for i in range(NUM_MAPS):
        seed = EVAL_SEED_BASE + i
        success, steps, reward = run_episode(
            model, seed, deterministic=True
        )
        det_results[seed] = {
            "success": success,
            "steps": steps,
            "reward": reward,
        }

    print("Running stochastic evaluation...")
    for i in range(NUM_MAPS):
        seed = EVAL_SEED_BASE + i
        runs = []

        for j in range(STOCHASTIC_RUNS_PER_MAP):
            action_seed = seed * 1000 + j
            success, steps, reward = run_episode(
                model,
                seed,
                deterministic=False,
                action_seed=action_seed,
            )
            runs.append({
                "success": success,
                "steps": steps,
                "reward": reward,
            })

        stochastic_results[seed] = runs

    # Deterministic summary
    det_successes = [
        r for r in det_results.values() if r["success"]
    ]
    det_failures = [
        r for r in det_results.values() if not r["success"]
    ]

    det_success_rate = len(det_successes) / NUM_MAPS * 100
    det_success_steps = (
        np.mean([r["steps"] for r in det_successes])
        if det_successes else np.nan
    )

    # Stochastic summary over all executions
    all_stoch = [
        r
        for runs in stochastic_results.values()
        for r in runs
    ]
    stoch_successes = [r for r in all_stoch if r["success"]]
    stoch_failures = [r for r in all_stoch if not r["success"]]

    stoch_success_rate = len(stoch_successes) / len(all_stoch) * 100
    stoch_success_steps = (
        np.mean([r["steps"] for r in stoch_successes])
        if stoch_successes else np.nan
    )

    # Map-level comparison
    det_success_seeds = [
        seed for seed, r in det_results.items() if r["success"]
    ]
    det_failure_seeds = [
        seed for seed, r in det_results.items() if not r["success"]
    ]

    def stochastic_rate(seed):
        runs = stochastic_results[seed]
        return sum(r["success"] for r in runs) / len(runs)

    # How well stochastic preserves deterministic-success maps
    preserved_rates = [stochastic_rate(seed) for seed in det_success_seeds]
    rescued_rates = [stochastic_rate(seed) for seed in det_failure_seeds]

    preserved_all = sum(rate == 1.0 for rate in preserved_rates)
    preserved_majority = sum(rate >= 0.5 for rate in preserved_rates)
    damaged_any = sum(rate < 1.0 for rate in preserved_rates)
    damaged_majority = sum(rate < 0.5 for rate in preserved_rates)

    rescued_any = sum(rate > 0.0 for rate in rescued_rates)
    rescued_majority = sum(rate >= 0.5 for rate in rescued_rates)
    rescued_all = sum(rate == 1.0 for rate in rescued_rates)

    print("\n" + "=" * 100)
    print("OVERALL RESULT")
    print("=" * 100)

    print(
        f"Deterministic: {len(det_successes)}/{NUM_MAPS} "
        f"({det_success_rate:.2f}%)"
    )
    print(f"  Avg successful steps: {det_success_steps:.2f}")

    print(
        f"\nStochastic: {len(stoch_successes)}/{len(all_stoch)} "
        f"({stoch_success_rate:.2f}%)"
    )
    print(f"  Avg successful steps: {stoch_success_steps:.2f}")

    print("\n" + "=" * 100)
    print("DETERMINISTIC-SUCCESS MAPS")
    print("=" * 100)
    print(f"Maps: {len(det_success_seeds)}")
    print(
        f"Stochastic success rate on these maps: "
        f"{np.mean(preserved_rates) * 100:.2f}%"
    )
    print(
        f"100% preserved: {preserved_all}/{len(det_success_seeds)}"
    )
    print(
        f">=50% stochastic success: "
        f"{preserved_majority}/{len(det_success_seeds)}"
    )
    print(
        f"At least one stochastic failure: "
        f"{damaged_any}/{len(det_success_seeds)}"
    )
    print(
        f"<50% stochastic success: "
        f"{damaged_majority}/{len(det_success_seeds)}"
    )

    print("\n" + "=" * 100)
    print("DETERMINISTIC-FAILURE MAPS")
    print("=" * 100)
    print(f"Maps: {len(det_failure_seeds)}")

    if det_failure_seeds:
        print(
            f"Stochastic success rate on these maps: "
            f"{np.mean(rescued_rates) * 100:.2f}%"
        )
        print(
            f"At least one stochastic success: "
            f"{rescued_any}/{len(det_failure_seeds)}"
        )
        print(
            f">=50% stochastic success: "
            f"{rescued_majority}/{len(det_failure_seeds)}"
        )
        print(
            f"100% stochastic success: "
            f"{rescued_all}/{len(det_failure_seeds)}"
        )

        print("\nPer deterministic-failure seed:")
        for seed in det_failure_seeds:
            rate = stochastic_rate(seed)
            print(f"  Seed {seed}: {rate * 100:6.2f}%")

    # Useful extremes among originally successful maps
    weak_preserved = sorted(
        [(seed, stochastic_rate(seed)) for seed in det_success_seeds],
        key=lambda x: x[1]
    )[:15]

    print("\n" + "=" * 100)
    print("LOWEST STOCHASTIC RATES AMONG DETERMINISTIC-SUCCESS MAPS")
    print("=" * 100)
    for seed, rate in weak_preserved:
        print(f"  Seed {seed}: {rate * 100:6.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
