import copy
import numpy as np
import torch
from stable_baselines3 import PPO

from partial_obstacle_direction_memory_env import PartialObstacleDirectionMemoryEnv


MODEL_PATH = "models/partial_obstacle_direction_memory_ppo"

# K에서 실제로 문제가 있었던 대표 seed.
# 성공 에피소드의 상태도 함께 수집해서 특정 실패 상태에만 치우치지 않게 본다.
SEEDS = [
    10000, 10001, 10002, 10003, 10004,
    10020, 10040, 10042, 10044, 10059,
    10132, 10312, 10489,
]

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

# direction score만 바꾼 counterfactual 입력.
# 총합 40을 유지한다.
PATTERNS = {
    "BALANCED": np.array([10, 10, 10, 10], dtype=np.float32),
    "UP_USED": np.array([0, 10, 15, 15], dtype=np.float32),
    "DOWN_USED": np.array([10, 0, 15, 15], dtype=np.float32),
    "LEFT_USED": np.array([15, 15, 0, 10], dtype=np.float32),
    "RIGHT_USED": np.array([15, 15, 10, 0], dtype=np.float32),
    "UP_HIGH": np.array([25, 5, 5, 5], dtype=np.float32),
    "DOWN_HIGH": np.array([5, 25, 5, 5], dtype=np.float32),
    "LEFT_HIGH": np.array([5, 5, 25, 5], dtype=np.float32),
    "RIGHT_HIGH": np.array([5, 5, 5, 25], dtype=np.float32),
}


def get_action_probs(model, obs):
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        probs = distribution.distribution.probs.detach().cpu().numpy()[0]
    return probs


def collect_states(model):
    """
    실제 K rollout에서 observation을 모은다.
    direction score가 아닌 앞 17개 feature는 실제 상태 그대로 보존한다.
    """
    states = []

    for seed in SEEDS:
        env = PartialObstacleDirectionMemoryEnv()
        obs, _ = env.reset(seed=seed)

        for step in range(env.max_steps):
            states.append({
                "seed": seed,
                "step": step,
                "position": tuple(int(v) for v in env.agent_pos),
                "goal": tuple(int(v) for v in env.goal_pos),
                "obs": obs.copy(),
            })

            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))

            if terminated or truncated:
                break

        env.close()

    return states


def counterfactual_obs(original_obs, raw_direction_scores):
    result = original_obs.copy()

    # K 환경과 동일하게 /40 정규화
    result[17:21] = raw_direction_scores / 40.0
    return result


def main():
    print("=== Experiment K: Why Did Direction Memory Help? ===")
    print("Only direction-score features are changed.")
    print("All other 17 observation values are held fixed.")
    print()

    model = PPO.load(MODEL_PATH)
    states = collect_states(model)

    print(f"Collected actual states: {len(states)}")
    print()

    # ---------------------------------------------------------
    # 1. 방향값을 바꿨을 때 policy distribution이 얼마나 변하는가
    # ---------------------------------------------------------
    changes = []
    argmax_changes = 0
    total_comparisons = 0

    pattern_action_counts = {
        name: np.zeros(4, dtype=np.int64)
        for name in PATTERNS
    }

    pattern_prob_sums = {
        name: np.zeros(4, dtype=np.float64)
        for name in PATTERNS
    }

    for state in states:
        original_obs = state["obs"]

        base_obs = counterfactual_obs(
            original_obs,
            PATTERNS["BALANCED"],
        )
        base_probs = get_action_probs(model, base_obs)
        base_argmax = int(np.argmax(base_probs))

        for name, scores in PATTERNS.items():
            test_obs = counterfactual_obs(
                original_obs,
                scores,
            )
            probs = get_action_probs(model, test_obs)
            argmax = int(np.argmax(probs))

            pattern_action_counts[name][argmax] += 1
            pattern_prob_sums[name] += probs

            if name != "BALANCED":
                total_comparisons += 1

                # 확률분포 변화량: L1 distance
                delta = float(np.abs(probs - base_probs).sum())
                changes.append(delta)

                if argmax != base_argmax:
                    argmax_changes += 1

    print("=" * 100)
    print("1) DOES DIRECTION MEMORY ACTUALLY CHANGE THE POLICY?")
    print("=" * 100)

    print(
        f"Average probability-distribution L1 change "
        f"vs BALANCED: {np.mean(changes):.4f}"
    )
    print(
        f"Median probability-distribution L1 change  "
        f"vs BALANCED: {np.median(changes):.4f}"
    )
    print(
        f"Argmax action changed: "
        f"{argmax_changes}/{total_comparisons} "
        f"({argmax_changes / total_comparisons * 100:.2f}%)"
    )

    print("\nAverage action probabilities by direction pattern:")
    for name in PATTERNS:
        avg = pattern_prob_sums[name] / len(states)
        print(
            f"  {name:10s} "
            f"[U {avg[0]*100:6.2f}% | "
            f"D {avg[1]*100:6.2f}% | "
            f"L {avg[2]*100:6.2f}% | "
            f"R {avg[3]*100:6.2f}%]"
        )

    # ---------------------------------------------------------
    # 2. 높은 값/0 값에 단순한 선호 규칙이 생겼는가
    # ---------------------------------------------------------
    print("\n" + "=" * 100)
    print("2) DID PPO LEARN 'HIGH = PREFER' OR 'ZERO = AVOID'?")
    print("=" * 100)

    for direction, name in enumerate(
        ["UP_HIGH", "DOWN_HIGH", "LEFT_HIGH", "RIGHT_HIGH"]
    ):
        count = pattern_action_counts[name][direction]
        rate = count / len(states) * 100.0

        print(
            f"  {name:10s}: "
            f"high-score direction became deterministic argmax "
            f"{count}/{len(states)} ({rate:.2f}%)"
        )

    used_patterns = [
        ("UP_USED", 0),
        ("DOWN_USED", 1),
        ("LEFT_USED", 2),
        ("RIGHT_USED", 3),
    ]

    print()
    for name, zero_direction in used_patterns:
        count = pattern_action_counts[name][zero_direction]
        rate = count / len(states) * 100.0

        print(
            f"  {name:10s}: "
            f"ZERO-score direction was STILL deterministic argmax "
            f"{count}/{len(states)} ({rate:.2f}%)"
        )

    # ---------------------------------------------------------
    # 3. 실제 상태 몇 개에서 before/after를 직접 보여준다
    # ---------------------------------------------------------
    print("\n" + "=" * 100)
    print("3) REPRESENTATIVE COUNTERFACTUAL STATES")
    print("=" * 100)

    # direction feature 변경으로 argmax가 실제 바뀌는 사례를 최대 12개 출력
    shown = 0

    for state in states:
        original_obs = state["obs"]

        balanced_probs = get_action_probs(
            model,
            counterfactual_obs(
                original_obs,
                PATTERNS["BALANCED"],
            ),
        )
        balanced_action = int(np.argmax(balanced_probs))

        variants = []

        for name in [
            "UP_USED", "DOWN_USED",
            "LEFT_USED", "RIGHT_USED",
            "UP_HIGH", "DOWN_HIGH",
            "LEFT_HIGH", "RIGHT_HIGH",
        ]:
            probs = get_action_probs(
                model,
                counterfactual_obs(
                    original_obs,
                    PATTERNS[name],
                ),
            )
            action = int(np.argmax(probs))

            if action != balanced_action:
                variants.append((name, probs, action))

        if not variants:
            continue

        print(
            f"\nSeed {state['seed']} step {state['step']} "
            f"pos={state['position']} goal={state['goal']}"
        )
        print(
            "  BALANCED "
            f"-> {ACTION_NAMES[balanced_action]:5s} "
            f"[U {balanced_probs[0]*100:5.1f} "
            f"D {balanced_probs[1]*100:5.1f} "
            f"L {balanced_probs[2]*100:5.1f} "
            f"R {balanced_probs[3]*100:5.1f}]"
        )

        for name, probs, action in variants[:4]:
            print(
                f"  {name:10s} "
                f"-> {ACTION_NAMES[action]:5s} "
                f"[U {probs[0]*100:5.1f} "
                f"D {probs[1]*100:5.1f} "
                f"L {probs[2]*100:5.1f} "
                f"R {probs[3]*100:5.1f}]"
            )

        shown += 1
        if shown >= 12:
            break

    if shown == 0:
        print(
            "No sampled state changed deterministic argmax "
            "when only direction scores were modified."
        )


if __name__ == "__main__":
    main()
