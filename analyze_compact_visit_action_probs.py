import numpy as np
import torch as th
from stable_baselines3 import PPO

from partial_obstacle_compact_memory_env import PartialObstacleCompactMemoryEnv

MODEL_PATH = "models/partial_obstacle_compact_memory_ppo"

# 대표적인 순환 실패. 필요하면 아래 리스트에 seed 추가.
SEEDS = [10040, 10020, 10059, 10237, 10442, 10447]

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


def get_action_probabilities(model, obs):
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with th.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        probs = distribution.distribution.probs.detach().cpu().numpy()[0]
    return probs


def analyze_seed(model, seed):
    env = PartialObstacleCompactMemoryEnv()
    obs, _ = env.reset(seed=seed)

    print("\n" + "=" * 100)
    print(f"SEED {seed}")
    print("Start:", tuple(env.agent_pos), "| Goal:", tuple(env.goal_pos))
    print("=" * 100)

    # 같은 위치에 다시 돌아왔을 때 정책이 방문횟수 변화에 반응하는지 보기 위해
    # 위치별 기록을 저장한다.
    records_by_position = {}

    for step in range(1, env.max_steps + 1):
        pos = tuple(int(v) for v in env.agent_pos)
        visit_count = int(env.visit_count_map[env.agent_pos[1], env.agent_pos[0]])

        probs = get_action_probabilities(model, obs)
        deterministic_action = int(np.argmax(probs))

        record = {
            "step": step,
            "visit": visit_count,
            "probs": probs.copy(),
            "action": deterministic_action,
        }
        records_by_position.setdefault(pos, []).append(record)

        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))

        if terminated or truncated:
            break

    # 반복 방문이 가장 많은 위치부터 출력.
    repeated = [
        (pos, records)
        for pos, records in records_by_position.items()
        if len(records) >= 3
    ]
    repeated.sort(key=lambda x: len(x[1]), reverse=True)

    if not repeated:
        print("3회 이상 반복 방문한 위치가 없습니다.")
        env.close()
        return

    # 너무 길어지지 않도록 seed당 상위 3개 위치.
    for pos, records in repeated[:3]:
        print()
        print(
            f"Position {pos} | policy decisions here: {len(records)} times"
        )
        print(
            f"{'Step':>5} {'Visit':>6} "
            f"{'UP':>8} {'DOWN':>8} {'LEFT':>8} {'RIGHT':>8} {'Pick':>7}"
        )
        print("-" * 70)

        # 처음 5개 + 중간 샘플 + 마지막 5개를 보여준다.
        if len(records) <= 15:
            selected = records
        else:
            indexes = {
                0, 1, 2, 3, 4,
                len(records) // 4,
                len(records) // 2,
                (len(records) * 3) // 4,
                len(records) - 5,
                len(records) - 4,
                len(records) - 3,
                len(records) - 2,
                len(records) - 1,
            }
            selected = [records[i] for i in sorted(indexes)]

        for r in selected:
            p = r["probs"]
            print(
                f'{r["step"]:5d} {r["visit"]:6d} '
                f'{p[0]*100:7.2f}% {p[1]*100:7.2f}% '
                f'{p[2]*100:7.2f}% {p[3]*100:7.2f}% '
                f'{ACTION_NAMES[r["action"]]:>7}'
            )

        first = records[0]
        last = records[-1]
        delta = (last["probs"] - first["probs"]) * 100.0

        print("\nFirst -> Last probability change (percentage points)")
        for i, name in enumerate(ACTION_NAMES):
            print(f"  {name:5}: {delta[i]:+7.2f} pp")

        print(
            f'Visit count: {first["visit"]} -> {last["visit"]}'
        )
        print(
            f'Deterministic pick: '
            f'{ACTION_NAMES[first["action"]]} -> '
            f'{ACTION_NAMES[last["action"]]}'
        )

    env.close()


def main():
    print("=== Experiment I: Visit Count vs Action Probability ===")
    print("Model:", MODEL_PATH)
    print(
        "Purpose: check whether the learned policy actually reacts "
        "to increasing current-tile visit count."
    )

    model = PPO.load(MODEL_PATH)

    for seed in SEEDS:
        analyze_seed(model, seed)


if __name__ == "__main__":
    main()
