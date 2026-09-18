import numpy as np
import torch

from stable_baselines3 import PPO

from partial_obstacle_full_memory_env import (
    PartialObstacleFullMemoryEnv,
)


MODEL_PATH = "models/partial_obstacle_full_memory_ppo"

# 벽에 95번 박았던 대표 STUCK 케이스
TARGET_SEED = 10003

ACTION_NAMES = [
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
]


def get_action_probabilities(
    model,
    observation,
):
    """
    현재 Observation을 PPO Policy에 넣고
    UP / DOWN / LEFT / RIGHT 확률을 직접 가져온다.
    """

    obs_tensor, _ = (
        model.policy.obs_to_tensor(
            observation
        )
    )

    with torch.no_grad():

        distribution = (
            model.policy.get_distribution(
                obs_tensor
            )
        )

        probabilities = (
            distribution.distribution.probs
            .cpu()
            .numpy()[0]
        )

    return probabilities


def format_probs(
    probabilities,
):
    parts = []

    for index, name in enumerate(
        ACTION_NAMES
    ):
        parts.append(
            f"{name} "
            f"{probabilities[index] * 100:6.2f}%"
        )

    return " | ".join(parts)


def main():

    print()
    print(
        "=== PPO Action Probability Analysis ==="
    )

    print(
        f"Model : {MODEL_PATH}"
    )

    print(
        f"Seed  : {TARGET_SEED}"
    )

    # --------------------------------------------------------
    # Model / Environment
    # --------------------------------------------------------

    model = PPO.load(
        MODEL_PATH
    )

    env = (
        PartialObstacleFullMemoryEnv()
    )

    observation, _ = env.reset(
        seed=TARGET_SEED
    )

    print()
    print(
        f"Start : "
        f"{tuple(int(v) for v in env.agent_pos)}"
    )

    print(
        f"Goal  : "
        f"{tuple(int(v) for v in env.goal_pos)}"
    )

    print()

    # --------------------------------------------------------
    # 분석용 기록
    # --------------------------------------------------------

    previous_position = tuple(
        int(v)
        for v in env.agent_pos
    )

    stuck_start_step = None

    unchanged_count = 0

    records = []

    # --------------------------------------------------------
    # Episode
    # --------------------------------------------------------

    while True:

        # 현재 Observation에서 PPO가 생각하는
        # 4개 행동 확률
        probabilities = (
            get_action_probabilities(
                model,
                observation,
            )
        )

        # deterministic=True와 동일하게
        # 가장 확률 높은 행동 선택
        action = int(
            np.argmax(
                probabilities
            )
        )

        position_before = tuple(
            int(v)
            for v in env.agent_pos
        )

        visit_count_before = int(
            env.visit_count_map[
                env.agent_pos[1],
                env.agent_pos[0],
            ]
        )

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        position_after = tuple(
            int(v)
            for v in env.agent_pos
        )

        moved = (
            position_before
            != position_after
        )

        # ----------------------------------------------------
        # 벽/경계에 막혔는지
        # ----------------------------------------------------

        if not moved:

            unchanged_count += 1

            if stuck_start_step is None:
                stuck_start_step = (
                    env.steps
                )

        else:
            unchanged_count = 0
            stuck_start_step = None

        # ----------------------------------------------------
        # 현재 위치 방문횟수
        # ----------------------------------------------------

        visit_count_after = int(
            env.visit_count_map[
                env.agent_pos[1],
                env.agent_pos[0],
            ]
        )

        # ----------------------------------------------------
        # 기록
        # ----------------------------------------------------

        record = {
            "step":
                env.steps,

            "before":
                position_before,

            "after":
                position_after,

            "action":
                action,

            "action_name":
                ACTION_NAMES[action],

            "probabilities":
                probabilities.copy(),

            "moved":
                moved,

            "reward":
                float(reward),

            "goal_distance":
                int(
                    info[
                        "goal_distance"
                    ]
                ),

            "visit_before":
                visit_count_before,

            "visit_after":
                visit_count_after,

            "unchanged_count":
                unchanged_count,
        }

        records.append(
            record
        )

        observation = (
            next_observation
        )

        previous_position = (
            position_after
        )

        if terminated or truncated:
            success = bool(
                terminated
            )

            break

    env.close()

    # ========================================================
    # 전체 경로 요약
    # ========================================================

    print(
        "=============================================="
    )

    print(
        "EPISODE RESULT"
    )

    print(
        "=============================================="
    )

    print(
        f"Result : "
        f"{'SUCCESS' if success else 'FAILURE'}"
    )

    print(
        f"Steps  : "
        f"{len(records)}"
    )

    final_position = (
        records[-1]["after"]
    )

    print(
        f"Final  : "
        f"{final_position}"
    )

    print()

    # ========================================================
    # 모든 이동 출력
    #
    # 정상 이동은 간단하게,
    # 이동 실패는 확률까지 자세히 출력
    # ========================================================

    print(
        "=============================================="
    )

    print(
        "STEP TRACE"
    )

    print(
        "=============================================="
    )

    for record in records:

        if record["moved"]:

            print(
                f"Step "
                f"{record['step']:3d} | "
                f"{record['before']} "
                f"-> "
                f"{record['after']} | "
                f"{record['action_name']:<5} | "
                f"Moved"
            )

        else:

            print(
                f"Step "
                f"{record['step']:3d} | "
                f"{record['before']} "
                f"-> "
                f"{record['after']} | "
                f"{record['action_name']:<5} | "
                f"BLOCKED #{record['unchanged_count']:02d} | "
                f"Visit "
                f"{record['visit_before']}"
                f"->{record['visit_after']} | "
                f"{format_probs(record['probabilities'])}"
            )

    # ========================================================
    # 가장 긴 연속 BLOCKED 구간 찾기
    # ========================================================

    longest_sequence = []
    current_sequence = []

    for record in records:

        if not record["moved"]:

            current_sequence.append(
                record
            )

            if (
                len(current_sequence)
                > len(longest_sequence)
            ):
                longest_sequence = (
                    current_sequence.copy()
                )

        else:
            current_sequence = []

    print()
    print(
        "=============================================="
    )

    print(
        "LONGEST BLOCKED SEQUENCE"
    )

    print(
        "=============================================="
    )

    if not longest_sequence:

        print(
            "연속 BLOCKED 구간 없음"
        )

        return

    first = (
        longest_sequence[0]
    )

    last = (
        longest_sequence[-1]
    )

    print(
        f"Position : "
        f"{first['before']}"
    )

    print(
        f"Steps    : "
        f"{first['step']} "
        f"~ "
        f"{last['step']}"
    )

    print(
        f"Length   : "
        f"{len(longest_sequence)}"
    )

    print(
        f"Action   : "
        f"{first['action_name']}"
    )

    print()

    # --------------------------------------------------------
    # 긴 구간 전체 95줄 대신
    # 대표 지점 출력
    # --------------------------------------------------------

    indexes = [
        0,
        1,
        2,
        4,
        9,
        19,
        39,
        59,
        79,
        len(longest_sequence) - 1,
    ]

    # 중복 제거 + 실제 범위 내 값만
    indexes = sorted(
        set(
            index
            for index in indexes
            if (
                0
                <= index
                < len(longest_sequence)
            )
        )
    )

    print(
        "대표 Step별 정책 확률"
    )

    print()

    for index in indexes:

        record = (
            longest_sequence[
                index
            ]
        )

        print(
            f"Step "
            f"{record['step']:3d} | "
            f"Blocked "
            f"{index + 1:3d}회째 | "
            f"Visit "
            f"{record['visit_before']:3d}"
            f"->{record['visit_after']:3d}"
        )

        print(
            "    "
            + format_probs(
                record[
                    "probabilities"
                ]
            )
        )

    # ========================================================
    # 처음 vs 마지막 비교
    # ========================================================

    first_probs = (
        longest_sequence[0][
            "probabilities"
        ]
    )

    last_probs = (
        longest_sequence[-1][
            "probabilities"
        ]
    )

    print()
    print(
        "=============================================="
    )

    print(
        "FIRST vs LAST BLOCKED STATE"
    )

    print(
        "=============================================="
    )

    for action_index, name in enumerate(
        ACTION_NAMES
    ):

        first_probability = (
            first_probs[
                action_index
            ]
            * 100
        )

        last_probability = (
            last_probs[
                action_index
            ]
            * 100
        )

        difference = (
            last_probability
            - first_probability
        )

        print(
            f"{name:<5} | "
            f"{first_probability:6.2f}% "
            f"-> "
            f"{last_probability:6.2f}% "
            f"| Change "
            f"{difference:+7.2f}%p"
        )

    print()

    print(
        "이 결과에서는 해결책을 적용하지 않습니다."
    )

    print(
        "방문 횟수/이전 행동이 변할 때 "
        "정책 확률이 실제로 어떻게 변하는지만 확인합니다."
    )


if __name__ == "__main__":
    main()