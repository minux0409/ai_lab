import numpy as np
import torch
from stable_baselines3 import PPO

from partial_obstacle_map_reward_env import (
    PartialObstacleMapRewardEnv,
)


MODEL_PATH = "models/partial_obstacle_map_position_reward_ppo"

EPISODES = 500
MAX_EXAMPLES = 10

ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
}


def get_policy_info(model, obs):
    """
    현재 Observation에서 PPO가 각 Action을
    몇 % 확률로 선택하려 하는지와 Critic Value를 반환.
    """

    obs_tensor, _ = model.policy.obs_to_tensor(obs)

    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)

        probabilities = (
            distribution.distribution.probs
            .cpu()
            .numpy()[0]
        )

        value = (
            model.policy.predict_values(obs_tensor)
            .cpu()
            .numpy()
            .flatten()[0]
        )

    return probabilities, float(value)


def get_blocked_actions(env):
    """
    현재 위치에서 실제로 이동할 수 없는 Action 확인.

    이 함수는 분석용일 뿐 PPO에게 정보를 추가하지 않는다.
    """

    x, y = env.agent_pos

    candidates = {
        0: (x, y - 1),  # UP
        1: (x, y + 1),  # DOWN
        2: (x - 1, y),  # LEFT
        3: (x + 1, y),  # RIGHT
    }

    blocked = {}

    for action, (nx, ny) in candidates.items():

        # 맵 바깥
        if not (
            0 <= nx < env.grid_size
            and 0 <= ny < env.grid_size
        ):
            blocked[action] = "MAP OUT"
            continue

        # 장애물
        if env.grid[ny, nx] == 1:
            blocked[action] = "WALL"
            continue

        blocked[action] = "OPEN"

    return blocked


def print_state(
    episode,
    step,
    env,
    probabilities,
    value,
    selected_action,
):
    blocked = get_blocked_actions(env)

    x, y = env.agent_pos
    gx, gy = env.goal_pos

    print()
    print("=" * 55)

    print(
        f"Episode {episode} | Step {step}"
    )

    print(
        f"Agent: ({x}, {y}) | "
        f"Goal: ({gx}, {gy})"
    )

    print(
        f"Goal relative: "
        f"dx={gx - x}, dy={gy - y}"
    )

    print()
    print("Action probabilities:")

    for action in range(4):

        marker = ""

        if action == selected_action:
            marker = " <-- SELECTED"

        print(
            f"  {ACTION_NAMES[action]:5} "
            f"{probabilities[action] * 100:6.2f}% "
            f"[{blocked[action]}]"
            f"{marker}"
        )

    print()

    print(
        f"Critic Value: {value:.3f}"
    )

    print("=" * 55)


def main():

    print("모델 로딩...")

    model = PPO.load(MODEL_PATH)

    env = PartialObstacleMapRewardEnv()

    shown = 0

    # -------------------------------------------------
    # 실패 Episode 탐색
    # -------------------------------------------------

    for episode in range(EPISODES):

        seed = 10_000 + episode

        obs, _ = env.reset(seed=seed)

        # 같은 위치에서 같은 행동을 반복하는지 추적
        previous_position = None
        previous_action = None
        repeat_count = 0

        for step in range(1, 101):

            # ------------------------------------------
            # 현재 Policy 확률 확인
            # ------------------------------------------
            probabilities, value = (
                get_policy_info(
                    model,
                    obs,
                )
            )

            # deterministic=True와 동일:
            # 가장 확률 높은 Action
            selected_action = int(
                np.argmax(probabilities)
            )

            current_position = tuple(
                int(v)
                for v in env.agent_pos
            )

            # ------------------------------------------
            # 같은 위치 + 같은 행동 반복 검사
            # ------------------------------------------
            if (
                current_position
                == previous_position
                and selected_action
                == previous_action
            ):
                repeat_count += 1

            else:
                repeat_count = 1

            previous_position = current_position
            previous_action = selected_action

            # ------------------------------------------
            # 실제 Action 실행
            # ------------------------------------------
            (
                next_obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(selected_action)

            next_position = tuple(
                int(v)
                for v in env.agent_pos
            )

            # ------------------------------------------
            # 이동 불가능한 행동을 선택했고
            # 같은 행동을 5번 이상 반복하면 출력
            # ------------------------------------------
            if (
                next_position == current_position
                and repeat_count == 5
            ):
                print()
                print(
                    "!!! BLOCKED ACTION 반복 발견 !!!"
                )

                print_state(
                    episode=episode,
                    step=step,
                    env=env,
                    probabilities=probabilities,
                    value=value,
                    selected_action=selected_action,
                )

                shown += 1

                if shown >= MAX_EXAMPLES:
                    env.close()
                    return

            obs = next_obs

            if terminated or truncated:
                break

    env.close()

    print()
    print(
        f"분석 완료. "
        f"반복 사례 {shown}개 발견."
    )


if __name__ == "__main__":
    main()