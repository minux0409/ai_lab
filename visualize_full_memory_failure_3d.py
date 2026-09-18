import os

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation, PillowWriter
from stable_baselines3 import PPO

from partial_obstacle_full_memory_env import (
    PartialObstacleFullMemoryEnv,
)


# ============================================================
# 설정
# ============================================================

MODEL_PATH = (
    "models/partial_obstacle_full_memory_ppo"
)

# 여기 숫자만 바꾸면 원하는 Episode를 볼 수 있음.
#
# 추천 실패 Seed
# 10003 = STUCK / 같은 위치 95회
# 10004 = A-B 왕복 87회
# 10009 = Goal 거리 1까지 접근 후 실패
# 10067 = Goal 거리 1까지 접근 후 최종 거리 7
TARGET_SEED = 10003

OUTPUT_DIR = "visualizations"

FPS = 4

ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
}


# ============================================================
# Episode 실행
# ============================================================

def run_episode(
    model,
    seed,
):
    env = PartialObstacleFullMemoryEnv()

    observation, _ = env.reset(
        seed=seed
    )

    frames = []
    total_reward = 0.0

    start_position = tuple(
        int(v)
        for v in env.agent_pos
    )

    goal_position = tuple(
        int(v)
        for v in env.goal_pos
    )

    frames.append(
        {
            "step": 0,
            "grid": env.grid.copy(),
            "discovered_map":
                env.discovered_map.copy(),
            "visit_count_map":
                env.visit_count_map.copy(),
            "agent_pos":
                env.agent_pos.copy(),
            "goal_pos":
                env.goal_pos.copy(),
            "action": None,
            "reward": 0.0,
            "total_reward": 0.0,
            "moved": True,
            "goal_distance":
                abs(
                    int(env.agent_pos[0])
                    - int(env.goal_pos[0])
                )
                + abs(
                    int(env.agent_pos[1])
                    - int(env.goal_pos[1])
                ),
            "goal_score":
                100
                - (
                    abs(
                        int(env.agent_pos[0])
                        - int(env.goal_pos[0])
                    )
                    + abs(
                        int(env.agent_pos[1])
                        - int(env.goal_pos[1])
                    )
                ),
            "newly_discovered": 0,
            "discovered_cells":
                int(
                    np.sum(
                        env.discovered_map
                        != -1
                    )
                ),
            "unique_visited_cells":
                int(
                    np.sum(
                        env.visit_count_map
                        > 0
                    )
                ),
        }
    )

    while True:
        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        action = int(action)

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += float(reward)

        frames.append(
            {
                "step": env.steps,
                "grid": env.grid.copy(),

                "discovered_map":
                    env.discovered_map.copy(),

                "visit_count_map":
                    env.visit_count_map.copy(),

                "agent_pos":
                    env.agent_pos.copy(),

                "goal_pos":
                    env.goal_pos.copy(),

                "action": action,

                "reward":
                    float(reward),

                "total_reward":
                    total_reward,

                "moved":
                    bool(
                        info["moved"]
                    ),

                "goal_distance":
                    int(
                        info["goal_distance"]
                    ),

                "goal_score":
                    float(
                        info["goal_score"]
                    ),

                "newly_discovered":
                    int(
                        info[
                            "newly_discovered"
                        ]
                    ),

                "discovered_cells":
                    int(
                        info[
                            "discovered_cells"
                        ]
                    ),

                "unique_visited_cells":
                    int(
                        info[
                            "unique_visited_cells"
                        ]
                    ),
            }
        )

        if terminated or truncated:
            success = bool(terminated)
            break

    env.close()

    return {
        "seed": seed,
        "success": success,
        "frames": frames,
        "start": start_position,
        "goal": goal_position,
    }


# ============================================================
# 실패 통계
# ============================================================

def analyze_episode(
    result,
):
    frames = result["frames"]

    positions = [
        tuple(
            int(v)
            for v in frame[
                "agent_pos"
            ]
        )
        for frame in frames
    ]

    moved_flags = [
        frame["moved"]
        for frame in frames[1:]
    ]

    unchanged = sum(
        not moved
        for moved in moved_flags
    )

    unique_positions = len(
        set(positions)
    )

    # 같은 위치 연속 횟수
    max_same_run = 1
    current_same_run = 1

    for i in range(
        1,
        len(positions)
    ):
        if (
            positions[i]
            == positions[i - 1]
        ):
            current_same_run += 1

            max_same_run = max(
                max_same_run,
                current_same_run,
            )

        else:
            current_same_run = 1

    # A-B-A-B 패턴 횟수
    ab_count = 0

    for i in range(
        2,
        len(positions)
    ):
        if (
            positions[i]
            == positions[i - 2]
            and
            positions[i]
            != positions[i - 1]
        ):
            ab_count += 1

    min_goal_distance = min(
        frame["goal_distance"]
        for frame in frames
    )

    return {
        "unique_positions":
            unique_positions,

        "unchanged":
            unchanged,

        "max_same_run":
            max_same_run,

        "ab_count":
            ab_count,

        "min_goal_distance":
            min_goal_distance,
    }


# ============================================================
# 3D Animation
# ============================================================

def create_animation(
    result,
):
    frames = result["frames"]
    seed = result["seed"]

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    gif_path = os.path.join(
        OUTPUT_DIR,
        f"failure_seed_{seed}.gif",
    )

    fig = plt.figure(
        figsize=(13, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    def update(
        frame_index,
    ):
        ax.clear()

        frame = frames[
            frame_index
        ]

        grid = frame["grid"]

        discovered = frame[
            "discovered_map"
        ]

        visits = frame[
            "visit_count_map"
        ]

        agent_x = int(
            frame["agent_pos"][0]
        )

        agent_y = int(
            frame["agent_pos"][1]
        )

        goal_x = int(
            frame["goal_pos"][0]
        )

        goal_y = int(
            frame["goal_pos"][1]
        )

        # ----------------------------------------------------
        # Tiles
        # ----------------------------------------------------

        for y in range(9):
            for x in range(9):

                known = (
                    discovered[y, x]
                    != -1
                )

                # 아직 발견하지 못한 칸
                if not known:
                    ax.bar3d(
                        x,
                        y,
                        0,
                        0.9,
                        0.9,
                        0.02,
                        alpha=0.08,
                        shade=True,
                    )

                    continue

                # 장애물
                if grid[y, x] == 1:
                    ax.bar3d(
                        x,
                        y,
                        0,
                        0.9,
                        0.9,
                        0.8,
                        alpha=0.75,
                        shade=True,
                    )

                    continue

                visit_count = int(
                    visits[y, x]
                )

                height = (
                    0.025
                    + min(
                        visit_count,
                        10,
                    )
                    * 0.012
                )

                ax.bar3d(
                    x,
                    y,
                    0,
                    0.9,
                    0.9,
                    height,
                    alpha=0.30,
                    shade=True,
                )

                # 방문 횟수
                if visit_count > 0:
                    ax.text(
                        x + 0.45,
                        y + 0.45,
                        height + 0.025,
                        str(
                            visit_count
                        ),
                        ha="center",
                        va="center",
                        fontsize=7,
                    )

        # ----------------------------------------------------
        # Goal
        # ----------------------------------------------------

        ax.scatter(
            [
                goal_x + 0.45
            ],
            [
                goal_y + 0.45
            ],
            [0.42],
            marker="*",
            s=400,
            label="Goal",
        )

        ax.text(
            goal_x + 0.45,
            goal_y + 0.45,
            0.68,
            "GOAL",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

        # ----------------------------------------------------
        # 이동 경로
        # ----------------------------------------------------

        path_x = []
        path_y = []
        path_z = []

        for old_frame in (
            frames[
                :frame_index + 1
            ]
        ):
            path_x.append(
                int(
                    old_frame[
                        "agent_pos"
                    ][0]
                )
                + 0.45
            )

            path_y.append(
                int(
                    old_frame[
                        "agent_pos"
                    ][1]
                )
                + 0.45
            )

            path_z.append(
                0.25
            )

        ax.plot(
            path_x,
            path_y,
            path_z,
            linewidth=2,
            alpha=0.8,
            label="Path",
        )

        # ----------------------------------------------------
        # Agent
        # ----------------------------------------------------

        ax.scatter(
            [
                agent_x + 0.45
            ],
            [
                agent_y + 0.45
            ],
            [0.42],
            s=280,
            label="Agent",
        )

        ax.text(
            agent_x + 0.45,
            agent_y + 0.45,
            0.68,
            "A",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

        # ----------------------------------------------------
        # Action
        # ----------------------------------------------------

        action = frame[
            "action"
        ]

        if action is None:
            action_name = "START"
        else:
            action_name = (
                ACTION_NAMES[
                    action
                ]
            )

        if frame["moved"]:
            moved_text = "YES"
        else:
            moved_text = "NO"

        # ----------------------------------------------------
        # Title / Debug Information
        # ----------------------------------------------------

        title = (
            f"Seed {seed}"
            f" | "
            f"{'SUCCESS' if result['success'] else 'FAILURE'}"
            f" | Step "
            f"{frame['step']}/100\n"

            f"Action: {action_name}"
            f" | Moved: {moved_text}"
            f" | A: ({agent_x},{agent_y})"
            f" | Goal: ({goal_x},{goal_y})\n"

            f"Goal Distance: "
            f"{frame['goal_distance']}"
            f" | Goal Score: "
            f"{frame['goal_score']:.0f}"
            f" | Step Reward: "
            f"{frame['reward']:.1f}"
            f" | Total: "
            f"{frame['total_reward']:.1f}\n"

            f"Known: "
            f"{frame['discovered_cells']}/81"
            f" | Unique Visited: "
            f"{frame['unique_visited_cells']}/81"
        )

        ax.set_title(
            title,
            fontsize=10,
        )

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        ax.set_xlim(
            0,
            9,
        )

        ax.set_ylim(
            9,
            0,
        )

        ax.set_zlim(
            0,
            1.4,
        )

        ax.set_xticks(
            range(9)
        )

        ax.set_yticks(
            range(9)
        )

        ax.set_zticks(
            []
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        ax.view_init(
            elev=55,
            azim=-55,
        )

        ax.legend(
            loc="upper left",
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=(
            1000 / FPS
        ),
        repeat=True,
    )

    print()
    print(
        f"GIF 생성 중: {gif_path}"
    )

    animation.save(
        gif_path,
        writer=PillowWriter(
            fps=FPS
        ),
    )

    print(
        "GIF 저장 완료!"
    )

    print(
        gif_path
    )

    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    print()
    print(
        "=== PPO Episode 3D Visualizer ==="
    )

    print(
        f"Target Seed : {TARGET_SEED}"
    )

    print(
        f"Model       : {MODEL_PATH}"
    )

    model = PPO.load(
        MODEL_PATH
    )

    result = run_episode(
        model,
        TARGET_SEED,
    )

    analysis = analyze_episode(
        result
    )

    final_frame = (
        result["frames"][-1]
    )

    print()
    print(
        "======================================"
    )

    print(
        "EPISODE SUMMARY"
    )

    print(
        "======================================"
    )

    print(
        f"Seed              : "
        f"{TARGET_SEED}"
    )

    print(
        f"Result            : "
        f"{'SUCCESS' if result['success'] else 'FAILURE'}"
    )

    print(
        f"Start             : "
        f"{result['start']}"
    )

    print(
        f"Goal              : "
        f"{result['goal']}"
    )

    print(
        f"Final             : "
        f"{tuple(int(v) for v in final_frame['agent_pos'])}"
    )

    print(
        f"Steps             : "
        f"{final_frame['step']}"
    )

    print(
        f"Unique Positions  : "
        f"{analysis['unique_positions']}"
    )

    print(
        f"Unchanged Moves   : "
        f"{analysis['unchanged']}"
    )

    print(
        f"Max Same Pos Run  : "
        f"{analysis['max_same_run']}"
    )

    print(
        f"A-B Oscillations  : "
        f"{analysis['ab_count']}"
    )

    print(
        f"Min Goal Distance : "
        f"{analysis['min_goal_distance']}"
    )

    print(
        f"Known Cells       : "
        f"{final_frame['discovered_cells']}/81"
    )

    print(
        f"Total Reward      : "
        f"{final_frame['total_reward']:.2f}"
    )

    create_animation(
        result
    )


if __name__ == "__main__":
    main()