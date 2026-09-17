from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env


class PartialObstacleMapRewardEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.grid_size = 9
        self.max_steps = 100
        self.obstacle_probability = 0.18

        # Reward
        self.step_penalty = -1.0
        self.discovery_reward = 0.2
        self.goal_reward = 100.0

        # Action
        # 0: 위
        # 1: 아래
        # 2: 왼쪽
        # 3: 오른쪽
        self.action_space = spaces.Discrete(4)

        # Observation
        #
        # 현재 주변 3x3 = 9
        # Goal 상대 위치 dx, dy = 2
        # 누적 발견 지도 9x9 = 81
        # 현재 Agent 절대 위치 x, y = 2
        #
        # 총 94
        #
        # 누적 지도:
        # -1 = 아직 관측하지 못함
        #  0 = 관측된 이동 가능 칸
        #  1 = 관측된 장애물
        #
        # Agent x/y:
        # 0 ~ 8 좌표를 0 ~ 1로 정규화
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(94,),
            dtype=np.float32,
        )

        self.grid = None
        self.agent_pos = None
        self.goal_pos = None
        self.steps = 0

        # PPO에게도 전달되는 누적 발견 지도
        self.discovered_map = None

    def _random_position(self):
        return np.array(
            [
                self.np_random.integers(0, self.grid_size),
                self.np_random.integers(0, self.grid_size),
            ],
            dtype=np.int32,
        )

    def _is_path_available(self):
        """
        맵 생성 시 실제 Agent -> Goal 경로가 존재하는지 BFS로 검사한다.

        해결 가능한 맵만 생성하기 위한 용도이며,
        실제 전체 경로 정보는 PPO에게 제공하지 않는다.
        """

        start = tuple(self.agent_pos)
        goal = tuple(self.goal_pos)

        queue = deque([start])
        visited = {start}

        directions = [
            (0, -1),
            (0, 1),
            (-1, 0),
            (1, 0),
        ]

        while queue:
            x, y = queue.popleft()

            if (x, y) == goal:
                return True

            for dx, dy in directions:
                nx = x + dx
                ny = y + dy

                if not (
                    0 <= nx < self.grid_size
                    and 0 <= ny < self.grid_size
                ):
                    continue

                if self.grid[ny, nx] == 1:
                    continue

                pos = (nx, ny)

                if pos in visited:
                    continue

                visited.add(pos)
                queue.append(pos)

        return False

    def _update_discovered_map(self):
        """
        현재 Agent 위치 기준 주변 3x3을 관측하여
        누적 발견 지도에 기록한다.

        반환값:
            이번 관측에서 새롭게 밝혀진 맵 내부 칸 수

        장애물도 처음 관측했다면 새로운 정보이므로
        새로 발견한 칸으로 계산한다.
        """

        ax, ay = self.agent_pos

        newly_discovered = 0

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                x = ax + dx
                y = ay + dy

                # 맵 바깥은 발견 칸으로 계산하지 않음
                if not (
                    0 <= x < self.grid_size
                    and 0 <= y < self.grid_size
                ):
                    continue

                # 아직 한 번도 관측하지 않은 칸
                if self.discovered_map[y, x] == -1.0:
                    newly_discovered += 1

                    if self.grid[y, x] == 1:
                        self.discovered_map[y, x] = 1.0
                    else:
                        self.discovered_map[y, x] = 0.0

        return newly_discovered

    def _get_observation(self):
        ax, ay = self.agent_pos

        local_cells = []

        # -------------------------------------------------
        # 1. 현재 주변 3x3 = 9
        # -------------------------------------------------
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                x = ax + dx
                y = ay + dy

                if (
                    x < 0
                    or x >= self.grid_size
                    or y < 0
                    or y >= self.grid_size
                ):
                    local_cells.append(-1.0)

                elif self.grid[y, x] == 1:
                    local_cells.append(1.0)

                else:
                    local_cells.append(0.0)

        # -------------------------------------------------
        # 2. Goal 상대 위치 = 2
        # -------------------------------------------------
        goal_dx = (
            self.goal_pos[0] - self.agent_pos[0]
        ) / (self.grid_size - 1)

        goal_dy = (
            self.goal_pos[1] - self.agent_pos[1]
        ) / (self.grid_size - 1)

        # -------------------------------------------------
        # 3. 누적 발견 지도 = 81
        # -------------------------------------------------
        discovered_flat = (
            self.discovered_map
            .flatten()
            .tolist()
        )

        # -------------------------------------------------
        # 4. 현재 Agent의 월드맵상 절대 위치 = 2
        # -------------------------------------------------
        agent_x = (
            self.agent_pos[0]
            / (self.grid_size - 1)
        )

        agent_y = (
            self.agent_pos[1]
            / (self.grid_size - 1)
        )

        # 총 94개
        observation = np.array(
            local_cells
            + [goal_dx, goal_dy]
            + discovered_flat
            + [agent_x, agent_y],
            dtype=np.float32,
        )

        return observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0

        # -------------------------------------------------
        # 반드시 실제 경로가 존재하는 랜덤 맵 생성
        # -------------------------------------------------
        while True:
            self.grid = (
                self.np_random.random(
                    (self.grid_size, self.grid_size)
                )
                < self.obstacle_probability
            ).astype(np.int32)

            self.agent_pos = self._random_position()
            self.goal_pos = self._random_position()

            if np.array_equal(
                self.agent_pos,
                self.goal_pos,
            ):
                continue

            ax, ay = self.agent_pos
            gx, gy = self.goal_pos

            # 시작점과 Goal은 항상 이동 가능
            self.grid[ay, ax] = 0
            self.grid[gy, gx] = 0

            # 해결 가능한 맵만 사용
            if self._is_path_available():
                break

        # -------------------------------------------------
        # 누적 지도 초기화
        # -------------------------------------------------
        # 처음에는 모든 칸을 모름(-1)
        self.discovered_map = np.full(
            (self.grid_size, self.grid_size),
            -1.0,
            dtype=np.float32,
        )

        # 시작 위치에서 보이는 주변 3x3은
        # 처음부터 관측한 상태로 시작한다.
        #
        # 시작 Observation이므로 이 발견에는
        # Reward를 지급하지 않는다.
        self._update_discovered_map()

        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1

        new_pos = self.agent_pos.copy()

        # -------------------------------------------------
        # PPO가 선택한 Action 실행
        # -------------------------------------------------
        if action == 0:
            new_pos[1] -= 1

        elif action == 1:
            new_pos[1] += 1

        elif action == 2:
            new_pos[0] -= 1

        elif action == 3:
            new_pos[0] += 1

        # -------------------------------------------------
        # 맵 밖이면 이동 실패 → 제자리
        # -------------------------------------------------
        if (
            new_pos[0] < 0
            or new_pos[0] >= self.grid_size
            or new_pos[1] < 0
            or new_pos[1] >= self.grid_size
        ):
            new_pos = self.agent_pos.copy()

        # -------------------------------------------------
        # 장애물이면 이동 실패 → 제자리
        # -------------------------------------------------
        nx, ny = new_pos

        if self.grid[ny, nx] == 1:
            new_pos = self.agent_pos.copy()

        self.agent_pos = new_pos

        # -------------------------------------------------
        # Reward
        # -------------------------------------------------

        # 행동 한 번마다 -1
        reward = self.step_penalty

        # 이동 후 현재 주변 3x3을 관측하고
        # 누적 지도 업데이트
        newly_discovered = self._update_discovered_map()

        # 새롭게 밝혀낸 칸 1개당 +0.2
        reward += (
            newly_discovered
            * self.discovery_reward
        )

        terminated = False

        # Goal 도착 시 +100
        if np.array_equal(
            self.agent_pos,
            self.goal_pos,
        ):
            reward += self.goal_reward
            terminated = True

        # 100 Step 사용 시 실패 종료
        truncated = self.steps >= self.max_steps

        info = {
            "newly_discovered": newly_discovered,
            "discovered_cells": int(
                np.sum(
                    self.discovered_map != -1.0
                )
            ),
            "agent_x": int(self.agent_pos[0]),
            "agent_y": int(self.agent_pos[1]),
        }

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        """
        디버깅용 실제 전체 맵.
        PPO는 이 전체 맵을 볼 수 없다.
        """

        print()

        for y in range(self.grid_size):
            row = []

            for x in range(self.grid_size):
                pos = np.array([x, y])

                if np.array_equal(
                    pos,
                    self.agent_pos,
                ):
                    row.append("A")

                elif np.array_equal(
                    pos,
                    self.goal_pos,
                ):
                    row.append("G")

                elif self.grid[y, x] == 1:
                    row.append("#")

                else:
                    row.append(".")

            print(" ".join(row))

        print()

    def render_discovered_map(self):
        """
        PPO에게 전달되는 누적 지도를 사람이 보기 쉽게 출력한다.

        ? = 아직 모름
        . = 확인된 이동 가능 칸
        # = 확인된 장애물
        A = 현재 Agent 위치

        A는 출력 편의를 위한 표시다.
        PPO에게는 별도로 agent_x / agent_y가 전달된다.
        """

        print()

        for y in range(self.grid_size):
            row = []

            for x in range(self.grid_size):
                if (
                    x == self.agent_pos[0]
                    and y == self.agent_pos[1]
                ):
                    row.append("A")
                    continue

                value = self.discovered_map[y, x]

                if value == -1.0:
                    row.append("?")

                elif value == 1.0:
                    row.append("#")

                else:
                    row.append(".")

            print(" ".join(row))

        print()


if __name__ == "__main__":
    env = PartialObstacleMapRewardEnv()

    check_env(env)

    print("환경 검사 성공!")
    print(
        "Observation shape:",
        env.observation_space.shape,
    )

    obs, _ = env.reset(seed=0)

    print(
        "시작 Observation 길이:",
        len(obs),
    )

    print(
        "시작 Agent 위치:",
        tuple(env.agent_pos),
    )

    print(
        "Observation의 Agent 정규화 위치:",
        f"x={obs[-2]:.3f},",
        f"y={obs[-1]:.3f}",
    )

    print("\n=== 실제 전체 맵 ===")
    env.render()

    print(
        "=== PPO가 전달받는 누적 지도 + 현재 위치 ==="
    )
    env.render_discovered_map()

    for step in range(5):
        action = env.action_space.sample()

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        print(
            f"Step {step + 1} | "
            f"Action {action} | "
            f"Position "
            f"({info['agent_x']}, {info['agent_y']}) | "
            f"New {info['newly_discovered']} | "
            f"Known {info['discovered_cells']}/81 | "
            f"Reward {reward:.2f}"
        )

        env.render_discovered_map()

        if terminated or truncated:
            break