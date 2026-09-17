from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env


class PartialObstacleExploreRewardEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.grid_size = 9
        self.max_steps = 100
        self.obstacle_probability = 0.18

        # Reward 설정
        self.step_penalty = -1.0
        self.discovery_reward = 0.2
        self.goal_reward = 100.0

        # 0: 위
        # 1: 아래
        # 2: 왼쪽
        # 3: 오른쪽
        self.action_space = spaces.Discrete(4)

        # Observation은 Baseline과 완전히 동일
        #
        # 주변 3x3 = 9개
        # Goal 상대 위치 dx, dy = 2개
        #
        # 총 11개
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(11,),
            dtype=np.float32,
        )

        self.grid = None
        self.agent_pos = None
        self.goal_pos = None
        self.steps = 0

        # Reward 계산용으로만 사용한다.
        # PPO Observation에는 포함하지 않는다.
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
        BFS로 Agent -> Goal 경로가 실제 존재하는지 검사한다.
        맵 생성 시 해결 가능한 맵만 사용하기 위한 용도다.
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

    def _observe_current_area(self):
        """
        현재 Agent 주변 3x3을 discovered_map에 기록한다.

        반환값:
            이번 관측으로 처음 밝혀진 '맵 내부' 칸 수

        장애물도 처음 확인한 경우 새로운 정보이므로 1칸으로 센다.
        맵 바깥은 세지 않는다.
        """

        ax, ay = self.agent_pos
        newly_discovered = 0

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                x = ax + dx
                y = ay + dy

                if not (
                    0 <= x < self.grid_size
                    and 0 <= y < self.grid_size
                ):
                    continue

                if not self.discovered_map[y, x]:
                    self.discovered_map[y, x] = True
                    newly_discovered += 1

        return newly_discovered

    def _get_observation(self):
        ax, ay = self.agent_pos

        local_cells = []

        # Agent를 중심으로 주변 3x3 관측
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

        # Goal까지의 상대 방향
        goal_dx = (
            self.goal_pos[0] - self.agent_pos[0]
        ) / (self.grid_size - 1)

        goal_dy = (
            self.goal_pos[1] - self.agent_pos[1]
        ) / (self.grid_size - 1)

        observation = np.array(
            local_cells + [goal_dx, goal_dy],
            dtype=np.float32,
        )

        return observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0

        # 반드시 해결 가능한 맵이 나올 때까지 생성
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

            # 시작점과 Goal에는 장애물 제거
            ax, ay = self.agent_pos
            gx, gy = self.goal_pos

            self.grid[ay, ax] = 0
            self.grid[gy, gx] = 0

            # 실제 경로가 있는 맵만 사용
            if self._is_path_available():
                break

        # 아직 아무것도 발견하지 않은 상태
        self.discovered_map = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=bool,
        )

        # 시작 위치에서 보이는 3x3은 이미 알고 시작한다.
        # 여기에는 Reward를 지급하지 않는다.
        self._observe_current_area()

        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1

        new_pos = self.agent_pos.copy()

        if action == 0:
            new_pos[1] -= 1
        elif action == 1:
            new_pos[1] += 1
        elif action == 2:
            new_pos[0] -= 1
        elif action == 3:
            new_pos[0] += 1

        # 맵 밖이면 이동 실패
        if (
            new_pos[0] < 0
            or new_pos[0] >= self.grid_size
            or new_pos[1] < 0
            or new_pos[1] >= self.grid_size
        ):
            new_pos = self.agent_pos.copy()

        # 장애물이면 이동 실패
        nx, ny = new_pos

        if self.grid[ny, nx] == 1:
            new_pos = self.agent_pos.copy()

        self.agent_pos = new_pos

        # -------------------------------------------------
        # 사용자가 설계한 Reward
        # -------------------------------------------------

        # 모든 행동에는 1점 비용
        reward = self.step_penalty

        # 이동 후 현재 위치 주변 3x3을 관측하고
        # 새롭게 밝혀진 칸 수를 계산
        newly_discovered = self._observe_current_area()

        # 새로 밝힌 칸 1개당 +0.2
        reward += (
            newly_discovered
            * self.discovery_reward
        )

        terminated = False

        # Goal 도착
        if np.array_equal(
            self.agent_pos,
            self.goal_pos,
        ):
            reward += self.goal_reward
            terminated = True

        truncated = self.steps >= self.max_steps

        info = {
            "newly_discovered": newly_discovered,
            "discovered_cells": int(
                np.sum(self.discovered_map)
            ),
        }

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        output = []

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

            output.append(" ".join(row))

        print()

        for row in output:
            print(row)

        print()


if __name__ == "__main__":
    env = PartialObstacleExploreRewardEnv()

    check_env(env)

    print("환경 검사 성공!")
    print("Observation shape:", env.observation_space.shape)

    obs, _ = env.reset(seed=0)

    print("시작 시 발견된 칸:", np.sum(env.discovered_map))
    print("Observation:")
    print(obs)

    for i in range(5):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"Step {i + 1} | "
            f"Action {action} | "
            f"New {info['newly_discovered']} | "
            f"Discovered {info['discovered_cells']} | "
            f"Reward {reward:.2f}"
        )

        if terminated or truncated:
            break