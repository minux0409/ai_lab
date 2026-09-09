from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env


class PartialObstacleGridEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.grid_size = 9
        self.max_steps = 100
        self.obstacle_probability = 0.18

        # 0: 위
        # 1: 아래
        # 2: 왼쪽
        # 3: 오른쪽
        self.action_space = spaces.Discrete(4)

        # Observation
        # 주변 3x3 = 9개
        # Goal 상대 위치 dx, dy = 2개
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

        # 이번 실험에서 추가:
        # 현재 Episode에서 방문했던 위치 기록
        self.visited_positions = set()

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
        BFS로 Agent -> Goal 경로가 실제 존재하는지 검사
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

    def _get_observation(self):
        ax, ay = self.agent_pos

        local_cells = []

        # Agent 중심 주변 3x3 관측
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

        # Goal 상대 방향
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

        # 매 Episode마다 방문기록 초기화
        self.visited_positions = set()

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

            # 시작점과 Goal에는 장애물 제거
            self.grid[ay, ax] = 0
            self.grid[gy, gx] = 0

            # 실제 경로가 존재하는 맵만 사용
            if self._is_path_available():
                break

        # 시작 위치는 이미 방문한 것으로 기록
        self.visited_positions.add(
            tuple(self.agent_pos)
        )

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

        # 맵 밖으로 나가면 이동 실패
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

        position = tuple(self.agent_pos)

        reward = 0.0
        terminated = False

        # 이번 실험 핵심:
        # 이미 방문했던 위치를 다시 방문하면 작은 패널티
        if position in self.visited_positions:
            reward -= 0.05

        self.visited_positions.add(position)

        # Goal 도착
        if np.array_equal(
            self.agent_pos,
            self.goal_pos,
        ):
            reward = 10.0
            terminated = True

        truncated = self.steps >= self.max_steps

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            {},
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
    env = PartialObstacleGridEnv()

    check_env(env)

    print("Revisit Penalty 환경 검사 성공!")

    obs, _ = env.reset(seed=10000)

    print("초기 Observation:")
    print(obs)

    env.render()