from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env


class PartialObstacleMemoryGridEnv(gym.Env):
    """
    Experiment #02

    Baseline과 동일한 9x9 partial-observation GridWorld.

    추가 정보:
    1. Agent가 실제로 관측한 3x3 영역을 누적해서 기억
    2. Agent가 실제로 밟은 위치를 별도로 기억

    실제 전체 맵의 미관측 영역은 Agent에게 공개하지 않는다.
    """

    def __init__(self):
        super().__init__()

        self.grid_size = 9
        self.max_steps = 100
        self.obstacle_probability = 0.18

        # 0: UP
        # 1: DOWN
        # 2: LEFT
        # 3: RIGHT
        self.action_space = spaces.Discrete(4)

        # ----------------------------------------------------
        # Observation
        #
        # current local 3x3 :  9
        # goal dx/dy        :  2
        # explored map      : 81
        # visited map       : 81
        # -----------------------
        # total             : 173
        #
        # explored map:
        # -1 = 아직 관측하지 않음 (UNKNOWN)
        #  0 = 관측했고 이동 가능
        #  1 = 관측한 장애물
        #
        # visited map:
        #  0 = 실제로 밟은 적 없음
        #  1 = 실제로 밟은 적 있음
        # ----------------------------------------------------

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(173,),
            dtype=np.float32,
        )

        self.grid = None
        self.agent_pos = None
        self.goal_pos = None
        self.steps = 0

        self.explored_map = None
        self.visited_map = None

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
        실제 맵에서 Start -> Goal 경로가 존재하는지 확인한다.

        이것은 환경 생성용 검사일 뿐이며,
        Agent Observation에는 전체 경로를 제공하지 않는다.
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
        현재 위치에서 실제로 볼 수 있는 3x3만 explored_map에 기록한다.
        미관측 영역의 실제 grid 값은 복사하지 않는다.
        """

        ax, ay = self.agent_pos

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                x = ax + dx
                y = ay + dy

                if not (
                    0 <= x < self.grid_size
                    and 0 <= y < self.grid_size
                ):
                    continue

                if self.grid[y, x] == 1:
                    self.explored_map[y, x] = 1.0
                else:
                    self.explored_map[y, x] = 0.0

    def _mark_current_position_visited(self):
        ax, ay = self.agent_pos
        self.visited_map[ay, ax] = 1.0

    def _get_local_cells(self):
        ax, ay = self.agent_pos

        local_cells = []

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

        return local_cells

    def _get_observation(self):
        local_cells = self._get_local_cells()

        goal_dx = (
            self.goal_pos[0] - self.agent_pos[0]
        ) / (self.grid_size - 1)

        goal_dy = (
            self.goal_pos[1] - self.agent_pos[1]
        ) / (self.grid_size - 1)

        observation = np.concatenate(
            [
                np.asarray(local_cells, dtype=np.float32),
                np.asarray(
                    [goal_dx, goal_dy],
                    dtype=np.float32,
                ),
                self.explored_map.flatten(),
                self.visited_map.flatten(),
            ]
        ).astype(np.float32)

        return observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0

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

            self.grid[ay, ax] = 0
            self.grid[gy, gx] = 0

            if self._is_path_available():
                break

        # 모든 칸은 처음에는 UNKNOWN.
        self.explored_map = np.full(
            (self.grid_size, self.grid_size),
            -1.0,
            dtype=np.float32,
        )

        # 아직 실제 방문한 칸 없음.
        self.visited_map = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=np.float32,
        )

        # 시작 위치에서 3x3을 관측하고,
        # 시작 위치는 방문한 것으로 기록.
        self._observe_current_area()
        self._mark_current_position_visited()

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

        # 맵 바깥
        if (
            new_pos[0] < 0
            or new_pos[0] >= self.grid_size
            or new_pos[1] < 0
            or new_pos[1] >= self.grid_size
        ):
            new_pos = self.agent_pos.copy()

        # 장애물
        nx, ny = new_pos

        if self.grid[ny, nx] == 1:
            new_pos = self.agent_pos.copy()

        self.agent_pos = new_pos

        # 실제로 도달한 현재 위치를 방문 처리하고,
        # 현재 위치에서 새롭게 보이는 3x3을 누적한다.
        self._mark_current_position_visited()
        self._observe_current_area()

        # Baseline과 동일한 Sparse Reward.
        reward = 0.0
        terminated = False

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

                if np.array_equal(pos, self.agent_pos):
                    row.append("A")
                elif np.array_equal(pos, self.goal_pos):
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
    env = PartialObstacleMemoryGridEnv()

    check_env(env)

    obs, _ = env.reset(seed=0)

    print("Environment check passed.")
    print("Observation shape:", obs.shape)
    print("Expected shape   :", (173,))