from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env


class PartialObstacleMapVisitedEnv(gym.Env):
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
        # 현재 주변 3x3       = 9
        # Goal 상대 dx/dy     = 2
        # 누적 발견 지도      = 81
        # 실제 방문 지도      = 81
        #
        # 총 173
        #
        # discovered_map
        # -1 = 아직 모름
        #  0 = 확인된 이동 가능 칸
        #  1 = 확인된 장애물
        #
        # visited_map
        #  0 = 실제로 방문하지 않음
        #  1 = 실제로 방문함
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

        self.discovered_map = None
        self.visited_map = None

    def _random_position(self):
        return np.array(
            [
                self.np_random.integers(
                    0,
                    self.grid_size,
                ),
                self.np_random.integers(
                    0,
                    self.grid_size,
                ),
            ],
            dtype=np.int32,
        )

    def _is_path_available(self):
        """
        맵 생성 시 Agent -> Goal 경로가 실제 존재하는지 검사.

        이 BFS 결과나 전체 맵 정보는 PPO에게 주지 않는다.
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

                position = (nx, ny)

                if position in visited:
                    continue

                visited.add(position)
                queue.append(position)

        return False

    def _update_discovered_map(self):
        """
        현재 위치 주변 3x3을 누적 발견 지도에 기록.

        반환:
            이번 관측에서 처음 밝혀진 맵 내부 칸 수
        """

        ax, ay = self.agent_pos

        newly_discovered = 0

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                x = ax + dx
                y = ay + dy

                # 맵 바깥은 발견 칸으로 세지 않는다.
                if not (
                    0 <= x < self.grid_size
                    and 0 <= y < self.grid_size
                ):
                    continue

                # 아직 한 번도 보지 못한 칸
                if self.discovered_map[y, x] == -1.0:
                    newly_discovered += 1

                    if self.grid[y, x] == 1:
                        self.discovered_map[y, x] = 1.0
                    else:
                        self.discovered_map[y, x] = 0.0

        return newly_discovered

    def _mark_current_position_visited(self):
        """
        Agent가 실제로 위치한 칸을 방문했다고 기록한다.
        """

        x, y = self.agent_pos

        self.visited_map[y, x] = 1.0

    def _get_observation(self):
        ax, ay = self.agent_pos

        local_cells = []

        # -------------------------------------------------
        # 1. 현재 주변 3x3
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
        # 2. Goal 상대좌표
        # -------------------------------------------------
        goal_dx = (
            self.goal_pos[0]
            - self.agent_pos[0]
        ) / (self.grid_size - 1)

        goal_dy = (
            self.goal_pos[1]
            - self.agent_pos[1]
        ) / (self.grid_size - 1)

        # -------------------------------------------------
        # 3. 지금까지 밝혀진 전체 지도
        # -------------------------------------------------
        discovered_flat = (
            self.discovered_map
            .flatten()
            .tolist()
        )

        # -------------------------------------------------
        # 4. 실제 방문한 위치 지도
        # -------------------------------------------------
        visited_flat = (
            self.visited_map
            .flatten()
            .tolist()
        )

        observation = np.array(
            local_cells
            + [goal_dx, goal_dy]
            + discovered_flat
            + visited_flat,
            dtype=np.float32,
        )

        return observation

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        self.steps = 0

        # -------------------------------------------------
        # 해결 가능한 랜덤 맵 생성
        # -------------------------------------------------
        while True:
            self.grid = (
                self.np_random.random(
                    (
                        self.grid_size,
                        self.grid_size,
                    )
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

        # -------------------------------------------------
        # 지도 초기화
        # -------------------------------------------------

        # 아직 모든 월드맵을 모름
        self.discovered_map = np.full(
            (
                self.grid_size,
                self.grid_size,
            ),
            -1.0,
            dtype=np.float32,
        )

        # 아직 아무 곳도 방문하지 않음
        self.visited_map = np.zeros(
            (
                self.grid_size,
                self.grid_size,
            ),
            dtype=np.float32,
        )

        # 시작 위치는 실제 방문 위치
        self._mark_current_position_visited()

        # 시작점에서 보이는 3x3을 지도에 기록
        # 시작 시 관측에는 Reward를 주지 않는다.
        self._update_discovered_map()

        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1

        previous_pos = self.agent_pos.copy()
        new_pos = self.agent_pos.copy()

        # -------------------------------------------------
        # Action
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
        # 맵 밖
        # -------------------------------------------------
        if (
            new_pos[0] < 0
            or new_pos[0] >= self.grid_size
            or new_pos[1] < 0
            or new_pos[1] >= self.grid_size
        ):
            new_pos = self.agent_pos.copy()

        # -------------------------------------------------
        # 장애물
        # -------------------------------------------------
        nx, ny = new_pos

        if self.grid[ny, nx] == 1:
            new_pos = self.agent_pos.copy()

        self.agent_pos = new_pos

        moved = not np.array_equal(
            previous_pos,
            self.agent_pos,
        )

        # 현재 위치를 방문 지도에 기록
        self._mark_current_position_visited()

        # -------------------------------------------------
        # Reward
        # -------------------------------------------------

        # 행동 1회당 -1
        reward = self.step_penalty

        # 이동 후 현재 위치의 주변 3x3으로 지도 업데이트
        newly_discovered = (
            self._update_discovered_map()
        )

        # 처음 밝혀낸 칸 1개당 +0.2
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

        truncated = (
            self.steps >= self.max_steps
        )

        info = {
            "newly_discovered": newly_discovered,
            "discovered_cells": int(
                np.sum(
                    self.discovered_map != -1.0
                )
            ),
            "visited_cells": int(
                np.sum(
                    self.visited_map == 1.0
                )
            ),
            "moved": moved,
            "agent_x": int(
                self.agent_pos[0]
            ),
            "agent_y": int(
                self.agent_pos[1]
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
        """
        실제 전체 맵.
        디버깅용이며 PPO에게 전체 맵을 제공하지 않는다.
        """

        print()

        for y in range(self.grid_size):
            row = []

            for x in range(self.grid_size):
                position = np.array(
                    [x, y]
                )

                if np.array_equal(
                    position,
                    self.agent_pos,
                ):
                    row.append("A")

                elif np.array_equal(
                    position,
                    self.goal_pos,
                ):
                    row.append("G")

                elif self.grid[y, x] == 1:
                    row.append("#")

                else:
                    row.append(".")

            print(" ".join(row))

        print()

    def render_memory(self):
        """
        누적 발견 지도 + 방문 여부 확인용.

        ? = 아직 관측하지 않음
        # = 관측된 장애물
        . = 관측했지만 실제 방문하지 않은 빈 칸
        v = 실제 방문한 칸
        A = 현재 위치
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

                if self.discovered_map[y, x] == -1.0:
                    row.append("?")

                elif self.discovered_map[y, x] == 1.0:
                    row.append("#")

                elif self.visited_map[y, x] == 1.0:
                    row.append("v")

                else:
                    row.append(".")

            print(" ".join(row))

        print()


if __name__ == "__main__":
    env = PartialObstacleMapVisitedEnv()

    check_env(env)

    print("환경 검사 성공!")
    print(
        "Observation shape:",
        env.observation_space.shape,
    )

    obs, _ = env.reset(seed=0)

    print(
        "Observation 길이:",
        len(obs),
    )

    print(
        "시작 위치:",
        tuple(env.agent_pos),
    )

    print(
        "Goal 위치:",
        tuple(env.goal_pos),
    )

    print("\n=== 실제 전체 맵 ===")
    env.render()

    print("=== PPO가 가진 지도/방문 기록 ===")
    env.render_memory()

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
            f"({info['agent_x']}, "
            f"{info['agent_y']}) | "
            f"Moved {info['moved']} | "
            f"New {info['newly_discovered']} | "
            f"Known {info['discovered_cells']}/81 | "
            f"Visited {info['visited_cells']}/81 | "
            f"Reward {reward:.2f}"
        )

        env.render_memory()

        if terminated or truncated:
            break