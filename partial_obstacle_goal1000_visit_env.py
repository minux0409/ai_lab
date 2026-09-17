from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env


class PartialObstacleGoal1000VisitEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.grid_size = 9
        self.max_steps = 100
        self.obstacle_probability = 0.18

        # -------------------------------------------------
        # Reward
        # -------------------------------------------------
        self.step_penalty = 0.0
        self.discovery_reward = 0.0
        self.collision_penalty = -1.0

        # 이번 실험 변경
        self.goal_reward = 1000.0

        # 9x9에서 최대 Manhattan Distance = 16
        # 거리 1 -> 15
        # 거리 2 -> 14
        # ...
        # 거리 15 -> 1
        # 거리 16 -> 1
        self.tile_reward_base = 16.0

        # -------------------------------------------------
        # History
        # -------------------------------------------------
        self.history_length = 4

        # Visit Count 정규화 상한
        self.visit_count_cap = 10.0

        # -------------------------------------------------
        # Action
        # -------------------------------------------------
        # 0 UP
        # 1 DOWN
        # 2 LEFT
        # 3 RIGHT
        self.action_space = spaces.Discrete(4)

        # -------------------------------------------------
        # Observation
        #
        # Local 3x3          9
        # Goal dx/dy         2
        # Discovered Map    81
        # Agent x/y          2
        # Last Action        4
        # Last Collision     1
        # Position History   8
        # Visit Count Map   81
        # --------------------
        # Total            188
        # -------------------------------------------------
        self.observation_size = 188

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.observation_size,),
            dtype=np.float32,
        )

        self.grid = None
        self.agent_pos = None
        self.goal_pos = None

        self.steps = 0

        self.discovered_map = None

        # 첫 방문 Reward 판정용
        self.visited_positions = None

        # 각 칸 실제 방문 횟수
        self.visit_count_map = None

        self.last_action = None
        self.last_collision = False

        self.position_history = None

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
        Start -> Goal 경로 존재 여부만 BFS로 검사.
        PPO에는 BFS 결과나 경로를 제공하지 않는다.
        """

        start = tuple(
            int(v)
            for v in self.agent_pos
        )

        goal = tuple(
            int(v)
            for v in self.goal_pos
        )

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
        현재 위치 주변 3x3을 누적 지도에 기록.

        -1 = Unknown
         0 = Free
         1 = Obstacle
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

                if self.discovered_map[y, x] == -1.0:
                    newly_discovered += 1

                    if self.grid[y, x] == 1:
                        self.discovered_map[y, x] = 1.0
                    else:
                        self.discovered_map[y, x] = 0.0

        return newly_discovered

    def _get_tile_reward(self):
        """
        Goal까지 Manhattan Distance 기반 첫 방문 Reward.

        distance 1  -> +15
        distance 2  -> +14
        ...
        distance 15 -> +1
        distance 16 -> +1

        Goal 자체는 +1000.
        """

        ax, ay = self.agent_pos
        gx, gy = self.goal_pos

        distance = (
            abs(int(gx) - int(ax))
            + abs(int(gy) - int(ay))
        )

        return float(
            max(
                1.0,
                self.tile_reward_base
                - distance,
            )
        )

    def _get_last_action_one_hot(self):
        one_hot = np.zeros(
            4,
            dtype=np.float32,
        )

        if self.last_action is not None:
            one_hot[
                int(self.last_action)
            ] = 1.0

        return one_hot.tolist()

    def _get_position_history_observation(self):
        values = []

        for x, y in self.position_history:
            values.append(
                x / (self.grid_size - 1)
            )

            values.append(
                y / (self.grid_size - 1)
            )

        return values

    def _get_visit_count_observation(self):
        """
        각 칸 방문 횟수를 0~1로 정규화.

        0회   -> 0.0
        1회   -> 0.1
        2회   -> 0.2
        ...
        10회+ -> 1.0

        이것은 Observation일 뿐,
        방문 횟수 자체에 Reward/Penalty는 없다.
        """

        normalized = np.minimum(
            self.visit_count_map,
            self.visit_count_cap,
        ) / self.visit_count_cap

        return (
            normalized
            .astype(np.float32)
            .flatten()
            .tolist()
        )

    def _get_observation(self):
        ax, ay = self.agent_pos

        # -------------------------------------------------
        # 1. Local 3x3 = 9
        # -------------------------------------------------
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

        # -------------------------------------------------
        # 2. Goal dx/dy = 2
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
        # 3. Discovered Map = 81
        # -------------------------------------------------
        discovered_flat = (
            self.discovered_map
            .flatten()
            .tolist()
        )

        # -------------------------------------------------
        # 4. Agent x/y = 2
        # -------------------------------------------------
        agent_x = (
            self.agent_pos[0]
            / (self.grid_size - 1)
        )

        agent_y = (
            self.agent_pos[1]
            / (self.grid_size - 1)
        )

        # -------------------------------------------------
        # 5. Last Action = 4
        # -------------------------------------------------
        last_action_one_hot = (
            self._get_last_action_one_hot()
        )

        # -------------------------------------------------
        # 6. Last Collision = 1
        # -------------------------------------------------
        last_collision = (
            1.0
            if self.last_collision
            else 0.0
        )

        # -------------------------------------------------
        # 7. Position History = 8
        # -------------------------------------------------
        position_history = (
            self._get_position_history_observation()
        )

        # -------------------------------------------------
        # 8. Visit Count Map = 81
        # -------------------------------------------------
        visit_count_flat = (
            self._get_visit_count_observation()
        )

        observation = np.array(
            local_cells
            + [goal_dx, goal_dy]
            + discovered_flat
            + [agent_x, agent_y]
            + last_action_one_hot
            + [last_collision]
            + position_history
            + visit_count_flat,
            dtype=np.float32,
        )

        assert len(observation) == 188, (
            f"Observation 길이 오류: "
            f"{len(observation)}"
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

            self.agent_pos = (
                self._random_position()
            )

            self.goal_pos = (
                self._random_position()
            )

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
        # Discovered Map
        # -------------------------------------------------
        self.discovered_map = np.full(
            (
                self.grid_size,
                self.grid_size,
            ),
            -1.0,
            dtype=np.float32,
        )

        self._update_discovered_map()

        start_position = tuple(
            int(v)
            for v in self.agent_pos
        )

        # -------------------------------------------------
        # First Visit 기록
        # -------------------------------------------------
        self.visited_positions = {
            start_position
        }

        # -------------------------------------------------
        # Visit Count Map
        #
        # 시작 위치는 이미 현재 서 있는 칸이므로 1회
        # -------------------------------------------------
        self.visit_count_map = np.zeros(
            (
                self.grid_size,
                self.grid_size,
            ),
            dtype=np.float32,
        )

        sx, sy = start_position

        self.visit_count_map[
            sy,
            sx,
        ] = 1.0

        # -------------------------------------------------
        # Short History
        # -------------------------------------------------
        self.last_action = None
        self.last_collision = False

        self.position_history = deque(
            [
                start_position
                for _ in range(
                    self.history_length
                )
            ],
            maxlen=self.history_length,
        )

        return (
            self._get_observation(),
            {},
        )

    def step(self, action):
        self.steps += 1

        action = int(action)

        old_pos = self.agent_pos.copy()
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

        reward = 0.0
        collision = False

        # -------------------------------------------------
        # Map Out
        # -------------------------------------------------
        if (
            new_pos[0] < 0
            or new_pos[0] >= self.grid_size
            or new_pos[1] < 0
            or new_pos[1] >= self.grid_size
        ):
            collision = True
            new_pos = old_pos.copy()

        # -------------------------------------------------
        # Wall
        # -------------------------------------------------
        if not collision:
            nx, ny = new_pos

            if self.grid[ny, nx] == 1:
                collision = True
                new_pos = old_pos.copy()

        if collision:
            reward += (
                self.collision_penalty
            )

        self.agent_pos = new_pos

        # -------------------------------------------------
        # Visit Count 갱신
        #
        # 충돌 시 위치가 그대로여도
        # "현재 칸에 다시 머문 것"으로 방문 횟수를 증가시킨다.
        #
        # 따라서 같은 벽에 계속 박으면
        # 해당 칸의 Visit Count가 빠르게 올라간다.
        # -------------------------------------------------
        current_position = tuple(
            int(v)
            for v in self.agent_pos
        )

        cx, cy = current_position

        self.visit_count_map[
            cy,
            cx,
        ] += 1.0

        # -------------------------------------------------
        # Discovered Map 갱신
        # -------------------------------------------------
        newly_discovered = (
            self._update_discovered_map()
        )

        terminated = False
        first_visit = False
        tile_reward = 0.0

        # -------------------------------------------------
        # Goal
        # -------------------------------------------------
        if np.array_equal(
            self.agent_pos,
            self.goal_pos,
        ):
            reward += (
                self.goal_reward
            )

            terminated = True

        # -------------------------------------------------
        # 처음 방문한 일반 타일
        # -------------------------------------------------
        elif (
            not collision
            and current_position
            not in self.visited_positions
        ):
            first_visit = True

            tile_reward = (
                self._get_tile_reward()
            )

            reward += tile_reward

            self.visited_positions.add(
                current_position
            )

        # -------------------------------------------------
        # Short History 갱신
        # -------------------------------------------------
        self.last_action = action
        self.last_collision = collision

        self.position_history.append(
            current_position
        )

        truncated = (
            self.steps >= self.max_steps
            and not terminated
        )

        info = {
            "newly_discovered":
                newly_discovered,

            "discovered_cells": int(
                np.sum(
                    self.discovered_map
                    != -1.0
                )
            ),

            "agent_x":
                int(self.agent_pos[0]),

            "agent_y":
                int(self.agent_pos[1]),

            "collision":
                collision,

            "first_visit":
                first_visit,

            "tile_reward":
                tile_reward,

            "visited_positions":
                len(
                    self.visited_positions
                ),

            "current_visit_count":
                int(
                    self.visit_count_map[
                        cy,
                        cx,
                    ]
                ),

            "max_visit_count":
                int(
                    np.max(
                        self.visit_count_map
                    )
                ),

            "last_action":
                self.last_action,

            "last_collision":
                self.last_collision,

            "position_history":
                list(
                    self.position_history
                ),
        }

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self):
        print()

        for y in range(
            self.grid_size
        ):
            row = []

            for x in range(
                self.grid_size
            ):
                pos = np.array(
                    [x, y]
                )

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

            print(
                " ".join(row)
            )

        print()

    def render_visit_counts(self):
        print()
        print("=== Visit Count Map ===")

        for y in range(
            self.grid_size
        ):
            row = []

            for x in range(
                self.grid_size
            ):
                row.append(
                    f"{int(self.visit_count_map[y, x]):3d}"
                )

            print(
                " ".join(row)
            )

        print()


if __name__ == "__main__":
    env = PartialObstacleGoal1000VisitEnv()

    check_env(env)

    print("환경 검사 성공!")

    print(
        "Observation shape:",
        env.observation_space.shape,
    )

    print()
    print("=== Observation ===")
    print("Local 3x3        :   9")
    print("Goal dx/dy       :   2")
    print("Discovered Map   :  81")
    print("Agent x/y        :   2")
    print("Last Action      :   4")
    print("Last Collision   :   1")
    print("Position History :   8")
    print("Visit Count Map  :  81")
    print("-----------------------")
    print("Total            : 188")

    print()
    print("=== Reward ===")
    print("Step              : +0")
    print("Discovery         : +0")
    print("Collision         : -1")
    print("Revisit           : +0")
    print("Distance 1        : +15")
    print("Distance 2        : +14")
    print("...")
    print("Distance 15       : +1")
    print("Distance 16       : +1")
    print("Goal              : +1000")

    obs, _ = env.reset(seed=0)

    print()
    print(
        "Observation length:",
        len(obs),
    )

    print(
        "Start:",
        tuple(env.agent_pos),
    )

    print(
        "Goal:",
        tuple(env.goal_pos),
    )

    print()
    print("=== Random 10 Step ===")

    for i in range(10):
        action = (
            env.action_space.sample()
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        print(
            f"{i + 1:2d} | "
            f"Action {action} | "
            f"Pos "
            f"({info['agent_x']},"
            f"{info['agent_y']}) | "
            f"Collision "
            f"{info['collision']} | "
            f"First "
            f"{info['first_visit']} | "
            f"Tile "
            f"{info['tile_reward']:.1f} | "
            f"Visit "
            f"{info['current_visit_count']} | "
            f"Reward "
            f"{reward:.1f}"
        )

        if terminated or truncated:
            break

    env.render_visit_counts()
    env.close()