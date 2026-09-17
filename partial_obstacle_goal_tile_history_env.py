from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env


class PartialObstacleGoalTileHistoryEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.grid_size = 9
        self.max_steps = 100
        self.obstacle_probability = 0.18

        # -------------------------------------------------
        # Reward
        # 이전 Goal Tile Reward 실험과 동일
        # -------------------------------------------------
        self.step_penalty = 0.0
        self.discovery_reward = 0.0
        self.collision_penalty = -1.0
        self.goal_reward = 100.0
        self.tile_reward_max = 9.0

        # -------------------------------------------------
        # History
        # -------------------------------------------------
        # 최근 4개 위치를 기억
        self.history_length = 4

        # -------------------------------------------------
        # Action
        # -------------------------------------------------
        # 0: UP
        # 1: DOWN
        # 2: LEFT
        # 3: RIGHT
        self.action_space = spaces.Discrete(4)

        # -------------------------------------------------
        # Observation
        #
        # 기존:
        # Local 3x3        9
        # Goal dx/dy       2
        # Discovered Map  81
        # Agent x/y        2
        # ------------------
        #                  94
        #
        # 추가:
        # Last Action      4 (one-hot)
        # Last Collision   1
        # Position History 8 (4 positions x/y)
        # ------------------
        # 총              107
        # -------------------------------------------------
        self.observation_size = 107

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
        self.visited_positions = None

        # 직전 행동
        # reset 직후에는 행동이 없으므로 None
        self.last_action = None

        # 직전 행동이 충돌했는지
        self.last_collision = False

        # 최근 위치 4개
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
        랜덤 생성된 맵에서
        Start -> Goal 경로가 실제 존재하는지 BFS 검사.

        PPO에게 전체 경로나 BFS 결과는 제공하지 않는다.
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

        이번 실험에서는 발견 자체에 Reward 없음.
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
        처음 방문한 일반 타일에 지급하는
        Goal Manhattan Distance 기반 Reward.

        distance 1 -> +8
        distance 2 -> +7
        ...
        distance >= 8 -> +1

        Goal은 별도로 +100.
        """

        ax, ay = self.agent_pos
        gx, gy = self.goal_pos

        manhattan_distance = (
            abs(int(gx) - int(ax))
            + abs(int(gy) - int(ay))
        )

        return float(
            max(
                1.0,
                self.tile_reward_max
                - manhattan_distance,
            )
        )

    def _get_last_action_one_hot(self):
        """
        직전 Action을 one-hot으로 반환.

        reset 직후:
        [0, 0, 0, 0]

        예: 직전 Action RIGHT(3)
        [0, 0, 0, 1]
        """

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
        """
        최근 4개 위치를 x/y 정규화하여 반환.

        예:
        [
            x1, y1,
            x2, y2,
            x3, y3,
            x4, y4
        ]

        좌표 0~8 -> 0~1
        """

        values = []

        for x, y in self.position_history:
            values.append(
                x / (self.grid_size - 1)
            )

            values.append(
                y / (self.grid_size - 1)
            )

        return values

    def _get_observation(self):
        ax, ay = self.agent_pos

        local_cells = []

        # -------------------------------------------------
        # 1. Local 3x3 = 9
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
        # 4. Current Agent x/y = 2
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

        observation = np.array(
            local_cells
            + [goal_dx, goal_dy]
            + discovered_flat
            + [agent_x, agent_y]
            + last_action_one_hot
            + [last_collision]
            + position_history,
            dtype=np.float32,
        )

        assert len(observation) == 107, (
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
        # Discovered Map 초기화
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

        # -------------------------------------------------
        # 최초 방문 Reward 기록
        # -------------------------------------------------
        start_position = tuple(
            int(v)
            for v in self.agent_pos
        )

        self.visited_positions = {
            start_position
        }

        # -------------------------------------------------
        # History 초기화
        # -------------------------------------------------
        self.last_action = None
        self.last_collision = False

        # 아직 과거 위치가 없으므로
        # 시작 위치로 4칸 모두 채운다.
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

        # -------------------------------------------------
        # Collision Reward
        # -------------------------------------------------
        if collision:
            reward += (
                self.collision_penalty
            )

        # 실제 위치 반영
        self.agent_pos = new_pos

        # -------------------------------------------------
        # 지도 갱신
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
            reward += self.goal_reward
            terminated = True

        # -------------------------------------------------
        # 처음 방문한 일반 타일
        # -------------------------------------------------
        else:
            current_position = tuple(
                int(v)
                for v in self.agent_pos
            )

            if (
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
        # ★ 이번 실험의 핵심
        # Action 결과를 다음 Observation에 기록
        # -------------------------------------------------
        self.last_action = action
        self.last_collision = collision

        current_position = tuple(
            int(v)
            for v in self.agent_pos
        )

        self.position_history.append(
            current_position
        )

        # -------------------------------------------------
        # Max Step
        # -------------------------------------------------
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

            "agent_x": int(
                self.agent_pos[0]
            ),

            "agent_y": int(
                self.agent_pos[1]
            ),

            "collision":
                collision,

            "first_visit":
                first_visit,

            "tile_reward":
                tile_reward,

            "visited_positions": len(
                self.visited_positions
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

                elif (
                    self.grid[y, x] == 1
                ):
                    row.append("#")

                else:
                    row.append(".")

            print(
                " ".join(row)
            )

        print()


if __name__ == "__main__":
    env = (
        PartialObstacleGoalTileHistoryEnv()
    )

    check_env(env)

    print("환경 검사 성공!")

    print(
        "Observation shape:",
        env.observation_space.shape,
    )

    print()
    print("=== Observation ===")
    print("Local 3x3        : 9")
    print("Goal dx/dy       : 2")
    print("Discovered map   : 81")
    print("Agent x/y        : 2")
    print("Last Action      : 4")
    print("Last Collision   : 1")
    print("Position History : 8")
    print("----------------------")
    print("Total            : 107")

    print()
    print("=== Reward ===")

    print(
        f"Step penalty      : "
        f"{env.step_penalty:+.1f}"
    )

    print(
        f"Discovery reward  : "
        f"{env.discovery_reward:+.1f}"
    )

    print(
        f"Collision penalty : "
        f"{env.collision_penalty:+.1f}"
    )

    print(
        "Tile reward       : "
        "first visit only, "
        "max(1, 9-distance)"
    )

    print(
        f"Goal reward       : "
        f"{env.goal_reward:+.1f}"
    )

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
            f"Reward "
            f"{reward:.1f} | "
            f"History "
            f"{info['position_history']}"
        )

        if (
            terminated
            or truncated
        ):
            break

    env.close()