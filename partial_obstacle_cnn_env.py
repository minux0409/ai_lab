from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env


class PartialObstacleCnnEnv(gym.Env):
    """
    Observation
    -----------
    1. 밝혀진 9x9 지도             : 81
    2. Goal 거리 점수 9x9          : 81
    3. 현재 Agent 위치 x/y         : 2
    4. 각 칸 방문 횟수             : 81
    5. 직전 Action one-hot          : 4

    Total                         : 249

    Goal 거리 점수
    -------------
    Goal          = 100
    Manhattan 1   = 99
    Manhattan 2   = 98
    ...

    Reward
    ------
    Step                  = -1
    Newly discovered cell = +0.2
    Goal                  = +100
    """

    metadata = {
        "render_modes": []
    }

    def __init__(self):
        super().__init__()

        self.grid_size = 9
        self.max_steps = 100
        self.obstacle_probability = 0.18

        self.step_penalty = -1.0
        self.discovery_reward = 0.1
        self.collision_penalty = -2.0
        self.first_visit_reward = 1.0
        self.goal_reward = 100.0

        # 0 = UP
        # 1 = DOWN
        # 2 = LEFT
        # 3 = RIGHT
        self.action_space = spaces.Discrete(4)

        # CNN용 공간 Observation.
        # 5 channels x 9 x 9:
        # 0 unknown mask
        # 1 known obstacle mask
        # 2 visit count (0~1)
        # 3 current agent position
        # 4 goal position
        #
        # 지도는 flatten하지 않고 2D 공간 구조를 그대로 유지한다.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5, self.grid_size, self.grid_size),
            dtype=np.float32,
        )

        self.grid = None
        self.agent_pos = None
        self.goal_pos = None

        self.steps = 0

        self.discovered_map = None
        self.visit_count_map = None

        self.previous_action = None

    # ---------------------------------------------------------
    # Map generation
    # ---------------------------------------------------------

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
        Agent -> Goal까지 실제 경로가 존재하는지 BFS로 확인한다.

        이 정보는 맵 생성 검증에만 사용하며
        PPO Observation에는 제공하지 않는다.
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

    # ---------------------------------------------------------
    # Memory
    # ---------------------------------------------------------

    def _update_discovered_map(self):
        """
        현재 위치에서 보이는 3x3을 누적 지도에 기록한다.

        return:
            이번 Step에서 새로 밝혀진 맵 내부 칸 수
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

                if self.discovered_map[y, x] != -1.0:
                    continue

                newly_discovered += 1

                if self.grid[y, x] == 1:
                    self.discovered_map[y, x] = 1.0
                else:
                    self.discovered_map[y, x] = 0.0

        return newly_discovered

    def _record_visit(self):
        """
        현재 위치의 방문 횟수를 +1 한다.

        벽에 부딪혀 실제 위치가 변하지 않았더라도
        현재 칸에 머문 것이므로 해당 위치 횟수가 증가한다.
        """

        x, y = self.agent_pos

        self.visit_count_map[y, x] += 1

    # ---------------------------------------------------------
    # Goal field
    # ---------------------------------------------------------

    def _get_goal_score_map(self):
        """
        각 칸의 Goal Manhattan Distance를 점수로 변환.

        Goal = 100
        거리 1 = 99
        거리 2 = 98
        ...

        9x9 최대 Manhattan Distance = 16
        따라서 최소 점수 = 84.

        PPO 입력에서는 /100 하여
        0.84 ~ 1.00 범위로 제공한다.
        """

        goal_x, goal_y = self.goal_pos

        score_map = np.zeros(
            (
                self.grid_size,
                self.grid_size,
            ),
            dtype=np.float32,
        )

        for y in range(self.grid_size):
            for x in range(self.grid_size):

                distance = (
                    abs(x - goal_x)
                    + abs(y - goal_y)
                )

                score = (
                    100.0
                    - float(distance)
                )

                score_map[y, x] = (
                    score / 100.0
                )

        return score_map

    # ---------------------------------------------------------
    # Observation
    # ---------------------------------------------------------

    def _get_previous_action_one_hot(self):

        result = np.zeros(
            4,
            dtype=np.float32,
        )

        if self.previous_action is not None:
            result[
                self.previous_action
            ] = 1.0

        return result

    def _get_observation(self):
        obs = np.zeros(
            (5, self.grid_size, self.grid_size),
            dtype=np.float32,
        )

        # discovered_map: -1 unknown, 0 known free, 1 obstacle
        obs[0] = (self.discovered_map == -1.0).astype(np.float32)
        obs[1] = (self.discovered_map == 1.0).astype(np.float32)

        obs[2] = np.clip(
            self.visit_count_map / float(self.max_steps),
            0.0,
            1.0,
        ).astype(np.float32)

        ax, ay = self.agent_pos
        gx, gy = self.goal_pos

        obs[3, ay, ax] = 1.0
        obs[4, gy, gx] = 1.0

        return obs

    # ---------------------------------------------------------
    # Gym
    # ---------------------------------------------------------

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(
            seed=seed
        )

        self.steps = 0
        self.previous_action = None

        # 해결 가능한 랜덤맵 생성
        while True:

            self.grid = (
                self.np_random.random(
                    (
                        self.grid_size,
                        self.grid_size,
                    )
                )
                < self.obstacle_probability
            ).astype(
                np.int32
            )

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

        # 아직 보지 않은 지도
        self.discovered_map = np.full(
            (
                self.grid_size,
                self.grid_size,
            ),
            -1.0,
            dtype=np.float32,
        )

        # 방문 횟수
        self.visit_count_map = np.zeros(
            (
                self.grid_size,
                self.grid_size,
            ),
            dtype=np.float32,
        )

        # 시작 위치도 방문 1회
        self._record_visit()

        # 시작 위치에서 보이는 영역 기록
        # Reset에서는 발견 Reward 없음
        self._update_discovered_map()

        return (
            self._get_observation(),
            {},
        )

    def step(
        self,
        action,
    ):
        self.steps += 1

        action = int(action)

        previous_position = (
            self.agent_pos.copy()
        )

        new_position = (
            self.agent_pos.copy()
        )

        # -----------------------------------------------------
        # Movement
        # -----------------------------------------------------

        if action == 0:
            new_position[1] -= 1

        elif action == 1:
            new_position[1] += 1

        elif action == 2:
            new_position[0] -= 1

        elif action == 3:
            new_position[0] += 1

        # 맵 밖
        if (
            new_position[0] < 0
            or new_position[0] >= self.grid_size
            or new_position[1] < 0
            or new_position[1] >= self.grid_size
        ):
            new_position = (
                self.agent_pos.copy()
            )

        # 장애물
        nx, ny = new_position

        if self.grid[ny, nx] == 1:
            new_position = (
                self.agent_pos.copy()
            )

        self.agent_pos = (
            new_position
        )

        moved = not np.array_equal(
            previous_position,
            self.agent_pos,
        )

        # -----------------------------------------------------
        # Memory update
        # -----------------------------------------------------

        ax, ay = self.agent_pos
        first_visit = (
            moved
            and self.visit_count_map[ay, ax] == 0
        )

        self._record_visit()

        newly_discovered = (
            self._update_discovered_map()
        )

        # 이번 Action은 다음 Observation에서
        # previous action이 된다.
        self.previous_action = action

        # -----------------------------------------------------
        # Reward
        # -----------------------------------------------------

        reward = (
            self.step_penalty
        )

        # 벽/경계 충돌: 기본 Step -1에 추가 -2 => 총 -3
        if not moved:
            reward += self.collision_penalty

        # 실제 이동으로 처음 밟은 칸에만 +1
        if first_visit:
            reward += self.first_visit_reward

        reward += (
            newly_discovered
            * self.discovery_reward
        )

        terminated = False

        if np.array_equal(
            self.agent_pos,
            self.goal_pos,
        ):
            reward += (
                self.goal_reward
            )

            terminated = True

        truncated = (
            self.steps
            >= self.max_steps
        )

        info = {
            "moved": moved,
            "first_visit": bool(first_visit),

            "newly_discovered":
                newly_discovered,

            "discovered_cells":
                int(
                    np.sum(
                        self.discovered_map
                        != -1.0
                    )
                ),

            "unique_visited_cells":
                int(
                    np.sum(
                        self.visit_count_map
                        > 0
                    )
                ),

            "agent_x":
                int(
                    self.agent_pos[0]
                ),

            "agent_y":
                int(
                    self.agent_pos[1]
                ),

            "goal_x":
                int(
                    self.goal_pos[0]
                ),

            "goal_y":
                int(
                    self.goal_pos[1]
                ),

            "goal_distance":
                int(
                    abs(
                        self.agent_pos[0]
                        - self.goal_pos[0]
                    )
                    + abs(
                        self.agent_pos[1]
                        - self.goal_pos[1]
                    )
                ),

            "goal_score":
                float(
                    100
                    - (
                        abs(
                            self.agent_pos[0]
                            - self.goal_pos[0]
                        )
                        + abs(
                            self.agent_pos[1]
                            - self.goal_pos[1]
                        )
                    )
                ),
        }

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    # ---------------------------------------------------------
    # Debug
    # ---------------------------------------------------------

    def render(self):

        print()

        for y in range(
            self.grid_size
        ):

            row = []

            for x in range(
                self.grid_size
            ):

                if (
                    x == self.agent_pos[0]
                    and y == self.agent_pos[1]
                ):
                    row.append("A")

                elif (
                    x == self.goal_pos[0]
                    and y == self.goal_pos[1]
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

    def render_memory(self):

        print()

        for y in range(
            self.grid_size
        ):

            row = []

            for x in range(
                self.grid_size
            ):

                if (
                    x == self.agent_pos[0]
                    and y == self.agent_pos[1]
                ):
                    row.append("A")

                elif (
                    self.discovered_map[y, x]
                    == -1.0
                ):
                    row.append("?")

                elif (
                    self.discovered_map[y, x]
                    == 1.0
                ):
                    row.append("#")

                elif (
                    self.visit_count_map[y, x]
                    > 0
                ):
                    row.append(
                        str(
                            min(
                                int(
                                    self.visit_count_map[
                                        y,
                                        x
                                    ]
                                ),
                                9,
                            )
                        )
                    )

                else:
                    row.append(".")

            print(
                " ".join(row)
            )

        print()


if __name__ == "__main__":

    env = (
        PartialObstacleCnnEnv()
    )

    check_env(env)

    observation, _ = (
        env.reset(
            seed=42
        )
    )

    print(
        "환경 검사 성공!"
    )

    print(
        "Observation shape:",
        env.observation_space.shape,
    )

    print(
        "Observation 길이:",
        len(observation),
    )

    print(
        "Start:",
        tuple(
            env.agent_pos
        ),
    )

    print(
        "Goal:",
        tuple(
            env.goal_pos
        ),
    )

    print(
        "\n=== 실제 전체 맵 ==="
    )

    env.render()

    print(
        "=== 현재까지 알고 있는 맵 ==="
    )

    env.render_memory()

    goal_score_map = (
        env._get_goal_score_map()
        * 100
    )

    print(
        "=== Goal Score Map ==="
    )

    print(
        goal_score_map.astype(int)
    )