import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env


class StrategicGridWorldEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.grid_size = 7
        self.max_steps = 80

        # 0: 위, 1: 아래, 2: 왼쪽, 3: 오른쪽
        self.action_space = spaces.Discrete(4)

        # 상태:
        # [agent_x, agent_y, goal_x, goal_y]
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size - 1,
            shape=(4,),
            dtype=np.float32,
        )

        self.agent_pos = None
        self.goal_pos = None

        # 장애물: 이동 불가
        self.obstacles = {
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 4),
            (2, 5),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 5),
        }

        # 위험 지역: 지나갈 수 있지만 패널티
        self.danger_zones = {
            (1, 3),
            (2, 3),
            (3, 3),
            (4, 4),
            (5, 4),
        }

        self.steps = 0

    def _get_observation(self):
        return np.array(
            [
                self.agent_pos[0],
                self.agent_pos[1],
                self.goal_pos[0],
                self.goal_pos[1],
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([6, 6])
        self.steps = 0

        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1

        old_pos = self.agent_pos.copy()
        new_pos = self.agent_pos.copy()

        if action == 0:
            new_pos[1] -= 1
        elif action == 1:
            new_pos[1] += 1
        elif action == 2:
            new_pos[0] -= 1
        elif action == 3:
            new_pos[0] += 1

        # 맵 밖으로 못 나감
        new_pos = np.clip(
            new_pos,
            0,
            self.grid_size - 1,
        )

        # 장애물이면 이동 실패
        if tuple(new_pos) in self.obstacles:
            new_pos = old_pos

        self.agent_pos = new_pos

        reward = -0.1
        terminated = False

        # 장애물에 막혀 제자리면 추가 패널티
        if np.array_equal(self.agent_pos, old_pos):
            reward -= 0.2

        # 위험 지역
        if tuple(self.agent_pos) in self.danger_zones:
            reward -= 1.0

        # Goal 도착
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 20.0
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
        grid = [
            ["." for _ in range(self.grid_size)]
            for _ in range(self.grid_size)
        ]

        for x, y in self.obstacles:
            grid[y][x] = "#"

        for x, y in self.danger_zones:
            grid[y][x] = "!"

        gx, gy = self.goal_pos
        ax, ay = self.agent_pos

        grid[gy][gx] = "G"
        grid[ay][ax] = "A"

        print()

        for row in grid:
            print(" ".join(row))

        print()


if __name__ == "__main__":
    env = StrategicGridWorldEnv()

    check_env(env)

    print("환경 검사 성공!")

    observation, _ = env.reset()

    print("초기 상태:", observation)
    env.render()