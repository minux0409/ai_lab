import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env


class RandomGridWorldEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.grid_size = 5
        self.max_steps = 50

        self.action_space = spaces.Discrete(4)

        # [agent_x, agent_y, goal_x, goal_y, enemy_x, enemy_y]
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size - 1,
            shape=(6,),
            dtype=np.float32,
        )

        self.agent_pos = None
        self.goal_pos = None
        self.enemy_pos = None
        self.steps = 0

    def _get_observation(self):
        return np.array(
            [
                self.agent_pos[0],
                self.agent_pos[1],
                self.goal_pos[0],
                self.goal_pos[1],
                self.enemy_pos[0],
                self.enemy_pos[1],
            ],
            dtype=np.float32,
        )

    def _random_position(self):
        return np.array(
            [
                self.np_random.integers(0, self.grid_size),
                self.np_random.integers(0, self.grid_size),
            ]
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0

        # 세 위치가 겹치지 않도록 생성
        while True:
            self.agent_pos = self._random_position()
            self.goal_pos = self._random_position()
            self.enemy_pos = self._random_position()

            if (
                not np.array_equal(self.agent_pos, self.goal_pos)
                and not np.array_equal(self.agent_pos, self.enemy_pos)
                and not np.array_equal(self.goal_pos, self.enemy_pos)
            ):
                break

        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1

        if action == 0:
            self.agent_pos[1] -= 1
        elif action == 1:
            self.agent_pos[1] += 1
        elif action == 2:
            self.agent_pos[0] -= 1
        elif action == 3:
            self.agent_pos[0] += 1

        self.agent_pos = np.clip(
            self.agent_pos,
            0,
            self.grid_size - 1,
        )

        reward = -0.1
        terminated = False

        if np.array_equal(self.agent_pos, self.enemy_pos):
            reward = -10.0
            terminated = True

        elif np.array_equal(self.agent_pos, self.goal_pos):
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
        grid = [["." for _ in range(self.grid_size)]
                for _ in range(self.grid_size)]

        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        ex, ey = self.enemy_pos

        grid[gy][gx] = "G"
        grid[ey][ex] = "X"
        grid[ay][ax] = "A"

        print()

        for row in grid:
            print(" ".join(row))

        print()


if __name__ == "__main__":
    env = RandomGridWorldEnv()

    check_env(env)

    print("랜덤 환경 검사 성공!")

    for episode in range(3):
        observation, _ = env.reset()

        print(f"\nEpisode {episode + 1}")
        print("Observation:", observation)

        env.render()