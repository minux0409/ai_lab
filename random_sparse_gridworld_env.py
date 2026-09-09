import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env


class RandomSparseGridWorldEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.grid_size = 10
        self.max_steps = 50

        # 0: 위, 1: 아래, 2: 왼쪽, 3: 오른쪽
        self.action_space = spaces.Discrete(4)

        # [agent_x, agent_y, goal_x, goal_y]
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size - 1,
            shape=(4,),
            dtype=np.float32,
        )

        self.agent_pos = None
        self.goal_pos = None
        self.steps = 0

    def _random_position(self):
        return np.array(
            [
                self.np_random.integers(0, self.grid_size),
                self.np_random.integers(0, self.grid_size),
            ]
        )

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

        self.steps = 0

        while True:
            self.agent_pos = self._random_position()
            self.goal_pos = self._random_position()

            if not np.array_equal(
                self.agent_pos,
                self.goal_pos,
            ):
                break

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

        new_pos = np.clip(
            new_pos,
            0,
            self.grid_size - 1,
        )

        self.agent_pos = new_pos

        # Sparse Reward:
        # Goal에 도착하기 전에는 아무 보상도 없다.
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
        grid = [
            ["." for _ in range(self.grid_size)]
            for _ in range(self.grid_size)
        ]

        ax, ay = self.agent_pos
        gx, gy = self.goal_pos

        grid[gy][gx] = "G"
        grid[ay][ax] = "A"

        print()

        for row in grid:
            print(" ".join(row))

        print()


if __name__ == "__main__":
    env = RandomSparseGridWorldEnv()

    check_env(env)

    print("환경 검사 성공!")

    # 위치가 실제로 랜덤화되는지 확인
    for episode in range(3):
        obs, _ = env.reset()

        print(f"\nEpisode {episode + 1}")
        print("Observation:", obs)
        env.render()