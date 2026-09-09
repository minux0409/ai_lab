import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env


class SparseGridWorldEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # 기존 5x5보다 크게 만들어
        # 우연히 Goal을 발견하기 어렵게 한다.
        self.grid_size = 10
        self.max_steps = 50

        # 0: 위
        # 1: 아래
        # 2: 왼쪽
        # 3: 오른쪽
        self.action_space = spaces.Discrete(4)

        # [agent_x, agent_y, goal_x, goal_y]
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size - 1,
            shape=(4,),
            dtype=np.float32,
        )

        self.start_pos = np.array([0, 0])
        self.goal_pos = np.array([9, 9])

        self.agent_pos = None
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

        self.agent_pos = self.start_pos.copy()
        self.steps = 0

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

        # 핵심:
        # Goal 전까지는 아무 보상도 주지 않는다.
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
    env = SparseGridWorldEnv()

    check_env(env)

    print("Sparse Reward 환경 검사 성공!")

    observation, _ = env.reset()

    print("초기 Observation:", observation)

    env.render()