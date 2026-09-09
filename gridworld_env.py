import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env


class GridWorldEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.grid_size = 5
        self.max_steps = 50

        # 행동
        # 0: 위
        # 1: 아래
        # 2: 왼쪽
        # 3: 오른쪽
        self.action_space = spaces.Discrete(4)

        # 상태:
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 처음에는 위치를 고정해서 학습이 잘 되는지부터 확인
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([4, 4])
        self.enemy_pos = np.array([2, 2])

        self.steps = 0

        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1

        if action == 0:      # 위
            self.agent_pos[1] -= 1
        elif action == 1:    # 아래
            self.agent_pos[1] += 1
        elif action == 2:    # 왼쪽
            self.agent_pos[0] -= 1
        elif action == 3:    # 오른쪽
            self.agent_pos[0] += 1

        # 맵 밖으로 못 나가게 제한
        self.agent_pos = np.clip(
            self.agent_pos,
            0,
            self.grid_size - 1,
        )

        reward = -0.1
        terminated = False

        # 적에게 닿음
        if np.array_equal(self.agent_pos, self.enemy_pos):
            reward = -10.0
            terminated = True

        # 목표 도착
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
    env = GridWorldEnv()

    # Gymnasium 규격에 맞는 환경인지 검사
    check_env(env)

    print("환경 검사 성공!")

    observation, _ = env.reset()

    print("초기 상태:")
    print(observation)

    env.render()

    # 아직 AI가 아니라 랜덤하게 움직이는 Agent
    for step in range(10):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, _ = env.step(action)

        print(
            f"Step {step + 1} | "
            f"Action: {action} | "
            f"Reward: {reward}"
        )

        env.render()

        if terminated or truncated:
            print("Episode 종료")
            break