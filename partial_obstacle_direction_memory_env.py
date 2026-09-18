import numpy as np
from gymnasium import spaces

from partial_obstacle_compact_memory_env import PartialObstacleCompactMemoryEnv


class PartialObstacleDirectionMemoryEnv(PartialObstacleCompactMemoryEnv):
    """
    Experiment K: Compact Per-Cell Direction Memory

    각 칸은 환경 내부에 [UP, DOWN, LEFT, RIGHT] 방향 점수 4개를 가진다.
    PPO에는 현재 서 있는 칸의 방향 점수 4개만 제공한다.

    초기값:
        [10, 10, 10, 10]

    행동 선택 시 (이동 성공/충돌 여부와 관계없이):
      - 선택한 방향의 현재 점수를 0으로 만든다.
      - 그 점수를 선택 방향의 좌/우 직교 방향에 절반씩 분배한다.

    예:
        [10, 10, 10, 10] 에서 UP
        -> [0, 10, 15, 15]

    Observation 21D:
      0..8   local 3x3 obstacle/boundary
      9..10  relative goal dx/dy
      11..14 previous action one-hot
      15     previous move succeeded
      16     previous collision
      17..20 current-cell direction scores

    Experiment I의 current tile visit count는 observation에서 제거한다.
    Reward는 부모(Experiment I) 그대로 사용한다.
    """

    INITIAL_DIRECTION_SCORE = 10.0

    # 한 칸의 총 방향 점수는 항상 40으로 보존된다.
    # 40으로 나누면 각 방향 입력은 0~1 범위가 된다.
    SCORE_SCALE = 40.0

    # action:
    # 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
    PERPENDICULAR = {
        0: (2, 3),  # UP    -> LEFT, RIGHT
        1: (2, 3),  # DOWN  -> LEFT, RIGHT
        2: (0, 1),  # LEFT  -> UP, DOWN
        3: (0, 1),  # RIGHT -> UP, DOWN
    }

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(21,),
            dtype=np.float32,
        )

        self.direction_scores = None

    def reset(self, seed=None, options=None):
        # 부모 환경의 실제 크기 변수는 grid_size.
        # 부모 reset() 내부에서 self._get_observation()을 호출하므로
        # direction memory를 먼저 준비한다.
        self.direction_scores = np.full(
            (self.grid_size, self.grid_size, 4),
            self.INITIAL_DIRECTION_SCORE,
            dtype=np.float32,
        )

        observation, info = super().reset(
            seed=seed,
            options=options,
        )

        return observation, info

    def _get_observation(self):
        """
        부모의 compact observation을 그대로 만들되,
        마지막 current visit count 1개를 제거하고
        현재 칸의 direction score 4개를 붙인다.
        """
        base = super()._get_observation()

        # 부모 observation:
        # 0..8   local
        # 9..10  goal relative
        # 11..14 previous action
        # 15     previous moved
        # 16     previous collision
        # 17     current visit count
        compact_without_visit_count = base[:17]

        if self.direction_scores is None:
            current_direction_scores = np.full(
                4,
                self.INITIAL_DIRECTION_SCORE,
                dtype=np.float32,
            )
        else:
            x = int(self.agent_pos[0])
            y = int(self.agent_pos[1])

            current_direction_scores = (
                self.direction_scores[y, x].astype(np.float32)
            )

        normalized_direction_scores = (
            current_direction_scores / self.SCORE_SCALE
        )

        return np.concatenate(
            [
                compact_without_visit_count,
                normalized_direction_scores,
            ]
        ).astype(np.float32)

    def _redistribute_direction_score(self, x, y, action):
        """
        현재 칸에서 선택한 방향의 점수를 0으로 만들고
        직교 방향 2개에 절반씩 분배한다.

        총점은 항상 40으로 보존된다.
        """
        action = int(action)

        selected_score = float(
            self.direction_scores[y, x, action]
        )

        if selected_score <= 0.0:
            return

        side_a, side_b = self.PERPENDICULAR[action]

        self.direction_scores[y, x, action] = 0.0
        self.direction_scores[y, x, side_a] += (
            selected_score / 2.0
        )
        self.direction_scores[y, x, side_b] += (
            selected_score / 2.0
        )

    def step(self, action):
        action = int(action)

        # 이동하기 전, 현재 칸에 '이 방향을 시도했다'는 메모리를 남긴다.
        # 벽/경계에 충돌해도 방향 메모리는 변화한다.
        x = int(self.agent_pos[0])
        y = int(self.agent_pos[1])

        self._redistribute_direction_score(
            x,
            y,
            action,
        )

        # 이동 / reward / discovery / collision / first visit 등은
        # Experiment I 부모 환경을 그대로 사용한다.
        observation, reward, terminated, truncated, info = (
            super().step(action)
        )

        info = dict(info)

        current_x = int(self.agent_pos[0])
        current_y = int(self.agent_pos[1])

        info["current_direction_scores"] = (
            self.direction_scores[
                current_y,
                current_x,
            ].copy()
        )

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    env = PartialObstacleDirectionMemoryEnv()
    check_env(env)

    observation, _ = env.reset(seed=42)

    print("환경 검사 성공!")
    print("Observation shape:", env.observation_space.shape)
    print("Observation 길이:", len(observation))
    print("Start:", tuple(env.agent_pos))
    print("Goal:", tuple(env.goal_pos))

    x = int(env.agent_pos[0])
    y = int(env.agent_pos[1])

    print(
        "현재 칸 Direction Scores:",
        env.direction_scores[y, x],
    )
    print(
        "PPO 입력 Direction Scores:",
        observation[17:21],
    )
