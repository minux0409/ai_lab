import numpy as np
from gymnasium import spaces

from partial_obstacle_goal_tile_history_env import (
    PartialObstacleGoalTileHistoryEnv,
)


class PartialObstacleGoalTileHistoryVisitEnv(
    PartialObstacleGoalTileHistoryEnv
):
    """
    통제 실험

    기준:
        기존 Goal Tile + Short History 환경 (107)

    유일한 변경:
        Visit Count Map 81개를 Observation에 추가

    Reward / History / Map / PPO 조건은
    기존 42.6% 실험과 동일.
    """

    def __init__(self):
        super().__init__()

        self.visit_count_cap = 10.0

        # 기존 107 + Visit Count 81
        self.observation_size = 188

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(188,),
            dtype=np.float32,
        )

        self.visit_count_map = None

    def _get_base_observation(self):
        """
        부모의 기존 107차원 Observation을 그대로 사용.
        """
        return super()._get_observation()

    def _get_visit_count_observation(self):
        """
        0회   -> 0.0
        1회   -> 0.1
        ...
        10회+ -> 1.0

        Observation 정보일 뿐,
        Reward/Penalty에는 사용하지 않는다.
        """

        normalized = np.minimum(
            self.visit_count_map,
            self.visit_count_cap,
        ) / self.visit_count_cap

        return (
            normalized
            .astype(np.float32)
            .flatten()
        )

    def _get_observation(self):
        base_observation = (
            self._get_base_observation()
        )

        visit_observation = (
            self._get_visit_count_observation()
        )

        observation = np.concatenate(
            [
                base_observation,
                visit_observation,
            ]
        ).astype(np.float32)

        assert observation.shape == (188,), (
            "Observation shape 오류: "
            f"{observation.shape}"
        )

        return observation

    def reset(
        self,
        seed=None,
        options=None,
    ):
        """
        부모 reset 중에는 부모의 _get_observation()이
        동적 디스패치로 이 클래스의 메서드를 호출할 수 있으므로,
        Visit Count Map을 먼저 준비한다.
        """

        self.visit_count_map = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=np.float32,
        )

        obs, info = super().reset(
            seed=seed,
            options=options,
        )

        # 시작 위치 = 최초 1회 방문
        x = int(self.agent_pos[0])
        y = int(self.agent_pos[1])

        self.visit_count_map.fill(0.0)
        self.visit_count_map[y, x] = 1.0

        # 부모 reset에서 만들어진 obs는
        # 시작 위치 방문 횟수 반영 전일 수 있으므로 다시 생성
        obs = self._get_observation()

        info["current_visit_count"] = 1
        info["max_visit_count"] = 1

        return obs, info

    def step(self, action):
        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = super().step(action)

        # 부모 step 완료 후 현재 위치 방문 횟수 +1
        #
        # 충돌해서 제자리에 남아도 증가.
        # 따라서 같은 벽에 계속 박으면
        # 현재 칸의 Visit Count가 올라간다.
        x = int(self.agent_pos[0])
        y = int(self.agent_pos[1])

        self.visit_count_map[y, x] += 1.0

        # 부모 step에서 만들어진 Observation에는
        # 이번 Visit Count 증가분이 아직 없으므로 재생성
        obs = self._get_observation()

        info["current_visit_count"] = int(
            self.visit_count_map[y, x]
        )

        info["max_visit_count"] = int(
            np.max(self.visit_count_map)
        )

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import (
        check_env,
    )

    env = (
        PartialObstacleGoalTileHistoryVisitEnv()
    )

    check_env(env)

    obs, info = env.reset(seed=42)

    print("환경 검사 성공!")
    print(
        "Observation shape:",
        env.observation_space.shape,
    )
    print(
        "Observation length:",
        len(obs),
    )

    print()
    print("=== 통제 조건 ===")
    print("기존 Observation : 107")
    print("Visit Count Map   : +81")
    print("Total             : 188")
    print()
    print(
        "Reward는 기존 42.6% "
        "History 환경 그대로"
    )
    print(
        "Visit Count에 별도 "
        "Reward/Penalty 없음"
    )

    env.close()