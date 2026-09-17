from partial_obstacle_goal_tile_history_env import (
    PartialObstacleGoalTileHistoryEnv,
)


class PartialObstacleGoal1000HistoryEnv(
    PartialObstacleGoalTileHistoryEnv
):
    """
    통제 실험:

    Observation:
        기존 Goal Tile + History 환경과 완전히 동일 = 107

    변경되는 것:
        Goal Reward
        Tile Reward 크기

    추가하지 않는 것:
        Visit Count Map
        Visit Penalty
        Action Mask
        Pathfinding
    """

    def __init__(self):
        super().__init__()

        # Goal 성공 보상
        self.goal_reward = 1000.0

        # 기존 Env의 _get_tile_reward()가
        # max(1, tile_reward_max - ManhattanDistance)
        # 구조이므로 16으로 설정.
        #
        # distance 1  -> 15
        # distance 2  -> 14
        # ...
        # distance 15 -> 1
        # distance 16 -> 1
        self.tile_reward_max = 16.0


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import (
        check_env,
    )

    env = PartialObstacleGoal1000HistoryEnv()

    check_env(env)

    obs, _ = env.reset(seed=42)

    print("환경 검사 성공!")
    print(
        "Observation shape:",
        env.observation_space.shape,
    )

    print()
    print("=== 이번 통제 실험 ===")
    print("Observation       : 107")
    print("Visit Count Map   : 없음")
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

    print()
    print(
        "Observation length:",
        len(obs),
    )

    env.close()