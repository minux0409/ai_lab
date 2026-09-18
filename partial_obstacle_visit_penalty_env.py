import numpy as np
from partial_obstacle_compact_memory_env import PartialObstacleCompactMemoryEnv


class PartialObstacleVisitPenaltyEnv(PartialObstacleCompactMemoryEnv):
    """
    Experiment J

    Experiment I와 observation / map generation / transition은 그대로 유지하고,
    reward에 '도착 칸의 기존 방문 횟수'만큼의 추가 penalty를 더한다.

    추가 reward:
        revisit_penalty = -previous_visit_count_of_destination

    예:
        처음 방문: 기존 0회 ->  0
        두 번째:   기존 1회 -> -1
        세 번째:   기존 2회 -> -2

    충돌/경계로 이동하지 못한 경우에는 '도착한 칸'이 없으므로
    이 revisit penalty는 추가하지 않는다.
    기존 collision penalty는 부모 환경 그대로 적용된다.
    """

    def step(self, action):
        # 부모 step 전에 현재 위치와 방문횟수를 보관한다.
        before_pos = tuple(int(v) for v in self.agent_pos)

        obs, reward, terminated, truncated, info = super().step(action)

        after_pos = tuple(int(v) for v in self.agent_pos)

        revisit_penalty = 0.0

        # 실제로 다른 칸으로 이동한 경우에만 적용.
        if after_pos != before_pos:
            # 부모 step에서 목적지 방문횟수가 이미 +1 된 상태이므로
            # 기존 방문횟수 = 현재 방문횟수 - 1
            x, y = after_pos
            current_visit_count = int(self.visit_count_map[y, x])
            previous_visit_count = max(0, current_visit_count - 1)

            revisit_penalty = -float(previous_visit_count)
            reward += revisit_penalty
        else:
            previous_visit_count = None

        # 분석용 정보. 기존 info를 깨지 않고 추가만 한다.
        info = dict(info)
        info["revisit_penalty"] = revisit_penalty
        info["destination_previous_visit_count"] = previous_visit_count

        return obs, reward, terminated, truncated, info
