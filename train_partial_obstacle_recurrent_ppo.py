import numpy as np

from sb3_contrib import RecurrentPPO

from partial_obstacle_env import PartialObstacleGridEnv


CHECKPOINTS = [
    0,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
]


def evaluate(model, episodes=500):
    env = PartialObstacleGridEnv()

    success = 0
    total_steps = 0

    for episode in range(episodes):
        # 기존 실험과 동일한 500개 평가 맵
        obs, _ = env.reset(seed=10_000 + episode)

        steps = 0

        # LSTM의 기억 상태
        lstm_states = None

        # 새로운 Episode가 시작됐음을 LSTM에 알려준다.
        episode_starts = np.ones(
            (1,),
            dtype=bool,
        )

        while True:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )

            obs, reward, terminated, truncated, _ = env.step(
                int(action.item())
            )

            steps += 1

            # 첫 행동 이후에는 같은 Episode
            episode_starts = np.zeros(
                (1,),
                dtype=bool,
            )

            if terminated or truncated:
                if terminated:
                    success += 1

                total_steps += steps
                break

    env.close()

    return {
        "success_rate": success / episodes * 100,
        "avg_steps": total_steps / episodes,
    }


if __name__ == "__main__":
    train_env = PartialObstacleGridEnv()

    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        verbose=0,
        learning_rate=0.0003,
        seed=42,
    )

    previous_checkpoint = 0
    results = []

    for checkpoint in CHECKPOINTS:
        additional_steps = checkpoint - previous_checkpoint

        if additional_steps > 0:
            print(
                f"\n{previous_checkpoint:,} → "
                f"{checkpoint:,} timestep LSTM 학습..."
            )

            model.learn(
                total_timesteps=additional_steps,
                reset_num_timesteps=False,
            )

        result = evaluate(
            model,
            episodes=500,
        )

        results.append(
            {
                "timesteps": checkpoint,
                **result,
            }
        )

        print(
            f"{checkpoint:,} timestep | "
            f"Success {result['success_rate']:.1f}% | "
            f"Steps {result['avg_steps']:.2f}"
        )

        previous_checkpoint = checkpoint

    model.save(
        "models/partial_obstacle_recurrent_ppo"
    )

    train_env.close()

    print("\n=== 전체 학습 결과 ===")

    for result in results:
        print(
            f"{result['timesteps']:>7,} | "
            f"Success {result['success_rate']:>5.1f}% | "
            f"Steps {result['avg_steps']:>6.2f}"
        )

    print("\n=== 기존 모델과 비교 ===")
    print(
        "Baseline MLP 200k : "
        "Success 84.4% | Steps 20.65"
    )
    print(
        "Revisit Reward 200k: "
        "Success 81.4% | Steps 23.43"
    )
    print(
        "History Obs 200k   : "
        "Success 90.2% | Steps 15.60"
    )

    print(
        "Recurrent LSTM 200k: "
        f"Success {results[-1]['success_rate']:.1f}% | "
        f"Steps {results[-1]['avg_steps']:.2f}"
    )