import numpy as np
from stable_baselines3 import PPO

from partial_obstacle_goal_tile_history_visit_env import (
    PartialObstacleGoalTileHistoryVisitEnv,
)


CHECKPOINTS = [
    0,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
]

EVAL_EPISODES = 500
EVAL_SEED_BASE = 10_000


def evaluate_model(
    model,
    episodes=EVAL_EPISODES,
):
    env = (
        PartialObstacleGoalTileHistoryVisitEnv()
    )

    successes = 0
    success_steps = []
    failure_steps = []

    for episode in range(episodes):
        obs, _ = env.reset(
            seed=EVAL_SEED_BASE + episode
        )

        steps = 0

        while True:
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            steps += 1

            if terminated or truncated:
                if terminated:
                    successes += 1
                    success_steps.append(
                        steps
                    )
                else:
                    failure_steps.append(
                        steps
                    )

                break

    env.close()

    success_rate = (
        successes
        / episodes
        * 100.0
    )

    avg_success_steps = (
        float(np.mean(success_steps))
        if success_steps
        else 0.0
    )

    avg_failure_steps = (
        float(np.mean(failure_steps))
        if failure_steps
        else 0.0
    )

    return (
        success_rate,
        avg_success_steps,
        avg_failure_steps,
    )


def evaluate_random(
    episodes=EVAL_EPISODES,
):
    env = (
        PartialObstacleGoalTileHistoryVisitEnv()
    )

    successes = 0
    success_steps = []
    failure_steps = []

    for episode in range(episodes):
        obs, _ = env.reset(
            seed=EVAL_SEED_BASE + episode
        )

        rng = np.random.default_rng(
            20_000 + episode
        )

        steps = 0

        while True:
            action = int(
                rng.integers(
                    0,
                    env.action_space.n,
                )
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            steps += 1

            if terminated or truncated:
                if terminated:
                    successes += 1
                    success_steps.append(
                        steps
                    )
                else:
                    failure_steps.append(
                        steps
                    )

                break

    env.close()

    success_rate = (
        successes
        / episodes
        * 100.0
    )

    avg_success_steps = (
        float(np.mean(success_steps))
        if success_steps
        else 0.0
    )

    avg_failure_steps = (
        float(np.mean(failure_steps))
        if failure_steps
        else 0.0
    )

    return (
        success_rate,
        avg_success_steps,
        avg_failure_steps,
    )


if __name__ == "__main__":
    test_env = (
        PartialObstacleGoalTileHistoryVisitEnv()
    )

    print(
        "=== History + Visit Count "
        "Controlled Experiment ==="
    )

    print(
        "Observation:",
        test_env.observation_space.shape,
    )

    print()
    print("기준 모델:")
    print(
        "  Goal Tile + History = "
        "200k 42.6%"
    )

    print()
    print("유일한 변경:")
    print(
        "  Visit Count Map +81"
    )

    print()
    print("유지:")
    print("  Goal Reward")
    print("  Tile Reward")
    print("  Collision Reward")
    print("  Revisit Reward")
    print("  Discovered Map")
    print("  Position History")
    print("  Last Action")
    print("  Last Collision")

    test_env.close()

    # -----------------------------------------
    # Random
    # -----------------------------------------
    (
        random_success,
        random_success_steps,
        random_failure_steps,
    ) = evaluate_random()

    print()
    print(
        f"Random | "
        f"Success "
        f"{random_success:.1f}% | "
        f"Success Steps "
        f"{random_success_steps:.2f} | "
        f"Failure Steps "
        f"{random_failure_steps:.2f}"
    )

    # -----------------------------------------
    # PPO
    # -----------------------------------------
    train_env = (
        PartialObstacleGoalTileHistoryVisitEnv()
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        learning_rate=0.0003,
        seed=42,
    )

    previous_checkpoint = 0
    results = []

    for checkpoint in CHECKPOINTS:
        additional_steps = (
            checkpoint
            - previous_checkpoint
        )

        if additional_steps > 0:
            print()
            print(
                f"{previous_checkpoint:,} "
                f"-> {checkpoint:,} "
                "timestep 학습..."
            )

            model.learn(
                total_timesteps=additional_steps,
                reset_num_timesteps=False,
            )

        (
            success_rate,
            success_steps,
            failure_steps,
        ) = evaluate_model(model)

        results.append(
            (
                checkpoint,
                success_rate,
                success_steps,
                failure_steps,
            )
        )

        print(
            f"{checkpoint:,} timestep | "
            f"Success "
            f"{success_rate:.1f}% | "
            f"Success Steps "
            f"{success_steps:.2f} | "
            f"Failure Steps "
            f"{failure_steps:.2f}"
        )

        previous_checkpoint = checkpoint

    MODEL_PATH = (
        "models/"
        "partial_obstacle_"
        "goal_tile_history_visit_ppo"
    )

    model.save(MODEL_PATH)
    train_env.close()

    print()
    print("=== 전체 학습 결과 ===")

    print(
        f"Random  | "
        f"Success "
        f"{random_success:>5.1f}% | "
        f"Success Steps "
        f"{random_success_steps:>6.2f} | "
        f"Failure Steps "
        f"{random_failure_steps:>6.2f}"
    )

    for (
        checkpoint,
        success_rate,
        success_steps,
        failure_steps,
    ) in results:
        print(
            f"{checkpoint:>7,} | "
            f"Success "
            f"{success_rate:>5.1f}% | "
            f"Success Steps "
            f"{success_steps:>6.2f} | "
            f"Failure Steps "
            f"{failure_steps:>6.2f}"
        )

    print()
    print(
        "모델 저장:",
        MODEL_PATH + ".zip",
    )