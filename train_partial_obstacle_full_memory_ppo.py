import os

from stable_baselines3 import PPO

from partial_obstacle_full_memory_env import (
    PartialObstacleFullMemoryEnv,
)


TOTAL_TIMESTEPS = 200_000

CHECKPOINTS = [
    0,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
]

EVAL_EPISODES = 500

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "partial_obstacle_full_memory_ppo",
)


def evaluate_random():
    env = (
        PartialObstacleFullMemoryEnv()
    )

    successes = 0

    success_steps = []
    failure_steps = []

    for episode in range(
        EVAL_EPISODES
    ):
        observation, _ = env.reset(
            seed=10_000 + episode
        )

        steps = 0

        while True:

            action = (
                env.action_space.sample()
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

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
        / EVAL_EPISODES
        * 100.0
    )

    avg_success_steps = (
        sum(success_steps)
        / len(success_steps)
        if success_steps
        else 0.0
    )

    avg_failure_steps = (
        sum(failure_steps)
        / len(failure_steps)
        if failure_steps
        else 0.0
    )

    return (
        success_rate,
        avg_success_steps,
        avg_failure_steps,
    )


def evaluate_model(
    model,
):
    env = (
        PartialObstacleFullMemoryEnv()
    )

    successes = 0

    success_steps = []
    failure_steps = []

    for episode in range(
        EVAL_EPISODES
    ):
        observation, _ = env.reset(
            seed=10_000 + episode
        )

        steps = 0

        while True:

            action, _ = (
                model.predict(
                    observation,
                    deterministic=True,
                )
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

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
        / EVAL_EPISODES
        * 100.0
    )

    avg_success_steps = (
        sum(success_steps)
        / len(success_steps)
        if success_steps
        else 0.0
    )

    avg_failure_steps = (
        sum(failure_steps)
        / len(failure_steps)
        if failure_steps
        else 0.0
    )

    return (
        success_rate,
        avg_success_steps,
        avg_failure_steps,
    )


def print_result(
    name,
    success_rate,
    success_steps,
    failure_steps,
):
    print(
        f"{name:<10} | "
        f"Success "
        f"{success_rate:>6.2f}% | "
        f"Success Steps "
        f"{success_steps:>6.2f} | "
        f"Failure Steps "
        f"{failure_steps:>6.2f}"
    )


def main():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    print()
    print(
        "=== Full Memory PPO Experiment ==="
    )

    print()
    print(
        "Observation"
    )

    print(
        "  Discovered 9x9 Map : 81"
    )

    print(
        "  Goal Score 9x9 Map : 81"
    )

    print(
        "  Agent Position      : 2"
    )

    print(
        "  Visit Count Map     : 81"
    )

    print(
        "  Previous Action     : 4"
    )

    print(
        "  TOTAL               : 249"
    )

    print()

    print(
        "Goal Score"
    )

    print(
        "  Goal       : 100"
    )

    print(
        "  Distance 1 : 99"
    )

    print(
        "  Distance 2 : 98"
    )

    print(
        "  ..."
    )

    print()

    print(
        "Reward"
    )

    print(
        "  Step       : -1"
    )

    print(
        "  Discovery  : +0.2 / cell"
    )

    print(
        "  Goal       : +100"
    )

    print()

    # -----------------------------------------------------
    # Random baseline
    # -----------------------------------------------------

    print(
        "Random Agent 평가 중..."
    )

    (
        random_success,
        random_success_steps,
        random_failure_steps,
    ) = evaluate_random()

    print_result(
        "Random",
        random_success,
        random_success_steps,
        random_failure_steps,
    )

    # -----------------------------------------------------
    # PPO
    # -----------------------------------------------------

    train_env = (
        PartialObstacleFullMemoryEnv()
    )

    model = PPO(
        "MlpPolicy",
        train_env,

        learning_rate=3e-4,

        n_steps=2048,
        batch_size=64,
        n_epochs=10,

        gamma=0.99,
        gae_lambda=0.95,

        clip_range=0.2,

        ent_coef=0.01,
        vf_coef=0.5,

        max_grad_norm=0.5,

        seed=42,

        verbose=0,
    )

    previous_checkpoint = 0

    results = []

    # -----------------------------------------------------
    # 0 timestep
    # -----------------------------------------------------

    (
        success_rate,
        success_steps,
        failure_steps,
    ) = evaluate_model(
        model
    )

    results.append(
        (
            0,
            success_rate,
            success_steps,
            failure_steps,
        )
    )

    print_result(
        "0",
        success_rate,
        success_steps,
        failure_steps,
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    for checkpoint in CHECKPOINTS[1:]:

        additional_steps = (
            checkpoint
            - previous_checkpoint
        )

        print()
        print(
            f"{previous_checkpoint:,}"
            f" -> "
            f"{checkpoint:,} "
            f"학습 중..."
        )

        model.learn(
            total_timesteps=(
                additional_steps
            ),
            reset_num_timesteps=False,
        )

        (
            success_rate,
            success_steps,
            failure_steps,
        ) = evaluate_model(
            model
        )

        results.append(
            (
                checkpoint,
                success_rate,
                success_steps,
                failure_steps,
            )
        )

        print_result(
            f"{checkpoint:,}",
            success_rate,
            success_steps,
            failure_steps,
        )

        previous_checkpoint = (
            checkpoint
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    model.save(
        MODEL_PATH
    )

    train_env.close()

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print()
    print(
        "======================================"
    )

    print(
        "FINAL RESULT"
    )

    print(
        "======================================"
    )

    print_result(
        "Random",
        random_success,
        random_success_steps,
        random_failure_steps,
    )

    for (
        checkpoint,
        success_rate,
        success_steps,
        failure_steps,
    ) in results:

        print_result(
            f"{checkpoint:,}",
            success_rate,
            success_steps,
            failure_steps,
        )

    print()

    print(
        "Model saved:"
    )

    print(
        MODEL_PATH
    )


if __name__ == "__main__":
    main()