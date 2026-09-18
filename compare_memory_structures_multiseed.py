import csv
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from partial_obstacle_cnn_env import PartialObstacleCnnEnv
from partial_obstacle_compact_memory_env import PartialObstacleCompactMemoryEnv
from partial_obstacle_direction_memory_env import PartialObstacleDirectionMemoryEnv


# ============================================================
# Fair 5-seed structure comparison
#
# Compared structures:
#   H: Spatial Map + CNN
#   I: Compact Short-Term Memory + MLP
#   K: Per-Cell Direction Memory + MLP
#
# These three experiments use the same reward setup:
#   Step             : -1
#   Collision        : additional -2
#   First visit      : +1
#   Newly discovered : +0.1 / cell
#   Goal             : +100
#
# Training:
#   seeds 0..4
#   200,000 timesteps each
#
# Evaluation:
#   deterministic
#   fixed maps 10000..10499
# ============================================================

TRAIN_SEEDS = [0, 1, 2, 3, 4]
TOTAL_TIMESTEPS = 200_000

EVAL_EPISODES = 500
EVAL_SEED_BASE = 10_000

MODEL_DIR = "models"
RESULT_DIR = "results"

# 이미 동일 조건으로 학습된 seed 모델이 있으면 다시 학습하지 않고 재사용.
# 완전히 새로 다시 돌리고 싶으면 False.
REUSE_EXISTING_MODELS = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GridCNN(BaseFeaturesExtractor):
    """
    9x9 spatial observation용 작은 CNN.
    기존 Experiment H와 동일한 구조.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)

        channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(
                channels,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(
                observation_space.sample()[None]
            ).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor):
        return self.linear(self.cnn(observations))


@dataclass
class StructureSpec:
    key: str
    label: str
    env_class: type
    policy: str
    model_prefix: str
    policy_kwargs: dict | None = None


STRUCTURES = [
    StructureSpec(
        key="H",
        label="Spatial Map + CNN",
        env_class=PartialObstacleCnnEnv,
        policy="CnnPolicy",
        model_prefix="partial_obstacle_cnn_ppo",
        policy_kwargs={
            "features_extractor_class": GridCNN,
            "features_extractor_kwargs": {
                "features_dim": 128,
            },
            "net_arch": {
                "pi": [128, 128],
                "vf": [128, 128],
            },
        },
    ),
    StructureSpec(
        key="I",
        label="Compact Short-Term Memory",
        env_class=PartialObstacleCompactMemoryEnv,
        policy="MlpPolicy",
        model_prefix="partial_obstacle_compact_memory_ppo",
    ),
    StructureSpec(
        key="K",
        label="Per-Cell Direction Memory",
        env_class=PartialObstacleDirectionMemoryEnv,
        policy="MlpPolicy",
        model_prefix="partial_obstacle_direction_memory_ppo",
    ),
]


def model_path(spec: StructureSpec, seed: int):
    return os.path.join(
        MODEL_DIR,
        f"{spec.model_prefix}_seed{seed}",
    )


def model_exists(path: str):
    return os.path.exists(path) or os.path.exists(path + ".zip")


def create_model(spec: StructureSpec, env, seed: int):
    return PPO(
        spec.policy,
        env,
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
        policy_kwargs=spec.policy_kwargs,
        seed=seed,
        verbose=0,
    )


def train_or_load(spec: StructureSpec, seed: int):
    set_seed(seed)

    path = model_path(spec, seed)
    env = spec.env_class()

    if REUSE_EXISTING_MODELS and model_exists(path):
        print(f"Loading existing model: {path}")
        model = PPO.load(path, env=env)
        return model, env, path, False

    model = create_model(spec, env, seed)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
    )

    model.save(path)

    return model, env, path, True


def evaluate(
    model,
    env_class,
    train_seed: int,
    structure_key: str,
):
    episode_rows = []

    successes = 0
    success_steps = []
    failure_steps = []

    for episode in range(EVAL_EPISODES):
        eval_seed = EVAL_SEED_BASE + episode

        env = env_class()
        obs, _ = env.reset(seed=eval_seed)

        success = False
        steps = 0

        for step in range(1, env.max_steps + 1):
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, _, terminated, truncated, _ = env.step(
                int(action)
            )

            steps = step

            if terminated:
                success = True
                break

            if truncated:
                break

        env.close()

        if success:
            successes += 1
            success_steps.append(steps)
        else:
            failure_steps.append(steps)

        episode_rows.append({
            "structure": structure_key,
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "success": 1 if success else 0,
            "steps": steps,
        })

    return {
        "successes": successes,
        "success_rate": successes / EVAL_EPISODES * 100.0,
        "success_steps": (
            float(np.mean(success_steps))
            if success_steps
            else float("nan")
        ),
        "failure_steps": (
            float(np.mean(failure_steps))
            if failure_steps
            else float("nan")
        ),
        "episode_rows": episode_rows,
    }


def save_episode_csv(rows):
    path = os.path.join(
        RESULT_DIR,
        "structure_comparison_5seed_episode_results.csv",
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "structure",
                "train_seed",
                "eval_seed",
                "success",
                "steps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def save_summary_csv(rows):
    path = os.path.join(
        RESULT_DIR,
        "structure_comparison_5seed_summary.csv",
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "structure",
                "label",
                "success_mean",
                "success_std",
                "success_min",
                "success_max",
                "success_steps_mean",
                "success_steps_std",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def save_map_comparison_csv(all_episode_rows):
    """
    같은 eval_seed에서 각 구조가 얼마나 자주 성공했는지 비교.

    5개의 train seed 중:
      5 = 모든 모델 성공
      0 = 모든 모델 실패
    """

    lookup = {}

    for row in all_episode_rows:
        key = (
            row["structure"],
            row["eval_seed"],
        )

        if key not in lookup:
            lookup[key] = []

        lookup[key].append(row["success"])

    path = os.path.join(
        RESULT_DIR,
        "structure_comparison_fixed_map_matrix.csv",
    )

    fieldnames = ["eval_seed"]

    for spec in STRUCTURES:
        fieldnames += [
            f"{spec.key}_success_count",
            f"{spec.key}_success_rate",
        ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for eval_seed in range(
            EVAL_SEED_BASE,
            EVAL_SEED_BASE + EVAL_EPISODES,
        ):
            row = {
                "eval_seed": eval_seed,
            }

            for spec in STRUCTURES:
                values = lookup.get(
                    (spec.key, eval_seed),
                    [],
                )

                count = int(sum(values))

                row[f"{spec.key}_success_count"] = count
                row[f"{spec.key}_success_rate"] = (
                    count / len(values) * 100.0
                    if values
                    else float("nan")
                )

            writer.writerow(row)

    return path


def print_summary(
    per_structure_results,
):
    print()
    print("=" * 112)
    print("STRUCTURE COMPARISON - 5 TRAINING SEEDS")
    print("=" * 112)

    print(
        f"{'Key':<4}"
        f"{'Structure':<32}"
        f"{'Success mean±std':<24}"
        f"{'Range':<20}"
        f"{'Success Steps mean±std':<28}"
    )

    print("-" * 112)

    summary_rows = []

    for spec in STRUCTURES:
        seed_results = per_structure_results[spec.key]

        rates = np.array(
            [
                r["success_rate"]
                for r in seed_results
            ],
            dtype=np.float64,
        )

        steps = np.array(
            [
                r["success_steps"]
                for r in seed_results
            ],
            dtype=np.float64,
        )

        rate_mean = rates.mean()
        rate_std = rates.std(ddof=0)
        rate_min = rates.min()
        rate_max = rates.max()

        step_mean = steps.mean()
        step_std = steps.std(ddof=0)

        print(
            f"{spec.key:<4}"
            f"{spec.label:<32}"
            f"{rate_mean:6.2f}% ± {rate_std:5.2f}"
            f"{'':<7}"
            f"{rate_min:6.2f}~{rate_max:6.2f}%"
            f"{'':<7}"
            f"{step_mean:6.2f} ± {step_std:5.2f}"
        )

        summary_rows.append({
            "structure": spec.key,
            "label": spec.label,
            "success_mean": f"{rate_mean:.4f}",
            "success_std": f"{rate_std:.4f}",
            "success_min": f"{rate_min:.4f}",
            "success_max": f"{rate_max:.4f}",
            "success_steps_mean": f"{step_mean:.4f}",
            "success_steps_std": f"{step_std:.4f}",
        })

    print("=" * 112)

    return summary_rows


def print_seed_detail(
    per_structure_results,
):
    print()
    print("=" * 90)
    print("SEED DETAIL")
    print("=" * 90)

    for spec in STRUCTURES:
        print()
        print(
            f"[{spec.key}] {spec.label}"
        )

        for result in per_structure_results[spec.key]:
            print(
                f"  Seed {result['train_seed']} | "
                f"Success {result['success_rate']:6.2f}% "
                f"({result['successes']}/{EVAL_EPISODES}) | "
                f"Success Steps {result['success_steps']:6.2f} | "
                f"Failure Steps {result['failure_steps']:6.2f}"
            )


def main():
    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )
    os.makedirs(
        RESULT_DIR,
        exist_ok=True,
    )

    print(
        "=== Fair Structure Comparison: "
        "H vs I vs K ==="
    )

    print(
        f"Training seeds      : "
        f"{TRAIN_SEEDS}"
    )

    print(
        f"Training timesteps  : "
        f"{TOTAL_TIMESTEPS:,} / seed"
    )

    print(
        f"Evaluation          : "
        f"deterministic, "
        f"{EVAL_EPISODES} fixed maps"
    )

    print(
        f"Evaluation seeds    : "
        f"{EVAL_SEED_BASE}.."
        f"{EVAL_SEED_BASE + EVAL_EPISODES - 1}"
    )

    print(
        f"Reuse trained models: "
        f"{REUSE_EXISTING_MODELS}"
    )

    print()

    per_structure_results = {
        spec.key: []
        for spec in STRUCTURES
    }

    all_episode_rows = []

    total_jobs = (
        len(STRUCTURES)
        * len(TRAIN_SEEDS)
    )

    job = 0

    for spec in STRUCTURES:
        print()
        print("#" * 90)
        print(
            f"[{spec.key}] "
            f"{spec.label}"
        )
        print("#" * 90)

        for seed in TRAIN_SEEDS:
            job += 1

            print()
            print("=" * 90)
            print(
                f"[{job}/{total_jobs}] "
                f"{spec.key} | "
                f"Training seed {seed}"
            )
            print("=" * 90)

            model, train_env, path, trained = (
                train_or_load(
                    spec,
                    seed,
                )
            )

            result = evaluate(
                model=model,
                env_class=spec.env_class,
                train_seed=seed,
                structure_key=spec.key,
            )

            result["train_seed"] = seed
            result["model_path"] = path
            result["trained_now"] = trained

            per_structure_results[
                spec.key
            ].append(
                result
            )

            all_episode_rows.extend(
                result["episode_rows"]
            )

            print(
                f"{spec.key} Seed {seed} | "
                f"Success "
                f"{result['success_rate']:.2f}% "
                f"({result['successes']}/{EVAL_EPISODES}) | "
                f"Success Steps "
                f"{result['success_steps']:.2f} | "
                f"Failure Steps "
                f"{result['failure_steps']:.2f}"
            )

            print(
                "Model:",
                path,
                "(trained)"
                if trained
                else "(reused)",
            )

            train_env.close()

            # GPU 메모리 정리
            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print_seed_detail(
        per_structure_results
    )

    summary_rows = print_summary(
        per_structure_results
    )

    episode_csv = save_episode_csv(
        all_episode_rows
    )

    summary_csv = save_summary_csv(
        summary_rows
    )

    map_csv = save_map_comparison_csv(
        all_episode_rows
    )

    print()
    print("Saved:")
    print(" ", episode_csv)
    print(" ", summary_csv)
    print(" ", map_csv)

    print()
    print(
        "Finished all "
        f"{total_jobs} training/evaluation jobs."
    )


if __name__ == "__main__":
    main()
