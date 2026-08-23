"""
Replay an online training run offline.

Re-trains the rewarder along the *same trajectory* as a live online run, using the
labeled data that was saved per-step under ``out/<profile>/<config>/online_steps/``.

For each online step N, the live loop fuses the labeled pairs of steps 1..N and calls
``rewarder.train(...)`` on a single, reused rewarder object (warm-start). This script
reproduces exactly that: it builds one fresh rewarder and trains it successively on the
growing fused dataset (1, then 1+2, then 1+2+3, ...). It does NOT re-train the generator
or generate new simulations (those are stochastic and produce unlabeled samples) - it
replays the *rewarder* trajectory only.

Test-set evaluation/logging is the same as a normal run: it happens inside
``rewarder.train()`` every epoch, provided ``test_set_path`` is set in the config and a
simulator is available. Use ``--test-set-path`` to enable it.

Run from the repository root (so that 'profiles/' and 'out/' resolve):

Usage examples:
    uv run python scripts/replay_online_training.py -p lenia -c base
    uv run python scripts/replay_online_training.py -p lenia -c base --steps 3
    uv run python scripts/replay_online_training.py -p lenia -c base --test-set-path out/lenia/benchmark --wandb
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

# Allow importing the repo-root modules (main.py, rlhfalife) when run from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the exact same machinery as main.py / the live online loop.
from main import load_profile_module, setup_paths
from rlhfalife.data_managers import DatasetManager, PairsManager, TrainingDataset
from rlhfalife.online_training import OnlineTrainingController


def parse_args():
    parser = argparse.ArgumentParser(description="Replay an online training run offline (rewarder trajectory).")
    parser.add_argument("--profile", "-p", type=str, required=True, help="Profile name (e.g. lenia).")
    parser.add_argument("--config", "-c", type=str, required=True, help="Config name (e.g. base).")
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Number of online steps to replay (e.g. 3 to replay only the first 3). "
             "Default: replay all existing steps.",
    )
    parser.add_argument(
        "--test-set-path", type=str, default=None,
        help="Directory containing benchmark.csv (and its outputs) to evaluate/log the test "
             "set each epoch, exactly like a normal run. E.g. 'out/lenia/benchmark'. "
             "If omitted, test-set evaluation is skipped.",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Log to wandb in a single fresh run across all replayed steps "
             "(distinguished by other/online_step), mirroring a live run. "
             "Default: off (console logging only).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override the device from the config (e.g. 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not save a per-step rewarder checkpoint.",
    )
    return parser.parse_args()


def build_fresh_rewarder(loader, out_paths, config_dict, replay_model_path):
    """
    Build a *fresh* (randomly-initialized) rewarder and the simulator, mirroring the
    profile Loader but skipping ``rewarder.load()`` so the replay starts from scratch
    like the very first step of the real online run.

    We call ``loader.load()`` to obtain a correctly-built simulator, then discard its
    rewarder (which may have loaded the final trained checkpoint) and instantiate a new
    one of the same class pointed at a replay-only model_path.
    """
    print("Loading profile (simulator + reference models)...")
    _generator, loaded_rewarder, simulator = loader.load(out_paths, config_dict)

    rewarder_cls = type(loaded_rewarder)
    wandb_params = getattr(loaded_rewarder, "wandb_params", None)

    # Free the discarded rewarder (and its CLIP-VIP weights) before building the fresh one.
    del loaded_rewarder, _generator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Building a fresh rewarder for the replay...")
    rewarder = rewarder_cls(
        config=config_dict,
        model_path=str(replay_model_path),
        device=config_dict["device"],
        simulator=simulator,
        wandb_params=wandb_params,
    )
    return rewarder, simulator


def prefix_fuse(controller, steps, fused_dir):
    """Fuse the per-step datasets/pairs for the given list of steps (a 1..N prefix)."""
    dataset_managers = []
    pairs_managers = []
    for step in steps:
        dataset_manager, pairs_manager = controller.load_step(step)
        dataset_managers.append(dataset_manager)
        pairs_managers.append(pairs_manager)

    fused_dataset_path = fused_dir / "replay_fused_dataset.csv"
    fused_pairs_path = fused_dir / "replay_fused_pairs.csv"
    fused_dataset = DatasetManager.fuse(dataset_managers, fused_dataset_path, controller.out_paths, controller.simulator)
    fused_pairs = PairsManager.fuse(pairs_managers, fused_pairs_path)
    return fused_dataset, fused_pairs


def main():
    args = parse_args()

    # --- Mirror main.py setup ------------------------------------------------
    profile_module = load_profile_module(args.profile)
    if profile_module is None:
        sys.exit(1)

    config_file_path = Path("profiles") / args.profile / "configs" / f"{args.config}.json"
    with open(config_file_path) as f:
        config_dict = json.load(f)

    if args.device:
        config_dict["device"] = args.device
    if args.test_set_path:
        config_dict["test_set_path"] = args.test_set_path

    out_path, out_paths = setup_paths(args.profile, args.config)

    # Replay checkpoints go to a dedicated folder so we never clobber the real model.
    replay_dir = Path(out_paths["rewarder"]) / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    replay_model_path = replay_dir / f"{config_dict.get('name', 'rewarder')}.pt"

    loader = profile_module.Loader()
    rewarder, simulator = build_fresh_rewarder(loader, out_paths, config_dict, replay_model_path)

    # --- Optional: single wandb run across all replayed steps ----------------
    if args.wandb and getattr(rewarder, "wandb_params", None) is not None:
        import wandb
        run_id = wandb.util.generate_id()
        rewarder.wandb_params = dict(rewarder.wandb_params)
        rewarder.wandb_params["id"] = run_id
        rewarder.wandb_params["resume"] = "allow"
        rewarder.wandb_params["name"] = f"{rewarder.wandb_params.get('name', args.config)}-replay"
        print(f"wandb logging enabled (run id: {run_id}).")
    else:
        # Make sure no stray wandb run is created if not requested.
        rewarder.wandb_params = None

    # --- Determine which steps to replay -------------------------------------
    controller = OnlineTrainingController(out_path, out_paths, simulator)
    all_steps = controller.existing_steps()
    if not all_steps:
        print(f"No online steps found under {controller.steps_root}. Nothing to replay.")
        sys.exit(1)

    n_to_replay = args.steps if args.steps is not None else len(all_steps)
    if n_to_replay < 1:
        print("--steps must be >= 1.")
        sys.exit(1)
    if n_to_replay > len(all_steps):
        print(f"Requested {n_to_replay} steps but only {len(all_steps)} exist; replaying all {len(all_steps)}.")
        n_to_replay = len(all_steps)

    print(f"\nFound online steps: {all_steps}. Replaying the first {n_to_replay} as phases.\n")

    # --- Replay loop: mirror launch_next_step()'s training phase per step ----
    for step in all_steps[:n_to_replay]:
        prefix = all_steps[:all_steps.index(step) + 1]
        print(f"\n{'='*70}\n[replay] Phase {step}: training rewarder on fused steps {prefix}\n{'='*70}")

        fused_dataset, fused_pairs = prefix_fuse(controller, prefix, controller.steps_root)
        training_dataset = TrainingDataset(fused_pairs, fused_dataset)
        if len(training_dataset) == 0:
            print(f"[replay] Phase {step}: no ranked pairs, skipping.")
            del training_dataset, fused_dataset, fused_pairs
            gc.collect()
            continue

        # Same field the live loop sets so logs carry the online step.
        rewarder.current_online_step = step

        # Warm-start: the SAME rewarder object is reused across phases (weights carry over),
        # while train() resets optimizer/scheduler each call - exactly like the live run.
        rewarder.train(training_dataset)

        if not args.no_save:
            step_ckpt = replay_dir / f"{config_dict.get('name', 'rewarder')}_step{step}.pt"
            rewarder.model_path = str(step_ckpt)
            rewarder.save()
            rewarder.model_path = str(replay_model_path)

        del training_dataset, fused_dataset, fused_pairs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[replay] Done. Per-step checkpoints saved under:", replay_dir)


if __name__ == "__main__":
    main()
