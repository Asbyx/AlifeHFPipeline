import traceback
import os
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import sv_ttk

from .quad_labeler import QuadLabelerApp
from .data_managers import DatasetManager, PairsManager, TrainingDataset
from .utils import Generator, Rewarder, Simulator


class OnlineTrainingController:
    def __init__(self, out_path: Path, out_paths: dict, simulator: Simulator) -> None:
        self.out_path = Path(out_path)
        self.out_paths = out_paths
        self.simulator = simulator
        self.steps_root = self.out_path / "online_steps"
        self.steps_root.mkdir(parents=True, exist_ok=True)
        self.config_path = self.steps_root / "config.json"
        self.initial_num_sims = self._load_config().get("initial_num_sims")

    def _step_dir(self, step: int) -> Path:
        return self.steps_root / f"step_{step}"

    def _step_paths(self, step: int) -> Tuple[Path, Path, dict]:
        step_dir = self._step_dir(step)
        step_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = step_dir / "dataset.csv"
        pairs_path = step_dir / "pairs.csv"

        return dataset_path, pairs_path, self.out_paths

    def existing_steps(self) -> List[int]:
        steps = []
        for path in self.steps_root.iterdir():
            if path.is_dir() and path.name.startswith("step_"):
                try:
                    steps.append(int(path.name.split("_")[1]))
                except (ValueError, IndexError):
                    continue
        return sorted(steps)

    def ensure_initial_step(self, prompt_fn: Callable[[], int]) -> Tuple[int, DatasetManager, PairsManager, bool]:
        steps = self.existing_steps()
        if steps:
            last_step = steps[-1]
            # Ensure we have a stored initial number even if config was missing
            if self.initial_num_sims is None:
                self.initial_num_sims = 8
                self._save_config({"initial_num_sims": self.initial_num_sims})
            return last_step, *self.load_step(last_step), False

        step_number = 1
        dataset_path, pairs_path, step_out_paths = self._step_paths(step_number)
        dataset_manager = DatasetManager(dataset_path, step_out_paths, self.simulator)
        pairs_manager = PairsManager(pairs_path)

        if self.initial_num_sims is None:
            self.initial_num_sims = prompt_fn()
            if not self.initial_num_sims or self.initial_num_sims <= 0:
                self.initial_num_sims = 8
            self._save_config({"initial_num_sims": self.initial_num_sims})

        needs_generation = False
        if self.initial_num_sims and self.initial_num_sims > 0:
            needs_generation = True

        return step_number, dataset_manager, pairs_manager, needs_generation

    def load_step(self, step: int) -> Tuple[DatasetManager, PairsManager]:
        dataset_path, pairs_path, step_out_paths = self._step_paths(step)
        dataset_manager = DatasetManager(dataset_path, step_out_paths, self.simulator)
        pairs_manager = PairsManager(pairs_path)
        return dataset_manager, pairs_manager

    def fuse_all_steps(self) -> Tuple[DatasetManager, PairsManager]:
        steps = self.existing_steps()
        dataset_managers = []
        pairs_managers = []
        for step in steps:
            dataset_path, pairs_path, step_out_paths = self._step_paths(step)
            dataset_managers.append(DatasetManager(dataset_path, step_out_paths, self.simulator))
            pairs_managers.append(PairsManager(pairs_path))

        fused_dataset_path = self.steps_root / "fused_dataset.csv"
        fused_pairs_path = self.steps_root / "fused_pairs.csv"
        fused_dataset = DatasetManager.fuse(dataset_managers, fused_dataset_path, self.out_paths, self.simulator)
        fused_pairs = PairsManager.fuse(pairs_managers, fused_pairs_path)
        return fused_dataset, fused_pairs

    def create_step(self, step: int) -> Tuple[DatasetManager, PairsManager]:
        dataset_path, pairs_path, step_out_paths = self._step_paths(step)
        dataset_manager = DatasetManager(dataset_path, step_out_paths, self.simulator)
        pairs_manager = PairsManager(pairs_path)
        return dataset_manager, pairs_manager

    def get_ranked_pairs_per_step(self) -> Dict[int, int]:
        stats: Dict[int, int] = {}
        for step in self.existing_steps():
            _, pairs_path, _ = self._step_paths(step)
            pairs_manager = PairsManager(pairs_path)
            stats[step] = pairs_manager.get_nb_ranked_pairs()
        return stats

    def _load_config(self) -> dict:
        if self.config_path.exists():
            try:
                return json.loads(self.config_path.read_text())
            except Exception:
                return {}
        return {}

    def _save_config(self, data: dict) -> None:
        try:
            self.config_path.write_text(json.dumps(data))
        except Exception:
            pass


class OnlineTrainingApp:
    DESCRIPTION = ("Online training iteratively generates simulations, ranks a subset, "
                   "trains the rewarder and generator, then regenerates improved simulations.")

    def __init__(self, root: tk.Tk, simulator: Simulator, generator: Generator, rewarder: Rewarder,
                 controller: OnlineTrainingController, frame_size: tuple = (300, 300), verbose: bool = False) -> None:
        self.root = root
        self.simulator = simulator
        self.generator = generator
        self.rewarder = rewarder
        self.controller = controller
        self.frame_size = frame_size
        self.verbose = verbose
        self.is_training = False

        # Hide main window until initial step is resolved to avoid showing an empty labeler if generating
        self.root.withdraw()
        self.current_step, self.dataset_manager, self.pairs_manager, needs_generation = self.controller.ensure_initial_step(
            prompt_fn=self._prompt_initial_sims
        )

        self.root.title("Online Training")
        sv_ttk.set_theme("dark")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.panes = ttk.Panedwindow(main_frame, orient=tk.HORIZONTAL)
        self.panes.pack(fill=tk.BOTH, expand=True)

        self.left_frame = ttk.Frame(self.panes)
        self.right_frame = ttk.Frame(self.panes, width=360)
        self.panes.add(self.left_frame, weight=3)
        self.panes.add(self.right_frame, weight=1)

        self.labeler = None
        self._build_side_panel()
        self.stats_job = None
        self._build_overlay()

        if needs_generation:
            self.root.deiconify()
            self.show_overlay("Generating initial simulations...")
            
            def run_initial_generation():
                try:
                    self.simulator.generate_pairs(
                        self.controller.initial_num_sims, 
                        self.dataset_manager, 
                        self.pairs_manager, 
                        verbose=self.verbose,
                        progress_callback=lambda msg: self.root.after(0, lambda: self.show_overlay(msg))
                    )
                finally:
                    self.root.after(0, self._finish_init)
            
            threading.Thread(target=run_initial_generation, daemon=True).start()
        else:
            self.root.deiconify()
            self._finish_init()

    def _finish_init(self):
        self.hide_overlay()
        self.labeler = QuadLabelerApp(
            self.root,
            self.simulator,
            self.dataset_manager,
            self.pairs_manager,
            verbose=self.verbose,
            frame_size=self.frame_size,
            container=self.left_frame
        )
        self.refresh_stats()

    def _build_side_panel(self):
        self.step_title_var = tk.StringVar(value=f"Online training - Step {self.current_step}")
        ttk.Label(self.right_frame, textvariable=self.step_title_var, font=("Arial", 14, "bold")).pack(pady=(0, 12))

        ttk.Label(self.right_frame, text=self.DESCRIPTION, wraplength=320, justify=tk.LEFT).pack(fill=tk.X, pady=(0, 12))

        self.dataset_count_var = tk.StringVar()
        ttk.Label(self.right_frame, textvariable=self.dataset_count_var, font=("Arial", 11)).pack(anchor=tk.W, pady=(0, 12))

        ttk.Label(self.right_frame, text="Pairs ranked per step:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.pairs_dropdown = ttk.Combobox(self.right_frame, state="readonly")
        self.pairs_dropdown.pack(fill=tk.X, pady=(4, 16))

        self.next_step_button = ttk.Button(
            self.right_frame,
            text="Launch next step",
            command=self.launch_next_step,
            style="Accent.TButton"
        )
        self.next_step_button.pack(fill=tk.X)

    def _build_overlay(self):
        self.overlay_var = tk.StringVar(value="")
        self.overlay_frame = tk.Frame(self.root, bg="#111111")
        self.overlay_frame.place_forget()
        ttk.Label(self.overlay_frame, textvariable=self.overlay_var, font=("Arial", 14, "bold")).pack(expand=True)

    def _prompt_initial_sims(self) -> int:
        return simpledialog.askinteger(
            "Online training setup",
            "How many simulations should be generated per online training step?\n"
            "This number will be applied to all online training steps.",
            parent=self.root,
            minvalue=1,
            initialvalue=8,
        ) or 8

    def show_overlay(self, message: str):
        self.overlay_var.set(message)
        self.overlay_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.overlay_frame.lift()
        self.next_step_button.state(["disabled"])

    def hide_overlay(self):
        self.overlay_frame.place_forget()
        self.next_step_button.state(["!disabled"])

    def refresh_stats(self):
        self.dataset_count_var.set(f"Number of simulations: {len(self.dataset_manager)}")
        stats = self.controller.get_ranked_pairs_per_step()
        stats[self.current_step] = self.pairs_manager.get_nb_ranked_pairs()

        current_ranked = stats.get(self.current_step, 0)
        values = [f"Current step ({self.current_step}): {current_ranked} ranked pairs"]
        for step in sorted((s for s in stats.keys() if s != self.current_step), reverse=True):
            values.append(f"Step {step}: {stats[step]} ranked pairs")

        current_sel = self.pairs_dropdown.get()
        self.pairs_dropdown["values"] = values
        if current_sel in values:
            self.pairs_dropdown.set(current_sel)
        elif values:
            self.pairs_dropdown.set(values[0])

        if self.stats_job:
            self.root.after_cancel(self.stats_job)
        # keep refreshing to reflect ongoing labeling
        self.stats_job = self.root.after(1500, self.refresh_stats)

    def launch_next_step(self):
        if self.is_training:
            return
        num_new = self.controller.initial_num_sims or 8

        self.is_training = True
        self.show_overlay("Training rewarder...")

        def run_pipeline():
            try:
                fused_dataset, fused_pairs = self.controller.fuse_all_steps()
                training_dataset = TrainingDataset(fused_pairs, fused_dataset)
                if len(training_dataset) == 0:
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Online training",
                        "No ranked pairs available to train on. Please rank more pairs before training."
                    ))
                    return

                self.rewarder.train(training_dataset)

                self.root.after(0, lambda: self.show_overlay("Training generator..."))
                self.generator.train(self.simulator, self.rewarder)
                self.rewarder.save()
                self.generator.save()

                next_step_id = self.current_step + 1
                next_dataset_manager, next_pairs_manager = self.controller.create_step(next_step_id)

                self.root.after(0, lambda: self.show_overlay("Generating new simulations..."))
                self.simulator.generate_pairs(
                    num_new,
                    next_dataset_manager,
                    next_pairs_manager,
                    verbose=self.verbose,
                    progress_callback=lambda msg: self.root.after(0, lambda: self.show_overlay(msg))
                )

                self.root.after(0, lambda: self._activate_step(next_step_id, next_dataset_manager, next_pairs_manager))
            except Exception as exc:
                traceback.print_exc()
                os._exit(1)
            finally:
                self.is_training = False
                self.root.after(0, self.hide_overlay)

        threading.Thread(target=run_pipeline, daemon=True).start()

    def _activate_step(self, step_number: int, dataset_manager: DatasetManager, pairs_manager: PairsManager):
        self.current_step = step_number
        self.dataset_manager = dataset_manager
        self.pairs_manager = pairs_manager
        self.step_title_var.set(f"Online training - Step {self.current_step}")

        if self.labeler:
            self.labeler.clear_video_widgets()
            self.labeler.history = []
            self.labeler.ignored_quads = set()
            self.labeler.current_hashes = []
            self.labeler.dataset_manager = dataset_manager
            self.labeler.pairs_manager = pairs_manager
            self.labeler.update_progress_percentage()
            self.labeler.load_next_videos()

        self.refresh_stats()

    def on_close(self):
        try:
            if self.labeler:
                self.labeler.save_and_exit()
            if self.stats_job:
                try:
                    self.root.after_cancel(self.stats_job)
                except Exception:
                    pass
                self.stats_job = None
        finally:
            self.root.destroy()


def launch_online_training(simulator: Simulator, generator: Generator, rewarder: Rewarder,
                           out_path: Path, out_paths: dict, frame_size: tuple = (300, 300), verbose: bool = False) -> None:
    """
    Launch the online training GUI window.
    """
    root = tk.Tk()
    controller = OnlineTrainingController(out_path, out_paths, simulator)
    OnlineTrainingApp(root, simulator, generator, rewarder, controller, frame_size=frame_size, verbose=verbose)
    root.mainloop()
