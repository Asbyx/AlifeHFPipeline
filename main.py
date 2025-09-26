import json
import argparse
import importlib
import sys
import shutil
from pathlib import Path
from rlhfalife.labeler import launch_video_labeler
from rlhfalife.quad_labeler import launch_quad_labeler
from rlhfalife.benchmarker import launch_benchmarker
from rlhfalife.trainer import launch_training
from rlhfalife.utils import *
from rlhfalife.data_managers import DatasetManager, PairsManager
from exporter import export_profile_interactive
from dataset_analyzer import analyze_existing_dataset, print_analysis

def get_available_profiles():
    """Get list of available profiles."""
    profiles_path = Path("profiles")
    return [d.name for d in profiles_path.iterdir() if d.is_dir() and d.name != "__pycache__"]

def get_available_configs(profile):
    """Get list of available configs for a profile."""
    configs_path = Path("profiles") / profile / "configs"
    return [c.stem for c in configs_path.iterdir() if c.name != "__pycache__" and c.suffix == '.json']

def select_profile(profile=None):
    """Select a profile, either from argument or via prompt."""
    profiles = get_available_profiles()

    if len(profiles) == 0:
        print("No profiles found. Please create a profile first. See the README for more information.")
        exit(1)

    if profile is None:
        print("Available profiles:")
        for i, p_name in enumerate(profiles):
            print(f"  {i + 1}. {p_name}")
        
        while True:
            try:
                choice = input(f"\nPlease enter the profile you want to use (number or name): ")
                if choice.isdigit():
                    index = int(choice) - 1
                    if 0 <= index < len(profiles):
                        profile = profiles[index]
                        break
                    else:
                        print(f"Invalid number. Please enter a number between 1 and {len(profiles)}.")
                elif choice in profiles:
                    profile = choice
                    break
                else:
                    print(f"Profile '{choice}' not found. Please enter a valid profile name or number.")
            except ValueError:
                print("Invalid input. Please enter a valid profile name or number.")
                
    elif profile not in profiles:
        print(f"Error: Profile '{profile}' not found.")
        exit(1)

    return profile

def select_config(profile, config=None):
    """Select a config, either from argument or via prompt."""
    configs = get_available_configs(profile)

    if len(configs) == 0:
        print(f"No configs found for profile '{profile}'. Please create configs first. See the README for more information.")
        exit(1)

    if config is None:
        print(f"\nAvailable configs for profile '{profile}':")
        for i, c_name in enumerate(configs):
            print(f"  {i + 1}. {c_name}")
        
        while True:
            try:
                choice = input(f"\nPlease enter the config you want to use (number or name): ")
                if choice.isdigit():
                    index = int(choice) - 1
                    if 0 <= index < len(configs):
                        config = configs[index]
                        break
                    else:
                        print(f"Invalid number. Please enter a number between 1 and {len(configs)}.")
                elif choice in configs:
                    config = choice
                    break
                else:
                    print(f"Config '{choice}' not found. Please enter a valid config name or number.")
            except ValueError:
                print("Invalid input. Please enter a valid config name or number.")

    elif config not in configs:
        print(f"Error: Config '{config}' not found in profile '{profile}'.")
        exit(1)

    return config

def setup_paths(profile, config):
    """Setup and return all necessary paths."""
    out_path = Path("out") / profile / config
    out_paths = {
        'outputs': out_path / "outputs",
        'videos': out_path / "videos",
        'params': out_path / "params",
        'rewarder': out_path / "rewarder",
        'generator': out_path / "generator",
        'saved_simulations': out_path / "saved_simulations",
        'benchmark': Path("out") / profile / "benchmark",
    }

    # Create all directories
    for path in out_paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return out_path, out_paths

def load_profile_module(profile):
    """Load the profile module and return it, forcing a reload if it was previously loaded."""
    module_name = f"profiles.{profile}"
    try:
        if module_name in sys.modules:
            submodules = []
            # First identify all submodules that need to be reloaded
            for name in list(sys.modules.keys()):
                if name.startswith(f"{module_name}."):
                    submodules.append(name)
            
            # Reload all submodules
            for submodule in submodules:
                if submodule in sys.modules:
                    importlib.reload(sys.modules[submodule])
                    
            profile_module = importlib.reload(sys.modules[module_name])
        else:
            profile_module = importlib.import_module(module_name)
        return profile_module
    except Exception as e:
        print(f"Error loading profile '{profile}': {str(e)}")
        return None

def generate_pairs_cli(simulator, dataset_manager, pairs_manager, num_sims, batch_size=None, verbose=True):
    """
    Generate pairs without GUI.
    
    Args:
        simulator: Simulator object to use for generating pairs
        dataset_manager: DatasetManager instance for storing simulation data
        pairs_manager: PairsManager instance for storing pairs
        num_sims: Number of simulations to generate
        batch_size: Number of simulations to generate in each batch. If None, all are done in one batch.
        verbose: Whether to print verbose output
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:            
        print(f"Generating {num_sims} simulations...")
        simulator.generate_pairs(
            num_sims, 
            dataset_manager, 
            pairs_manager, 
            batch_size=batch_size,
            verbose=verbose
        )
        print(f"Successfully generated {num_sims} simulations.")
        return True
    except Exception as e:
        print(f"Error generating pairs: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def print_menu():
    print("\nAlifeHub - Main Menu")
    print("\n--- Data Labeling ---")
    print("  1. Label Pairs (needs GUI)")
    print("  2. Quad Labeler (needs GUI)")
    print("\n--- Training & Generation ---")
    print("  3. Launch training")
    print("  4. Generate pairs (no GUI)")
    print("  5. Benchmark rewarder")
    print("\n--- Profile & Configuration ---")
    print("  6. Export profile")
    print("  7. Reload new code")
    print("  8. Reload models and data managers (updates config)")
    print("  9. Change frame size")
    print(" 10. Custom script")
    print("\n--- Data Management ---")
    print("  A. Analyze training dataset")
    print("  B. Reset labels (keep simulations and pairs)")
    print("  C. Reset config (erase everything)")
    print("\n--- General ---")
    print("  0. Exit")
    return input("\nPlease choose an option: ").upper()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AlifeHub CLI')
    parser.add_argument('--profile', '-p', type=str, help='Profile name to use')
    parser.add_argument('--config', '-c', type=str, help='Config name to use')
    parser.add_argument('--frame_size', '-f', type=int, default=300, help='Frame size to use')
    args = parser.parse_args()

    # Select profile and config
    profile = select_profile(args.profile)
    config = select_config(profile, args.config)

    print(f"Using profile: {profile}, config: {config}, frame size: {args.frame_size}.")

    # Load the profile module
    print(f"Loading profile...")
    profile_module = load_profile_module(profile)
    if profile_module is None:
        exit(1)

    # Load the config
    config_file_path = Path("profiles") / profile / "configs" / f"{config}.json"
    with open(config_file_path) as f:
        config_dict = json.load(f)

    # Setup paths
    out_path, out_paths = setup_paths(profile, config)
    dataset_path = out_path / "dataset.csv"
    pairs_path = out_path / "pairs.csv"

    # Main loop
    loader = profile_module.Loader()

    menu_actions = {
        "1": lambda: launch_video_labeler(simulator, dataset_manager, pairs_manager, verbose=False, frame_size=(args.frame_size, args.frame_size)),
        "2": lambda: launch_quad_labeler(simulator, dataset_manager, pairs_manager, verbose=False, frame_size=(args.frame_size, args.frame_size)),
        "3": lambda: launch_training(generator, rewarder, simulator, pairs_manager, dataset_manager),
        "4": lambda: generate_pairs_cli_action(simulator, dataset_manager, pairs_manager),
        "5": lambda: launch_benchmarker(simulator, generator, rewarder, out_paths, frame_size=(args.frame_size, args.frame_size)),
        "6": lambda: export_profile_interactive(),
        "9": lambda: change_frame_size_action(args),
        "A": lambda: analyze_dataset_action(dataset_manager, pairs_manager),
        "B": lambda: reset_labels_action(pairs_manager),
        "C": lambda: reset_config_action(out_path, profile, config),
        "10": lambda: loader.custom_script(generator, rewarder, simulator),
    }
    
    while True:
        print("\nLoading the generator, rewarder and simulator...")
        generator, rewarder, simulator = loader.load(out_paths, config_dict)
        
        print("\nBuilding the data managers...")
        dataset_manager = DatasetManager(dataset_path, out_paths, simulator)
        pairs_manager = PairsManager(pairs_path)
        print(f"Number of simulations: {len(dataset_manager)}")
        print(f"Number of ranked pairs: {pairs_manager.get_nb_ranked_pairs()}")
        
        choice = print_menu()
        print()
        
        if choice == "0":
            print("Exiting AlifeHub...")
            break
        elif choice == "7":
            print("Reloading new code...")            
            profile_module = load_profile_module(profile)
            loader = profile_module.Loader()
            print("Code reloaded. Models and data managers will be reloaded on next action or if you select option 8.")
        elif choice == "8":
            print("Reloading models and data managers...")
            config_file_path = Path("profiles") / profile / "configs" / f"{config}.json"
            with open(config_file_path) as f:
                config_dict = json.load(f)
            print("Models and data managers will be reloaded.")
        elif choice in menu_actions:
            menu_actions[choice]()
        else:
            print("Invalid option. Please try again.")

# Helper functions for menu actions to keep the main loop cleaner
def generate_pairs_cli_action(simulator, dataset_manager, pairs_manager):
    try:
        num_sims_str = input("Enter total number of simulations to generate (default: 5): ") or "5"
        num_sims = int(num_sims_str)
        if num_sims <= 0:
            print("Number of simulations must be positive")
            return
        
        batch_size_str = input(f"Enter batch size (press Enter for no batching): ") or str(num_sims)
        batch_size = int(batch_size_str)
        if batch_size <= 0:
            print("Batch size must be positive.")
            return
        if batch_size > num_sims:
            batch_size = num_sims
            print(f"Batch size was larger than total simulations, setting batch size to {num_sims}.")

        generate_pairs_cli(simulator, dataset_manager, pairs_manager, num_sims, batch_size=batch_size)
    except ValueError:
        print("Please enter a valid number.")

def change_frame_size_action(args):
    try:
        new_size_str = input(f"Enter new frame size (current: {args.frame_size}, default: 300): ") or "300"
        new_size = int(new_size_str)
        if new_size > 0:
            args.frame_size = new_size
            print(f"Frame size updated to {new_size}.")
        else:
            print("Frame size must be positive.")
    except ValueError:
        print("Please enter a valid number.")

def analyze_dataset_action(dataset_manager, pairs_manager):
    print("\nAnalyzing Training Dataset")
    analysis = analyze_existing_dataset(dataset_manager, pairs_manager)
    print_analysis(analysis)
    input("\nPress Enter to continue...")

def reset_labels_action(pairs_manager):
    print("\nResetting Labels")
    confirmation = input("Are you sure you want to reset all labels? This will clear all winners. (y/n): ").lower()
    if confirmation == 'y':
        pairs_manager.reset_rankings()
        print(f"All rankings have been reset. Number of unranked pairs: {pairs_manager.get_nb_unranked_pairs()}")
    else:
        print("Reset operation cancelled.")

def reset_config_action(out_path, profile, config):
    print("\nResetting Config")
    confirmation = input(f"Are you sure you want to reset the config '{config}' for profile '{profile}'? This will delete all associated data. (y/n): ").lower()
    if confirmation == 'y':
        try:
            shutil.rmtree(out_path)
            print(f"Config '{config}' for profile '{profile}' has been reset.")
            print("Please restart the application.")
            exit(0)
        except Exception as e:
            print(f"Error resetting config: {str(e)}")
    else:
        print("Reset operation cancelled.")

if __name__ == "__main__":
    main()