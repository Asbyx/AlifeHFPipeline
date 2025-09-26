import tkinter as tk
from tkinter import messagebox, ttk, filedialog, simpledialog
import threading
import cv2
from PIL import Image, ImageTk
import pandas as pd
import shutil
import sys
from scipy.stats import kendalltau
from rlhfalife.utils import Generator, Rewarder, Simulator
import traceback
from pathlib import Path
import sv_ttk


class DarkThemeManager:
    """Utility class to manage dark theme configuration for tkinter applications."""
    
    # Dark theme color palette
    COLORS = {
        'bg': '#1c1c1c',           # Dark background
        'fg': '#ffffff',           # White text
        'select_bg': '#404040',    # Selection background
        'select_fg': '#ffffff',    # Selection text
        'button_bg': '#404040',    # Button background
        'button_hover': '#505050', # Button hover
        'button_press': '#303030', # Button pressed
    }
    
    @classmethod
    def setup_theme(cls, master_window):
        """Apply dark theme to the given master window and configure ttk styles."""
        # Apply sv_ttk dark theme
        sv_ttk.set_theme("dark")
        
        # Configure the main window background
        master_window.configure(bg=cls.COLORS['bg'])
        
        # Configure ttk styles for better dark theme integration
        style = ttk.Style()
        
        # Configure Frame styles
        style.configure('Dark.TFrame', background=cls.COLORS['bg'])
        
        # Configure Label styles
        style.configure('Dark.TLabel', 
                       background=cls.COLORS['bg'], 
                       foreground=cls.COLORS['fg'])
        
        # Configure Button styles
        style.configure('Dark.TButton',
                       background=cls.COLORS['button_bg'],
                       foreground=cls.COLORS['fg'],
                       borderwidth=1,
                       focuscolor='none')
        style.map('Dark.TButton',
                 background=[('active', cls.COLORS['button_hover']),
                           ('pressed', cls.COLORS['button_press'])])
        
        # Configure Progressbar styles
        style.configure('Dark.Horizontal.TProgressbar',
                       background=cls.COLORS['button_bg'],
                       troughcolor=cls.COLORS['bg'],
                       borderwidth=1)
        
        return cls.COLORS
    
    @classmethod
    def apply_to_toplevel(cls, toplevel_window):
        """Apply dark theme background to a Toplevel window."""
        toplevel_window.configure(bg=cls.COLORS['bg'])

class LiveBenchmarkApp:
    """App for live benchmarking with auto-generated simulations and rewarder scoring."""
    
    def __init__(self, master: tk.Toplevel, number_of_videos: int, simulator: Simulator, generator: Generator, 
                 rewarder: Rewarder, out_paths: dict, frame_size: tuple = (300, 300),
                 on_close=None) -> None:
        self.master = master
        self.number_of_videos = number_of_videos
        self.simulator = simulator
        self.generator = generator
        self.rewarder = rewarder
        self.out_paths = out_paths
        self.frame_size = frame_size
        self.on_close_handler = on_close
        self.thread = None
        self._is_closing = False # Flag to signal closing process
        
        self.scores = []
        self.videos = []
        self.params = []
        self.current_index = 0
        self.after_id = None
        self.cap = None
        
        # Apply dark theme
        self.dark_colors = DarkThemeManager.setup_theme(self.master)
        
        self.master.title("Live Benchmarking")
        self.create_widgets()
        self.run_benchmark()
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.master.lift()
        self.master.focus_force()
        self.master.attributes('-topmost', True)
        self.master.attributes('-topmost', False)
        
        self.master.bind('<space>', lambda event: self.restart_video())
        self.master.bind('<Left>', lambda event: self.show_previous())
        self.master.bind('<Right>', lambda event: self.show_next())
    
    def create_widgets(self):
        """Create the UI widgets for the live benchmark app."""
        main_frame = ttk.Frame(self.master, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.status_label = ttk.Label(main_frame, text="Initializing...", style='Dark.TLabel')
        self.status_label.pack(pady=10)

        self.video_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        self.video_frame.pack()

        self.video_label = ttk.Label(self.video_frame, style='Dark.TLabel')
        self.video_label.pack()

        self.score_label = ttk.Label(main_frame, text="Score: ", style='Dark.TLabel')
        self.score_label.pack(pady=5)

        self.button_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        self.button_frame.pack(pady=10)

        self.save_button = ttk.Button(self.button_frame, text="Save Video & Params", command=self.save_video, style='Dark.TButton')
        self.save_button.pack(side="left", padx=5)

        self.save_snapshot_button = ttk.Button(self.button_frame, text="Save Benchmark snapshot", command=self.save_benchmark_snapshot, style='Dark.TButton')
        self.save_snapshot_button.pack(side="left", padx=5)

        self.prev_button = ttk.Button(self.button_frame, text="Previous", command=self.show_previous, style='Dark.TButton')
        self.prev_button.pack(side="left", padx=5)

        self.next_button = ttk.Button(self.button_frame, text="Next", command=self.show_next, style='Dark.TButton')
        self.next_button.pack(side="left", padx=5)

        self.restart_button = ttk.Button(self.button_frame, text="Restart", command=self.restart_video, style='Dark.TButton')
        self.restart_button.pack(side="left", padx=5)

        self.reroll_button = ttk.Button(self.button_frame, text="Reroll Benchmark", command=self.reroll_benchmark, style='Dark.TButton')
        self.reroll_button.pack(side="left", padx=5)
        
    def run_benchmark(self):
        """Start the benchmarking process in a separate thread."""
        self.thread = threading.Thread(target=self.benchmark_process)
        self.thread.start()

    def benchmark_process(self):
        """Run the benchmark process: generate parameters, run simulations, and score them."""
        if self._is_closing: return
        self.update_status("Generating parameters...")
        try:
            self.params = self.generator.generate(self.number_of_videos)
        except Exception as e:
            print(f"Error during parameter generation: {e}")
            self.update_status("Error during generation.") # update_status will check _is_closing
            return
        if self._is_closing: return

        # Check if two params are the same
        if any(str(self.params[i]) == str(self.params[j]) for i in range(len(self.params)) for j in range(i+1, len(self.params))):
            print("\n" + "="*50)
            print("!!! WARNING !!!: Generator generated at least two identical parameters.")
            # Filter out the identical parameters
            self.params = [self.params[i] for i in range(len(self.params)) if not any(str(self.params[i]) == str(self.params[j]) for j in range(i+1, len(self.params)))]
            print(f"Unique parameters: {len(self.params)}, over {self.number_of_videos} generated.")
            print("="*50 + "\n")
        if self._is_closing: return

        self.update_status("Running simulations...")
        try:
            outputs = self.simulator.run(self.params)
        except Exception as e:
            print(f"Error during simulation run: {e}")
            self.update_status("Error during simulation.")
            return
        if self._is_closing: return
        
        self.update_status("Scoring simulations...")
        try:
            self.scores = self.rewarder.rank(outputs)
        except Exception as e:
            print(f"Error during rewarder ranking: {e}")
            self.update_status("Error during scoring.")
            return
        if self._is_closing: return

        self.update_status("Saving videos...")
        try:
            self.videos = self.simulator.save_videos(self.generator.hash_params(self.params), outputs, self.out_paths['videos'])
        except Exception as e:
            print(f"Error during video saving: {e}")
            self.update_status("Error saving videos.")
            return
        if self._is_closing: return

        # Sort videos by score
        if not self.videos:
            self.update_status("No videos to display.")
            return
        if self._is_closing: return

        sorted_videos = sorted(zip(self.scores, self.videos, self.params), key=lambda x: x[0], reverse=True)
        self.scores, self.videos, self.params = zip(*sorted_videos)
        if self._is_closing: return

        self.update_status("Videos (best to worst):")
        if self._is_closing or not self.master.winfo_exists(): return
        self.master.after(0, lambda: self.show_video(0))

    def update_status(self, message):
        """Update the status label."""
        if not self.master.winfo_exists() or self._is_closing:
            return
        self.master.after(0, lambda: self.status_label.config(text=message))

    def show_video(self, index):
        """Display the video at the specified index."""
        self.current_index = index
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = cv2.VideoCapture(self.videos[index])
        self.update_frame()
        self.score_label.config(text=f"Score: {self.scores[index]}")

    def update_frame(self):
        """Update the video frame."""
        if self.after_id is not None:
            self.master.after_cancel(self.after_id)
            self.after_id = None

        if not self.cap or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame to match the specified frame size
            frame = cv2.resize(frame, self.frame_size)
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.video_label.config(image=img)
            self.video_label.image = img
            self.after_id = self.master.after(30, self.update_frame)  # Update every 30 ms for smooth playback
        else:
            # Reset to start when video ends
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.after_id = self.master.after(30, self.update_frame)

    def show_next(self):
        """Show the next video in the list."""
        if hasattr(self, 'videos') and self.videos and self.current_index < len(self.videos) - 1:
            if self.cap:
                self.cap.release()
            self.show_video(self.current_index + 1)

    def show_previous(self):
        """Show the previous video in the list."""
        if hasattr(self, 'videos') and self.videos and self.current_index > 0:
            if self.cap:
                self.cap.release()
            self.show_video(self.current_index - 1)

    def save_video(self):
        """Save the current video and its parameters."""
        if not hasattr(self, 'videos') or not self.videos:
            messagebox.showinfo("Info", "No videos available to save.")
            return
            
        # Save the video (duplicate the file) and params
        save_path = Path(self.out_paths['saved_simulations']) / Path(self.videos[self.current_index]).name
        shutil.copy2(self.videos[self.current_index], save_path)
        
        # Save parameters
        hash_value = self.generator.hash_params([self.params[self.current_index]])[0]
        param_path = Path(self.out_paths['saved_simulations']) / str(hash_value)
        self.simulator.save_param(self.params[self.current_index], str(param_path))

        print(f"Video saved to {self.out_paths['saved_simulations']}.")
        self.update_status(f"Video and its parameters saved to {self.out_paths['saved_simulations']} !")

    def restart_video(self):
        """Restart the current video from the beginning."""
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def reroll_benchmark(self):
        """Clean up current benchmark and generate a new one."""
        if self.thread and self.thread.is_alive():
            messagebox.showinfo("Info", "A benchmark generation is already in progress. Please wait.")
            return

        # Stop current video playback and release capture
        if self.after_id is not None:
            self.master.after_cancel(self.after_id)
            self.after_id = None
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        
        # Clear video display
        self.video_label.config(image=None)
        self.video_label.image = None
        self.score_label.config(text="Score: ")

        # Delete old video files
        if hasattr(self, 'videos') and self.videos:
            for video_path in self.videos:
                try:
                    if Path(video_path).exists(): # Check if file exists before attempting to delete
                        Path(video_path).unlink()
                except Exception as e:
                    print(f"Error deleting old video {video_path}: {e}")
            print("Deleted old benchmark videos.")

        # Reset benchmark data
        self.scores = []
        self.videos = []
        self.params = []
        self.current_index = 0
        
        # Update status and run new benchmark
        self.update_status("Rerolling benchmark...")
        self.run_benchmark()

    def save_benchmark_snapshot(self):
        """Saves a snapshot of the current benchmark."""
        if not hasattr(self, 'videos') or not self.videos:
            messagebox.showinfo("Info", "No videos available to save.")
            return

        snapshot_path_str = filedialog.askdirectory(
            title="Save Benchmark Snapshot In",
            initialdir=str(Path(self.out_paths.get('root', '.')) / "benchmarks"),
        )

        snapshot_name = simpledialog.askstring("Save Benchmark Snapshot As", "Enter the name of the snapshot: ")

        if not snapshot_path_str:
            return

        snapshot_path = Path(snapshot_path_str) / snapshot_name
        
        should_overwrite = False
        if snapshot_path.exists():
            if messagebox.askyesno("Overwrite?", f"The destination '{snapshot_path.name}' already exists. Do you want to overwrite it?"):
                should_overwrite = True
            else:
                return
        
        # Create and show progress window
        progress_window = tk.Toplevel(self.master)
        progress_window.title("Saving Snapshot")
        DarkThemeManager.apply_to_toplevel(progress_window)

        self.master.update_idletasks()
        master_x = self.master.winfo_x()
        master_y = self.master.winfo_y()
        master_w = self.master.winfo_width()
        master_h = self.master.winfo_height()
        prog_w = 400
        prog_h = 120
        prog_x = master_x + (master_w - prog_w) // 2
        prog_y = master_y + (master_h - prog_h) // 2
        progress_window.geometry(f"{prog_w}x{prog_h}+{prog_x}+{prog_y}")
        progress_window.transient(self.master)
        progress_window.grab_set()

        status_label = ttk.Label(progress_window, text="Saving benchmark snapshot...", style='Dark.TLabel')

        status_label.pack(pady=10, padx=10)

        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=350, mode="determinate")
        progress_bar.pack(pady=5, padx=10)
        
        # Start saving in a new thread
        threading.Thread(
            target=self._save_snapshot_worker,
            args=(snapshot_path, should_overwrite, progress_window, progress_bar, status_label),
            daemon=True
        ).start()

    def _save_snapshot_worker(self, snapshot_path, should_overwrite, progress_window, progress_bar, status_label):
        """Worker thread to save the benchmark snapshot."""
        try:
            # Create directories
            videos_dir = snapshot_path / "videos"
            params_dir = snapshot_path / "parameters"
            
            if snapshot_path.exists() and should_overwrite:
                shutil.rmtree(snapshot_path)
            
            snapshot_path.mkdir(parents=True, exist_ok=False)
            videos_dir.mkdir()
            params_dir.mkdir()

            num_videos = len(self.videos)
            if progress_window.winfo_exists():
                 progress_window.after(0, lambda: progress_bar.config(maximum=num_videos))

            # Data is already sorted by score (desc) in benchmark_process
            for i, (score, video_path, param) in enumerate(zip(self.scores, self.videos, self.params)):
                if not progress_window.winfo_exists(): # Check if user closed the window
                    print("Snapshot saving cancelled by user.")
                    shutil.rmtree(snapshot_path)
                    return

                rank = i + 1
                
                # Format filenames
                base_filename = f"{rank:02d}_score_{score:.4f}"
                video_filename = base_filename + ".mp4"
                
                # Copy video
                shutil.copy2(video_path, videos_dir / video_filename)
                
                # Save parameter
                self.simulator.save_param(param, str(params_dir / base_filename))
                
                # Update progress bar
                if progress_window.winfo_exists():
                    progress_window.after(0, lambda i=i: progress_bar.config(value=i + 1))

            if progress_window.winfo_exists():
                progress_window.after(0, lambda: status_label.config(text="Snapshot saved successfully!"))

        except Exception as e:
            error_message = f"Failed to save snapshot: {e}"
            print(error_message)
            traceback.print_exc()
            if progress_window.winfo_exists():
                progress_window.after(0, lambda: status_label.config(text=error_message))
        
        finally:
            if progress_window.winfo_exists():
                close_button = ttk.Button(progress_window, text="Close", command=progress_window.destroy, style='Dark.TButton')
                progress_window.after(0, lambda: close_button.pack(pady=10))
                progress_window.after(0, progress_window.grab_release)

    def on_close(self):
        """Handle window close event."""
        self._is_closing = True # Signal that the window is closing

        # Release video resources
        if self.thread and self.thread.is_alive():
            try:
                self.thread.join(1) # Wait a bit for the thread to finish
            except RuntimeError: # Can happen if thread is already in a bad state
                pass
        self.thread = None

        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

        # Delete video files
        if hasattr(self, 'videos') and self.videos:
            for video_path in self.videos:
                try:
                    if Path(video_path).exists(): # Check if file exists before attempting to delete
                        Path(video_path).unlink()
                except Exception as e:
                    print(f"Error deleting {video_path}: {e}")
            print("Deleted all videos from the benchmark.")

        # Ask if user wants to save the generator and rewarder
        save_generator = messagebox.askyesno("Save Generator", "Do you want to save the generator?")
        save_rewarder = messagebox.askyesno("Save Rewarder", "Do you want to save the rewarder?")

        if save_generator:
            self.generator.save()   
        if save_rewarder:
            self.rewarder.save()

        # Call the on_close callback if provided
        if self.on_close_handler:
            self.on_close_handler()

class CreateBenchmarkApp:
    """App for creating a benchmark by manually ranking videos."""
    
    def __init__(self, master: tk.Toplevel, simulator: Simulator, generator: Generator, 
                 rewarder: Rewarder, out_paths: dict, frame_size: tuple = (300, 300),
                 on_close=None) -> None:
        self.master = master
        self.simulator = simulator
        self.generator = generator
        self.rewarder = rewarder
        self.out_paths = out_paths
        self.frame_size = frame_size
        self.on_close_handler = on_close
        self.thread = None
        self._is_closing = False # Flag to signal closing process
        
        self.videos = []
        self.params = []
        self.hashs = []
        self.video_widgets = []
        self.relationship_buttons = []  # Store INLINE relationship buttons
        self.inter_row_relationship_button = None # Button between rows
        self.outputs = []
        
        # For drag and drop
        self.drag_source = None
        self.highlight_widget = None
        
        # Apply dark theme
        self.dark_colors = DarkThemeManager.setup_theme(self.master)
        
        self.master.title("Create Benchmark")
        self.create_widgets()
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.master.lift()
        self.master.focus_force()
        self.master.attributes('-topmost', True)
        self.master.attributes('-topmost', False)
        
        self.master.bind('<space>', lambda event: self.restart_video())
        self.master.bind('<Left>', lambda event: self.show_previous())
        self.master.bind('<Right>', lambda event: self.show_next())
        
        # Flag to prevent concurrent generation
        self._is_generating = False
        
        # Initial generation
        self.generate_videos()
    
    def create_widgets(self):
        """Create the UI widgets for the benchmark creation app."""
        # Main frame
        main_frame = ttk.Frame(self.master, padding=10, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title and instruction
        ttk.Label(main_frame, text="Create Benchmark", font=("Arial", 16, "bold"), style='Dark.TLabel').pack(pady=(0, 5))
        instructions = """Drag videos to rank them from best (top-left) to worst (bottom-right).\nUse the switches between videos to mark them as equal (=) or greater than (<)."""
        ttk.Label(main_frame, text=instructions, justify=tk.CENTER, style='Dark.TLabel').pack(pady=(0, 10))
        
        # Create a frame for the two rows of videos
        self.videos_area_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        self.videos_area_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Frame for the top row of videos
        self.top_row_frame = ttk.Frame(self.videos_area_frame, style='Dark.TFrame')
        self.top_row_frame.pack(side=tk.TOP, fill=tk.X, expand=True, padx=10, pady=5)

        # Frame for the button between rows
        self.inter_row_frame = ttk.Frame(self.videos_area_frame, style='Dark.TFrame')
        self.inter_row_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        # Frame for the bottom row of videos
        self.bottom_row_frame = ttk.Frame(self.videos_area_frame, style='Dark.TFrame')
        self.bottom_row_frame.pack(side=tk.TOP, fill=tk.X, expand=True, padx=10, pady=5)

        # Button frame
        button_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        button_frame.pack(pady=10)
        
        # Save benchmark button
        self.save_button = ttk.Button(button_frame, text="Save Benchmark as default", command=self.save_benchmark, style='Dark.TButton')
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # Save benchmark as button
        self.save_as_button = ttk.Button(button_frame, text="Save Benchmark in custom location", command=self.save_benchmark_as, style='Dark.TButton')
        self.save_as_button.pack(side=tk.LEFT, padx=5)
        
        # Generate new videos button
        self.generate_button = ttk.Button(button_frame, text="Generate New Videos", command=self.generate_videos, style='Dark.TButton')
        self.generate_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready", style='Dark.TLabel')
        self.status_label.pack(pady=5)
    
    def toggle_relationship(self, button_or_idx):
        """Toggle the relationship between videos from = to > or vice versa.
           Handles both inline buttons (by index) and the inter-row button (by object)."""
        if isinstance(button_or_idx, int):
            button = self.relationship_buttons[button_or_idx]
        elif button_or_idx == 'inter_row':
             if not self.inter_row_relationship_button: return
             button = self.inter_row_relationship_button
        else:
             # Should ideally not happen if button object is passed, but check anyway
             button = button_or_idx # Assume it's the button object

        current_text = button.cget("text")
        
        if current_text == "=":
            button.config(text="<")
        else:
            button.config(text="=")
    
    def generate_videos(self):
        """Start the video generation process in a background thread."""
        if self._is_generating:
            print("Generation already in progress.")
            return

        self._is_generating = True

        # Clear current videos and update UI for loading state
        self.clear_video_widgets()
        self.status_label.config(text="Generating videos...")
        self.generate_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.save_as_button.config(state=tk.DISABLED)
        self.master.update_idletasks() # Ensure UI updates before starting thread

        # Start generation in a new thread
        self.thread = threading.Thread(target=self._generation_worker, daemon=True)
        self.thread.start()

    def _update_status_safe(self, message):
        """Safely update the status label from any thread."""
        if self.master.winfo_exists() and not self._is_closing: # Check if window still exists and not closing
            self.master.after(0, lambda: self.status_label.config(text=message))

    def _generation_worker(self):
        """Worker function to generate parameters, run simulations, and save videos."""
        try:
            if self._is_closing: return
            # Generate parameters
            self._update_status_safe("Generating parameters...")
            try:
                params = self.generator.generate(10)
            except Exception as e:
                error_message = f"Error during parameter generation: {e}"
                print(error_message)
                traceback.print_exc()
                self._update_status_safe(error_message)
                if self.master.winfo_exists() and not self._is_closing: self.master.after(0, self._reset_ui_after_generation)
                return
            if self._is_closing: return

            # Check for duplicates
            unique_params = []
            seen_params_str = set()
            for p in params:
                p_str = str(p)
                if p_str not in seen_params_str:
                    unique_params.append(p)
                    seen_params_str.add(p_str)
                else:
                     print(f"Warning: Duplicate parameter generated and skipped: {p_str}")

            if len(unique_params) < len(params):
                print("\n" + "="*50)
                print(f"!!! WARNING !!!: Generator generated {len(params) - len(unique_params)} duplicate parameters.")
                print(f"Proceeding with {len(unique_params)} unique parameters.")
                print("="*50 + "\n")

            if not unique_params:
                self._update_status_safe("Error: No unique parameters generated.")
                if self.master.winfo_exists() and not self._is_closing:
                    self.master.after(0, self._finish_generation, [], [], [], []) # Finish with empty results
                return

            params = unique_params
            if self._is_closing: return

            # Run simulations
            self._update_status_safe("Running simulations...")
            try:
                outputs = self.simulator.run(params)
            except Exception as e:
                error_message = f"Error during simulation run: {e}"
                print(error_message)
                traceback.print_exc()
                self._update_status_safe(error_message)
                if self.master.winfo_exists() and not self._is_closing: self.master.after(0, self._reset_ui_after_generation)
                return
            if self._is_closing: return

            # Get hashes
            hashs = self.generator.hash_params(params)
            if self._is_closing: return

            # Save videos
            self._update_status_safe("Saving videos...")
            try:
                videos = self.simulator.save_videos(hashs, outputs, self.out_paths['videos'])
            except Exception as e:
                error_message = f"Error during video saving: {e}"
                print(error_message)
                traceback.print_exc()
                self._update_status_safe(error_message)
                if self.master.winfo_exists() and not self._is_closing: self.master.after(0, self._reset_ui_after_generation)
                return
            if self._is_closing: 
                # Clean up videos if we generated some then closed
                if videos:
                    print("Window closed during video saving, cleaning up temporary videos.")
                    for video_path in videos:
                        try:
                            if Path(video_path).exists(): Path(video_path).unlink()
                        except Exception as e_del: print(f"Error deleting temp video {video_path}: {e_del}")
                return

            # Schedule the final UI update on the main thread
            if self.master.winfo_exists() and not self._is_closing:
                self.master.after(0, self._finish_generation, videos, params, hashs, outputs)

        except Exception as e:
            error_message = f"Error during generation: {e}"
            print(error_message)
            traceback.print_exc()
            self._update_status_safe(error_message)
            # Ensure UI is reset even on error
            if self.master.winfo_exists() and not self._is_closing:
                self.master.after(0, self._reset_ui_after_generation)


    def _finish_generation(self, videos, params, hashs, outputs):
        """Update the UI after video generation is complete. Runs in the main thread."""
        if not self.master.winfo_exists() or self._is_closing: # Check if window was closed or is closing
            # If closing, clean up any videos that might have been generated by this worker run
            # The main on_close will handle self.videos, but these are local to this generation run
            if videos:
                print("Finishing generation but window is closing/closed. Cleaning up videos from this run.")
                for video_path in videos:
                    try:
                        if Path(video_path).exists(): Path(video_path).unlink()
                    except Exception as e:
                        print(f"Error deleting temporary video {video_path} during _finish_generation on close: {e}")
            return

        self.videos = videos
        self.params = params
        self.hashs = [str(h) for h in hashs]
        self.outputs = outputs

        if not videos:
             self.status_label.config(text="Ready (No videos generated)")
        else:
            # Create video widgets
            self.create_video_widgets()
            self.status_label.config(text="Ready - Drag to reorder and use switches to set ranks")
            self.master.update_idletasks() # Ensure widgets are drawn
            # Make the window fullscreen instead of resizing to estimated dimensions
            try:
                # Different approach depending on platform
                if sys.platform == "win32":
                    self.master.state('zoomed')  # Windows-specific fullscreen
                else:
                    # For Linux/Mac
                    self.master.attributes('-fullscreen', True)
                    # Optional: Add an escape binding to exit fullscreen if needed
                    self.master.bind("<Escape>", lambda event: self.master.attributes("-fullscreen", False))
            except tk.TclError as e:
                print(f"Could not make window fullscreen: {e}")

        if self.thread:
            self.thread.join(1)
            self.thread = None

        self._reset_ui_after_generation()

    def _reset_ui_after_generation(self):
         """Reset UI elements after generation attempt (success or failure)."""
         if not self.master.winfo_exists() or self._is_closing:
              return
         # Re-enable buttons
         self.generate_button.config(state=tk.NORMAL)
         self.save_button.config(state=tk.NORMAL)
         self.save_as_button.config(state=tk.NORMAL)
         self._is_generating = False

    def create_video_widgets(self):
        """Create draggable video widgets for ranking across two rows."""
        self.clear_video_widgets()
        
        if not self.videos:
            print("No videos available to display.")
            return

        num_videos = len(self.videos)
        mid_point = (num_videos + 1) // 2 # Split point for rows

        # Create widgets row by row
        inline_button_idx = 0
        for i in range(num_videos):
            current_row_frame = self.top_row_frame if i < mid_point else self.bottom_row_frame
            
            # Add an inline relationship button before this video (if not the first overall AND not the first of the second row)
            if i > 0 and i != mid_point:
                relationship_button = ttk.Button(
                    current_row_frame, 
                    text="<",  # Default is "greater than"
                    width=3,
                    style='Dark.TButton',
                    # Use lambda with default argument to capture current index
                    command=lambda idx=inline_button_idx: self.toggle_relationship(idx)
                )
                relationship_button.pack(side=tk.LEFT, padx=5, pady=(self.frame_size[1] // 2)) # Center vertically approx
                self.relationship_buttons.append(relationship_button)
                inline_button_idx += 1
            
            # Add the video widget
            video_widget = DraggableVideo(
                current_row_frame,
                self.videos[i],
                i, # Overall index (rank)
                self.frame_size,
                hash_value=self.hashs[i],
                on_drag_start=self.start_drag,
                on_drag_release=self.end_drag,
                on_drag_motion=self.on_drag_motion
            )
            video_widget.pack(side=tk.LEFT, padx=10)
            self.video_widgets.append(video_widget)

        # Add the button between the two rows (if there's more than one video)
        if num_videos > 1 and mid_point < num_videos: # Ensure there's a split
             self.inter_row_relationship_button = ttk.Button(
                  self.inter_row_frame,
                  text="<",
                  width=3,
                  style='Dark.TButton',
                  command=lambda: self.toggle_relationship('inter_row')
             )
             # Pack it in the center of the inter_row_frame
             self.inter_row_relationship_button.pack(anchor=tk.CENTER)

    def clear_video_widgets(self):
        """Clear all video widgets and relationship buttons."""
        for widget in self.video_widgets:
            widget.release_resources()
            widget.destroy()
        
        for button in self.relationship_buttons:
            button.destroy()
        
        if self.inter_row_relationship_button:
             self.inter_row_relationship_button.destroy()
             self.inter_row_relationship_button = None

        self.video_widgets = []
        self.relationship_buttons = []
    
    def start_drag(self, source_widget):
        """Handle the start of a drag operation."""
        self.drag_source = source_widget
        self.highlight_widget = None
    
    def on_drag_motion(self, source_widget, x, y):
        """Handle drag motion to provide visual feedback."""
        # Find which widget is under the mouse cursor
        target_widget = self.find_widget_at_position(x, y)
        
        # If we've moved to a different widget, update highlighting
        if target_widget != self.highlight_widget and target_widget != source_widget:
            # Remove previous highlighting
            if self.highlight_widget and self.highlight_widget != self.drag_source:
                self.highlight_widget.config(relief=tk.RAISED)
            
            # Add highlighting to new widget
            if target_widget and target_widget != self.drag_source:
                target_widget.config(relief=tk.RIDGE)  # Light green highlight
                
            self.highlight_widget = target_widget
    
    def find_widget_at_position(self, x, y):
        """Find which video widget is at the given screen position."""
        for widget in self.video_widgets:
            widget_x = widget.winfo_rootx()
            widget_y = widget.winfo_rooty()
            widget_width = widget.winfo_width()
            widget_height = widget.winfo_height()
            
            if (widget_x <= x <= widget_x + widget_width and 
                widget_y <= y <= widget_y + widget_height):
                return widget
        
        return None
    
    def end_drag(self, source_widget, event):
        """Handle the end of a drag operation."""
        # Find which widget we're over
        x, y = event.x_root, event.y_root
        target_widget = self.find_widget_at_position(x, y)
        
        # Clean up any highlighting from drag_motion
        if self.highlight_widget and self.highlight_widget != self.drag_source:
            try:
                # Ensure the widget still exists before trying to configure it
                if self.highlight_widget.winfo_exists():
                    self.highlight_widget.config(relief=tk.RAISED)
            except tk.TclError:
                 pass # Widget might have been destroyed

        self.highlight_widget = None
        
        if target_widget and target_widget != source_widget:
            # Get indices
            source_idx = self.video_widgets.index(source_widget)
            target_idx = self.video_widgets.index(target_widget)

            # Swap elements in all lists
            self.video_widgets[source_idx], self.video_widgets[target_idx] = self.video_widgets[target_idx], self.video_widgets[source_idx]
            self.hashs[source_idx], self.hashs[target_idx] = self.hashs[target_idx], self.hashs[source_idx]
            self.videos[source_idx], self.videos[target_idx] = self.videos[target_idx], self.videos[source_idx]
            self.params[source_idx], self.params[target_idx] = self.params[target_idx], self.params[source_idx]
            self.outputs[source_idx], self.outputs[target_idx] = self.outputs[target_idx], self.outputs[source_idx]
            
            # Update rank indicators for all widgets after the swap
            for i, widget in enumerate(self.video_widgets):
                 if widget.winfo_exists(): # Check if widget still exists
                      widget.update_rank(i)
                 else:
                      print(f"Warning: Attempted to update rank on a non-existent widget at index {i}")

            # Repack all widgets visually based on the new order
            self.repack_widgets()

        # Reset drag source after operation
        self.drag_source = None
    
    def repack_widgets(self):
        """Repack all widgets into the two-row layout based on current order."""
        
        # Store current frame positions before destroying widgets
        frame_positions = {}
        for widget in self.video_widgets:
            # Use video_path as a unique key (assuming paths are unique and widgets exist)
            if widget.winfo_exists():
                frame_positions[widget.video_path] = widget.get_current_frame_pos()

        # Store relationship button states
        inline_button_states = [btn.cget("text") for btn in self.relationship_buttons]
        inter_row_button_state = None
        if self.inter_row_relationship_button:
            inter_row_button_state = self.inter_row_relationship_button.cget("text")

        # Destroy all existing widgets and buttons first
        for widget in self.video_widgets:
            widget.release_resources() # Release video capture first
            widget.destroy()
        self.video_widgets.clear()

        for button in self.relationship_buttons:
            button.destroy()
        self.relationship_buttons.clear()

        if self.inter_row_relationship_button:
            self.inter_row_relationship_button.destroy()
            self.inter_row_relationship_button = None

        num_videos = len(self.videos) # Use data list length
        if num_videos == 0:
            return

        mid_point = (num_videos + 1) // 2
        
        # Recreate widgets and inline buttons based on the potentially reordered data lists
        inline_button_idx = 0
        for i in range(num_videos):
            current_row_frame = self.top_row_frame if i < mid_point else self.bottom_row_frame
            
            # Add an inline relationship button before this video
            if i > 0 and i != mid_point:
                button_text = "<" # Default
                if inline_button_idx < len(inline_button_states):
                    button_text = inline_button_states[inline_button_idx]
                
                relationship_button = ttk.Button(
                    current_row_frame, 
                    text=button_text, # Restore state
                    width=3,
                    style='Dark.TButton',
                    command=lambda idx=inline_button_idx: self.toggle_relationship(idx)
                )
                relationship_button.pack(side=tk.LEFT, padx=5, pady=(self.frame_size[1] // 2))
                self.relationship_buttons.append(relationship_button)
                inline_button_idx += 1
                
            # Get the saved frame position
            start_frame_pos = frame_positions.get(self.videos[i], 0)

            # Recreate the video widget with the correct parent and start frame
            video_widget = DraggableVideo(
                current_row_frame, # Correct parent
                self.videos[i],
                i, # Overall index (rank)
                self.frame_size,
                hash_value=self.hashs[i],
                on_drag_start=self.start_drag,
                on_drag_release=self.end_drag,
                on_drag_motion=self.on_drag_motion,
                start_frame=start_frame_pos # Pass the saved position
            )
            video_widget.pack(side=tk.LEFT, padx=10)
            self.video_widgets.append(video_widget) # Add the new widget to the list

        # Recreate and repack the inter-row button if needed
        if num_videos > 1 and mid_point < num_videos: # Ensure there's a split
             button_text = "<" # Default
             if inter_row_button_state:
                 button_text = inter_row_button_state
             self.inter_row_relationship_button = ttk.Button(
                  self.inter_row_frame,
                  text=button_text, # Restore state
                  width=3,
                  style='Dark.TButton',
                  command=lambda: self.toggle_relationship('inter_row')
             )
             self.inter_row_relationship_button.pack(anchor=tk.CENTER)

        # Optional: Try to resize window again after repack
        self.master.update_idletasks()
        # Potentially recalculate and set geometry here if needed

    def save_benchmark_as(self):
        """Save the benchmark in a user-selected location."""
        if not self.outputs or not self.hashs:
            messagebox.showinfo("Info", "No outputs to save.")
            return
        
        # Ask user for directory
        from tkinter import filedialog
        custom_dir = filedialog.askdirectory(
            title="Select Directory to Save Benchmark",
            initialdir=Path(self.out_paths["benchmark"]).parent
        )
        
        if not custom_dir:  # User cancelled
            return
            
        # Save to the selected directory
        self._save_benchmark_to_location(custom_dir)
    
    def save_benchmark(self):
        """Save the current ranking as a benchmark to the default location."""
        if not self.outputs or not self.hashs:
            messagebox.showinfo("Info", "No outputs to save.")
            return
        
        # Use the default benchmark directory
        benchmark_dir = self.out_paths["benchmark"]
        self._save_benchmark_to_location(benchmark_dir)
        
    def _save_benchmark_to_location(self, benchmark_dir):
        """Save the benchmark to the specified location."""
        # Create directory if it doesn't exist
        if Path(benchmark_dir).exists():
            shutil.rmtree(benchmark_dir)
        Path(benchmark_dir).mkdir(parents=True, exist_ok=True)
        
        # Calculate actual ranks based on relationships (= buttons)
        ranks = self.calculate_ranks()
        
        output_files = []
        benchmark_data = []
        # Process outputs and create benchmark data in a single loop
        for i, (output, hash_val) in enumerate(zip(self.outputs, self.hashs)):
            # Save output and get the file path
            output_file = self.simulator.save_output(output, Path(benchmark_dir) / hash_val)
            output_files.append(output_file)
            
            # Add entry to benchmark data
            benchmark_data.append({
                'rank': ranks[i],
                'hash': hash_val,
                'output_file': Path(output_file).name,
            })
        
        # Save to CSV
        df = pd.DataFrame(benchmark_data)
        benchmark_csv = Path(benchmark_dir) / "benchmark.csv"
        df.to_csv(benchmark_csv, index=False)

        messagebox.showinfo("Success", f"Benchmark saved to {benchmark_dir}")
        self.status_label.config(text=f"Benchmark saved to {benchmark_dir}")

        # close the window
        self.master.quit()
        self.master.destroy()
    
    def calculate_ranks(self):
        """Calculate actual ranks based on relationship buttons across two rows."""
        num_videos = len(self.video_widgets)
        if num_videos == 0:
            return []

        ranks = list(range(1, num_videos + 1)) # Initial sequential ranks
        
        # Process relationships to adjust ranks for equalities ('=')
        current_rank = 1
        adjusted_ranks = [0] * num_videos # Store final ranks
        
        mid_point = (num_videos + 1) // 2
        inline_button_idx = 0
        
        for i in range(num_videos):
            adjusted_ranks[i] = current_rank
            
            is_last_in_row1 = (i == mid_point - 1)
            is_last_overall = (i == num_videos - 1)

            # Determine the relationship to the *next* item
            relationship = '<' # Default
            if is_last_in_row1 and self.inter_row_relationship_button:
                # Relationship is determined by the inter-row button
                relationship = self.inter_row_relationship_button.cget("text")
            elif not is_last_overall and i + 1 != mid_point:
                # Relationship is determined by the next inline button
                if inline_button_idx < len(self.relationship_buttons):
                     relationship = self.relationship_buttons[inline_button_idx].cget("text")
                     inline_button_idx += 1
                else:
                     print(f"Warning: Mismatch between video count and relationship buttons at index {i}")

            # If the relationship to the next item is NOT '=', increment the rank for the next item
            if relationship == '<':
                current_rank += 1
                
        # Final rank adjustment pass for tied ranks (could be simpler ways)
        # Re-iterate based on buttons to ensure consistency for '=' chains
        final_ranks = list(range(1, num_videos + 1))
        
        processed_buttons = 0
        for i in range(num_videos - 1):
             is_inter_row_comparison = (i == mid_point - 1)
             
             button_text = '<' # Default assumption
             if is_inter_row_comparison:
                  if self.inter_row_relationship_button:
                       button_text = self.inter_row_relationship_button.cget("text")
             else:
                  # This relies on buttons being in order
                  if processed_buttons < len(self.relationship_buttons):
                     button_text = self.relationship_buttons[processed_buttons].cget("text")
                     processed_buttons += 1

             if button_text == '=':
                  # Make rank i+1 equal to rank i
                  rank_to_assign = final_ranks[i]
                  final_ranks[i+1] = rank_to_assign
                  # Adjust all subsequent ranks
                  for j in range(i + 2, num_videos):
                       if final_ranks[j] > rank_to_assign: # If it was ranked higher
                           # Check previous rank; if it was also higher, decrement
                           if final_ranks[j] == final_ranks[j-1] + 1:
                                final_ranks[j] -= 1
                       elif final_ranks[j] == rank_to_assign: # If already equal, ensure consistency
                            pass # Keep it the same


        return final_ranks
    
    def on_close(self):
        """Handle window close event."""
        self._is_closing = True # Signal that the window is closing

        # Release video resources
        self.clear_video_widgets()
        
        if self.thread and self.thread.is_alive():
            try:
                self.thread.join(1) # Wait a bit for the thread to finish
            except RuntimeError:
                pass # Thread might be in a bad state or already joined/exited
        self.thread = None
        
        # Delete temporary videos (self.videos list)
        for video_path in self.videos:
            try:
                if Path(video_path).exists(): # Check if file exists before attempting to delete
                    Path(video_path).unlink()
            except Exception as e:
                print(f"Error deleting {video_path}: {e}")
        
        # Call the on_close callback
        if self.on_close_handler:
            self.on_close_handler()

class DraggableVideo(ttk.Frame):
    """A draggable video widget that can be reordered via drag and drop."""
    
    def __init__(self, parent, video_path, index, frame_size, hash_value, 
                 on_drag_start=None, on_drag_release=None, on_drag_motion=None,
                 start_frame=0):
        """
        Initialize a draggable video widget.
        
        Args:
            parent: The parent widget
            video_path: Path to the video file
            index: The index of this video (0-3)
            frame_size: Size of the video frame (width, height)
            hash_value: Hash value of the video for display
            on_drag_start: Callback function when drag starts
            on_drag_release: Callback function when drag ends
            on_drag_motion: Callback function when dragging
            start_frame: The starting frame for the video
        """
        super().__init__(parent)
        self.config(relief=tk.RAISED, borderwidth=2) # This might not work with ttk themes but let's try
        self.parent = parent
        self.index = index
        self.frame_size = frame_size
        self.video_path = video_path
        self.hash_value = hash_value
        self.on_drag_start = on_drag_start
        self.on_drag_release = on_drag_release
        self.on_drag_motion = on_drag_motion
        self.is_dragging = False
        
        # Create the video label
        self.video_label = ttk.Label(self)
        self.video_label.pack(padx=5, pady=5)
        
        # Create hash label
        self.hash_label = ttk.Label(self, text=f"Hash: {hash_value}", font=("Arial", 8), wraplength=frame_size[0]-10)
        self.hash_label.pack(pady=(0, 5))
        
        # Create ranking indicator
        self.rank_label = ttk.Label(self, text=f"#{index+1}", font=("Arial", 14, "bold"))
        self.rank_label.pack(pady=5)
        
        # Open the video
        self.cap = cv2.VideoCapture(video_path)
        
        # Set the starting frame if provided and valid
        if start_frame > 0 and self.cap.isOpened():
            set_success = self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            if not set_success:
                print(f"Warning: Could not set start frame for {video_path} to {start_frame}")

        # Bind events for dragging
        for widget in [self, self.video_label, self.rank_label, self.hash_label]:
            widget.bind("<ButtonPress-1>", self.on_press)
            widget.bind("<ButtonRelease-1>", self.on_release)
            widget.bind("<B1-Motion>", self.on_motion)
        
        # Track mouse offset for dragging
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        # For drag and drop feedback
        
        # Start playing the video
        self.after_id = None
        self.update_frame()
    
    def get_current_frame_pos(self):
        """Returns the current frame number of the video."""
        if self.cap and self.cap.isOpened():
            # Get the position, ensuring it's an integer
            pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            return int(pos) if pos is not None else 0
        return 0

    def update_frame(self):
        """Update the video frame."""
        # Prevent errors if cap is not ready or widget destroyed
        if not hasattr(self, 'cap') or not self.cap or not self.cap.isOpened() or not self.winfo_exists():
            return
            
        ret, frame = self.cap.read()
        
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            
        if ret:
            # Convert and resize frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.frame_size)
            
            # Convert to PIL Image and then to PhotoImage
            img = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=img)
            
            # Update label
            self.video_label.config(image=photo)
            self.video_label.image = photo
            
            # Schedule next update
            self.after_id = self.after(33, self.update_frame)  # ~30 fps
    
    def update_rank(self, new_index):
        """Update the rank label to show new position."""
        self.index = new_index
        self.rank_label.config(text=f"#{new_index+1}")
    
    def on_press(self, event):
        """Handle mouse button press event."""
        self.is_dragging = True
        # Record the offset of the mouse click relative to the widget
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        if self.on_drag_start:
            self.on_drag_start(self)
        
        # Change appearance to indicate dragging
        self.config(relief=tk.SUNKEN)
    
    def on_motion(self, event):
        """Handle drag motion to provide visual feedback."""
        if not self.is_dragging:
            return
            
        # Get mouse position relative to the screen
        x_root = event.x_root
        y_root = event.y_root
        
        # Call the motion callback if it exists
        if self.on_drag_motion:
            self.on_drag_motion(self, x_root, y_root)
    
    def on_release(self, event):
        """Handle mouse button release event."""
        if self.is_dragging:
            self.is_dragging = False
            # Reset appearance
            self.config(relief=tk.RAISED)
            
            if self.on_drag_release:
                self.on_drag_release(self, event)
    
    def release_resources(self):
        """Release video resources."""
        if self.after_id:
            self.after_cancel(self.after_id)
        if self.cap and self.cap.isOpened():
            self.cap.release()

def test_rewarder_on_benchmark(simulator: Simulator, rewarder: Rewarder, out_paths: dict, verbose: bool = True) -> None:
    """
    Test the rewarder against a benchmark.
    
    Args:
        simulator: The simulator to use for loading outputs
        rewarder: The rewarder to test
        out_paths: Dictionary containing a "benchmark" key with the path to the benchmark
            this is the path to the directory containing the benchmark.csv file and the outputs
        verbose: Whether to print detailed progress and results
    """
    def print_v(*args, **kwargs):
        """Print only if verbose is True."""
        if verbose:
            print(*args, **kwargs)
    
    benchmark_file = Path(out_paths["benchmark"]) / "benchmark.csv"
    
    # Check if benchmark exists
    if not benchmark_file.exists():
        print(f"Error: Benchmark file not found.")
        print("Please create a benchmark first using option 2.")
        return
    
    print_v(f"Loading benchmark from {benchmark_file}")
    try:
        # Load benchmark data
        benchmark_df = pd.read_csv(benchmark_file, dtype={'hash': str})
        
        # Check if benchmark has required columns
        if not all(col in benchmark_df.columns for col in ['rank', 'hash', 'output_file']):
            print("Error: Benchmark file has invalid format. Missing required columns.")
            return
        
        if len(benchmark_df) < 2:
            print("Error: Benchmark contains fewer than 2 samples. Need at least 2 for evaluation.")
            return
        
        # Load outputs
        print_v("Loading simulation outputs...")
        outputs = []
        hashes = []
        total_outputs = len(benchmark_df)
        for i, (_, row) in enumerate(benchmark_df.iterrows()):
            hash_val = row['hash']
            hashes.append(hash_val)
            output_path = Path(out_paths["benchmark"]) / hash_val
            try:
                # Show progress
                if verbose:
                    progress = f"[{i+1}/{total_outputs}]"
                    print(f"{progress} Loading output for hash {hash_val}...", end="\r")
                
                output = simulator.load_output(str(output_path))
                outputs.append(output)
            except Exception as e:
                print(f"\nError loading output for hash {hash_val}: {e}")
                return
        
        print_v(f"Loaded all {total_outputs} outputs successfully.            ")
        
        # Get ranks from benchmark
        benchmark_ranks = benchmark_df['rank'].tolist()
        
        # Get scores from rewarder
        print_v("Ranking using the rewarder...")
        rewarder_scores = rewarder.rank(outputs)
        
        # Create results dataframe for display
        results_df = pd.DataFrame({
            'Hash': hashes,
            'Benchmark Rank': benchmark_ranks,
            'Rewarder Score': rewarder_scores
        })
        
        # Add ranking by rewarder scores (highest score = rank 1)
        rewarder_ranking = [-score for score in rewarder_scores]  # Convert to "lower is better" for ranking
        results_df['Rewarder Ranking'] = pd.Series(rewarder_ranking).rank(method='min').map(int)
        
        # Calculate ranking error (difference between benchmark rank and rewarder ranking)
        results_df['Rank Error'] = abs(results_df['Benchmark Rank'] - results_df['Rewarder Ranking'])
        avg_rank_error = results_df['Rank Error'].mean()
        
        # Calculate Kendall Tau correlation
        kendall_tau, _ = kendalltau(results_df['Benchmark Rank'], results_df['Rewarder Ranking'])

        # Calculate precision for top 3 and bottom 3
        top_3_rewarder_hashes = results_df.sort_values(by="Rewarder Ranking")["Hash"].head(3)
        bottom_3_rewarder_hashes = results_df.sort_values(by="Rewarder Ranking")["Hash"].tail(3)

        top_3_precision = results_df["Hash"].head(3).isin(top_3_rewarder_hashes).sum() / 3
        bottom_3_precision = results_df["Hash"].tail(3).isin(bottom_3_rewarder_hashes).sum() / 3

        # Sort by benchmark rank for first display
        benchmark_sorted_df = results_df.sort_values('Benchmark Rank')
        benchmark_sorted_df = benchmark_sorted_df[['Hash', 'Benchmark Rank', 'Rewarder Ranking', 'Rewarder Score', 'Rank Error']]

        pair_wise_accuracy = 0
        for i in range(len(benchmark_sorted_df)):
            for j in range(i+1, len(benchmark_sorted_df)):
                benchmark_i = benchmark_sorted_df.iloc[i]["Benchmark Rank"]
                benchmark_j = benchmark_sorted_df.iloc[j]["Benchmark Rank"]
                rewarder_i = benchmark_sorted_df.iloc[i]["Rewarder Ranking"]
                rewarder_j = benchmark_sorted_df.iloc[j]["Rewarder Ranking"]
                
                # If the ordering is the same in both rankings +1 pt
                if (benchmark_i < benchmark_j and rewarder_i < rewarder_j) or \
                   (benchmark_i > benchmark_j and rewarder_i > rewarder_j):
                    pair_wise_accuracy += 1
        pair_wise_accuracy /= (len(benchmark_sorted_df) * (len(benchmark_sorted_df) - 1) / 2)
        
        
        # Display results
        print_v("\n===== REWARDER EVALUATION RESULTS =====")
        print_v(f"{pair_wise_accuracy:>6.2f}  Pair-wise Accuracy [0 to 1, > 0.9 is good]")
        print_v(f"{avg_rank_error:>6.2f}  Average Rank Error")
        print_v(f"{kendall_tau:>+6.2f}  Kendall Tau Correlation [-1 to 1]")
        print_v(f"{top_3_precision:>6.2f}  Top 3 Precision [0 to 1]")
        print_v(f"{bottom_3_precision:>6.2f}  Bottom 3 Precision [0 to 1]")
        
        print_v("\nResults sorted by Benchmark Rank:")
        print_v(benchmark_sorted_df.to_string(index=False))
        print_v("="*100)
        return pair_wise_accuracy
        
    except Exception as e:
        print(f"Error testing rewarder on benchmark: {e}")
        traceback.print_exc()

def launch_benchmarker(simulator: Simulator, generator: Generator, rewarder: Rewarder, out_paths: dict, frame_size: tuple = (300, 300)) -> None:
    """
    Launch the benchmarker. Handles CLI evaluation or launches the GUI.

    Args:
        simulator: The simulator to use
        generator: The generator to use
        rewarder: The rewarder to use
        out_paths: The paths to the outputs
        frame_size: Tuple of (width, height) for video frames
    """
    print("\nBenchmark Options:")
    print("1. Make a Live Benchmark (needs GUI)")
    print("2. Create a New Benchmark (needs GUI)")
    print("3. Test Rewarder on default benchmark")
    print("0. Exit")
    choice = input("Choose an option: ")

    if choice == '1':
        number_of_videos = input("Enter the number of simulations that will be used: ")
        number_of_videos = int(number_of_videos)

        print("Launching Live Benchmark GUI...")
        root = tk.Tk()
        sv_ttk.set_theme("dark")
        root.withdraw() # Hide the root window
        def close_handler():
            # print("Closing hidden root window")
            root.destroy()
        live_app = LiveBenchmarkApp(
            tk.Toplevel(root),
            number_of_videos,
            simulator,
            generator,
            rewarder,
            out_paths,
            frame_size,
            on_close=close_handler
        )
        root.mainloop() # Blocks until the Toplevel window (and hidden root) is closed
    elif choice == '2':
        print("Launching Create Benchmark GUI...")
        root = tk.Tk()
        sv_ttk.set_theme("dark")
        root.withdraw() # Hide the root window
        def close_handler():
            # print("Closing hidden root window")
            root.destroy()
        create_app = CreateBenchmarkApp(
            tk.Toplevel(root),
            simulator,
            generator,
            rewarder,
            out_paths,
            frame_size,
            on_close=close_handler
        )
        root.mainloop() # Blocks until the Toplevel window (and hidden root) is closed
    elif choice == '3':
        print("Testing rewarder on benchmark...")
        test_rewarder_on_benchmark(simulator, rewarder, out_paths)
    elif choice == '0':
        print("Exiting benchmark tool.")
    else:
        print("Invalid choice. Please try again.")
