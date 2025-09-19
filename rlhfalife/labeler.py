import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
from tkinter import ttk
import threading
from rlhfalife.utils import Simulator
from rlhfalife.data_managers import DatasetManager, PairsManager

def launch_video_labeler(simulator: Simulator, dataset_manager: DatasetManager, pairs_manager: PairsManager, verbose: bool = False, frame_size: tuple = (300, 300)) -> None:
    """
    Launch the video labeler app.
    
    Args:
        simulator: Simulator object that will be used for generating pairs
        dataset_manager: DatasetManager instance for storing simulation data
        pairs_manager: PairsManager instance for storing pairs
        verbose: Whether to print verbose output
        frame_size: Tuple of (width, height) for video frames
    
    Note: executes in the main thread.
    """
    root = tk.Tk()
    app = VideoLabelerApp(root, simulator, dataset_manager, pairs_manager, verbose, frame_size)
    root.protocol("WM_DELETE_WINDOW", app.save_and_exit)
    root.mainloop()

class VideoLabelerApp:
    def __init__(self, master: tk.Tk, simulator: Simulator, dataset_manager: DatasetManager, 
                 pairs_manager: PairsManager, verbose: bool = False, frame_size: tuple = (300, 300)) -> None:
        """
        Initialize the video labeler app

        Args:
            master: The master window
            simulator: The simulator to use
            dataset_manager: DatasetManager instance for storing simulation data
            pairs_manager: PairsManager instance for storing pairs
            verbose: Whether to print verbose output
            frame_size: Tuple of (width, height) for video frames
        """
        self.simulator = simulator
        self.dataset_manager = dataset_manager
        self.pairs_manager = pairs_manager
        self.verbose = verbose
        self.frame_size = frame_size

        self.is_playing = True
        self.master = master
        self.after_id = None
        self.cap1 = None
        self.cap2 = None

        self.master.title("Video Labeler")
        self.create_widgets()
        self.bind_keys()
        self.master.lift()
        self.master.focus_force()
        self.master.attributes('-topmost', True)
        self.master.attributes('-topmost', False)
        self.update_progress_percentage()
        self.load_next_videos()

    def create_widgets(self):
        self.video_frame = tk.Frame(self.master)
        self.video_frame.pack()

        self.left_video_label = tk.Label(self.video_frame)
        self.left_video_label.pack(side="left")

        self.right_video_label = tk.Label(self.video_frame)
        self.right_video_label.pack(side="right")

        # Create a new frame for progress bars
        self.progress_frame = tk.Frame(self.master)
        self.progress_frame.pack(pady=5)

        # Add progress bars to the new frame
        self.progress = ttk.Progressbar(self.progress_frame, orient="horizontal", length=200, mode="determinate")
        self.progress.pack(padx=5)
        self.progress['value'] = 0
        self.progress.bind("<Button-1>", self.on_progress_click)

        # Add play/pause button
        self.play_pause_button = tk.Button(self.progress_frame, text="Play/Pause", command=self.toggle_play_pause)
        self.play_pause_button.pack(side="left", padx=5)

        # Add pair info label
        self.pair_info_frame = tk.Frame(self.master)
        self.pair_info_frame.pack(pady=5)
        self.pair_info_label = tk.Label(self.pair_info_frame, text="Pair 0 of 0")
        self.pair_info_label.pack()

        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack()

        self.left_button = tk.Button(self.button_frame, text="Left Wins", command=self.left_wins)
        self.left_button.pack(side="left", padx=2)

        self.equal_button = tk.Button(self.button_frame, text="Equal", command=self.declare_equal)
        self.equal_button.pack(side="left", padx=2)
        
        self.right_button = tk.Button(self.button_frame, text="Right Wins", command=self.right_wins)
        self.right_button.pack(side="left", padx=2)

        self.generate_button = tk.Button(self.button_frame, text="Generate New Pairs", command=self.generate_new_pairs_dialog)
        self.generate_button.pack(side="left", padx=5)

        # Add reset & regenerate button
        self.reset_frame = tk.Frame(self.master)
        self.reset_frame.pack(pady=5)
        self.reset_button = tk.Button(self.reset_frame, text="Reset & Regenerate", command=self.reset_and_regenerate)
        self.reset_button.pack()

        # Add quit button frame at the bottom
        self.quit_frame = tk.Frame(self.master)
        self.quit_frame.pack(pady=10, fill=tk.X)
        
        # Add progress percentage label
        self.progress_label = tk.Label(self.quit_frame, text="Progress: 0% (0/0)")
        self.progress_label.pack(side="left", padx=10)
        
        self.quit_button = tk.Button(self.quit_frame, text="Quit", command=self.save_and_exit)
        self.quit_button.pack(side="right", padx=10)

        # Create a new frame for keybindings
        self.keybindings_frame = tk.Frame(self.master)
        self.keybindings_frame.pack(side="left", padx=10, pady=10)

        # Add a label to display keybindings
        self.keybindings_label = tk.Label(self.keybindings_frame, text="Keybindings:\nLeft Arrow: Left Wins\nRight Arrow: Right Wins\nDown Arrow: Equal\nSpace: Restart Videos\nBackspace: Previous Pair", justify="left")
        self.keybindings_label.pack()

    def toggle_play_pause(self):
        """Toggle the playing state of the videos."""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.update_frames()

    def bind_keys(self):
        self.master.bind('<Left>', lambda event: self.left_wins())
        self.master.bind('<Right>', lambda event: self.right_wins())
        self.master.bind('<Down>', lambda event: self.declare_equal())
        self.master.bind('<space>', lambda event: self.restart_videos())
        self.master.bind('<BackSpace>', lambda event: self.previous_pair())

    def load_next_videos(self, undo: bool = False):
        """Display the current pair of videos."""
        if undo:
            self.hash1, self.hash2 = self.pairs_manager.get_last_ranked_pair()
        else:
            self.hash1, self.hash2 = self.pairs_manager.get_next_unranked_pair()
        if self.hash1 is None or self.hash2 is None:
            self.prompt_generate_new_pairs()
            return

        # Get video paths from dataset manager
        video_path1, video_path2 = self.dataset_manager.get_video_paths([self.hash1, self.hash2])
        
        if not video_path1 or not video_path2:
            messagebox.showerror("Error", f"Video files not found for pair {self.hash1}, {self.hash2}")
            self.load_next_videos()
            return
        
        # Open the videos
        self.cap1 = cv2.VideoCapture(video_path1)
        self.cap2 = cv2.VideoCapture(video_path2)
        
        if not self.cap1.isOpened() or not self.cap2.isOpened():
            messagebox.showerror("Error", f"Failed to open video files {video_path1}, {video_path2}")
            return
        
        # Update the pair info label
        self.pair_info_label.config(text=f"Pair {self.pairs_manager.get_nb_ranked_pairs()} of {self.pairs_manager.get_nb_pairs()}: {self.hash1} vs {self.hash2}")
        
        self.progress['value'] = 0
        self.progress['maximum'] = self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)

        # Start playing the videos
        self.play_videos()

    def _update_gui_frames(self, frame1, frame2):
        """Helper to update the GUI with new frames."""
        # Convert frames from BGR to RGB
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        
        # Resize frames if needed
        frame1 = cv2.resize(frame1, self.frame_size)
        frame2 = cv2.resize(frame2, self.frame_size)
        
        # Convert to PIL Image
        img1 = Image.fromarray(frame1)
        img2 = Image.fromarray(frame2)
        
        # Convert to PhotoImage
        photo1 = ImageTk.PhotoImage(image=img1)
        photo2 = ImageTk.PhotoImage(image=img2)
        
        # Update labels
        self.left_video_label.config(image=photo1)
        self.left_video_label.image = photo1
        self.right_video_label.config(image=photo2)
        self.right_video_label.image = photo2
        
        self.progress['value'] = self.cap1.get(cv2.CAP_PROP_POS_FRAMES)

    def prompt_generate_new_pairs(self):
        """Prompt the user to generate new pairs."""
        num_pairs = simpledialog.askinteger("Generate Pairs", "No more pairs available. How many new simulations to run ?", minvalue=1, parent=self.master)
        if num_pairs:
            self.generate_new_pairs(num_pairs)

    def generate_new_pairs(self, num_pairs):
        """Generate new pairs of simulations."""
        loading_screen = LoadingScreen(self.master)
        loading_screen.run_generation_process(
            self.simulator, 
            num_pairs, 
            self.dataset_manager, 
            self.pairs_manager,
            self.verbose,
            on_complete=lambda: self.on_generation_complete(loading_screen),
            on_error=lambda: self.on_generation_error(loading_screen)
        )

    def on_generation_complete(self, loading_screen):
        """Called when generation is complete"""
        loading_screen.close()
        self.update_progress_percentage()
        self.load_next_videos()
        
    def on_generation_error(self, loading_screen):
        """Handle errors during generation"""
        loading_screen.close()
        # Do nothing here - the error dialog will handle closing the application
        pass

    def play_videos(self):
        """Start playing the videos."""
        self.update_frames()

    def update_frames(self):
        """Update the video frames."""
        if not self.is_playing:
            return
        
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()

        if not ret1:
            if self.verbose:
                print("Replaying first video.")
            self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret1, frame1 = self.cap1.read()

        if not ret2:
            if self.verbose:
                print("Replaying second video.")
            self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret2, frame2 = self.cap2.read()

        if ret1 and ret2:
            self._update_gui_frames(frame1, frame2)

            # Schedule the next frame update
            self.after_id = self.master.after(33, self.update_frames)  # ~30 fps
        else:
            if self.verbose:
                print("Error reading frames.")
            self.restart_videos()

    def on_progress_click(self, event):
        """Handle clicks on the progress bar to seek the video."""
        if self.cap1 is None:
            return

        clicked_x = event.x
        width = self.progress.winfo_width()
        if width == 0:
            return
        progress_percentage = clicked_x / width
        total_frames = self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)
        new_frame = int(total_frames * progress_percentage)
        
        self.set_frame(new_frame)

    def set_frame(self, frame_number):
        """Set the videos to a specific frame."""
        if self.cap1 is None or self.cap2 is None:
            return
            
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()

        if ret1 and ret2:
            self._update_gui_frames(frame1, frame2)
        else:
            if self.verbose:
                print("Error reading frames during seek.")

    def left_wins(self):
        self.record_winner('left')

    def right_wins(self):
        self.record_winner('right')

    def declare_equal(self):
        self.record_winner('equal')

    def record_winner(self, winner):
        """Record the winner of the current pair."""
        # Release video resources
        if hasattr(self, 'cap1') and self.cap1 is not None:
            self.cap1.release()
            self.cap1 = None
        if hasattr(self, 'cap2') and self.cap2 is not None:
            self.cap2.release()
            self.cap2 = None
        
        # Cancel any pending frame updates
        if self.after_id:
            self.master.after_cancel(self.after_id)
            self.after_id = None
        
        # Record the winner
        if winner == 'left':
            winner_val = 0
        elif winner == 'right':
            winner_val = 1
        else: # equal
            winner_val = 0.5
        self.pairs_manager.set_winner(self.hash1, self.hash2, winner_val)

        self.load_next_videos()
        self.update_progress_percentage()

    def update_progress_percentage(self):
        """Update the progress percentage label."""
        nb_pairs = self.pairs_manager.get_nb_pairs()
        nb_ranked_pairs = self.pairs_manager.get_nb_ranked_pairs()
        self.progress_label.config(text=f"Progress: {((nb_ranked_pairs) / nb_pairs) * 100 if nb_pairs > 0 else 0:.1f}% ({nb_ranked_pairs}/{nb_pairs}). Number of simulations: {len(self.dataset_manager)}")

    def save_and_exit(self):
        # Release video resources if they exist
        if self.cap1 is not None:
            self.cap1.release()
        if self.cap2 is not None:
            self.cap2.release()
        
        # Cancel any pending after callbacks
        if self.after_id is not None:
            self.master.after_cancel(self.after_id)
        
        # Save the pairs manager
        self.pairs_manager.save()

        # Save the dataset manager
        self.dataset_manager.save()

        # Destroy the window and quit
        self.master.destroy()
        self.master.quit()

    def restart_videos(self):
        if self.cap1 is not None:
            self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if self.cap2 is not None:
            self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def previous_pair(self):
        # Release video resources
        if self.cap1 is not None:
            self.cap1.release()
            self.cap1 = None
        if self.cap2 is not None:
            self.cap2.release()
            self.cap2 = None
        
        # Cancel any pending frame updates
        if self.after_id:
            self.master.after_cancel(self.after_id)
            self.after_id = None

        self.load_next_videos(undo=True)
        self.update_progress_percentage()

    def reset_and_regenerate(self):
        """Reset all pairs and regenerate new ones."""
        # Confirm with the user
        result = messagebox.askyesno("Confirm Reset", 
                                     "Are you sure you want to reset all pairs and files?\n"
                                     "This will delete all existing simulations and videos.")
        if not result:
            return
            
        # Release video resources
        if self.cap1 is not None:
            self.cap1.release()
            self.cap1 = None
        if self.cap2 is not None:
            self.cap2.release()
            self.cap2 = None
            
        # Cancel any pending frame updates
        if self.after_id:
            self.master.after_cancel(self.after_id)
            self.after_id = None
            
        # Reset dataset and pairs
        self.dataset_manager.reset()
        self.pairs_manager.reset()
        
        # Prompt for number of new pairs
        num_pairs = simpledialog.askinteger("Generate Pairs", 
                                           "How many new pairs would you like to generate?", 
                                           minvalue=1, 
                                           initialvalue=5,
                                           parent=self.master)
        if num_pairs:
            self.generate_new_pairs(num_pairs)
        else:
            # If user cancels, just update the UI
            self.update_progress_percentage()
            self.load_next_videos()

    def generate_new_pairs_dialog(self):
        """Prompt user to generate new pairs without resetting existing ones."""
        num_pairs = simpledialog.askinteger("Generate New Pairs", 
                                           "How many new pairs would you like to generate?", 
                                           minvalue=1, 
                                           initialvalue=5,
                                           parent=self.master)
        if num_pairs:
            self.generate_new_pairs(num_pairs)

    def show_error_dialog(self, debug_info):
        """Show a detailed error dialog with debug information."""
        error_window = tk.Toplevel(self.master)
        error_window.title("Error Details")
        error_window.geometry("800x600")
        
        # Make it modal
        error_window.transient(self.master)
        error_window.grab_set()
        
        # Add a frame for the content
        frame = ttk.Frame(error_window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Add a label at the top
        ttk.Label(frame, text="An error occurred during pair generation. Details below:", 
                 wraplength=780).pack(pady=(0, 10))
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=text_widget.yview)
        
        # Insert the debug information
        text_widget.insert(tk.END, debug_info)
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        # Add buttons at the bottom
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=10)
        
        # Copy button
        def copy_to_clipboard():
            error_window.clipboard_clear()
            error_window.clipboard_append(debug_info)
            error_window.update()
        
        ttk.Button(button_frame, text="Copy to Clipboard", 
                  command=copy_to_clipboard).pack(side=tk.LEFT, padx=5)
        
        # Close button that will exit the application when clicked
        def close_and_exit():
            error_window.destroy()
            self.master.quit()
            self.master.destroy()
            
        ttk.Button(button_frame, text="Close", 
                  command=close_and_exit).pack(side=tk.LEFT, padx=5)

class LoadingScreen:
    def __init__(self, master, title="Generating Pairs"):
        self.window = tk.Toplevel(master)
        self.window.title(title)
        self.window.geometry("300x200")  # Slightly taller to accommodate the animation
        self.window.resizable(False, False)
        
        # Make it modal
        self.window.transient(master)
        self.window.grab_set()
        
        # Center the window
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        
        # Create a frame for the content
        self.frame = ttk.Frame(self.window, padding=20)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Add a simple animation (spinning dots)
        self.animation_frame = ttk.Frame(self.frame)
        self.animation_frame.pack(pady=(0, 10))
        
        self.dots = []
        for i in range(5):
            dot = ttk.Label(self.animation_frame, text="●", font=("Arial", 14))
            dot.grid(row=0, column=i, padx=3)
            self.dots.append(dot)
        
        # Add a label for the status
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(self.frame, textvariable=self.status_var)
        self.status_label.pack(pady=(0, 10))
        
        # Add a progress bar
        self.progress = ttk.Progressbar(self.frame, orient=tk.HORIZONTAL, length=250, mode="indeterminate")
        self.progress.pack(pady=10)
        self.progress.start(10)
        
        self.total_steps = None
        self.current_step = 0
        self.master = master
        
        # Start the animation
        self.animation_index = 0
        self.animate()
        
    def animate(self):
        """Animate the dots"""
        if not hasattr(self, 'window') or not self.window.winfo_exists():
            return  # Stop animation if window is closed
            
        # Reset all dots to default color
        for dot in self.dots:
            dot.configure(foreground="gray")
            
        # Highlight the current dot
        self.dots[self.animation_index].configure(foreground="blue")
        
        # Move to the next dot
        self.animation_index = (self.animation_index + 1) % len(self.dots)
        
        # Schedule the next animation frame
        self.animation_id = self.window.after(200, self.animate)
        
    def set_determinate_mode(self, total_steps):
        """Switch to determinate mode with a known number of steps"""
        self.total_steps = total_steps
        self.current_step = 0
        self.progress.stop()
        self.progress.configure(mode="determinate", maximum=total_steps, value=0)
        
    def increment_progress(self):
        """Increment the progress bar by one step"""
        if self.total_steps is not None:
            self.current_step += 1
            self.progress.configure(value=self.current_step)
            self.window.update()
        
    def update_status(self, message):
        """Update the status message in the loading screen"""
        self.status_var.set(message)
        self.window.update()
        
    def close(self):
        """Close the loading screen"""
        if hasattr(self, 'animation_id'):
            self.window.after_cancel(self.animation_id)
        self.window.grab_release()
        self.window.destroy()
        
    def run_generation_process(self, simulator, num_pairs, dataset_manager, pairs_manager, 
                              verbose=False, on_complete=None, on_error=None):
        """
        Run the generation process in a separate thread.
        
        Args:
            simulator: The simulator to use
            num_pairs: Number of pairs to generate
            dataset_manager: DatasetManager instance for storing simulation data
            pairs_manager: PairsManager instance for storing pairs
            verbose: Whether to print verbose output
            on_complete: Callback function to call when generation is complete
            on_error: Callback function to call when an error occurs
        """
        self.update_status("Initializing...")
        
        def update_callback(message):
            self.update_status(message)
            return True  # Continue processing
        
        def run_generation():
            try:
                simulator.generate_pairs(
                    num_pairs, 
                    dataset_manager, 
                    pairs_manager, 
                    verbose=verbose, 
                    progress_callback=update_callback
                )
                if on_complete:
                    self.master.after(0, on_complete)
            except Exception as e:
                import traceback
                import sys
                
                # Get the full traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                
                # Get simulator state information
                simulator_info = f"Simulator Configuration:\n"
                simulator_info += f"- Type: {type(simulator).__name__}\n"
                for attr in dir(simulator):
                    if not attr.startswith('_'):  # Only show public attributes
                        try:
                            value = getattr(simulator, attr)
                            if not callable(value):  # Skip methods
                                simulator_info += f"- {attr}: {value}\n"
                        except Exception:
                            pass  # Skip attributes that can't be accessed
                
                # Get dataset and pairs manager state
                dataset_info = f"\nDataset Manager State:\n"
                dataset_info += f"- Total simulations: {len(dataset_manager)}\n"
                pairs_info = f"\nPairs Manager State:\n"
                pairs_info += f"- Total pairs: {pairs_manager.get_nb_pairs()}\n"
                pairs_info += f"- Ranked pairs: {pairs_manager.get_nb_ranked_pairs()}\n"
                
                # Combine all debug information
                debug_info = (
                    f"Error Details:\n"
                    f"{'='*50}\n"
                    f"{str(e)}\n\n"
                    f"Full Traceback:\n"
                    f"{'='*50}\n"
                    f"{tb_str}\n\n"
                    f"State Information:\n"
                    f"{'='*50}\n"
                    f"{simulator_info}\n"
                    f"{dataset_info}\n"
                    f"{pairs_info}"
                )
                
                if on_error:
                    self.master.after(0, lambda: self.show_error_dialog(debug_info))
                    self.master.after(0, lambda: on_error())
        
        threading.Thread(target=run_generation).start()
    
    def show_error_dialog(self, debug_info):
        """Show a detailed error dialog with debug information."""
        error_window = tk.Toplevel(self.master)
        error_window.title("Error Details")
        error_window.geometry("800x600")
        
        # Make it modal
        error_window.transient(self.master)
        error_window.grab_set()
        
        # Add a frame for the content
        frame = ttk.Frame(error_window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Add a label at the top
        ttk.Label(frame, text="An error occurred during pair generation. Details below:", 
                 wraplength=780).pack(pady=(0, 10))
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=text_widget.yview)
        
        # Insert the debug information
        text_widget.insert(tk.END, debug_info)
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        # Add buttons at the bottom
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=10)
        
        # Copy button
        def copy_to_clipboard():
            error_window.clipboard_clear()
            error_window.clipboard_append(debug_info)
            error_window.update()
        
        ttk.Button(button_frame, text="Copy to Clipboard", 
                  command=copy_to_clipboard).pack(side=tk.LEFT, padx=5)
        
        # Close button that will exit the application when clicked
        def close_and_exit():
            error_window.destroy()
            self.master.quit()
            self.master.destroy()
            
        ttk.Button(button_frame, text="Close", 
                  command=close_and_exit).pack(side=tk.LEFT, padx=5)
