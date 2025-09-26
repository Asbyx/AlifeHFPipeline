import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
from tkinter import ttk
import numpy as np
import threading
from collections import Counter
from rlhfalife.utils import Simulator
from rlhfalife.data_managers import DatasetManager, PairsManager
import sv_ttk

def launch_quad_labeler(simulator: Simulator, dataset_manager: DatasetManager, pairs_manager: PairsManager, verbose: bool = False, frame_size: tuple = (300, 300)) -> None:
    """
    Launch the quad labeler app that shows 4 videos for ranking.
    
    Args:
        simulator: Simulator object that will be used for generating quadruples
        dataset_manager: DatasetManager instance for storing simulation data
        pairs_manager: PairsManager instance for storing pairs
        verbose: Whether to print verbose output
        frame_size: Tuple of (width, height) for video frames
    
    Note: executes in the main thread.
    """
    root = tk.Tk()
    sv_ttk.set_theme("dark")
    app = QuadLabelerApp(root, simulator, dataset_manager, pairs_manager, verbose, frame_size)
    root.protocol("WM_DELETE_WINDOW", app.save_and_exit)
    root.mainloop()

class DraggableVideo(ttk.Frame):
    """A draggable video widget that can be reordered via drag and drop."""
    
    def __init__(self, parent, video_path, index, frame_size, hash_value, on_drag_start=None, on_drag_release=None, on_drag_motion=None):
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
        """
        super().__init__(parent)
        self.config(relief=tk.RAISED)
        self.parent = parent
        self.index = index
        self.frame_size = frame_size
        self.video_path = video_path
        self.hash_value = hash_value
        self.on_drag_start = on_drag_start
        self.on_drag_release = on_drag_release
        self.on_drag_motion = on_drag_motion
        self.is_dragging = False

        self.is_playing = True
        
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
        
        # Bind events for dragging
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<B1-Motion>", self.on_motion)
        self.video_label.bind("<ButtonPress-1>", self.on_press)
        self.video_label.bind("<ButtonRelease-1>", self.on_release)
        self.video_label.bind("<B1-Motion>", self.on_motion)
        self.rank_label.bind("<ButtonPress-1>", self.on_press)
        self.rank_label.bind("<ButtonRelease-1>", self.on_release)
        self.rank_label.bind("<B1-Motion>", self.on_motion)
        self.hash_label.bind("<ButtonPress-1>", self.on_press)
        self.hash_label.bind("<ButtonRelease-1>", self.on_release)
        self.hash_label.bind("<B1-Motion>", self.on_motion)
        
        # Track mouse offset for dragging
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        # Start playing the video
        self.after_id = None
        self.update_frame()
    
    def _display_frame(self, frame):
        """Helper to display a frame."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, self.frame_size)
        img = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=photo)
        self.video_label.image = photo
    
    def update_frame(self):
        """Update the video frame."""
        if not self.is_playing:
            if self.after_id:
                self.after_cancel(self.after_id)
                self.after_id = None
            return
        
        ret, frame = self.cap.read()
        
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            
        if ret:
            self._display_frame(frame)
            
        self.after_id = self.after(33, self.update_frame)  # ~30 fps
    
    def next_frame(self):
        """Go to the next frame."""
        if not self.cap.isOpened():
            return
        current_frame_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if current_frame_pos < total_frames:
             ret, frame = self.cap.read()
             if ret:
                self._display_frame(frame)

    def prev_frame(self):
        """Go to the previous frame."""
        if not self.cap.isOpened():
            return
        current_frame_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        new_pos = current_frame_pos - 2
        if new_pos >= 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            ret, frame = self.cap.read()
            if ret:
                self._display_frame(frame)

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
        """Handle mouse motion event."""
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
        self.is_playing = False
        if self.after_id:
            self.after_cancel(self.after_id)
        if self.cap and self.cap.isOpened():
            self.cap.release()

class QuadLabelerApp:
    def __init__(self, master: tk.Tk, simulator: Simulator, dataset_manager: DatasetManager, 
                 pairs_manager: PairsManager, verbose: bool = False, frame_size: tuple = (300, 300)) -> None:
        """
        Initialize the quad labeler app.
        
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
        
        self.master = master
        self.master.title("Quad Video Labeler")
        
        # Store currently displayed videos
        self.current_videos = []
        self.current_hashes = []
        self.video_widgets = []
        self.ignored_quads = set()
        self.history = []
        
        # For progress bar
        self.progress = None
        self.progress_after_id = None
        
        # Relationships between adjacent videos (e.g., '<' or '=')
        # For 4 videos, there are 3 relationships. Default to '<'.
        self.relationships = ['<'] * 3 
        self.relationship_button_frames = [] # Frames holding the < and = buttons
        self.relationship_buttons_widgets = [] # Stores (less_btn, equal_btn) tuples

        # Create widgets
        self.create_widgets()
        self.bind_keys()
        
        # Set window properties
        self.master.lift()
        self.master.focus_force()
        self.master.attributes('-topmost', True)
        self.master.attributes('-topmost', False)
        
        # Track dragging state
        self.drag_source = None
        self.drag_over_widget = None
        self.highlight_widget = None
        
        # Update UI and load videos
        self.update_progress_percentage()
        self.load_next_videos()
    
    def create_widgets(self):
        """Create the main UI widgets."""
        # Create title and instructions
        self.title_frame = ttk.Frame(self.master)
        self.title_frame.pack(pady=10)
        
        title_label = ttk.Label(self.title_frame, 
                              text="Drag videos to order. Use buttons (<, =) to set relationships.",
                              font=("Arial", 14))
        title_label.pack()
        
        # Create main frame for videos
        self.videos_frame = ttk.Frame(self.master)
        self.videos_frame.pack(padx=20, pady=10)
        
        # Create a new frame for video controls
        self.video_controls_frame = ttk.Frame(self.master)
        self.video_controls_frame.pack(pady=5)

        buttons_control_frame = ttk.Frame(self.video_controls_frame)
        buttons_control_frame.pack()

        self.prev_frame_button = ttk.Button(buttons_control_frame, text="< Prev Frame", command=self.prev_frame)
        self.prev_frame_button.pack(side=tk.LEFT, padx=5)

        self.play_pause_button = ttk.Button(buttons_control_frame, text="Play/Pause", command=self.toggle_play_pause)
        self.play_pause_button.pack(side=tk.LEFT, padx=5)

        self.next_frame_button = ttk.Button(buttons_control_frame, text="Next Frame >", command=self.next_frame)
        self.next_frame_button.pack(side=tk.LEFT, padx=5)

        self.progress = ttk.Progressbar(self.video_controls_frame, orient="horizontal", length=1200, mode="determinate")
        self.progress.pack(pady=5)
        self.progress.bind("<Button-1>", self.on_progress_click)
        
        # Create frame for action buttons
        self.button_frame = ttk.Frame(self.master)
        self.button_frame.pack(pady=15)
        
        # Submit button
        self.submit_button = ttk.Button(self.button_frame, text="Submit Ranking", 
                                      command=self.submit_ranking,
                                      style="Accent.TButton")
        self.submit_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Skip button
        self.skip_button = ttk.Button(self.button_frame, text="Skip Ranking",
                                     command=self.skip_ranking)
        self.skip_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Undo button
        self.undo_button = ttk.Button(self.button_frame, text="Undo",
                                     command=self.undo_ranking)
        self.undo_button.pack(side=tk.LEFT, padx=10, pady=5)

        # Add restart videos button
        self.restart_button = ttk.Button(self.button_frame, text="Restart Videos", 
                                      command=self.restart_videos)
        self.restart_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Add generate new videos button
        self.generate_button = ttk.Button(self.button_frame, text="Generate New Videos", 
                                        command=self.generate_new_videos_dialog)
        self.generate_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Create bottom frame for status and quit
        self.bottom_frame = ttk.Frame(self.master)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        # Create frame for keybindings
        self.key_frame = ttk.Frame(self.bottom_frame)
        self.key_frame.pack(side=tk.LEFT, padx=10)
        
        # Add keybindings label
        key_text_col1 = "Keybindings:\nEnter: Submit Ranking\nSpace: Restart Videos\nBackspace: Undo Ranking\nEsc: Quit"
        key_label_col1 = ttk.Label(self.key_frame, text=key_text_col1, justify=tk.LEFT)
        key_label_col1.pack(side=tk.LEFT, anchor='n')

        key_text_col2 = "\nLeft: Previous Frame\nRight: Next Frame\nP: Play/Pause\nS: Skip Ranking"
        key_label_col2 = ttk.Label(self.key_frame, text=key_text_col2, justify=tk.LEFT)
        key_label_col2.pack(side=tk.LEFT, anchor='n', padx=10)
        
        # Add progress percentage label
        self.progress_label = ttk.Label(self.bottom_frame, text="Progress: 0% (0/0)")
        self.progress_label.pack(side=tk.LEFT, padx=10)
        
        # Add quit button
        self.quit_button = ttk.Button(self.bottom_frame, text="Quit", 
                                    command=self.save_and_exit)
        self.quit_button.pack(side=tk.RIGHT, padx=10, pady=2)
    
    def bind_keys(self):
        """Bind keyboard shortcuts."""
        self.master.bind('<Return>', lambda event: self.submit_ranking())
        self.master.bind('s', lambda event: self.skip_ranking())
        self.master.bind('<BackSpace>', lambda event: self.undo_ranking())
        self.master.bind('<space>', lambda event: self.restart_videos())
        self.master.bind('<Escape>', lambda event: self.save_and_exit())
        self.master.bind('p', lambda event: self.toggle_play_pause())
        self.master.bind('<Left>', lambda event: self.prev_frame())
        self.master.bind('<Right>', lambda event: self.next_frame())
    
    def toggle_play_pause(self):
        """Toggle the playing state of the videos."""
        if not self.video_widgets:
            return

        is_currently_playing = self.video_widgets[0].is_playing
        
        for widget in self.video_widgets:
            widget.is_playing = not is_currently_playing
            if widget.is_playing:
                widget.update_frame()

        if not is_currently_playing: # If we just started playing
            self.update_progress_bar()
        else: # If we just paused
            if self.progress_after_id:
                self.master.after_cancel(self.progress_after_id)
                self.progress_after_id = None

    def update_progress_bar(self):
        """Update the progress bar to match the current video frame."""
        if self.video_widgets and self.video_widgets[0].is_playing:
            current_frame = self.video_widgets[0].cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.progress['value'] = current_frame
            self.progress_after_id = self.master.after(100, self.update_progress_bar)
        else:
            if self.progress_after_id:
                self.master.after_cancel(self.progress_after_id)
            self.progress_after_id = None

    def on_progress_click(self, event):
        """Handle clicks on the progress bar to seek the video."""
        if not self.video_widgets:
            return

        clicked_x = event.x
        width = self.progress.winfo_width()
        if width == 0:
            return
        progress_percentage = clicked_x / width
        total_frames = self.video_widgets[0].cap.get(cv2.CAP_PROP_FRAME_COUNT)
        new_frame = int(total_frames * progress_percentage)
        
        self.set_frame(new_frame)

    def set_frame(self, frame_number):
        """Set all videos to a specific frame."""
        if not self.video_widgets:
            return
        
        for widget in self.video_widgets:
            widget.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = widget.cap.read()
            if ret:
                widget._display_frame(frame)
        
        self.progress['value'] = frame_number

    def next_frame(self):
        """Advance all videos by one frame."""
        for widget in self.video_widgets:
            if widget.is_playing:
                widget.is_playing = False # Pause first
            widget.next_frame()

    def prev_frame(self):
        """Rewind all videos by one frame."""
        for widget in self.video_widgets:
            if widget.is_playing:
                widget.is_playing = False # Pause first
            widget.prev_frame()
    
    def load_next_videos(self):
        """Load the next set of 4 videos."""
        self.history.append(self.current_hashes)

        # Get 4 hashes that have the fewest ranks and are not ranked together
        all_hashes = self.dataset_manager.get_all_hashes()        
        ranked_pairs_df = self.pairs_manager._get_ranked_pairs() # Renamed for clarity
        hash_rankings = Counter()
        
        # Create a set of tuples for faster lookup of existing pairs
        existing_pairs_set = set()
        if not ranked_pairs_df.empty:
            for _, row in ranked_pairs_df.iterrows():
                # Store pairs lexicographically to avoid (h1, h2) vs (h2, h1) issues
                pair = tuple(sorted((row['hash1'], row['hash2'])))
                existing_pairs_set.add(pair)
                hash_rankings[row['hash1']] += 1
                hash_rankings[row['hash2']] += 1
        
        for h in all_hashes:
            if h not in hash_rankings:
                hash_rankings[h] = 0
        
        def _find_next_quad(nodes, path, ignored, existing_pairs): # DFS to find a quad that is not ranked together or ignored
            if len(path) == 4:
                sorted_path = tuple(sorted(path))
                if sorted_path not in ignored:
                    return path
                return None

            # Try to extend the current path
            for i, node in enumerate(nodes):
                # Check compatibility with current path
                is_compatible = all(tuple(sorted((node, p_node))) not in existing_pairs for p_node in path)
                
                if is_compatible:
                    # Recurse with remaining nodes
                    result = _find_next_quad(nodes[i+1:], path + [node], ignored, existing_pairs)
                    if result:
                        return result
            return None

        all_hashes_sorted = [h for h, _ in sorted(hash_rankings.items(), key=lambda x: x[1])]
        
        best_hashes = _find_next_quad(all_hashes_sorted, [], self.ignored_quads, existing_pairs_set)

        if best_hashes is None:
            # If no quad is found, set it to an empty list to trigger video generation prompt
            best_hashes = []
        
        if len(best_hashes) < 4:
            self.prompt_generate_new_videos()
            return
        
        self.current_hashes = best_hashes
        self.relationships = ['<'] * (len(self.current_hashes) -1) # Reset relationships

        # Clear any existing video widgets
        self.clear_video_widgets()
        
        # Get the video paths
        video_paths = self.dataset_manager.get_video_paths(self.current_hashes)
        
        # Check if all videos exist
        if None in video_paths or not all(video_paths):
            messagebox.showerror("Error", "One or more video files not found")
            self.load_next_videos()
            return
        
        # Randomly permute the videos for initial display
        indices = np.random.permutation(4)
        self.current_hashes = [self.current_hashes[i] for i in indices]
        video_paths = [video_paths[i] for i in indices]
        
        # Create the video widgets
        self.create_video_widgets(video_paths)
        if self.video_widgets:
            total_frames = self.video_widgets[0].cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.progress['maximum'] = total_frames
            self.progress['value'] = 0
            if self.video_widgets[0].is_playing:
                self.update_progress_bar()
    
    
    def create_video_widgets(self, video_paths):
        """Create the draggable video widgets and relationship buttons."""
        # Side by side layout for videos and relationship buttons
        for i, path in enumerate(video_paths):
            # Create the draggable video widget
            video_widget = DraggableVideo(
                self.videos_frame, 
                path, 
                i, 
                self.frame_size,
                hash_value=self.current_hashes[i],  # Pass the hash value
                on_drag_start=self.start_drag,
                on_drag_release=self.end_drag,
                on_drag_motion=self.on_drag_motion
            )
            video_widget.pack(side=tk.LEFT, padx=5, pady=10)
            self.video_widgets.append(video_widget)

            # Add relationship button if this is not the last video
            if i < len(video_paths) - 1:
                btn_frame = ttk.Frame(self.videos_frame)
                btn_frame.pack(side=tk.LEFT, padx=5, pady=10, anchor='center')
                self.relationship_button_frames.append(btn_frame)

                # Single button that toggles relationship
                rel_button = ttk.Button(btn_frame, text=self.relationships[i], 
                                     command=lambda idx=i: self.toggle_relationship(idx))
                rel_button.pack(pady=2)
                self.relationship_buttons_widgets.append(rel_button)
        
        self.update_relationship_buttons_visuals() # Initial visual state

    def clear_video_widgets(self):
        """Clear all video widgets and relationship buttons, and release resources."""
        for widget in self.video_widgets:
            widget.release_resources()
            widget.destroy()
        self.video_widgets = []

        for frame in self.relationship_button_frames:
            frame.destroy()
        self.relationship_button_frames = []
        self.relationship_buttons_widgets = [] # Clear the stored button widgets
    
    def toggle_relationship(self, index):
        """Toggle the relationship between video[index] and video[index+1]."""
        if 0 <= index < len(self.relationships):
            current_relationship = self.relationships[index]
            new_relationship = '=' if current_relationship == '<' else '<'
            self.relationships[index] = new_relationship
            self.update_relationship_buttons_visuals()

    def set_relationship(self, index, rel_type):
        """Set the relationship between video[index] and video[index+1]."""
        if 0 <= index < len(self.relationships):
            self.relationships[index] = rel_type
            self.update_relationship_buttons_visuals()

    def update_relationship_buttons_visuals(self):
        """Update the visual state of relationship buttons (text)."""
        for i, rel_button in enumerate(self.relationship_buttons_widgets):
            if 0 <= i < len(self.relationships):
                 rel_button.config(text=self.relationships[i])

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
        
        # Clean up any highlighting
        if self.highlight_widget and self.highlight_widget != self.drag_source:
            self.highlight_widget.config(relief=tk.RAISED)
        
        self.highlight_widget = None
        
        if target_widget and target_widget != source_widget:
            # Swap the widgets in the display
            self.swap_videos(source_widget, target_widget)
    
    def swap_videos(self, widget1, widget2):
        """Swap two video widgets in the display and update relationships."""
        # Get their current indices in the video_widgets list
        index1 = self.video_widgets.index(widget1)
        index2 = self.video_widgets.index(widget2)
        
        # Swap the widgets in the list
        self.video_widgets[index1], self.video_widgets[index2] = self.video_widgets[index2], self.video_widgets[index1]
        
        # Swap the corresponding hashes
        self.current_hashes[index1], self.current_hashes[index2] = self.current_hashes[index2], self.current_hashes[index1]
        
        # Update their visual rank indicators (DraggableVideo.update_rank handles its own label)
        widget1.update_rank(index2) # widget1 is now at index2
        widget2.update_rank(index1) # widget2 is now at index1
        
        # Rearrange all video widgets and relationship button frames in the display
        # First, remove all existing items from videos_frame
        for child in self.videos_frame.winfo_children():
            child.pack_forget()
        
        # Repack videos and relationship buttons in the new order
        for i, video_widget in enumerate(self.video_widgets):
            video_widget.pack(side=tk.LEFT, padx=5, pady=10)
            # Repack relationship buttons if they exist for this position
            if i < len(self.relationship_button_frames):
                self.relationship_button_frames[i].pack(side=tk.LEFT, padx=5, pady=10, anchor='center')
        

        self.update_relationship_buttons_visuals()
    
    def submit_ranking(self):
        """Process the current video ranking and relationships to generate pairs."""
        if len(self.current_hashes) != 4 or len(self.relationships) != 3:
            messagebox.showerror("Error", "Need exactly 4 videos and 3 relationships defined.")
            return

        ranked_groups = []
        current_rank_group = 0
        
        # The first video is always in its own (initial) rank group
        if self.current_hashes:
            ranked_groups.append((self.current_hashes[0], current_rank_group))
        
        # Determine rank groups based on relationships
        for i in range(len(self.relationships)):
            hash_i_plus_1 = self.current_hashes[i+1]
            relationship = self.relationships[i]
            
            if relationship == '<':
                current_rank_group += 1
            
            ranked_groups.append((hash_i_plus_1, current_rank_group))

        for i in range(len(ranked_groups)):
            hash1, group1 = ranked_groups[i]
            for j in range(i + 1, len(ranked_groups)):
                hash2, group2 = ranked_groups[j]
                
                if group1 < group2: # hash1 is preferred over hash2
                    self.pairs_manager.set_winner(hash1, hash2, 0) # hash1 wins
                elif group1 == group2: # hash1 and hash2 are equal
                    self.pairs_manager.set_winner(hash1, hash2, 0.5) # Equal
        
        # Load the next set of videos
        self.load_next_videos()
        self.update_progress_percentage()
    
    def skip_ranking(self):
        """Add the current quad to the ignored list and load the next one."""
        if len(self.current_hashes) == 4:
            # Sort to ensure canonical representation
            ignored_hash_tuple = tuple(sorted(self.current_hashes))
            self.ignored_quads.add(ignored_hash_tuple)
            if self.verbose:
                print(f"Added {ignored_hash_tuple} to ignored quads.")
            self.load_next_videos()

    def undo_ranking(self):
        """Undo the last ranking."""
        if len(self.history) == 0:
            messagebox.showinfo("Info", "No ranking to undo.")
            return
        self.current_hashes = self.history.pop()
        self.relationships = ['<'] * (len(self.current_hashes) -1) # Reset relationships

        # Clear any existing video widgets
        self.clear_video_widgets()
        
        # Get the video paths
        video_paths = self.dataset_manager.get_video_paths(self.current_hashes)
        
        # Check if all videos exist
        if None in video_paths or not all(video_paths):
            messagebox.showerror("Error", "One or more video files not found")
            self.load_next_videos()
            return
        
        # Create the video widgets
        self.create_video_widgets(video_paths)
        if self.video_widgets:
            total_frames = self.video_widgets[0].cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.progress['maximum'] = total_frames
            self.progress['value'] = 0
            if self.video_widgets[0].is_playing:
                self.update_progress_bar()
            

    def restart_videos(self):
        """Restart all videos from the beginning."""
        for widget in self.video_widgets:
            if widget.cap and widget.cap.isOpened():
                widget.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def prompt_generate_new_videos(self):
        """Prompt user to generate new videos if not enough available."""
        num_videos = simpledialog.askinteger("Generate Videos", 
                                           "Not enough videos available. How many new videos to generate?", 
                                           minvalue=4)
        if num_videos:
            self.generate_new_videos(num_videos)
    
    def generate_new_videos_dialog(self):
        """Prompt for generating new videos."""
        num_videos = simpledialog.askinteger("Generate Videos", 
                                           "How many new videos would you like to generate?", 
                                           minvalue=4, initialvalue=4)
        if num_videos:
            self.generate_new_videos(num_videos)
    
    def generate_new_videos(self, num_videos):
        """Generate new video simulations."""
        loading_screen = LoadingScreen(self.master)
        loading_screen.run_generation_process(
            self.simulator, 
            num_videos, 
            self.dataset_manager, 
            self.pairs_manager,
            self.verbose,
            on_complete=lambda: self.on_generation_complete(loading_screen),
            on_error=lambda: self.on_generation_error(loading_screen)
        )
    
    def on_generation_complete(self, loading_screen):
        """Called when generation is complete."""
        loading_screen.close()
        self.update_progress_percentage()
        self.load_next_videos()
    
    def on_generation_error(self, loading_screen):
        """Handle errors during generation."""
        loading_screen.close()
        # Error dialog is handled in the loading screen
    
    def update_progress_percentage(self):
        """Update the progress percentage label."""
        nb_pairs = self.pairs_manager.get_nb_pairs()
        nb_ranked_pairs = self.pairs_manager.get_nb_ranked_pairs()
        self.progress_label.config(
            text=f"Progress: {((nb_ranked_pairs) / nb_pairs) * 100 if nb_pairs > 0 else 0:.1f}% "
                 f"({nb_ranked_pairs}/{nb_pairs}). Number of simulations: {len(self.dataset_manager)}"
        )
    
    def save_and_exit(self):
        """Save data and exit the application."""
        # Release video resources
        self.clear_video_widgets()
        
        if self.progress_after_id:
            self.master.after_cancel(self.progress_after_id)

        # Save managers
        self.pairs_manager.save()
        self.dataset_manager.save()
        
        # Destroy window and quit
        self.master.destroy()
        self.master.quit()

class LoadingScreen:
    def __init__(self, master, title="Generating Videos"):
        self.window = tk.Toplevel(master)
        self.window.title(title)
        self.window.geometry("300x200")
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
        
    def update_status(self, message):
        """Update the status message."""
        self.status_var.set(message)
        self.window.update()
        
    def close(self):
        """Close the loading screen."""
        if hasattr(self, 'animation_id'):
            self.window.after_cancel(self.animation_id)
        self.window.grab_release()
        self.window.destroy()
        
    def run_generation_process(self, simulator, num_pairs, dataset_manager, pairs_manager, 
                              verbose=False, on_complete=None, on_error=None):
        """
        Run the generation process in a separate thread.
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
        ttk.Label(frame, text="An error occurred during video generation. Details below:", 
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
