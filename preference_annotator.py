"""
Human Preference Annotation Tool

Simple GUI tool to annotate crop preference pairs for RLHF training.
Shows two crops side by side and lets you choose which is better.

Usage:
  python preference_annotator.py --preferences results/step6/preference_pairs.json
"""

import argparse
import json
import os
import tkinter as tk
# from tkinter import tqdm
from typing import List, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from step6_incremental_improvement import PreferencePair, IncrementalImprovementPipeline


class PreferenceAnnotator:
    def __init__(self, preferences_path: str):
        self.preferences_path = preferences_path
        self.pipeline = IncrementalImprovementPipeline()
        self.preferences = self.pipeline._load_preferences(preferences_path)
        
        # Filter unannotated pairs
        self.unannotated = [i for i, p in enumerate(self.preferences) if p.preference == -1]
        self.current_idx = 0
        
        # GUI setup
        self.root = tk.Tk()
        self.root.title("Crop Preference Annotator")
        self.root.geometry("1200x600")
        
        self.setup_gui()
        self.load_current_pair()
        
    def setup_gui(self):
        """Setup the GUI components"""
        # Title
        title_label = tk.Label(self.root, text="Which crop looks better?", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Progress
        self.progress_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.progress_label.pack()
        
        # Main frame for crops
        main_frame = tk.Frame(self.root)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Crop A frame
        self.crop_a_frame = tk.Frame(main_frame, relief=tk.RAISED, bd=2)
        self.crop_a_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10)
        
        crop_a_label = tk.Label(self.crop_a_frame, text="Crop A", font=("Arial", 14, "bold"))
        crop_a_label.pack(pady=5)
        
        self.crop_a_image_label = tk.Label(self.crop_a_frame)
        self.crop_a_image_label.pack(expand=True, fill=tk.BOTH)
        
        # Crop B frame
        self.crop_b_frame = tk.Frame(main_frame, relief=tk.RAISED, bd=2)
        self.crop_b_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10)
        
        crop_b_label = tk.Label(self.crop_b_frame, text="Crop B", font=("Arial", 14, "bold"))
        crop_b_label.pack(pady=5)
        
        self.crop_b_image_label = tk.Label(self.crop_b_frame)
        self.crop_b_image_label.pack(expand=True, fill=tk.BOTH)
        
        # Buttons frame
        buttons_frame = tk.Frame(self.root)
        buttons_frame.pack(pady=20)
        
        # Choice buttons
        self.choice_a_btn = tk.Button(buttons_frame, text="A is Better", 
                                     command=lambda: self.make_choice(0),
                                     bg="lightgreen", font=("Arial", 12, "bold"),
                                     width=15, height=2)
        self.choice_a_btn.pack(side=tk.LEFT, padx=10)
        
        self.no_preference_btn = tk.Button(buttons_frame, text="No Preference", 
                                          command=lambda: self.make_choice(-1),
                                          bg="lightyellow", font=("Arial", 12),
                                          width=15, height=2)
        self.no_preference_btn.pack(side=tk.LEFT, padx=10)
        
        self.choice_b_btn = tk.Button(buttons_frame, text="B is Better", 
                                     command=lambda: self.make_choice(1),
                                     bg="lightcoral", font=("Arial", 12, "bold"),
                                     width=15, height=2)
        self.choice_b_btn.pack(side=tk.LEFT, padx=10)
        
        # Navigation buttons
        nav_frame = tk.Frame(self.root)
        nav_frame.pack(pady=10)
        
        self.prev_btn = tk.Button(nav_frame, text="Previous", command=self.previous_pair,
                                 state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.next_btn = tk.Button(nav_frame, text="Next", command=self.next_pair)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        # Save button
        self.save_btn = tk.Button(nav_frame, text="Save Progress", command=self.save_progress,
                                 bg="lightblue", font=("Arial", 10, "bold"))
        self.save_btn.pack(side=tk.LEFT, padx=20)
        
        # Keyboard bindings
        self.root.bind('<KeyPress-1>', lambda e: self.make_choice(0))
        self.root.bind('<KeyPress-2>', lambda e: self.make_choice(1))
        self.root.bind('<KeyPress-0>', lambda e: self.make_choice(-1))
        self.root.bind('<KeyPress-Left>', lambda e: self.previous_pair())
        self.root.bind('<KeyPress-Right>', lambda e: self.next_pair())
        self.root.bind('<KeyPress-s>', lambda e: self.save_progress())
        
        # Focus for keyboard input
        self.root.focus_set()
        
    def load_current_pair(self):
        """Load the current preference pair"""
        if self.current_idx >= len(self.unannotated):
            self.show_completion()
            return
            
        pair_idx = self.unannotated[self.current_idx]
        preference = self.preferences[pair_idx]
        
        # Update progress
        progress = f"Pair {self.current_idx + 1} of {len(self.unannotated)}"
        self.progress_label.config(text=progress)
        
        # Load image
        image = cv2.imread(preference.image_path)
        if image is None:
            print(f"Could not load image: {preference.image_path}")
            return
        
        # Extract crops
        x_a, y_a, w_a, h_a = preference.crop_a
        x_b, y_b, w_b, h_b = preference.crop_b
        
        crop_a = image[y_a:y_a+h_a, x_a:x_a+w_a]
        crop_b = image[y_b:y_b+h_b, x_b:x_b+w_b]
        
        # Resize crops for display
        crop_a = cv2.resize(crop_a, (300, 300))
        crop_b = cv2.resize(crop_b, (300, 300))
        
        # Convert BGR to RGB
        crop_a = cv2.cvtColor(crop_a, cv2.COLOR_BGR2RGB)
        crop_b = cv2.cvtColor(crop_b, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL and display
        crop_a_pil = Image.fromarray(crop_a)
        crop_b_pil = Image.fromarray(crop_b)
        
        crop_a_photo = ImageTk.PhotoImage(crop_a_pil)
        crop_b_photo = ImageTk.PhotoImage(crop_b_pil)
        
        self.crop_a_image_label.config(image=crop_a_photo)
        self.crop_a_image_label.image = crop_a_photo  # Keep a reference
        
        self.crop_b_image_label.config(image=crop_b_photo)
        self.crop_b_image_label.image = crop_b_photo  # Keep a reference
        
        # Update navigation buttons
        self.prev_btn.config(state=tk.NORMAL if self.current_idx > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_idx < len(self.unannotated) - 1 else tk.DISABLED)
        
    def make_choice(self, choice: int):
        """Record the user's preference choice"""
        if self.current_idx >= len(self.unannotated):
            return
            
        pair_idx = self.unannotated[self.current_idx]
        self.preferences[pair_idx].preference = choice
        
        # Move to next pair
        self.next_pair()
        
    def next_pair(self):
        """Move to the next preference pair"""
        if self.current_idx < len(self.unannotated) - 1:
            self.current_idx += 1
            self.load_current_pair()
        else:
            self.show_completion()
            
    def previous_pair(self):
        """Move to the previous preference pair"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_current_pair()
            
    def save_progress(self):
        """Save current progress to file"""
        self.pipeline._save_preferences(self.preferences, self.preferences_path)
        
        # Update unannotated list
        self.unannotated = [i for i, p in enumerate(self.preferences) if p.preference == -1]
        
        print(f"Progress saved. {len(self.unannotated)} pairs remaining.")
        
    def show_completion(self):
        """Show completion message"""
        completion_label = tk.Label(self.root, text="All pairs annotated! 🎉", 
                                   font=("Arial", 16, "bold"), fg="green")
        completion_label.pack(pady=50)
        
        # Disable choice buttons
        self.choice_a_btn.config(state=tk.DISABLED)
        self.choice_b_btn.config(state=tk.DISABLED)
        self.no_preference_btn.config(state=tk.DISABLED)
        
    def run(self):
        """Start the annotation tool"""
        print("Preference Annotator Started")
        print("Controls:")
        print("  1 - A is better")
        print("  2 - B is better") 
        print("  0 - No preference")
        print("  Left/Right arrows - Navigate")
        print("  S - Save progress")
        print(f"Total pairs to annotate: {len(self.unannotated)}")
        
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Human Preference Annotation Tool")
    parser.add_argument("--preferences", type=str, required=True,
                       help="Path to preference pairs JSON file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.preferences):
        print(f"Preference file not found: {args.preferences}")
        print("Run Stage 2 first to generate preference pairs.")
        return
    
    annotator = PreferenceAnnotator(args.preferences)
    annotator.run()


if __name__ == "__main__":
    main()
