import cv2
import numpy as np
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time

class ProgressDialog:
    def __init__(self, parent, title="Processing"):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.transient(parent)
        self.top.grab_set()
        
        # Center the dialog
        window_width = 300
        window_height = 100
        screen_width = parent.winfo_screenwidth()
        screen_height = parent.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.top.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Progress bar
        self.label = ttk.Label(self.top, text="Applying texture...", padding=(10, 5))
        self.label.pack()
        
        self.progress = ttk.Progressbar(
            self.top, length=250, mode='determinate',
            maximum=100, value=0
        )
        self.progress.pack(pady=10, padx=20)
        
        self.status_label = ttk.Label(self.top, text="", padding=(10, 5))
        self.status_label.pack()
        
    def update(self, value, status=""):
        self.progress['value'] = value
        if status:
            self.status_label['text'] = status
        self.top.update()
        
    def destroy(self):
        self.top.destroy()

class TextureMapperGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Texture Mapper")
        
        # Initialize paths
        self.images_folder = Path("./images/base_images")
        self.textures_folder = Path("./images/textures")
        
        # Store the current image and texture
        self.current_image = None
        self.original_image = None
        self.current_texture = None
        self.mask = None
        self.click_points = []
        self.result = None
        
        self.setup_gui()
        self.load_file_lists()
        
    def setup_gui(self):
        """Create the GUI elements"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure main window grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Dropdowns frame
        controls_frame = ttk.LabelFrame(main_frame, text="Image Selection", padding="5")
        controls_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        controls_frame.columnconfigure(1, weight=1)
        controls_frame.columnconfigure(3, weight=1)
        
        # Image selection
        ttk.Label(controls_frame, text="Image:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.image_var = tk.StringVar()
        self.image_dropdown = ttk.Combobox(controls_frame, textvariable=self.image_var)
        self.image_dropdown.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        self.image_dropdown.bind('<<ComboboxSelected>>', self.on_image_select)
        
        # Texture selection
        ttk.Label(controls_frame, text="Texture:").grid(row=0, column=2, padx=5, sticky=tk.W)
        self.texture_var = tk.StringVar()
        self.texture_dropdown = ttk.Combobox(controls_frame, textvariable=self.texture_var)
        self.texture_dropdown.grid(row=0, column=3, padx=5, sticky=(tk.W, tk.E))
        self.texture_dropdown.bind('<<ComboboxSelected>>', self.on_texture_select)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Effect Controls", padding="5")
        params_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        params_frame.columnconfigure(1, weight=1)
        
        # Tolerance slider with percentage
        ttk.Label(params_frame, text="Selection Tolerance:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.tolerance_var = tk.IntVar(value=15)
        self.tolerance_slider = ttk.Scale(
            params_frame, from_=1, to=100, 
            orient=tk.HORIZONTAL, variable=self.tolerance_var,
            command=self.on_tolerance_change
        )
        self.tolerance_slider.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        self.tolerance_label = ttk.Label(params_frame, text="15%")
        self.tolerance_label.grid(row=0, column=2, padx=5)

    # Effect strength slider with percentage
        ttk.Label(params_frame, text="Effect Strength:").grid(row=1, column=0, padx=5, sticky=tk.W)
        self.strength_var = tk.IntVar(value=50)
        self.strength_slider = ttk.Scale(
            params_frame, from_=1, to=100, 
            orient=tk.HORIZONTAL, variable=self.strength_var,
            command=self.on_strength_change
        )
        self.strength_slider.grid(row=1, column=1, padx=5, sticky=(tk.W, tk.E))
        self.strength_label = ttk.Label(params_frame, text="50%")
        self.strength_label.grid(row=1, column=2, padx=5)
        
        # Blend mode selection
        ttk.Label(params_frame, text="Blend Mode:").grid(row=2, column=0, padx=5, sticky=tk.W)
        self.blend_var = tk.StringVar(value="overlay")
        blend_modes = ["overlay", "multiply", "screen", "soft-light", "hard-light", "color-burn", "color-dodge"]
        self.blend_dropdown = ttk.Combobox(
            params_frame, textvariable=self.blend_var, 
            values=blend_modes, state="readonly"
        )
        self.blend_dropdown.grid(row=2, column=1, columnspan=2, padx=5, sticky=(tk.W, tk.E))
        self.blend_dropdown.bind('<<ComboboxSelected>>', self.on_blend_change)
        
        # Canvas for image display
        self.canvas_frame = ttk.LabelFrame(main_frame, text="Image Preview - Click to select areas", padding="5")
        self.canvas_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.canvas_frame.columnconfigure(0, weight=1)
        self.canvas_frame.rowconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(self.canvas_frame, width=800, height=600, bg='gray90')
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=5)
        
        ttk.Button(button_frame, text="Clear Selection", command=self.clear_points).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Apply Texture", command=self.apply_texture).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Reset Image", command=self.reset_image).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Save Result", command=self.save_result).grid(row=0, column=3, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))
        self.status_var.set("Ready")

    def load_file_lists(self):
        """Load lists of available images and textures"""
        try:
            images = list(self.images_folder.glob("*.webp"))
            self.image_dropdown['values'] = [img.name for img in images]
            
            textures = list(self.textures_folder.glob("*.webp"))
            self.texture_dropdown['values'] = [tex.name for tex in textures]
            
            if not images or not textures:
                messagebox.showwarning("Warning", "No .webp files found in images or textures folders!")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading files: {str(e)}")

    def load_webp(self, path):
        """Load a WebP image and convert to OpenCV format"""
        try:
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image {path}: {str(e)}")
            return None

    def create_mask(self, image, points, tolerance):
        """Create a more precise and controlled mask"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Scale tolerance more reasonably
        tolerance = tolerance * 2.5  # Reduced from 5x to 2.5x for more control
        
        # Create initial mask
        for point in points:
            temp_mask = np.zeros((h+2, w+2), np.uint8)
            flood_img = image.copy()
            
            # More controlled flood fill
            flood_diff = (tolerance, tolerance, tolerance)
            cv2.floodFill(
                flood_img, temp_mask, point,
                (255, 255, 255),
                flood_diff,
                flood_diff,
                cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
            )
            
            # Add to main mask
            mask = cv2.bitwise_or(mask, temp_mask[1:-1, 1:-1])
        
        # Refined mask processing
        # 1. Initial cleanup
        mask = cv2.medianBlur(mask, 5)  # Remove noise
        
        # 2. Controlled expansion
        kernel_small = np.ones((3,3), np.uint8)
        kernel_medium = np.ones((5,5), np.uint8)
        
        # Gradual mask refinement
        mask = cv2.dilate(mask, kernel_small, iterations=1)
        mask = cv2.erode(mask, kernel_small, iterations=1)
        mask = cv2.dilate(mask, kernel_medium, iterations=1)
        
        # 3. Edge smoothing
        # Calculate edge smoothing based on image size
        blur_size = max(3, min(9, int(min(h, w) * 0.01)))
        if blur_size % 2 == 0:
            blur_size += 1
            
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        
        # 4. Enhance mask contrast while preserving edges
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
        
        # 5. Apply adaptive thresholding for better edge preservation
        mask_float = mask.astype(float) / 255
        mask_float = np.power(mask_float, 0.8)  # Slightly sharper edges
        mask = (mask_float * 255).astype(np.uint8)
        
        return mask

    def apply_blend_mode(self, image, texture, mask, mode='overlay', strength=0.5):
        """Apply texture with refined blending"""
        # Convert to float
        img_float = image.astype(float) / 255.0
        texture_float = texture.astype(float) / 255.0
        mask_float = mask.astype(float) / 255.0
        
        # Convert mask to 3 channels with refined edge handling
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
        
        # Enhance texture contrast with more control
        texture_float = cv2.normalize(texture_float, None, 0, 1, cv2.NORM_MINMAX)
        texture_float = np.power(texture_float, 0.7)  # Slightly reduced from 0.6
        
        # Refined blending modes
        if mode == 'overlay':
            result_dark = 2.5 * img_float * texture_float  # Reduced from 3.0
            result_bright = 1.0 - 2.5 * (1.0 - img_float) * (1.0 - texture_float)
            texture_effect = np.where(img_float <= 0.5, result_dark, result_bright)
        elif mode == 'multiply':
            texture_effect = img_float * texture_float * 2.0  # Reduced from 2.5
        elif mode == 'screen':
            texture_effect = 1.0 - (1.0 - img_float) * (1.0 - texture_float) * 2.0
        elif mode == 'soft-light':
            texture_effect = np.power(img_float, np.power(2, 2 * (texture_float - 0.5)))
        elif mode == 'hard-light':
            texture_effect = np.where(texture_float <= 0.5,
                                    2.0 * img_float * texture_float,
                                    1.0 - 2.0 * (1.0 - img_float) * (1.0 - texture_float))
        else:
            # Default blend with more control
            texture_effect = cv2.addWeighted(img_float, 0.4, texture_float, 0.6, 0)
        
        # More controlled strength application
        strength = np.clip(strength * 2.0, 0, 1)  # Reduced from 3.0
        
        # Refined edge handling
        mask_strength = np.power(mask_3ch, 0.7) * strength  # Smoother transition
        result = img_float * (1.0 - mask_strength) + texture_effect * mask_strength
        
        # Final contrast enhancement with more control
        result = cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX)
        result = np.power(result, 0.9)  # Slightly reduced contrast boost
        
        return (np.clip(result * 255, 0, 255)).astype(np.uint8)

    def update_preview(self):
        """Update the preview image on canvas"""
        if self.current_image is not None:
            # Create display image
            display_img = self.current_image.copy()
            
            # Draw points
            for point in self.click_points:
                cv2.circle(display_img, point, 3, (0, 255, 0), -1)
            
            # Convert to RGB for tkinter
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            
            # Calculate scaling
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            img_height, img_width = display_img.shape[:2]
            
            # Calculate scale to fit image in canvas while maintaining aspect ratio
            scale = min(canvas_width/img_width, canvas_height/img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            if new_width > 0 and new_height > 0:  # Prevent scaling to zero size
                display_img = cv2.resize(display_img, (new_width, new_height))
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(display_img))
                
                # Center the image on the canvas
                x_offset = (canvas_width - new_width) // 2
                y_offset = (canvas_height - new_height) // 2
                
                # Update canvas
                self.canvas.delete("all")
                self.canvas.create_image(x_offset, y_offset, image=self.photo, anchor=tk.NW)
                
                # Update status
                self.status_var.set(f"Image size: {img_width}x{img_height} | Selected points: {len(self.click_points)}")
    

    def on_image_select(self, event=None):
        """Handle image selection with reset"""
        selected = self.image_var.get()
        if selected:
            path = self.images_folder / selected
            new_image = self.load_webp(path)
            if new_image is not None:
                self.original_image = new_image.copy()
                self.current_image = new_image.copy()
                self.click_points = []
                self.update_preview()
                self.status_var.set(f"Loaded new image: {selected}")
    
    def on_texture_select(self, event=None):
        """Handle texture selection"""
        selected = self.texture_var.get()
        if selected:
            path = self.textures_folder / selected
            self.current_texture = self.load_webp(path)
            if self.current_texture is not None:
                self.status_var.set(f"Loaded texture: {selected}")
    
    def on_canvas_click(self, event):
        """Handle canvas clicks"""
        if self.current_image is not None:
            # Convert canvas coordinates to image coordinates
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            img_height, img_width = self.current_image.shape[:2]
            
            # Calculate scaling and offsets
            scale = min(canvas_width/img_width, canvas_height/img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            x_offset = (canvas_width - new_width) // 2
            y_offset = (canvas_height - new_height) // 2
            
            # Convert click coordinates to image coordinates
            x = int((event.x - x_offset) / scale)
            y = int((event.y - y_offset) / scale)
            
            # Check if click is within image bounds
            if 0 <= x < img_width and 0 <= y < img_height:
                self.click_points.append((x, y))
                self.update_preview()
                self.status_var.set(f"Added point at ({x}, {y})")
    
    def on_tolerance_change(self, event=None):
        """Handle tolerance slider changes"""
        value = self.tolerance_var.get()
        self.tolerance_label.config(text=f"{value}%")
        
    def on_strength_change(self, event=None):
        """Handle effect strength slider changes"""
        value = self.strength_var.get()
        self.strength_label.config(text=f"{value}%")
        
    def on_blend_change(self, event=None):
        """Handle blend mode changes"""
        pass
    
    def clear_points(self):
        """Clear selection points without resetting the image"""
        self.click_points = []
        self.update_preview()
        self.status_var.set("Selection cleared - previous changes preserved")
    
    def reset_image(self):
        """Reset the image to its original state and clear points"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.click_points = []
            self.update_preview()
            self.status_var.set("Image reset to original")
    def apply_texture(self):
        """Apply texture progressively, using current state as base"""
        if self.current_image is None or self.current_texture is None:
            messagebox.showwarning("Warning", "Please select both an image and a texture first!")
            return
        
        if not self.click_points:
            messagebox.showwarning("Warning", "Please select areas by clicking on the image first!")
            return
        
        progress = ProgressDialog(self.root)
        
        try:
            # Create mask
            progress.update(10, "Creating selection mask...")
            tolerance = self.tolerance_var.get()
            self.mask = self.create_mask(self.current_image, self.click_points, tolerance)
            
            # Show refined mask preview
            mask_display = self.mask.copy()
            mask_preview = cv2.applyColorMap(mask_display, cv2.COLORMAP_JET)
            # cv2.imshow('Mask Preview - Blue=No Effect, Red=Full Effect', mask_preview)
            
            # Resize and prepare texture
            progress.update(30, "Preparing texture...")
            texture_resized = cv2.resize(self.current_texture, 
                                       (self.current_image.shape[1], self.current_image.shape[0]))
            
            # Use current image (with previous changes) as base instead of original
            progress.update(70, "Applying texture effect...")
            strength = self.strength_var.get() / 100.0
            self.result = self.apply_blend_mode(
                self.current_image,  # Use current image instead of original_image
                texture_resized, 
                self.mask, 
                self.blend_var.get(),
                strength
            )
            
            # Update preview
            self.current_image = self.result.copy()
            progress.update(100, "Done!")
            time.sleep(0.5)
            progress.destroy()
            
            self.update_preview()
            self.status_var.set(f"Applied texture with {self.blend_var.get()} blend mode at {int(strength*100)}% strength")
            
        except Exception as e:
            progress.destroy()
            messagebox.showerror("Error", f"Error applying texture: {str(e)}")
            
    def save_result(self):
        """Save the processed image"""
        if not hasattr(self, 'result') or self.result is None:
            messagebox.showwarning("Warning", "No result to save! Please apply texture first.")
            return
            
        try:
            output_folder = self.images_folder / "results"
            output_folder.mkdir(exist_ok=True)
            
            # Generate filename with parameters
            base_name = self.image_var.get().rsplit('.', 1)[0]
            texture_name = self.texture_var.get().rsplit('.', 1)[0]
            params = f"{self.blend_var.get()}_{self.strength_var.get()}pct"
            output_name = f"{base_name}_with_{texture_name}_{params}.webp"
            
            output_path = output_folder / output_name
            
            # Convert BGR to RGB
            result_rgb = cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB)
            
            # Save using PIL
            Image.fromarray(result_rgb).save(str(output_path), 'WEBP', quality=95)
            self.status_var.set(f"Saved result as: {output_name}")
            messagebox.showinfo("Success", f"Saved result to:\n{output_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error saving result: {str(e)}")

def main():
    try:
        root = tk.Tk()
        root.title("Texture Mapper")
        
        # Set minimum window size
        root.minsize(900, 700)
        
        # Configure grid weights for resizing
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)
        
        app = TextureMapperGUI(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"Application error: {str(e)}")

if __name__ == "__main__":
    main()