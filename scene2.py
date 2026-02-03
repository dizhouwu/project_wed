import os
import numpy as np
import cv2
import random
from moviepy import ImageSequenceClip, concatenate_videoclips

# ==========================================
# 1. Configuration Center
# ==========================================
IMG_FOLDER = os.path.join(os.getcwd(), "project_wed")
OUTPUT_NAME = "scene2.mp4"

# Standard 1080p vertical resolution for iPhone 15 Pro Max
FINAL_W, FINAL_H = 1080, 1920 

def is_image(fn):
    return fn.lower().endswith((".jpg", ".jpeg", ".png"))

def extract_number(fn):
    try: 
        # Assumes filename is a sequence index, e.g., 1.jpg, 2.jpg
        return float(os.path.splitext(fn)[0])
    except: 
        return 0.0

# ==========================================
# 2. Core Rendering Engine (OpenCV Backend)
# ==========================================
def process_frame(img_path, blur_sigma, shake_val):
    img = cv2.imread(img_path)
    if img is None: 
        print(f"Warning: Unable to read file {img_path}")
        return None
    
    # Convert BGR to RGB (OpenCV default to MoviePy compatible)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img.shape[:2]
    
    # Calculate Cover Fill ratio (CSS object-fit: cover logic)
    # Upscale by 1.1x to provide redundancy for the shake offset
    scale = max(FINAL_W / w, FINAL_H / h) * 1.10 
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Shake Logic: Randomized pixel offset
    dx = random.randint(-shake_val, shake_val)
    dy = random.randint(-shake_val, shake_val)
    
    # Precise Center Cropping to 1080x1920
    y_mid, x_mid = img.shape[0]//2, img.shape[1]//2
    # Boundary check for pixel-perfect cropping
    img_cropped = img[y_mid-(FINAL_H//2)+dy : y_mid+(FINAL_H//2)+dy, 
                      x_mid-(FINAL_W//2)+dx : x_mid+(FINAL_W//2)+dx]
    
    # Size Validation (Ensures no 1px deviation from odd/even rounding)
    if img_cropped.shape[0] != FINAL_H or img_cropped.shape[1] != FINAL_W:
        img_cropped = cv2.resize(img_cropped, (FINAL_W, FINAL_H))

    # Dynamic Motion Blur Logic
    if blur_sigma > 0.3:
        # Gaussian Kernel size must be odd
        ksize = int(blur_sigma * 5) | 1
        img_cropped = cv2.GaussianBlur(img_cropped, (ksize, ksize), 0)
        
    return img_cropped

# ==========================================
# 3. Automated Production Pipeline
# ==========================================
def main():
    if not os.path.exists(IMG_FOLDER):
        print(f"Error: Folder not found {IMG_FOLDER}")
        return

    all_files = [img for img in os.listdir(IMG_FOLDER) if is_image(img)]
    # Reverse Sorting: Traveling from Present back to Past
    images_sorted = sorted(all_files, key=extract_number, reverse=True)
    images_paths = [os.path.join(IMG_FOLDER, img) for img in images_sorted]

    if not images_paths:
        print("No images found in the source directory!")
        return

    # Temporal Curve: Acceleration from 0.2s down to 0.04s (Rewind effect)
    start_duration, end_duration, gamma = 0.2, 0.04, 1.5
    t_vals = np.linspace(0, 1, len(images_paths))
    durations = start_duration * (end_duration / start_duration) ** (t_vals ** gamma)

    clips = []
    print(f"🚀 Processing {len(images_paths)} frames. Simulating system reconstruction...")

    for i, (p, d) in enumerate(zip(images_paths, durations)):
        # Shake intensity correlates with acceleration
        shake_intensity = int(t_vals[i] * 25) 
        # Motion blur increases as frame duration decreases
        blur_s = max(0, (start_duration - d) * 12)
        
        frame = process_frame(p, blur_s, shake_intensity)
        if frame is not None:
            clips.append(ImageSequenceClip([frame], durations=[d]))

    # Memory Flicker Injection 
    if len(clips) > 10:
        flash_idx = int(len(clips) * 0.7)
        flash_frame = process_frame(images_paths[flash_idx], 5.0, 40) # Extreme blur
        flash_clip = ImageSequenceClip([flash_frame], durations=[0.02])
        clips.insert(flash_idx, flash_clip)

    # ==========================================
    # 4. Export (iOS Optimized Color & Format)
    # ==========================================
    final_video = concatenate_videoclips(clips, method="compose")
    
    print("🎬 Rendering Final Commit...")
    # pix_fmt yuv420p ensures compatibility with iOS devices
    final_video.write_videofile(
        OUTPUT_NAME, 
        fps=30, 
        audio=False, 
        codec="libx264",
        preset="slow",
        ffmpeg_params=["-pix_fmt", "yuv420p"]
    )
    print(f"✨ Rendering Complete: {OUTPUT_NAME}")

if __name__ == "__main__":
    main()
