import os
import cv2
import numpy as np
import data_loader
import algorithms
import tkinter as tk
from tkinter import filedialog

def get_available_datasets(dataset_path):
    if not os.path.exists(dataset_path):
        return []
    # List directories in dataset_path
    datasets = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    return sorted(datasets)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "..", "DATASETS")
    
    print("=" * 60)
    print("  Visual Anomaly Detection System - Interactive Pipeline")
    print("=" * 60)
    
    datasets = get_available_datasets(dataset_path)
    if not datasets:
        print(f"Error: No datasets found in {dataset_path}")
        return
        
    root = tk.Tk()
    root.withdraw()
        
    while True:
        print("\nAvailable Datasets:")
        for i, ds in enumerate(datasets):
            print(f"  [{i + 1}] {ds}")
        print("  [x] Exit")
        
        choice = input("\nSelect a dataset (number) or 'x' to exit: ").strip()
        if choice.lower() == 'x':
            print("Exiting pipeline. Goodbye!")
            break
            
        try:
            choice_idx = int(choice) - 1
            if choice_idx < 0 or choice_idx >= len(datasets):
                print("Invalid choice. Try again.")
                continue
        except ValueError:
            print("Please enter a valid number or 'x'.")
            continue
            
        selected_dataset = datasets[choice_idx]
        print(f"\n[+] Selected dataset: {selected_dataset}")
        
        # Load training images
        print(f"Loading normal training set for {selected_dataset}...")
        train_images_color = data_loader.load_train_images(dataset_path, selected_dataset)
        
        if len(train_images_color) == 0:
            print("Error: No training images found. Please check dataset.")
            continue
            
        print(f"Loaded {len(train_images_color)} normal training images.")
        
        print("Processing training images to generate golden template...")
        train_images_grayscale = [algorithms.convert_to_grayscale(img) for img in train_images_color]
        smooth_train_images = [algorithms.apply_gaussian_filter(img, kernel_size=5) for img in train_images_grayscale]
        mean_template, std_template = algorithms.generate_golden_template(smooth_train_images)
        print("Golden template and standard deviation generated successfully.")
        
        test_dir_path = os.path.abspath(os.path.join(dataset_path, selected_dataset, "test"))
        
        while True:
            print(f"\nOptions for dataset '{selected_dataset}':")
            print("  [1] Select a test image")
            print("  [b] Back to dataset selection")
            
            cat_choice = input("\nEnter option: ").strip().lower()
            
            if cat_choice == 'b':
                break
            elif cat_choice == '1':
                file_path = filedialog.askopenfilename(
                    initialdir=test_dir_path if os.path.exists(test_dir_path) else base_dir,
                    title=f"Select Test Image ({selected_dataset})",
                    filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp"), ("All Files", "*.*")]
                )
                
                if not file_path:
                    print("No file selected.")
                    continue
                    
                test_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if test_img is None:
                    print("Image could not be read.")
                    continue
                    
                # Process the image
                test_image_gray = algorithms.convert_to_grayscale(test_img)
                smooth_test_image = algorithms.apply_gaussian_filter(test_image_gray, kernel_size=5)
                
                # Calculate the difference for the heatmap visualization
                diff = np.abs(smooth_test_image.astype(np.float32) - mean_template).astype(np.uint8)
                
                # Compute the final mask using the Z-score function
                final_mask = algorithms.compute_anomaly(smooth_test_image, mean_template, std_template, z_threshold=3.5)
                
                # Create Heatmap based on the difference
                norm_diff = np.clip(diff.astype(np.float32) * 3.0, 0, 255).astype(np.uint8)
                
                heatmap_lut = algorithms.generate_heatmap_lut()
                heatmap = algorithms.apply_heatmap(norm_diff, heatmap_lut)

                # Display in 4 windows
                window_names = ["Original Test Image", "Mean Template", "Heatmap", "Anomaly Mask"]
                for name in window_names:
                    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                
                cv2.imshow("Original Test Image", test_img)
                
                # Convert mean_template to uint8 for proper display in OpenCV
                cv2.imshow("Mean Template", mean_template.astype(np.uint8))
                cv2.imshow("Heatmap", heatmap)
                cv2.imshow("Anomaly Mask", final_mask)
                
                print("Images displayed. Press 'q' or 'ESC' in any window, or close a window to return.")
                
                while True:
                    key = cv2.waitKey(100) & 0xFF
                    if key == 27 or key == ord('q'): # ESC or q
                        break
                        
                    # Check if any window was closed
                    windows_open = True
                    for name in window_names:
                        try:
                            if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                                windows_open = False
                                break
                        except cv2.error:
                            windows_open = False
                            break
                    
                    if not windows_open:
                        break
                        
                cv2.destroyAllWindows()
            else:
                print("Invalid option.")

if __name__ == "__main__":
    main()
