import cv2
import data_loader
import algorithms

class AnomalyDetectionApp:
    def __init__(self):
        self.dataset_path = ".\..\DATASETS\\"
        self.category = "pill"

    def run(self):
        print(f"Loading dataset from: {self.dataset_path}/{self.category}...")
        

        train_images_color = data_loader.load_train_images(
            self.dataset_path, 
            self.category
        )
        
        if len(train_images_color) == 0:
            print("Error: No images found. Please check your dataset path.")
            return

        print(f"Successfully loaded {len(train_images_color)} images.")

        test_image = train_images_color[0]

        basic_grayscale_image = algorithms.convert_to_grayscale(test_image)
        luminosity_grayscale_image = algorithms.convert_to_grayscale_using_luminosity(test_image)

        cv2.imshow("Original Image", test_image)
        cv2.imshow("Basic Grayscale Image", basic_grayscale_image)
        cv2.imshow("Luminosity Grayscale Image", luminosity_grayscale_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AnomalyDetectionApp()
    app.run()