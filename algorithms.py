import numpy as np
import cv2
from scipy.ndimage import median_filter
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt

def convert_to_grayscale(image):
    # image is a 3D array in a [height, width, channels(Blue Green Red)] format 

    b = image[:, :, 0]  # all the values of the blue channel
    g = image[:, :, 1]  # all the values of the green channel
    r = image[:, :, 2]  # all the values of the red channel

    grayscale_image =  b/3 + g/3 + r/3 

    return grayscale_image.astype(np.ubyte)


def convert_to_grayscale_using_luminosity(image):
    # image is a 3D array in a [height, width, channels(Blue Green Red)] format 

    b = image[:, :, 0]  # all the values of the blue channel
    g = image[:, :, 1]  # all the values of the green channel
    r = image[:, :, 2]  # all the values of the red channel

    grayscale_image = 0.299 * r + 0.587 * g + 0.114 * b

    return grayscale_image.astype(np.ubyte)


def generate_gaussian_kernel(size):
    sigma = size / 6.0
    d = size // 2
    
    # [-d, -d+1, ..., 0, ..., d-1, d]
    coords = np.arange(size) - d

    # Create 2D grid of coordinates [-d, ..., d] x [-d, ..., d]
    dx, dy = np.meshgrid(coords, coords, indexing='ij')
    
    aux = 1.0 / (2.0 * np.pi * sigma**2)
    exponent = -(dx**2 + dy**2) / (2.0 * sigma**2)
    kernel = aux * np.exp(exponent)

    kernel /= kernel.sum()

    return kernel

def apply_gaussian_filter(src_image, kernel_size):
    d = kernel_size // 2
    
    kernel = generate_gaussian_kernel(kernel_size)
    
    dst_image = np.copy(src_image).astype(np.float32)
    float_image = np.copy(src_image).astype(np.float32)
    
    # creaetes all the possible windows of the image with the same size as the kernel
    windows = sliding_window_view(float_image, (kernel_size, kernel_size))
    
    # tensordot performs a sum of products over specified axes. 
    # windows is a 4D array, windows[0][0] is the first window of the image with the same size as the kernel
    # axes    0  1  2  3
    # windows[0][0][0][0] this is how we actually access the pixel values of the first window
    # so we multiply each window from the image with the kernel
    # product is a 2d array, same size as the original image, where each pixel is 
    # the result of the convolution of the kernel with the corresponding window of the image
    product = np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))
    
    # map the values of the product to the range [0, 255] and convert to uint8
    result = np.clip(product, 0, 255).astype(np.uint8)
    
    # this is equivalent to padding the image
    rows = slice(d, -d, 1)
    cols = slice(d, -d, 1)

    # the borders keep the source image's pixels unchanged
    dst_image[rows, cols] = result
    
    return dst_image.astype(np.uint8)


def generate_golden_template(images):
    images_stack = np.array(images, dtype=np.float32)
    
    N = images_stack.shape[0]
    
    # the sum of all images
    sum_image = np.sum(images_stack, axis=0)

    # arithmetic mean
    mean_image = sum_image / N
    
    # diff_stack is a 3D array that holds each image(from the original stack) - the mean image
    diff_stack = images_stack - mean_image
    
    # square each image and penalize the pixels that are far from the mean
    squared_diff_stack = diff_stack ** 2
    
    # the sum of all squared difference images
    sum_squared_diffs = np.sum(squared_diff_stack, axis=0)

    # arithmetic mean of the squared difference images a.k.a. variance
    variance_image = sum_squared_diffs / N
    
    # standard deviation is obtained by taking the square root of the variance
    std_image = np.sqrt(variance_image)
    # this is really helpfull because it shows how much each pixel in the image varies across the training set
    
    return mean_image.astype(np.float32), std_image.astype(np.float32)


def compute_anomaly(test_image, mean_template, std_template, z_threshold=3.5):
    
    epsilon = 2.0
    safe_std = std_template + epsilon  # this way division by 0 is avoided
    
    diff = np.abs(test_image.astype(np.float32) - mean_template)
    
    # if a pixel has a high standard deviation, then a large difference compared to the standard deviation will be small 
    # so therefore it won't be considered as anomaly

    # but if a pixel varies slightly across the train images, then the same difference compared to the standard deviation will be large 
    # and the pixel will be considered as anomaly
    z_score = diff / safe_std

    anomaly_mask = np.zeros(test_image.shape, dtype=np.uint8)
    anomaly_mask[z_score > z_threshold] = 255
    
    opened_mask = opening(anomaly_mask)
    final_mask = closing_n_times(opened_mask, iterations=7)
 
    return final_mask

def convert_to_binary(src_image, threshold):
    dst_image = np.zeros(src_image.shape, dtype=np.uint8)
    
    dst_image[src_image > threshold] = 255
    
    return dst_image    

def compute_histogram(src_image):
    # src_image is a 1D array
    hist = np.zeros((256), dtype=np.int32)
    
    for pixel in src_image.flatten():
        hist[pixel] += 1
        
    return hist

def normalize_histogram(histo, total_pixels):
    dst_hist = np.zeros(histo.shape, dtype=np.float32)
    dst_hist = histo / total_pixels
    return dst_hist

def show_histogram(hist):
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

# def find_best_threshold_otsu(src_image):
#     # Create a histogram of the image
#     hist = compute_histogram(src_image)

#     # Normalize the histogram
#     norm_hist = normalize_histogram(hist, src_image.size)

#     # The mean intensity of all the pixels
#     global_mean = np.mean(src_image)

#     # Array of intensity levels
#     intensity_levels = np.arange(0, 256)

#     # Track both the maximum variance found and the 'k' that produced it
#     max_variance = -1
#     best_threshold = 0 
    
#     # Iterate through all possible thresholds
#     for k in range(0, 256):
#         # Calculate P1(k) - inclusive of k
#         sum_k = np.sum(norm_hist[0:k+1])
        
#         # Prevent division by zero
#         if sum_k == 0 or sum_k == 1:
#             continue
            
#         # Calculate m(k) - inclusive of k
#         mean_k = np.sum(norm_hist[0:k+1] * intensity_levels[0:k+1])
        
#         # Calculate between-class variance
#         variance_k = (global_mean * sum_k - mean_k) ** 2 / (sum_k * (1 - sum_k))  
        
#         # If we found a new maximum variance, update both variables
#         if variance_k > max_variance:
#             max_variance = variance_k
#             best_threshold = k

#     return best_threshold

# just for testing purposes, not used in the main app
def apply_median_filter(src_image, kernel_size=5):
    return median_filter(src_image, size=kernel_size)


def erosion(binary_image):
    # Pad the image with 0 (background) to handle the edges safely
    padded_image = np.pad(binary_image, pad_width=1, mode='constant', constant_values=0)
    
    # Extract all 3x3 windows
    windows = sliding_window_view(padded_image, (3, 3))
    
    # The minimum value in a 3x3 window simulates erosion for a white object (255)
    # If there is any 0 in the 3x3 window, the minimum will be 0.
    eroded_image = np.min(windows, axis=(2, 3))
    
    return eroded_image.astype(np.uint8)

def dilation(binary_image):
    # Pad the image with 0 (background) to handle the edges safely
    padded_image = np.pad(binary_image, pad_width=1, mode='constant', constant_values=0)
    
    # Extract all 3x3 windows
    windows = sliding_window_view(padded_image, (3, 3))
    
    # The maximum value in a 3x3 window simulates dilation for a white object (255)
    # If there is any 255 in the 3x3 window, the maximum will be 255.
    dilated_image = np.max(windows, axis=(2, 3))
    
    return dilated_image.astype(np.uint8)

def opening(binary_image):
    # removes small noise from the binary image
    eroded = erosion(binary_image)
    opened = dilation(eroded)
    
    return opened

def closing(binary_image):
    #fills the holes in the binary image 
    dilated =  dilation(binary_image)
    closed = erosion(dilated)
    
    return closed

def closing_n_times(binary_image, iterations):

    temp_image = binary_image.copy()
    for _ in range(iterations):
        temp_image = dilation(temp_image)
        
    for _ in range(iterations):
        temp_image = erosion(temp_image)
        
    return temp_image

def opening_n_times(binary_image, iterations):

    temp_image = binary_image.copy()
    for _ in range(iterations):
        temp_image = erosion(temp_image)
        
    for _ in range(iterations):
        temp_image = dilation(temp_image)
        
    return temp_image

def generate_heatmap_lut():
    # Create an empty array for 256 colors, each with 3 channels (B, G, R)
    lut = np.zeros((256, 3), dtype=np.uint8)
    
    for i in range(256):
        # Normalize the pixel intensity to a value between 0.0 and 1.0
        v = i / 255.0
        
        # The mathematical formulas for the colormap curves
        r = 1.5 - abs(4.0 * v - 3.0)
        g = 1.5 - abs(4.0 * v - 2.0)
        b = 1.5 - abs(4.0 * v - 1.0)
        
        # Clamp the values to keep them strictly between 0.0 and 1.0
        # Then multiply by 255 to get the standard 8-bit color scale
        r = int(np.clip(r, 0.0, 1.0) * 255)
        g = int(np.clip(g, 0.0, 1.0) * 255)
        b = int(np.clip(b, 0.0, 1.0) * 255)
        
        # Store the color in BGR format (because OpenCV uses BGR, not RGB)
        lut[i] = [b, g, r]
        
    return lut

def apply_heatmap(gray_image, lut):
    
    heatmap = lut[gray_image]
    
    return heatmap
