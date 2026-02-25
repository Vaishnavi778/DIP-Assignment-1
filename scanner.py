"""
Name: Vaishnavi Goswami
Roll No: 2301010372
Course: Image Processing & Computer Vision
Unit: Image Acquisition & Sampling
Assignment Title: Smart Document Scanner & Quality Analysis System
Date: 10-02-2026
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

print("=============================================")
print(" SMART DOCUMENT SCANNER & QUALITY ANALYSIS ")
print("=============================================")
print("This system analyzes effects of sampling and quantization\n")


# -----------------------------
# Create Output Folder
# -----------------------------
if not os.path.exists("outputs"):
    os.makedirs("outputs")


# -----------------------------
# Quantization Function
# -----------------------------
def quantize_image(image, levels):
    """
    Reduces gray levels of an image
    """
    max_val = 255
    step = max_val // (levels - 1)
    quantized = (image // step) * step
    return quantized


# -----------------------------
# Sampling Function
# -----------------------------
def sample_image(image, size):
    down = cv2.resize(image, (size, size))
    up = cv2.resize(down, (512, 512))
    return up


# -----------------------------
# Main Processing Function
# -----------------------------
def process_image(image_path):

    print(f"\nProcessing Image: {image_path}")

    # Load image
    original = cv2.imread(image_path)

    if original is None:
        print("Error loading image.")
        return

    original = cv2.resize(original, (512, 512))

    # Convert to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # -----------------------------
    # Sampling
    # -----------------------------
    high_res = gray
    med_res = sample_image(gray, 256)
    low_res = sample_image(gray, 128)

    # -----------------------------
    # Quantization
    # -----------------------------
    q8 = quantize_image(gray, 256)
    q4 = quantize_image(gray, 16)
    q2 = quantize_image(gray, 4)

    # -----------------------------
    # Display All in One Figure
    # -----------------------------
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(3, 3, 2)
    plt.imshow(high_res, cmap='gray')
    plt.title("512x512")
    plt.axis("off")

    plt.subplot(3, 3, 3)
    plt.imshow(med_res, cmap='gray')
    plt.title("256x256")
    plt.axis("off")

    plt.subplot(3, 3, 4)
    plt.imshow(low_res, cmap='gray')
    plt.title("128x128")
    plt.axis("off")

    plt.subplot(3, 3, 5)
    plt.imshow(q8, cmap='gray')
    plt.title("8-bit (256 levels)")
    plt.axis("off")

    plt.subplot(3, 3, 6)
    plt.imshow(q4, cmap='gray')
    plt.title("4-bit (16 levels)")
    plt.axis("off")

    plt.subplot(3, 3, 7)
    plt.imshow(q2, cmap='gray')
    plt.title("2-bit (4 levels)")
    plt.axis("off")

    plt.tight_layout()

    # Save output
    filename = os.path.basename(image_path)
    save_path = f"outputs/{filename}_output.png"
    plt.savefig(save_path)
    print("Saved:", save_path)

    plt.show()

    # -----------------------------
    # Observations
    # -----------------------------
    print("\n--- Quality Observations ---")
    print("1. Low resolution (128x128) causes loss of fine text edges.")
    print("2. Medium resolution preserves readability but slight blur exists.")
    print("3. 4-bit quantization introduces visible banding.")
    print("4. 2-bit quantization significantly reduces readability.")
    print("5. OCR works best with 512x512 and 8-bit grayscale images.")


# -----------------------------
# Run Program
# -----------------------------
if __name__ == "__main__":

    image_path = input("Enter path of document image: ")
    process_image(image_path)