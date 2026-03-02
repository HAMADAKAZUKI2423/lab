import cv2
import numpy as np
import os
import argparse

def apply_gamma(img, gamma=2.2, decode=True):
    """
    Apply gamma correction.
    decode=True: sRGB (0-255) -> Linear (0.0-1.0)
    decode=False: Linear (0.0-1.0) -> sRGB (0-255)
    """
    if decode:
        # Normalize to 0-1 and remove gamma
        norm = img.astype(np.float32) / 255.0
        return np.power(norm, gamma)
    else:
        # Apply gamma and scale back to 0-255
        corrected = np.power(img, 1.0 / gamma)
        return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

def superimpose_images(gabor_path, noise_path, output_path, gamma=2.2):
    """
    Superimposes a Gabor patch onto a Noise image assuming OST-AR (Additive Blending).
    The Noise image is cropped to the center to match the Gabor image size.
    """
    # 1. Read images
    img_gabor = cv2.imread(gabor_path)
    img_noise = cv2.imread(noise_path)

    if img_gabor is None:
        raise FileNotFoundError(f"Gabor image not found: {gabor_path}")
    if img_noise is None:
        raise FileNotFoundError(f"Noise image not found: {noise_path}")

    # 2. Get dimensions
    h_g, w_g = img_gabor.shape[:2]
    h_n, w_n = img_noise.shape[:2]

    # 3. Crop Noise to center (match Gabor size)
    if w_n < w_g or h_n < h_g:
        raise ValueError(f"Noise image ({w_n}x{h_n}) is smaller than Gabor image ({w_g}x{h_g}).")

    cx, cy = w_n // 2, h_n // 2
    x1 = cx - w_g // 2
    y1 = cy - h_g // 2
    x2 = x1 + w_g
    y2 = y1 + h_g

    img_noise_cropped = img_noise[y1:y2, x1:x2]

    # 4. Convert to linear space (physically linear luminance)
    gabor_linear = apply_gamma(img_gabor, gamma, decode=True)
    noise_linear = apply_gamma(img_noise_cropped, gamma, decode=True)

    # 5. Additive Blending (OST-AR simulation)
    # In OST-AR, the display light (Foreground) is added to the real world light (Background).
    # L_observed = L_background + L_foreground
    blended_linear = noise_linear + gabor_linear

    # 6. Clip to valid range [0, 1]
    # Note: In reality, the eye can perceive brightness > 1.0 (monitor max), 
    # but for standard image formats, we must clip.
    if np.max(blended_linear) > 1.0:
        print(f"Warning: Blended luminance exceeded 1.0 (Max: {np.max(blended_linear):.2f}). Clipping applied.")
    
    blended_linear = np.clip(blended_linear, 0.0, 1.0)

    # 7. Convert back to sRGB (Gamma Encode)
    blended_uint8 = apply_gamma(blended_linear, gamma, decode=False)

    # 8. Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, blended_uint8)
    print(f"Saved superimposed image: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Superimpose Gabor on Noise for OST-AR simulation.")
    parser.add_argument("--gabor", required=True, help="Path to Gabor image (Foreground)")
    parser.add_argument("--noise", required=True, help="Path to Noise image (Background)")
    parser.add_argument("--output", required=True, help="Path to output image")
    parser.add_argument("--gamma", type=float, default=2.2, help="Gamma value for linearization (default: 2.2)")

    args = parser.parse_args()

    try:
        superimpose_images(args.gabor, args.noise, args.output, args.gamma)
    except Exception as e:
        print(f"Error: {e}")