import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Setup
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Load images
A = cv2.imread("A.pgm", 0)
B = cv2.imread("B.pgm", 0)

# 1. Difference Field
D = np.abs(A.astype(np.float32) - B.astype(np.float32))

# normalize to [0,1]
D_norm = (D - D.min()) / (D.max() - D.min() + 1e-8)

# Save raw difference
cv2.imwrite(
    os.path.join(output_dir, "01_difference_raw.png"),
    (D_norm * 255).astype(np.uint8)
)

# 2. Pseudo-color (Heatmap)
plt.imshow(D_norm, cmap='turbo')
plt.colorbar()
plt.title("Difference Heatmap")
plt.savefig(os.path.join(output_dir, "02_heatmap.png"), dpi=300)
plt.close()

# 3. Smoothed Difference
D_smooth = cv2.GaussianBlur(D_norm, (5, 5), 0)

cv2.imwrite(
    os.path.join(output_dir, "03_smooth.png"),
    (D_smooth * 255).astype(np.uint8)
)

# 4. Gradient Magnitude
gy, gx = np.gradient(D_norm)
grad_mag = np.sqrt(gx**2 + gy**2)

plt.imshow(grad_mag, cmap='gray')
plt.title("Gradient Magnitude")
plt.savefig(os.path.join(output_dir, "04_gradient.png"), dpi=300)
plt.close()

# 5. Contour Visualization
plt.imshow(D_norm, cmap='turbo')
plt.contour(D_norm, levels=5, colors='white')
plt.title("Contour Visualization")
plt.savefig(os.path.join(output_dir, "05_contour.png"), dpi=300)
plt.close()

# 6. Thresholding (Localization)
tau = np.percentile(D_norm, 90)

mask = (D_norm > tau).astype(np.uint8) * 255

cv2.imwrite(
    os.path.join(output_dir, "06_threshold.png"),
    mask
)

print("All results saved in:", output_dir)