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
    os.path.join(output_dir, "difference_raw.png"),
    (D_norm * 255).astype(np.uint8)
)

# 2. Pseudo-color (Heatmap)
plt.imshow(D_norm, cmap='turbo')
plt.colorbar()
plt.title("Difference Heatmap")
plt.savefig(os.path.join(output_dir, "heatmap.png"), dpi=300)
plt.close()

# 3. Smoothed Difference
D_smooth = cv2.GaussianBlur(D_norm, (5, 5), 0)

cv2.imwrite(
    os.path.join(output_dir, "smooth.png"),
    (D_smooth * 255).astype(np.uint8)
)

# 4. Contour Visualization
plt.imshow(D_norm, cmap='turbo')
plt.contour(D_norm, levels=5, colors='white')
plt.title("Contour Visualization")
plt.savefig(os.path.join(output_dir, "contour.png"), dpi=300)
plt.close()

# 5. Thresholding (Localization)
tau = np.percentile(D_norm, 98)

mask = (D_norm > tau).astype(np.uint8) * 255

cv2.imwrite(
    os.path.join(output_dir, "threshold.png"),
    mask
)

# 6. Gradient Magnitude
gy, gx = np.gradient(D_norm)
grad_mag = np.sqrt(gx**2 + gy**2)

plt.imshow(grad_mag, cmap='gray')
plt.title("Gradient Magnitude")
plt.savefig(os.path.join(output_dir, "gradient.png"), dpi=300)
plt.close()

print("All results saved in:", output_dir)
