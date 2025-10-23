# import sys
# import cv2
# import numpy as np
# from PyQt5.QtWidgets import (
#     QApplication, QWidget, QVBoxLayout, QPushButton,
#     QLabel, QFileDialog, QSlider, QHBoxLayout, QSizePolicy
# )
# from PyQt5.QtGui import QPixmap, QImage
# from PyQt5.QtCore import Qt


# class SessileDropAnalyzer(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle('Sessile Drop Contact Angle Analyzer (Responsive UI)')
#         self.setGeometry(100, 100, 1000, 800)

#         # --- State Management ---
#         self.original_image = None
#         self.last_analyzed_image = None
#         self.current_pixmap = None

#         # --- Layout Setup ---
#         self.layout = QVBoxLayout()
#         self.setLayout(self.layout)

#         # --- Image Display ---
#         self.image_label = QLabel('Please load an image of a droplet.')
#         self.image_label.setAlignment(Qt.AlignCenter)
#         self.image_label.setStyleSheet("border: 2px solid gray;")
#         self.image_label.setScaledContents(False)        # Prevent QLabel from auto-scaling
#         self.image_label.setMinimumSize(600, 400)        # Minimum size to prevent collapsing
#         self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         self.layout.addWidget(self.image_label, stretch=1)

#         # --- Controls ---
#         controls_layout = QHBoxLayout()
#         self.btn_load = QPushButton('Load Image')
#         controls_layout.addWidget(self.btn_load)

#         self.slider_label = QLabel('Baseline Position:')
#         controls_layout.addWidget(self.slider_label)

#         self.slider = QSlider(Qt.Horizontal)
#         self.slider.setEnabled(False)
#         controls_layout.addWidget(self.slider)

#         self.layout.addLayout(controls_layout)

#         # --- Results ---
#         results_layout = QHBoxLayout()
#         self.left_angle_label = QLabel('Left Angle: --°')
#         self.right_angle_label = QLabel('Right Angle: --°')
#         results_layout.addWidget(self.left_angle_label)
#         results_layout.addWidget(self.right_angle_label)
#         self.layout.addLayout(results_layout)

#         # --- Connections ---
#         self.btn_load.clicked.connect(self.load_image)
#         self.slider.sliderReleased.connect(self.update_analysis)
#         self.slider.valueChanged.connect(self.preview_baseline)

#     # ------------------------------------------------------------

#     def load_image(self):
#         """Load and preprocess image from file dialog."""
#         filepath, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.png *.jpg *.bmp)')
#         if not filepath:
#             return

#         self.original_image = cv2.imread(filepath)
#         if self.original_image is None:
#             self.image_label.setText("Error: Could not load image.")
#             return

#         MAX_WIDTH = 1200
#         h, w, _ = self.original_image.shape
#         if w > MAX_WIDTH:
#             self.original_image = cv2.resize(self.original_image,
#                                              (MAX_WIDTH, int(h * MAX_WIDTH / w)),
#                                              interpolation=cv2.INTER_AREA)

#         h, _, _ = self.original_image.shape
#         self.slider.blockSignals(True)
#         self.slider.setRange(0, h - 1)
#         self.slider.setValue(int(h * 0.85))
#         self.slider.setEnabled(True)
#         self.slider.blockSignals(False)

#         # Perform first analysis
#         self.update_analysis()

#     # ------------------------------------------------------------

#     def preview_baseline(self):
#         """Lightweight visual update: only draws baseline."""
#         if self.last_analyzed_image is None:
#             return

#         preview_img = self.last_analyzed_image.copy()
#         baseline_y = self.slider.value()
#         cv2.line(preview_img, (0, baseline_y), (preview_img.shape[1], baseline_y), (255, 0, 0), 2)
#         self.display_image(preview_img)

#     # ------------------------------------------------------------

#     def update_analysis(self):
#         """Re-run full contour analysis and update result display."""
#         if self.original_image is None:
#             return

#         analysis_img = self.original_image.copy()
#         baseline_y = self.slider.value()

#         gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (7, 7), 0)
#         _, thresh = cv2.threshold(blurred, 0, 255,
#                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
#                                        cv2.CHAIN_APPROX_SIMPLE)
#         if not contours:
#             self.left_angle_label.setText('Left Angle: --°')
#             self.right_angle_label.setText('Right Angle: --°')
#             self.last_analyzed_image = self.original_image.copy()
#             self.preview_baseline()
#             return

#         droplet_contour = max(contours, key=cv2.contourArea)
#         epsilon = 0.001 * cv2.arcLength(droplet_contour, True)
#         approx_contour = cv2.approxPolyDP(droplet_contour, epsilon, True)

#         # Apex and intersection points
#         apex_point = tuple(approx_contour[approx_contour[:, :, 1].argmin()][0])
#         intersection_points = [tuple(p[0]) for p in approx_contour
#                                if abs(p[0][1] - baseline_y) < 5]

#         left_contact_point, right_contact_point = None, None
#         if len(intersection_points) >= 2:
#             intersection_points.sort()
#             left_contact_point = intersection_points[0]
#             right_contact_point = intersection_points[-1]

#         # Calculate angles
#         l_angle = self.calculate_contact_angle(approx_contour,
#                                                left_contact_point, 'left') if left_contact_point else None
#         r_angle = self.calculate_contact_angle(approx_contour,
#                                                right_contact_point, 'right') if right_contact_point else None

#         self.left_angle_label.setText(f'Left Angle: {l_angle:.2f}°'
#                                       if l_angle is not None else 'Left Angle: --°')
#         self.right_angle_label.setText(f'Right Angle: {r_angle:.2f}°'
#                                        if r_angle is not None else 'Right Angle: --°')

#         # Draw results
#         cv2.drawContours(analysis_img, [approx_contour], -1, (0, 255, 0), 2)
#         cv2.circle(analysis_img, apex_point, 8, (255, 255, 0), -1)
#         if left_contact_point:
#             cv2.circle(analysis_img, left_contact_point, 8, (0, 0, 255), -1)
#         if right_contact_point:
#             cv2.circle(analysis_img, right_contact_point, 8, (0, 0, 255), -1)

#         # Save and preview
#         self.last_analyzed_image = analysis_img
#         self.preview_baseline()

#     # ------------------------------------------------------------

#     def calculate_contact_angle(self, contour, contact_point, side):
#         """Estimate local tangent contact angle from contour."""
#         x, y, w, h = cv2.boundingRect(contour)
#         top_contour = [p for p in contour if p[0][1] < y + h * 0.75]
#         if len(top_contour) < 5:
#             return None

#         (cx, cy), radius = cv2.minEnclosingCircle(np.array(top_contour))
#         p1 = np.array(contact_point)
#         normal_vector = p1 - np.array([cx, cy])

#         if np.isclose(normal_vector[0], 0):
#             return 90.0

#         angle_rad = np.arctan2(normal_vector[1], normal_vector[0])
#         angle_deg = np.degrees(angle_rad)
#         return (180 - angle_deg) if side == 'left' else angle_deg

#     # ------------------------------------------------------------

#     def display_image(self, img_cv):
#         """Safely convert OpenCV image to QPixmap and display."""
#         h, w, ch = img_cv.shape
#         bytes_per_line = ch * w
#         q_img = QImage(img_cv.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
#         self.current_pixmap = QPixmap.fromImage(q_img)

#         scaled_pixmap = self.current_pixmap.scaled(
#             self.image_label.width(),
#             self.image_label.height(),
#             Qt.KeepAspectRatio,
#             Qt.SmoothTransformation
#         )
#         self.image_label.setPixmap(scaled_pixmap)

#     # ------------------------------------------------------------

#     def resizeEvent(self, event):
#         """Re-render image on window resize without zoom loops."""
#         super().resizeEvent(event)
#         if self.current_pixmap:
#             scaled_pixmap = self.current_pixmap.scaled(
#                 self.image_label.width(),
#                 self.image_label.height(),
#                 Qt.KeepAspectRatio,
#                 Qt.SmoothTransformation
#             )
#             self.image_label.setPixmap(scaled_pixmap)


# # ------------------------------------------------------------

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     analyzer = SessileDropAnalyzer()
#     analyzer.show()
#     sys.exit(app.exec_())

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QSlider, QHBoxLayout, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


# ------------------------------ Core helpers ------------------------------

def adaptive_drop_mask(bgr):
    """
    Robust foreground (drop) mask under uneven lighting and glare.
    Returns a clean binary mask (uint8 0/255).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive thresholding handles background gradients
    mask = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        51, 2
    )

    # Morphology to remove speckles and close tiny gaps
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)

    return mask


def choose_droplet_contour(mask):
    """
    Pick the most droplet-like contour using area × circularity scoring.
    Returns a contour with CHAIN_APPROX_NONE resolution (no simplification).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    best, best_score = None, 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # ignore tiny blobs
            continue
        peri = cv2.arcLength(cnt, True)
        circularity = 4.0 * np.pi * area / (peri * peri + 1e-6)
        score = area * circularity  # favors large & smooth-ish blobs
        if score > best_score:
            best, best_score = cnt, score

    return best


def find_auto_baseline_from_contour(cnt):
    """
    Estimate the solid–liquid baseline y from the lower part of the contour.
    We take a high percentile of y (image y increases downwards).
    """
    ys = cnt[:, 0, 1]
    # robust "near bottom" line; 95th percentile works well for most cases
    baseline_y = int(np.percentile(ys, 95))
    return baseline_y


def intersection_points_with_baseline(cnt, baseline_y, tol=3):
    """
    Find left/right contour points that lie on (or very near) the baseline.
    Returns (left_pt, right_pt) or (None, None) if not enough.
    """
    pts = cnt[:, 0, :]  # (N, 2) as (x, y)
    # collect candidates within ±tol pixels of baseline
    cand = pts[np.abs(pts[:, 1] - baseline_y) <= tol]
    if cand.shape[0] < 2:
        # fallback: take lowest points on left/right halves
        xs = pts[:, 0]
        median_x = np.median(xs)
        left_half  = pts[xs <= median_x]
        right_half = pts[xs >= median_x]
        if left_half.size and right_half.size:
            left_pt  = tuple(left_half[np.argmax(left_half[:, 1])])
            right_pt = tuple(right_half[np.argmax(right_half[:, 1])])
            return left_pt, right_pt
        return None, None

    # sort by x and pick extremes
    cand = cand[np.argsort(cand[:, 0])]
    left_pt = tuple(cand[0])
    right_pt = tuple(cand[-1])
    return left_pt, right_pt


def local_poly_contact_angle(cnt, contact_pt, window_px=25):
    """
    Estimate contact angle from a local quadratic fit around the contact point.

    - Select contour points within |x - x_c| < window_px
    - Fit y = ax^2 + bx + c (least squares)
    - Slope m = dy/dx at x_c = 2a*x_c + b
    - Contact angle (inside liquid) relative to horizontal: theta = atan(|m|)
    Returns theta in degrees (0..180)
    """
    if contact_pt is None:
        return None

    cx, cy = contact_pt
    pts = cnt[:, 0, :]  # (N, 2)
    neighborhood = pts[np.abs(pts[:, 0] - cx) < window_px]

    # if neighborhood is too small, widen once
    if neighborhood.shape[0] < 8:
        neighborhood = pts[np.abs(pts[:, 0] - cx) < window_px * 1.8]
    if neighborhood.shape[0] < 8:
        return None

    x = neighborhood[:, 0].astype(np.float64)
    y = neighborhood[:, 1].astype(np.float64)

    # robust quadratic fit (ordinary least squares is fine here)
    try:
        a, b, c = np.polyfit(x, y, 2)
    except Exception:
        return None

    m = 2.0 * a * cx + b  # slope dy/dx at the contact point
    theta = np.degrees(np.arctan(np.abs(m)))  # 0..90+ depending on slope

    # Clamp for sanity
    theta = float(np.clip(theta, 0.0, 175.0))
    return theta


def apex_of_contour(cnt):
    """Apex is the highest (minimum y) point on the contour."""
    if cnt is None or len(cnt) == 0:
        return None
    idx = np.argmin(cnt[:, 0, 1])
    return tuple(cnt[idx, 0, :])


# ------------------------------ UI Application ------------------------------

class SessileDropAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Sessile Drop Contact Angle Analyzer (Improved)')
        self.setGeometry(100, 100, 1100, 820)

        # State
        self.original_image = None
        self.draw_image = None
        self.current_pixmap = None
        self.auto_baseline_y = None

        # Layout
        layout = QVBoxLayout(self)
        self.setLayout(layout)

        # Image display
        self.image_label = QLabel('Load a droplet image (side view).')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #888;")
        self.image_label.setScaledContents(False)
        self.image_label.setMinimumSize(700, 450)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, stretch=1)

        # Controls
        controls = QHBoxLayout()
        layout.addLayout(controls)

        self.btn_load = QPushButton('Load Image')
        controls.addWidget(self.btn_load)

        self.slider_label = QLabel('Baseline (y):')
        controls.addWidget(self.slider_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        controls.addWidget(self.slider)

        # Results
        results = QHBoxLayout()
        layout.addLayout(results)
        self.left_angle_label = QLabel('Left Angle: --°')
        self.right_angle_label = QLabel('Right Angle: --°')
        results.addWidget(self.left_angle_label)
        results.addWidget(self.right_angle_label)

        # Signals
        self.btn_load.clicked.connect(self.load_image)
        self.slider.valueChanged.connect(self.preview_baseline_only)
        self.slider.sliderReleased.connect(self.run_full_analysis)

    # --------------------------- UI handlers ---------------------------

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self.image_label.setText('Error: Could not read image.')
            return

        # Downscale if very wide
        MAX_W = 1400
        h, w = img.shape[:2]
        if w > MAX_W:
            img = cv2.resize(img, (MAX_W, int(h * MAX_W / w)), interpolation=cv2.INTER_AREA)

        self.original_image = img
        self.run_full_analysis(initial=True)

    def preview_baseline_only(self):
        if self.draw_image is None:
            return
        y = self.slider.value()
        img = self.draw_image.copy()
        cv2.line(img, (0, y), (img.shape[1], y), (255, 0, 0), 2)
        self.display_image(img)

    def run_full_analysis(self, initial=False):
        if self.original_image is None:
            return

        img = self.original_image.copy()
        vis = img.copy()

        # 1) Robust foreground mask
        mask = adaptive_drop_mask(img)

        # 2) Pick droplet contour
        cnt = choose_droplet_contour(mask)
        if cnt is None:
            self.left_angle_label.setText('Left Angle: --°')
            self.right_angle_label.setText('Right Angle: --°')
            self.draw_image = img
            self.preview_baseline_only()
            return

        # 3) Baseline (automatic, with slider to fine-tune)
        auto_baseline = find_auto_baseline_from_contour(cnt)
        self.auto_baseline_y = int(auto_baseline)

        # Initialize slider range/value on first analysis
        if initial or not self.slider.isEnabled():
            h = img.shape[0]
            self.slider.blockSignals(True)
            self.slider.setRange(0, h - 1)
            # set near the detected baseline but keep within image
            self.slider.setValue(int(np.clip(self.auto_baseline_y, 0, h - 1)))
            self.slider.setEnabled(True)
            self.slider.blockSignals(False)

        baseline_y = int(self.slider.value())

        # 4) Intersection/contact points
        left_pt, right_pt = intersection_points_with_baseline(cnt, baseline_y, tol=3)

        # 5) Angles via local polynomial tangent
        left_angle = local_poly_contact_angle(cnt, left_pt, window_px=25)
        right_angle = local_poly_contact_angle(cnt, right_pt, window_px=25)

        # Labels
        self.left_angle_label.setText(f'Left Angle: {left_angle:.2f}°' if left_angle is not None else 'Left Angle: --°')
        self.right_angle_label.setText(f'Right Angle: {right_angle:.2f}°' if right_angle is not None else 'Right Angle: --°')

        # 6) Draw overlay
        cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)                 # droplet contour
        cv2.line(vis, (0, baseline_y), (vis.shape[1], baseline_y), (255, 0, 0), 2)  # baseline

        apx = apex_of_contour(cnt)
        if apx is not None:
            cv2.circle(vis, apx, 8, (255, 255, 0), -1)                   # apex

        if left_pt is not None:
            cv2.circle(vis, tuple(map(int, left_pt)), 8, (0, 0, 255), -1)
        if right_pt is not None:
            cv2.circle(vis, tuple(map(int, right_pt)), 8, (0, 0, 255), -1)

        # keep for preview scaling
        self.draw_image = vis
        self.display_image(vis)

    # --------------------------- utilities ---------------------------

    def display_image(self, img_bgr):
        h, w = img_bgr.shape[:2]
        qimg = QImage(img_bgr.data, w, h, 3 * w, QImage.Format_BGR888)
        self.current_pixmap = QPixmap.fromImage(qimg)

        scaled = self.current_pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_pixmap is not None:
            scaled = self.current_pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)


# ------------------------------ main ------------------------------

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = SessileDropAnalyzer()
    w.show()
    sys.exit(app.exec_())
