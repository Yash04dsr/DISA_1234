import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QSlider, QHBoxLayout, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


# ============================ Image / Geometry Utils ============================

def adaptive_drop_mask(bgr):
    """Create a robust binary mask for the droplet under uneven lighting."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    mask = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        51, 2
    )
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask


def choose_droplet_contour(mask):
    """Pick the most plausible droplet contour by area × circularity score."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    best, best_score = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        per = cv2.arcLength(c, True)
        circularity = 4.0 * np.pi * area / (per * per + 1e-6)
        score = area * circularity
        if score > best_score:
            best, best_score = c, score
    return best


def find_auto_baseline_from_contour(cnt):
    """Use a high percentile of contour y to place the baseline near the bottom."""
    ys = cnt[:, 0, 1]
    return int(np.percentile(ys, 95))


def refine_contact_point(gray, point, search_radius=5):
    """Shift the point to the location of maximum intensity gradient magnitude."""
    if point is None:
        return None
    x0, y0 = int(point[0]), int(point[1])
    h, w = gray.shape
    best = (x0, y0)
    best_g = 0
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            x, y = x0 + dx, y0 + dy
            if 1 <= x < w - 1 and 1 <= y < h - 1:
                gx = int(gray[y, x + 1]) - int(gray[y, x - 1])
                gy = int(gray[y + 1, x]) - int(gray[y - 1, x])
                g2 = gx * gx + gy * gy
                if g2 > best_g:
                    best_g = g2
                    best = (x, y)
    return best


def fit_circle_and_intersect(cnt, baseline_y, gray=None, refine=True):
    """
    Fit a circle to the *upper* contour (avoid distorted base) and intersect with baseline.
    Optionally refine intersection locations by local gradient.
    """
    pts = cnt[:, 0, :].astype(np.float32)
    if len(pts) < 10:
        return None, None

    # keep only the upper 70% of points (y smaller = higher in image)
    y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
    cutoff = y_min + 0.70 * (y_max - y_min)
    upper = pts[pts[:, 1] < cutoff]
    if len(upper) < 10:
        upper = pts  # fallback

    (cx, cy), r = cv2.minEnclosingCircle(upper)

    # Intersections of (x - cx)^2 + (y - cy)^2 = r^2 with y = baseline_y
    dy = baseline_y - cy
    if abs(dy) >= r:
        return None, None  # circle doesn't cross baseline
    dx = float(np.sqrt(max(r * r - dy * dy, 0.0)))

    left  = (int(round(cx - dx)), int(baseline_y))
    right = (int(round(cx + dx)), int(baseline_y))

    if refine and gray is not None:
        left  = refine_contact_point(gray, left, 5)
        right = refine_contact_point(gray, right, 5)

    return left, right


def local_poly_contact_angle(cnt, contact_pt, window_px=25):
    """
    Local quadratic fit around contact point, then angle = atan(|dy/dx|).
    Returns angle in degrees.
    """
    if contact_pt is None:
        return None
    cx, _ = contact_pt
    pts = cnt[:, 0, :]
    nb = pts[np.abs(pts[:, 0] - cx) < window_px]
    if len(nb) < 8:
        nb = pts[np.abs(pts[:, 0] - cx) < window_px * 1.8]
    if len(nb) < 8:
        return None

    x = nb[:, 0].astype(np.float64)
    y = nb[:, 1].astype(np.float64)
    try:
        a, b, c = np.polyfit(x, y, 2)
    except Exception:
        return None

    m = 2.0 * a * cx + b
    theta = np.degrees(np.arctan(np.abs(m)))
    return float(np.clip(theta, 0.0, 175.0))


def apex_of_contour(cnt):
    """Highest point (minimum y)."""
    if cnt is None or len(cnt) == 0:
        return None
    idx = np.argmin(cnt[:, 0, 1])
    return tuple(cnt[idx, 0, :])


# ============================ PyQt5 Application ============================

class SessileDropAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sessile Drop Contact Angle Analyzer (Circle-Fit + Refinement)")
        self.setGeometry(100, 100, 1150, 830)

        self.original_image = None
        self.current_pixmap = None
        self.overlay = None

        # --- Layout / Widgets ---
        layout = QVBoxLayout(self)
        self.setLayout(layout)

        self.image_label = QLabel("Load a side-view droplet image.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #888;")
        self.image_label.setMinimumSize(720, 460)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, stretch=1)

        controls = QHBoxLayout()
        layout.addLayout(controls)
        self.btn_load = QPushButton("Load Image")
        controls.addWidget(self.btn_load)
        controls.addWidget(QLabel("Baseline (y):"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        controls.addWidget(self.slider)

        results = QHBoxLayout()
        layout.addLayout(results)
        self.left_angle_label = QLabel("Left Angle: --°")
        self.right_angle_label = QLabel("Right Angle: --°")
        results.addWidget(self.left_angle_label)
        results.addWidget(self.right_angle_label)

        # --- Signals ---
        self.btn_load.clicked.connect(self.load_image)
        self.slider.valueChanged.connect(self.preview_baseline_only)
        self.slider.sliderReleased.connect(self.run_full_analysis)

    # ------------------------------- UI Actions -------------------------------

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self.image_label.setText("Error: Could not read image.")
            return
        h, w = img.shape[:2]
        if w > 1400:
            img = cv2.resize(img, (1400, int(h * 1400 / w)), interpolation=cv2.INTER_AREA)
        self.original_image = img
        self.run_full_analysis(initial=True)

    def preview_baseline_only(self):
        if self.overlay is None:
            return
        y = self.slider.value()
        temp = self.overlay.copy()
        cv2.line(temp, (0, y), (temp.shape[1], y), (255, 0, 0), 2)
        self.display_image(temp)

    # ----------------------------- Core Processing -----------------------------

    def run_full_analysis(self, initial=False):
        if self.original_image is None:
            return

        img = self.original_image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1) Mask & contour
        mask = adaptive_drop_mask(img)
        cnt = choose_droplet_contour(mask)
        if cnt is None:
            self.left_angle_label.setText("Left Angle: --°")
            self.right_angle_label.setText("Right Angle: --°")
            self.display_image(img)
            return

        # 2) Baseline: auto then user-tunable
        auto_y = find_auto_baseline_from_contour(cnt)
        if initial or not self.slider.isEnabled():
            h = img.shape[0]
            self.slider.blockSignals(True)
            self.slider.setRange(0, h - 1)
            self.slider.setValue(int(np.clip(auto_y, 0, h - 1)))
            self.slider.setEnabled(True)
            self.slider.blockSignals(False)
        baseline_y = int(self.slider.value())

        # 3) Contact points by circle-fit intersection + gradient refinement
        l_pt, r_pt = fit_circle_and_intersect(cnt, baseline_y, gray, refine=True)

        # Fallback: if circle intersection fails, try slope-based near baseline
        if (l_pt is None) or (r_pt is None):
            # Simple fallback: choose lowest y on left/right halves
            pts = cnt[:, 0, :]
            xs = pts[:, 0]
            medx = np.median(xs)
            left_half = pts[xs <= medx]
            right_half = pts[xs >= medx]
            if len(left_half):
                l_pt = tuple(left_half[np.argmax(left_half[:, 1])])
            if len(right_half):
                r_pt = tuple(right_half[np.argmax(right_half[:, 1])])
            # refine these as well
            l_pt = refine_contact_point(gray, l_pt, 5) if l_pt is not None else None
            r_pt = refine_contact_point(gray, r_pt, 5) if r_pt is not None else None

        # 4) Angles by local quadratic tangent
        left_ang = local_poly_contact_angle(cnt, l_pt, window_px=25)
        right_ang = local_poly_contact_angle(cnt, r_pt, window_px=25)
        self.left_angle_label.setText(
            f"Left Angle: {left_ang:.2f}°" if left_ang is not None else "Left Angle: --°"
        )
        self.right_angle_label.setText(
            f"Right Angle: {right_ang:.2f}°" if right_ang is not None else "Right Angle: --°"
        )

        # 5) Draw overlay & labels
        vis = img.copy()
        cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
        cv2.line(vis, (0, baseline_y), (vis.shape[1], baseline_y), (255, 0, 0), 2)

        apx = apex_of_contour(cnt)
        if apx is not None:
            cv2.circle(vis, apx, 7, (255, 255, 0), -1)

        for p, txt in [(l_pt, left_ang), (r_pt, right_ang)]:
            if p is not None:
                cv2.circle(vis, (int(p[0]), int(p[1])), 8, (0, 0, 255), -1)
                # put angle text slightly above/right of the point
                if txt is not None:
                    label = f"{txt:.1f}°"
                    px, py = int(p[0]), int(p[1])
                    cv2.putText(vis, label, (px + 8, max(py - 10, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        self.overlay = vis
        self.display_image(vis)

    # ------------------------------- Qt Helpers -------------------------------

    def display_image(self, img_bgr):
        h, w = img_bgr.shape[:2]
        qimg = QImage(img_bgr.data, w, h, 3 * w, QImage.Format_BGR888)
        self.current_pixmap = QPixmap.fromImage(qimg)
        scaled = self.current_pixmap.scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_pixmap is not None:
            scaled = self.current_pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)


# ================================== Main ==================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SessileDropAnalyzer()
    w.show()
    sys.exit(app.exec_())
