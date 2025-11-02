import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QSlider, QHBoxLayout, QSizePolicy, QScrollArea,
    QMainWindow, QFileDialog, QFrame, QGroupBox, QFormLayout,
    QSpacerItem
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QSize


# ============================ Image / Geometry Utils ============================
# (All your OpenCV helper functions are unchanged as they are excellent)

def adaptive_drop_mask(bgr):
    """Creates a binary mask of the droplet using adaptive thresholding."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # Adaptive thresholding to handle varying lighting
    mask = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        51, 2
    )
    # Morphological closing and opening to remove noise
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask


def choose_droplet_contour(mask):
    """Finds all contours and chooses the one most likely to be the droplet."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    
    best, best_score = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        # Filter out tiny contours
        if area < 500:
            continue
        
        per = cv2.arcLength(c, True)
        circularity = 4.0 * np.pi * area / (per * per + 1e-6)
        
        # Score favors large, circular objects
        score = area * circularity
        if score > best_score:
            best, best_score = c, score
            
    return best


def find_auto_baseline_from_contour(cnt):
    """Estimates baseline from the bottom 5% of contour points."""
    ys = cnt[:, 0, 1]
    return int(np.percentile(ys, 95))


def refine_contact_point(gray, point, search_radius=5):
    """Finds the max gradient location near an estimated contact point."""
    if point is None:
        return None
    
    x0, y0 = int(point[0]), int(point[1])
    h, w = gray.shape
    best = (x0, y0)
    best_g = 0
    
    # Search in a small window
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            x, y = x0 + dx, y0 + dy
            # Check bounds
            if 1 <= x < w - 1 and 1 <= y < h - 1:
                # Sobel-like gradient calculation
                gx = int(gray[y, x + 1]) - int(gray[y, x - 1])
                gy = int(gray[y + 1, x]) - int(gray[y - 1, x])
                g2 = gx * gx + gy * gy
                
                if g2 > best_g:
                    best_g = g2
                    best = (x, y)
    return best


def fit_circle_and_intersect(cnt, baseline_y, gray=None, refine=True):
    """
    Fits a circle to the top part of the contour and finds its
    intersection with the baseline.
    """
    pts = cnt[:, 0, :].astype(np.float32)
    if len(pts) < 10:
        return None, None
        
    # Use only the upper 70% of the drop for circle fitting
    # This avoids distortion from the substrate
    y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
    cutoff = y_min + 0.70 * (y_max - y_min)
    upper = pts[pts[:, 1] < cutoff]
    
    if len(upper) < 10:
        upper = pts # Fallback to all points if filter is too aggressive
        
    (cx, cy), r = cv2.minEnclosingCircle(upper)
    
    # Find intersection of circle and horizontal baseline
    dy = baseline_y - cy
    if abs(dy) >= r:
        return None, None # No intersection
        
    dx = np.sqrt(max(r * r - dy * dy, 0.0))
    left = (int(round(cx - dx)), int(baseline_y))
    right = (int(round(cx + dx)), int(baseline_y))
    
    # Refine points to max gradient
    if refine and gray is not None:
        left = refine_contact_point(gray, left, 5)
        right = refine_contact_point(gray, right, 5)
        
    return left, right


def local_poly_contact_angle(cnt, contact_pt, window_px=25):
    """
    Calculates the contact angle using a 2nd-degree polynomial fit
    in a small window around the contact point.
    """
    if contact_pt is None:
        return None
        
    cx, _ = contact_pt
    pts = cnt[:, 0, :]
    
    # Get points in a window around the contact point
    nb = pts[np.abs(pts[:, 0] - cx) < window_px]
    if len(nb) < 8:
        # Fallback to a wider window if needed
        nb = pts[np.abs(pts[:, 0] - cx) < window_px * 1.8]
    if len(nb) < 8:
        return None # Not enough points to fit

    x = nb[:, 0].astype(np.float64)
    y = nb[:, 1].astype(np.float64)
    
    try:
        # Fit y = ax^2 + bx + c
        a, b, c = np.polyfit(x, y, 2)
    except Exception:
        return None
        
    # Get the slope m = dy/dx = 2ax + b at the contact point
    m = 2.0 * a * cx + b
    
    # Angle is arctan of the absolute slope
    theta = np.degrees(np.arctan(np.abs(m)))
    
    return float(np.clip(theta, 0.0, 175.0))


def apex_of_contour(cnt):
    """Finds the highest point (minimum y-value) of the contour."""
    if cnt is None or len(cnt) == 0:
        return None
    idx = np.argmin(cnt[:, 0, 1])
    return tuple(cnt[idx, 0, :])


def detect_auto_baseline(gray):
    """
    Detects the substrate baseline using a Sobel filter.
    Strongest horizontal edges in the bottom half of the image.
    """
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.uint8(np.absolute(sobel_y))
    edges = cv2.GaussianBlur(edges, (9, 9), 0)
    
    h = gray.shape[0]
    bottom_half = edges[int(h * 0.5):, :]
    
    # Sum edge strength for each row
    row_strength = np.mean(bottom_half, axis=1)
    
    # Find rows with high edge strength
    y_rel = np.where(row_strength > np.percentile(row_strength, 95))[0]
    if len(y_rel) == 0:
        return int(h * 0.9) # Fallback
        
    # Baseline is the median of the strong edge rows
    baseline_y = int(h * 0.5 + np.median(y_rel))
    return baseline_y


# ============================ Main Analyzer GUI ============================

class SessileDropAnalyzer(QMainWindow):
    
    APP_STYLE = """
    QMainWindow {
        background-color: #2E2E2E;
    }
    QGroupBox {
        color: #E0E0E0;
        font-weight: bold;
        border: 1px solid #555;
        border-radius: 5px;
        margin-top: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px 0 5px;
        left: 10px;
    }
    QLabel {
        color: #E0E0E0;
        font-size: 14px;
    }
    QPushButton {
        background-color: #4A4A4A;
        color: #E0E0E0;
        border: 1px solid #666;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 14px;
    }
    QPushButton:hover {
        background-color: #5A5A5A;
        border: 1px solid #777;
    }
    QPushButton:pressed {
        background-color: #3A3A3A;
    }
    QPushButton:disabled {
        background-color: #3C3C3C;
        color: #777;
        border: 1px solid #555;
    }
    QSlider::groove:horizontal {
        border: 1px solid #555;
        height: 8px;
        background: #3C3C3C;
        margin: 2px 0;
        border-radius: 4px;
    }
    QSlider::handle:horizontal {
        background: #0078D7;
        border: 1px solid #0078D7;
        width: 16px;
        margin: -4px 0;
        border-radius: 8px;
    }
    QScrollArea {
        border: 1px solid #555;
    }
    /* Style for the result labels */
    QLabel#ResultLabel {
        color: #2D9CDB; /* A bright blue */
        font-weight: bold;
        font-size: 15px;
    }
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sessile Drop Contact Angle Analyzer")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(self.APP_STYLE)

        self.original_image = None
        self.current_pixmap = None

        # Create main central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Control Panel ---
        self.control_panel = QFrame()
        self.control_panel.setFixedWidth(350)
        self.control_panel.setLayout(QVBoxLayout())
        main_layout.addWidget(self.control_panel)

        # File Group
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout()
        self.btn_load = QPushButton("Load Image...")
        file_layout.addWidget(self.btn_load)
        file_group.setLayout(file_layout)
        self.control_panel.layout().addWidget(file_group)

        # Analysis Group
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QFormLayout()
        self.btn_auto_baseline = QPushButton("Auto-Detect Baseline")
        analysis_layout.addRow(self.btn_auto_baseline)
        
        self.slider = QSlider(Qt.Horizontal)
        analysis_layout.addRow("Baseline (y):", self.slider)
        analysis_group.setLayout(analysis_layout)
        self.control_panel.layout().addWidget(analysis_group)

        # Results Group
        results_group = QGroupBox("Results")
        results_layout = QFormLayout()
        self.left_angle_label = QLabel("--°")
        self.left_angle_label.setObjectName("ResultLabel")
        self.right_angle_label = QLabel("--°")
        self.right_angle_label.setObjectName("ResultLabel")
        self.avg_angle_label = QLabel("--°")
        self.avg_angle_label.setObjectName("ResultLabel")
        self.apex_label = QLabel("--")
        self.apex_label.setObjectName("ResultLabel")
        
        results_layout.addRow("Left Angle:", self.left_angle_label)
        results_layout.addRow("Right Angle:", self.right_angle_label)
        results_layout.addRow("Avg Angle:", self.avg_angle_label)
        results_layout.addRow("Drop Apex (x, y):", self.apex_label)
        results_group.setLayout(results_layout)
        self.control_panel.layout().addWidget(results_group)
        
        # Spacer to push all groups to the top
        self.control_panel.layout().addSpacerItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        # --- Right Image Panel ---
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True) # This is key
        self.scroll_area.setStyleSheet("background-color: #222;")
        
        self.image_label = QLabel("Click 'Load Image' to start...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("color: #999; font-size: 16px;")
        
        self.scroll_area.setWidget(self.image_label)
        main_layout.addWidget(self.scroll_area, stretch=1)

        # --- Signals ---
        self.btn_load.clicked.connect(self.load_image_dialog)
        self.btn_auto_baseline.clicked.connect(self.auto_detect_baseline)
        self.slider.valueChanged.connect(self.preview_baseline_only)
        self.slider.sliderReleased.connect(self.run_full_analysis)

        # --- Initial State ---
        self.clear_results() # Disables controls
        self.load_image_from_path(os.path.expanduser("~/sessile.png"))


    def load_image_dialog(self):
        """Opens a file dialog for the user to select an image."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if path:
            self.load_image_from_path(path)

    def load_image_from_path(self, path):
        """Loads and processes an image from a given file path."""
        if not os.path.exists(path):
            if not self.original_image: # Only show error if no image is loaded
                self.image_label.setText(f"File not found:\n{path}")
            return
            
        img = cv2.imread(path)
        if img is None:
            self.image_label.setText(f"Error reading image:\n{path}")
            self.clear_results()
            return
            
        h, w = img.shape[:2]
        # Optional: Resize very large images to prevent performance issues
        if w > 3000:
            scale = 3000 / w
            img = cv2.resize(img, (3000, int(h * scale)), interpolation=cv2.INTER_AREA)
            
        self.original_image = img
        self.image_label.setText("") # Clear "Loading..." text
        self.run_full_analysis(initial=True)

    def clear_results(self):
        """Resets the UI to a clean state and disables controls."""
        self.left_angle_label.setText("--°")
        self.right_angle_label.setText("--°")
        self.avg_angle_label.setText("--°")
        self.apex_label.setText("--")
        self.slider.setEnabled(False)
        self.btn_auto_baseline.setEnabled(False)
        if not self.original_image:
            self.image_label.setPixmap(QPixmap()) # Clear image
            self.image_label.setText("Click 'Load Image' to start...")

    def auto_detect_baseline(self):
        """Runs the Sobel-based baseline detection and updates the slider."""
        if self.original_image is None:
            return
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        baseline_y = detect_auto_baseline(gray)
        self.slider.setValue(baseline_y)
        self.run_full_analysis()

    def run_full_analysis(self, initial=False):
        """
        The main analysis pipeline. Finds contour, baseline, contact points,
        and angles, then updates the display.
        """
        if self.original_image is None:
            self.clear_results()
            return
            
        img = self.original_image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = adaptive_drop_mask(img)
        cnt = choose_droplet_contour(mask)

        if cnt is None:
            self.display_image(self.original_image)
            self.clear_results()
            self.slider.setEnabled(True) # Keep slider enabled
            self.btn_auto_baseline.setEnabled(True)
            self.image_label.setText("No droplet contour found.")
            return

        # Set up slider range and initial value
        auto_y = find_auto_baseline_from_contour(cnt)
        if initial or not self.slider.isEnabled():
            h = img.shape[0]
            self.slider.blockSignals(True)
            self.slider.setRange(0, h - 1)
            self.slider.setValue(int(np.clip(auto_y, 0, h - 1)))
            self.slider.setEnabled(True)
            self.btn_auto_baseline.setEnabled(True)
            self.slider.blockSignals(False)

        baseline_y = int(self.slider.value())

        # --- Perform Calculations ---
        l_pt, r_pt = fit_circle_and_intersect(cnt, baseline_y, gray, refine=True)
        
        # Fallback if circle fit fails (e.g., non-circular drop)
        if (l_pt is None) or (r_pt is None):
            pts = cnt[:, 0, :]
            xs = pts[:, 0]
            medx = np.median(xs)
            left_half = pts[xs <= medx]
            right_half = pts[xs >= medx]
            
            l_pt_fallback = None
            r_pt_fallback = None
            if len(left_half):
                l_pt_fallback = tuple(left_half[np.argmax(left_half[:, 1])])
            if len(right_half):
                r_pt_fallback = tuple(right_half[np.argmax(right_half[:, 1])])
            
            # Refine the fallback points
            l_pt = refine_contact_point(gray, l_pt_fallback, 5)
            r_pt = refine_contact_point(gray, r_pt_fallback, 5)

        left_ang = local_poly_contact_angle(cnt, l_pt, window_px=25)
        right_ang = local_poly_contact_angle(cnt, r_pt, window_px=25)
        apx = apex_of_contour(cnt)

        # --- Update Results Labels ---
        self.left_angle_label.setText(
            f"{left_ang:.2f}°" if left_ang is not None else "--°"
        )
        self.right_angle_label.setText(
            f"{right_ang:.2f}°" if right_ang is not None else "--°"
        )
        
        if left_ang is not None and right_ang is not None:
            avg_ang = (left_ang + right_ang) / 2.0
            self.avg_angle_label.setText(f"{avg_ang:.2f}°")
        else:
            self.avg_angle_label.setText("--°")
            
        if apx is not None:
            self.apex_label.setText(f"({apx[0]}, {apx[1]})")
        else:
            self.apex_label.setText("--")

        # --- Draw Visualization ---
        vis = img.copy()
        cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2) # Droplet Contour
        cv2.line(vis, (0, baseline_y), (vis.shape[1], baseline_y), (255, 0, 0), 2) # Baseline

        if apx is not None:
            cv2.circle(vis, apx, 7, (255, 255, 0), -1) # Apex

        # Draw contact points and angles
        for p, txt in [(l_pt, left_ang), (r_pt, right_ang)]:
            if p is not None:
                cv2.circle(vis, (int(p[0]), int(p[1])), 8, (0, 0, 255), -1)
                if txt is not None:
                    label = f"{txt:.1f}°"
                    px, py = int(p[0]), int(p[1])
                    cv2.putText(vis, label, (px + 10, max(py - 15, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                                
        self.display_image(vis)

    def preview_baseline_only(self):
        """Quickly draws just the baseline on the last-drawn image."""
        if self.original_image is None or self.current_pixmap is None:
            return
            
        # Create a pixmap from the *original* image to draw on
        h, w, _ = self.original_image.shape
        qimg = QImage(self.original_image.data, w, h, 3 * w, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        
        # We need to paint on the QPixmap, not the CV image, for speed
        from PyQt5.QtGui import QPainter, QPen
        painter = QPainter(pixmap)
        pen = QPen(Qt.red, 2, Qt.SolidLine)
        painter.setPen(pen)
        
        y = self.slider.value()
        painter.drawLine(0, y, w, y)
        painter.end()
        
        self.image_label.setPixmap(pixmap)
        self.current_pixmap = pixmap # Store this for the *next* preview
        

    def display_image(self, img_bgr):
        """Converts a BGR CV-matrix to a QPixmap and displays it."""
        h, w, ch = img_bgr.shape
        bytes_per_line = ch * w
        qimg = QImage(img_bgr.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.current_pixmap = QPixmap.fromImage(qimg)
        
        # Set the pixmap at 1:1 scale. The QScrollArea will handle scrolling.
        self.image_label.setPixmap(self.current_pixmap)
        self.image_label.setFixedSize(w, h) # Fix label size to image size
        

# ================================== Main ==================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SessileDropAnalyzer()
    window.show()
    sys.exit(app.exec_())
