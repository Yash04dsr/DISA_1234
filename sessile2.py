import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QSlider, QHBoxLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class SessileDropAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Sessile Drop Contact Angle Analyzer (Responsive UI)')
        self.setGeometry(100, 100, 1000, 800)
        
        # --- State Management Attributes ---
        self.original_image = None
        self.last_analyzed_image = None # Stores the image with contours/points, but no baseline
        self.current_pixmap = None

        # --- GUI Setup ---
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.image_label = QLabel('Please load an image of a droplet.')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid gray;")
        self.layout.addWidget(self.image_label, 1)

        controls_layout = QHBoxLayout()
        self.btn_load = QPushButton('Load Image')
        controls_layout.addWidget(self.btn_load)

        self.slider_label = QLabel('Baseline Position:')
        controls_layout.addWidget(self.slider_label)
        self.slider = QSlider(Qt.Horizontal)
        controls_layout.addWidget(self.slider)
        
        self.layout.addLayout(controls_layout)

        results_layout = QHBoxLayout()
        self.left_angle_label = QLabel('Left Angle: --°')
        self.right_angle_label = QLabel('Right Angle: --°')
        results_layout.addWidget(self.left_angle_label)
        results_layout.addWidget(self.right_angle_label)
        self.layout.addLayout(results_layout)
        
        # --- Signal Connections ---
        self.btn_load.clicked.connect(self.load_image)
        self.slider.sliderReleased.connect(self.update_analysis)
        self.slider.valueChanged.connect(self.preview_baseline)

    def load_image(self):
        """Loads and preprocesses an image from the user."""
        filepath, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.png *.jpg *.bmp)')
        if not filepath: return
            
        self.original_image = cv2.imread(filepath)
        if self.original_image is None:
            self.image_label.setText("Error: Could not load image."); return

        # Resize large images for consistent performance
        MAX_WIDTH = 1200
        h, w, _ = self.original_image.shape
        if w > MAX_WIDTH:
            self.original_image = cv2.resize(self.original_image, (MAX_WIDTH, int(h * MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)

        h, _, _ = self.original_image.shape
        self.slider.blockSignals(True)
        self.slider.setRange(0, h - 1)
        self.slider.setValue(int(h * 0.85))
        self.slider.blockSignals(False)

        # Perform the first full analysis
        self.update_analysis()

    def preview_baseline(self):
        """## FIX: Lightweight preview. Uses the last analyzed image and only draws the baseline."""
        if self.last_analyzed_image is None: return
        
        preview_img = self.last_analyzed_image.copy()
        baseline_y = self.slider.value()
        cv2.line(preview_img, (0, baseline_y), (preview_img.shape[1], baseline_y), (255, 0, 0), 2)
        
        self.display_image(preview_img)
        
    def update_analysis(self):
        """## FIX: Heavy analysis. Recalculates everything and updates the base analyzed image."""
        if self.original_image is None: return

        # Base image for analysis drawings
        analysis_img = self.original_image.copy()
        baseline_y = self.slider.value()

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self.left_angle_label.setText('Left Angle: --°')
            self.right_angle_label.setText('Right Angle: --°')
            self.last_analyzed_image = self.original_image.copy() # Reset analyzed image
            self.preview_baseline() # Show the baseline on the clean image
            return

        droplet_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.001 * cv2.arcLength(droplet_contour, True)
        approx_contour = cv2.approxPolyDP(droplet_contour, epsilon, True)
        
        apex_point = tuple(approx_contour[approx_contour[:, :, 1].argmin()][0])
        intersection_points = [tuple(p[0]) for p in approx_contour if abs(p[0][1] - baseline_y) < 5]

        left_contact_point, right_contact_point = None, None
        if len(intersection_points) >= 2:
            intersection_points.sort(); left_contact_point = intersection_points[0]; right_contact_point = intersection_points[-1]

        # Calculate angles and update labels
        l_angle = self.calculate_contact_angle(approx_contour, left_contact_point, 'left') if left_contact_point else None
        r_angle = self.calculate_contact_angle(approx_contour, right_contact_point, 'right') if right_contact_point else None
        self.left_angle_label.setText(f'Left Angle: {l_angle:.2f}°' if l_angle is not None else 'Left Angle: --°')
        self.right_angle_label.setText(f'Right Angle: {r_angle:.2f}°' if r_angle is not None else 'Right Angle: --°')

        # Draw analysis results (contour, points) onto the analysis_img
        cv2.drawContours(analysis_img, [approx_contour], -1, (0, 255, 0), 2)
        cv2.circle(analysis_img, apex_point, 8, (255, 255, 0), -1)
        if left_contact_point: cv2.circle(analysis_img, left_contact_point, 8, (0, 0, 255), -1)
        if right_contact_point: cv2.circle(analysis_img, right_contact_point, 8, (0, 0, 255), -1)
        
        # Save this fully drawn image (without baseline) for the preview function to use
        self.last_analyzed_image = analysis_img
        
        # Now call the preview function to draw the final baseline and display
        self.preview_baseline()

    def calculate_contact_angle(self, contour, contact_point, side):
        x, y, w, h = cv2.boundingRect(contour)
        top_contour = [p for p in contour if p[0][1] < y + h * 0.75]
        if len(top_contour) < 5: return None
        (cx, cy), radius = cv2.minEnclosingCircle(np.array(top_contour))
        p1 = np.array(contact_point)
        normal_vector = p1 - np.array([cx, cy])
        if np.isclose(normal_vector[0], 0): return 90.0
        angle_rad = np.arctan2(normal_vector[1], normal_vector[0])
        angle_deg = np.degrees(angle_rad)
        return (180 - angle_deg) if side == 'left' else angle_deg

    def display_image(self, img_cv):
        h, w, ch = img_cv.shape
        q_img = QImage(img_cv.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
        self.current_pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(self.current_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        if self.current_pixmap:
            self.image_label.setPixmap(self.current_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    analyzer = SessileDropAnalyzer()
    analyzer.show()
    sys.exit(app.exec_())