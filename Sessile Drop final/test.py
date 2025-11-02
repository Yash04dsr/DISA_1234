# import sys
# import os
# import cv2
# import numpy as np
# from PyQt5.QtWidgets import (
#     QApplication, QWidget, QVBoxLayout, QPushButton,
#     QLabel, QSlider, QHBoxLayout, QSizePolicy, QScrollArea,
#     QFileDialog, QMessageBox
# )
# from PyQt5.QtGui import QPixmap, QImage
# from PyQt5.QtCore import Qt


# # ============================ Image / Geometry Utils ============================

# def adaptive_drop_mask(bgr):
#     gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (7, 7), 0)
#     mask = cv2.adaptiveThreshold(
#         blurred, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
#         51, 2
#     )
#     k = np.ones((3, 3), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
#     return mask


# def choose_droplet_contour(mask):
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     if not contours:
#         return None
#     best, best_score = None, 0.0
#     for c in contours:
#         area = cv2.contourArea(c)
#         if area < 500:
#             continue
#         per = cv2.arcLength(c, True)
#         circularity = 4.0 * np.pi * area / (per * per + 1e-6)
#         score = area * circularity
#         if score > best_score:
#             best, best_score = c, score
#     return best


# def find_auto_baseline_from_contour(cnt):
#     ys = cnt[:, 0, 1]
#     return int(np.percentile(ys, 95))


# def refine_contact_point(gray, point, search_radius=5):
#     if point is None:
#         return None
#     x0, y0 = int(point[0]), int(point[1])
#     h, w = gray.shape
#     best = (x0, y0)
#     best_g = 0
#     for dy in range(-search_radius, search_radius + 1):
#         for dx in range(-search_radius, search_radius + 1):
#             x, y = x0 + dx, y0 + dy
#             if 1 <= x < w - 1 and 1 <= y < h - 1:
#                 gx = int(gray[y, x + 1]) - int(gray[y, x - 1])
#                 gy = int(gray[y + 1, x]) - int(gray[y - 1, x])
#                 g2 = gx * gx + gy * gy
#                 if g2 > best_g:
#                     best_g = g2
#                     best = (x, y)
#     return best


# def fit_circle_and_intersect(cnt, baseline_y, gray=None, refine=True):
#     pts = cnt[:, 0, :].astype(np.float32)
#     if len(pts) < 10:
#         return None, None, None
#     y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
#     cutoff = y_min + 0.70 * (y_max - y_min)
#     upper = pts[pts[:, 1] < cutoff]
#     if len(upper) < 10:
#         upper = pts
#     (cx, cy), r = cv2.minEnclosingCircle(upper)
#     dy = baseline_y - cy
#     if abs(dy) >= r:
#         return None, None, None
#     dx = np.sqrt(max(r * r - dy * dy, 0.0))
#     left = (int(round(cx - dx)), int(baseline_y))
#     right = (int(round(cx + dx)), int(baseline_y))
#     if refine and gray is not None:
#         left = refine_contact_point(gray, left, 5)
#         right = refine_contact_point(gray, right, 5)
#     return left, right, ((cx, cy), r)


# def calculate_contact_angle(cnt, contact_point, baseline_y, side='left', window_size=30):
#     """
#     Calculate contact angle using polynomial fit with proper tangent calculation
#     """
#     if contact_point is None:
#         return None, None
    
#     cx, cy = contact_point
#     pts = cnt[:, 0, :]
    
#     # Get points along the contour near the contact point
#     if side == 'left':
#         # For left side, get points to the right of contact point (moving upward along contour)
#         nearby_pts = pts[(pts[:, 0] >= cx - window_size) & (pts[:, 0] <= cx + window_size) & 
#                         (pts[:, 1] < baseline_y)]
#     else:
#         # For right side, get points to the left of contact point (moving upward along contour)
#         nearby_pts = pts[(pts[:, 0] >= cx - window_size) & (pts[:, 0] <= cx + window_size) & 
#                         (pts[:, 1] < baseline_y)]
    
#     if len(nearby_pts) < 10:
#         # Fallback: get points within larger window
#         nearby_pts = pts[(pts[:, 0] >= cx - window_size*2) & (pts[:, 0] <= cx + window_size*2) & 
#                         (pts[:, 1] < baseline_y)]
    
#     if len(nearby_pts) < 5:
#         return None, None
    
#     # Sort points by y-coordinate (ascending - from baseline upward)
#     nearby_pts = nearby_pts[np.argsort(nearby_pts[:, 1])]
    
#     x = nearby_pts[:, 0].astype(np.float64)
#     y = nearby_pts[:, 1].astype(np.float64)
    
#     try:
#         # Fit 2nd order polynomial
#         coeffs = np.polyfit(x, y, 2)
#         a, b, c = coeffs
        
#         # Calculate derivative (slope) at contact point
#         slope = 2 * a * cx + b
        
#         # Calculate angle between tangent and horizontal
#         angle_rad = np.arctan(np.abs(slope))
#         contact_angle = np.degrees(angle_rad)
        
#         # For sessile drops, contact angle is > 90° if hydrophobic, < 90° if hydrophilic
#         # But the geometric calculation should give the acute/obtuse angle correctly
#         # based on the slope direction
        
#         # Adjust angle based on the side and slope direction
#         if side == 'left':
#             # For left side, if slope is positive, angle > 90°
#             if slope > 0:
#                 contact_angle = 180 - contact_angle
#         else:  # right side
#             # For right side, if slope is negative, angle > 90°
#             if slope < 0:
#                 contact_angle = 180 - contact_angle
        
#         # Ensure physically reasonable angles
#         contact_angle = np.clip(contact_angle, 10, 170)
        
#         # Calculate tangent line points for visualization
#         tangent_length = 50
#         if side == 'left':
#             dx = tangent_length
#             dy = tangent_length * slope
#             # For left side, tangent extends to the right
#             tangent_start = (int(cx), int(cy))
#             tangent_end = (int(cx + dx), int(cy + dy))
#         else:
#             dx = -tangent_length
#             dy = -tangent_length * slope  
#             # For right side, tangent extends to the left
#             tangent_start = (int(cx), int(cy))
#             tangent_end = (int(cx + dx), int(cy + dy))
        
#         return contact_angle, (tangent_start, tangent_end)
        
#     except Exception as e:
#         print(f"Error in contact angle calculation: {e}")
#         return None, None


# def draw_tangent_line(image, tangent_points, color=(0, 0, 255), thickness=2):
#     """Draw tangent line on image"""
#     if tangent_points is None:
#         return
#     start, end = tangent_points
#     cv2.line(image, start, end, color, thickness)


# def apex_of_contour(cnt):
#     if cnt is None or len(cnt) == 0:
#         return None
#     idx = np.argmin(cnt[:, 0, 1])
#     return tuple(cnt[idx, 0, :])


# def detect_auto_baseline(gray):
#     sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#     edges = np.uint8(np.absolute(sobel_y))
#     edges = cv2.GaussianBlur(edges, (9, 9), 0)
#     h = gray.shape[0]
#     bottom_half = edges[int(h * 0.5):, :]
#     row_strength = np.mean(bottom_half, axis=1)
#     y_rel = np.where(row_strength > np.percentile(row_strength, 95))[0]
#     if len(y_rel) == 0:
#         return int(h * 0.9)
#     baseline_y = int(h * 0.5 + np.median(y_rel))
#     return baseline_y


# # ============================ Main Analyzer GUI ============================

# class SessileDropAnalyzer(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Sessile Drop Contact Angle Analyzer")
#         self.setGeometry(100, 100, 1200, 850)

#         self.original_image = None
#         self.current_pixmap = None
#         self.overlay = None

#         layout = QVBoxLayout(self)
        
#         # Upload button
#         upload_layout = QHBoxLayout()
#         layout.addLayout(upload_layout)
        
#         self.btn_upload = QPushButton("Upload Image")
#         self.btn_upload.setStyleSheet("QPushButton { background-color: #2196F3; color: white; padding: 8px; font-weight: bold; }")
#         self.btn_upload.clicked.connect(self.upload_image)
#         upload_layout.addWidget(self.btn_upload)
        
#         upload_layout.addStretch()
        
#         # Image display
#         self.image_label = QLabel("Please upload an image of a sessile drop for analysis")
#         self.image_label.setAlignment(Qt.AlignCenter)
#         self.image_label.setStyleSheet("border: 2px solid #888; background-color: #f0f0f0; color: #666; font-size: 14px;")
#         self.image_label.setMinimumSize(800, 500)
#         self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         layout.addWidget(self.image_label, stretch=1)

#         # Controls
#         controls = QHBoxLayout()
#         layout.addLayout(controls)
        
#         self.btn_auto_baseline = QPushButton("Auto Baseline")
#         self.btn_auto_baseline.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 8px; }")
#         self.btn_auto_baseline.clicked.connect(self.auto_detect_baseline)
#         self.btn_auto_baseline.setEnabled(False)
#         controls.addWidget(self.btn_auto_baseline)
        
#         controls.addWidget(QLabel("Baseline Y:"))
#         self.slider = QSlider(Qt.Horizontal)
#         self.slider.setEnabled(False)
#         self.slider.setStyleSheet("QSlider::handle:horizontal { background-color: #2196F3; }")
#         controls.addWidget(self.slider)
        
#         controls.addStretch()

#         # Results display
#         results_layout = QVBoxLayout()
#         layout.addLayout(results_layout)
        
#         # Main results
#         results_main = QHBoxLayout()
#         results_layout.addLayout(results_main)
        
#         self.left_angle_label = QLabel("Left Contact Angle: --°")
#         self.left_angle_label.setStyleSheet("QLabel { font-weight: bold; color: #D32F2F; font-size: 14px; padding: 5px; }")
#         self.right_angle_label = QLabel("Right Contact Angle: --°")
#         self.right_angle_label.setStyleSheet("QLabel { font-weight: bold; color: #1976D2; font-size: 14px; padding: 5px; }")
#         self.average_label = QLabel("Average: --°")
#         self.average_label.setStyleSheet("QLabel { font-weight: bold; color: #388E3C; font-size: 14px; padding: 5px; }")
        
#         results_main.addWidget(self.left_angle_label)
#         results_main.addWidget(self.right_angle_label)
#         results_main.addWidget(self.average_label)
        
#         # Additional info
#         info_layout = QHBoxLayout()
#         results_layout.addLayout(info_layout)
        
#         self.baseline_label = QLabel("Baseline Position: --")
#         self.contour_label = QLabel("Contour Area: --")
#         self.apex_label = QLabel("Apex: --")
#         self.image_info_label = QLabel("Image: None")
        
#         for label in [self.baseline_label, self.contour_label, self.apex_label, self.image_info_label]:
#             label.setStyleSheet("QLabel { color: #666; font-size: 12px; padding: 2px; }")
#             info_layout.addWidget(label)

#         # Signals
#         self.slider.valueChanged.connect(self.preview_baseline_only)
#         self.slider.sliderReleased.connect(self.run_full_analysis)

#     def upload_image(self):
#         """Open file dialog to upload an image"""
#         file_path, _ = QFileDialog.getOpenFileName(
#             self,
#             "Select Sessile Drop Image",
#             "",
#             "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)"
#         )
        
#         if file_path:
#             self.load_image(file_path)

#     def load_image(self, file_path):
#         """Load and process the selected image"""
#         try:
#             img = cv2.imread(file_path)
#             if img is None:
#                 QMessageBox.critical(self, "Error", "Could not read the image file.")
#                 return
                
#             # Store filename for display
#             file_name = os.path.basename(file_path)
#             self.image_info_label.setText(f"Image: {file_name}")
            
#             # Resize if too large
#             h, w = img.shape[:2]
#             if w > 1400:
#                 img = cv2.resize(img, (1400, int(h * 1400 / w)), interpolation=cv2.INTER_AREA)
                
#             self.original_image = img
#             self.btn_auto_baseline.setEnabled(True)
#             self.run_full_analysis(initial=True)
            
#         except Exception as e:
#             QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

#     def preview_baseline_only(self):
#         if self.overlay is None or self.original_image is None:
#             return
#         y = self.slider.value()
#         temp = self.original_image.copy()
        
#         # Draw baseline on original image
#         cv2.line(temp, (0, y), (temp.shape[1], y), (255, 0, 0), 2)
#         self.baseline_label.setText(f"Baseline Position: {y} px")
#         self.display_image(temp)

#     def auto_detect_baseline(self):
#         if self.original_image is None:
#             return
#         gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
#         baseline_y = detect_auto_baseline(gray)
#         self.slider.setValue(baseline_y)
#         self.run_full_analysis()

#     def run_full_analysis(self, initial=False):
#         if self.original_image is None:
#             return
            
#         img = self.original_image.copy()
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         mask = adaptive_drop_mask(img)
#         cnt = choose_droplet_contour(mask)
        
#         if cnt is None:
#             self.left_angle_label.setText("Left Contact Angle: --°")
#             self.right_angle_label.setText("Right Contact Angle: --°")
#             self.average_label.setText("Average: --°")
#             self.contour_label.setText("Contour Area: --")
#             self.apex_label.setText("Apex: --")
#             self.display_image(img)
#             QMessageBox.warning(self, "Analysis Warning", "No droplet contour found in the image. Please ensure the drop is clearly visible.")
#             return
            
#         # Calculate contour area
#         area = cv2.contourArea(cnt)
#         self.contour_label.setText(f"Contour Area: {area:.0f} px²")
            
#         auto_y = find_auto_baseline_from_contour(cnt)
#         if initial or not self.slider.isEnabled():
#             h = img.shape[0]
#             self.slider.blockSignals(True)
#             self.slider.setRange(0, h - 1)
#             self.slider.setValue(int(np.clip(auto_y, 0, h - 1)))
#             self.slider.setEnabled(True)
#             self.slider.blockSignals(False)
            
#         baseline_y = int(self.slider.value())
#         self.baseline_label.setText(f"Baseline Position: {baseline_y} px")
        
#         l_pt, r_pt, circle_data = fit_circle_and_intersect(cnt, baseline_y, gray, refine=True)
        
#         # Fallback if circle fit fails
#         if (l_pt is None) or (r_pt is None):
#             pts = cnt[:, 0, :]
#             xs = pts[:, 0]
#             medx = np.median(xs)
#             left_half = pts[xs <= medx]
#             right_half = pts[xs >= medx]
#             if len(left_half):
#                 l_pt = tuple(left_half[np.argmax(left_half[:, 1])])
#             if len(right_half):
#                 r_pt = tuple(right_half[np.argmax(right_half[:, 1])])
#             l_pt = refine_contact_point(gray, l_pt, 5) if l_pt is not None else None
#             r_pt = refine_contact_point(gray, r_pt, 5) if r_pt is not None else None
        
#         # Calculate contact angles with proper tangent lines
#         left_ang, left_tangent = calculate_contact_angle(cnt, l_pt, baseline_y, 'left', 30)
#         right_ang, right_tangent = calculate_contact_angle(cnt, r_pt, baseline_y, 'right', 30)
        
#         # Calculate average
#         if left_ang is not None and right_ang is not None:
#             avg_ang = (left_ang + right_ang) / 2
#             self.average_label.setText(f"Average: {avg_ang:.1f}°")
#         else:
#             self.average_label.setText("Average: --°")
        
#         # Update angle labels
#         self.left_angle_label.setText(
#             f"Left Contact Angle: {left_ang:.1f}°" if left_ang is not None else "Left Contact Angle: --°"
#         )
#         self.right_angle_label.setText(
#             f"Right Contact Angle: {right_ang:.1f}°" if right_ang is not None else "Right Contact Angle: --°"
#         )
        
#         vis = img.copy()
        
#         # Draw contour with subtle color
#         cv2.drawContours(vis, [cnt], -1, (0, 120, 0), 2)  # Dark green
        
#         # Draw baseline
#         cv2.line(vis, (0, baseline_y), (vis.shape[1], baseline_y), (255, 0, 0), 2)
        
#         # Draw fitted circle if available
#         if circle_data is not None:
#             (cx, cy), r = circle_data
#             cv2.circle(vis, (int(cx), int(cy)), int(r), (200, 200, 0), 2)  # Light blue-green
        
#         # Draw apex
#         apx = apex_of_contour(cnt)
#         if apx is not None:
#             cv2.circle(vis, apx, 6, (0, 255, 255), -1)  # Yellow
#             self.apex_label.setText(f"Apex: ({apx[0]}, {apx[1]})")
        
#         # Draw contact points, tangent lines, and angles
#         for i, (p, ang, tangent) in enumerate([(l_pt, left_ang, left_tangent), (r_pt, right_ang, right_tangent)]):
#             if p is not None:
#                 # Draw contact point
#                 cv2.circle(vis, (int(p[0]), int(p[1])), 7, (0, 0, 255), -1)
                
#                 # Draw tangent line
#                 if tangent is not None:
#                     draw_tangent_line(vis, tangent, (0, 0, 255), 2)
                
#                 if ang is not None:
#                     # Choose color based on side
#                     color = (50, 50, 220) if i == 0 else (220, 50, 50)
                    
#                     # Create angle label
#                     label = f"{ang:.1f}°"
#                     px, py = int(p[0]), int(p[1])
                    
#                     # Position text based on side
#                     if i == 0:  # Left side
#                         text_pos = (px + 12, max(py - 12, 25))
#                     else:  # Right side
#                         text_pos = (px - 70, max(py - 12, 25))
                    
#                     # Draw text with background for better visibility
#                     (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
#                     cv2.rectangle(vis, 
#                                 (text_pos[0] - 5, text_pos[1] - text_height - 5),
#                                 (text_pos[0] + text_width + 5, text_pos[1] + baseline - 5),
#                                 (255, 255, 255), -1)
#                     cv2.putText(vis, label, text_pos,
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
#         self.overlay = vis
#         self.display_image(vis)

#     def display_image(self, img_bgr):
#         h, w = img_bgr.shape[:2]
#         qimg = QImage(img_bgr.data, w, h, 3 * w, QImage.Format_BGR888)
#         self.current_pixmap = QPixmap.fromImage(qimg)
#         scaled = self.current_pixmap.scaled(
#             self.image_label.width(), self.image_label.height(),
#             Qt.KeepAspectRatio, Qt.SmoothTransformation
#         )
#         self.image_label.setPixmap(scaled)

#     def resizeEvent(self, event):
#         super().resizeEvent(event)
#         if self.current_pixmap is not None:
#             scaled = self.current_pixmap.scaled(
#                 self.image_label.width(), self.image_label.height(),
#                 Qt.KeepAspectRatio, Qt.SmoothTransformation
#             )
#             self.image_label.setPixmap(scaled)


# # ================================== Scroll Wrapper ==================================

# class ScrollableAnalyzer(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Sessile Drop Contact Angle Analyzer")
#         self.setGeometry(100, 100, 800, 600)
#         layout = QVBoxLayout(self)
#         scroll = QScrollArea()
#         scroll.setWidgetResizable(True)
#         layout.addWidget(scroll)
#         self.analyzer = SessileDropAnalyzer()
#         scroll.setWidget(self.analyzer)


# # ================================== Main ==================================

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = ScrollableAnalyzer()
#     window.show()
#     sys.exit(app.exec_())


# import sys
# import cv2
# import numpy as np
# from PyQt5.QtWidgets import (
#     QApplication, QWidget, QVBoxLayout, QPushButton,
#     QLabel, QFileDialog, QSlider, QHBoxLayout, QSizePolicy, QScrollArea
# )
# from PyQt5.QtGui import QPixmap, QImage
# from PyQt5.QtCore import Qt


# # ============================ Image / Geometry Utils ============================

# def adaptive_drop_mask(bgr):
#     gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (7, 7), 0)
#     mask = cv2.adaptiveThreshold(
#         blurred, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
#         51, 2
#     )
#     k = np.ones((3, 3), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
#     return mask


# def choose_droplet_contour(mask):
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     if not contours:
#         return None
#     best, best_score = None, 0.0
#     for c in contours:
#         area = cv2.contourArea(c)
#         if area < 500:
#             continue
#         per = cv2.arcLength(c, True)
#         circularity = 4.0 * np.pi * area / (per * per + 1e-6)
#         score = area * circularity
#         if score > best_score:
#             best, best_score = c, score
#     return best


# def find_auto_baseline_from_contour(cnt):
#     ys = cnt[:, 0, 1]
#     return int(np.percentile(ys, 95))


# def refine_contact_point(gray, point, search_radius=5):
#     if point is None:
#         return None
#     x0, y0 = int(point[0]), int(point[1])
#     h, w = gray.shape
#     best = (x0, y0)
#     best_g = 0
#     for dy in range(-search_radius, search_radius + 1):
#         for dx in range(-search_radius, search_radius + 1):
#             x, y = x0 + dx, y0 + dy
#             if 1 <= x < w - 1 and 1 <= y < h - 1:
#                 gx = int(gray[y, x + 1]) - int(gray[y, x - 1])
#                 gy = int(gray[y + 1, x]) - int(gray[y - 1, x])
#                 g2 = gx * gx + gy * gy
#                 if g2 > best_g:
#                     best_g = g2
#                     best = (x, y)
#     return best


# def fit_circle_and_intersect(cnt, baseline_y, gray=None, refine=True):
#     pts = cnt[:, 0, :].astype(np.float32)
#     if len(pts) < 10:
#         return None, None
#     y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
#     cutoff = y_min + 0.70 * (y_max - y_min)
#     upper = pts[pts[:, 1] < cutoff]
#     if len(upper) < 10:
#         upper = pts
#     (cx, cy), r = cv2.minEnclosingCircle(upper)
#     dy = baseline_y - cy
#     if abs(dy) >= r:
#         return None, None
#     dx = np.sqrt(max(r * r - dy * dy, 0.0))
#     left = (int(round(cx - dx)), int(baseline_y))
#     right = (int(round(cx + dx)), int(baseline_y))
#     if refine and gray is not None:
#         left = refine_contact_point(gray, left, 5)
#         right = refine_contact_point(gray, right, 5)
#     return left, right


# def local_poly_contact_angle(cnt, contact_pt, window_px=25):
#     if contact_pt is None:
#         return None
#     cx, _ = contact_pt
#     pts = cnt[:, 0, :]
#     nb = pts[np.abs(pts[:, 0] - cx) < window_px]
#     if len(nb) < 8:
#         nb = pts[np.abs(pts[:, 0] - cx) < window_px * 1.8]
#     if len(nb) < 8:
#         return None
#     x = nb[:, 0].astype(np.float64)
#     y = nb[:, 1].astype(np.float64)
#     try:
#         a, b, c = np.polyfit(x, y, 2)
#     except Exception:
#         return None
#     m = 2.0 * a * cx + b
#     theta = np.degrees(np.arctan(np.abs(m)))
#     return float(np.clip(theta, 0.0, 175.0))


# def apex_of_contour(cnt):
#     if cnt is None or len(cnt) == 0:
#         return None
#     idx = np.argmin(cnt[:, 0, 1])
#     return tuple(cnt[idx, 0, :])


# def detect_auto_baseline(gray):
#     sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#     edges = np.uint8(np.absolute(sobel_y))
#     edges = cv2.GaussianBlur(edges, (9, 9), 0)
#     h = gray.shape[0]
#     bottom_half = edges[int(h * 0.5):, :]
#     row_strength = np.mean(bottom_half, axis=1)
#     y_rel = np.where(row_strength > np.percentile(row_strength, 95))[0]
#     if len(y_rel) == 0:
#         return int(h * 0.9)
#     baseline_y = int(h * 0.5 + np.median(y_rel))
#     return baseline_y


# # ============================ Main Analyzer GUI ============================

# class SessileDropAnalyzer(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Sessile Drop Contact Angle Analyzer (Circle-Fit + Auto Baseline Add-on)")
#         self.setGeometry(100, 100, 1150, 830)

#         self.original_image = None
#         self.current_pixmap = None
#         self.overlay = None

#         layout = QVBoxLayout(self)
#         self.image_label = QLabel("Load a side-view droplet image.")
#         self.image_label.setAlignment(Qt.AlignCenter)
#         self.image_label.setStyleSheet("border: 2px solid #888;")
#         self.image_label.setMinimumSize(720, 460)
#         self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         layout.addWidget(self.image_label, stretch=1)

#         controls = QHBoxLayout()
#         layout.addLayout(controls)
#         self.btn_load = QPushButton("Load Image")
#         self.btn_auto_baseline = QPushButton("Auto Baseline")
#         controls.addWidget(self.btn_load)
#         controls.addWidget(self.btn_auto_baseline)
#         controls.addWidget(QLabel("Baseline (y):"))
#         self.slider = QSlider(Qt.Horizontal)
#         self.slider.setEnabled(False)
#         controls.addWidget(self.slider)

#         results = QHBoxLayout()
#         layout.addLayout(results)
#         self.left_angle_label = QLabel("Left Angle: --°")
#         self.right_angle_label = QLabel("Right Angle: --°")
#         results.addWidget(self.left_angle_label)
#         results.addWidget(self.right_angle_label)

#         # Signals
#         self.btn_load.clicked.connect(self.load_image)
#         self.btn_auto_baseline.clicked.connect(self.auto_detect_baseline)
#         self.slider.valueChanged.connect(self.preview_baseline_only)
#         self.slider.sliderReleased.connect(self.run_full_analysis)

#     # ------------------------------- UI Actions -------------------------------

#     def load_image(self):
#         path, _ = QFileDialog.getOpenFileName(
#             self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
#         )
#         if not path:
#             return
#         img = cv2.imread(path)
#         if img is None:
#             self.image_label.setText("Error: Could not read image.")
#             return
#         h, w = img.shape[:2]
#         if w > 1400:
#             img = cv2.resize(img, (1400, int(h * 1400 / w)), interpolation=cv2.INTER_AREA)
#         self.original_image = img
#         self.run_full_analysis(initial=True)

#     def preview_baseline_only(self):
#         if self.overlay is None:
#             return
#         y = self.slider.value()
#         temp = self.overlay.copy()
#         cv2.line(temp, (0, y), (temp.shape[1], y), (255, 0, 0), 2)
#         self.display_image(temp)

#     def auto_detect_baseline(self):
#         if self.original_image is None:
#             return
#         gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
#         baseline_y = detect_auto_baseline(gray)
#         self.slider.setValue(baseline_y)
#         self.run_full_analysis()

#     # ----------------------------- Core Processing -----------------------------

#     def run_full_analysis(self, initial=False):
#         if self.original_image is None:
#             return
#         img = self.original_image.copy()
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         mask = adaptive_drop_mask(img)
#         cnt = choose_droplet_contour(mask)
#         if cnt is None:
#             self.left_angle_label.setText("Left Angle: --°")
#             self.right_angle_label.setText("Right Angle: --°")
#             self.display_image(img)
#             return
#         auto_y = find_auto_baseline_from_contour(cnt)
#         if initial or not self.slider.isEnabled():
#             h = img.shape[0]
#             self.slider.blockSignals(True)
#             self.slider.setRange(0, h - 1)
#             self.slider.setValue(int(np.clip(auto_y, 0, h - 1)))
#             self.slider.setEnabled(True)
#             self.slider.blockSignals(False)
#         baseline_y = int(self.slider.value())
#         l_pt, r_pt = fit_circle_and_intersect(cnt, baseline_y, gray, refine=True)
#         if (l_pt is None) or (r_pt is None):
#             pts = cnt[:, 0, :]
#             xs = pts[:, 0]
#             medx = np.median(xs)
#             left_half = pts[xs <= medx]
#             right_half = pts[xs >= medx]
#             if len(left_half):
#                 l_pt = tuple(left_half[np.argmax(left_half[:, 1])])
#             if len(right_half):
#                 r_pt = tuple(right_half[np.argmax(right_half[:, 1])])
#             l_pt = refine_contact_point(gray, l_pt, 5) if l_pt is not None else None
#             r_pt = refine_contact_point(gray, r_pt, 5) if r_pt is not None else None
#         left_ang = local_poly_contact_angle(cnt, l_pt, window_px=25)
#         right_ang = local_poly_contact_angle(cnt, r_pt, window_px=25)
#         self.left_angle_label.setText(
#             f"Left Angle: {left_ang:.2f}°" if left_ang is not None else "Left Angle: --°"
#         )
#         self.right_angle_label.setText(
#             f"Right Angle: {right_ang:.2f}°" if right_ang is not None else "Right Angle: --°"
#         )
#         vis = img.copy()
#         cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
#         cv2.line(vis, (0, baseline_y), (vis.shape[1], baseline_y), (255, 0, 0), 2)
#         apx = apex_of_contour(cnt)
#         if apx is not None:
#             cv2.circle(vis, apx, 7, (255, 255, 0), -1)
#         for p, txt in [(l_pt, left_ang), (r_pt, right_ang)]:
#             if p is not None:
#                 cv2.circle(vis, (int(p[0]), int(p[1])), 8, (0, 0, 255), -1)
#                 if txt is not None:
#                     label = f"{txt:.1f}°"
#                     px, py = int(p[0]), int(p[1])
#                     cv2.putText(vis, label, (px + 8, max(py - 10, 10)),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
#         self.overlay = vis
#         self.display_image(vis)

#     def display_image(self, img_bgr):
#         h, w = img_bgr.shape[:2]
#         qimg = QImage(img_bgr.data, w, h, 3 * w, QImage.Format_BGR888)
#         self.current_pixmap = QPixmap.fromImage(qimg)
#         scaled = self.current_pixmap.scaled(
#             self.image_label.width(), self.image_label.height(),
#             Qt.KeepAspectRatio, Qt.SmoothTransformation
#         )
#         self.image_label.setPixmap(scaled)

#     def resizeEvent(self, event):
#         super().resizeEvent(event)
#         if self.current_pixmap is not None:
#             scaled = self.current_pixmap.scaled(
#                 self.image_label.width(), self.image_label.height(),
#                 Qt.KeepAspectRatio, Qt.SmoothTransformation
#             )
#             self.image_label.setPixmap(scaled)


# # ================================== Scroll Wrapper ==================================

# class ScrollableAnalyzer(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Sessile Drop Analyzer (Scrollable Version)")
#         self.setGeometry(100, 100, 800, 600)

#         layout = QVBoxLayout(self)
#         scroll = QScrollArea()
#         scroll.setWidgetResizable(True)
#         layout.addWidget(scroll)

#         self.analyzer = SessileDropAnalyzer()
#         scroll.setWidget(self.analyzer)


# # ================================== Main ==================================

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = ScrollableAnalyzer()
#     window.show()
#     sys.exit(app.exec_())

import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QSlider, QHBoxLayout, QSizePolicy, QScrollArea
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


# ============================ Image / Geometry Utils ============================

def adaptive_drop_mask(bgr):
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
    ys = cnt[:, 0, 1]
    return int(np.percentile(ys, 95))


def refine_contact_point(gray, point, search_radius=5):
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
    pts = cnt[:, 0, :].astype(np.float32)
    if len(pts) < 10:
        return None, None
    y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
    cutoff = y_min + 0.70 * (y_max - y_min)
    upper = pts[pts[:, 1] < cutoff]
    if len(upper) < 10:
        upper = pts
    (cx, cy), r = cv2.minEnclosingCircle(upper)
    dy = baseline_y - cy
    if abs(dy) >= r:
        return None, None
    dx = np.sqrt(max(r * r - dy * dy, 0.0))
    left = (int(round(cx - dx)), int(baseline_y))
    right = (int(round(cx + dx)), int(baseline_y))
    if refine and gray is not None:
        left = refine_contact_point(gray, left, 5)
        right = refine_contact_point(gray, right, 5)
    return left, right


def local_poly_contact_angle(cnt, contact_pt, window_px=25):
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
    if cnt is None or len(cnt) == 0:
        return None
    idx = np.argmin(cnt[:, 0, 1])
    return tuple(cnt[idx, 0, :])


def detect_auto_baseline(gray):
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.uint8(np.absolute(sobel_y))
    edges = cv2.GaussianBlur(edges, (9, 9), 0)
    h = gray.shape[0]
    bottom_half = edges[int(h * 0.5):, :]
    row_strength = np.mean(bottom_half, axis=1)
    y_rel = np.where(row_strength > np.percentile(row_strength, 95))[0]
    if len(y_rel) == 0:
        return int(h * 0.9)
    baseline_y = int(h * 0.5 + np.median(y_rel))
    return baseline_y


# ============================ Main Analyzer GUI ============================

class SessileDropAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sessile Drop Contact Angle Analyzer (Auto Load: ~/sessile.png)")
        self.setGeometry(100, 100, 1150, 830)

        self.original_image = None
        self.current_pixmap = None
        self.overlay = None

        layout = QVBoxLayout(self)
        self.image_label = QLabel("Loading sessile.png from home directory...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #888;")
        self.image_label.setMinimumSize(720, 460)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, stretch=1)

        controls = QHBoxLayout()
        layout.addLayout(controls)
        self.btn_auto_baseline = QPushButton("Auto Baseline")
        controls.addWidget(self.btn_auto_baseline)
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

        # Signals
        self.btn_auto_baseline.clicked.connect(self.auto_detect_baseline)
        self.slider.valueChanged.connect(self.preview_baseline_only)
        self.slider.sliderReleased.connect(self.run_full_analysis)

        # Automatically load sessile.png
        self.load_default_image()

    def load_default_image(self):
        path = os.path.expanduser("~/sessile.png")
        if not os.path.exists(path):
            self.image_label.setText("Error: ~/sessile.png not found.")
            return
        img = cv2.imread(path)
        if img is None:
            self.image_label.setText("Error: Could not read sessile.png.")
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

    def auto_detect_baseline(self):
        if self.original_image is None:
            return
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        baseline_y = detect_auto_baseline(gray)
        self.slider.setValue(baseline_y)
        self.run_full_analysis()

    def run_full_analysis(self, initial=False):
        if self.original_image is None:
            return
        img = self.original_image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = adaptive_drop_mask(img)
        cnt = choose_droplet_contour(mask)
        if cnt is None:
            self.left_angle_label.setText("Left Angle: --°")
            self.right_angle_label.setText("Right Angle: --°")
            self.display_image(img)
            return
        auto_y = find_auto_baseline_from_contour(cnt)
        if initial or not self.slider.isEnabled():
            h = img.shape[0]
            self.slider.blockSignals(True)
            self.slider.setRange(0, h - 1)
            self.slider.setValue(int(np.clip(auto_y, 0, h - 1)))
            self.slider.setEnabled(True)
            self.slider.blockSignals(False)
        baseline_y = int(self.slider.value())
        l_pt, r_pt = fit_circle_and_intersect(cnt, baseline_y, gray, refine=True)
        if (l_pt is None) or (r_pt is None):
            pts = cnt[:, 0, :]
            xs = pts[:, 0]
            medx = np.median(xs)
            left_half = pts[xs <= medx]
            right_half = pts[xs >= medx]
            if len(left_half):
                l_pt = tuple(left_half[np.argmax(left_half[:, 1])])
            if len(right_half):
                r_pt = tuple(right_half[np.argmax(right_half[:, 1])])
            l_pt = refine_contact_point(gray, l_pt, 5) if l_pt is not None else None
            r_pt = refine_contact_point(gray, r_pt, 5) if r_pt is not None else None
        left_ang = local_poly_contact_angle(cnt, l_pt, window_px=25)
        right_ang = local_poly_contact_angle(cnt, r_pt, window_px=25)
        self.left_angle_label.setText(
            f"Left Angle: {left_ang:.2f}°" if left_ang is not None else "Left Angle: --°"
        )
        self.right_angle_label.setText(
            f"Right Angle: {right_ang:.2f}°" if right_ang is not None else "Right Angle: --°"
        )
        vis = img.copy()
        cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
        cv2.line(vis, (0, baseline_y), (vis.shape[1], baseline_y), (255, 0, 0), 2)
        apx = apex_of_contour(cnt)
        if apx is not None:
            cv2.circle(vis, apx, 7, (255, 255, 0), -1)
        for p, txt in [(l_pt, left_ang), (r_pt, right_ang)]:
            if p is not None:
                cv2.circle(vis, (int(p[0]), int(p[1])), 8, (0, 0, 255), -1)
                if txt is not None:
                    label = f"{txt:.1f}°"
                    px, py = int(p[0]), int(p[1])
                    cv2.putText(vis, label, (px + 8, max(py - 10, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        self.overlay = vis
        self.display_image(vis)

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


# ================================== Scroll Wrapper ==================================

class ScrollableAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sessile Drop Analyzer (Auto-load ~/sessile.png)")
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.analyzer = SessileDropAnalyzer()
        scroll.setWidget(self.analyzer)


# ================================== Main ==================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScrollableAnalyzer()
    window.show()
    sys.exit(app.exec_())
