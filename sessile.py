import sys, cv2, numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QSlider, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# ============================ Safe Baseline Detection ============================
def detect_baseline_hough(gray):
    h, w = gray.shape
    roi_y0 = int(h * 0.55)
    roi = gray[roi_y0:, :].copy()  # ✅ ensure contiguous memory
    if roi.size == 0:
        return int(h * 0.9)
    try:
        edges = cv2.Canny(cv2.GaussianBlur(roi, (5,5), 0), 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, int(w*0.25), 10)
        if lines is None or len(lines)==0:
            return int(h*0.9)
        ys=[]
        for x1,y1,x2,y2 in lines[:,0,:]:
            if abs(y2-y1)<15: ys.append((y1+y2)/2)
        return int(roi_y0 + np.median(ys)) if ys else int(h*0.9)
    except cv2.error as e:
        print("⚠️ Hough failed:", e)
        return int(h*0.9)

# ============================ Safe Droplet Segmentation ============================
def detect_droplet_contour_smart(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    baseline_y = detect_baseline_hough(gray)
    h, w = gray.shape
    top = max(0, baseline_y - int(0.65 * h))
    roi = gray[top:baseline_y, :].copy()
    if roi.size == 0:
        print("⚠️ Empty ROI"); return None, baseline_y

    try:
        blur = cv2.GaussianBlur(roi, (5,5), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low, high = max(5, int(0.5*otsu)), max(10, int(1.5*otsu))
        edges = cv2.Canny(blur, low, high)
        k = np.ones((3,3),np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
        cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts: return None, baseline_y

        def score(c):
            a = cv2.contourArea(c)
            if a < 200: return -1
            p = cv2.arcLength(c, True)
            circ = 4*np.pi*a/(p*p+1e-6)
            ys = c[:,0,1] + top
            frac = np.mean(ys < baseline_y)
            return a * (0.6*circ + 0.3*frac)
        best = max(cnts, key=score)
        best[:,0,1] += top
        return best, baseline_y
    except cv2.error as e:
        print("⚠️ Contour detect failed:", e)
        return None, baseline_y

# ============================ Other Geometry Utils ============================
def refine_contact_point(gray, pt, search_radius=5):
    if pt is None: return None
    x0,y0=int(pt[0]),int(pt[1])
    h,w=gray.shape;best=(x0,y0);bg=0
    for dy in range(-search_radius,search_radius+1):
        for dx in range(-search_radius,search_radius+1):
            x,y=x0+dx,y0+dy
            if 1<=x<w-1 and 1<=y<h-1:
                gx=int(gray[y,x+1])-int(gray[y,x-1])
                gy=int(gray[y+1,x])-int(gray[y-1,x])
                g2=gx*gx+gy*gy
                if g2>bg:bg,g2;best=(x,y)
    return best

def fit_circle_and_intersect(cnt, baseline_y, gray=None):
    try:
        pts=cnt[:,0,:].astype(np.float32)
        if len(pts)<10: return None,None
        (cx,cy),r=cv2.minEnclosingCircle(pts)
        dy=baseline_y-cy
        if abs(dy)>=r: return None,None
        dx=np.sqrt(max(r*r-dy*dy,0))
        L,R=(int(cx-dx),int(baseline_y)),(int(cx+dx),int(baseline_y))
        return L,R
    except cv2.error as e:
        print("⚠️ Circle fit failed:", e)
        return None,None

def local_angle(cnt, pt):
    if pt is None: return None
    cx,_=pt;pts=cnt[:,0,:];nb=pts[np.abs(pts[:,0]-cx)<25]
    if len(nb)<8: return None
    x,y=nb[:,0].astype(float),nb[:,1].astype(float)
    try:a,b,c=np.polyfit(x,y,2)
    except: return None
    m=2*a*cx+b
    return float(np.degrees(np.arctan(abs(m))))

# ============================ GUI ============================
class DropApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crash-Proof Sessile Drop Analyzer")
        self.resize(1100,800)
        self.original=None;self.overlay=None
        lay=QVBoxLayout(self)
        self.lbl=QLabel("Load droplet image");self.lbl.setAlignment(Qt.AlignCenter)
        self.lbl.setStyleSheet("border:1px solid #777;");lay.addWidget(self.lbl,stretch=1)
        row=QHBoxLayout();lay.addLayout(row)
        self.load=QPushButton("Load Image");self.auto=QPushButton("Auto Baseline")
        row.addWidget(self.load);row.addWidget(self.auto);row.addWidget(QLabel("Baseline:"))
        self.slider=QSlider(Qt.Horizontal);self.slider.setEnabled(False);row.addWidget(self.slider)
        res=QHBoxLayout();lay.addLayout(res)
        self.ltxt=QLabel("Left: --°");self.rtxt=QLabel("Right: --°")
        res.addWidget(self.ltxt);res.addWidget(self.rtxt)
        self.load.clicked.connect(self.load_image)
        self.auto.clicked.connect(self.auto_baseline)
        self.slider.valueChanged.connect(self.preview)
        self.slider.sliderReleased.connect(self.analyze)

    def load_image(self):
        p,_=QFileDialog.getOpenFileName(self,"Open","","Images (*.png *.jpg *.jpeg *.bmp)")
        if not p:return
        img=cv2.imread(p)
        if img is None:self.lbl.setText("Error reading");return
        if img.shape[1]>1400:
            img=cv2.resize(img,(1400,int(img.shape[0]*1400/img.shape[1])))
        self.original=img;self.analyze(initial=True)

    def auto_baseline(self):
        if self.original is None:return
        gray=cv2.cvtColor(self.original,cv2.COLOR_BGR2GRAY)
        y=detect_baseline_hough(gray)
        self.slider.setValue(y);self.analyze()

    def preview(self):
        if self.overlay is None:return
        y = self.slider.value()
        tmp = self.overlay.copy()
        cv2.line(tmp, (0, y), (tmp.shape[1], y), (255,0,0), 2)
        self.display_image(tmp)
    def analyze(self, initial=False):
        if self.original is None:
            return
        img = self.original.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cnt, base_y = detect_droplet_contour_smart(img)
        if cnt is None:
            self.ltxt.setText("Left:--")
            self.rtxt.setText("Right:--")
            self.display_image(img)
            return
        h = img.shape[0]
        if initial or not self.slider.isEnabled():
            self.slider.blockSignals(True)
            self.slider.setRange(0, h-1)
            self.slider.setValue(base_y)
            self.slider.setEnabled(True)
            self.slider.blockSignals(False)
        y = self.slider.value()
        L, R = fit_circle_and_intersect(cnt, y, gray)
        left, right = local_angle(cnt, L), local_angle(cnt, R)
        self.ltxt.setText(f"Left:{left:.1f}°" if left else "Left:--")
        self.rtxt.setText(f"Right:{right:.1f}°" if right else "Right:--")
        vis = img.copy()
        cv2.drawContours(vis, [cnt], -1, (0,255,0), 2)
        cv2.line(vis, (0, y), (vis.shape[1], y), (255,0,0), 2)
        for p, a in [(L, left), (R, right)]:
            if p:
                cv2.circle(vis, (int(p[0]), int(p[1])), 6, (0,0,255), -1)
            if p and a:
                cv2.putText(vis, f"{a:.1f}°", (int(p[0]) + 8, int(p[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
        self.overlay = vis
        self.display_image(vis)

    def display_image(self, img):
        img_copy = img.copy()  # ✅ ensure independent buffer
        h, w = img_copy.shape[:2]
        qimg = QImage(img_copy.data, w, h, 3*w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qimg)
        self.lbl.setPixmap(pix.scaled(self.lbl.width(), self.lbl.height(),
                                      Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self,e):
        if self.overlay is not None:self.display_image(self.overlay)
        super().resizeEvent(e)

# ============================ Main ============================
if __name__=="__main__":
    app=QApplication(sys.argv)
    w=DropApp();w.show()
    sys.exit(app.exec_())
