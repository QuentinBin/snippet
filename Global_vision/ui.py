import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, pyqtSignal
from sklearn.cluster import KMeans

class VisionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        # 变量初始化
        self.cap = None  # 摄像头/视频
        self.clicked_signal_roi_position_1 = pyqtSignal(tuple) # 初始roi区域信号1
        self.clicked_signal_roi_position_2 = pyqtSignal(tuple) # 初始roi区域信号2
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.roi = None  # (x, y, w, h)
        self.tracking = False  # 是否启用追踪

    def initUI(self):
        """ 初始化 UI 界面 """
        self.setWindowTitle("全局视觉定位")
        self.setGeometry(100, 100, 800, 600)

        # 界面布局
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)

        self.btn_start_cam = QPushButton("启动摄像头", self)
        self.btn_load_video = QPushButton("导入视频", self)
        self.btn_select_roi = QPushButton("选择ROI", self)
        self.btn_toggle_tracking = QPushButton("开始追踪", self)

        # 布局管理
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.btn_start_cam)
        layout.addWidget(self.btn_load_video)
        layout.addWidget(self.btn_select_roi)
        layout.addWidget(self.btn_toggle_tracking)
        self.setLayout(layout)

        # 连接按钮
        self.btn_start_cam.clicked.connect(self.start_camera)
        self.btn_load_video.clicked.connect(self.load_video)
        self.btn_select_roi.clicked.connect(self.select_roi)
        self.btn_toggle_tracking.clicked.connect(self.toggle_tracking)

    def start_camera(self):
        """ 启动摄像头 """
        self.cap = cv2.VideoCapture(0)  # 摄像头
        self.timer.start(30)  # 每 30ms 读取一帧

    def load_video(self):
        """ 选择并加载视频 """
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.timer.start(30)

    def select_roi(self):
        """ 选择 ROI 区域 """
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.roi = cv2.selectROI("选择ROI", frame, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("选择ROI")

    def toggle_tracking(self):
        """ 开启/关闭追踪 """
        if self.tracking:
            self.tracking = False
            self.btn_toggle_tracking.setText("开始追踪")
        else:
            self.tracking = True
            self.btn_toggle_tracking.setText("停止追踪")

    def update_frame(self):
        """ 读取摄像头/视频帧并更新 """
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return

        # 获取 QLabel 的大小
        label_width = self.video_label.width()
        label_height = self.video_label.height()

        # 计算缩放比例，保持纵横比
        frame_height, frame_width = frame.shape[:2]
        scale = min(label_width / frame_width, label_height / frame_height)
        new_size = (int(frame_width * scale), int(frame_height * scale))

        # 缩放帧到 QLabel 尺寸
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

        # 处理ROI
        if self.roi and self.tracking:
            

                # 绘制标记点
                cv2.circle(frame, (new_x, new_y), 5, (0, 255, 0), -1)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 转换为 Qt 格式
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def extract_color_centers(self, image, n_clusters=2):
        """ KMeans 提取红、黄标志点 """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 颜色范围
        yellow_lower, yellow_upper = np.array([20, 100, 100]), np.array([30, 255, 255])
        red_lower1, red_upper1 = np.array([0, 100, 100]), np.array([10, 255, 255])
        red_lower2, red_upper2 = np.array([170, 100, 100]), np.array([180, 255, 255])

        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
        mask_red = mask_red1 | mask_red2

        mask = mask_yellow | mask_red
        points = np.column_stack(np.where(mask > 0))

        if len(points) >= n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, n_init=10)
            kmeans.fit(points)
            centers = kmeans.cluster_centers_[:, ::-1]
            return [tuple(map(int, c)) for c in centers]

        return []

    def closeEvent(self, event):
        """ 关闭窗口时释放资源 """
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisionApp()
    window.show()
    sys.exit(app.exec_())
