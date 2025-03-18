import sys
import cv2
import numpy as np
import csv
import serial
from serial.tools import list_ports
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSlider,QSizePolicy, QComboBox
from PyQt5.QtWidgets import QLineEdit, QFormLayout, QFrame, QShortcut
from PyQt5.QtCore import Qt, QDateTime, pyqtSignal, QThread, QObject, QTimer
from PyQt5.QtGui import QImage, QPixmap
from queue import Queue
from PyQt5.QtWidgets import QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import struct
import re
import time
import gxipy as gx
import random



K = np.array([[515.8139513744953, 0.0, 1196.667955373116], [0.0, 514.3908924982283, 1037.0549098545412], [0.0, 0.0, 1.0]])
D = np.array([[0.007190500081416133], [-0.004884565917566122], [0.006158251646770605], [-0.003034926245524286]])
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (2448, 2048), cv2.CV_16SC2)


def data_augment(image, factor):
    float_image = image.astype(np.float32)

    # 创建一个高斯核
    kernel = cv2.GaussianBlur(float_image, (0, 0), sigmaX=factor, sigmaY=factor, borderType=cv2.BORDER_REFLECT_101)

    # 提亮图片
    enhanced_image = float_image + kernel

    # 限制像素值并转换回uint8类型
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)

    return enhanced_image


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(PlotCanvas, self).__init__(fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, data):
        self.axes.clear()
        self.axes.plot(data)
        self.draw()


class ImageProcessor(QThread):
    frame_processed = pyqtSignal(np.ndarray)

    def __init__(self, frame_queue, hsv1, hsv2):
        super().__init__()
        self.frame_queue = frame_queue
        self.running = False
        self.hsv1 = hsv1
        self.hsv2 = hsv2
        self.selected_point_1 = None
        self.selected_point_2 = None

        self.SUCCESS_TRACK_1 = False
        self.SUCCESS_TRACK_2 = False

        self.prev_positions_marker1 = (None, None)
        self.prev_positions_marker2 = (None, None)

        self.marker1_hsv_history = None
        self.marker2_hsv_history = None

        self.V_global = (None,None)
        self.angle_velocity = None
        self.u = 0
        self.v = 0

        self.min_area = 50  # 根据需要调整
        self.max_area = 1000  # 根据需要调整
        self.aspect_ratio_range = (0.8, 1.2)  # 根据需要调整

        # 储存位置和角度
        self.heading_angle = 0
        self.heading_angle_last = 0
        self.pixel_x = 0
        self.pixel_y = 0
        self.pixel_x_lasts = [0]*100
        self.pixel_y_lasts = [0]*100


    def run(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                if frame is not None:
                    result= self.process_frame(frame)
                    self.frame_processed.emit(result)
            QThread.msleep(10)  # Control processing frame rate

    def process_frame(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        test_frame = frame.copy()
        mask1 = cv2.inRange(hsv, np.array(self.hsv1[:3]), np.array(self.hsv1[3:]))
        mask2 = cv2.inRange(hsv, np.array(self.hsv2[:3]), np.array(self.hsv2[3:]))
        ###############################################################################################
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        ###############################################################################################

        thresh1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel, iterations=4)
        thresh2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=4)
        # thresh1 = mask1
        # thresh2 = mask2
        contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_dist_1 = float('inf')
        nearest_contour1 = None
        for cnt1 in contours1:
            # if len(cnt1) >= 4:
            area = cv2.contourArea(cnt1)
            if not (self.min_area <= area <= self.max_area):
                continue
            M = cv2.moments(cnt1)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                dist = (cX - self.selected_point_1[0]) ** 2 + (cY - self.selected_point_1[1]) ** 2
                if dist < min_dist_1:
                    min_dist_1 = dist
                    nearest_contour1 = cnt1

        if nearest_contour1 is not None:

            cv2.drawContours(test_frame, [nearest_contour1], -1, (0, 255, 0), 2)
            M = cv2.moments(nearest_contour1)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # self.marker_positions.append((cX, cY))
            self.selected_point_1 = (cX, cY)

            # Calculate HSV average within the contour
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [nearest_contour1], -1, color=255, thickness=-1)
            hsv_values = hsv[mask == 255]
            average_hsv = np.mean(hsv_values, axis=0)
            self.marker1_hsv_history = average_hsv

            if self.SUCCESS_TRACK_1 == False:
                self.SUCCESS_TRACK_1 = True
                self.prev_positions_marker1 = (cX, cY)


        else:
            self.SUCCESS_TRACK_1 = False
            self.prev_positions_marker1 = (None, None)
            self.marker1_hsv_history = self.hsv1




        contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_dist_2 = float('inf')
        nearest_contour2 = None
        for cnt2 in contours2:
            area = cv2.contourArea(cnt2)
            if not (self.min_area <= area <= self.max_area):
                continue
            M = cv2.moments(cnt2)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                dist = (cX - self.selected_point_2[0]) ** 2 + (cY - self.selected_point_2[1]) ** 2
                if dist < min_dist_2:
                    min_dist_2 = dist
                    nearest_contour2 = cnt2

        if nearest_contour2 is not None:
            cv2.drawContours(test_frame, [nearest_contour2], -1, (0, 255, 0), 2)
            M = cv2.moments(nearest_contour2)
            cX2 = int(M["m10"] / M["m00"])
            cY2 = int(M["m01"] / M["m00"])
            # self.marker_positions.append((cX, cY))
            self.selected_point_2 = (cX2, cY2)

            # Calculate HSV average within the contour
            mask_2 = np.zeros(hsv.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask_2, [nearest_contour2], -1, color=255, thickness=-1)
            hsv2_values = hsv[mask_2 == 255]
            average_hsv2 = np.mean(hsv2_values, axis=0)
            self.marker2_hsv_history = average_hsv2
            # print(average_hsv2)

            if self.SUCCESS_TRACK_2 == False:
                self.SUCCESS_TRACK_2 = True
                self.prev_positions_marker2 = (cX2, cY2)

        else:
            self.SUCCESS_TRACK_2 = False
            self.prev_positions_marker2 = (None,None)
            self.marker2_hsv_history = self.hsv2

        if self.SUCCESS_TRACK_1 and self.SUCCESS_TRACK_2:
            # print('entered this')
            V_marker1_x = self.selected_point_1[0] - self.prev_positions_marker1[0]
            V_marker1_y = self.selected_point_1[1] - self.prev_positions_marker1[1]
            V_marker2_x = self.selected_point_2[0] - self.prev_positions_marker2[0]
            V_marker2_y = self.selected_point_2[1] - self.prev_positions_marker2[1]

            angle_prev = np.arctan2(self.prev_positions_marker1[1]-self.prev_positions_marker2[1], self.prev_positions_marker1[0]-self.prev_positions_marker2[0])
            angle_curr = np.arctan2(self.selected_point_1[1]-self.selected_point_2[1], self.selected_point_1[0]-self.selected_point_2[0])

            V_global_x = (V_marker1_x + V_marker2_x) / 2
            V_global_y = (V_marker1_y + V_marker2_y) / 2
            self.V_global = (V_global_x, V_global_y)
            self.u = 0
            self.v = 0

            self.angle_velocity = angle_curr - angle_prev

            self.prev_positions_marker1 = self.selected_point_1
            self.prev_positions_marker2 = self.selected_point_2

            center_x = int((self.selected_point_1[0]+ self.selected_point_2[0])/2)
            center_y = int((self.selected_point_1[1]+ self.selected_point_2[1])/2)
            start_point = (center_x, center_y)
            end_point = (int(start_point[0] + V_global_x * 20+1), int(start_point[1] + V_global_y * 20+1))
            cv2.arrowedLine(test_frame, start_point, end_point, (0, 255, 0), 2)

        # combined_mask = cv2.bitwise_or(mask1, mask2)
        result = test_frame
        return result


    def start_processing(self):
        if not self.running:
            self.running = True
            self.start()


    def stop_processing(self):
        self.running = False
        self.wait()


    def update_thresholds(self, hsv1, hsv2):
        self.hsv1 = hsv1
        self.hsv2 = hsv2


    def update_selected_point1(self, pos):
        self.selected_point_1 = pos
        print(pos)


    def update_selected_point2(self, pos):
        self.selected_point_2 = pos
        print(pos)


class CameraCapture(QThread):
    frame_captured = pyqtSignal(np.ndarray)


    def __init__(self):
        super().__init__()
        self.running = False
        self.cap = cv2.VideoCapture(0)
        # self.cam = cam
        self.video_file = None

    def set_video_file(self, video_file):
        self.video_file = video_file
        if self.cap.isOpened():
            self.cap.release()
        self.cap = cv2.VideoCapture(video_file)

    def run(self):
        device_manager = gx.DeviceManager()
        dev_num, dev_info_list = device_manager.update_device_list()
        if dev_num == 0:
            print("Number of enumerated devices is 0")
            return

        # open the first device
        cam = device_manager.open_device_by_index(1)

        # exit when the camera is a mono camera
        if cam.PixelColorFilter.is_implemented() is False:
            print("This sample does not support mono camera.")
            cam.close_device()
            return

        # set continuous acquisition
        cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

        # set exposure
        cam.ExposureTime.set(60000.0)

        # set gain
        cam.Gain.set(2.)

        # get param of improving image quality
        if cam.GammaParam.is_readable():
            gamma_value = cam.GammaParam.get()
            self.gamma_lut = gx.Utility.get_gamma_lut(gamma_value)
        else:
            self.gamma_lut = None
        if cam.ContrastParam.is_readable():
            contrast_value = cam.ContrastParam.get()
            self.contrast_lut = gx.Utility.get_contrast_lut(contrast_value)
        else:
            self.contrast_lut = None
        if cam.ColorCorrectionParam.is_readable():
            self.color_correction_param = cam.ColorCorrectionParam.get()
        else:
            self.color_correction_param = 0

        # set the acq buffer count
        cam.data_stream[0].set_acquisition_buffer_number(10)
        # start data acquisition
        cam.stream_on()
        while self.running:
            raw_image = cam.data_stream[0].get_image()

            if raw_image is None:
                print("Getting image failed.")
                continue

            if raw_image.get_status() == gx.GxFrameStatusList.INCOMPLETE:
                pass
            # get RGB image from raw image
            rgb_image = raw_image.convert("RGB")
            # if rgb_image is None:
            #     continue

            # improve image quality
            rgb_image.image_improvement(self.color_correction_param, self.contrast_lut, self.gamma_lut)

            # create numpy array with data from raw image
            numpy_image = rgb_image.get_numpy_array()
            frame = cv2.cvtColor(np.asarray(numpy_image), cv2.COLOR_BGR2RGB)
            # undistorted_img = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR,
            #                             borderMode=cv2.BORDER_CONSTANT)
            # undistorted_img = cv2.rotate(undistorted_img[400:-150, 620:-680], cv2.ROTATE_90_COUNTERCLOCKWISE)
            undistorted_img = frame
            if undistorted_img is not None:
                self.frame_captured.emit(undistorted_img)
            else:
                if self.video_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
                else:
                    self.stop_capture()  # Stop if it's a camera
            # QThread.msleep(20)  # Control capture frame rate

    def start_capture(self):
        if not self.running:
            self.running = True
            self.start()

    def stop_capture(self):
        self.running = False
        self.wait()
        self.cap.release()

class SignalEmitter(QObject):
    clicked_position_signal1 = pyqtSignal(tuple)
    clicked_position_signal2 = pyqtSignal(tuple)

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.SignalEmitter = SignalEmitter()
        self.initUI()
        self.is_recording = False
        self.is_processing = False
        self.recorded_frames = []
        self.video_writer = None
        self.plot_data = []
        self.frame_queue = Queue(maxsize=10)

        self.is_saving_data = False
        self.data_file = None
        self.csv_writer = None
        self.data_file_name = ""


        self.camera_capture = CameraCapture()
        self.camera_capture.frame_captured.connect(self.update_frame)


        self.hsv1 = [0, 0, 0, 255, 255, 255]
        self.hsv2 = [0, 0, 0, 255, 255, 255]
        self.processor = ImageProcessor(self.frame_queue, self.hsv1, self.hsv2)
        self.processor.frame_processed.connect(self.display_processed_image)
        self.SignalEmitter.clicked_position_signal1.connect(self.processor.update_selected_point1)
        self.SignalEmitter.clicked_position_signal2.connect(self.processor.update_selected_point2)

        self.f_L = 0.
        self.f_R = 0.
        self.f_tail = 0.
        self.amp_foward = 0
        self.amp_turn = 0
        self.amp_L = 30
        self.amp_R = 30
        self.bias_tail = 0
        
        self.marker1_x = 0
        self.marker1_y = 0
        self.marker2_x = 0
        self.marker2_y = 0


        # backstepping:
        self.goal_pos_pixel = np.array([700, 185])
        self.final_orientation_rad = np.deg2rad(0)
        self.control_points = None

        self.fish_pos_pixel = None
        self.fish_pos = None
        self.fish_heading = None


        self.RoboDact_Contoller = RobotDact()
        self.BackStepping_timer = QTimer()
        self.BackStepping_timer.timeout.connect(self.backstepping_task)
        self.BackStepping_timer.start(100) # 0.1s


    def initUI(self):
        # 设置窗口标题
        self.setWindowTitle("Real-Time Camera with Dual HSV Processing and Sliders")

        # 创建用于显示图像的标签
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640,480)
        self.result_label = QLabel(self)
        self.result_label.setFixedSize(640, 480)

        # 创建用于显示速度变化曲线的绘图画布
        self.plot_canvas = PlotCanvas(self, width=5, height=4)

        # 创建用于显示选定颜色的标签
        self.selected_color_label = QLabel(self)
        self.selected_color_label.setFixedSize(100, 50)
        self.selected_color_label.setStyleSheet("background-color: rgb(0, 0, 0);")

        self.serial_port_label = QLabel("选择串口:", self)
        self.serial_port_combobox = QComboBox(self)
        self.update_serial_ports()
        self.serial_port_combobox.currentIndexChanged.connect(self.on_serial_port_change)
        self.serial_port = None

        # 创建用于显示标记状态的标签
        self.marker1_label = QLabel("Marker 1: None", self)
        self.marker2_label = QLabel("Marker 2: None", self)

        # 创建按钮并连接到各自的函数
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)

        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.clicked.connect(self.stop_camera)

        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.start_recording)

        self.stop_record_button = QPushButton("Stop Recording")
        self.stop_record_button.clicked.connect(self.stop_recording)

        self.screenshot_button = QPushButton("Take Screenshot")
        self.screenshot_button.clicked.connect(self.take_screenshot)

        self.save_button = QPushButton("Start Saving Data")
        self.save_button.clicked.connect(self.start_saving_data)

        self.stop_save_button = QPushButton("Stop Saving Data")
        self.stop_save_button.clicked.connect(self.stop_saving_data)

        self.process_button = QPushButton("Start Processing")
        self.process_button.clicked.connect(self.start_processing)

        self.stop_process_button = QPushButton("Stop Processing")
        self.stop_process_button.clicked.connect(self.stop_processing)

        self.select_color_button_1 = QPushButton("Select Color for 1")
        self.select_color_button_1.clicked.connect(self.enable_color_selection_1)

        self.select_color_button_2 = QPushButton("Select Color for 2")
        self.select_color_button_2.clicked.connect(self.enable_color_selection_2)

        self.select_video_button = QPushButton("Select Video File")
        self.select_video_button.clicked.connect(self.select_video_file)

        # 创建HSV阈值滑块并连接到更新函数
        self.create_hsv_sliders()

        # 创建用于滑块的布局
        hbox_sliders = self.create_slider_layouts()

        # 创建按钮的垂直布局
        buttons_vbox = self.create_button_layout()

        # 将滑块和状态显示添加到按钮布局
        buttons_vbox.addLayout(hbox_sliders)
        buttons_vbox.addWidget(self.marker1_label)
        buttons_vbox.addWidget(self.marker2_label)

        # 添加selected_color_label到按钮布局中
        buttons_vbox.addWidget(QLabel("Selected Color:"))
        buttons_vbox.addWidget(self.selected_color_label)
        self.data_file_label = QLabel("当前没有正在记录的数据文件", self)
        buttons_vbox.addWidget(self.data_file_label)

        # 创建图像的水平布局
        images_hbox = QHBoxLayout()
        images_hbox.addWidget(self.image_label)
        images_hbox.addWidget(self.result_label)

        # 创建绘图画布的垂直布局
        canvas_hbox = QVBoxLayout()
        canvas_hbox.addWidget(self.plot_canvas)

        # 创建右侧布局，包含图像和绘图画布
        right_layout = QVBoxLayout()
        right_layout.addLayout(images_hbox)
        right_layout.addLayout(canvas_hbox)

        # 添加控制命令的部分
        control_layout = QVBoxLayout()
        self.control_frame = QFrame(self)
        self.control_frame.setFrameShape(QFrame.StyledPanel)
        self.control_frame.setLayout(control_layout)

        # 添加参数输入框
        # 控制命令部分
        self.param1_slider = QSlider(Qt.Horizontal)
        self.param1_slider.setRange(0, 100)
        self.param1_slider.setValue(50)

        self.param2_slider = QSlider(Qt.Horizontal)
        self.param2_slider.setRange(0, 100)
        self.param2_slider.setValue(50)

        param_form_layout = QFormLayout()
        param_form_layout.addRow("胸鳍波动频率:", self.param1_slider)
        param_form_layout.addRow("尾鳍摆动频率:", self.param2_slider)



        # 添加控制命令按钮
        self.command1_button = QPushButton("前进(W)", self)
        self.command1_button.clicked.connect(self.send_command1)
        command1_shortcut = QShortcut("W", self)
        command1_shortcut.activated.connect(self.send_command1)

        self.command2_button = QPushButton("左转(A)", self)
        self.command2_button.clicked.connect(self.send_command2)
        command2_shortcut = QShortcut("A", self)
        command2_shortcut.activated.connect(self.send_command2)

        self.command3_button = QPushButton("右转(D)", self)
        self.command3_button.clicked.connect(self.send_command3)
        command3_shortcut = QShortcut("D", self)
        command3_shortcut.activated.connect(self.send_command3)

        self.command4_button = QPushButton("停止运动(S)", self)
        self.command4_button.clicked.connect(self.send_command4)
        command4_shortcut = QShortcut("S", self)
        command4_shortcut.activated.connect(self.send_command4)

        # 添加串口状态显示
        self.serial_status_label = QLabel("串口状态: 未连接", self)

        # 打开串口按钮
        self.open_serial_button = QPushButton("打开串口", self)
        self.open_serial_button.clicked.connect(self.open_serial_port)

        # 关闭串口按钮
        self.close_serial_button = QPushButton("关闭串口", self)
        self.close_serial_button.clicked.connect(self.close_serial_port)

        # 测试串口
        self.test_serial_button = QPushButton("测试命令", self)
        self.test_serial_button.clicked.connect(self.test_serial_comm)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.command1_button)
        button_layout.addWidget(self.command2_button)
        button_layout.addWidget(self.command3_button)
        button_layout.addWidget(self.command4_button)
        control_layout.addLayout(param_form_layout)
        control_layout.addLayout(button_layout)

        # 布局
        serial_layout = QVBoxLayout()
        serial_layout.addWidget(self.serial_port_label)
        serial_layout.addWidget(self.serial_port_combobox)
        serial_layout.addWidget(self.serial_status_label)
        serial_layout.addWidget(self.open_serial_button)
        serial_layout.addWidget(self.close_serial_button)
        serial_layout.addWidget(self.test_serial_button)



        # 创建顶部和底部的水平布局
        top_layout = QHBoxLayout()
        bottom_layout = QHBoxLayout()

        # 将图像标签添加到顶部布局
        top_layout.addWidget(self.image_label)
        top_layout.addWidget(self.result_label)

        # 将控制帧和按钮布局添加到底部布局
        bottom_layout.addWidget(self.control_frame)
        bottom_layout.addLayout(buttons_vbox)

        # 创建主水平布局并添加顶部和底部布局
        main_hbox = QHBoxLayout()
        main_hbox.addLayout(top_layout)
        main_hbox.addLayout(bottom_layout)
        main_hbox.addWidget(self.plot_canvas)
        # 将串口布局添加到主布局中
        main_hbox.addLayout(serial_layout)

        # 设置主布局
        self.setLayout(main_hbox)

        self.hsv_history_1 = []
        self.hsv_history_2 = []
        self.max_history_size = 10

    def create_hsv_sliders(self):
        # 创建HSV阈值滑块并连接到更新函数
        self.h1_min_slider = QSlider(Qt.Horizontal)
        self.h1_min_slider.setRange(0, 255)
        self.h1_min_slider.setValue(0)
        self.h1_min_slider.valueChanged.connect(self.update_thresholds)

        self.s1_min_slider = QSlider(Qt.Horizontal)
        self.s1_min_slider.setRange(0, 255)
        self.s1_min_slider.setValue(0)
        self.s1_min_slider.valueChanged.connect(self.update_thresholds)

        self.v1_min_slider = QSlider(Qt.Horizontal)
        self.v1_min_slider.setRange(0, 255)
        self.v1_min_slider.setValue(0)
        self.v1_min_slider.valueChanged.connect(self.update_thresholds)

        self.h1_max_slider = QSlider(Qt.Horizontal)
        self.h1_max_slider.setRange(0, 255)
        self.h1_max_slider.setValue(255)
        self.h1_max_slider.valueChanged.connect(self.update_thresholds)

        self.s1_max_slider = QSlider(Qt.Horizontal)
        self.s1_max_slider.setRange(0, 255)
        self.s1_max_slider.setValue(255)
        self.s1_max_slider.valueChanged.connect(self.update_thresholds)

        self.v1_max_slider = QSlider(Qt.Horizontal)
        self.v1_max_slider.setRange(0, 255)
        self.v1_max_slider.setValue(255)
        self.v1_max_slider.valueChanged.connect(self.update_thresholds)

        self.h2_min_slider = QSlider(Qt.Horizontal)
        self.h2_min_slider.setRange(0, 255)
        self.h2_min_slider.setValue(0)
        self.h2_min_slider.valueChanged.connect(self.update_thresholds)

        self.s2_min_slider = QSlider(Qt.Horizontal)
        self.s2_min_slider.setRange(0, 255)
        self.s2_min_slider.setValue(0)
        self.s2_min_slider.valueChanged.connect(self.update_thresholds)

        self.v2_min_slider = QSlider(Qt.Horizontal)
        self.v2_min_slider.setRange(0, 255)
        self.v2_min_slider.setValue(0)
        self.v2_min_slider.valueChanged.connect(self.update_thresholds)

        self.h2_max_slider = QSlider(Qt.Horizontal)
        self.h2_max_slider.setRange(0, 255)
        self.h2_max_slider.setValue(255)
        self.h2_max_slider.valueChanged.connect(self.update_thresholds)

        self.s2_max_slider = QSlider(Qt.Horizontal)
        self.s2_max_slider.setRange(0, 255)
        self.s2_max_slider.setValue(255)
        self.s2_max_slider.valueChanged.connect(self.update_thresholds)

        self.v2_max_slider = QSlider(Qt.Horizontal)
        self.v2_max_slider.setRange(0, 255)
        self.v2_max_slider.setValue(255)
        self.v2_max_slider.valueChanged.connect(self.update_thresholds)

    def create_slider_layouts(self):
        # 创建用于滑块的布局
        hbox_sliders = QHBoxLayout()

        # 第一个滑块组
        vbox_min_sliders1 = QVBoxLayout()
        vbox_max_sliders1 = QVBoxLayout()

        vbox_min_sliders1.addWidget(QLabel('H1 Min'))
        vbox_min_sliders1.addWidget(self.h1_min_slider)
        vbox_min_sliders1.addWidget(QLabel('S1 Min'))
        vbox_min_sliders1.addWidget(self.s1_min_slider)
        vbox_min_sliders1.addWidget(QLabel('V1 Min'))
        vbox_min_sliders1.addWidget(self.v1_min_slider)

        vbox_max_sliders1.addWidget(QLabel('H1 Max'))
        vbox_max_sliders1.addWidget(self.h1_max_slider)
        vbox_max_sliders1.addWidget(QLabel('S1 Max'))
        vbox_max_sliders1.addWidget(self.s1_max_slider)
        vbox_max_sliders1.addWidget(QLabel('V1 Max'))
        vbox_max_sliders1.addWidget(self.v1_max_slider)

        # 第二个滑块组
        vbox_min_sliders2 = QVBoxLayout()
        vbox_max_sliders2 = QVBoxLayout()

        vbox_min_sliders2.addWidget(QLabel('H2 Min'))
        vbox_min_sliders2.addWidget(self.h2_min_slider)
        vbox_min_sliders2.addWidget(QLabel('S2 Min'))
        vbox_min_sliders2.addWidget(self.s2_min_slider)
        vbox_min_sliders2.addWidget(QLabel('V2 Min'))
        vbox_min_sliders2.addWidget(self.v2_min_slider)

        vbox_max_sliders2.addWidget(QLabel('H2 Max'))
        vbox_max_sliders2.addWidget(self.h2_max_slider)
        vbox_max_sliders2.addWidget(QLabel('S2 Max'))
        vbox_max_sliders2.addWidget(self.s2_max_slider)
        vbox_max_sliders2.addWidget(QLabel('V2 Max'))
        vbox_max_sliders2.addWidget(self.v2_max_slider)

        hbox_sliders.addLayout(vbox_min_sliders1)
        hbox_sliders.addLayout(vbox_max_sliders1)
        hbox_sliders.addLayout(vbox_min_sliders2)
        hbox_sliders.addLayout(vbox_max_sliders2)

        return hbox_sliders

    def create_button_layout(self):
        # 创建按钮的垂直布局
        buttons_vbox = QVBoxLayout()
        buttons_vbox.addWidget(self.start_button)
        buttons_vbox.addWidget(self.stop_button)
        buttons_vbox.addWidget(self.record_button)
        buttons_vbox.addWidget(self.stop_record_button)
        buttons_vbox.addWidget(self.screenshot_button)
        buttons_vbox.addWidget(self.save_button)
        buttons_vbox.addWidget(self.stop_save_button)
        buttons_vbox.addWidget(self.process_button)
        buttons_vbox.addWidget(self.stop_process_button)
        buttons_vbox.addWidget(self.select_color_button_1)
        buttons_vbox.addWidget(self.select_color_button_2)
        buttons_vbox.addWidget(self.select_video_button)

        return buttons_vbox

    def start_camera(self):
        self.camera_capture.start_capture()

    def stop_camera(self):
        self.camera_capture.stop_capture()

    def select_video_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        video_file, _ = QFileDialog.getOpenFileName(self, "Select Video File", "",
                                                    "Video Files (*.avi *.mp4 *.mov);;All Files (*)", options=options)
        if video_file:
            self.camera_capture.set_video_file(video_file)
            self.start_camera()
    def start_recording(self):
        # if self.is_processing:
        self.is_recording = True
        current_time = QDateTime.currentDateTime().toString("hh_mm_ss")
        self.video_writer = cv2.VideoWriter(f"{current_time}.avi", cv2.VideoWriter.fourcc(*'XVID'), 20.0,
                                            (640, 480))

    def stop_recording(self):
        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

    def take_screenshot(self):
        current_time = QDateTime.currentDateTime().toString("hh_mm_ss")
        cv2.imwrite(f"{current_time}.png", self.frame)

    def start_saving_data(self):
        if not self.is_saving_data:
            self.is_saving_data = True
            current_time = QDateTime.currentDateTime().toString("hh_mm_ss")
            self.data_file_name = current_time + ".csv"
            self.csv_file = open(f"{current_time}.csv", mode='w', newline='')
            self.data_file_label.setText(f"正在记录数据文件: {self.data_file_name}")
            self.save_button.setEnabled(False)
            self.csv_writer = csv.writer(self.csv_file)
            # self.csv_writer.writerow(["Mean Pixel Value"])  # Change this header as needed

    def stop_saving_data(self):
        if self.is_saving_data:
            self.is_saving_data = False
            if self.csv_writer:
                self.csv_writer = None
                self.csv_file.close()

            self.save_button.setEnabled(True)

            self.data_file_label.setText("没有正在记录的数据文件")

    def start_processing(self):
        self.is_processing = True
        self.processor.start_processing()

    def stop_processing(self):
        self.is_processing = False
        self.processor.stop_processing()

    def enable_color_selection_1(self):
        self.image_label.mousePressEvent = self.select_color_1

    def select_color_1(self, event):
        label_width = self.image_label.width()
        label_height = self.image_label.height()

        # 获取图像的实际尺寸
        frame_height, frame_width, _ = self.frame.shape

        # 计算缩放比例
        scale_width = frame_width / label_width
        scale_height = frame_height / label_height

        # 获取鼠标点击的相对位置
        x = event.pos().x()
        y = event.pos().y()

        # 将相对位置转换为图像中的实际位置
        x = int(x * scale_width)
        y = int(y * scale_height)

        clicked_position = (x, y)
        self.SignalEmitter.clicked_position_signal1.emit(clicked_position)

        frame = self.frame  # 需要确保在 `update_frame` 中保存当前帧到 `self.current_frame`

        if frame is not None and 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            selected_color = frame[y, x]
            selected_color_hsv = np.int_(cv2.cvtColor(np.uint8([[selected_color]]), cv2.COLOR_BGR2HSV)[0][0])

            self.h1_min_slider.setValue(max(0, selected_color_hsv[0] - 40))
            self.s1_min_slider.setValue(max(0, selected_color_hsv[1] - 40))
            self.v1_min_slider.setValue(max(0, selected_color_hsv[2] - 40))
            self.h1_max_slider.setValue(min(255, selected_color_hsv[0] + 40))
            self.s1_max_slider.setValue(min(255, selected_color_hsv[1] + 40))
            self.v1_max_slider.setValue(min(255, selected_color_hsv[2] + 40))

            print(selected_color_hsv)


            self.selected_color_label.setStyleSheet(
                f"background-color: rgb({selected_color[2]}, {selected_color[1]}, {selected_color[0]});")

    def enable_color_selection_2(self):
        self.image_label.mousePressEvent = self.select_color_2

    def select_color_2(self, event):

        label_width = self.image_label.width()
        label_height = self.image_label.height()

        # 获取图像的实际尺寸
        frame_height, frame_width, _ = self.frame.shape

        # 计算缩放比例
        scale_width = frame_width / label_width
        scale_height = frame_height / label_height

        # 获取鼠标点击的相对位置
        x = event.pos().x()
        y = event.pos().y()

        # 将相对位置转换为图像中的实际位置
        x = int(x * scale_width)
        y = int(y * scale_height)

        clicked_position = (x, y)
        self.SignalEmitter.clicked_position_signal2.emit(clicked_position)
        frame = self.frame  # 需要确保在 `update_frame` 中保存当前帧到 `self.current_frame`

        if frame is not None and 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            selected_color = frame[y, x]

            selected_color_hsv = np.int_(cv2.cvtColor(np.uint8([[selected_color]]), cv2.COLOR_BGR2HSV)[0][0])
            self.h2_min_slider.setValue(max(0, selected_color_hsv[0] - 60))
            self.s2_min_slider.setValue(max(0, selected_color_hsv[1] - 60))
            self.v2_min_slider.setValue(max(0, selected_color_hsv[2] - 60))
            self.h2_max_slider.setValue(min(255, selected_color_hsv[0] + 60))
            self.s2_max_slider.setValue(min(255, selected_color_hsv[1] + 60))
            self.v2_max_slider.setValue(min(255, selected_color_hsv[2] + 60))

            print(selected_color)

            self.selected_color_label.setStyleSheet(
                f"background-color: rgb({selected_color[2]}, {selected_color[1]}, {selected_color[0]});")

    def update_thresholds(self):
        hsv1 = [
            self.h1_min_slider.value(),
            self.s1_min_slider.value(),
            self.v1_min_slider.value(),
            self.h1_max_slider.value(),
            self.s1_max_slider.value(),
            self.v1_max_slider.value()
        ]
        hsv2 = [
            self.h2_min_slider.value(),
            self.s2_min_slider.value(),
            self.v2_min_slider.value(),
            self.h2_max_slider.value(),
            self.s2_max_slider.value(),
            self.v2_max_slider.value()
        ]
        self.processor.update_thresholds(hsv1, hsv2)

    def update_frame(self, frame):
        self.frame = frame
        height, width, _ = frame.shape
        if self.is_recording:
            self.video_writer.write(frame)
        if self.is_processing:
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
        self.image_label.setFixedSize(width, height)
        self.result_label.setFixedSize(width, height)
        self.display_image(frame, self.image_label)

        if self.processor.SUCCESS_TRACK_1:
            self.update_hsv_history_1(self.processor.marker1_hsv_history)
            if len(self.hsv_history_1)==self.max_history_size:
                mean_hsv1 = np.int_(np.mean(np.array(self.hsv_history_1), axis=0))
                # print(mean_hsv1)
                self.h1_min_slider.setValue(max(0, mean_hsv1[0] - 40))
                self.s1_min_slider.setValue(max(0, mean_hsv1[1] - 40))
                self.v1_min_slider.setValue(max(0, mean_hsv1[2] - 40))
                self.h1_max_slider.setValue(min(255, mean_hsv1[0] + 40))
                self.s1_max_slider.setValue(min(255, mean_hsv1[1] + 40))
                self.v1_max_slider.setValue(min(255, mean_hsv1[2] + 40))
        #
        if self.processor.SUCCESS_TRACK_2:
            self.update_hsv_history_2(self.processor.marker2_hsv_history)
            if len(self.hsv_history_2)==self.max_history_size:
                mean_hsv2 = np.int_(np.mean(np.array(self.hsv_history_2), axis=0))
                self.h2_min_slider.setValue(max(0, mean_hsv2[0] - 40))
                self.s2_min_slider.setValue(max(0, mean_hsv2[1] - 40))
                self.v2_min_slider.setValue(max(0, mean_hsv2[2] - 40))
                self.h2_max_slider.setValue(min(255, mean_hsv2[0] + 40))
                self.s2_max_slider.setValue(min(255, mean_hsv2[1] + 40))
                self.v2_max_slider.setValue(min(255, mean_hsv2[2] + 40))
        # if self.processor.angle_velocity is not None:
        #     self.plot_data.append(self.processor.V_global[0])
        #     self.plot_canvas.plot(self.plot_data)

    def display_processed_image(self, frame):

        self.display_image(frame, self.result_label)
        if self.processor.SUCCESS_TRACK_1:
            self.marker1_label.setText(f"Marker 1: {self.processor.selected_point_1}")
            self.marker1_x = int(self.processor.selected_point_1[0])
            self.marker1_y = int(self.processor.selected_point_1[1])
        else:
            self.marker1_label.setText("Marker 1: None")

        if self.processor.SUCCESS_TRACK_2:
            self.marker2_la+bel.setText(f"Marker 2: {self.processor.selected_point_2}")
            self.marker2_x = int(self.processor.selected_point_2[0])
            self.marker2_y = int(self.processor.selected_point_2[1])
        else:
            self.marker2_label.setText("Marker 2: None")

        if self.csv_writer:
            ####
            write_data = [time.time(), self.f_L, self.f_R, self.f_tail, self.marker1_x, self.marker1_y, self.marker2_x, self.marker2_y]
            self.csv_writer.writerow(write_data)  # Change this line to record the desired data
            ####

    def display_image(self, frame, label):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_qt_format)
        label.setPixmap(pixmap)

    def send_command1(self):
        f_chest = self.param1_slider.value()/100*1.5
        f_tail = self.param2_slider.value()/100*2
        f_L = f_chest
        f_R = f_chest
        self.f_L = f_L
        self.f_R = f_R
        self.f_tail = f_tail
        # command = f"COMMAND1 {f_L} {f_R} {f_tail}"
        command = [f_L, f_R, f_tail]
        self.send_serial_command(command)

    def send_command2(self):
        f_chest = self.param1_slider.value()/100*1.5
        f_tail = self.param2_slider.value()/100*2
        f_L = 0.
        f_R = f_chest
        self.f_L = f_L
        self.f_R = f_R
        self.f_tail = f_tail
        command = [f_L, f_R, f_tail]
        self.send_serial_command(command)

    def send_command3(self):
        f_chest = self.param1_slider.value()/100*1.5
        f_tail = self.param2_slider.value()/100*2
        f_L = f_chest
        f_R = 0.
        self.f_L = f_L
        self.f_R = f_R
        self.f_tail = f_tail
        command = [f_L, f_R, f_tail]
        self.send_serial_command(command)

    def send_command4(self):
        f_chest = 0.
        f_tail = 0.
        f_L = f_chest
        f_R = f_chest
        self.f_L = f_L
        self.f_R = f_R
        self.f_tail = f_tail
        command = [f_L, f_R, f_tail]
        self.send_serial_command(command)

    def send_serial_command(self, command):
        if self.serial_port and self.serial_port.is_open:
            input_s = '55 aa 99 11 00 00 d5 ff 00 00 00 d5 ff 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ff fd'
            Amp_L = 40
            Amp_R = 40
            if command[0] >= 0:
                phasel = 'b5 ff' ## 75°相位差
            else:
                phasel = '5a 00'
            if command[1] >= 0:
                phaser = 'b5 ff' ## 75°相位差
            else:
                phaser = '5a 00'
            # phaser = 'b5 ff'
            phase_t = 30
            biasl = str(hex(0))[2:].zfill(2)
            biasr = str(hex(0))[2:].zfill(2)
            Amp_T1 = 30
            Amp_T2 = 30

            f_l = abs(command[0])
            f_r = abs(command[1])
            freq_t1 = command[2]
            freq_t2 = command[2]
            # f_l = 0.5
            # f_r = 0.5
            # freq_t1 = 0.5
            # freq_t2 = 0.5

            input_s = (input_s[0:12] + str(hex(Amp_L))[2:].zfill(2) + input_s[14:15] + str(hex(int(10 * f_l)))[2:].zfill(2) +
                       input_s[17:18] + phasel + input_s[23:24] + biasl + input_s[26:27] + str(hex(Amp_R))[2:].zfill(2) +
                       input_s[29:30] + str(hex(int(10 * f_r)))[2:].zfill(2) + input_s[32:33] + phaser + input_s[38:39] + biasr + input_s[41:45] +
                       str(hex(phase_t))[2:].zfill(2) + input_s[47:48] + str(hex(Amp_T1))[2:].zfill(2) + input_s[50:60] + re.sub(r"(?<=\w)(?=(?:\w\w)+$)", " ", struct.pack('<f', freq_t1).hex()) +
                       input_s[71:84] + str(hex(Amp_T2))[2:].zfill(2) + input_s[86:96] + re.sub(r"(?<=\w)(?=(?:\w\w)+$)", " ", struct.pack('<f', freq_t2).hex()) + input_s[107:])
            print(input_s)
            #55 aa 99 11 28 05 b5 ff 00 28 05 b5 ff 00 00 1e 1e 00 00 00 00 00 00 00 00 00 00 00 1e 00 00 00 00 00 00 00 00 00 00 00 ff fd
            #55 aa 99 11 28 05 b5 ff 00 28 05 b5 ff 00 00 1e 1e 00 00 00 00 00 00 00 00 00 00 00 1e 00 00 00 00 00 00 00 00 00 00 00 ff fd
            input_s = input_s.strip()
            send_list = []
            while input_s != '':
                try:
                    num = int(input_s[0:2], 16)
                except ValueError:
                    return None
                input_s = input_s[2:].strip()
                send_list.append(num)
            input_s = bytes(send_list)
            self.serial_port.write(input_s)
            # self.serial_port.write(command.encode())
            print(command)
        else:
            print("Serial port not open")

    def send_serial_command_backsteping(self, pec_L, pec_R):
        if self.serial_port and self.serial_port.is_open:
            input_s = '55 aa 99 11 00 00 d5 ff 00 00 00 d5 ff 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ff fd'
            Amp_L = pec_L
            Amp_R = pec_R
            phasel = 'c4 ff' ## 60°相位差
            phaser = 'c4 ff' ## 75°相位差

            phase_t = 30
            biasl = str(hex(0))[2:].zfill(2)
            biasr = str(hex(0))[2:].zfill(2)
            Amp_T1 = 30
            Amp_T2 = 30

            f_l = 1
            f_r = 1
            freq_t1 = 0
            freq_t2 =0
            # f_l = 0.5
            # f_r = 0.5
            # freq_t1 = 0.5
            # freq_t2 = 0.5

            input_s = (input_s[0:12] + str(hex(Amp_L))[2:].zfill(2) + input_s[14:15] + str(hex(int(10 * f_l)))[2:].zfill(2) +
                       input_s[17:18] + phasel + input_s[23:24] + biasl + input_s[26:27] + str(hex(Amp_R))[2:].zfill(2) +
                       input_s[29:30] + str(hex(int(10 * f_r)))[2:].zfill(2) + input_s[32:33] + phaser + input_s[38:39] + biasr + input_s[41:45] +
                       str(hex(phase_t))[2:].zfill(2) + input_s[47:48] + str(hex(Amp_T1))[2:].zfill(2) + input_s[50:60] + re.sub(r"(?<=\w)(?=(?:\w\w)+$)", " ", struct.pack('<f', freq_t1).hex()) +
                       input_s[71:84] + str(hex(Amp_T2))[2:].zfill(2) + input_s[86:96] + re.sub(r"(?<=\w)(?=(?:\w\w)+$)", " ", struct.pack('<f', freq_t2).hex()) + input_s[107:])
            print(input_s)
            #55 aa 99 11 28 05 b5 ff 00 28 05 b5 ff 00 00 1e 1e 00 00 00 00 00 00 00 00 00 00 00 1e 00 00 00 00 00 00 00 00 00 00 00 ff fd
            #55 aa 99 11 28 05 b5 ff 00 28 05 b5 ff 00 00 1e 1e 00 00 00 00 00 00 00 00 00 00 00 1e 00 00 00 00 00 00 00 00 00 00 00 ff fd
            input_s = input_s.strip()
            send_list = []
            while input_s != '':
                try:
                    num = int(input_s[0:2], 16)
                except ValueError:
                    return None
                input_s = input_s[2:].strip()
                send_list.append(num)
            input_s = bytes(send_list)
            self.serial_port.write(input_s)
            # self.serial_port.write(command.encode())
            print(pec_L, pec_R)
        else:
            print("Serial port not open")
# 55 aa 99 11 14 05 a6 ff 00 14 05 a6 ff 00 08 1e 1e 00 00 00 00 00 00 3f 00 00 00 00 1e 00 00 00 00 00 00 3f 00 00 00 00 ff fd
# 55 aa 99 11 14 05 c4 ff 00 14 05 c4 ff 00 00 1e 1e 00 00 00 00 00 40 3f 00 00 00 00 1e 00 00 00 00 00 40 3f 00 00 00 00 ff fd
    def update_serial_ports(self):
        ports = list_ports.comports()
        self.serial_port_combobox.clear()
        for port in ports:
            self.serial_port_combobox.addItem(port[0])

    def open_serial_port(self):
        # self.update_serial_ports()
        port_name = self.serial_port_combobox.currentText()
        if port_name:
            try:
                self.serial_port = serial.Serial(port_name, 115200, timeout=1)
                self.serial_status_label.setText("串口状态: 已连接")
            except serial.SerialException as e:
                self.serial_status_label.setText(f"串口状态: 打开失败 ({e})")

    def close_serial_port(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.serial_status_label.setText("串口状态: 未连接")

    def on_serial_port_change(self):
        if self.serial_port and self.serial_port.is_open:
            self.close_serial_port()
        self.serial_status_label.setText("串口状态: 未连接")

    def test_serial_comm(self):
        input_s = '55 aa 99 11 00 00 00 00 00 00 00 00 00 00 03 02 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ff fd'
        # input_s = '55 aa 99 11 14 07 a6 ff 00 14 07 a6 ff 00 00 1e 1e 00 00 00 00 00 80 3f 00 00 00 00 1e 00 00 00 00 00 80 3f 00 00 00 00 ff fd'
        input_s = input_s.strip()
        send_list = []
        while input_s != '':
            try:
                num = int(input_s[0:2], 16)
            except ValueError:
                return None
            input_s = input_s[2:].strip()
            send_list.append(num)
        input_s = bytes(send_list)
        self.serial_port.write(input_s)

    def update_hsv_history_1(self, hsv_value):
        # Append the new HSV value to the history
        self.hsv_history_1.append(hsv_value)

        # If the history size exceeds the maximum size, remove the oldest entry
        if len(self.hsv_history_1) > self.max_history_size:
            self.hsv_history_1.pop(0)

    def update_hsv_history_2(self, hsv_value):
        # Append the new HSV value to the history
        self.hsv_history_2.append(hsv_value)

        # If the history size exceeds the maximum size, remove the oldest entry
        if len(self.hsv_history_2) > self.max_history_size:
            self.hsv_history_2.pop(0)

    def backstepping_task(self):
        if self.processor.SUCCESS_TRACK_1 and self.processor.SUCCESS_TRACK_2:
            
            self.fish_pos_pixel = [self.processor.pixel_x, self.processor.pixel_y]
            self.fish_pos = self.fish_pos_pixel / 515 * 2
            self.fish_heading = self.processor.heading_angle
            self.goal_pos = self.goal_pos_pixel / 515*2 

            velocity_global = np.array(self.processor.V_global) / 515 * 2   
            theta = self.fish_heading
            R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
            # 计算随体速度
            velocity_body = R @ velocity_global
            u, v = velocity_body
            w = self.processor.angle_velocity
            delta_pos = self.goal_pos - self.fish_pos
            r = np.linalg.norm(delta_pos)  # 计算欧几里得距离

            pec_amp_forward, pec_amp_turn = self.RoboDact_Contoller.compute_control_law(u, v, w, theta, r)

            self.amp_foward = pec_amp_forward
            self.amp_turn = pec_amp_turn

            self.amp_L = self.amp_foward - self.amp_turn
            self.amp_R = self.amp_foward + self.amp_turn



class RobotDact:
    def __init__(self):
        self.m = 17.446
        self.J = 1.366
        self.m_ax = 0
        self.m_ay = 5.559


        self.C_dx = 0.1445 + 0.1309 + 0.1070
        self.C_dy = 0.5 + 0.4098 + 0.2273
        self.K_d = 5.76
        self.rho = 1000
        
        # control params
        self.pec_amp_foward = 0
        self.pec_amp_turn = 0
        self.tail_bias = 0
        # control hyper_params
        self.pec_freq = 1.0
        self.tail_freq = 1.0
        self.tail_amp = 60
        self.k_pec1 = 0
        self.k_pec2 = 0
        self.pec_l = 0.2 # unit:m
        self.compute_pec_force_coef()

        # controller params
        self.k1 = 5.9433664191087345
        self.k2 = 2.8158204403815574
        self.kr = 5.9433664191087345
        self.ktheta = 1
    
    def compute_pec_force_coef(self):
        # 参数表
        params = np.array([
            [0.9262, -1.6480,  1.5960,  0.6419,  0.0442, -1.5290],
            [0.1016,  1.5580, -2.0650,  1.4700, -0.1958,  3.0170],
            [0.0535,  0.7496,  1.3190,  0.3941,  0.1191, -0.7917]
        ])
        # 频率 f
        f = self.pec_freq
        # 计算 k1
        k1 = np.sum((params[:, 0] * f**3 + params[:, 1] * f**2 + params[:, 2] * f + params[:, 3]) * params[:, 4])
        # 计算 k2
        k2 = np.sum((params[:, 0] * f**3 + params[:, 1] * f**2 + params[:, 2] * f + params[:, 3]) * params[:, 5])
        self.k_pec1 = k1
        self.k_pec2 = k2
        print("k1:",k1)
        print("k2:",k2)
        return k1, k2
    
    def compute_pec_force(self, amp_foward, amp_turn):
        pec_force_u = 2*(self.k_pec1 * amp_foward)
        pec_force_v = 0
        pec_force_w = 2 * self.k_pec1 * amp_turn
        return pec_force_u, pec_force_v, pec_force_w


    def fup(self, u, v, w):
        term1 = (self.m - self.m_ay)/(self.m-self.m_ax) * v * w
        term2 = -0.5 * self.rho * self.C_dx * u**2 / (self.m-self.m_ax)
        term3 = 2* self.k_pec2 /(self.m-self.m_ax)
        return term1+term2+term3
    
    def fvp(self, u, v, w):
        term1 = -(self.m - self.m_ax)/(self.m-self.m_ay) * v * w
        term2 = -0.5 * self.rho * self.C_dy * v**2 / (self.m-self.m_ay)
        return term1+term2
    
    def fwp(self, u, v, w):
        term1 = (self.m_ay - self.m_ax)/(self.J) * u * v
        term2 = -self.K_d*np.sign(w)*w**2 /(self.J)
        return term1+term2

    def dynamics(self, u, v, w, pec_amp_foward, pec_amp_turn):
        """Compute the dynamics based on the given system of equations"""
        
        # Compute forces from fu and fv
        f_u = self.fup(u, v, w)
        f_v = self.fvp(u, v, w)
        f_w = self.fwp(u, v, w)

        pec_force_u,pec_force_v, pec_force_w = self.compute_pec_force(pec_amp_foward, pec_amp_turn)
        # Dynamics equations
        u_dot = f_u + pec_force_u / (self.m - self.m_ax) - 2* self.k_pec2 /(self.m-self.m_ax) # 减去重复项
        v_dot = f_v + pec_force_v / (self.m - self.m_ay)
        w_dot = f_w + pec_force_w / (self.J)
        return u_dot, v_dot, w_dot
    
    def kinematics(self, u, v, psi, omega):
        """Compute the kinematics based on the system of equations"""
        
        # Kinematic equations
        x_dot = u * np.cos(psi) - v * np.sin(psi)
        y_dot = u * np.sin(psi) + v * np.cos(psi)
        psi_dot = omega
        
        return x_dot, y_dot, psi_dot
    
    def target_tracking_dynamics(self, x, y, xs, ys, u, v, psi, omega):
        """Compute the dynamics for target tracking (r, θ)"""
        
        # Compute the relative coordinates to the target
        xe = xs - x
        ye = ys - y
        
        # Compute the relative angle φ
        phi = np.arctan2(ye, xe)
        
        # Compute the relative distance r and angle θ
        r = np.sqrt(xe**2 + ye**2)
        theta = psi - phi
        
        # Dynamics equations for r and θ
        r_dot = -u * np.cos(theta) + v * np.sin(theta)
        theta_dot = (u / r) * np.sin(theta) + (v / r) * np.cos(theta) + omega
        
        return r, theta, r_dot, theta_dot
    


    def compute_control_law(self, u, v, w, theta, r):
        
        f_u = self.fup(u, v, w)
        f_v = self.fvp(u, v, w)
        f_w = self.fwp(u, v, w)

        pec_amp_forward = (-f_u*np.cos(theta) + f_v*np.sin(theta) + (u*np.sin(theta)+v*np.cos(theta))*(u/r*np.sin(theta)+v/r*np.cos(theta)+w) \
                           +(self.k1 + self.kr)*(-u*np.cos(theta)+v*np.sin(theta)) + self.k1*self.kr*r) /(self.k_pec1 * np.cos(theta))
        # pec_amp_forward = 30
        
        pec_amp_turn =  (self.k_pec1*np.sin(theta)*pec_amp_forward + f_u*np.sin(theta) + f_v*np.cos(theta) + f_w*r + (self.ktheta+u*np.cos(theta)-v*np.sin(theta))*(u/r*np.sin(theta)+v/r*np.cos(theta)+w) \
                    +w*(-u*np.cos(theta)+v*np.sin(theta))+self.k2*self.ktheta*theta + self.k2*(u*np.sin(theta)+v*np.cos(theta)+w*r) )/ (self.k_pec2 * r)
        
        if pec_amp_forward < 30:
            pec_amp_forward = 30
        if pec_amp_forward > 40:
            pec_amp_forward = 40
        if pec_amp_turn > 20:
            pec_amp_turn = 20
        if pec_amp_turn < -20:
            pec_amp_turn = -20   
        # pec_amp_turn = -abs(pec_amp_turn) * np.sign(theta)
        print("pec_amp_foward:", pec_amp_forward)
        print("pec_amp_turn:", pec_amp_turn)
        return pec_amp_forward, pec_amp_turn

    def update_position(self, x, y, psi, x_dot, y_dot, psi_dot, dt):
        """Update position based on kinematics"""
        x_new = x + x_dot * dt
        y_new = y + y_dot * dt
        psi_new = psi + psi_dot * dt
        return x_new, y_new, psi_new
    
    
    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
