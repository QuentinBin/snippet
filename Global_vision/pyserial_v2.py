import sys, time
import serial
import serial.tools.list_ports
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSlider,QSizePolicy, QComboBox
from PyQt5.QtWidgets import QLineEdit, QFormLayout, QFrame, QShortcut
from PyQt5.QtCore import Qt, QDateTime, pyqtSignal, QThread, QObject, QTimer, QMutex
from PyQt5.QtGui import QImage, QPixmap
from robodactII import Ui_Form
from cushy_serial import CushySerial
import gxipy as gx
import struct
import re
from queue import Queue

import cv2
import numpy as np
import csv
import os
from sklearn.cluster import KMeans
from datetime import datetime

# 相机超参数
K = np.array([[515.8139513744953, 0.0, 1196.667955373116], [0.0, 514.3908924982283, 1037.0549098545412], [0.0, 0.0, 1.0]])
D = np.array([[0.007190500081416133], [-0.004884565917566122], [0.006158251646770605], [-0.003034926245524286]])


class Pyqt5_Serial(QtWidgets.QWidget, Ui_Form):

    def __init__(self):
        super(Pyqt5_Serial, self).__init__()
        self.setupUi(self)
        self.init()
        self.setWindowTitle("Robot Fish")
        self.port_check()

        self.ser = CushySerial() #异步串口
        self._send_mutex = QMutex() #串口发送进程锁 #防止发送错位
        self._mutex = QMutex() #主进程锁
        self._rec_data = [] #储存接收到的一帧数据
        self._rec_data_buff = [] #储存缓冲
        self._send_data_buff = [] #发送缓冲
        self._time_marker = time.time()
        self._rec_cnt = 0 #用于计算收发帧数
        self._wait_cnt = 0 #用于等待收的次数

        # 接收数据和发送数据数目置零
        self.data_num_received = 0
        self.data_num_sended = 0
        self.box_s2_1.setText(str(self.data_num_received))
        self.box_s2_2.setText(str(self.data_num_sended))

        
        self.fname = ""        #['55','AA','99','11','90','00','00','00','00','00','00','00','00','91','00','00','93','00','00','00','00','FF','FD']
        self.data_Send =       '55 AA 99 11 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 FF FD'
        self.data_CPG_Stop =   '55 AA 99 11 90 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 FF FD' # ['55','AA','99','11','90','00','00','00','00','00','00','00','00','00','00','00','00','00','00','00','00','FF','FD']    
        self.data_CPG_W =      '55 AA 99 11 90 28 28 14 14 c4 c4 00 00 00 00 00 93 05 00 00 00 FF FD'
        self.data_CPG_D =      '55 AA 99 11 90 28 28 1e 14 c4 5a 00 00 00 00 00 93 05 00 78 06 FF FD'
        self.data_CPG_S =      '55 AA 99 11 90 28 28 14 14 5a 5a 00 00 00 00 00 93 05 00 00 00 FF FD'
        self.data_CPG_A =      '55 AA 99 11 90 28 28 14 14 5a a6 00 00 00 00 00 93 05 00 88 06 FF FD'
        self.data_CPG_I =      '55 AA 99 11 90 1e 1e 0a 0a 00 00 1e 1e 00 00 00 93 05 00 00 00 FF FD'   
        self.data_CPG_K =      '55 AA 99 11 90 1e 1e 0a 0a 00 00 e2 e2 00 00 00 93 05 00 00 00 FF FD'
        self.data_CPG_FlapL =  '55 AA 99 11 90 28 00 14 00 00 00 00 00 00 00 00 93 00 00 00 06 FF FD'
        self.data_CPG_FlapR =  '55 AA 99 11 90 00 28 00 14 00 00 00 00 00 00 00 93 00 00 00 06 FF FD'
        self.data_Fold =       '55 AA 99 11 00 00 00 00 00 00 00 00 00 91 01 68 00 00 00 00 00 FF FD'  # 1是收鳍 2是开鳍
        self.data_UnFold =     '55 AA 99 11 00 00 00 00 00 00 00 00 00 91 00 00 00 00 00 00 00 FF FD'  

        
        self.data_Tail_Stop =   '55 AA 99 11 90 00 00 00 00 00 00 00 00 00 00 00 93 00 00 00 00 FF FD'
        self.data_Tail_J =      '55 AA 99 11 90 00 00 0a 0a a6 a6 00 00 00 00 00 93 14 64 00 00 FF FD'
        self.data_Tail_N =      '55 AA 99 11 00 00 00 00 00 00 00 00 00 00 00 00 93 00 00 00 00 FF FD'
        self.data_Tail_M =      '55 AA 99 11 00 00 00 00 00 00 00 00 00 00 00 00 93 00 00 00 00 FF FD'
        self.data_Reset =       '55 AA 99 11 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 01 FF FD'
        self.data_MotorA5 =     '55 AA 99 11 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 01 02 FF FD'
        self.data_MotorD5 =     '55 AA 99 11 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 02 02 FF FD'
        self.data_MotorA1 =     '55 AA 99 11 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 03 02 FF FD'
        self.data_MotorD1 =     '55 AA 99 11 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 04 02 FF FD'
        self.data_FoldReset =   '55 AA 99 11 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 03 FF FD'
        self.data_save_data =   '55 AA 99 11 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 04 FF FD'
        self.data_stop_save_data =   '55 AA 99 11 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 05 FF FD'
        
      
    def init(self):
        # 串口检测按钮
        self.box_s1_1.clicked.connect(self.port_check)
        # 串口信息显示
        self.box_s1_2.currentTextChanged.connect(self.port_imf)
        # 打开串口按钮
        self.box_s1_7.clicked.connect(self.port_open)
        # 关闭串口按钮
        self.box_s1_8.clicked.connect(self.port_close)

        # 清除接收窗口
        self.box_s3_2.clicked.connect(self.receive_data_clear)
        # 清除发送窗口
        self.box_s4_2.clicked.connect(self.send_data_clear)
        # 发送数据按钮
        self.box_s4_4.clicked.connect(self.send_data_clear)

        self.box_s5_1.clicked.connect(self.send_cpg_stop)
        self.box_s5_2.clicked.connect(self.send_fold)
        self.box_s5_3.clicked.connect(self.send_unfold)
        self.box_s5_4.clicked.connect(self.send_cpg_w)
        self.box_s5_5.clicked.connect(self.send_cpg_s)
        self.box_s5_6.clicked.connect(self.send_cpg_a)
        self.box_s5_7.clicked.connect(self.send_cpg_d)
        self.box_s5_15.clicked.connect(self.send_cpg_i)
        self.box_s5_16.clicked.connect(self.send_cpg_k)
        self.box_s5_17.clicked.connect(self.send_cpg_flapl)
        self.box_s5_18.clicked.connect(self.send_cpg_flapr)

        self.box_s6_1.clicked.connect(self.send_tail_stop)
        self.box_s6_2.clicked.connect(self.send_tail_j)
        self.box_s6_3.clicked.connect(self.send_tail_n)
        self.box_s6_4.clicked.connect(self.send_tail_m)

        self.box_s7_1.clicked.connect(self.SendKey_Reset)
        self.box_s7_2.clicked.connect(self.SendKey_FoldReset)
        self.box_s7_3.clicked.connect(self.SendKey_MotorA5)
        self.box_s7_4.clicked.connect(self.SendKey_MotorD5)
        self.box_s7_5.clicked.connect(self.SendKey_MotorA1)
        self.box_s7_6.clicked.connect(self.SendKey_MotorD1)
        self.box_s7_7.clicked.connect(self.send_all)
        self.box_s7_8.clicked.connect(self.save_data)
        self.box_s7_9.clicked.connect(self.stop_save_data)
        
        self.box_s10_1.clicked.connect(self.start_stop_camera)
        self.box_s10_2.clicked.connect(self.select_roi)
        self.box_s10_3.clicked.connect(self.start_stop_prosessing)
        self.box_s10_4.clicked.connect(self.plot_location)
        self.box_s10_5.clicked.connect(self.plot_path)
        self.box_s10_6.clicked.connect(self.start_saving_data)
        self.box_s10_7.clicked.connect(self.stop_saving_data)
        self.box_s10_8.clicked.connect(self.start_recording)
        self.box_s10_9.clicked.connect(self.stop_recording)
        

        # 初始化摄像头
        self.frame = None # 摄像头帧
        self.frame_queue = Queue(maxsize=10) # 帧队列
        self.frame_risized = None # 缩放后的帧
        self.camera_capture = CameraCapture() # 摄像头捕获线程
        self.camera_capture.frame_captured.connect(self.update_frame) # 摄像头捕获信号
        self.processor = ImageProcessor(self.frame_queue, None) # 图像处理线程
        self.processor.frame_processed.connect(self.display_processed_image) # 图像处理信号

        # 初始化参数
        self.is_start_camera = False # 是否开始摄像头
        self.is_processing = False # 是否正在处理
        self.is_plot_location = False # 是否正在绘制位置
        self.is_plot_path = False # 是否正在绘制路径
        self.is_saving_data = False # 是否正在保存数据
        self.data_file = None # 数据文件
        self.csv_writer = None # CSV 写入器
        self.data_file_name = "" # 数据文件名
        self.is_recording = False # 是否正在录制
        self.video_writer = None # 视频写入器
        self.video_file_name = "" # 视频文件名
        self.track_roi = None # 追踪 ROI

        # 初始化机器人运动信息
        self.track_points = None # 追踪点
        self.position = None # 位置
        self.angle = None # 方向
        self.velocity = None # 速度
        self.angular_velocity = None
        self.track_points_list = [] # 追踪点记录
    
    def start_recording(self):
        if self.is_processing and not self.is_recording and self.frame is not None:
            self.is_recording = True
            self.box_s10_8.setEnabled(False)
            if not os.path.exists("./video_data"):
                os.makedirs("./video_data")
            self.video_file_name = "./video_data/" + QDateTime.currentDateTime().toString("hh_mm_ss") + ".mp4"
            self.video_writer = cv2.VideoWriter(self.video_file_name, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.frame.shape[1], self.frame.shape[0]))
            self.processor.video_writer = self.video_writer

    def stop_recording(self):
        self.is_recording = False
        self.box_s10_8.setEnabled(True)
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.processor.video_writer = self.video_writer

    def start_saving_data(self):
        if not self.is_saving_data:
            self.is_saving_data = True
            current_time = QDateTime.currentDateTime().toString("hh_mm_ss")
            if not os.path.exists("./data"):
                os.makedirs("./data")
            self.data_file_name = "./data/" + current_time + ".csv"
            self.csv_file = open(f"./data/{current_time}.csv", mode='w', newline='')
            self.data_file_label.setText(f"正在记录数据文件: {self.data_file_name}")
            self.box_s10_6.setEnabled(False)
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["Time", "Track Points", "Position", "Angle", "Velocity", "Angular Velocity"])
            # self.csv_writer.writerow(["Mean Pixel Value"])  # Change this header as needed

    def stop_saving_data(self):
        if self.is_saving_data:
            self.is_saving_data = False
            if self.csv_writer:
                self.csv_writer = None
                self.csv_file.close()

            self.box_s10_6.setEnabled(True)

            self.data_file_label.setText("没有正在记录的数据文件")

    def plot_location(self):
        if not self.is_plot_location:
            self.is_plot_location = True
            self.processor.draw_location_flag = self.is_plot_location
        else:
            self.is_plot_location = False
            self.processor.draw_location_flag = self.is_plot_location
        

    def plot_path(self):
        self.draw_past_points_num = int(self.box_s10_10.value())
        self.processor.draw_past_points_num = self.draw_past_points_num
        print(self.draw_past_points_num)
        if not self.is_plot_path:
            self.is_plot_path = True
            self.processor.draw_path_flag = self.is_plot_path  
        else:
            self.is_plot_path = False
            self.processor.draw_path_flag = self.is_plot_path  
        


    def update_frame(self, frame):
        self.frame = frame
        
        if self.is_processing:
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
                
        self.frame_risized = self.resize_cap(self.frame, self.cap_qlabel)
        # 显示帧
        # 转换为 Qt 格式
        frame = cv2.cvtColor(self.frame_risized, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.cap_qlabel.setPixmap(QPixmap.fromImage(qt_image))

    def display_processed_image(self, frame):
        frame_risized = self.resize_cap(frame, self.cap_qlabel_2)
        # 显示帧
        # 转换为 Qt 格式
        frame = cv2.cvtColor(frame_risized, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.cap_qlabel_2.setPixmap(QPixmap.fromImage(qt_image))

        if self.csv_writer:
            ####
            write_data = [time.time(), self.processor.track_points, self.processor.position, self.processor.angle, self.processor.velocity, self.processor.angular_velocity]
            self.csv_writer.writerow(write_data)  # Change this line to record the desired data
            ####
    
    def resize_cap(self, frame, cap_qlabel):
        # 获取 QLabel 的大小
        label_width = cap_qlabel.width()
        label_height = cap_qlabel.height()
        # 计算缩放比例，保持纵横比
        frame_height, frame_width = frame.shape[:2]
        scale = min(label_width / frame_width, label_height / frame_height)
        new_size = (int(frame_width * scale), int(frame_height * scale))
        # 缩放帧到 QLabel 尺寸
        frame_risized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        return frame_risized
    

    def select_roi(self):
        """ 选择 ROI 区域 """
        if self.is_start_camera and self.frame is not None:
            self.track_roi = cv2.selectROI("选择ROI", self.frame, fromCenter=False, showCrosshair=True)
            self.processor.update_roi(self.track_roi)
            cv2.destroyWindow("选择ROI")

    def start_stop_camera(self):
        if not self.is_start_camera:
            self.camera_capture.start_capture()
            self.is_start_camera = True
            self.fps = self.camera_capture.cap.get(cv2.CAP_PROP_FPS)
            # self.fps = 500
        else:
            self.camera_capture.stop_capture()
            self.frame = None
            self.frame_risized = None
            self.cap_qlabel.clear()

            # 重置参数
            self.is_start_camera = False # 是否开始摄像头
            self.is_recording = False # 是否正在录制
            self.is_processing = False # 是否正在处理
            self.is_plot_location = False # 是否正在绘制位置
            self.is_plot_path = False # 是否正在绘制路径
            self.is_saving_data = False # 是否正在保存数据
            self.track_roi = None # 追踪 ROI
            

    def start_stop_prosessing(self):
        if not self.is_processing:
            self.is_processing = True
            self.processor.start_processing()
        else:
            self.is_processing = False
            self.processor.stop_processing()


    # 串口检测
    def port_check(self):
        # 检测所有存在的串口，将信息存储在字典中
        self.Com_Dict = {}
        port_list = list(serial.tools.list_ports.comports())
        self.box_s1_2.clear()
        for port in port_list:
            self.Com_Dict["%s" % port[0]] = "%s" % port[1]
            self.box_s1_2.addItem(port[0])
        if len(self.Com_Dict) == 0:
            self.state_label.setText(" 无串口")
        else:
            pass

    # 串口信息
    def port_imf(self):
        # 显示选定的串口的详细信息
        imf_s = self.box_s1_2.currentText()
        if imf_s != "":
            self.state_label.setText(
                self.Com_Dict[self.box_s1_2.currentText()])
        else:
            pass

    # 打开串口
    def port_open(self):
        self._mutex.lock()
        self.ser.port = self.box_s1_2.currentText()
        self.ser.baudrate = int(self.box_s1_3.currentText())
        self.ser.bytesize = int(self.box_s1_4.currentText())
        self.ser.stopbits = int(self.box_s1_6.currentText())
        self.ser.parity = self.box_s1_5.currentText()
        
        try:
            self.ser.open()
        except BaseException:
            QMessageBox.critical(self, "Port Error", "此串口不能被打开！")
            return None

        if self.ser.isOpen():
            @self.ser.on_message()
            def handle_serial_message(msg: bytes):
                self._rec_data_buff+=msg
                if(len(self._rec_data_buff)>=69):
                    print(len(self._rec_data_buff))
                    self._rec_data = self._rec_data_buff
                    self._rec_cnt += 1
                    self._send_data_buff = []
                    self._rec_data_buff = []
                    if(self._rec_cnt == 100):
                        print(time.time()-self._time_marker)
                        self._time_marker = time.time()
                        self._rec_cnt = 0
                        
            self.box_s1_7.setEnabled(False)
            self.box_s1_8.setEnabled(True)
            self.groupBox_s1.setTitle("串口状态（已开启）")
        else:
            pass
        self._mutex.unlock()

    # 关闭串口
    def port_close(self):
        self._mutex.lock()
        try:
            self.ser.close()
        except BaseException:
            pass
        self.box_s1_7.setEnabled(True)
        self.box_s1_8.setEnabled(False)

        self.groupBox_s1.setTitle("串口状态（已关闭）")
        self._mutex.unlock()

    # 发送数据
    def data_send_empty(self):
        self._mutex.lock()
        self._send_mutex.lock()
        if self.ser.isOpen() and len(self._send_data_buff)==0: #只有接收到rec后才重新清0
            self._wait_cnt = 0
            input_s = self.data_Send
            input_s = input_s.strip()
            self.box_s4_1.setText(input_s)
            send_list = []
            while input_s != '':
                num = int(input_s[0:2], 16)
                input_s = input_s[2:].strip()
                send_list.append(num)
            input_s = bytes(send_list)
            self.ser.send(input_s)
            self._send_data_buff = input_s
            self.data_num_sended += len(input_s)
            self.box_s2_2.setText(str(self.data_num_sended))
        elif len(self._send_data_buff)!=0:
            self._wait_cnt += 1
            if(self._wait_cnt > 100): #200ms
                self._wait_cnt = 0
                input_s = self.data_Send
                input_s = input_s.strip()
                self.box_s4_1.setText(input_s)
                send_list = []
                while input_s != '':
                    num = int(input_s[0:2], 16)
                    input_s = input_s[2:].strip()
                    send_list.append(num)
                input_s = bytes(send_list)
                self.ser.write(input_s)
                self._send_data_buff = input_s
                self.data_num_sended += len(input_s)
                self.box_s2_2.setText(str(self.data_num_sended))
        self._send_mutex.unlock()
        self._mutex.unlock()

    def send_data_str(self, str_data):
        self._send_mutex.lock()
        self._send_data_buff = []
        if self.ser.isOpen(): #只有接收到rec后才重新清0
            self._wait_cnt = 0
            input_s = str_data
            input_s = input_s.strip()
            self.box_s4_1.setText(input_s)
            send_list = []
            while input_s != '':
                num = int(input_s[0:2], 16)
                input_s = input_s[2:].strip()
                send_list.append(num)
            input_s = bytes(send_list)
            self.ser.send(input_s)
            self._send_data_buff = input_s
            self.data_num_sended += len(input_s)
            self.box_s2_2.setText(str(self.data_num_sended))
        self._send_mutex.unlock()
    
    def data_init_send(self):
        global send_rece_init
        if(send_rece_init==0):
            self.send_data_alltime()
        else:
           pass
    # 清除显示
    def send_data_clear(self):
        self.box_s4_1.setText("")

    def receive_data_clear(self):
        self.box_s3_1.setText("")

    # 发送停止数据
    def send_cpg_stop(self):
        self._mutex.lock()
        self.send_data_str(self.data_CPG_Stop)
        self._mutex.unlock()

    # 发送停止数据
    def send_fold(self):
        self._mutex.lock()
        chest_angle = str(hex(self.box_s9_17.value()))[2:].zfill(4)
        input_s = ['55','AA','99','11','00','00','00','00','00','00','00','00','00','91',chest_angle[0:2],chest_angle[2:4],'00','00','00','00','00','FF','FD']
        input_s = ' '.join(input_s)
        self.send_data_str(input_s)

        self._mutex.unlock()

    # 发送停止数据
    def send_unfold(self):
        self._mutex.lock()
        self.send_data_str(self.data_UnFold)
        self._mutex.unlock()

     # 发送停止数据
    def send_cpg_w(self):
        self._mutex.lock()
        self.send_data_str(self.data_CPG_W)
        self._mutex.unlock()
         # 发送停止数据
    def send_cpg_a(self):
        self._mutex.lock()
        self.send_data_str(self.data_CPG_A)
        self._mutex.unlock()
         # 发送停止数据
    def send_cpg_s(self):
        self._mutex.lock()
        self.send_data_str(self.data_CPG_S)
        self._mutex.unlock()
         # 发送停止数据
    def send_cpg_d(self):
        self._mutex.lock()
        self.send_data_str(self.data_CPG_D)
        self._mutex.unlock()
         # 发送停止数据
    def send_cpg_i(self):
        self._mutex.lock()
        self.send_data_str(self.data_CPG_I)
        self._mutex.unlock()
         # 发送停止数据
    def send_cpg_k(self):
        self._mutex.lock()
        self.send_data_str(self.data_CPG_K)
        self._mutex.unlock()
    def send_cpg_flapl(self):
        self._mutex.lock()
        self.send_data_str(self.data_CPG_FlapL)
        self._mutex.unlock()
    def send_cpg_flapr(self):
        self._mutex.lock()
        self.send_data_str(self.data_CPG_FlapR)
        self._mutex.unlock()
    # 发送停止数据
    def send_tail_stop(self):
        self._mutex.lock()
        self.send_data_str(self.data_Tail_Stop)
        self._mutex.unlock()

    def send_tail_j(self):
        self._mutex.lock()
        self.send_data_str(self.data_Tail_J)
        self._mutex.unlock()
    
    def send_tail_n(self):
        self._mutex.lock()
        self.send_data_str(self.data_Tail_N)
        self._mutex.unlock()
    
    def send_tail_m(self):
        self._mutex.lock()
        self.send_data_str(self.data_Tail_M)
        self._mutex.unlock()


    def send_all(self):
        self._mutex.lock()

        if self.box_s9_5.value() < 0:
            cpg_biasl = str(struct.pack("i", self.box_s9_5.value()))[4:6]
        else:
            cpg_biasl = str(hex(self.box_s9_5.value()))[2:].zfill(2)

        if self.box_s9_6.value() < 0:
            cpg_biasr = str(struct.pack("i", self.box_s9_6.value()))[4:6]
        else:
            cpg_biasr = str(hex(self.box_s9_6.value()))[2:].zfill(2)

        if self.box_s9_7.value() > 0:
            cpg_phase = str(struct.pack("i", -self.box_s9_7.value()))[4:6]
        else:
            cpg_phase = str(hex(-self.box_s9_7.value()))[2:].zfill(2)
        if self.box_s9_8.value() > 0:
            tail_bias = str(struct.pack("i", -self.box_s9_8.value()))[4:6]
        else:
            tail_bias = str(hex(-self.box_s9_8.value()))[2:].zfill(2)

        cpg_ampl = str(hex(self.box_s9_1.value()))[2:].zfill(2)
        cpg_freql = str(hex(int(self.box_s9_2.value()*10)))[2:].zfill(2)
        cpg_ampr = str(hex(self.box_s9_3.value()))[2:].zfill(2)
        cpg_freqr = str(hex(int(self.box_s9_4.value()*10)))[2:].zfill(2)

        tail_amp = str(hex(self.box_s8_1.value()))[2:].zfill(2)
        tail_freq = str(hex(int(self.box_s8_2.value()*10)))[2:].zfill(2)

        input_s = ['55','AA','99','11','90',cpg_ampl,cpg_ampr,cpg_freql,cpg_freqr,cpg_phase,cpg_phase,cpg_biasl,cpg_biasr,'00','00','00','93',tail_freq,tail_amp,tail_bias,'06','FF','FD']
        input_s = ' '.join(input_s)
        self.send_data_str(input_s)
        self._mutex.unlock()

    def save_data(self):
        # 发送快捷键save_data
        self._mutex.lock()
        self.send_data_str(self.data_save_data)
        self._mutex.unlock()

    def stop_save_data(self):
        # 发送快捷键stop_save_data
        self._mutex.lock()
        self.send_data_str(self.data_stop_save_data)
        self._mutex.unlock()

    # 发送快捷键reset
    def SendKey_Reset(self):
        self._mutex.lock()
        self.send_data_str(self.data_Reset)
        self._mutex.unlock()

    # 发送快捷键胸鳍复位
    def SendKey_FoldReset(self):
        self._mutex.lock()
        self.send_data_str(self.data_FoldReset)
        self._mutex.unlock()

    # 发送快捷键电机偏置+5
    def SendKey_MotorA5(self):
        self._mutex.lock()
        self.send_data_str(self.data_MotorA5)
        self._mutex.unlock()

    # 发送快捷键电机偏置-5
    def SendKey_MotorD5(self):
        self._mutex.lock()
        self.send_data_str(self.data_MotorD5)
        self._mutex.unlock()

    # 发送快捷键电机偏置+1
    def SendKey_MotorA1(self):
        self._mutex.lock()
        self.send_data_str(self.data_MotorA1)
        self._mutex.unlock()

    # 发送快捷键电机偏置-1
    def SendKey_MotorD1(self):
        self._mutex.lock()
        self.send_data_str(self.data_MotorD1)
        self._mutex.unlock()

    def SendKey_SendALL(self):
        self._mutex.lock()

        if self.box_s9_5.value() < 0:
            biasl = str(struct.pack("i", self.box_s9_5.value()))[4:6]
        else:
            biasl = str(hex(self.box_s9_5.value()))[2:].zfill(2)

        if self.box_s9_6.value() < 0:
            biasr = str(struct.pack("i", self.s6__box_6.value()))[4:6]
        else:
            biasr = str(hex(self.box_s9_6.value()))[2:].zfill(2)

        phase = -self.box_s9_7.value()
        if phase == 30:
            phasel = 'e2'
            phaser = 'e2'
        elif phase == 45:
            phasel = 'd3'
            phaser = 'd3'
        elif phase == 60:
            phasel = 'c4'
            phaser = 'c4'
        elif phase == 75:
            phasel = 'b5'
            phaser = 'b5'
        elif phase == 90:
            phasel = 'a6'
            phaser = 'a6'
        elif phase == -30:
            phasel = '1e'
            phaser = '1e'
        elif phase == -45:
            phasel = '2d'
            phaser = '2d'
        elif phase == -60:
            phasel = '3c'
            phaser = '3c'
        elif phase == -75:
            phasel = '4b'
            phaser = '4b'
        elif phase == -90:
            phasel = '5a'
            phaser = '5a'
        else:
            phasel = '00'
            phaser = '00'
        
        self.data_Send = self.data_Send[0]
        input_s = input_s[0:12] + str(hex(self.s6__box_1.value()))[2:].zfill(2) + input_s[14:15] + str(hex(int(10 * self.s6__box_2.value())))[2:].zfill(2) + input_s[17:18] + phasel + input_s[23:24] + biasl + input_s[26:27] + str(hex(self.s6__box_3.value()))[2:].zfill(2) + input_s[29:30] + str(hex(int(10 * self.s6__box_4.value())))[2:].zfill(2) + input_s[32:33] + phaser + input_s[38:39] + biasr + input_s[41:45] + str(hex(self.s5__box_5.value()))[2:].zfill(2) + input_s[47:48] + str(hex(self.s5__box_1.value()))[2:].zfill(2) + input_s[50:60] + re.sub(r"(?<=\w)(?=(?:\w\w)+$)", " ", struct.pack('<f', self.s5__box_2.value()).hex()) + input_s[71:84] + str(hex(self.s5__box_3.value()))[2:].zfill(2) + input_s[86:96] + re.sub(r"(?<=\w)(?=(?:\w\w)+$)", " ", struct.pack('<f', self.s5__box_4.value()).hex()) + input_s[107:]
        self.s3__send_text.setText(input_s)

        send_list = []
        while input_s != '':
            num = int(input_s[0:2], 16)
            input_s = input_s[2:].strip()
            send_list.append(num)
        input_s = bytes(send_list)

        num = self.ser.write(input_s)
        self.data_num_sended += num
        self.lineEdit_2.setText(str(self.data_num_sended))

        self._mutex.unlock()


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
        cam.ExposureTime.set(40000.0)

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

class ImageProcessor(QThread):
    frame_processed = pyqtSignal(np.ndarray)
    def __init__(self, frame_queue, roi, draw_location_flag=False, draw_path_flag=False, draw_past_points_num=120, video_writer=None):
        super().__init__()
        self.frame_queue = frame_queue
        self.running = False
        self.roi = roi
        self.SUCCESS_TRACK_yellow = False
        self.SUCCESS_TRACK_red = False
        self.draw_location_flag = draw_location_flag
        self.draw_path_flag = draw_path_flag
        self.draw_past_points_num = draw_past_points_num
        self.video_writer = video_writer
        
        # 初始化机器人运动信息
        self.track_points = None # 追踪点
        self.position = None # 位置
        self.angle = None # 方向
        self.velocity = None # 速度
        self.angular_velocity = None
        self.timestamp = None # 时间戳
        self.track_points_list = [] # 追踪点记录
        self.position_list = [] # 位置记录
        self.angle_list = [] # 方向记录
        self.timestamp_list = [] # 时间戳记录

    def start_processing(self):
        if not self.running:
            self.running = True
            self.start()

    def stop_processing(self):
        self.running = False
        self.track_points = None # 追踪点
        self.position = None # 位置
        self.angle = None # 方向
        self.velocity = None # 速度
        self.angular_velocity = None
        self.timestamp = None # 时间戳
        self.track_points_list = [] # 追踪点记录
        self.position_list = [] # 位置记录
        self.angle_list = [] # 方向记录
        self.timestamp_list = [] # 时间戳记录
        self.roi = None
        self.wait()
    
    def update_roi(self, roi):
        self.roi = roi

    def run(self):
        while self.running:
            elapsed_time_list = []
            if not self.frame_queue.empty():
                frame = self.frame_queue.get() # Get the latest frame
                start_time = time.time()
                if frame is not None:
                    result= self.process_frame(frame)
                    self.frame_processed.emit(result)
                elapsed_time = time.time() - start_time
                # elapsed_time_list.append(elapsed_time)
                # print("fps:", 1 / np.mean(elapsed_time_list))
            QThread.msleep(10)  # Control processing frame rate

    def process_frame(self, frame):
        if self.roi is not None:
            x, y, w, h = self.roi
            roi_frame = frame[y:y+h, x:x+w]
            centers = self.extract_color_centers(roi_frame, self.angle)

            if centers["yellow"] is not None:
                self.SUCCESS_TRACK_yellow = True
            else:
                self.SUCCESS_TRACK_yellow = False
            if centers["red"] is not None:
                self.SUCCESS_TRACK_red = True
            else:
                self.SUCCESS_TRACK_red = False
            
            if self.SUCCESS_TRACK_yellow and self.SUCCESS_TRACK_red:
                # 计算机器人位置
                new_x = int((centers["red"][0] + centers["yellow"][0]) / 2) + x
                new_y = int((centers["red"][1] + centers["yellow"][1]) / 2) + y
                # 更新ROI
                self.roi = (max(0, new_x - w // 2), max(0, new_y - h // 2), w, h)

                self.track_points = centers
                self.track_points_list.append(self.track_points)
                self.position = [new_x, new_y]
                self.position_list.append(self.position)
                self.angle = np.arctan2(centers["yellow"][1] - centers["red"][1], centers["yellow"][0] - centers["red"][0])
                self.angle_list.append(self.angle)
                self.timestamp = time.time()
                self.timestamp_list.append(self.timestamp)

                # 根据过去60个数值平均计算二维速度、角速度
                if len(self.position_list) > 60:
                    velocities = []
                    angular_velocities = []
                    for i in range(9):
                        time_diff = self.timestamp_list[-1] - self.timestamp_list[-1-i]
                        if time_diff != 0:
                            velocities.append((np.array(self.position_list[-1]) - np.array(self.position_list[-2-i])) / time_diff)
                            angular_velocities.append((self.angle_list[-1] - self.angle_list[-2-i]) / time_diff)
                    if velocities:
                        self.velocity = np.mean(velocities, axis=0)
                    else:
                        self.velocity = np.array([0, 0])
                    if angular_velocities:
                        self.angular_velocity = np.mean(angular_velocities)
                    else:
                        self.angular_velocity = 0
                else:
                    self.velocity = np.array([0, 0])
                    self.angular_velocity = 0
                
                # 创建帧的副本用于绘制
                frame_copy = frame.copy()
                
                if self.draw_location_flag:
                    cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.arrowedLine(frame_copy, (new_x, new_y), (new_x + int(1 * self.velocity[0]), new_y + int(1 * self.velocity[1])), (0, 255, 0), 2)
                
                if self.draw_path_flag:
                    if len(self.position_list) <= self.draw_past_points_num:
                        position_list = self.position_list
                    else:
                        position_list = self.position_list[-self.draw_past_points_num:0]
                    for i in range(len(position_list) - 1):
                        cv2.line(frame_copy, tuple(position_list[i]), tuple(position_list[i+1]), (0, 255, 0), 2)
                
                if self.video_writer:
                    self.video_writer.write(frame_copy)

                # 获取当前时间
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # 设置文本属性
                frame_width = frame_copy.shape[1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_color = (0, 255, 255)  # 黄色
                thickness = 2
                position = (frame_width - 300, 30)  # 右上角 (300 像素宽度)
                cv2.putText(frame_copy, timestamp, position, font, font_scale, font_color, thickness)

                return frame_copy
        if self.video_writer:
            self.video_writer.write(frame)
        return frame


    def extract_color_centers(self, image, angle=None):
        """ 通过连通域分析，仅保留面积最接近红色区域的黄色区域，并进行角度筛选 """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 调整颜色范围
        yellow_lower, yellow_upper = np.array([20, 100, 100]), np.array([30, 255, 255])  # 黄色
        red_lower1, red_upper1 = np.array([0, 150, 150]), np.array([10, 255, 255])  # 红色
        red_lower2, red_upper2 = np.array([170, 150, 150]), np.array([180, 255, 255])  # 红色

        # 颜色掩码
        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        mask_red = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)

        # 形态学去噪
        kernel = np.ones((3, 3), np.uint8)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

        centers = {"yellow": None, "red": None}

        # 计算红色连通区域
        num_labels_r, labels_r, stats_r, centroids_r = cv2.connectedComponentsWithStats(mask_red, connectivity=8)

        if num_labels_r > 1:  # 至少有一个红色区域
            # 选择面积最大的红色区域
            largest_idx = np.argmax(stats_r[1:, cv2.CC_STAT_AREA]) + 1
            red_area = stats_r[largest_idx, cv2.CC_STAT_AREA]
            red_center = tuple(map(int, centroids_r[largest_idx]))  # (x, y)
            centers["red"] = red_center
        else:
            return centers  # 没有红色区域，无法筛选黄色点

        # 计算黄色连通区域
        num_labels_y, labels_y, stats_y, centroids_y = cv2.connectedComponentsWithStats(mask_yellow, connectivity=8)

        if num_labels_y > 1:  # 至少有一个黄色区域
            min_area_diff = float("inf")
            best_yellow_center = None

            for i in range(1, num_labels_y):
                yellow_area = stats_y[i, cv2.CC_STAT_AREA]
                area_diff = abs(yellow_area - red_area)  # 计算面积差距

                yellow_x, yellow_y = int(centroids_y[i][0]), int(centroids_y[i][1])  # 黄色区域中心
                vector = np.array([yellow_x - red_center[0], yellow_y - red_center[1]])  # 方向向量

                # 计算角度筛选
                if angle is not None:
                    angle_rad = np.arctan2(vector[1], vector[0])  # 计算方向角（弧度）
                    diff_angle = (np.degrees(angle_rad) - np.degrees(angle)) % 360

                    if not (0 <= diff_angle <= 90 or 270 <= diff_angle <= 360):  # 只保留前方的黄色点
                        continue

                # 选择面积最接近红色区域的黄色点
                if area_diff < min_area_diff:
                    min_area_diff = area_diff
                    best_yellow_center = (yellow_x, yellow_y)

            centers["yellow"] = best_yellow_center

        return centers


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myshow = Pyqt5_Serial()
    myshow.show()
    print("RoboDactII")
    sys.exit(app.exec_())


