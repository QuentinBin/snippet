import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_color_centers(image, n_clusters=2):
    """
    使用KMeans提取黄色和红色色块的中心坐标。
    
    :param image: 输入图像（BGR格式）
    :param n_clusters: KMeans聚类的簇数
    :return: 黄色和红色色块的中心点坐标 [(x1, y1), (x2, y2), ...]
    """
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 颜色范围（HSV）
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 100, 100])
    red_upper2 = np.array([180, 255, 255])

    # 颜色掩码
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = mask_red1 | mask_red2

    # 合并黄色和红色区域
    mask = mask_yellow | mask_red

    # 获取所有符合颜色的像素点坐标
    points = np.column_stack(np.where(mask > 0))  # (y, x) 坐标

    # 进行KMeans聚类
    if len(points) >= n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(points)
        centers = kmeans.cluster_centers_[:, ::-1]  # 转换为 (x, y)
        centers = [tuple(map(int, c)) for c in centers]
    else:
        centers = []

    return centers


# 处理视频
video_path = "Global_vision\Real-Time Camera with Dual HSV Processing and Sliders 2025-03-09 00-10-00.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"无法打开视频文件: {video_path}")

# 读取第一帧
ret, frame = cap.read()
if not ret:
    raise RuntimeError("无法读取视频第一帧")

# 手动选择初始ROI
roi = cv2.selectROI("选择初始裁剪区域", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("选择初始裁剪区域")

# 解析ROI坐标
x, y, w, h = roi

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 视频结束

    # 提取ROI
    roi_frame = frame[y:y+h, x:x+w]
    roi_frame_b = cv2.bilateralFilter(roi_frame, 9, 75, 75)  # 自适应双边滤波

    # 提取颜色中心
    centers = extract_color_centers(roi_frame_b, n_clusters=2)

    if len(centers) == 2:
        # 在ROI图像中标记点
        cv2.circle(roi_frame, centers[0], 5, (0, 255, 0), -1)  # 绿色
        cv2.circle(roi_frame, centers[1], 5, (0, 0, 255), -1)  # 红色
        
        # 计算新的ROI中心
        new_x = int((centers[0][0] + centers[1][0]) / 2) + x
        new_y = int((centers[0][1] + centers[1][1]) / 2) + y

        # 更新ROI位置（保持宽高不变）
        x = max(0, new_x - w // 2)
        y = max(0, new_y - h // 2)

    # 显示结果
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("视频跟踪", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # 按 "ESC" 退出
        break

cap.release()
cv2.destroyAllWindows()
