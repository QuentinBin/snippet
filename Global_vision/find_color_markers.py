import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_color_centers(image):
    """ 分别提取黄色和红色标志点的中心坐标 """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 颜色范围（HSV）
    yellow_lower, yellow_upper = np.array([20, 100, 100]), np.array([30, 255, 255])
    red_lower1, red_upper1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    red_lower2, red_upper2 = np.array([170, 100, 100]), np.array([180, 255, 255])

    # 颜色掩码
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_red = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)

    # 提取像素点坐标
    yellow_points = np.column_stack(np.where(mask_yellow > 0))  # 黄色 (y, x)
    red_points = np.column_stack(np.where(mask_red > 0))  # 红色 (y, x)

    # KMeans 计算中心点
    centers = {"yellow": None, "red": None}
    
    if len(yellow_points) > 0:
        kmeans_yellow = KMeans(n_clusters=1, n_init=10)
        kmeans_yellow.fit(yellow_points)
        centers["yellow"] = tuple(map(int, kmeans_yellow.cluster_centers_[0][::-1]))  # (x, y)

    if len(red_points) > 0:
        kmeans_red = KMeans(n_clusters=1, n_init=10)
        kmeans_red.fit(red_points)
        centers["red"] = tuple(map(int, kmeans_red.cluster_centers_[0][::-1]))  # (x, y)

    return centers


# 示例用法
image_path = 'Global_vision/test_pic.png'
output_path = 'Global_vision/output_image.jpg'

# 读取图片
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"无法读取图片: {image_path}")

# 选择裁剪区域
roi = cv2.selectROI("选择裁剪区域", image, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("选择裁剪区域")

# 裁剪图像
x, y, w, h = roi
cropped_image = image[y:y+h, x:x+w]
# 对裁剪图像使用自适应双边滤波
cropped_image_b = cv2.bilateralFilter(cropped_image, 9, 75, 75)

# k means聚类算法提取黄色区域
centers = extract_color_centers(cropped_image_b)
cv2.circle(cropped_image, centers['yellow'], 5, (0, 255, 0), -1) # 绿色
cv2.circle(cropped_image, centers['red'], 5, (0, 0, 255), -1) # 红色


# 展现裁剪图像滤波前后对比
cv2.imshow("原图", cropped_image)
cv2.waitKey(0)
# cv2.imshow("滤波后", cropped_image_b)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite(output_path, cropped_image)
# # 在原图上绘制裁剪区域
# cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


