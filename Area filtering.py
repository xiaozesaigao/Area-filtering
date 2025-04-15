import cv2
import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

def remove_small_objects(image_path, min_area=100, output_path="cleaned_image.png"):
    """
    完整流程去除小碎屑(面积<100像素)
    
    参数：
    image_path: 输入图像路径
    min_area: 最小保留面积阈值(像素)
    output_path: 输出图像保存路径
    """
    # -------------------- 1. 读取图像 --------------------
    # 读取原始图像(支持彩色/灰度)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法加载图像：{image_path}")
    
    # -------------------- 2. 灰度转换 --------------------
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # -------------------- 3. 二值化处理 --------------------
    # 使用Otsu自动阈值(适合背景/前景对比明显的情况)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # -------------------- 4. 连通区域分析 --------------------
    # 8邻域连通标记
    labeled = label(binary, connectivity=2, background=0)
    regions = regionprops(labeled)
    
    # -------------------- 5. 创建过滤掩膜 --------------------
    mask = np.zeros_like(binary, dtype=np.uint8)
    for region in regions:
        if region.area >= min_area:
            # 获取区域坐标切片
            minr, minc, maxr, maxc = region.bbox
            # 精确提取区域形状(避免矩形框内包含其他区域)
            mask[minr:maxr, minc:maxc] += (labeled[minr:maxr, minc:maxc] == region.label).astype(np.uint8) * 255
    
    # -------------------- 6. 后处理优化 --------------------
    # (可选)闭运算填补小孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # -------------------- 7. 结果保存与可视化 --------------------
    cv2.imwrite(output_path, cleaned)
    
    # 可视化对比
    plt.figure(figsize=(12,6))
    plt.subplot(121), plt.imshow(binary, cmap='gray'), plt.title('原始二值图像')
    plt.subplot(122), plt.imshow(cleaned, cmap='gray'), plt.title(f'去噪后(最小面积={min_area}像素)')
    plt.show()

if __name__ == "__main__":
    # 使用示例
    input_image = "sample_image.jpg"  # 替换为实际图像路径
    remove_small_objects(input_image, min_area=100)