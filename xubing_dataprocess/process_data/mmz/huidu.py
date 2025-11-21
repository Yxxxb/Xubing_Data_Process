import numpy as np
from PIL import Image

kernels = {
    # 锐化卷积核（增强边缘）
    'sharpen': np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]),
}

def heatmap_to_gray(txt_path, output_path=None):
    """
    将热力图txt数据转换为灰度图
    
    参数:
        txt_path: 热力图数据txt文件路径
        output_path: 输出灰度图路径，若为None则不保存仅返回图像对象
    """
    # 读取txt文件
    with open(txt_path, 'r') as f:
        # 读取每一行，转换为浮点数列表
        heatmap_data = []
        for line in f:
            # 去除空白字符并按空格分割
            row = list(map(float, line.strip().split()))
            heatmap_data.append(row)
    
    # 转换为numpy数组
    heatmap_array = np.array(heatmap_data, dtype=np.float32)
    
    # 归一化到0-255范围（灰度图范围）
    min_val = np.min(heatmap_array)
    max_val = np.max(heatmap_array)
    
    # 防止除零错误
    if max_val > min_val:
        normalized = (heatmap_array - min_val) / (max_val - min_val) * 255.0
    else:
        normalized = np.zeros_like(heatmap_array)
    
    # 转换为8位无符号整数
    gray_array = normalized.astype(np.uint8)
    
    # 创建PIL图像
    gray_image = Image.fromarray(gray_array, mode='L')  # 'L'表示灰度模式
    
    # 保存图像（如果指定了输出路径）
    if output_path:
        gray_image.save(output_path)
        print(f"灰度图已保存至: {output_path}")
    
    return gray_image

# 使用示例
if __name__ == "__main__":
    # 替换为你的txt文件路径
    txt_file_path = "/mnt/cephfs/xubingye/tsp/ft_local/532nm.txt"
    # 输出图像路径
    output_image_path = "/mnt/cephfs/xubingye/tsp/ft_local/heatmap_gray.png"
    
    # 转换并保存
    image = heatmap_to_gray(txt_file_path, output_image_path)
    
    # 显示图像（可选）
    # image.show()
