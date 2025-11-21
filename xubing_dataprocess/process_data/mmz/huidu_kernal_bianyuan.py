import numpy as np
from PIL import Image

def apply_edge_kernel(heatmap_array, sensitivity=1.0):
    """应用边缘检测卷积核对热力图数组进行边缘提取
    
    参数:
        heatmap_array: 归一化后的热力图数组
        sensitivity: 边缘敏感度（值越大，边缘越明显）
        
    返回:
        边缘提取后的数组
    """
    # 定义边缘检测卷积核（增强水平和垂直边缘）
    kernel = np.array([
        [-1, -1, -1],
        [-1,  6, -1],
        [-1, -1, -1]
    ]) * sensitivity  # 敏感度调节因子
    
    # 获取数组尺寸并进行边缘填充
    height, width = heatmap_array.shape
    padded_array = np.pad(heatmap_array, pad_width=1, mode='edge')
    
    # 初始化输出数组
    output_array = np.zeros_like(heatmap_array, dtype=np.float32)
    
    # 应用卷积操作提取边缘
    for i in range(height):
        for j in range(width):
            region = padded_array[i:i+3, j:j+3]
            output_array[i, j] = np.sum(region * kernel)
    
    # 边缘后处理：取绝对值+归一化+对比度增强
    output_array = np.abs(output_array)
    max_edge = np.max(output_array)
    if max_edge > 0:
        output_array = (output_array / max_edge) * 255.0
    output_array = np.clip(output_array, 0, 255)  # 限制范围
    
    return output_array.astype(np.uint8)

def image_to_edge_detection(img_path, sensitivities=[1.0, 1.5, 2.0, 2.5, 3.0]):
    """
    读取灰度图像，应用不同敏感度的边缘检测卷积核，保存所有结果
    
    参数:
        img_path: 输入灰度图像路径
        sensitivities: 边缘敏感度列表，值越大边缘越明显
    """
    # 读取灰度图像并转换为numpy数组
    img = Image.open(img_path).convert('L')  # 确保转为灰度模式
    normalized = np.array(img, dtype=np.float32)  # 图像已在0-255范围，无需重新归一化
    
    # 对每个敏感度生成并保存边缘提取结果
    for s in sensitivities:
        edge_array = apply_edge_kernel(normalized, sensitivity=s)
        edge_image = Image.fromarray(edge_array, mode='L')
        # 生成输出路径（在原文件名后添加敏感度信息）
        output_path = f"/mnt/cephfs/xubingye/tsp/ft_local/heatmap_edge_sensitivity_{s}.png"
        edge_image.save(output_path)
        print(f"边缘提取结果（敏感度={s}）已保存至: {output_path}")
    
    return True

if __name__ == "__main__":
    # 输入灰度图像路径（修改为新路径）
    input_img_path = "/mnt/cephfs/xubingye/tsp/ft_local/ft_local/heatmap_gray.png"
    
    # 运行边缘检测
    image_to_edge_detection(
        input_img_path,
        sensitivities=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    )