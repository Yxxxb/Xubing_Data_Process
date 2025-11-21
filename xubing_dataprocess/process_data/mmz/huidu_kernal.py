import numpy as np
from PIL import Image

def apply_sharpen_kernel(heatmap_array, center_value, edge_value):
    """应用    对热力图数组组应用锐化卷积核
    
    参数:
        heatmap_array: 归一化后的热力图数组
        center_value: 锐化卷积核的中间值
        
    返回:
        锐化处理后的数组
    """
    # 定义锐化卷积核，中间值为自定义输入
    kernel = np.array([
        [0, -1, 0],
        [-1, center_value, -1],
        [0, -1, 0]
    ])

    kernel = np.array([
        [0, -0.08392 , 0],
        [-0.08392 ,  0.63661, -0.08392 ],
        [0, -0.08392 , 0]
    ])

    kernel = np.array([
        [0, -0.08392 , 0],
        [-0.08392 ,  0.63661, -0.08392 ],
        [0, -0.08392 , 0]
    ]) * 5

    kernel = np.array([
        [0, -0.08392 , -0.08392 , -0.08392 , 0],
        [-0.08392 ,  0.63661, 0.63661, 0.63661, -0.08392],
        [-0.08392 ,  0.63661, 0.63661, 0.63661, -0.08392],
        [-0.08392 ,  0.63661, 0.63661, 0.63661, -0.08392],
        [0, -0.08392, -0.08392, -0.08392, 0],
    ]) * 1

    kernel = np.array([
        [0, edge_value, edge_value, edge_value, 0],
        [edge_value, center_value, center_value, center_value, edge_value],
        [edge_value, center_value, center_value, center_value, edge_value],
        [edge_value, center_value, center_value, center_value, edge_value],
        [0, edge_value, edge_value, edge_value, 0],
    ])
    
    # 获取数组尺寸并进行边缘填充
    height, width = heatmap_array.shape
    padded_array = np.pad(heatmap_array, pad_width=2, mode='edge')
    
    # 初始化输出数组
    output_array = np.zeros_like(heatmap_array, dtype=np.float32)
    
    # 应用卷积操作
    for i in range(height):
        for j in range(width):
            # 提取3x3区域并应用卷积
            region = padded_array[i:i+5, j:j+5]
            output_array[i, j] = np.sum(region * kernel)
    
    # 裁剪超出0-255的数值并转换为整数
    output_array = np.clip(output_array, 0, 255)
    return output_array.astype(np.uint8)

def heatmap_to_sharpened_gray(txt_path):
    """
    读取热力图数据，应用不同中间值的锐化卷积核，保存所有结果
    
    参数:
        txt_path: 热力图数据txt文件路径
        center_values: 卷积核中间值的列表，默认包含3到10
    """
    # 读取txt文件
    with open(txt_path, 'r') as f:
        heatmap_data = [list(map(float, line.strip().split())) for line in f]
    
    # 转换为numpy数组
    heatmap_array = np.array(heatmap_data, dtype=np.float32)
    
    # 归一化到0-255范围
    min_val = np.min(heatmap_array)
    max_val = np.max(heatmap_array)
    if max_val > min_val:
        normalized = (heatmap_array - min_val) / (max_val - min_val) * 255.0
    else:
        normalized = np.zeros_like(heatmap_array)
    
    # 对每个中间值生成并保存锐化结果
    center_values = [ 0.63661, 0.54769, 0.66709, 0.4973, 0.5406, 0.53292 , 0.53292 ]
    edge_values = [ -0.08392, -0.10569, -0.13897, -0.08829, -0.14445, -0.06142 , -0.53292 ]
    center_values = [ 0.06125, 0.12458, 0.15782, 0.18647, 0.21664, 0.24538, 0.27149, 0.30583, 0.33285, 0.36322, 0.39993, 0.4327, 0.46475, 0.4973, 0.51997, 0.54769, 0.57227, 0.63661, 0.66709 ]
    edge_values = [ -0.03527, -0.03527, -0.03527, -0.03527, -0.03527, -0.03527, -0.03527, -0.03527, -0.03527, -0.03527, -0.03527, -0.03527, -0.03527, -0.03527, -0.03527, -0.03527, -0.03527, -0.03527, -0.03527 ]
    # center_values = [0.41814, 0.63661, 0.48882, 0.43767, 0.57227, 0.48016, 0.51997, 0.80224, 0.80742]
    # edge_values = [-0.02233, -0.02233, -0.02233, -0.02233, -0.02233, -0.02233, -0.02233, -0.02233, -0.02233]
    for center, edge in zip(center_values, edge_values):
        # 应用锐化卷积
        sharpened_array = apply_sharpen_kernel(normalized, center, edge)
        
        # 创建图像并生成保存路径
        sharpened_image = Image.fromarray(sharpened_array, mode='L')
        output_path = f"/mnt/cephfs/xubingye/tsp/ft_local/0811/heatmap_sharpen_center_{center}_edge_{edge}.png"
        
        # 保存图像
        sharpened_image.save(output_path)
        print(f"锐化结果（中间值={center}）已保存至: {output_path}")
    
    return True

# 使用示例
if __name__ == "__main__":
    # 输入txt文件路径（保持不变）
    txt_file_path = "/mnt/cephfs/xubingye/tsp/ft_local/532nm.txt"
    
    # 可自定义需要测试的中间值列表，这里使用默认的3-10
    heatmap_to_sharpened_gray(txt_file_path)
