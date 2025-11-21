import numpy as np
from PIL import Image

def apply_thick_bright_edge_kernel(heatmap_array, center, edge):
    """应用增强边缘检测卷积核，使边缘更粗更白
    
    参数:
        heatmap_array: 灰度图像数组
        sensitivity: 边缘敏感度（值越大，边缘越明显）
        edge_thickness: 边缘粗细（1-3，值越大边缘越粗）
        
    返回:
        增强处理后的边缘数组
    """
    # 根据边缘粗细选择不同尺寸的卷积核
    # if edge_thickness == 1:
    #     # 3x3核（基础边缘）
    #     kernel = np.array([
    #         [edge, edge, edge],
    #         [edge,  8, edge],
    #         [edge, edge, edge]
    #     ])
    # elif edge_thickness == 2:
    #     # 5x5核（中等粗细边缘）
    #     kernel = np.array([
    #         [edge, edge, edge, edge, edge],
    #         [edge, edge, edge, edge, edge],
    #         [edge, edge, 24, edge, edge],
    #         [edge, edge, edge, edge, edge],
    #         [edge, edge, edge, edge, edge]
    #     ])
    # else:
    #     # 7x7核（粗边缘）
    #     kernel = np.array([
    #         [edge, edge, edge, edge, edge, edge, edge],
    #         [edge, edge, edge, edge, edge, edge, edge],
    #         [edge, edge, edge, edge, edge, edge, edge],
    #         [edge, edge, edge, 48, edge, edge, edge],
    #         [edge, edge, edge, edge, edge, edge, edge],
    #         [edge, edge, edge, edge, edge, edge, edge],
    #         [edge, edge, edge, edge, edge, edge, edge]
    #     ])
    
    # # 应用敏感度缩放
    # kernel = kernel * sensitivity

    kernel = np.array([
        [edge, edge, edge, edge, edge],
        [edge, edge, edge, edge, edge],
        [edge, edge, center, edge, edge],
        [edge, edge, edge, edge, edge],
        [edge, edge, edge, edge, edge]
    ])
    
    # 根据核大小计算填充宽度
    pad_width = kernel.shape[0] // 2
    padded_array = np.pad(heatmap_array, pad_width=pad_width, mode='edge')
    
    # 获取图像尺寸并初始化输出数组
    height, width = heatmap_array.shape
    output_array = np.zeros_like(heatmap_array, dtype=np.float32)
    
    # 应用卷积操作
    kernel_size = kernel.shape[0]
    for i in range(height):
        for j in range(width):
            region = padded_array[i:i+kernel_size, j:j+kernel_size]
            output_array[i, j] = np.sum(region * kernel)
    
    # 边缘后处理（增强亮度和对比度）
    output_array = np.abs(output_array)
    
    # 提升亮度：增加基础亮度并限制最大值
    max_edge = np.max(output_array)
    if max_edge > 0:
        # 归一化到0-280（超过255部分会被截断，增强亮度）
        output_array = (output_array / max_edge) * 280.0
    
    # 高对比度处理：暗部更暗，亮部更亮
    # 阈值以下的像素设为0（更暗），以上的保持高亮度
    threshold = 30  # 可调整阈值控制边缘明显程度
    output_array[output_array < threshold] = 0
    
    # 限制在0-255范围并转换为整数
    output_array = np.clip(output_array, 0, 255)
    
    return output_array.astype(np.uint8)

def image_to_thick_bright_edges(img_path, sensitivities=[1.0, 1.5, 2.0], 
                               edge_thicknesses=[1, 2, 3]):
    """
    读取灰度图像，生成不同粗细和亮度的边缘检测结果
    
    参数:
        img_path: 输入灰度图像路径
        sensitivities: 边缘敏感度列表
        edge_thicknesses: 边缘粗细列表（1-3）
    """
    # 读取灰度图像并转换为numpy数组
    img = Image.open(img_path).convert('L')
    img_array = np.array(img, dtype=np.float32)

    center_values = [0.41814, 0.63661, 0.48882, 0.43767, 0.57227, 0.48016, 0.51997, 0.80224, 0.80742]
    edge_values = [-0.02233, -0.02233, -0.02233, -0.02233, -0.02233, -0.02233, -0.02233, -0.02233, -0.02233]
    center_values = [0.53292, 0.53869]
    edge_values = [-0.02233, -0.02233]
    
    # 生成所有组合的边缘结果
    # for thickness in edge_thicknesses:
    #     for s in sensitivities:
    for center, edge in zip(center_values, edge_values):
        edge_array = apply_thick_bright_edge_kernel(
            img_array, 
            center,
            edge
        )
        edge_image = Image.fromarray(edge_array, mode='L')
        
        # 生成输出路径
        output_path = f"/mnt/cephfs/xubingye/tsp/ft_local/0811_edge/heatmap_edge_center_{center}_edge_{edge}.png"
        edge_image.save(output_path)
        print(f"边缘结果（center={center}, edge={edge}）已保存至: {output_path}")
    
    return True

if __name__ == "__main__":
    # 输入灰度图像路径
    input_img_path = "/mnt/cephfs/xubingye/tsp/ft_local/ft_local/heatmap_gray.png"
    
    # 运行边缘检测（可调整参数组合）
    image_to_thick_bright_edges(
        input_img_path,
        sensitivities=[1.2, 1.5, 1.8],  # 敏感度参数
        edge_thicknesses=[1, 2, 3]      # 边缘粗细（1-3）
    )
