import os
from PIL import Image

def flip_images_in_directory(directory):
    """
    对目录内所有PNG图像执行先左右翻转再上下翻转，并覆盖保存
    
    参数:
        directory: 包含PNG图像的目录路径
    """
    # 遍历目录内所有文件
    for filename in os.listdir(directory):
        # 检查是否为PNG文件
        if filename.lower().endswith('.png'):
            file_path = os.path.join(directory, filename)
            
            try:
                # 打开图像
                with Image.open(file_path) as img:
                    # 先左右翻转（水平翻转），再上下翻转（垂直翻转）
                    flipped_img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)  # 左右翻转
                    flipped_img = flipped_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)  # 上下翻转
                    
                    # 覆盖原文件保存
                    flipped_img.save(file_path)
                    print(f"已处理: {filename}")
                    
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

if __name__ == "__main__":
    # 目标目录路径
    target_directory = "/mnt/cephfs/xubingye/tsp/ft_local/mmz/ft_local/mmz"
    
    # 检查目录是否存在
    if not os.path.isdir(target_directory):
        print(f"错误: 目录 {target_directory} 不存在")
    else:
        # 执行批量翻转操作
        flip_images_in_directory(target_directory)
        print("所有PNG图像处理完成")
