#!/bin/bash

# 指定要处理的目录
DIRECTORY="/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video/QA/LLaVA-Video-178K/0_30_s_academic_v0_1"

# 遍历目录中的所有 .tar.gz 文件
for file in "$DIRECTORY"/*.tar.gz; do
    # 检查文件是否存在
    if [ -f "$file" ]; then
        # 获取文件名（不带扩展名）
        base_name=$(basename "$file" .tar.gz)
        # 构建输出的 .tar 文件路径
        tar_file="$DIRECTORY/$base_name.tar"
        # 解压 .tar.gz 文件为 .tar 文件
        tar -xzf "$file" -O > "$tar_file"
        echo "Extracted $file to $tar_file"
    fi
done
