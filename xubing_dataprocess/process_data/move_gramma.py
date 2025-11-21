import os
import shutil

# 你的txt文件路径
txt_path = '/mnt/cephfs/xubingye/vlm/MMDataKit/workspace/complex_data_to_filter_cot_3B.txt'
# 根目录
root_dir = '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_cot'

# 读取数据集名
with open(txt_path, 'r') as f:
    dataset_names = [os.path.basename(line.strip()) for line in f if line.strip()]

for dataset in dataset_names:
    # 三个相关文件夹
    folder_ori = os.path.join(root_dir, dataset)
    folder_vlm = os.path.join(root_dir, dataset + '_vlm_infer_Qwen72B_complexity_vllm')
    folder_vlm_pe = os.path.join(root_dir, dataset + '_vlm_infer_Qwen72B_complexity_vllm_pe')

    # 新建三个子文件夹
    sub_ori = os.path.join(folder_ori, 'grammar_correct')
    sub_vlm = os.path.join(folder_ori, 'grammar_correct_vlm_infer_Qwen72B_complexity_vllm')
    sub_vlm_pe = os.path.join(folder_ori, 'grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe')
    os.makedirs(sub_ori, exist_ok=True)
    os.makedirs(sub_vlm, exist_ok=True)
    os.makedirs(sub_vlm_pe, exist_ok=True)

    # 移动jsonl文件
    if os.path.exists(folder_ori):
        for file in os.listdir(folder_ori):
            if file.endswith('.jsonl'):
                shutil.move(os.path.join(folder_ori, file), os.path.join(sub_ori, file))
    if os.path.exists(folder_vlm):
        for file in os.listdir(folder_vlm):
            if file.endswith('.jsonl'):
                shutil.move(os.path.join(folder_vlm, file), os.path.join(sub_vlm, file))
    if os.path.exists(folder_vlm_pe):
        for file in os.listdir(folder_vlm_pe):
            if file.endswith('.jsonl'):
                shutil.move(os.path.join(folder_vlm_pe, file), os.path.join(sub_vlm_pe, file))

    print(f'处理完成: {dataset}')

print('全部完成！')
