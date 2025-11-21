from datakit.utils.jsonl_recordio_index import jsonl_to_recordio_to_index
from datakit.utils.generate_index import generate_index
import os
from tqdm import tqdm


if __name__ == "__main__":
    print("seccessfully import jsonl_to_recordio_to_index!!!")

    # SFT Qwen72 重刷
    # 这里存储路径为：
    # .../recordio/raw_root_list_foldername/rename/raw_root_list_foldername
    # .../OCR/Chinese-Card-OCR/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_test_recordio/qwen_2_5_vl_72b_vllm/grammar_correct_vlm_infer_Qwen72B_vllm_test_recordio
    # with open("/mnt/cephfs/xubingye/vlm/MMDataKit/workspace/data_to_filter.txt", 'r') as f:
    #     datasets = f.readlines()
    #     datasets = [dataset.strip() for dataset in datasets]
    #     idx = 0
    #     for dataset in tqdm(datasets):
    #         # if dataset == "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/CAPTION/allava_cap/data/grammar_correct":
    #         #     continue
    #         print("#############", idx, dataset)
    #         cur_raw_root_list = dataset + "_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text"
    #         jsonl_to_recordio_to_index(raw_root_list = cur_raw_root_list, alert_path = f"/mnt/cephfs/xubingye/envs/loop_test_{idx}",
    #                                 index_file_root = "/mnt/cephfs/haichengwang/wfs/datasets/mm/sft/composition", index_file_name = None, 
    #                                 base_train_config = "/mnt/cephfs/haichengwang/code/mimikyu_h800/configs/202503/20250327/20250327e5-sft.yaml",model_save_folder="/mnt/cephfs/haichengwang/wfs/weights/mm/",
    #                                 train_config_file="/mnt/cephfs/haichengwang/code/mimikyu_h800/configs/202503/20250328/20250328e5-sft.yaml", base_datasets_file=None, num_epochs=1, micro_batch_tokens_limit=8192, batch_size=64,
    #                                 maxframes=64, rename="qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text", maxpixels=14 * 14 * 4 * 2048, num_workers=64,
    #                                 )
    #         idx += 1

    # CoTls
    # with open("/mnt/cephfs/xubingye/vlm/MMDataKit/workspace/complex_data_to_filter_cot_3B.txt", 'r') as f:
    #     datasets = f.readlines()
    #     datasets = [dataset.strip() for dataset in datasets]
    #     idx = 0
    #     for dataset in tqdm(datasets):
    #         print("#############", idx, dataset)
    #         cur_raw_root_list = dataset + "_vlm_infer_Qwen72B_complexity_vllm_pe"
    #         jsonl_to_recordio_to_index(raw_root_list = cur_raw_root_list, alert_path = f"/mnt/cephfs/xubingye/envs/loop_test_{idx}",
    #                                 index_file_root = "/mnt/cephfs/haichengwang/wfs/datasets/mm/sft/composition", index_file_name = None, 
    #                                 base_train_config = "/mnt/cephfs/haichengwang/code/mimikyu_h800/configs/202503/20250327/20250327e5-sft.yaml",model_save_folder="/mnt/cephfs/haichengwang/wfs/weights/mm/",
    #                                 train_config_file="/mnt/cephfs/haichengwang/code/mimikyu_h800/configs/202503/20250328/20250328e5-sft.yaml", base_datasets_file=None, num_epochs=1, micro_batch_tokens_limit=8192, batch_size=64,
    #                                 maxframes=64, rename="qwen_2_5_vl_72b_vllm_cot", maxpixels=14 * 14 * 4 * 2048, num_workers=64,
    #                                 )
    #         idx += 1

    # CoT w/o gt
    # with open("/mnt/cephfs/xubingye/vlm/MMDataKit/workspace/complex_data_to_filter_cot_3B_wo_gt.txt", 'r') as f:
    #     datasets = f.readlines()
    #     datasets = [dataset.strip() for dataset in datasets]
    #     idx = 0
    #     for dataset in tqdm(datasets):
    #         print("#############", idx, dataset)
    #         cur_raw_root_list = dataset + "_vlm_infer_Qwen72B_complexity_vllm_pe"
    #         jsonl_to_recordio_to_index(raw_root_list = cur_raw_root_list, alert_path = f"/mnt/cephfs/xubingye/envs/loop_test_{idx}",
    #                                 index_file_root = "/mnt/cephfs/haichengwang/wfs/datasets/mm/sft/composition", index_file_name = None, 
    #                                 base_train_config = "/mnt/cephfs/haichengwang/code/mimikyu_h800/configs/202503/20250327/20250327e5-sft.yaml",model_save_folder="/mnt/cephfs/haichengwang/wfs/weights/mm/",
    #                                 train_config_file="/mnt/cephfs/haichengwang/code/mimikyu_h800/configs/202503/20250328/20250328e5-sft.yaml", base_datasets_file=None, num_epochs=1, micro_batch_tokens_limit=8192, batch_size=64,
    #                                 maxframes=64, rename="qwen_2_5_vl_72b_vllm_cot_wo_gt", maxpixels=14 * 14 * 4 * 2048, num_workers=64,
    #                                 )
    #         idx += 1
    
    # interleave test
    # dataset = "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_few_shot/ChartQA/data/grammar_correct"
    # cur_raw_root_list = dataset 
    # jsonl_to_recordio_to_index(raw_root_list = cur_raw_root_list, alert_path = f"/mnt/cephfs/xubingye/envs/loop_test_0",
    #                         index_file_root = "/mnt/cephfs/haichengwang/wfs/datasets/mm/sft/composition", index_file_name = None, 
    #                         base_train_config = "/mnt/cephfs/haichengwang/code/mimikyu_h800/configs/202503/20250327/20250327e5-sft.yaml",model_save_folder="/mnt/cephfs/haichengwang/wfs/weights/mm/",
    #                         train_config_file="/mnt/cephfs/haichengwang/code/mimikyu_h800/configs/202503/20250328/20250328e5-sft.yaml", base_datasets_file=None, num_epochs=1, micro_batch_tokens_limit=8192, batch_size=64,
    #                         maxframes=64, rename="qwen_2_5_vl_72b_vllm_few_shot", maxpixels=14 * 14 * 4 * 2048, num_workers=64,
    #                         )
    
    # few shot
    # raw_tor_recordio_fs.jsonl
    # with open("/mnt/cephfs/xubingye/vlm/MMDataKit/workspace/complex_data_to_filter_few_shot_3B.txt", 'r') as f:
    #     datasets = f.readlines()
    #     datasets = [dataset.strip() for dataset in datasets]
    #     idx = 0
    #     for dataset in tqdm(datasets):
    #         # # geoqa+ iconqa_choose_txt iconqa_fill_blank tqa
    #         # if not ("iconqa_choose_txt" in dataset or "iconqa_fill_blank" in dataset or "tqa" in dataset):
    #         #     continue
    #         print("#############", idx, dataset)
    #         cur_raw_root_list = dataset
    #         jsonl_to_recordio_to_index(raw_root_list = cur_raw_root_list, alert_path = f"/mnt/cephfs/xubingye/envs/loop_test_{idx}",
    #                                 index_file_root = "/mnt/cephfs/haichengwang/wfs/datasets/mm/sft/composition", index_file_name = None, 
    #                                 base_train_config = "/mnt/cephfs/haichengwang/code/mimikyu_h800/configs/202503/20250327/20250327e5-sft.yaml",model_save_folder="/mnt/cephfs/haichengwang/wfs/weights/mm/",
    #                                 train_config_file="/mnt/cephfs/haichengwang/code/mimikyu_h800/configs/202503/20250328/20250328e5-sft.yaml", base_datasets_file=None, num_epochs=1, micro_batch_tokens_limit=8192, batch_size=64,
    #                                 maxframes=64, rename="qwen_2_5_vl_72b_vllm_few_shot", maxpixels=14 * 14 * 4 * 2048, num_workers=64,
    #                                 )
    #         idx += 1

    # long cot text OpenMathReasoning
    # dataset = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/LONG-COT-TEXT/OpenMathReasoning/data/grammar_correct"
    # dataset = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/LONG-COT-TEXT/AM-DeepSeek-R1-Distilled-1.4M/data/grammar_correct"
    # dataset = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/LONG-COT-TEXT/OpenThoughts2-1M/data/grammar_correct"
    # cur_raw_root_list = dataset 
    # jsonl_to_recordio_to_index(raw_root_list = cur_raw_root_list, alert_path = f"/mnt/cephfs/xubingye/envs/loop_test_0",
    #                         index_file_root = "/mnt/cephfs/haichengwang/wfs/datasets/mm/sft/composition", index_file_name = None, 
    #                         base_train_config = "/mnt/cephfs/haichengwang/code/mimikyu_h800/configs/202503/20250327/20250327e5-sft.yaml",model_save_folder="/mnt/cephfs/haichengwang/wfs/weights/mm/",
    #                         train_config_file="/mnt/cephfs/haichengwang/code/mimikyu_h800/configs/202503/20250328/20250328e5-sft.yaml", base_datasets_file=None, num_epochs=1, micro_batch_tokens_limit=8192, batch_size=64,
    #                         maxframes=64, rename="openmathreasoning_text_long_cot", maxpixels=14 * 14 * 4 * 2048, num_workers=64,
    #                         )
    
    # # long cot text OpenMathReasoning pretrain
    # # dataset = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/LONG-COT-TEXT/OpenMathReasoning/data/grammar_correct"
    # # dataset = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/LONG-COT-TEXT/AM-DeepSeek-R1-Distilled-1.4M/data/grammar_correct"
    # dataset = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/LONG-COT-TEXT/OpenThoughts2-1M/data/grammar_correct"
    # cur_raw_root_list = dataset 
    # jsonl_to_recordio_to_index(raw_root_list = cur_raw_root_list, alert_path = f"/mnt/cephfs/xubingye/envs/loop_test_0",
    #                         index_file_root = "/mnt/cephfs/haichengwang/wfs/datasets/mm/sft/composition", index_file_name = None, 
    #                         base_train_config = "/mnt/cephfs/haichengwang/code/mimikyu_h800/configs/202503/20250327/20250327e5-sft.yaml",model_save_folder="/mnt/cephfs/haichengwang/wfs/weights/mm/",
    #                         train_config_file="/mnt/cephfs/haichengwang/code/mimikyu_h800/configs/202503/20250328/20250328e5-sft.yaml", base_datasets_file=None, num_epochs=1, micro_batch_tokens_limit=8192, batch_size=64,
    #                         maxframes=64, rename="openmathreasoning_text_long_cot_pt_no_role", maxpixels=14 * 14 * 4 * 2048, num_workers=64,
    #                         is_pretrain_decay=True,  # 续写
    #                         )

    # cold start counting
    # dataset = "/mnt/cephfs/xubingye/wfs/datasets/sft-category/Clevr_CoGenT_TrainA_R1_cold_start/data/grammar_correct"
    # cur_raw_root_list = dataset 
    # jsonl_to_recordio_to_index(raw_root_list = cur_raw_root_list, alert_path = f"/mnt/cephfs/xubingye/envs/loop_test_0",
    #                         index_file_root = "/mnt/cephfs/haichengwang/wfs/datasets/mm/sft/composition", index_file_name = None, 
    #                         base_train_config = "/mnt/cephfs/haichengwang/code/mimikyu_h800/configs/202503/20250327/20250327e5-sft.yaml",model_save_folder="/mnt/cephfs/haichengwang/wfs/weights/mm/",
    #                         train_config_file="/mnt/cephfs/haichengwang/code/mimikyu_h800/configs/202503/20250328/20250328e5-sft.yaml", base_datasets_file=None, num_epochs=1, micro_batch_tokens_limit=8192, batch_size=64,
    #                         maxframes=64, rename="openmathreasoning_text_long_cot_pt_no_role", maxpixels=14 * 14 * 4 * 2048, num_workers=64,
    #                         is_pretrain_decay=False,
    #                         )

    # Obj365
    # dataset = "/mnt/cephfs/xubingye/wfs/datasets/sft-category/Objects365/data/grammar_correct"
    # cur_raw_root_list = dataset 
    # jsonl_to_recordio_to_index(raw_root_list = cur_raw_root_list, alert_path = f"/mnt/cephfs/xubingye/envs/loop_test_0",
    #                         index_file_root = "/mnt/cephfs/haichengwang/wfs/datasets/mm/sft/composition", index_file_name = None, 
    #                         base_train_config = "/mnt/cephfs/haichengwang/code/mimikyu_h800/configs/202503/20250327/20250327e5-sft.yaml",model_save_folder="/mnt/cephfs/haichengwang/wfs/weights/mm/",
    #                         train_config_file="/mnt/cephfs/haichengwang/code/mimikyu_h800/configs/202503/20250328/20250328e5-sft.yaml", base_datasets_file=None, num_epochs=1, micro_batch_tokens_limit=8192, batch_size=64,
    #                         maxframes=64, rename="openmathreasoning_text_long_cot_pt_no_role", maxpixels=14 * 14 * 4 * 2048, num_workers=64,
    #                         is_pretrain_decay=True,
    #                         )

    # Qwen3 CoT with noise
    
    output_txt_path = "/mnt/cephfs/xubingye/vlm/MMDataKit/workspace/complex_data_to_filter_caption.txt"
    with open(output_txt_path, "r") as f:
        datasets = f.readlines()
    datasets = [dataset.strip() + '_vlm_infer_Qwen72B_caption_qwen3_cot_raw' for dataset in datasets]
    for dataset in tqdm(datasets):
        cur_raw_root_list = dataset 
        jsonl_to_recordio_to_index(raw_root_list = cur_raw_root_list, alert_path = f"/mnt/cephfs/xubingye/envs/loop_test_0",
                                index_file_root = "/mnt/cephfs/haichengwang/wfs/datasets/mm/sft/composition", index_file_name = None, 
                                base_train_config = "/mnt/cephfs/haichengwang/code/mimikyu_h800/configs/202503/20250327/20250327e5-sft.yaml",model_save_folder="/mnt/cephfs/haichengwang/wfs/weights/mm/",
                                train_config_file="/mnt/cephfs/haichengwang/code/mimikyu_h800/configs/202503/20250328/20250328e5-sft.yaml", base_datasets_file=None, num_epochs=1, micro_batch_tokens_limit=8192, batch_size=64,
                                maxframes=64, rename="openmathreasoning_text_long_cot_pt_no_role", maxpixels=14 * 14 * 4 * 2048, num_workers=64,
                                is_pretrain_decay=False,
                                )