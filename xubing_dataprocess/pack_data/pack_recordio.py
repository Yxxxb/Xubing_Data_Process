import os
import sys
import argparse
from transformers import AutoTokenizer
from datakit.utils.files import read_mmq_index, read_mmq_recordio
reader_all_sft = read_mmq_index("/mnt/cephfs/bensenliu/wfs/datasets/mm/sft/composition/202505/20250515/e1_sft_index.recordio")
reader_doc_ocr = read_mmq_index("/mnt/cephfs/bensenliu/wfs/datasets/mm/sft/composition/202506/20250620/e1_sft_index.recordio")
# breakpoint()

# reader2 = read_mmq_index("/mnt/cephfs/xubingye/wfs/datasets/sft/202505/20250513/e1_sft_qwen25_72b_12m_pxls_reorg_text_index.recordio")
# reader3 = read_mmq_index("/mnt/cephfs/xubingye/wfs/datasets/sft/202504/20250429/e2_sft_qwen25_72b_index.recordio")
# reader4 = read_mmq_index("/mnt/cephfs/xubingye/wfs/datasets/sft/202505/20250513/e1_sft_qwen25_72b_12m_pxls_reorg_text_index.recordio")
# reader1 = read_mmq_index("/mnt/cephfs/bensenliu/wfs/datasets/mm/sft/composition/202504/20250427/e2_sft_index.recordio")
# reader2 = read_mmq_index("/mnt/cephfs/xubingye/wfs/datasets/sft/202504/20250427/e2_sft_qwen25_72b_index.recordio")
# reader_test1 = read_mmq_recordio("/mnt/cephfs/xubingye/wfs/datasets/rl-category/LONG-COT-TEXT/OpenMathReasoning/recordio/grammar_correct/openmathreasoning_text_long_cot/grammar_correct/0_0_data.recordio")
# reader_test2 = read_mmq_recordio("/mnt/cephfs/xubingye/wfs/datasets/rl-category/LONG-COT-TEXT/OpenThoughts2-1M/recordio/grammar_correct/openmathreasoning_text_long_cot/grammar_correct/0_0_data.recordio")
# reader_test3 = read_mmq_recordio("/mnt/cephfs/xubingye/wfs/datasets/rl-category/LONG-COT-TEXT/AM-DeepSeek-R1-Distilled-1.4M/recordio/grammar_correct/openmathreasoning_text_long_cot/grammar_correct/0_0_data.recordio")
# reader_test2 = read_mmq_recordio("/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/TEXT/welm_sft_20250120/recordio/qwen2vl-grammar_correct-qwen-2-5/welm_sft_20250120_image.recordio")
# reader_test3 = read_mmq_recordio("/mnt/cephfs/xubingye/wfs/datasets/rl-category/long-cot-text/OpenMathReasoning/recordio/grammar_correct/openmathreasoning_text_long_cot/grammar_correct/0_0_data.recordio")
# reader_test4 = read_mmq_recordio("/mnt/cephfs/xubingye/wfs/datasets/rl-category/long-cot-text/OpenMathReasoning/recordio/grammar_correct/openmathreasoning_text_long_cot/grammar_correct/0_0_image.recordio")

reader = read_mmq_index("/mnt/cephfs/bensenliu/wfs/datasets/mm/sft/composition/202412/20241230/e2_sft_index.recordio")
data_list_previous = reader.header.filenames
reader_zhongyin_decay_0618 = read_mmq_index("/mnt/cephfs/zhonyinzhao/wfs/datasets/mm/sft/composition/202506/20250612/e1_decay_index.recordio")
data_list_zhongyin_decay_0618 = reader_zhongyin_decay_0618.header.filenames

data_list = [i for i in data_list_previous if 'enhanced_data.recordio' not in i]
data_list = list(set(data_list))

# sft data e2_sft_index去重、去掉Qwen72B刷数据中重复的，（science的三个数据集除外）
e2_sft_filtered_data = [
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/QA/Mantis-Instruct-multi_vqa/recordio/qwen2vl-grammar_correct-qwen-2-5/Mantis-Instruct-multi_vqa_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/openhermes2.5/recordio/qwen2vl-grammar-correct-qwen-2-5/openhermes2.5_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/mini-gemini/recordio/qwen2vl-grammar-correct-qwen-2-5/mini-gemini_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/kvqa/recordio/qwen2vl-grammar-correct-qwen-2-5/kvqa_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/k12-print/recordio/qwen2vl-grammar_correct-qwen-2-5/k12-print_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/TEXT2IMAGE/NuminaMath-CoT/recordio/qwen2vl-grammar_correct-qwen-2-5/NuminaMath-CoT_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/infovqa/recordio/qwen2vl-grammar-correct-qwen-2-5/infovqa_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/docvqa/recordio/qwen2vl-grammar-correct-qwen-2-5/docvqa_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/LaTeX_OCR_column_rename_full/recordio/qwen2vl-grammar_correct-qwen-2-5/LaTeX_OCR_column_rename_full_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/llavar/recordio/qwen2vl-grammar-correct-qwen-2-5/llavar_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/textvqa/recordio/qwen2vl-grammar-correct-qwen-2-5/textvqa_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/QA/Mantis-Instruct-spot-the-diff/recordio/qwen2vl-grammar_correct-qwen-2-5/Mantis-Instruct-spot-the-diff_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/gpt4o-complex-20240809-en/recordio/qwen2vl-grammar-correct-qwen-2-5/gpt4o-complex-20240809-en_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/QA/Mantis-Instruct-dreamsim/recordio/qwen2vl-grammar_correct-qwen-2-5/Mantis-Instruct-dreamsim_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/SCIENCE/mm_ai2d/recordio/qwen2vl-grammar_correct-qwen2-vl-reasoning-mamoothvl-en-100-per-qwen-2-5/mm_ai2d_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/hme100k/recordio/qwen2vl-grammar_correct-qwen-2-5/hme100k_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/scienceqa/recordio/qwen2vl-grammar-correct-qwen-2-5/scienceqa_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/dvqa/recordio/qwen2vl-grammar-correct-qwen-2-5/dvqa_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/geo3k/recordio/qwen2vl-grammar-correct-qwen-2-5/geo3k_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/LaTeX_OCR/recordio/qwen2vl-grammar_correct-qwen-2-5/LaTeX_OCR_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/MathInstruct/recordio/qwen2vl-grammar-correct-qwen-2-5/MathInstruct_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/mm_ai2drecordio/qwen2vl-grammar-correct-backup-qwen-2-5/mm_ai2d_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/tqa/recordio/qwen2vl-grammar-correct-qwen-2-5/tqa_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/MetaMathQA/recordio/qwen2vl-grammar-correct-qwen-2-5/MetaMathQA_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/MathV360K/recordio/qwen2vl-grammar-correct-qwen-2-5/MathV360K_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/QA/Mantis-Instruct-nlvr2/recordio/qwen2vl-grammar_correct-qwen-2-5/Mantis-Instruct-nlvr2_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/iconqa_choose_txt/recordio/qwen2vl-grammar-correct-qwen-2-5/iconqa_choose_txt_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/alpaca-gpt4/recordio/qwen2vl-grammar-correct-qwen-2-5/alpaca-gpt4_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/iconqa_fill_blank/recordio/qwen2vl-grammar-correct-qwen-2-5/iconqa_fill_blank_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/orca-math-word-problems-200k/recordio/qwen2vl-grammar-correct-qwen-2-5/orca-math-word-problems-200k_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/QA/Mantis-Instruct-contrastive_caption/recordio/qwen2vl-grammar_correct-qwen-2-5/Mantis-Instruct-contrastive_caption_data.recordio",
    "/mnt/cephfs/bensenliu/dataset/mm/sharegpt4v/recordio/qwen2vl-grammar-correct-with-image-token-qwen-2-5/sharegpt4v_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/SROIE_2019_text_recognition/recordio/qwen2vl-grammar_correct-qwen-2-5/SROIE_2019_text_recognition_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/ESTVQA/recordio/qwen2vl-grammar_correct-qwen-2-5/ESTVQA_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/TEXT/belle_cn_1m/recordio/qwen2vl-grammar_correct-qwen-2-5/belle_cn_1m_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/icdar_2015/recordio/qwen2vl-grammar-correct-qwen-2-5/icdar_2015_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v_qa_cn/recordio/qwen2vl-grammar_correct_translated_correct-qwen-2-5/sharegpt4v_qa_cn_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/ChartQA/recordio/qwen2vl-grammar_correct-qwen-2-5/ChartQA_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/poie/recordio/qwen2vl-grammar-correct-qwen-2-5/poie_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/gpt4v/recordio/qwen2vl-grammar-correct-qwen-2-5/gpt4v_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/geoqa+/recordio/qwen2vl-grammar-correct-qwen-2-5/geoqa+_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/SCIENCE/tqa/recordio/qwen2vl-grammar_correct-qwen2-vl-reasoning-mamoothvl-en-100-per-qwen-2-5/tqa_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/QA/Mantis-Instruct-lrv_multi/recordio/qwen2vl-grammar_correct-qwen-2-5/Mantis-Instruct-lrv_multi_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/QA/Mantis-Instruct-birds-to-words/recordio/qwen2vl-grammar_correct-qwen-2-5/Mantis-Instruct-birds-to-words_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/SCIENCE/scienceqa/recordio/qwen2vl-grammar_correct-qwen2-vl-reasoning-mamoothvl-en-100-per-qwen-2-5/scienceqa_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v_choice_cn/recordio/qwen2vl-grammar_correct-qwen-2-5/sharegpt4v_choice_cn_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/lima/recordio/qwen2vl-grammar-correct-qwen-2-5/lima_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/QA/Mantis-Instruct-iconqa/recordio/qwen2vl-grammar_correct-qwen-2-5/Mantis-Instruct-iconqa_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/stvqa/recordio/qwen2vl-grammar-correct-qwen-2-5/stvqa_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/super_clever/recordio/qwen2vl-grammar-correct-qwen-2-5/super_clever_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/tabwp_cot/recordio/qwen2vl-grammar-correct-qwen-2-5/tabwp_cot_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/500k-atlas-math/recordio/qwen2vl-grammar-correct-qwen-2-5/500k-atlas-math_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/clevr_math_5w/recordio/qwen2vl-grammar-correct-qwen-2-5/clevr_math_5w_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/vsr/recordio/qwen2vl-grammar-correct-qwen-2-5/vsr_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/math/recordio/qwen2vl-grammar-correct-qwen-2-5/math_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/handwritten_cn/recordio/qwen2vl-grammar_correct-qwen-2-5/handwritten_cn_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/TEXT/welm_sft_20250120/recordio/qwen2vl-grammar_correct-qwen-2-5/welm_sft_20250120_data.recordio"
]

# sft data qwen2.5 72B Points resolution Qwen-2.5-VL72B蒸馏数据
qwen_72b_reinfer_data = [
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/Chinese-Card-OCR/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/Chinese-OCR/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/value-added-tax-invoice-cn/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/aiplane-itinerary-cn/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio", 
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/bank_card_cn/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/bus_ticket_cn/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/customs_declaration_cn/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/duty-paid-proof-cn/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/express_waybill_cn/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/invoice-mix-up-cn/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/medical_bill_cn/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/quota_invoice_cn/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/railway_ticket_cn/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/ride_sharing_cn/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/steamer_ticket_cn/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/taxi_ticket_cn/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/toll_invoice_cn/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/vat_invoice_roll_cn/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/invoices-and-receipts_ocr_v1_image_translation/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/CAPTION/allava_cap/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/CAPTION/gpt4v-cn/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/CAPTION/lvis_instruct4v_cap_cn/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/QA/free_style_qa/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MEMES/memes-500/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    # "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/SCIENCE/mm_ai2d/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    # "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/SCIENCE/scienceqa/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
    # "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/SCIENCE/tqa/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/qwen_2_5_vl_72b_vllm_12M_reso_reorganize_text/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_reorganize_text/*_data.recordio",
]

# CoT思维链难度数据，其中部分为困难的，剩余为简单的
cot_data = [
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_cot/ChartQA/recordio/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/qwen_2_5_vl_72b_vllm_cot/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_cot/clevr_math_5w/recordio/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/qwen_2_5_vl_72b_vllm_cot/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_cot/geo3k/recordio/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/qwen_2_5_vl_72b_vllm_cot/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_cot/geoqa+/recordio/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/qwen_2_5_vl_72b_vllm_cot/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_cot/iconqa_choose_txt/recordio/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/qwen_2_5_vl_72b_vllm_cot/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_cot/iconqa_fill_blank/recordio/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/qwen_2_5_vl_72b_vllm_cot/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_cot/infovqa/recordio/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/qwen_2_5_vl_72b_vllm_cot/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_cot/super_clever/recordio/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/qwen_2_5_vl_72b_vllm_cot/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_cot/mm_ai2d/recordio/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/qwen_2_5_vl_72b_vllm_cot/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_cot/scienceqa/recordio/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/qwen_2_5_vl_72b_vllm_cot/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_cot/tqa/recordio/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/qwen_2_5_vl_72b_vllm_cot/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/*_data.recordio"
]

# Few Shot难度数据，其中部分为困难的，剩余为简单的
few_shot_data = [
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_few_shot/ChartQA/recordio/grammar_correct/qwen_2_5_vl_72b_vllm_few_shot/grammar_correct/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_few_shot/clevr_math_5w/recordio/grammar_correct/qwen_2_5_vl_72b_vllm_few_shot/grammar_correct/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_few_shot/geo3k/recordio/grammar_correct/qwen_2_5_vl_72b_vllm_few_shot/grammar_correct/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_few_shot/geoqa+/recordio/grammar_correct/qwen_2_5_vl_72b_vllm_few_shot/grammar_correct/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_few_shot/iconqa_choose_txt/recordio/grammar_correct/qwen_2_5_vl_72b_vllm_few_shot/grammar_correct/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_few_shot/iconqa_fill_blank/recordio/grammar_correct/qwen_2_5_vl_72b_vllm_few_shot/grammar_correct/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_few_shot/infovqa/recordio/grammar_correct/qwen_2_5_vl_72b_vllm_few_shot/grammar_correct/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_few_shot/super_clever/recordio/grammar_correct/qwen_2_5_vl_72b_vllm_few_shot/grammar_correct/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_few_shot/mm_ai2d/recordio/grammar_correct/qwen_2_5_vl_72b_vllm_few_shot/grammar_correct/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_few_shot/scienceqa/recordio/grammar_correct/qwen_2_5_vl_72b_vllm_few_shot/grammar_correct/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_few_shot/tqa/recordio/grammar_correct/qwen_2_5_vl_72b_vllm_few_shot/grammar_correct/*_data.recordio"
]

# video
video_data = [
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video/QA/finevideo/recordio/2_fps_all_conversations/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video/QA/VideoChat2-IT/video_content/sub2/recordio/maxframe64/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video/CAPTION/VideoUFO/recordio/2fps_64maxframes/maxframe64_160token/2fps_64maxframes/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video/CAPTION/ShareGPT4Video/recordio/grammar_correct_2fps_64maxframes/maxframe64/grammar_correct_2fps_64maxframes/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video/CAPTION/OpenVid-1M/recordio/full/maxframe64/full/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video/CAPTION/Vript/recordio/grammar_correct_2FPS_200maxframes_prompted/maxframe64_160token/grammar_correct_2FPS_200maxframes_prompted/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video/QA/LLaVA-Hound/recordio/grammar_correct_1fps_64maxframes/maxframe64/grammar_correct_1fps_64maxframes/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video/QA/LLaVA-Video-178K-refine/recordio/grammar_correct_2fps_200maxframes/maxframe64_160token/grammar_correct_2fps_200maxframes/*_data.recordio",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video/QA/cinepile/v2/recordio/fps1_maxframe3000/maxframe64_160token/fps1_maxframe3000/*_data.recordio"
]

# text cot
text_cot_data = [
    "/mnt/cephfs/xubingye/wfs/datasets/rl-category/LONG-COT-TEXT/OpenMathReasoning/recordio/grammar_correct/openmathreasoning_text_long_cot/grammar_correct/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/rl-category/LONG-COT-TEXT/OpenThoughts2-1M/recordio/grammar_correct/openmathreasoning_text_long_cot/grammar_correct/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/rl-category/LONG-COT-TEXT/AM-DeepSeek-R1-Distilled-1.4M/recordio/grammar_correct/openmathreasoning_text_long_cot/grammar_correct/*_data.recordio",
]

# text cot pt
text_cot_pt_data = [
    "/mnt/cephfs/xubingye/wfs/datasets/rl-category/LONG-COT-TEXT/OpenMathReasoning/recordio/grammar_correct/openmathreasoning_text_long_cot_pt_no_role/grammar_correct/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/rl-category/LONG-COT-TEXT/OpenThoughts2-1M/recordio/grammar_correct/openmathreasoning_text_long_cot_pt_no_role/grammar_correct/*_data.recordio",
    "/mnt/cephfs/xubingye/wfs/datasets/rl-category/LONG-COT-TEXT/AM-DeepSeek-R1-Distilled-1.4M/recordio/grammar_correct/openmathreasoning_text_long_cot_pt_no_role/grammar_correct/*_data.recordio",
]

# cold start cot sft
count_cold_start_data = [
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/Clevr_CoGenT_TrainA_R1_cold_start/recordio/grammar_correct/openmathreasoning_text_long_cot_pt_no_role/grammar_correct/*_data.recordio",
]

# Obj365
grounding_obj365_data = [
    "/mnt/cephfs/xubingye/wfs/datasets/sft-category/Objects365/recordio/grammar_correct/openmathreasoning_text_long_cot_pt_no_role/grammar_correct/*_data.recordio",
]

# decay doc ocr
reader_all_sft = read_mmq_index("/mnt/cephfs/bensenliu/wfs/datasets/mm/sft/composition/202505/20250515/e1_sft_index.recordio")
reader_doc_ocr = read_mmq_index("/mnt/cephfs/bensenliu/wfs/datasets/mm/sft/composition/202506/20250620/e1_sft_index.recordio")
decay_doc_ocr_data = [i for i in reader_doc_ocr.header.filenames if i not in reader_all_sft.header.filenames]

# hunyuan
hunyuan_data = [
    "/mnt/wfs/mmnanjingwfssh/project_pr-nlp-large_pretrain/zhonyinzhao/datasets/mm/decay/Tiku/recordio/aidatacos/base/*_data.recordio"
]

# chuhan tiku
chuhan_data = [
    "/mnt/wfs/mmnanjingwfssh/project_pr-nlp-large_pretrain/zhonyinzhao/datasets/mm/decay/PDF/recordio/tiku/PR@base_ori05/*_data.recordio"
]

# qwen2.5 caption + qwen3 cot + with noise
mm_cot_noise = [
    '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/CHART/ChartQA/recordio/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/openmathreasoning_text_long_cot_pt_no_role/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/*_data.recordio',
    '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/DIAGRAM/infovqa/recordio/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/openmathreasoning_text_long_cot_pt_no_role/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/*_data.recordio',
    '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/clevr_math_5w/recordio/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/openmathreasoning_text_long_cot_pt_no_role/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/*_data.recordio',
    '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/geo3k/recordio/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/openmathreasoning_text_long_cot_pt_no_role/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/*_data.recordio',
    '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/geoqa+/recordio/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/openmathreasoning_text_long_cot_pt_no_role/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/*_data.recordio',
    '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/iconqa_choose_txt/recordio/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/openmathreasoning_text_long_cot_pt_no_role/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/*_data.recordio',
    '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/iconqa_fill_blank/recordio/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/openmathreasoning_text_long_cot_pt_no_role/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/*_data.recordio',
    '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/super_clever/recordio/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/openmathreasoning_text_long_cot_pt_no_role/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/*_data.recordio',
    '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/SCIENCE/mm_ai2d/recordio/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/openmathreasoning_text_long_cot_pt_no_role/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/*_data.recordio',
    '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/SCIENCE/scienceqa/recordio/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/openmathreasoning_text_long_cot_pt_no_role/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/*_data.recordio',
    '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/SCIENCE/tqa/recordio/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/openmathreasoning_text_long_cot_pt_no_role/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_raw/*_data.recordio',
]

# data_list = qwen_72b_reinfer_data

# 20250519-e1xubing
# data_list = qwen_72b_reinfer_data + e2_sft_filtered_data + cot_data

# 20250519-e2xubing
# data_list = qwen_72b_reinfer_data + e2_sft_filtered_data + few_shot_data

# 20250527-e1xubing
# data_list = qwen_72b_reinfer_data + e2_sft_filtered_data + video_data

# 20250605-e1xubing
# data_list = text_cot_data

# 20250619-e1xubing
# data_list = data_list_zhongyin_decay_0618 + text_cot_data

# 20250621-e1xubing
# data_list = data_list_zhongyin_decay_0618 + text_cot_pt_data

# 20250707-e1xubing
# data_list = count_cold_start_data

# 20250707-e2xubing
# data_list = decay_doc_ocr_data + data_list_zhongyin_decay_0618

# 20250711-e1xubing
# data_list = data_list_zhongyin_decay_0618 + chuhan_data + hunyuan_data + grounding_obj365_data

# 20250729-e1xubing 675840000 tokens 580000 samples mm
# data_list = mm_cot_noise

# 20250729-e2xubing 
data_list = mm_cot_noise + reader_all_sft.header.filenames

data_seq = " ".join(data_list)
micro_batch_size = 8192*4
output_dir = "/mnt/cephfs/xubingye/wfs/datasets/sft/202507/20250729"
if not os.path.exists(output_dir):
    os.system(f"mkdir -p {output_dir}")
os.system(f"mmq_batch_collator --input_files {data_seq} --output_file {os.path.join(output_dir,'e2_sft_e1_mm_cot_with_noise_full_sft_index.recordio')} --num_epochs 1 --batch_strategy by_token --micro_batch_tokens_limit {micro_batch_size} --batch_tokens {micro_batch_size*64} --split_doc False")
