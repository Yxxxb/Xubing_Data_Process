import os
import json

# Input and output directories
input_dir = "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/CAPTION/allava_cap/data/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso"
output_dir = "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/CAPTION/allava_cap/data/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso_filtered"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each .jsonl file in the input directory
count = 0
none_count = 0
for filename in os.listdir(input_dir):
    if filename.endswith(".jsonl"):
        input_file_path = os.path.join(input_dir, filename)
        output_file_path = os.path.join(output_dir, filename)

        filtered_data = []
        with open(input_file_path, "r", encoding="utf-8") as infile:
            for line in infile:
                item = json.loads(line)
                # Check if the required field is not None
                assert "conversations" in item and len(item["conversations"]) == 2
                if item["conversations"][1].get("qwen2.5vl-72b") is not None:
                    filtered_data.append(item)
                else:
                    none_count += 1
                count += 1
        # Write the filtered data to the output file
        with open(output_file_path, "w", encoding="utf-8") as outfile:
            for entry in filtered_data:
                outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("Processing complete. Filtered files are saved in:", output_dir)
print(f"Total entries processed: {count}")
print(f"Entries with None values: {none_count}")
