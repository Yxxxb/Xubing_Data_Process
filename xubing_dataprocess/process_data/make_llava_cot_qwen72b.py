from datakit.utils.files import find_all_files, read_jsonl_file, dump_list_to_jsonl_file
from concurrent.futures import ThreadPoolExecutor, as_completed
# from datakit.api_call.gpt4_o import gpt4o_api 
import os

cot_prompt = (
    "I have an image and a question that I want you to answer. "
    "I need you to strictly follow the format with four specific sections: SUMMARY, CAPTION, REASONING, and CONCLUSION. "
    "It is crucial that you adhere to this structure exactly as outlined and that the final answer in the CONCLUSION matches the standard correct answer precisely. "
    "To explain further: "
    "In SUMMARY, briefly explain what steps you'll take to solve the problem. "
    "In CAPTION, describe the contents of the image, specifically focusing on details relevant to the question. "
    "In REASONING, outline a step-by-step thought process you would use to solve the problem based on the image. "
    "In CONCLUSION, give the final answer in a direct format, and it must match the correct answer exactly. "
    "If it's a multiple choice question, the conclusion should only include the option without repeating what the option is. "
    "Here's how the format should look: "
    "<SUMMARY> [Summarize how you will approach the problem and explain the steps you will take to reach the answer.] </SUMMARY> "
    "<CAPTION> [Provide a detailed description of the image, particularly emphasizing the aspects related to the question.] </CAPTION> "
    "<REASONING> [Provide a chain-of-thought, logical explanation of the problem. This should outline step-by-step reasoning.] </REASONING> "
    "<CONCLUSION> [State the final answer in a clear and direct format. It must match the correct answer exactly.] </CONCLUSION> "
    "(Do not forget </CONCLUSION>!) "
    "Please apply this format meticulously to analyze the given image and answer the related question."
    # "Please apply this format meticulously to analyze the given image and answer the related question, ensuring that the answer matches the standard one perfectly."
)

# complex results to cot input
# 带有是否为困难标签的text、easy、diff数据，输出仅仅包含diff text
root = "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results"
save_root = "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_cot"
jsonls = find_all_files(root, '.jsonl')
for _idx, jsonl in enumerate(jsonls):
    data = read_jsonl_file(jsonl)
    data = [{'id': item['id'], 'base64_image': item['base64_image'], 'conversations': item['conversations']} for item in data if item['complexity'] == 'hard']
    for item in data:
        # item['conversations'][0]['text'] = cot_prompt + "\n" + \
        #     "Question: " + item['conversations'][0]['text'] + "\n" + \
        #         "Standard answer: " + item['conversations'][1]['text']
        item['conversations'][0]['text'] = cot_prompt + "\n" + \
            "Question: " + item['conversations'][0]['text']
    dataset_name = jsonl.split("/")[-1].split(".")[0]
    os.makedirs(os.path.join(save_root, dataset_name, "data", "grammar_correct_wo_gt"), exist_ok=True)
    dump_list_to_jsonl_file(os.path.join(save_root, dataset_name, "data", "grammar_correct_wo_gt", f"{dataset_name}.jsonl"), data)


# import shutil

# for filename in os.listdir(save_root):
#     if filename.endswith('.jsonl'):
#         folder_name = os.path.splitext(filename)[0]
#         os.makedirs(os.path.join(save_root, folder_name), exist_ok=True)
#         shutil.move(os.path.join(save_root, filename), os.path.join(save_root, folder_name, filename))

"""
content.append({
                        "type": "image_url",
                        "image_url": {'url': 'data:image/jpeg;base64,' + base64_image}
                    })
                    content.append({
                        "type": "text",
                        "text": (
                            "I have an image and a question that I want you to answer. I need you to strictly follow the format with four specific sections: SUMMARY, CAPTION, REASONING, and CONCLUSION. It is crucial that you adhere to this structure exactly as outlined and that the final answer in the CONCLUSION matches the standard correct answer precisely. To explain further: In SUMMARY, briefly explain what steps you'll take to solve the problem. In CAPTION, describe the contents of the image, specifically focusing on details relevant to the question. In REASONING, outline a step-by-step thought process you would use to solve the problem based on the image. In CONCLUSION, give the final answer in a direct format, and it must match the correct answer exactly. If it's a multiple choice question, the conclusion should only include the option without repeating what the option is. Here's how the format should look: <SUMMARY> [Summarize how you will approach the problem and explain the steps you will take to reach the answer.] </SUMMARY> <CAPTION> [Provide a detailed description of the image, particularly emphasizing the aspects related to the question.] </CAPTION> <REASONING> [Provide a chain-of-thought, logical explanation of the problem. This should outline step-by-step reasoning.] </REASONING> <CONCLUSION> [State the final answer in a clear and direct format. It must match the correct answer exactly.] </CONCLUSION> (Do not forget </CONCLUSION>!) Please apply this format meticulously to analyze the given image and answer the related question, ensuring that the answer matches the standard one perfectly."
                        )
                    })
                    
                    assert index > 0 and conversations[index - 1]['from'] == 'human'
                    conversations[index - 1]['value'] = conversations[index - 1]['value'].replace('<image>\n', '').replace('\n<image>', '')
                    question = conversations[index - 1]['value']

                    standard_answer = value
                    
                    content.append({
                        "type": "text",
                        "text": "Question: " + question + "\n"
                    })
                    content.append({
                        "type": "text",
                        "text": "Standard answer: " + standard_answer
                    })
                    if hints:
                        added_hints = "".join([f"\nHint: {hint}" for hint in hints])
                        content.append({
                            "type": "text",
                            "text": added_hints
                        })

                    messages = [
                        {
                            'role': 'user',
                            'content': content
                        }
                    ]
"""
