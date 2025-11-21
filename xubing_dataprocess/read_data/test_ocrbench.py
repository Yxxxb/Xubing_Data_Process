import pandas as pd

# 读取Excel文件
df = pd.read_excel('/mnt/cephfs/xubingye/vlm/VLMEvalKit/eval_res/202506/20250606e1xubing/POINTSV15-API/T20250623_G93533403/POINTSV15-API_OCRBench.xlsx')
df = pd.read_excel('/mnt/cephfs/xubingye/vlm/VLMEvalKit/eval_res/202505/20250519e2xubing/POINTSV15-Qwen-2.5-7B-Chat/T20250520_G93533403/POINTSV15-Qwen-2.5-7B-Chat_OCRBench.xlsx')

answer = df['answer'].tolist()
prediction = df['prediction'].tolist()
category = df['category'].tolist()

# breakpoint()
error_list = []

count = 0
all = 0
for i in range(len(answer)):
    if isinstance(prediction[i], str) and '<think>' in prediction[i]:
        continue
    # if category[i] != 'Regular Text Recognition':
    #     continue
    is_in = False
    for cur_answer in eval(answer[i]):
        # breakpoint()
        cur_answer = str(cur_answer).strip().replace('\n', ' ').replace(' ', '').lower()
        cur_prediction = str(prediction[i]).strip().replace('\n', ' ').replace(' ', '').lower()
        if 'erroroccurredduringapicall.pleasecheckthelogs.' in cur_prediction or cur_answer in cur_prediction:
            count += 1
            is_in = True
            break
    if not is_in:
        print(f'Error: {cur_answer} vs {cur_prediction}')
        error_list.append(i)
    all += 1
print(f'Accuracy: {count / all}')
print(f'Total: {all}')
print(f'Correct: {count}')
print(error_list)
