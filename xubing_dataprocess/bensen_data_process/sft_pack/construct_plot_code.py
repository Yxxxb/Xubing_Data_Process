import openai
import random
import re
from tqdm import tqdm
from datakit.utils.mp import multi_process_with_append
from datakit.utils.files import dump_list_to_jsonl_file
from datakit.utils.distributed import dist_split_files, get_distributed_env


with open('points-1.2/prompts/gen_plot_code.txt') as f:
    PROMPT = f.read()   

with open('points-1.2/prompts/code/plot.txt') as f:
    examples = f.read()
    examples = examples.split('---')
    examples = [e.strip() for e in examples]

pattern = r'```python(.*?)```'


def obtain_result(prompt):
    index, prompt = prompt
    client = openai.Client(base_url="http://127.0.0.1:8081/v1", api_key="None")
    seed = random.randint(0, 1000000)
    temperature = random.uniform(0.3, 1.0)
    top_p = random.uniform(0.2, 0.8)
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=2048,
    )
    response = response.choices[0].message.content
    extracted_code = re.findall(pattern, response, re.DOTALL)
    if len(extracted_code) > 0 and len(extracted_code[0].strip()) > 0:
        response = extracted_code[0].strip()
    print(response)
    result = {
        'id': str(index),
        'code': response
    }
    return result


if __name__ == '__main__':
    _, rank, _ = get_distributed_env()
    output_file = f'/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/CODE/image2code/raw/image2code_{rank}.jsonl'
    sample_num = 30000
    data = []
    for i in tqdm(range(sample_num)):
        random_example = random.choice(examples)
        prompt = PROMPT + '\n' + random_example + '\n\n' + '### 生成的代码\n'
        data.append([str(i) + '_' + str(rank), prompt])
    data_cur_rank = dist_split_files(data)
    results = multi_process_with_append(obtain_result, data_cur_rank, 
                                        num_workers=16)
    dump_list_to_jsonl_file(output_file, results)