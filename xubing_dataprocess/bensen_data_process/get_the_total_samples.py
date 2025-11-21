from datakit.utils.distributed import dist_split_files, get_distributed_env
from datakit.utils.mp import multi_process_with_append
from datakit.utils.files import read_jsonl_file, find_all_files
from datakit.utils.distributed import barrier_all_processes
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--root-folder', type=str, required=True)
parser.add_argument('--task-name', type=str, required=True)



def get_total_samples(file_path):
    data = read_jsonl_file(file_path)
    return len(data)



if __name__ == '__main__':
    args = parser.parse_args()
    root_folder = args.root_folder
    _, rank, _ = get_distributed_env()
    all_jsonls = find_all_files(root_folder, 'jsonl')
    jsonls_cur_rank = dist_split_files(all_jsonls)
    total_samples_cur_rank = multi_process_with_append(get_total_samples, 
                                                       jsonls_cur_rank,
                                                       128)
    total_samples = sum(total_samples_cur_rank)
    output_file = os.path.join(root_folder, f'total_samples_{rank}.txt')
    with open(output_file, 'w') as f:
        f.write(str(total_samples))
    barrier_all_processes(args.task_name)

