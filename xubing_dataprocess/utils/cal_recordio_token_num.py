import argparse

from datakit.utils.files import find_all_files, read_mmq_recordio
from datakit.utils.mp import multi_process_with_append
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input-folders',
                    type=str,
                    nargs='+',
                    required=True,
                    help='The input folders of the recordio files.')


def get_the_token_num_of_single_recordio(recordio_path: str) -> int:
    '''Get the token number of a single recordio file.

    Args:
        recordio_path (str): The path of the recordio file.

    Returns:
        int: The token number of the recordio file.
    '''
    data = read_mmq_recordio(recordio_path)
    total_token_num = 0
    for item in tqdm(data):
        total_token_num += len(item['text'])
    return total_token_num


if __name__ == '__main__':
    args = parser.parse_args()
    all_recordios = []
    for folder in args.input_folders:
        all_recordios.extend(find_all_files(folder, 'data.recordio'))
    total_token_nums = multi_process_with_append(
        get_the_token_num_of_single_recordio, all_recordios, 128)
    print(
        f'The total token number of the recordio files is {sum(total_token_nums)}'  # noqa
    )
