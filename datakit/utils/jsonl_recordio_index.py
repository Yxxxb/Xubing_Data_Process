import time

from .distributed import delete_folder
from .generate_index import generate_index
from .jsonl_to_recordio import jsonl_to_recordio
from .raw_to_jsonl import raw_to_jsonl


def jsonl_to_recordio_to_index(raw_root_list: str,
                               alert_path: str,
                               index_file_root: str,
                               index_file_name: str,
                               base_train_config: str,
                               train_config_file: str,
                               model_save_folder: str,
                               base_datasets_file: str = None,
                               num_epochs: int = 2,
                               micro_batch_tokens_limit: int = 32768,
                               batch_size: int = 64,
                               batch_strategy: int = 'by_token',
                               maxframes: int = 64,
                               rename: str = None,
                               is_pretrain_decay: bool = False,
                               **kwargs) -> None:
    """
    Convert JSONL to RecordIO then to RecordIO index.

    Args:
        raw_root_list (str): String list of raw root folders.
                             Separated by comma.
        alert_path (str): Path to alert folder.
        index_file_root (str): Root folder of index file.
        index_file_name (str): Name of index file.
        base_train_config (str): Base training configuration file.
        train_config_file (str): Training configuration file.
        model_save_folder (str): Folder to save model.
        base_datasets_file (str): Base datasets file.
        num_epochs (int): Number of epochs.
        micro_batch_tokens_limit (int): Micro batch tokens limit.
        batch_size (int): Batch size.
        batch_strategy (str): Batch strategy.
        maxframes (int): Maximum number of frames.
        rename (str): string of renamed saving directory.
        **kwargs: Additional keyword arguments for `raw_to_jsonl`.

    Returns:
        None.
    """
    output_jsonl_list = raw_to_jsonl(raw_root_list,
                                     alert_path=alert_path,
                                     maxframes=maxframes,
                                     rename=rename,
                                     **kwargs)

    output_jsonl_str = ','.join(output_jsonl_list)
    print(output_jsonl_str)
    output_recordio_list = jsonl_to_recordio(output_jsonl_str,
                                             alert_path=alert_path,
                                            is_pretrain_decay=is_pretrain_decay,)

    output_recordio_str = ','.join(output_recordio_list)
    print(output_recordio_str)

    if index_file_name is not None:
        generate_index(root_list=output_recordio_str,
                       base_datasets_file=base_datasets_file,
                       index_file_root=index_file_root,
                       index_file_name=index_file_name,
                       base_train_config=base_train_config,
                       train_config_file=train_config_file,
                       model_save_folder=model_save_folder,
                       num_epochs=num_epochs,
                       seq_length=micro_batch_tokens_limit,
                       batch_size=batch_size,
                       alert_path=alert_path)
    # sleep for 1 min
    time.sleep(60)
    delete_folder('jsonl', alert_path)
    delete_folder('recordio', alert_path)
    delete_folder('recordio_index', alert_path)
