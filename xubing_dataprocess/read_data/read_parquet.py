import pandas as pd
import re

from datakit.utils.files import (dump_list_to_jsonl_file, find_all_files, mem_efficient_read_jsonl_file,
                                 read_jsonl_file)
from datakit.utils.image import decode_base64_image_to_pil

pdf = '/mnt/cephfs/zhonyinzhao/mount/vlmdatasets/pt/PDF/scihub/PR/214_91.jsonl'
data = '/mnt/cephfs/zhonyinzhao/mount/vlmdatasets/pt/PDF/scihub/data/214_91.jsonl'
