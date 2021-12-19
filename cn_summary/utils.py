import os
import re

def get_src_path(path):
    prefix = os.environ.setdefault(
        'CN_SUMMARY_MODEL_DIR', 'D:\gitWarehouse\cn-summary\model'
    ) 
    return os.path.abspath(os.path.join(prefix, path))

def text_legal(func):
    def wrapper(text, *args, **kwargs):
        text = re.sub('\s', '', text)
        seperators = ['。', '!', '！', '？', '?']
        if text[-1] not in seperators:
            text += '。'
        return func(text, *args, **kwargs)
    return wrapper