import os

def get_src_path(path):
    prefix = os.environ.setdefault(
        'CN_SUMMARY_MODEL_DIR', 'D:\gitWarehouse\cn-summary\model'
    ) 
    return os.path.abspath(os.path.join(prefix, path))

