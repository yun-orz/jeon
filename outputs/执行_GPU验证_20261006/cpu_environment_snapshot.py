# -*- coding: utf-8 -*-
"""只读核对原CPU环境版本、导入位置和关键源码/二进制指纹。"""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import torch


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def snapshot():
    package = Path(torch.__file__).parent
    files = [package/'__init__.py', package/'version.py', Path(torch._C.__file__), package/'lib/torch_cpu.dll']
    distribution = metadata.distribution('torch')
    files += [Path(distribution.locate_file(item)) for item in distribution.files if str(item).endswith('.dist-info/METADATA')]
    return dict(torch_version=torch.__version__, cuda_runtime=torch.version.cuda, torch_file=str(package),
                versions={name: metadata.version(name) for name in ['torch', 'sympy', 'numpy', 'scipy', 'matplotlib']},
                files_sha256={str(path): sha(path) for path in files})


if __name__ == '__main__':
    print(json.dumps(snapshot(), ensure_ascii=False, indent=2))
