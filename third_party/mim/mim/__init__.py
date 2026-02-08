# Copyright (c) OpenMMLab. All rights reserved.

# NOTE: We could got AssertionError when importing pip before
# setuptools. A workaround is to import setuptools first and filter
# warnings that are caused by setuptools replacing distutils.
# Related issues:
# - https://github.com/pypa/setuptools/issues/3621
# - https://github.com/open-mmlab/mmclassification/issues/1343
try:
    import warnings

    import setuptools  # noqa: F401
    warnings.filterwarnings('ignore', 'Setuptools is replacing distutils')
except ImportError:
    pass

from .commands import (
    download,
    get_model_info,
    gridsearch,
    install,
    list_package,
    run,
    test,
    train,
    uninstall,
)

__all__ = [
    'download', 'install', 'list_package', 'download', 'get_model_info',
    'install', 'uninstall', 'train', 'test', 'run', 'gridsearch'
]
