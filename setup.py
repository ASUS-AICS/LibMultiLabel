import setuptools

name = 'libmultilabel'
version = '0.1.11'
author = 'LibMultiLabel Team'
license = 'MIT License'
license_file = 'LICENSE'
description = 'A library for multi-label text classification'
long_description = 'See documentation here: https://www.csie.ntu.edu.tw/~cjlin/libmultilabel'
url = 'https://github.com/ASUS-AICS/LibMultiLabel'
project_urls = {
    'Bug Tracker' : 'https://github.com/ASUS-AICS/LibMultiLabel/issues',
    'Documentation' : 'https://www.csie.ntu.edu.tw/~cjlin/libmultilabel',
    'Source Code' : 'https://github.com/ASUS-AICS/LibMultiLabel/',
}
classifiers = [
    'Environment :: GPU :: NVIDIA CUDA :: 11.6',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
]

packages = setuptools.find_packages(
    exclude=['docs'],
)

install_requires = [
    'nltk',
    'pandas>1.3.0',
    'PyYAML',
    'tqdm',
]

python_requires = '>=3.7'

extras_require = {
    'all': [
        'pytorch-lightning==1.7.7',
        'torch>=1.13.1',
        'torchmetrics==0.10.3',
        'torchtext>=0.13.0',
        'transformers',
        'liblinear-multicore',
        'numba',
        'scikit-learn',
        'scipy',
    ],
    'nn': [
        'pytorch-lightning==1.7.7',
        'torch>=1.13.1',
        'torchmetrics==0.10.3',
        'torchtext>=0.13.0',
        'transformers',
    ],
    'linear': [
        'liblinear-multicore',
        'numba',
        'scikit-learn',
        'scipy',
    ],
    'nn-param-search': [
        'pytorch-lightning==1.7.7',
        'torch>=1.13.1',
        'torchmetrics==0.10.3',
        'torchtext>=0.13.0',
        'transformers',
        'bayesian-optimization',
        'optuna',
        'ray>=2.0.1',
        'ray[tune]',
        'grpcio==1.43.0',
    ],
}

if __name__ == '__main__':
    setuptools.setup(
        name=name,
        version=version,
        author=author,
        license=license,
        license_file=license_file,
        description=description,
        long_description=long_description,
        url=url,
        project_urls=project_urls,
        classifiers=classifiers,
        packages=packages,
        install_requires=install_requires,
        python_requires=python_requires,
        extras_require=extras_require,
    )