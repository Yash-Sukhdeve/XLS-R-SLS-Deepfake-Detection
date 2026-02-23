from setuptools import setup, find_packages

setup(
    name='sls-asvspoof',
    version='1.0.0',
    description='Audio Deepfake Detection with XLS-R and SLS classifier',
    author='Qishan Zhang, Shuangbing Wen, Tao Hu',
    url='https://github.com/Yash-Sukhdeve/XLS-R-SLS-Deepfake-Detection',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.7,<3.8',
    # Note: torch, torchvision, torchaudio are installed via conda/requirements.txt
    # with CUDA support (+cu117). They are NOT listed here to avoid pip downgrading
    # to CPU-only versions. See environment.yml or requirements.txt for full deps.
    install_requires=[
        'librosa==0.9.1',
        'tensorboardX==2.5',
        'tensorboard==2.11.2',
        'pandas==1.3.5',
        'numpy==1.21.6',
        'scipy==1.7.3',
        'pyyaml==6.0.1',
        'tqdm',
        'matplotlib==3.5.3',
    ],
)
