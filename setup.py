from setuptools import setup, find_packages

setup(
    name='vtp_lipreading',
    version='0.1.0',
    description='Sub-word level Lipreading with Visual Attention',
    author='prajwalkr',
    author_email='prajwalrenukanand@gmail.com',
    url='https://github.com/prajwalkr/vtp',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'vtp_lipreading': ['checkpoints/tokenizers/**/*'],
    },
    install_requires=[
        'ConfigArgParse==1.7',
        'decord==0.6.0',
        'einops==0.8.0',
        'huggingface_hub==0.26.5',
        'linear-attention-transformer==0.19.1',
        'local-attention==1.10.0',
        'numpy==1.24.4',
        'pandas==2.0.3',
        'Pillow==10.4.0',
        'scipy==1.10.1',
        'tokenizers==0.20.0',
        'torch',
        'torchvision',
        'tqdm==4.67.1',
        'transformers==4.46.3'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords='lipreading deep-learning visual-attention transformer',
    python_requires='>=3.6'
)
