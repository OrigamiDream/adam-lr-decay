from setuptools import setup, find_packages

setup(
    name='adam-lr-decay',
    packages=find_packages(exclude=[]),
    version='0.0.1',
    license='MIT',
    description='Adam Layer-wise LR Decay',
    author='OrigamiDream',
    author_email='hello@origamidream.me',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/OrigamiDream/adam-lr-decay',
    python_requires='>=3.8.0',
    extras_require={
        'gpu': ['tensorflow>=2.11'],
        'cpu': ['tensorflow-cpu>=2.11']
    },
    install_requires=[
        'packaging'
    ],
    keywords=[
        'machine learning',
        'deep learning',
        'tensorflow',
        'optimizers',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8'
    ]
)
