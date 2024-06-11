from setuptools import setup, find_packages

setup(
    name='hc_refloco',
    version='1.0',
    packages=find_packages(),
    py_modules=['dataloader', 'evaluation', 'overlaps'],
    install_requires=[
        # Add any dependencies your package needs here
        'torch>=2.1.0', 
        'datasets>=2.19.0',
        'Pillow>=10.3.0', 
        'statistics>=1.0.3.5', 
        'pandas>=2.2.2'
    ],
    entry_points={
        'console_scripts': [
            # Add any command line scripts here
            # e.g., 'mycommand = mypackage.module:function'
        ],
    },
    author='Jinjing Zhao',
    description='Dataloader and evaluation of several LMMs on HC-RefLoCo benchmark.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ZhaoJingjing713/HC-RefLoCo',  # Replace with your package's URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    license='MIT',
    python_requires='>=3.6',
)
