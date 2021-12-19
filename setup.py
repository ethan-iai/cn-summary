from setuptools import setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='cn_summary',
    version='0.0.1',
    
    description='Package for generation of Chinese text summary',
    
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='',
    
    author='',
    author_email='',
    
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
    ],

    packages=['cn_summary'],

    python_requires='>=3.6, <4',

    install_requires=[],   
)