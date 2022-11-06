import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

version = '0.0.3'

setuptools.setup(
    name='pytorch_histogram_matching',
    version=version,
    url='https://github.com/nemodleo/pytorch-histogram-matching',
    author='Hyun Park',
    author_email='nemod.leo@snu.ac.kr',
    license="MIT",
    description='pytorch implementation of histogram matching',
    keywords="hm color pytorch histogram matching",
    long_description=long_description,
    long_description_content_type='text/markdown', 
    python_requires=">= 3.6",
    packages=['pytorch_histogram_matching'],
    install_requires=[],
    include_package_data=True,
    zip_safe=False,
    package_data={},
    classifiers=[
        'License :: OSI Approved :: MIT License'
    ] 
)