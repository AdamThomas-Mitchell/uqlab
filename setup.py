from setuptools import setup, find_packages

setup(
    name='uqlab',
    version='0.1.1',
    url="https://github.com/AdamThomas-Mitchell/uqlab",
    author='Adam Thomas-Mitchell',
    author_email='atmitchell017@gmail.com',
    description='A package to investigate uncertainty quantification for machine learning force fields',
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    include_package_data=True,
    package_data={
        '': [
            'data/waterDimer/*.csv',
            'data/glycine/train/*.csv',
            'data/glycine/test/*.csv'
        ]
    },
)
