from setuptools import setup, find_packages

setup(
    name='uqlab',
    version='0.1.0',
    url="https://github.com/AdamThomas-Mitchell/uqlab",
    author='Adam Thomas-Mitchell',
    author_email='atmitchell017@gmail.com',
    description='A package for uncertainty quantification and post-hoc calibration',
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
)
