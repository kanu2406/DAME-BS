from setuptools import setup, find_packages

setup(
    name="dame_ts",
    version="0.1.0",
    description="DAME-TS: Differentially Private Mean Estimation via Ternary Search",
    author="Kanupriya Jain",
    author_email="k.jain@criteo.com",
    url="https://github.com/kanu2406/DAME-TS",  
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "tqdm"
    ],
    python_requires='>=3.7',
)
