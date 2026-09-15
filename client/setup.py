from setuptools import setup, find_packages

setup(
    name="iagi_gui",
    version="2.0.0",
    description="Клиентское приложение для системы оптимизации раскроя плоских деталей (ИАГИ)",
    author="IAGI Team",
    packages=find_packages(),
    install_requires=[
        "PyQt5>=5.15.0",
        "matplotlib>=3.5.0",
        "numpy>=1.21.0",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "iagi-client=main:main",
        ],
    },
)