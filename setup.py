from setuptools import setup, find_packages

def get_requirements(file_path):
    with open(file_path) as f:
        return f.read().splitlines()

setup(
    name="Components",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=get_requirements("requirements.txt"),
)
