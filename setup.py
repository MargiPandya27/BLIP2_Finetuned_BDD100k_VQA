from pathlib import Path

from setuptools import find_packages, setup


HYPEN_E_DOT = "-e ."
def read_requirements() -> list[str]:
    requirements_path = Path(__file__).parent / "requirements.txt"
    requirements = []
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements


def read_readme() -> str:
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return ""


setup(
    name="blip2-vqa",
    version="0.1.0",
    description="BLIP-2 fine-tuning for BDD100K risk assessment VQA",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(exclude=["scripts*", "inference*", "tests*"]), # Exclude scripts, inference, and tests   
    install_requires=read_requirements(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
