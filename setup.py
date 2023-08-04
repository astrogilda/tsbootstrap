from pathlib import Path

from setuptools import find_packages, setup

if __name__ == "__main__":
    with Path("./requirements.in").open("r") as f:
        reqs = f.read().split("\n")

    with Path("./requirements-dev.in").open("r") as f:
        dev_reqs = [
            req for req in f.read().split("\n") if not req.startswith("-c ")
        ]

    setup(
        name="ts_bs",
        author="Sankalp Gilda",
        author_email="sankalp.gilda@gmail.com",
        description="A Python package to generate bootstrapped time series",
        long_description=Path("README.md").open("r").read(),
        long_description_content_type="text/markdown",
        license="MIT",
        license_file="LICENSE",
        install_requires=reqs,
        python_requires=">=3.8",
        version="0.0.1",
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        zip_safe=False,
        include_package_data=True,
        extras_require={"dev": dev_reqs},
        options={"bdist_wheel": {"universal": True}},
        package_data={"": ["py.typed"]},
        classifiers=[
            "Development Status :: 1 - Planning",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
        ],
        platforms=["unix", "linux", "osx", "cygwin", "win32"],
    )
