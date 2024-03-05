from pathlib import Path

import tomlkit


def update_requirements():
    with Path("pyproject.toml").open("r") as pyproject:
        data = tomlkit.parse(pyproject.read())
        # Convert tomlkit containers to Python dicts
        dependencies = dict(data["project"]["dependencies"])
        dev_dependencies = dict(data["project"]["dev-dependencies"])

        with Path("docs/requirements.txt").open("w") as requirements:
            for dep, version in dependencies.items():
                if dep != "python":
                    requirements.write(f"{dep}{version}\n")
            for dev_dep, version in dev_dependencies.items():
                requirements.write(f"{dev_dep}{version}\n")


if __name__ == "__main__":
    update_requirements()
