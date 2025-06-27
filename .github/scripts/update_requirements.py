from pathlib import Path

import tomlkit


def update_requirements():
    # Navigate up two levels to the root directory, then to 'pyproject.toml'
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"

    with Path(pyproject_path).open("r") as pyproject:
        data = tomlkit.parse(pyproject.read())

        # Get the dependencies as a list
        dependencies = data["project"]["dependencies"]  # type: ignore

        docs_dependencies = data["project"]["optional-dependencies"]["docs"]  # type: ignore

        requirements_path = Path(__file__).parent.parent.parent / "docs/requirements.txt"
        with Path(requirements_path).open("w") as requirements:
            for dep in dependencies:  # type: ignore
                if dep != "python":
                    # Directly write the dependency string to requirements.txt
                    requirements.write(f"{dep}\n")
            for docs_dep in docs_dependencies:  # type: ignore
                requirements.write(f"{docs_dep}\n")


if __name__ == "__main__":
    update_requirements()
