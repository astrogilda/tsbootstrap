#!/usr/bin/env python3
"""Pre-commit hook to sync docs/requirements.txt with pyproject.toml."""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the sync script and stage changes if any."""
    # Get the repository root
    repo_root = Path(__file__).resolve().parent.parent.parent

    # Run the update script
    update_script = repo_root / ".github" / "scripts" / "update_requirements.py"
    if not update_script.exists():
        print(f"Error: Update script not found at {update_script}")
        return 1

    # Run the update script
    try:
        subprocess.run([sys.executable, str(update_script)], check=True)  # noqa: S603
    except subprocess.CalledProcessError:
        print("Error: Failed to run update_requirements.py")
        return 1

    # Check if docs/requirements.txt was modified
    docs_req = repo_root / "docs" / "requirements.txt"
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", str(docs_req)],  # noqa: S603, S607
            capture_output=True,
            text=True,
            check=True,
        )

        if result.stdout.strip():
            # File was modified, add it to the commit
            subprocess.run(["git", "add", str(docs_req)], check=True)  # noqa: S603, S607
            print("✅ docs/requirements.txt was updated and staged")
            return 0
        else:
            print("✅ docs/requirements.txt is already in sync")
            return 0

    except subprocess.CalledProcessError as e:
        print(f"Error checking git status: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
