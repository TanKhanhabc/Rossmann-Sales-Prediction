import argparse
from pathlib import Path

# Các thư mục của dự án Rossmann Sales Prediction
DIRECTORIES = [
    "backend/app/model",
    "backend/app/routers",
    "backend/app/services",
    "backend/tests",
    "frontend/src",
    "config",
    "data/raw",
    "data/processed",
    "data/merged",
    "data/external",
    "docs",
    "logs",
    "models",
    "notebooks/exploratory",
    "results",
    "scripts/preprocessing",
    "scripts/analysis",
]


def create_project_structure(root: Path):
    # Tạo thư mục gốc nếu chưa tồn tại
    root.mkdir(parents=True, exist_ok=True)

    # Tạo các thư mục con
    for directory in DIRECTORIES:
        path = root / directory
        path.mkdir(parents=True, exist_ok=True)
        print(f"[OK] {path}")

    print("\nProject structure created successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Create Rossmann ML project structure"
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root directory"
    )

    args = parser.parse_args()

    create_project_structure(args.root.resolve())


if __name__ == "__main__":
    main()