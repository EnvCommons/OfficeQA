"""Download and prepare OfficeQA data for the OpenReward environment.

Downloads question CSVs and transformed Treasury Bulletin text files from the
databricks/officeqa GitHub repository.

Produces:
  - officeqa_pro.csv, officeqa_full.csv — question CSVs (for env server)
  - bucket_data/officeqa/documents/     — 696 .txt files (for sandbox bucket)

Usage:
    uv run python prepare_data.py
"""

import os
import urllib.request
import zipfile
from pathlib import Path

GITHUB_RAW = "https://raw.githubusercontent.com/databricks/officeqa/main"
OUTPUT_DIR = Path(__file__).parent

TRANSFORMED_ZIP_URL = (
    "https://github.com/databricks/officeqa/raw/refs/heads/main/"
    "treasury_bulletins_parsed/transformed/treasury_bulletins_transformed.zip"
)


def download_file(url: str, dest: Path) -> None:
    """Download a file from URL to destination path."""
    if dest.exists():
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading: {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved: {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")


def download_csvs() -> None:
    """Download question CSV files."""
    print("Downloading question CSVs...")
    for name in ["officeqa_pro.csv", "officeqa_full.csv"]:
        download_file(f"{GITHUB_RAW}/{name}", OUTPUT_DIR / name)


def download_and_extract_documents() -> None:
    """Download and extract transformed Treasury Bulletin text files."""
    docs_dir = OUTPUT_DIR / "bucket_data" / "officeqa" / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    existing_txt = list(docs_dir.glob("*.txt"))
    if len(existing_txt) > 600:
        print(f"  Documents already extracted: {len(existing_txt)} .txt files")
        return

    zip_path = OUTPUT_DIR / "treasury_bulletins_transformed.zip"

    print("Downloading transformed Treasury Bulletin documents...")
    download_file(TRANSFORMED_ZIP_URL, zip_path)

    print("  Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        # Extract all .txt files into the documents directory
        for member in zf.namelist():
            if member.endswith(".txt"):
                # Extract just the filename, ignoring any subdirectory structure
                filename = os.path.basename(member)
                if filename:
                    dest = docs_dir / filename
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())

    txt_count = len(list(docs_dir.glob("*.txt")))
    print(f"  Extracted {txt_count} .txt files to {docs_dir}")

    # Clean up zip
    zip_path.unlink()
    print("  Removed zip file")


def print_summary() -> None:
    """Print summary of downloaded data."""
    import pandas as pd

    print("\n=== Summary ===")

    for name in ["officeqa_full.csv", "officeqa_pro.csv"]:
        path = OUTPUT_DIR / name
        if path.exists():
            df = pd.read_csv(path)
            print(f"\n{name}:")
            print(f"  Tasks: {len(df)}")
            if "difficulty" in df.columns:
                for diff, count in df["difficulty"].value_counts().items():
                    print(f"  {diff}: {count}")

    docs_dir = OUTPUT_DIR / "bucket_data" / "officeqa" / "documents"
    if docs_dir.exists():
        txt_count = len(list(docs_dir.glob("*.txt")))
        print(f"\nDocuments: {txt_count} .txt files in {docs_dir}")


def main():
    print("=== OfficeQA Data Preparation ===\n")

    download_csvs()
    download_and_extract_documents()
    print_summary()

    print("\n=== Next Steps ===")
    print("1. Upload bucket_data/ contents to the OpenReward bucket")
    print("2. Create GeneralReasoning/OfficeQA namespace on openreward.ai")
    print("3. Run: pytest tests.py")


if __name__ == "__main__":
    main()
