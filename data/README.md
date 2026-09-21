# Data

Raw flight records are not included in this repository.

1. Download monthly U.S. on-time performance files from the Bureau of Transportation Statistics.
2. Create a local `raw/` directory here.
3. Place CSV files in `data/raw/`.
4. Run `python src/train.py --data-dir data/raw` from the repository root.

The repository's `.gitignore` prevents raw files from being committed.

