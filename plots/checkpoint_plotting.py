import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dir_prefix",
        type=str,
        help="Prefix for the directory to get data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/saves",
        help="Directory to save the plots.",
    )
    args = parser.parse_args()
    dir_prefix = Path(args.dir_prefix)
    output_dir = Path(args.output_dir)
    
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    