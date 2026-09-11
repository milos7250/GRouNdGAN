from pathlib import Path

import click
import pandas as pd


def tsv_to_csv(input: Path, output: Path):

    df = pd.read_csv(input, sep="\t", names=["TF", "target", "importance"])
    df.to_csv(output, index=False)
    
if __name__ == "__main__":
    @click.command()
    @click.option("--input", "-i", type=click.Path(exists=True, path_type=Path), required=True, help="Input tsv file path.")
    @click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Output csv file path.")
    def _main(input: Path, output: Path):
        tsv_to_csv(input, output)
    
    _main()