import pandas as pd

data = (
    pd.read_csv(
        "data/monolingual/oci/oci-fr_web_2020_10K-sentences.txt",
        quoting=3,
        sep="\t",
        usecols=[1],
        header=None,
    )
    .iloc[:, 0]
    .tolist()
)

print(data)
