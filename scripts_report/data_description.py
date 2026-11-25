from pathlib import Path

import pandas as pd

from cmapss.features import get_data


def table_metadata() -> None:
    rows = []
    for subdataset in ("DS01", "DS02", "DS03"):
        data = get_data(subdataset)
        for split, meta_key in (("train", "train"), ("eval", "test")):
            rows.append({
                "Dataset": f"{subdataset} {split}",
                r"\#Units": data["meta"][meta_key]["unit"].nunique(),
                "Flights": len(data[split]["y"]),
                "Flight Classes": ", ".join(
    str(fc) for fc in sorted(data["meta"][meta_key]["Fc"].unique())),
            })

    df = pd.DataFrame(rows)
    Path("report/tables").mkdir(parents=True, exist_ok=True)
    with open("report/tables/table_metadata.tex", "w") as f:
        f.write(df.to_latex(index=False))

  
if __name__ == "__main__":
  table_metadata()
    