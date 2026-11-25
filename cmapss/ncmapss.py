from dataclasses import dataclass
from pathlib import Path
 
import h5py
import numpy as np
import pandas as pd
 
# W    (4)  altitude, Mach, throttle angle, fan inlet temperature
# X_s (14)  physical sensors
# X_v (14)  virtual sensors, not measurable on a real engine
# T   (10)  health parameters, RUL is derived from these
# A    (4)  unit, cycle, flight class, health state
# Y    (1)  RUL in cycles
 
ROOT = Path("data/raw/NCMAPSS")
 
NAMES = ("DS01", "DS02", "DS03")
 
FEATURE_SETS = {
    "observable": ("W", "X_s"),
    "oracle": ("W", "X_s", "X_v", "T"),
}
 
AUX = ("unit", "cycle", "Fc", "hs")
 
DTYPES = {"unit": "int32", "cycle": "int32", "Fc": "int8", "hs": "int8"}
 
 
# Container for Dev and Test split of Subdatasets 
@dataclass
class NCmapssSplit:
    name: str
    split: str
    frame: pd.DataFrame
    feature_names: list[str]
    stride: int
 
 
def _get_path(name: str) -> Path:
    if name not in NAMES:
        raise ValueError
 
    return ROOT / f"N-CMAPSS_{name}.h5"
 
# Core function to get data from .h5 files
def load(
    name: str, split: str, features: str = "observable", stride: int = 10
) -> NCmapssSplit:
    if split not in ("dev", "test"):
        raise ValueError
    if stride < 1:
        raise ValueError
 
    groups = FEATURE_SETS[features]
    columns = {}
 
    with h5py.File(_get_path(name), "r") as file:
        # every n-th sample, sliced on read ->  full array never in memory
        sel = slice(None, None, stride) # Downsampling
 
        # get features we want from h5 files
        for g in groups:
            block = file[f"{g}_{split}"][sel]
            names = [v.decode() for v in np.asarray(file[f"{g}_var"]).ravel()]
            for i, col in enumerate(names):
                columns[col] = block[:, i]
 
        aux = file[f"A_{split}"][sel]
        for i, col in enumerate(AUX):
            columns[col] = aux[:, i]
 
        columns["RUL"] = file[f"Y_{split}"][sel].ravel()
 
    frame = pd.DataFrame(columns).astype(DTYPES)
    feature_names = [c for c in frame.columns if c not in AUX and c != "RUL"]
 
    return NCmapssSplit(name, split, frame, feature_names, stride)
 
 
def load_flights(name: str, split: str) -> pd.DataFrame:
    # only the aux and label groups, so no sensor channel is read and no stride is needed
    if split not in ("dev", "test"):
        raise ValueError
 
    with h5py.File(_get_path(name), "r") as file:
        aux = file[f"A_{split}"][:]
        rul = file[f"Y_{split}"][:].ravel()
 
    frame = pd.DataFrame({col: aux[:, i] for i, col in enumerate(AUX)}).astype(DTYPES)
    frame["RUL"] = rul
 
    flights = frame.groupby(["unit", "cycle"], as_index=False).agg(
        Fc=("Fc", "first"),
        RUL=("RUL", "first"),
        samples=("RUL", "size"),
    )
    flights["duration_h"] = flights["samples"] / 3600.0
 
    return flights