from __future__ import annotations

"""ibge_io_reader
===================
Light‑weight helper to extract vectors and matrices from the IBGE supply‑use
(IO) workbooks bundled with the *iotbr* package.  Four aggregation levels are
supported (12, 20, 51 and 68).

The public API boils down to three convenience functions:

* ``read_var`` – load a single variable (vector, matrix or VA block)
* ``read_vars`` – load several variables at once and concatenate by column
* ``read_var_def`` – same as ``read_var`` but deflated to constant prices

Examples
--------
>>> df = read_var(year="2019", level="68", var="PT", unit="t")
>>> bundle = read_vars(year="2019", level="20", vars_=["OT_pm", "IPI"])
"""

from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Dict, List, Literal, Union

import io

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Metadata tables
# ---------------------------------------------------------------------------

Level = Literal["12", "20", "51", "68"]
Unit = Literal["t", "t-1"]

_LEVEL_META: Dict[Level, Dict[str, Union[int, str]]] = {
    "12": {"path": "iotbr.IBGE.nivel_12_2000_2021_xls", "rows": 12, "rows_va": 14},
    "20": {"path": "iotbr.IBGE.nivel_20_2010_2021_xls", "rows": 20, "rows_va": 14},
    "51": {"path": "iotbr.IBGE.nivel_51_2000_2021_xls", "rows": 107, "rows_va": 12},
    "68": {"path": "iotbr.IBGE.nivel_68_2010_2021_xls", "rows": 128, "rows_va": 14},
}

_Y_INDEX: Dict[Level, int] = {"51": 0, "12": 1, "20": 1, "68": 1}

# ---------------------------------------------------------------------------
# Dictionary (metadata) helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _load_dictionary() -> pd.DataFrame:
    """Load *dictionary_1.csv* only once (cached)."""
    with resources.open_binary("iotbr.IBGE", "dictionary_1.csv") as fp:
        return pd.read_csv(io.BytesIO(fp.read()))


def _var_entry(var: str, reference: int) -> pd.Series:
    dic = _load_dictionary()
    try:
        return dic.query("reference == @reference and var == @var").iloc[0]
    except IndexError as exc:
        raise KeyError(f"Variable '{var}' not found for reference {reference}") from exc


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _reference_year(year: int) -> int:
    """Return reference year for a sheet (2010 or 2000)."""
    return 2010 if year >= 2010 else 2000


def _excel_path(level: Level) -> str:
    return _LEVEL_META[level]["path"]  # type: ignore[return-value]


def _n_rows(level: Level, value_added: bool = False) -> int:
    key = "rows_va" if value_added else "rows"
    return int(_LEVEL_META[level][key])  # type: ignore[arg-type]


def _y_index_col(level: Level) -> int:
    return _Y_INDEX[level]


def _var_type(var: str, reference: int = 2010) -> str:
    return _var_entry(var, reference)["type"]  # type: ignore[return-value]


def _var_col(var: str, level: Level, year: int) -> int:
    ref = _reference_year(year)
    col = _var_entry(var, ref)[f"Ncol_{level}"]
    if col == "x":
        raise ValueError(
            f"Variable '{var}' not available for year={year} at level {level}.")
    return int(col)


def _var_sheet(var: str, reference: int = 2010) -> str:
    return _var_entry(var, reference)["sheet"]  # type: ignore[return-value]


def _table_name(var: str, unit: Unit, reference: int = 2010) -> str:
    """Return *tab1* … *tab4* depending on variable/unit combination."""
    tab = _var_entry(var, reference)["table"]
    if tab == "tab1" and unit == "t":
        return "tab1"
    if tab == "tab2" and unit == "t":
        return "tab2"
    if tab == "tab1" and unit == "t-1":
        return "tab3"
    return "tab4"


# ---------------------------------------------------------------------------
# IO workbook access
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def _read_excel(year: int, level: Level, var: str, unit: Unit) -> pd.DataFrame:
    """Read a single sheet from an embedded workbook (cached)."""
    tab = _table_name(var, unit)
    sheet = _var_sheet(var)

    file_name = f"{level}_{tab}_{year}.xls"
    with resources.open_binary(_excel_path(level), file_name) as fp:
        bio = io.BytesIO(fp.read())
    return pd.read_excel(bio, sheet_name=sheet, engine="xlrd")


# ---------------------------------------------------------------------------
# Public loading helpers
# ---------------------------------------------------------------------------

def _prepare_header(df: pd.DataFrame, level: Level, value_added: bool = False):
    rows = _n_rows(level, value_added)
    col_y = _y_index_col(level)
    y_index = df.iloc[4 : 4 + rows, col_y]

    if value_added:
        x_index = df.iloc[2, 1 : 1 + int(level)]
    else:
        x_index = df.iloc[2, col_y + 1 : col_y + 1 + int(level)]
    return y_index, x_index


def read_vector(year: Union[int, str], level: Level, var: str, unit: Unit = "t") -> pd.DataFrame:
    year = int(year)
    df = _read_excel(year, level, var, unit)
    col = _var_col(var, level, year)
    rows = _n_rows(level)
    y_index, _ = _prepare_header(df, level)

    # Special cases where the IO workbook splits the variable across several columns
    if level == "51" and var == "X_bens_serv":
        values = df.iloc[4 : 4 + rows, col : col + 2].sum(axis=1)
    elif level == "51" and var == "M_bens_serv" and year < 2010:
        values = df.iloc[4 : 4 + rows, col : col + 3].sum(axis=1)
    elif level == "12" and var == "X_bens_serv" and year < 2010:
        values = df.iloc[4 : 4 + rows, col : col + 2].sum(axis=1)
    elif level == "12" and var == "M_bens_serv" and year < 2010:
        values = df.iloc[4 : 4 + rows, col : col + 3].sum(axis=1)
    else:
        values = df.iloc[4 : 4 + rows, col]

    out = pd.DataFrame(values.values, index=y_index, columns=[var])
    out.index.name = "produtos"
    return out


def read_matrix(year: Union[int, str], level: Level, var: str, unit: Unit = "t") -> pd.DataFrame:
    year = int(year)
    df = _read_excel(year, level, var, unit)
    col_init = _var_col(var, level, year)
    col_final = col_init + int(level)
    rows = _n_rows(level)

    y_index, x_index = _prepare_header(df, level)
    data = df.iloc[4 : 4 + rows, col_init:col_final]
    data.columns = x_index.values
    data.index = y_index
    data.index.name = "produtos"
    return data


def read_va(year: Union[int, str], level: Level, var: str, unit: Unit = "t") -> pd.DataFrame:
    year = int(year)
    df = _read_excel(year, level, var, unit)
    col_init = _var_col(var, level, year)
    col_final = col_init + int(level)
    rows = _n_rows(level, value_added=True)

    y_index, x_index = _prepare_header(df, level, value_added=True)
    data = df.iloc[4 : 4 + rows, col_init:col_final]
    data.columns = x_index.values
    data.index = y_index
    data.index.name = "setor"
    return data.T  # transpose so sectors are rows like other matrices


# ---------------------------------------------------------------------------
# High‑level wrappers
# ---------------------------------------------------------------------------

def read_var(
    year: Union[int, str] = 2019,
    level: Level = "68",
    var: str = "PT",
    unit: Unit = "t",
):
    year = int(year)
    kind = _var_type(var)
    if kind == "vector":
        return read_vector(year, level, var, unit)
    if kind == "matrix":
        return read_matrix(year, level, var, unit)
    # otherwise value‑added block
    return read_va(year, level, var, unit)


def read_vars(
    year: Union[int, str] = 2019,
    level: Level = "68",
    vars_: Union[str, List[str]] = "PT",
    unit: Unit = "t",
):
    if isinstance(vars_, str):
        vars_ = [vars_]
    frames = [read_var(year, level, v, unit) for v in vars_]
    return pd.concat(frames, axis=1)


# Optional deflator wrapper (depends on user‑provided *deflate* module)

def read_var_def(
    year: Union[int, str] = 2019,
    level: Level = "68",
    var: str = "PT",
    unit: Unit = "t",
    reference_year: int = 2011,
):
    from . import deflate as _deflate  # local import to avoid hard dependency

    df_pc = read_var(year, level, var, unit)
    deflators = _deflate.deflators_df(reference_year)
    factor = deflators.loc[deflators["year"] == int(year), "def_cum_pro"].iat[0]
    return df_pc / factor


__all__ = [
    "read_var",
    "read_vars",
    "read_var_def",
    "read_vector",
    "read_matrix",
    "read_va",
]
