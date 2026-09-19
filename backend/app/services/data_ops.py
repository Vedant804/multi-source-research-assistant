"""Whitelisted pandas operations. The LLM only chooses parameters; it never writes code."""
import pandas as pd

from app.schemas import DataOp


def read_table(path: str, table: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
    else:
        sheet = table.split("::", 1)[1] if "::" in table else 0
        df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c) for c in df.columns]
    return df


def inspect_tables(path: str, filename: str, ext: str) -> list[dict]:
    if ext == ".csv":
        frames = {filename: read_table(path, filename)}
    else:
        sheets = pd.ExcelFile(path).sheet_names
        frames = {f"{filename}::{s}": read_table(path, f"{filename}::{s}") for s in sheets}

    tables = []
    for name, df in frames.items():
        tables.append(
            {
                "name": name,
                "rows": int(len(df)),
                "columns": [{"name": str(c), "dtype": str(t)} for c, t in df.dtypes.items()],
                "sample": df.head(3).astype(str).to_dict(orient="records"),
            }
        )
    return tables


def _col(df: pd.DataFrame, name: str | None) -> str:
    if not name or name not in df.columns:
        raise ValueError(f"Unknown column {name!r}. Available: {list(df.columns)}")
    return name


def execute_op(df: pd.DataFrame, op: DataOp) -> str:
    n = max(1, min(op.n, 50))

    if op.op == "describe":
        sub = df[[_col(df, op.column)]] if op.column else df
        desc = sub.describe(include="all").T.round(3).head(40)
        return f"Rows: {len(df)}, Columns: {len(df.columns)}\n\n{desc.to_markdown()}"

    if op.op == "value_counts":
        c = _col(df, op.column)
        out = df[c].value_counts(dropna=False).head(n).rename_axis(c).reset_index(name="count")
        return out.to_markdown(index=False)

    if op.op == "groupby_agg":
        g = _col(df, op.group_by)
        if op.agg == "count" and not op.column:
            res = df.groupby(g).size().rename("count")
        else:
            c = _col(df, op.column)
            res = df.groupby(g)[c].agg(op.agg)
        res = res.sort_values(ascending=op.ascending).head(n).round(3)
        return res.reset_index().to_markdown(index=False)

    if op.op == "top_n":
        c = _col(df, op.column)
        return df.sort_values(c, ascending=op.ascending).head(n).to_markdown(index=False)

    if op.op == "correlation":
        num = df.select_dtypes("number")
        if num.shape[1] < 2:
            raise ValueError("Need at least two numeric columns for correlation")
        return num.iloc[:, :12].corr().round(3).to_markdown()

    if op.op == "time_trend":
        d, c = _col(df, op.date_column), _col(df, op.column)
        dates = pd.to_datetime(df[d], errors="coerce")
        tmp = pd.DataFrame({"period": dates.dt.to_period(op.freq), "value": df[c]}).dropna(subset=["period"])
        res = tmp.groupby("period")["value"].agg(op.agg).tail(36).round(3).reset_index()
        res["period"] = res["period"].astype(str)
        return res.to_markdown(index=False)

    raise ValueError(f"Unsupported operation {op.op}")