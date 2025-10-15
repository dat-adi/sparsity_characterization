import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Sparsity Evaluation

    This notebook is supposed to be a visualization of the sparsities for the wanda repository. We should have the following here:
    - [ ] Histograms of feature relevance at 50% sparsity in Wanda
    - [ ] Histograms of feature relevance at 50% sparsity in SparseGPT
    """
    )
    return


@app.cell
def _():
    import duckdb

    DATABASE_URL = "./sparsity.db"
    engine = duckdb.connect(DATABASE_URL, read_only=False)
    return (engine,)


@app.cell
def _(engine, mo, wanda_0_5_down_proj):
    _df = mo.sql(
        f"""
        SELECT * FROM wanda_0_5_down_proj
        """,
        engine=engine
    )
    return


if __name__ == "__main__":
    app.run()
