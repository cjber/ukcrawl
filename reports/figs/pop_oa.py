import duckdb
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

conn = duckdb.connect(database=":memory:", read_only=False)
conn.sql("CREATE TABLE pop_oa AS SELECT * FROM './data/raw/oapop21/ts001.parquet'")
conn.sql(
    """
    CREATE TABLE ukcrawl AS 
        SELECT *, len(all_postcodes) as pc_count 
        FROM './data/out/pc_year/*.parquet'
    """
)


conn.sql(
    "CREATE TABLE msoa2lad AS SELECT * FROM './data/raw/Middle_Layer_Super_Output_Area_(2021)_to_LAD_(April_2023)_Lookup_in_England_and_Wales.csv'"
)

df = conn.sql(
    "SELECT count(url) AS count, LAD23CD FROM ukcrawl JOIN msoa2lad ON ukcrawl.MSOA11 = msoa2lad.MSOA21CD GROUP BY LAD23CD"
).df()

df = gpd.read_file(
    "./data/raw/Local_Authority_Districts_December_2023_Boundaries_UK_BUC_-9001365094975426864.gpkg"
).merge(df, left_on="LAD23CD", right_on="LAD23CD", how="left")

fig, ax = plt.subplots(figsize=(10, 10))
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="3%", pad=0.1)
cmap = plt.cm.viridis
num_quantiles = 5

quantiles = np.linspace(0, 1, num_quantiles + 1)
quantile_values = df["count"].quantile(quantiles).values
cmap = plt.cm.viridis
norm = BoundaryNorm(quantile_values, cmap.N)
df.plot(
    column="count",
    figsize=(10, 10),
    edgecolor="face",
    scheme="quantiles",
    ax=ax,
)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []
cbar = fig.colorbar(
    sm,
    cax=cax,
    orientation="horizontal",
    boundaries=quantile_values,
    ticks=quantile_values,
)
# cbar.set_label("Number of Websites")
cbar.ax.tick_params(labelsize=8)
ax.axis("off")
plt.show()
