# pip install streamlit matplotlib seaborn plotly folium geopandas datashader altair shapely streamlit-folium streamlit run visualization_libraries_streamlit_app.py
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import folium
import geopandas as gpd
from shapely.geometry import Point, Polygon
import datashader as ds
import datashader.transfer_functions as tf

try:
    from streamlit_folium import st_folium
    HAS_STREAMLIT_FOLIUM = True
except Exception:
    HAS_STREAMLIT_FOLIUM = False

st.set_page_config(
    page_title="Python Visualization Libraries Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Python Visualization Libraries Explorer")
st.caption("Select a visualization library from the dropdown and explore at least five runnable examples with outputs.")

st.sidebar.header("Controls")
library = st.sidebar.selectbox(
    "Choose a library",
    ["Matplotlib", "Seaborn", "Plotly", "Folium", "GeoPandas", "Datashader", "Altair"],
)
np.random.seed(42)

# shared data
Tips = sns.load_dataset("tips")
Iris = sns.load_dataset("iris")
Gap = px.data.gapminder()
Gap2007 = Gap.query("year == 2007").copy()


def show_example(title, code_text, plot_func):
    with st.expander(title, expanded=True):
        c1, c2 = st.columns([1, 1])
        with c1:
            st.code(code_text.strip(), language="python")
        with c2:
            plot_func()


def render_map(m):
    if HAS_STREAMLIT_FOLIUM:
        st_folium(m, width=700, height=500)
    else:
        st.components.v1.html(m._repr_html_(), height=520)


# ----------------------------
# Matplotlib
# ----------------------------
def matplotlib_examples():
    st.subheader("Matplotlib")
    st.write("Classic static plotting with fine-grained control.")

    show_example(
        "Example 1: Line Plot",
        '''
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([1, 2, 3, 4, 5], [2, 4, 5, 7, 11], marker="o")
ax.set_title("Simple Line Plot")
ax.grid(True)
st.pyplot(fig)
''',
        lambda: (
            lambda fig, ax: (
                ax.plot([1, 2, 3, 4, 5], [2, 4, 5, 7, 11], marker="o"),
                ax.set_title("Simple Line Plot"),
                ax.set_xlabel("X"),
                ax.set_ylabel("Y"),
                ax.grid(True),
                st.pyplot(fig),
            )
        )(*plt.subplots(figsize=(6, 4))),
    )

    show_example(
        "Example 2: Bar Chart",
        '''
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["A", "B", "C", "D"], [12, 19, 7, 15])
ax.set_title("Bar Chart")
st.pyplot(fig)
''',
        lambda: (
            lambda fig, ax: (
                ax.bar(["A", "B", "C", "D"], [12, 19, 7, 15]),
                ax.set_title("Bar Chart"),
                ax.set_xlabel("Category"),
                ax.set_ylabel("Value"),
                st.pyplot(fig),
            )
        )(*plt.subplots(figsize=(6, 4))),
    )

    show_example(
        "Example 3: Scatter Plot",
        '''
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(np.random.randn(100), np.random.randn(100), alpha=0.75)
ax.set_title("Scatter Plot")
st.pyplot(fig)
''',
        lambda: (
            lambda fig, ax: (
                ax.scatter(np.random.randn(250), np.random.randn(250), alpha=0.75),
                ax.set_title("Scatter Plot"),
                ax.set_xlabel("Feature 1"),
                ax.set_ylabel("Feature 2"),
                st.pyplot(fig),
            )
        )(*plt.subplots(figsize=(6, 4))),
    )

    show_example(
        "Example 4: Histogram",
        '''
data = np.random.normal(loc=50, scale=10, size=500)
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(data, bins=20, edgecolor="black")
ax.set_title("Histogram")
st.pyplot(fig)
''',
        lambda: (
            lambda fig, ax: (
                ax.hist(np.random.normal(loc=50, scale=10, size=500), bins=20, edgecolor="black"),
                ax.set_title("Histogram"),
                ax.set_xlabel("Value"),
                ax.set_ylabel("Frequency"),
                st.pyplot(fig),
            )
        )(*plt.subplots(figsize=(6, 4))),
    )

    show_example(
        "Example 5: Subplots",
        '''
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot([1, 2, 3, 4], [1, 4, 2, 5], marker="o")
axes[1].bar(["Jan", "Feb", "Mar"], [10, 15, 8])
st.pyplot(fig)
''',
        lambda: (
            lambda fig, axes: (
                axes[0].plot([1, 2, 3, 4], [1, 4, 2, 5], marker="o"),
                axes[1].bar(["Jan", "Feb", "Mar"], [10, 15, 8]),
                axes[0].set_title("Line"),
                axes[1].set_title("Bar"),
                fig.suptitle("Multiple Subplots"),
                st.pyplot(fig),
            )
        )(*plt.subplots(1, 2, figsize=(10, 4))),
    )


# ----------------------------
# Seaborn
# ----------------------------
def seaborn_examples():
    st.subheader("Seaborn")
    st.write("Statistical visualizations with elegant defaults.")

    show_example(
        "Example 1: Scatter Plot",
        '''
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="sex", ax=ax)
st.pyplot(fig)
''',
        lambda: (
            lambda fig, ax: (
                sns.scatterplot(data=Tips, x="total_bill", y="tip", hue="sex", ax=ax),
                ax.set_title("Scatter Plot"),
                st.pyplot(fig),
            )
        )(*plt.subplots(figsize=(6, 4))),
    )

    show_example(
        "Example 2: Heatmap",
        '''
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(iris.drop(columns=["species"]).corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
''',
        lambda: (
            lambda fig, ax: (
                sns.heatmap(Iris.drop(columns=["species"]).corr(), annot=True, cmap="coolwarm", ax=ax),
                ax.set_title("Correlation Heatmap"),
                st.pyplot(fig),
            )
        )(*plt.subplots(figsize=(6, 4))),
    )

    show_example(
        "Example 3: Box Plot",
        '''
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=tips, x="day", y="total_bill", ax=ax)
st.pyplot(fig)
''',
        lambda: (
            lambda fig, ax: (
                sns.boxplot(data=Tips, x="day", y="total_bill", ax=ax),
                ax.set_title("Box Plot"),
                st.pyplot(fig),
            )
        )(*plt.subplots(figsize=(6, 4))),
    )

    show_example(
        "Example 4: Pair Plot",
        '''
pair = sns.pairplot(iris, hue="species", diag_kind="hist")
st.pyplot(pair.fig)
''',
        lambda: st.pyplot(sns.pairplot(Iris, hue="species", diag_kind="hist").fig),
    )

    show_example(
        "Example 5: Violin Plot",
        '''
fig, ax = plt.subplots(figsize=(6, 4))
sns.violinplot(data=tips, x="time", y="total_bill", ax=ax)
st.pyplot(fig)
''',
        lambda: (
            lambda fig, ax: (
                sns.violinplot(data=Tips, x="time", y="total_bill", ax=ax),
                ax.set_title("Violin Plot"),
                st.pyplot(fig),
            )
        )(*plt.subplots(figsize=(6, 4))),
    )


# ----------------------------
# Plotly
# ----------------------------
def plotly_examples():
    st.subheader("Plotly")
    st.write("Interactive charts with zoom, hover, and click interactions.")

    show_example(
        "Example 1: Interactive Scatter",
        '''
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
st.plotly_chart(fig, use_container_width=True)
''',
        lambda: st.plotly_chart(
            px.scatter(px.data.iris(), x="sepal_width", y="sepal_length", color="species", title="Interactive Scatter"),
            use_container_width=True,
        ),
    )

    show_example(
        "Example 2: Bar Chart",
        '''
df = tips.groupby("day", as_index=False)["total_bill"].mean()
fig = px.bar(df, x="day", y="total_bill")
st.plotly_chart(fig, use_container_width=True)
''',
        lambda: st.plotly_chart(
            px.bar(Tips.groupby("day", as_index=False, observed=True)["total_bill"].mean(), x="day", y="total_bill", title="Average Bill by Day"),
            use_container_width=True,
        ),
    )

    show_example(
        "Example 3: Box Plot",
        '''
fig = px.box(iris, x="species", y="petal_length")
st.plotly_chart(fig, use_container_width=True)
''',
        lambda: st.plotly_chart(
            px.box(Iris, x="species", y="petal_length", title="Box Plot"),
            use_container_width=True,
        ),
    )

    show_example(
        "Example 4: Bubble Chart",
        '''
fig = px.scatter(gap_2007, x="gdpPercap", y="lifeExp", size="pop", color="continent", hover_name="country")
st.plotly_chart(fig, use_container_width=True)
''',
        lambda: st.plotly_chart(
            px.scatter(
                Gap2007,
                x="gdpPercap",
                y="lifeExp",
                size="pop",
                color="continent",
                hover_name="country",
                title="Bubble Chart",
            ),
            use_container_width=True,
        ),
    )

    show_example(
        "Example 5: Graph Objects Line Plot",
        '''
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[3, 1, 4, 2], mode="lines+markers"))
st.plotly_chart(fig, use_container_width=True)
''',
        lambda: (
            lambda fig: (fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[3, 1, 4, 2], mode="lines+markers", name="Series")), fig.update_layout(title="Graph Objects Line Plot"), st.plotly_chart(fig, use_container_width=True))
        )(go.Figure()),
    )


# ----------------------------
# Folium
# ----------------------------
def folium_examples():
    st.subheader("Folium")
    st.write("Interactive maps built on Leaflet.")

    show_example(
        "Example 1: Marker Map",
        '''
m = folium.Map(location=[20.2961, 85.8245], zoom_start=11)
folium.Marker([20.2961, 85.8245], popup="Bhubaneswar").add_to(m)
render_map(m)
''',
        lambda: (
            lambda m: (folium.Marker([20.2961, 85.8245], popup="Bhubaneswar").add_to(m), render_map(m))
        )(folium.Map(location=[20.2961, 85.8245], zoom_start=11)),
    )

    show_example(
        "Example 2: Circle Map",
        '''
m = folium.Map(location=[20.2961, 85.8245], zoom_start=8)
folium.Circle([20.2961, 85.8245], radius=5000, popup="Coverage Area").add_to(m)
render_map(m)
''',
        lambda: (
            lambda m: (folium.Circle([20.2961, 85.8245], radius=5000, popup="Coverage Area").add_to(m), render_map(m))
        )(folium.Map(location=[20.2961, 85.8245], zoom_start=8)),
    )

    show_example(
        "Example 3: Route Line",
        '''
m = folium.Map(location=[20.2961, 85.8245], zoom_start=8)
folium.PolyLine([[20.2961, 85.8245], [20.4625, 85.8828], [21.9320, 86.7510]], color="blue").add_to(m)
render_map(m)
''',
        lambda: (
            lambda m: (folium.PolyLine([[20.2961, 85.8245], [20.4625, 85.8828], [21.9320, 86.7510]], color="blue").add_to(m), render_map(m))
        )(folium.Map(location=[20.2961, 85.8245], zoom_start=8)),
    )

    show_example(
        "Example 4: Multiple Markers",
        '''
m = folium.Map(location=[20.2961, 85.8245], zoom_start=7)
for lat, lon, name in [[20.2961, 85.8245, "Bhubaneswar"], [20.4625, 85.8828, "Cuttack"], [21.9320, 86.7510, "Baripada"]]:
    folium.Marker([lat, lon], popup=name).add_to(m)
render_map(m)
''',
        lambda: (
            lambda m: (
                [folium.Marker([lat, lon], popup=name).add_to(m) for lat, lon, name in [[20.2961, 85.8245, "Bhubaneswar"], [20.4625, 85.8828, "Cuttack"], [21.9320, 86.7510, "Baripada"]]],
                render_map(m)
            )
        )(folium.Map(location=[20.2961, 85.8245], zoom_start=7)),
    )

    show_example(
        "Example 5: Circle Markers",
        '''
m = folium.Map(location=[20.2961, 85.8245], zoom_start=7)
folium.CircleMarker([20.2961, 85.8245], radius=10, popup="City Center").add_to(m)
folium.CircleMarker([21.9320, 86.7510], radius=10, popup="Baripada").add_to(m)
render_map(m)
''',
        lambda: (
            lambda m: (
                folium.CircleMarker([20.2961, 85.8245], radius=10, popup="City Center").add_to(m),
                folium.CircleMarker([21.9320, 86.7510], radius=10, popup="Baripada").add_to(m),
                render_map(m)
            )
        )(folium.Map(location=[20.2961, 85.8245], zoom_start=7)),
    )


# ----------------------------
# GeoPandas
# ----------------------------
def geopandas_examples():
    st.subheader("GeoPandas")
    st.write("Geospatial analysis with a Pandas-like interface.")

    show_example(
        "Example 1: GeoDataFrame Table",
        '''
gdf = gpd.GeoDataFrame(
    {"city": ["Baripada", "Bhubaneswar", "Cuttack"]},
    geometry=[Point(86.7510, 21.9320), Point(85.8245, 20.2961), Point(85.8828, 20.4625)],
    crs="EPSG:4326"
)
st.dataframe(gdf)
''',
        lambda: st.dataframe(
            gpd.GeoDataFrame(
                {"city": ["Baripada", "Bhubaneswar", "Cuttack"]},
                geometry=[Point(86.7510, 21.9320), Point(85.8245, 20.2961), Point(85.8828, 20.4625)],
                crs="EPSG:4326",
            ).assign(geometry=lambda df: df.geometry.astype(str)),
            use_container_width=True,
        ),
    )

    show_example(
        "Example 2: Point Map",
        '''
gdf.plot(column="value", legend=True)
st.pyplot(fig)
''',
        lambda: (
            lambda fig, ax: (
                gpd.GeoDataFrame(
                    {"city": ["Baripada", "Bhubaneswar", "Cuttack"], "value": [10, 20, 15]},
                    geometry=[Point(86.7510, 21.9320), Point(85.8245, 20.2961), Point(85.8828, 20.4625)],
                    crs="EPSG:4326",
                ).plot(ax=ax, marker="o", column="value", legend=True),
                ax.set_title("Point Map"),
                st.pyplot(fig),
            )
        )(*plt.subplots(figsize=(6, 4))),
    )

    show_example(
        "Example 3: Polygon Map",
        '''
poly1 = Polygon([(85.7, 20.2), (85.9, 20.2), (85.9, 20.4), (85.7, 20.4)])
poly2 = Polygon([(86.6, 21.8), (86.9, 21.8), (86.9, 22.0), (86.6, 22.0)])
''',
        lambda: (
            lambda fig, ax: (
                gpd.GeoDataFrame(
                    {"region": ["R1", "R2"], "value": [5, 9]},
                    geometry=[Polygon([(85.7, 20.2), (85.9, 20.2), (85.9, 20.4), (85.7, 20.4)]), Polygon([(86.6, 21.8), (86.9, 21.8), (86.9, 22.0), (86.6, 22.0)])],
                    crs="EPSG:4326",
                ).plot(ax=ax, column="value", legend=True),
                ax.set_title("Polygon Map"),
                st.pyplot(fig),
            )
        )(*plt.subplots(figsize=(6, 4))),
    )

    show_example(
        "Example 4: Buffer Output",
        '''
points = gpd.GeoSeries([Point(85.82, 20.29), Point(85.88, 20.46)], crs="EPSG:4326")
buffered = points.to_crs("EPSG:3857").buffer(5000).to_crs("EPSG:4326")
st.write(buffered)
''',
        lambda: st.dataframe(
            gpd.GeoSeries([Point(85.82, 20.29), Point(85.88, 20.46)], crs="EPSG:4326").to_crs("EPSG:3857").buffer(5000).to_crs("EPSG:4326").astype(str).rename("geometry").to_frame()
        ),
    )

    show_example(
        "Example 5: Annotated Map",
        '''
gdf.plot(color="red")
''',
        lambda: (
            lambda fig, ax: (
                gdf := gpd.GeoDataFrame(
                    {"city": ["Baripada", "Bhubaneswar"], "score": [88, 95]},
                    geometry=[Point(86.7510, 21.9320), Point(85.8245, 20.2961)],
                    crs="EPSG:4326",
                ),
                gdf.plot(ax=ax, color="red"),
                [ax.annotate(row.city, (row.geometry.x, row.geometry.y), xytext=(3, 3), textcoords="offset points") for _, row in gdf.iterrows()],
                ax.set_title("Annotated Points"),
                st.pyplot(fig),
            )
        )(*plt.subplots(figsize=(6, 4))),
    )


# ----------------------------
# Datashader
# ----------------------------
def datashader_examples():
    st.subheader("Datashader")
    st.write("Excellent for very large datasets.")

    big_df = pd.DataFrame({"x": np.random.randn(100000), "y": np.random.randn(100000)})

    show_example(
        "Example 1: Dense Scatter",
        '''
canvas = ds.Canvas(plot_width=700, plot_height=400)
agg = canvas.points(big_df, "x", "y")
img = tf.shade(agg)
st.image(np.array(img.to_pil()))
''',
        lambda: st.image(np.array(tf.shade(ds.Canvas(plot_width=700, plot_height=400).points(big_df, "x", "y")).to_pil()), caption="Dense Scatter", use_container_width=True),
    )

    show_example(
        "Example 2: Dense Line",
        '''
sorted_df = pd.DataFrame({"x": np.sort(big_df["x"]), "y": np.sort(big_df["y"] )})
agg = ds.Canvas(plot_width=700, plot_height=400).line(sorted_df, "x", "y")
st.image(np.array(tf.shade(agg).to_pil()))
''',
        lambda: st.image(
            np.array(
                tf.shade(
                    ds.Canvas(plot_width=700, plot_height=400).line(
                        pd.DataFrame({"x": np.sort(big_df["x"]), "y": np.sort(big_df["y"])}),
                        "x",
                        "y",
                    )
                ).to_pil()
            ),
            caption="Dense Line",
            use_container_width=True,
        ),
    )

    show_example(
        "Example 3: Density View",
        '''
agg = ds.Canvas(plot_width=700, plot_height=400).points(big_df, "x", "y")
st.image(np.array(tf.shade(agg).to_pil()))
''',
        lambda: st.image(np.array(tf.shade(ds.Canvas(plot_width=700, plot_height=400).points(big_df, "x", "y")).to_pil()), caption="Density View", use_container_width=True),
    )

    show_example(
        "Example 4: Dark Background",
        '''
img = tf.set_background(tf.shade(agg), "black")
st.image(np.array(img.to_pil()))
''',
        lambda: st.image(np.array(tf.set_background(tf.shade(ds.Canvas(plot_width=700, plot_height=400).points(big_df, "x", "y")), "black").to_pil()), caption="Dark Background", use_container_width=True),
    )

    show_example(
        "Example 5: Summary Metrics",
        '''
st.metric("Rows", len(big_df))
st.metric("X Mean", round(big_df["x"].mean(), 4))
st.metric("Y Mean", round(big_df["y"].mean(), 4))
''',
        lambda: st.columns(3),
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(big_df))
    c2.metric("X Mean", round(big_df["x"].mean(), 4))
    c3.metric("Y Mean", round(big_df["y"].mean(), 4))


# ----------------------------
# Altair
# ----------------------------
def altair_examples():
    st.subheader("Altair")
    st.write("Compact declarative charts with a clean grammar of graphics.")

    show_example(
        "Example 1: Line Chart",
        '''
df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 3, 5, 7, 11]})
chart = alt.Chart(df).mark_line(point=True).encode(x="x", y="y")
st.altair_chart(chart, use_container_width=True)
''',
        lambda: st.altair_chart(
            alt.Chart(pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 3, 5, 7, 11]})).mark_line(point=True).encode(x="x", y="y").properties(title="Line Chart"),
            use_container_width=True,
        ),
    )

    show_example(
        "Example 2: Bar Chart",
        '''
df = pd.DataFrame({"category": ["A", "B", "C", "D"], "value": [12, 18, 7, 15]})
chart = alt.Chart(df).mark_bar().encode(x="category", y="value")
st.altair_chart(chart, use_container_width=True)
''',
        lambda: st.altair_chart(
            alt.Chart(pd.DataFrame({"category": ["A", "B", "C", "D"], "value": [12, 18, 7, 15]})).mark_bar().encode(x="category", y="value").properties(title="Bar Chart"),
            use_container_width=True,
        ),
    )

    show_example(
        "Example 3: Scatter Plot",
        '''
df = pd.DataFrame({"x": np.random.randn(100), "y": np.random.randn(100), "group": np.random.choice(["A", "B"], 100)})
chart = alt.Chart(df).mark_point(filled=True, size=60).encode(x="x", y="y", color="group")
st.altair_chart(chart, use_container_width=True)
''',
        lambda: st.altair_chart(
            alt.Chart(pd.DataFrame({"x": np.random.randn(250), "y": np.random.randn(250), "group": np.random.choice(["A", "B"], 250)})).mark_point(filled=True, size=60).encode(x="x", y="y", color="group").properties(title="Scatter Plot"),
            use_container_width=True,
        ),
    )

    show_example(
        "Example 4: Area Chart",
        '''
df = pd.DataFrame({"day": ["Mon", "Tue", "Wed", "Thu", "Fri"], "bill": [10, 15, 13, 18, 21]})
chart = alt.Chart(df).mark_area().encode(x="day", y="bill")
st.altair_chart(chart, use_container_width=True)
''',
        lambda: st.altair_chart(
            alt.Chart(pd.DataFrame({"day": ["Mon", "Tue", "Wed", "Thu", "Fri"], "bill": [10, 15, 13, 18, 21]})).mark_area().encode(x="day", y="bill").properties(title="Area Chart"),
            use_container_width=True,
        ),
    )

    show_example(
        "Example 5: Grouped Points",
        '''
df = pd.DataFrame({"x": [1, 2, 1, 2], "y": [4, 5, 3, 4], "group": ["A", "A", "B", "B"]})
chart = alt.Chart(df).mark_circle(size=120).encode(x="x", y="y", color="group")
st.altair_chart(chart, use_container_width=True)
''',
        lambda: st.altair_chart(
            alt.Chart(pd.DataFrame({"x": [1, 2, 1, 2], "y": [4, 5, 3, 4], "group": ["A", "A", "B", "B"]})).mark_circle(size=120).encode(x="x", y="y", color="group").properties(title="Grouped Points"),
            use_container_width=True,
        ),
    )


col1, = st.columns(1)
col1.metric("Selected Library", library)

main_tabs = st.tabs(["Examples", "About", "Run Notes"])
with main_tabs[0]:
    if library == "Matplotlib":
        matplotlib_examples()
    elif library == "Seaborn":
        seaborn_examples()
    elif library == "Plotly":
        plotly_examples()
    elif library == "Folium":
        folium_examples()
    elif library == "GeoPandas":
        geopandas_examples()
    elif library == "Datashader":
        datashader_examples()
    elif library == "Altair":
        altair_examples()

with main_tabs[1]:
    st.markdown(
        """
        ### What this app does
        - Lets participants pick a visualization library from a dropdown.
        - Shows **five examples** for each library.
        - Displays both the **code** and the **output**.
        - Uses Streamlit widgets, metrics, expanders, tabs, and columns for a workshop-friendly UI.

        ### Installation
        ```bash
        pip install streamlit matplotlib seaborn plotly folium geopandas datashader altair shapely streamlit-folium
        ```
        """
    )

with main_tabs[2]:
    st.markdown(
        """
        ### Running the app on Windows
        1. Save this file as `app.py`
        2. Open Command Prompt / PowerShell in the same folder
        3. Run:
        ```bash
        streamlit run app.py
        ```

        ### Notes
        - Folium maps are embedded directly in the app.
        - GeoPandas examples use synthetic data.
        - Datashader works best for very large datasets.
        - The code is written to be easy to demonstrate live in a workshop.
        """
    )
