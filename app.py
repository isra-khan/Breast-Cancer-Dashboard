"""Interactive breast cancer EDA dashboard."""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Breast Cancer EDA | Pink Ribbon",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROSE = "#D81B60"
PINK = "#E91E63"
BERRY = "#5A0A30"
PLUM = "#3B1024"
MUTED = "#6E4758"
PALETTE = [ROSE, "#A94B78", "#E995B5", "#7C3158", "#F2B9CE", "#C66A91", "#8A6A79"]
STATUS_COLORS = {"Alive": ROSE, "Dead": "#5A0A30"}


def normalized_name(value: str) -> str:
    """Match the notebook's strip/lower/space-to-underscore normalization."""
    return re.sub(r"\s+", "_", str(value).strip().lower())


def load_data() -> tuple[pd.DataFrame, int, str]:
    """Load the workspace CSV (or an uploaded replacement) and apply notebook logic."""
    default_path = Path(__file__).resolve().parent / "Breast_Cancer .csv"
    uploaded = st.sidebar.file_uploader("Use another CSV", type=["csv"], help="Optional. The bundled dataset is used by default.")
    try:
        raw = pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(default_path)
        source_name = uploaded.name if uploaded is not None else default_path.name
    except Exception as exc:
        st.error(f"Could not load the dataset: {exc}")
        st.stop()

    # Same cleaning sequence as the notebook: remove exact duplicate, normalize names.
    before = len(raw)
    df = raw.drop_duplicates().copy()
    removed = before - len(df)
    df.columns = [normalized_name(col) for col in df.columns]

    # The notebook dataset spells this column "reginol". Resolve that safely while
    # also accepting the corrected spelling in alternate datasets.
    positive_node_col = next(
        (name for name in ("reginol_node_positive", "regional_node_positive", "regional_nodes_positive") if name in df.columns),
        None,
    )

    if "age" in df.columns:
        # Notebook definitions retained; out-of-range/missing ages remain unassigned.
        df["age_group"] = pd.cut(
            pd.to_numeric(df["age"], errors="coerce"),
            bins=[29, 40, 50, 60, 69],
            labels=["<=40", "41-50", "51-60", "61-69"],
            include_lowest=True,
        )
    if "tumor_size" in df.columns:
        df["tumor_size_category"] = pd.cut(
            pd.to_numeric(df["tumor_size"], errors="coerce"),
            bins=[0, 20, 50, float("inf")],
            labels=["Small", "Medium", "Large"],
        )
    if positive_node_col and "regional_node_examined" in df.columns:
        positive = pd.to_numeric(df[positive_node_col], errors="coerce")
        examined = pd.to_numeric(df["regional_node_examined"], errors="coerce").replace(0, np.nan)
        # Keep the notebook's ratio, then bound it and fill undefined values so the
        # engineered feature is always numeric, finite, and in [0, 1].
        df["positive_node_ratio"] = (positive / examined).replace([np.inf, -np.inf], np.nan).clip(0, 1).fillna(0.0)
    if "estrogen_status" in df.columns and "progesterone_status" in df.columns:
        df["hormone_receptor_status"] = df["estrogen_status"].astype("string") + "_" + df["progesterone_status"].astype("string")

    return df, removed, source_name


def add_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
        :root { --rose:#D81B60; --pink:#E91E63; --berry:#5A0A30; --plum:#3B1024; --muted:#6E4758; }
        .stApp { background: linear-gradient(145deg,#FFF8FA 0%,#FEF5F8 55%,#FFF9FB 100%); color:var(--plum); }
        html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
        h1,h2,h3 { color:var(--plum); letter-spacing:-.02em; }
        h1 { font-family:'Playfair Display',serif; }
        [data-testid="stSidebar"] { background:#FCEBF1; border-right:1px solid #F3D7E2; }
        [data-testid="stSidebar"] * { color:var(--plum); }
        .hero { position:relative; overflow:hidden; border-radius:24px; padding:27px 32px; margin:2px 0 22px;
          min-height:172px; background:linear-gradient(112deg,#5A0A30 0%,#8D164A 56%,#D81B60 100%); color:#fff;
          box-shadow:0 12px 32px rgba(90,10,48,.14); }
        .hero h1 { margin:0 0 6px; color:#fff; font:700 32px 'Playfair Display',serif; position:relative; z-index:2; }
        .hero p { margin:0; max-width:620px; color:#FFE7EF; font-size:14px; position:relative; z-index:2; }
        .hero .eyebrow { color:#FFC9DC; text-transform:uppercase; letter-spacing:.17em; font-size:10px; font-weight:700; margin-bottom:9px; }
        .cell-strip { position:absolute; right:18px; top:5px; width:360px; height:164px; opacity:.76; }
        .cell { transform-box:fill-box; transform-origin:center; animation:drift 10s ease-in-out infinite alternate, pulse 5s ease-in-out infinite; }
        .cell:nth-child(2) { animation-delay:-3s,-1s; }.cell:nth-child(3) { animation-delay:-6s,-2s; }
        .ribbon { animation:float 7s ease-in-out infinite; transform-origin:center; }
        @keyframes drift { from { transform:translate(0,5px) rotate(-3deg); } to { transform:translate(7px,-7px) rotate(4deg); } }
        @keyframes pulse { 0%,100% { opacity:.55; } 50% { opacity:.92; } }
        @keyframes float { 0%,100% { transform:translateY(0) rotate(-5deg); } 50% { transform:translateY(-4px) rotate(4deg); } }
        @media (prefers-reduced-motion: reduce) { *,*::before,*::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; scroll-behavior:auto !important; } }
        [data-testid="stMetric"] { display:flex; flex-direction:column; justify-content:center; box-sizing:border-box;
          height:132px; background:#fff; border:1px solid #F2D7E1; padding:16px 18px; border-radius:18px;
          box-shadow:0 5px 18px rgba(90,10,48,.055); }
        [data-testid="stMetricLabel"] { color:var(--muted); font-weight:600; }
        [data-testid="stMetricValue"] { color:var(--berry); font-weight:700; }
        .note-card { background:#FCEBF1; border:1px solid #F4D4E0; border-left:4px solid #D81B60; border-radius:13px; padding:14px 17px; color:#5A0A30; margin:8px 0 12px; }
        .takeaway { height:100%; background:#fff; border:1px solid #F2D7E1; border-radius:16px; padding:17px 18px; box-shadow:0 5px 18px rgba(90,10,48,.045); }
        .takeaway b { color:#D81B60; }
        div[data-testid="stExpander"] { background:rgba(255,255,255,.72); border:1px solid #F2D7E1; border-radius:14px; }
        </style>
        <div class="hero">
          <div class="eyebrow">Breast cancer awareness · Exploratory data</div>
          <h1>Understanding the story in the data</h1>
          <p>Explore patient characteristics, tumor features, and recorded survival months across the cohort.</p>
          <svg class="cell-strip" viewBox="0 0 360 164" aria-hidden="true">
            <g class="cell" transform="translate(214 78)"><path d="M-42-5 C-47-27-24-39-5-34 C13-47 37-29 33-10 C49 5 33 29 13 28 C-2 43-27 31-31 14 C-47 10-52 0-42-5Z" fill="#F6A8C4" fill-opacity=".32" stroke="#FFDCE8" stroke-opacity=".8" stroke-width="2"/><ellipse cx="0" cy="0" rx="13" ry="16" fill="#65123D" fill-opacity=".82"/><circle cx="-20" cy="-14" r="3" fill="#FFE6EF"/></g>
            <g class="cell" transform="translate(298 43) scale(.72)"><path d="M-42-5 C-47-27-24-39-5-34 C13-47 37-29 33-10 C49 5 33 29 13 28 C-2 43-27 31-31 14 C-47 10-52 0-42-5Z" fill="#FFBDD2" fill-opacity=".35" stroke="#FFE2EC" stroke-width="2"/><ellipse cx="1" cy="-1" rx="13" ry="16" fill="#7D1949" fill-opacity=".84"/><circle cx="-19" cy="-14" r="3" fill="#FFF0F5"/></g>
            <g class="cell" transform="translate(160 126) scale(.52)"><path d="M-42-5 C-47-27-24-39-5-34 C13-47 37-29 33-10 C49 5 33 29 13 28 C-2 43-27 31-31 14 C-47 10-52 0-42-5Z" fill="#F9C5D7" fill-opacity=".32" stroke="#FFEAF1" stroke-width="2"/><ellipse cx="0" cy="0" rx="13" ry="16" fill="#751541" fill-opacity=".82"/></g>
            <g class="ribbon" transform="translate(330 115) rotate(-18)"><path d="M-14-17 C-13-31 10-30 12-15 C13-6 3 3 0 7 C-4 2-15-7-14-17Z" fill="none" stroke="#FFE6EF" stroke-width="6"/><path d="M-3 3 L-15 31 L-2 25 L4 38 L8 8 M2 5 L19 29 L6 25" fill="none" stroke="#FFE6EF" stroke-width="6" stroke-linejoin="round"/></g>
          </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=48, b=8),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="DM Sans, sans-serif", color=PLUM, size=12),
        title=dict(font=dict(color=PLUM, size=16), x=0.02),
        legend=dict(bgcolor="rgba(255,255,255,0)", font=dict(color=MUTED)),
        colorway=PALETTE,
        hoverlabel=dict(bgcolor="#FFF8FA", bordercolor="#E8A9C0", font_color=PLUM),
    )
    fig.update_xaxes(showgrid=False, linecolor="#EBD6DF", tickfont_color=MUTED, title_font_color=MUTED)
    fig.update_yaxes(gridcolor="#F1E2E8", zerolinecolor="#EBD6DF", tickfont_color=MUTED, title_font_color=MUTED)
    return fig


def main() -> None:
    add_theme()
    df, removed, source_name = load_data()
    if df.empty:
        st.warning("The selected file contains no records.")
        st.stop()

    st.sidebar.markdown("### Refine the cohort")
    filters: dict[str, object] = {}
    primary_filters = [c for c in ("status", "age_group", "tumor_size_category") if c in df.columns]
    for col in primary_filters:
        values = list(df[col].dropna().unique())
        values = sorted(values, key=str)
        selected = st.sidebar.multiselect(col.replace("_", " ").title(), values, default=values, key=f"primary_{col}")
        filters[col] = selected

    derived = {"age_group", "tumor_size_category", "positive_node_ratio", "hormone_receptor_status"}
    categorical = [c for c in df.select_dtypes(include=["object", "category", "string"]).columns if c not in primary_filters]
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns]
    with st.sidebar.expander("More clinical filters", expanded=False):
        for col in categorical:
            vals = sorted(df[col].dropna().astype(str).unique().tolist())
            if 1 < len(vals) <= 60:
                filters[col] = st.multiselect(col.replace("_", " ").title(), vals, default=vals, key=f"cat_{col}")
        for col in numeric:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty or series.min() == series.max():
                continue
            is_int = pd.api.types.is_integer_dtype(df[col])
            if is_int:
                lo, hi, step = int(series.min()), int(series.max()), 1
            else:
                lo, hi = float(series.min()), float(series.max())
                step = max((hi - lo) / 100, 0.01)
            filters[col] = st.slider(col.replace("_", " ").title(), lo, hi, (lo, hi), step=step, key=f"num_{col}")

    filtered = df.copy()
    for col, selected in filters.items():
        if isinstance(selected, list):
            if col in filtered.columns:
                filtered = filtered[filtered[col].astype("string").isin([str(v) for v in selected])]
        elif isinstance(selected, tuple) and col in filtered.columns:
            values = pd.to_numeric(filtered[col], errors="coerce")
            filtered = filtered[values.between(selected[0], selected[1])]

    if filtered.empty:
        st.info("No records match these filters. Adjust the sidebar selections to continue.")
        st.stop()

    st.caption(f"Source: **{source_name}** · {len(df):,} unique records · {removed:,} duplicate row removed during cleaning")
    status_counts = filtered["status"].value_counts() if "status" in filtered.columns else pd.Series(dtype=int)
    alive = int(status_counts.get("Alive", 0))
    dead = int(status_counts.get("Dead", 0))
    survival = pd.to_numeric(filtered.get("survival_months", pd.Series(dtype=float)), errors="coerce").dropna()
    ratio = pd.to_numeric(filtered.get("positive_node_ratio", pd.Series(dtype=float)), errors="coerce").dropna()

    metrics = st.columns(6)
    metrics[0].metric("Total records", f"{len(filtered):,}")
    metrics[1].metric("Alive", f"{alive:,}", f"{alive / len(filtered):.1%}" if "status" in filtered else None)
    metrics[2].metric("Dead", f"{dead:,}", f"{dead / len(filtered):.1%}" if "status" in filtered else None)
    metrics[3].metric("Mean survival", f"{survival.mean():.1f} mo" if not survival.empty else "—")
    metrics[4].metric("Median survival", f"{survival.median():.1f} mo" if not survival.empty else "—")
    metrics[5].metric("Avg positive node ratio", f"{ratio.mean():.3f}" if not ratio.empty else "—")

    if "status" in filtered.columns:
        n_status = int(status_counts.sum())
        largest = int(status_counts.max()) if len(status_counts) else 0
        imbalance = largest / n_status if n_status else 0
        dominant = status_counts.idxmax() if len(status_counts) else "—"
        st.markdown(f"<div class='note-card'><b>Status balance:</b> {dominant} is the larger class ({imbalance:.1%} of records). Use percentage-aware comparisons when category sizes differ.</div>", unsafe_allow_html=True)

    st.markdown("## Cohort overview")
    left, right = st.columns(2)
    if "status" in filtered.columns:
        counts = filtered["status"].value_counts().rename_axis("Status").reset_index(name="Records")
        with left:
            fig = px.pie(counts, names="Status", values="Records", hole=.64, title="Status distribution", color="Status", color_discrete_map=STATUS_COLORS)
            fig.update_traces(textposition="inside", textinfo="percent+label", marker_line_color="#FFF8FA", marker_line_width=3)
            st.plotly_chart(style_fig(fig), use_container_width=True)
    if "survival_months" in filtered.columns:
        with right:
            fig = px.histogram(filtered, x="survival_months", color="status" if "status" in filtered.columns else None, barmode="overlay", opacity=.78, nbins=28, title="Recorded survival months", color_discrete_map=STATUS_COLORS)
            st.plotly_chart(style_fig(fig), use_container_width=True)

    if "survival_months" in filtered.columns:
        st.markdown("## Survival months across groups")
        group_options = [c for c in ("status", "age_group", "tumor_size_category", "n_stage", "t_stage", "grade", "race", "estrogen_status", "progesterone_status") if c in filtered.columns]
        if group_options:
            selected_group = st.selectbox("Compare survival months by", group_options, index=0, key="survival_group")
            c1, c2 = st.columns([1.35, 1])
            with c1:
                fig = px.box(filtered, x=selected_group, y="survival_months", color=selected_group, points="outliers", title=f"Survival months by {selected_group.replace('_',' ')}", color_discrete_sequence=PALETTE)
                st.plotly_chart(style_fig(fig, 370), use_container_width=True)
            with c2:
                summary = filtered.groupby(selected_group, observed=True)["survival_months"].agg(Records="count", Mean="mean", Median="median").reset_index()
                fig = px.bar(summary, x=selected_group, y="Mean", color=selected_group, text=summary["Mean"].round(1), title="Mean survival months", color_discrete_sequence=PALETTE)
                fig.update_traces(textposition="outside")
                st.plotly_chart(style_fig(fig, 370), use_container_width=True)

    st.markdown("## Clinical feature breakdowns")
    charts = st.columns(3)
    if "age_group" in filtered.columns:
        counts = filtered["age_group"].value_counts(sort=False, dropna=False).rename_axis("Age group").reset_index(name="Records")
        counts["Age group"] = counts["Age group"].astype(str).replace("nan", "Unassigned")
        with charts[0]:
            fig = px.bar(counts, x="Age group", y="Records", color="Age group", title="Records by age group", color_discrete_sequence=PALETTE)
            st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    if "tumor_size_category" in filtered.columns:
        counts = filtered["tumor_size_category"].value_counts(sort=False, dropna=False).rename_axis("Tumor size").reset_index(name="Records")
        counts["Tumor size"] = counts["Tumor size"].astype(str).replace("nan", "Unassigned")
        with charts[1]:
            fig = px.bar(counts, x="Tumor size", y="Records", color="Tumor size", title="Records by tumor size", color_discrete_sequence=PALETTE)
            st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    if "positive_node_ratio" in filtered.columns:
        with charts[2]:
            fig = px.histogram(filtered, x="positive_node_ratio", nbins=24, title="Positive node ratio (0–1)", color_discrete_sequence=[ROSE])
            st.plotly_chart(style_fig(fig, 320), use_container_width=True)

    extra_group = next((c for c in ("age_group", "tumor_size_category", "positive_node_ratio") if c in filtered.columns), None)
    if extra_group and "survival_months" in filtered.columns:
        with st.expander("Explore survival and node involvement", expanded=False):
            if extra_group == "positive_node_ratio":
                temp = filtered.copy()
                temp["positive_node_ratio_band"] = pd.cut(temp["positive_node_ratio"], bins=[-0.001, .25, .5, .75, 1.001], labels=["0–0.25", ".25–0.50", ".50–0.75", ".75–1.00"])
                group = "positive_node_ratio_band"
            else:
                temp, group = filtered, extra_group
            fig = px.box(temp, x=group, y="survival_months", color="status" if "status" in temp.columns else group, title="Survival months by clinical group", color_discrete_map=STATUS_COLORS, color_discrete_sequence=PALETTE)
            st.plotly_chart(style_fig(fig, 350), use_container_width=True)

    st.markdown("## EDA takeaways")
    takeaways = []
    if "status" in filtered.columns:
        takeaways.append(("Status mix", f"{alive:,} Alive and {dead:,} Dead records in the current cohort."))
    if not survival.empty:
        takeaways.append(("Recorded follow-up", f"Survival months have a median of {survival.median():.1f} and a mean of {survival.mean():.1f}. These are descriptive summaries."))
    if "tumor_size_category" in filtered.columns:
        top_tumor = filtered["tumor_size_category"].mode(dropna=True)
        if not top_tumor.empty:
            takeaways.append(("Tumor size", f"{top_tumor.iloc[0]} is the most frequent tumor-size band in this cohort."))
    if "positive_node_ratio" in filtered.columns:
        takeaways.append(("Node involvement", f"The average positive-to-examined node ratio is {filtered['positive_node_ratio'].mean():.3f}; ratios are bounded to 0–1."))
    for start in range(0, len(takeaways), 2):
        cols = st.columns(2)
        for col, (title, body) in zip(cols, takeaways[start:start + 2]):
            with col:
                st.markdown(f"<div class='takeaway'><b>{title}</b><br>{body}</div>", unsafe_allow_html=True)

    with st.expander(f"View filtered data ({len(filtered):,} records)", expanded=False):
        st.dataframe(filtered, use_container_width=True, hide_index=True, height=420)
        st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"), "breast_cancer_filtered.csv", "text/csv")

    st.caption("This dashboard is for exploratory analysis of the supplied dataset and does not provide clinical guidance.")


if __name__ == "__main__":
    main()
