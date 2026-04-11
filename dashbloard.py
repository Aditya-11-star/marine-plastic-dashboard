import io
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import seaborn as sns
except Exception:
    sns = None


st.set_page_config(
    page_title="Marine Plastic Pollution Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        :root {
            --bg: #0A0E1A;
            --card: rgba(255, 255, 255, 0.06);
            --text: #E0E0E0;
            --muted: #8ECAE6;
            --teal: #00F5D4;
            --coral: #FF6B6B;
            --glass-border: rgba(255, 255, 255, 0.18);
        }

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            color: var(--text);
        }

        .stApp {
            background: radial-gradient(circle at 5% 10%, rgba(0, 245, 212, 0.13), transparent 35%),
                        radial-gradient(circle at 95% 90%, rgba(255, 107, 107, 0.12), transparent 35%),
                        linear-gradient(145deg, #0A0E1A 0%, #0B1125 50%, #101937 100%);
            color: var(--text);
        }

        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF !important;
        }

        p, li, span, div, label {
            color: var(--text);
        }

        section[data-testid="stSidebar"],
        [data-testid="stSidebar"],
        .css-1d391kg {
            background: #0D1B2A !important;
            border-right: 1px solid var(--glass-border);
            border-left: 2px solid #00F5D4;
            padding-top: 0.5rem;
        }

        section[data-testid="stSidebar"] * ,
        [data-testid="stSidebar"] * ,
        .css-1d391kg * {
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #00F5D4 !important;
            font-weight: 700 !important;
            letter-spacing: 0.2px;
        }

        /* Sidebar radio options spacing + visibility */
        section[data-testid="stSidebar"] .stRadio > div,
        [data-testid="stSidebar"] .stRadio > div {
            gap: 0.45rem;
        }

        section[data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] .stRadio label p,
        [data-testid="stSidebar"] .stRadio label p,
        .stRadio label {
            color: #FFFFFF !important;
            opacity: 1 !important;
            font-weight: 600 !important;
        }

        /* Selected radio option glow */
        section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked),
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) {
            background: rgba(0, 245, 212, 0.16) !important;
            border: 1px solid #00F5D4 !important;
            box-shadow: 0 0 10px rgba(0, 245, 212, 0.45);
            border-radius: 10px;
            padding: 0.35rem 0.5rem;
        }

        section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) p,
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) p {
            color: #00F5D4 !important;
            font-weight: 700 !important;
        }

        /* Fallback for browsers without :has support */
        [data-testid="stSidebar"] input[type="radio"] {
            accent-color: #00F5D4 !important;
        }

        [data-testid="stSidebar"] .stMarkdown p {
            color: #8ECAE6 !important;
            font-weight: 600 !important;
        }

        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 0.15rem;
            text-shadow: 0 0 16px rgba(0,245,212,0.45);
        }

        .subtitle {
            font-size: 1rem;
            color: #8ECAE6;
            margin-bottom: 1rem;
        }

        .glass-card {
            background: var(--card);
            border: 1px solid var(--glass-border);
            border-radius: 18px;
            padding: 16px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            box-shadow: 0 0 14px rgba(0,245,212,0.16), 0 0 22px rgba(255,107,107,0.10);
            margin-bottom: 12px;
        }

        .metric-title {
            color: #8ECAE6;
            font-size: 0.95rem;
            margin-bottom: 0.2rem;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--teal);
            text-shadow: 0 0 12px rgba(0,245,212,0.5);
        }

        .insight-box {
            border-left: 3px solid var(--coral);
            background: rgba(255, 107, 107, 0.10);
            border-radius: 12px;
            padding: 10px 12px;
            margin: 8px 0 16px;
            color: var(--text);
        }

        .stAlert {
            border-radius: 14px;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data(path: str = "preprocessed_marine_plastic.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def canon(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def build_column_map(df: pd.DataFrame) -> Dict[str, str]:
    return {canon(col): col for col in df.columns}


def pick_col(df: pd.DataFrame, candidates) -> str:
    mapping = build_column_map(df)
    for c in candidates:
        key = canon(c)
        if key in mapping:
            return mapping[key]
    return ""


def metric_card(title: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(title: str, value: float) -> None:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="font-size:1.45rem;">{value:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dark_figure(figsize=(8, 4)):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0B1125")
    ax.set_facecolor("#0B1125")
    return fig, ax


def page_overview(df: pd.DataFrame) -> None:
    st.markdown("## 🌊 Overview — Marine Plastic Snapshot")
    total_rows = len(df)
    total_cols = len(df.columns)
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("🧾 Total Rows", str(total_rows))
    with c2:
        metric_card("🧱 Columns", str(total_cols))
    with c3:
        metric_card("🧼 Missing Values", str(missing))
    with c4:
        metric_card("🧬 Duplicates", str(duplicates))

    st.markdown("### 🔄 Before vs After Preprocessing")
    compare_df = pd.DataFrame(
        {
            "Feature": [
                "Rows",
                "Columns",
                "Missing Values",
                "Duplicates",
                "Encoding/Scaling",
                "Log Features",
                "Season Flags",
            ],
            "Before Preprocessing": [
                "Raw rows",
                "Base columns",
                "Present",
                "Possible",
                "Not applied",
                "Not available",
                "Not available",
            ],
            "After Preprocessing": [
                f"{total_rows}",
                f"{total_cols}",
                f"{missing}",
                f"{duplicates}",
                "Source + Sea State encoded",
                "Log_CD* and Log_WD* added",
                "Season_Autumn/Spring/Summer/Winter added",
            ],
        }
    )
    st.dataframe(compare_df, use_container_width=True)

    st.markdown("### 🔍 First 10 Rows")
    st.dataframe(df.head(10), use_container_width=True)


def page_statistical(df: pd.DataFrame) -> None:
    st.markdown("## 📊 Statistical Measures")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    selected_col = st.selectbox("Choose a numeric column", options=numeric_cols, index=0)

    series = df[selected_col].dropna()
    mean_v = series.mean()
    median_v = series.median()
    mode_v = series.mode().iloc[0] if not series.mode().empty else float("nan")
    std_v = series.std()
    var_v = series.var()
    skew_v = series.skew()

    a, b, c = st.columns(3)
    with a:
        stat_card("Mean", mean_v)
        stat_card("Std Deviation", std_v)
    with b:
        stat_card("Median", median_v)
        stat_card("Variance", var_v)
    with c:
        stat_card("Mode", mode_v)
        stat_card("Skewness", skew_v)

    st.markdown("### 🔥 Correlation Heatmap")
    corr_df = df[numeric_cols].corr(numeric_only=True)
    fig, ax = dark_figure(figsize=(11, 7))
    if sns is not None:
        sns.heatmap(
            corr_df,
            cmap="coolwarm",
            annot=False,
            linewidths=0.3,
            linecolor="#1A2446",
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
    else:
        # Fallback heatmap when seaborn/scipy import fails on some setups.
        im = ax.imshow(corr_df.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(corr_df.columns)))
        ax.set_yticks(np.arange(len(corr_df.index)))
        ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr_df.index, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        st.info("Seaborn unavailable; using matplotlib fallback heatmap.")
    ax.set_title("Feature Correlation Matrix", color="#EAF2FF", fontsize=13, pad=12)
    st.pyplot(fig, use_container_width=True)

    if skew_v > 0.5:
        st.error("⚠️ Right-skewed distribution detected for selected feature.")
    elif skew_v < -0.5:
        st.warning("⚠️ Left-skewed distribution detected for selected feature.")
    else:
        st.success("✅ Distribution is approximately balanced.")


def page_visualizations(df: pd.DataFrame) -> None:
    st.markdown("## 🎨 Visualizations Hub")

    cd1 = pick_col(df, ["CD1 (/km^2)", "CD1  (/km^2)"])
    cd2 = pick_col(df, ["CD2 (/km^2)", "CD2  (/km^2)"])
    cd3 = pick_col(df, ["CD3 (/km^2)", "CD3  (/km^2)"])
    cd4 = pick_col(df, ["CD4 (/km^2)", "CD4  (/km^2)"])
    year_col = pick_col(df, ["Year"])
    source_col = pick_col(df, ["Researcher", "Source"])

    st.markdown("### 1) 📈 Histogram of CD1")
    if cd1:
        s = pd.to_numeric(df[cd1], errors="coerce").dropna()
        pos = s[s > 0]
        fig, ax = dark_figure(figsize=(9, 4))
        if len(pos) > 2:
            bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 38)
            ax.hist(pos, bins=bins, color="#00F5D4", edgecolor="#FF6B6B", alpha=0.85)
            ax.set_xscale("log")
            ax.set_xlabel("CD1 (/km^2) [log scale]")
        else:
            ax.hist(s, bins=30, color="#00F5D4", edgecolor="#FF6B6B", alpha=0.85)
            ax.set_xlabel("CD1 (/km^2)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of CD1")
        st.pyplot(fig, use_container_width=True)
        st.markdown(
            '<div class="insight-box">💡 <b>Insight:</b> CD1 shows heavy right-tail behavior, indicating most areas have low plastic density while few hotspots are extremely high.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("CD1 column not found.")

    st.markdown("### 2) 📦 Box Plot of CD1 to CD4")
    box_cols = [c for c in [cd1, cd2, cd3, cd4] if c]
    if len(box_cols) >= 2:
        fig, ax = dark_figure(figsize=(9, 4))
        data = [pd.to_numeric(df[c], errors="coerce").dropna() for c in box_cols]
        ax.boxplot(
            data,
            labels=[c.split()[0] for c in box_cols],
            patch_artist=True,
            boxprops=dict(facecolor="#0FD9BF", color="#00F5D4"),
            medianprops=dict(color="#FF6B6B", linewidth=2),
            whiskerprops=dict(color="#00F5D4"),
            capprops=dict(color="#00F5D4"),
            flierprops=dict(marker="o", markerfacecolor="#FF6B6B", markeredgecolor="#FF6B6B", alpha=0.4, markersize=4),
        )
        ax.set_title("Outlier Spread in CD1–CD4")
        ax.set_ylabel("Particle Count (/km^2)")
        st.pyplot(fig, use_container_width=True)
        st.markdown(
            '<div class="insight-box">🧪 <b>Insight:</b> Boxplots reveal strong outliers across categories, suggesting uneven and localized pollution accumulation.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Not enough CD columns found.")

    st.markdown("### 3) 📅 Bar Chart of Average CD1 by Year")
    if cd1 and year_col:
        fig, ax = dark_figure(figsize=(9, 4))
        yearly = df.groupby(year_col, as_index=False)[cd1].mean()
        ax.bar(yearly[year_col].astype(str), yearly[cd1], color="#00F5D4", edgecolor="#FF6B6B")
        ax.set_title("Average CD1 by Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Avg CD1")
        ax.tick_params(axis="x", rotation=35)
        st.pyplot(fig, use_container_width=True)
        st.markdown(
            '<div class="insight-box">📌 <b>Insight:</b> Year-wise variation highlights changing marine plastic intensity, useful for policy impact tracking.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Year or CD1 column not found.")

    st.markdown("### 4) 🌦️ Bar Chart of Samples by Season")
    season_cols = [c for c in df.columns if c.startswith("Season_")]
    if season_cols:
        fig, ax = dark_figure(figsize=(9, 4))
        season_counts = {c.replace("Season_", ""): int((df[c] == 1).sum()) for c in season_cols}
        season_series = pd.Series(season_counts).sort_values(ascending=False)
        ax.bar(season_series.index, season_series.values, color="#FF6B6B", edgecolor="#00F5D4")
        ax.set_title("Samples by Season")
        ax.set_ylabel("Sample Count")
        st.pyplot(fig, use_container_width=True)
        st.markdown(
            '<div class="insight-box">🍂 <b>Insight:</b> Seasonal sample concentration can influence observed pollution trends and interpretability.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Season columns not found.")

    st.markdown("### 5) 👩‍🔬 Horizontal Bar Chart of Samples by Researcher")
    if source_col:
        fig, ax = dark_figure(figsize=(9, 5))
        src_counts = df[source_col].astype(str).value_counts().head(15).sort_values()
        ax.barh(src_counts.index, src_counts.values, color="#00F5D4", edgecolor="#FF6B6B")
        ax.set_title(f"Samples by {'Researcher' if canon(source_col) == canon('Researcher') else 'Source'}")
        ax.set_xlabel("Count")
        st.pyplot(fig, use_container_width=True)
        st.markdown(
            '<div class="insight-box">🧭 <b>Insight:</b> Contributor/source dominance may reflect sampling effort differences rather than true pollution differences.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Researcher/Source column not found.")


def page_explorer(df: pd.DataFrame) -> None:
    st.markdown("## 🧰 Data Explorer")
    year_col = pick_col(df, ["Year"])

    if year_col:
        year_options = ["All"] + sorted(df[year_col].dropna().astype(int).unique().tolist())
        selected = st.selectbox("Filter by Year", year_options)
        if selected == "All":
            filtered = df.copy()
        else:
            filtered = df[df[year_col].astype("Int64") == int(selected)].copy()
    else:
        st.warning("Year column not found. Showing complete dataset.")
        filtered = df.copy()

    st.dataframe(filtered, use_container_width=True, height=500)

    csv_buffer = io.StringIO()
    filtered.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download Filtered CSV",
        data=csv_buffer.getvalue(),
        file_name="filtered_marine_plastic.csv",
        mime="text/csv",
    )


def main() -> None:
    inject_css()
    st.markdown('<div class="main-title">🌊 Marine Plastic Pollution Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Dark-mode, neon analytics for your Python Data Science project ♻️🐢</div>',
        unsafe_allow_html=True,
    )

    df = load_data("preprocessed_marine_plastic.csv")

    st.sidebar.markdown("## 🧭 Navigation")
    page = st.sidebar.radio(
        "Go to page",
        ["Overview", "Statistical Measures", "Visualizations", "Data Explorer"],
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("🌐 **Theme:** Dark Ocean Glass")
    st.sidebar.markdown("💎 **Accent:** #00F5D4 + #FF6B6B")

    if page == "Overview":
        page_overview(df)
    elif page == "Statistical Measures":
        page_statistical(df)
    elif page == "Visualizations":
        page_visualizations(df)
    else:
        page_explorer(df)


if __name__ == "__main__":
    main()
