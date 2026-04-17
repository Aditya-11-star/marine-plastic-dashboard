"""
MARINE PLASTIC POLLUTION — FULL DASHBOARD WITH 3 ML MODELS
===========================================================
Models: K-Means, DBSCAN, Hierarchical Clustering

Instructions:
  1. Place in same folder as: preprocessed_marine_plastic.csv
  2. pip install streamlit pandas matplotlib seaborn scikit-learn scipy
  3. python -m streamlit run dashbloard.py
"""

import warnings
warnings.filterwarnings("ignore")

import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

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

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Poppins',sans-serif;color:#E0E0E0}
.stApp{
  background:radial-gradient(circle at 5% 10%,rgba(0,245,212,.13),transparent 35%),
             radial-gradient(circle at 95% 90%,rgba(255,107,107,.12),transparent 35%),
             linear-gradient(145deg,#0A0E1A 0%,#0B1125 50%,#101937 100%);
}
h1,h2,h3,h4,h5,h6{color:#FFFFFF !important}
section[data-testid="stSidebar"],[data-testid="stSidebar"]{
  background:#0D1B2A !important;
  border-left:3px solid #00F5D4;
}
section[data-testid="stSidebar"] *,[data-testid="stSidebar"] *{
  color:#FFFFFF !important;font-weight:600 !important
}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{
  color:#00F5D4 !important;font-weight:700 !important
}
[data-testid="stSidebar"] .stMarkdown p{color:#8ECAE6 !important}
[data-testid="stSidebar"] input[type="radio"]{accent-color:#00F5D4 !important}
.glass-card{
  background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.18);
  border-radius:18px;padding:16px;backdrop-filter:blur(10px);
  box-shadow:0 0 14px rgba(0,245,212,.16);margin-bottom:12px
}
.metric-title{color:#8ECAE6;font-size:.95rem;margin-bottom:.2rem}
.metric-value{font-size:2rem;font-weight:700;color:#00F5D4;text-shadow:0 0 12px rgba(0,245,212,.5)}
.insight-box{
  border-left:3px solid #FF6B6B;background:rgba(255,107,107,.10);
  border-radius:12px;padding:10px 12px;margin:8px 0 16px;color:#E0E0E0
}
</style>""", unsafe_allow_html=True)


# ── Load & Cluster data ───────────────────────────────────
@st.cache_data
def load_and_cluster():
    df = pd.read_csv("preprocessed_marine_plastic.csv")
    df.columns = df.columns.str.strip()

    features = [c for c in df.columns if "Log_" in c] + \
               ["Sea State Normalized", "Latitude", "Longitude"]
    features = [f for f in features if f in df.columns]

    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    df_c = df.loc[X.index].copy()
    df_c["Cluster"] = labels
    log_cd = [c for c in features if "Log_CD" in c]
    df_c["_poll"] = df_c[log_cd].sum(axis=1)
    means = df_c.groupby("Cluster")["_poll"].mean().sort_values()
    zone_map = {
        means.index[0]: "Low Pollution 🟢",
        means.index[1]: "Medium Pollution 🟡",
        means.index[2]: "High Pollution 🔴",
    }
    df_c["Pollution_Zone"] = df_c["Cluster"].map(zone_map)
    df_c = df_c.drop(columns=["_poll"])

    sil = silhouette_score(X_scaled, labels)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    return df, df_c, X_scaled, X_pca, labels, zone_map, sil, scaler, kmeans, features


@st.cache_data
def run_clustering_models(X_scaled):
    results = {}

    # K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(X_scaled)
    results['K-Means'] = {
        'labels': km_labels,
        'n_clusters': len(set(km_labels)),
        'n_noise': 0,
    }

    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    db_labels = dbscan.fit_predict(X_scaled)
    n_noise = int((db_labels == -1).sum())
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    results['DBSCAN'] = {
        'labels': db_labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
    }

    # Hierarchical
    hc = AgglomerativeClustering(n_clusters=3)
    hc_labels = hc.fit_predict(X_scaled)
    results['Hierarchical'] = {
        'labels': hc_labels,
        'n_clusters': len(set(hc_labels)),
        'n_noise': 0,
    }

    for name, info in results.items():
        lbl = info['labels']
        metrics = {
            'Silhouette Score': np.nan,
            'Davies-Bouldin Score': np.nan,
            'Calinski-Harabasz Score': np.nan,
            'Clusters Found': info['n_clusters'],
            'Noise Points': info['n_noise'],
        }
        try:
            if name == 'DBSCAN':
                mask = lbl != -1
                if len(set(lbl[mask])) > 1:
                    metrics['Silhouette Score'] = round(silhouette_score(X_scaled[mask], lbl[mask]), 4)
                    metrics['Davies-Bouldin Score'] = round(davies_bouldin_score(X_scaled[mask], lbl[mask]), 4)
                    metrics['Calinski-Harabasz Score'] = round(calinski_harabasz_score(X_scaled[mask], lbl[mask]), 4)
            else:
                if len(set(lbl)) > 1:
                    metrics['Silhouette Score'] = round(silhouette_score(X_scaled, lbl), 4)
                    metrics['Davies-Bouldin Score'] = round(davies_bouldin_score(X_scaled, lbl), 4)
                    metrics['Calinski-Harabasz Score'] = round(calinski_harabasz_score(X_scaled, lbl), 4)
        except Exception:
            pass
        results[name]['metrics'] = metrics

    return results


df, df_c, X_scaled, X_pca, labels, zone_map, sil, scaler, kmeans, features = load_and_cluster()
model_results = run_clustering_models(X_scaled)


# ── Helpers ───────────────────────────────────────────────
def metric_card(title, value):
    st.markdown(f"""
    <div class="glass-card">
      <div class="metric-title">{title}</div>
      <div class="metric-value">{value}</div>
    </div>""", unsafe_allow_html=True)

def dark_fig(figsize=(9, 4)):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0B1125")
    ax.set_facecolor("#0B1125")
    return fig, ax


# ── Sidebar ───────────────────────────────────────────────
st.markdown('<div style="font-size:2rem;font-weight:700;color:#FFF;text-shadow:0 0 16px rgba(0,245,212,.5)">🌊 Marine Plastic Pollution Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div style="color:#8ECAE6;margin-bottom:1rem">Python for Data Science | CA Component 2 | K-Means, DBSCAN, Hierarchical ♻️🐢</div>', unsafe_allow_html=True)

st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio("Go to page", [
    "🏠 Overview",
    "📊 Statistical Measures",
    "📈 Visualizations",
    "🤖 ML Clusters",
    "🔬 Model Comparison",
    "🔍 Search by Cluster",
    "🔮 Predict My Zone",
    "📋 Data Explorer",
])
st.sidebar.markdown("---")
st.sidebar.markdown("🌐 **Theme:** Dark Ocean Glass")
st.sidebar.markdown(f"🎯 **K-Means Silhouette:** {sil:.4f}")
st.sidebar.markdown("🤖 **Models:** K-Means | DBSCAN | Hierarchical")


# ══════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("## 🏠 Dataset Overview")
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("🧾 Total Rows","589")
    with c2: metric_card("🧱 Columns","28")
    with c3: metric_card("🧼 Missing Values","0")
    with c4: metric_card("🧬 Duplicates","0")

    st.markdown("### 🔄 Before vs After Preprocessing")
    st.dataframe(pd.DataFrame({
        "Metric":["Rows","Columns","Missing Values","Duplicates","Features"],
        "Original":[1572,20,16357,2,"14 columns"],
        "Preprocessed":[589,28,0,0,"28 columns (+13 new)"]
    }))
    st.markdown("### 👀 Dataset Preview")
    st.dataframe(df.head(10))


# ══════════════════════════════════════════════════════════
# PAGE 2 — STATISTICAL MEASURES
# ══════════════════════════════════════════════════════════
elif page == "📊 Statistical Measures":
    st.markdown("## 📊 Statistical Measures")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    col = st.selectbox("Select column:", num_cols)
    s = df[col].dropna()

    a,b,c = st.columns(3)
    with a:
        metric_card("📐 Mean", f"{s.mean():.4f}")
        metric_card("📏 Std Deviation", f"{s.std():.4f}")
    with b:
        metric_card("📍 Median", f"{s.median():.4f}")
        metric_card("📦 Variance", f"{s.var():.4f}")
    with c:
        metric_card("🎯 Mode", f"{s.mode()[0]:.4f}")
        metric_card("📈 Skewness", f"{s.skew():.4f}")

    if s.mean() > s.median():
        st.error(f"⚠️ Mean ({s.mean():.2f}) > Median ({s.median():.2f}) → RIGHT SKEWED → Log transformation applied!")
    else:
        st.success("✅ Data is fairly balanced.")

    st.markdown("### 🔥 Correlation Heatmap")
    corr = df[num_cols].corr(numeric_only=True)
    fig, ax = dark_fig(figsize=(11,7))
    if sns:
        sns.heatmap(corr, cmap="coolwarm", annot=False,
                    linewidths=0.3, linecolor="#1A2446", ax=ax)
    ax.set_title("Feature Correlation Matrix", color="#EAF2FF", fontsize=13)
    st.pyplot(fig)


# ══════════════════════════════════════════════════════════
# PAGE 3 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════
elif page == "📈 Visualizations":
    st.markdown("## 🎨 Visualizations Hub")

    cd1 = next((c for c in df.columns if "CD1" in c and "Log" not in c), None)
    year_col = "Year" if "Year" in df.columns else None
    src_col  = "Source" if "Source" in df.columns else None

    if cd1:
        st.markdown("### 1) 📈 Histogram — CD1 Distribution")
        fig, ax = dark_fig()
        s = pd.to_numeric(df[cd1], errors="coerce").dropna()
        ax.hist(s[s>0], bins=40, color="#00F5D4", edgecolor="#FF6B6B", alpha=0.85)
        ax.set_title("Distribution of CD1"); ax.set_xlabel("CD1 (/km²)"); ax.set_ylabel("Frequency")
        st.pyplot(fig)
        st.markdown('<div class="insight-box">💡 Most zones have near-zero plastic — confirms right-skewed data needing log transform.</div>', unsafe_allow_html=True)

    cd_raw = [c for c in df.columns if c.startswith("CD") and "Log" not in c and "/" in c]
    if len(cd_raw) >= 2:
        st.markdown("### 2) 📦 Box Plot — CD1 to CD4")
        fig, ax = dark_fig()
        ax.boxplot([pd.to_numeric(df[c], errors="coerce").dropna() for c in cd_raw],
                   labels=[c.split()[0] for c in cd_raw], patch_artist=True,
                   boxprops=dict(facecolor="#0FD9BF", color="#00F5D4"),
                   medianprops=dict(color="#FF6B6B", linewidth=2),
                   flierprops=dict(marker="o", markerfacecolor="#FF6B6B", alpha=0.4, markersize=4))
        ax.set_title("Outliers in CD1–CD4"); ax.set_ylabel("Count (/km²)")
        st.pyplot(fig)
        st.markdown('<div class="insight-box">🧪 Outlier dots confirm extreme pollution hotspots in certain ocean regions.</div>', unsafe_allow_html=True)

    if cd1 and year_col:
        st.markdown("### 3) 📅 Average CD1 by Year")
        fig, ax = dark_fig()
        yearly = df.groupby(year_col)[cd1].mean().reset_index()
        ax.bar(yearly[year_col].astype(str), yearly[cd1], color="#00F5D4", edgecolor="#FF6B6B")
        ax.set_title("Average CD1 by Year"); ax.set_xlabel("Year"); ax.set_ylabel("Avg CD1")
        st.pyplot(fig)
        st.markdown('<div class="insight-box">📌 Yearly variation in plastic intensity across 2007–2013.</div>', unsafe_allow_html=True)

    season_cols = [c for c in df.columns if c.startswith("Season_")]
    if season_cols:
        st.markdown("### 4) 🍂 Samples by Season")
        fig, ax = dark_fig()
        counts = {c.replace("Season_",""):(df[c]==1).sum() for c in season_cols}
        ax.bar(counts.keys(), counts.values(), color="#FF6B6B", edgecolor="#00F5D4")
        ax.set_title("Samples by Season"); ax.set_ylabel("Count")
        st.pyplot(fig)
        st.markdown('<div class="insight-box">🍂 Seasonal distribution shows possible temporal bias in data collection.</div>', unsafe_allow_html=True)

    if src_col:
        st.markdown("### 5) 👨‍🔬 Samples by Researcher")
        fig, ax = dark_fig(figsize=(9,5))
        src = df[src_col].value_counts().sort_values()
        ax.barh(src.index, src.values, color="#00F5D4", edgecolor="#FF6B6B")
        ax.set_title("Samples by Researcher"); ax.set_xlabel("Count")
        st.pyplot(fig)
        st.markdown('<div class="insight-box">🧭 M. Eriksen contributed most samples — possible geographic bias.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 4 — ML CLUSTERS (K-Means)
# ══════════════════════════════════════════════════════════
elif page == "🤖 ML Clusters":
    st.markdown("## 🤖 K-Means Clustering Results")
    st.markdown(f"**Algorithm:** K-Means | **Clusters:** 3 | **Silhouette Score:** `{sil:.4f}`")

    zone_counts = df_c["Pollution_Zone"].value_counts()
    col_list = st.columns(3)
    for i,(zone,count) in enumerate(zone_counts.items()):
        color = "#1D9E75" if "Low" in zone else ("#EF9F27" if "Medium" in zone else "#E24B4A")
        with col_list[i]:
            st.markdown(f"""
            <div style="background:rgba(0,0,0,.3);border:2px solid {color};border-radius:14px;
                        padding:20px;text-align:center;margin-bottom:10px">
              <div style="font-size:2.5rem">{zone.split()[-1]}</div>
              <div style="font-size:1.2rem;font-weight:700;color:{color}">{zone.replace(zone.split()[-1],'').strip()}</div>
              <div style="font-size:2rem;font-weight:700;color:#FFF">{count}</div>
              <div style="color:#8ECAE6">ocean zones</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("### 🔵 PCA Cluster Visualization")
    color_map = {0:"#E24B4A", 1:"#EF9F27", 2:"#1D9E75"}
    fig, ax = dark_fig(figsize=(10,6))
    ax.scatter(X_pca[:,0], X_pca[:,1], c=[color_map[l] for l in labels], alpha=0.6, s=40)
    pca2 = PCA(n_components=2)
    pca2.fit(X_scaled)
    centers_pca = pca2.transform(kmeans.cluster_centers_)
    for cx,cy in centers_pca:
        ax.scatter(cx, cy, c="white", s=250, marker="*", edgecolors="black", zorder=5)
    ax.set_title("K-Means Clusters — 2D PCA View", fontsize=13)
    ax.set_xlabel("PCA Component 1"); ax.set_ylabel("PCA Component 2")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#1D9E75", label="Low Pollution 🟢"),
        Patch(facecolor="#EF9F27", label="Medium Pollution 🟡"),
        Patch(facecolor="#E24B4A", label="High Pollution 🔴"),
    ])
    st.pyplot(fig)

    st.markdown("### 🗺️ Global Pollution Map")
    fig, ax = dark_fig(figsize=(12,6))
    for zone, grp in df_c.groupby("Pollution_Zone"):
        color = "#1D9E75" if "Low" in zone else ("#EF9F27" if "Medium" in zone else "#E24B4A")
        ax.scatter(grp["Longitude"], grp["Latitude"], c=color, label=zone, alpha=0.6, s=25)
    ax.set_title("Marine Plastic Pollution Zones — World Map", fontsize=13)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(); ax.grid(alpha=0.2)
    st.pyplot(fig)


# ══════════════════════════════════════════════════════════
# PAGE 5 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════
elif page == "🔬 Model Comparison":
    st.markdown("## 🔬 Unsupervised Model Comparison")
    st.markdown("Comparing **K-Means**, **DBSCAN**, and **Hierarchical Clustering** on the same features.")

    # Metrics Table
    rows = []
    for name, info in model_results.items():
        m = info['metrics']
        rows.append({
            'Model': name,
            'Silhouette Score': m['Silhouette Score'],
            'Davies-Bouldin Score': m['Davies-Bouldin Score'],
            'Calinski-Harabasz Score': m['Calinski-Harabasz Score'],
            'Clusters Found': m['Clusters Found'],
            'Noise Points': m['Noise Points'],
        })
    compare_df = pd.DataFrame(rows)
    st.markdown("### 📊 Performance Metrics Table")
    st.dataframe(compare_df)

    # Winner
    valid = compare_df.dropna(subset=['Silhouette Score'])
    if not valid.empty:
        best_idx = valid['Silhouette Score'].idxmax()
        best_model = valid.loc[best_idx, 'Model']
        best_score = valid.loc[best_idx, 'Silhouette Score']
        st.markdown(f"""
        <div class="glass-card" style="border:2px solid #00F5D4;text-align:center;padding:25px">
          <div style="font-size:1rem;color:#8ECAE6">🏆 Best Model</div>
          <div style="font-size:2rem;font-weight:700;color:#00F5D4">{best_model}</div>
          <div style="color:#8ECAE6">Silhouette Score: {best_score:.4f}</div>
        </div>""", unsafe_allow_html=True)

    # Silhouette Bar Chart
    st.markdown("### 📈 Silhouette Score Comparison")
    fig, ax = dark_fig(figsize=(8,4))
    colors = ["#00F5D4","#FF6B6B","#1D9E75"]
    scores = compare_df['Silhouette Score'].fillna(0).tolist()
    ax.bar(compare_df['Model'], scores, color=colors, edgecolor="white")
    ax.set_title("Silhouette Scores by Model", color="#EAF2FF")
    ax.set_ylabel("Score", color="#EAF2FF")
    ax.tick_params(colors="#EAF2FF")
    st.pyplot(fig)

    # PCA Side by Side
    st.markdown("### 🧭 PCA 2D View — All 3 Models")
    X_pca_c = PCA(n_components=2).fit_transform(X_scaled)
    color_map = {0:"#E24B4A", 1:"#EF9F27", 2:"#1D9E75"}
    fig, axes = plt.subplots(1, 3, figsize=(20,5), facecolor="#0B1125")
    for ax, (name, info) in zip(axes, model_results.items()):
        ax.set_facecolor("#0B1125")
        lbl = info['labels']
        if name == 'DBSCAN':
            mask = lbl != -1
            ax.scatter(X_pca_c[mask,0], X_pca_c[mask,1],
                       c=[color_map.get(l,"#8ECAE6") for l in lbl[mask]], s=30, alpha=0.6)
            ax.scatter(X_pca_c[~mask,0], X_pca_c[~mask,1],
                       c='red', s=20, alpha=0.8, marker='x', label='Noise')
            ax.legend(frameon=False, labelcolor='white')
        else:
            ax.scatter(X_pca_c[:,0], X_pca_c[:,1],
                       c=[color_map.get(l,"#8ECAE6") for l in lbl], s=30, alpha=0.7)
        ax.set_title(name, color="#EAF2FF")
        ax.set_xlabel("PCA 1", color="#EAF2FF")
        ax.set_ylabel("PCA 2", color="#EAF2FF")
        ax.tick_params(colors="#EAF2FF")
    plt.tight_layout()
    st.pyplot(fig)

    # Dendrogram
    st.markdown("### 🔗 Hierarchical Clustering Dendrogram")
    Z = linkage(X_scaled, method='ward')
    fig, ax = dark_fig(figsize=(12,5))
    dendrogram(Z, ax=ax, truncate_mode='level', p=20,
               color_threshold=0, above_threshold_color='#00F5D4')
    ax.set_title("Dendrogram — Top 20 Levels", color="#EAF2FF")
    ax.tick_params(colors="#EAF2FF")
    st.pyplot(fig)

    # DBSCAN Noise Points
    st.markdown("### 🔴 DBSCAN — Noise / Outlier Points")
    db_labels = model_results['DBSCAN']['labels']
    noise_count = int((db_labels == -1).sum())
    metric_card("🔴 Noise Points Detected by DBSCAN", str(noise_count))
    fig, ax = dark_fig(figsize=(10,5))
    mask = db_labels != -1
    ax.scatter(X_pca_c[mask,0], X_pca_c[mask,1],
               c=[color_map.get(l,"#8ECAE6") for l in db_labels[mask]],
               s=30, alpha=0.6, label="Clustered")
    ax.scatter(X_pca_c[~mask,0], X_pca_c[~mask,1],
               c='red', s=50, alpha=0.9, marker='x', label=f"Noise ({noise_count} pts)")
    ax.set_title("DBSCAN — Noise Points in Red", color="#EAF2FF")
    ax.legend(frameon=False, labelcolor='white')
    st.pyplot(fig)

    # Overfitting Analysis
    st.markdown("### 🧠 Overfitting vs Underfitting Analysis")
    st.info("Unsupervised models do not overfit in the same way as supervised models because there is no labeled target to memorize. However underfitting can occur if parameters are poorly set.")
    st.markdown("""
    | Model | Overfitting Risk | Underfitting Risk |
    |---|---|---|
    | **K-Means** | Low — no labels to memorize | Medium — assumes spherical clusters |
    | **DBSCAN** | Low | High — if eps too small, many points become noise |
    | **Hierarchical** | Low | Low — flexible tree structure |
    """)

    # Model Insights
    st.markdown("### 🔎 Model Insights — What Clusters Mean")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="glass-card" style="border:2px solid #1D9E75">
        <div class="metric-title">🟢 Low Pollution Zone</div>
        <div style="color:#E0E0E0;font-size:0.9rem">Ocean regions with near-zero plastic density. Focus: monitoring and prevention.</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="glass-card" style="border:2px solid #EF9F27">
        <div class="metric-title">🟡 Medium Pollution Zone</div>
        <div style="color:#E0E0E0;font-size:0.9rem">Moderate plastic accumulation. Focus: targeted cleanup campaigns.</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="glass-card" style="border:2px solid #E24B4A">
        <div class="metric-title">🔴 High Pollution Zone</div>
        <div style="color:#E0E0E0;font-size:0.9rem">Severe contamination hotspots. Focus: urgent cleanup and policy action.</div>
        </div>""", unsafe_allow_html=True)

    # Business Interpretation
    st.markdown("### 💼 Business Interpretation")
    st.markdown("""
    - **Low zones** → Monitor and protect. Minimal intervention needed.
    - **Medium zones** → Deploy preventive cleanup operations and improve coastal waste management.
    - **High zones** → Immediate cleanup operations, strict marine policy, international cooperation needed.
    - These cluster insights help governments and NGOs **prioritize funding and cleanup resources** efficiently.
    """)


# ══════════════════════════════════════════════════════════
# PAGE 6 — SEARCH BY CLUSTER
# ══════════════════════════════════════════════════════════
elif page == "🔍 Search by Cluster":
    st.markdown("## 🔍 Search & Filter by Pollution Zone")

    zones = ["All Zones"] + sorted(df_c["Pollution_Zone"].unique().tolist())
    selected_zone = st.selectbox("🌊 Select Pollution Zone:", zones)
    years = ["All Years"] + sorted(df_c["Year"].dropna().astype(int).unique().tolist()) if "Year" in df_c.columns else ["All Years"]
    selected_year = st.selectbox("📅 Filter by Year:", years)
    if "Source" in df_c.columns:
        sources = ["All Researchers"] + sorted(df_c["Source"].dropna().unique().tolist())
        selected_source = st.selectbox("👨‍🔬 Filter by Researcher:", sources)
    else:
        selected_source = "All Researchers"

    filtered = df_c.copy()
    if selected_zone != "All Zones":
        filtered = filtered[filtered["Pollution_Zone"] == selected_zone]
    if selected_year != "All Years":
        filtered = filtered[filtered["Year"] == int(selected_year)]
    if selected_source != "All Researchers" and "Source" in filtered.columns:
        filtered = filtered[filtered["Source"] == selected_source]

    st.markdown(f"### 📋 Results — {len(filtered)} ocean zones found")
    if len(filtered) > 0:
        c1,c2,c3 = st.columns(3)
        with c1: metric_card("🔢 Matching Zones", str(len(filtered)))
        with c2:
            cd1_col = next((c for c in filtered.columns if "CD1" in c and "Log" not in c), None)
            if cd1_col: metric_card("🧪 Avg CD1 (/km²)", f"{filtered[cd1_col].mean():.2f}")
        with c3:
            if "Sea State" in filtered.columns:
                metric_card("🌊 Avg Sea State", f"{filtered['Sea State'].mean():.2f}")

        fig, ax = dark_fig(figsize=(10,5))
        for zone, grp in filtered.groupby("Pollution_Zone"):
            color = "#1D9E75" if "Low" in zone else ("#EF9F27" if "Medium" in zone else "#E24B4A")
            ax.scatter(grp["Longitude"], grp["Latitude"], c=color, label=zone, alpha=0.7, s=40)
        ax.set_title("Filtered Ocean Zones Map")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.legend(); ax.grid(alpha=0.2)
        st.pyplot(fig)

        show_cols = ["Latitude","Longitude","Pollution_Zone","Cluster","Year","Source","Sea State"]
        show_cols = [c for c in show_cols if c in filtered.columns]
        st.dataframe(filtered[show_cols])
        st.download_button("⬇️ Download Results",
                           data=filtered.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_clusters.csv", mime="text/csv")
    else:
        st.warning("No results found. Try different filters!")


# ══════════════════════════════════════════════════════════
# PAGE 7 — PREDICT ZONE
# ══════════════════════════════════════════════════════════
elif page == "🔮 Predict My Zone":
    st.markdown("## 🔮 Predict Pollution Zone")

    col1, col2 = st.columns(2)
    with col1:
        latitude  = st.number_input("📍 Latitude",   min_value=-90.0,  max_value=90.0,  value=20.0)
        longitude = st.number_input("📍 Longitude",  min_value=-180.0, max_value=180.0, value=-64.0)
        cd1 = st.number_input("CD1 (/km²)", min_value=0.0, value=10.0)
        cd2 = st.number_input("CD2 (/km²)", min_value=0.0, value=5.0)
        cd3 = st.number_input("CD3 (/km²)", min_value=0.0, value=2.0)
        cd4 = st.number_input("CD4 (/km²)", min_value=0.0, value=1.0)
    with col2:
        wd1 = st.number_input("WD1 (g/km²)", min_value=0.0, value=50.0)
        wd2 = st.number_input("WD2 (g/km²)", min_value=0.0, value=30.0)
        wd3 = st.number_input("WD3 (g/km²)", min_value=0.0, value=20.0)
        wd4 = st.number_input("WD4 (g/km²)", min_value=0.0, value=10.0)
        sea = st.slider("🌊 Sea State (0–15)", 0, 15, 3)

    if st.button("🔮 Predict Pollution Zone", type="primary"):
        log_vals = [np.log1p(v) for v in [cd1,cd2,cd3,cd4,wd1,wd2,wd3,wd4]]
        sea_norm = sea / 15.0
        input_arr = np.array([*log_vals, sea_norm, latitude, longitude]).reshape(1,-1)
        input_arr = input_arr[:, :len(features)]
        input_scaled = scaler.transform(input_arr)
        cluster = kmeans.predict(input_scaled)[0]
        zone = zone_map.get(cluster, f"Cluster {cluster}")
        color = "#1D9E75" if "Low" in zone else ("#EF9F27" if "Medium" in zone else "#E24B4A")
        emoji = zone.split()[-1]
        st.markdown(f"""
        <div style="background:rgba(0,0,0,.4);border:3px solid {color};border-radius:18px;
                    padding:40px;text-align:center;margin-top:20px">
          <div style="font-size:5rem">{emoji}</div>
          <div style="font-size:1.8rem;font-weight:700;color:{color};margin-top:10px">{zone}</div>
          <div style="color:#8ECAE6;margin-top:8px">Cluster ID: {cluster}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 8 — DATA EXPLORER
# ══════════════════════════════════════════════════════════
elif page == "📋 Data Explorer":
    st.markdown("## 📋 Data Explorer")
    years = ["All"] + sorted(df["Year"].dropna().astype(int).unique().tolist()) if "Year" in df.columns else ["All"]
    sel_year = st.selectbox("Filter by Year:", years)
    filtered = df if sel_year == "All" else df[df["Year"] == int(sel_year)]
    st.write(f"Showing **{len(filtered)}** rows")
    st.dataframe(filtered)
    st.download_button("⬇️ Download CSV",
                       data=filtered.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_data.csv", mime="text/csv")

st.markdown("---")
st.markdown("**Marine Plastic Waste Management | Python for Data Science | CA Component 2**")Science | CA Component 2**")
