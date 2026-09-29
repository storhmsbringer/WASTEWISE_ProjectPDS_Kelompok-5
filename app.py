import html
import re
from turtle import left
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from scipy.stats import kstest, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DEFAULT_DATA_PATH = Path(__file__).parent / "Dataset_Jawa_Tengah.xlsx"

# Palette
SAFETY_ORANGE = "#FF6B00"
TAUPE_BROWN = "#8C7C68"
EARTHY_KHAKI = "#BFAFA0"
MUTED_BLUE = "#4A90E2"
CHARCOAL_GRAY = "#333333"
SOFT_OFF_WHITE = "#F9F9F9"
MUSTARD_YELLOW = "#E5A93B"

BG_APP = "#1A1A1A"
BG_CARD = CHARCOAL_GRAY
BG_CARD_ALT = "#3D3D3D"
TEXT_PRIMARY = SOFT_OFF_WHITE
TEXT_MUTED = EARTHY_KHAKI
GRID_COLOR = "rgba(191,175,160,0.15)"

PALETTE_CATEGORICAL = [MUTED_BLUE, SAFETY_ORANGE, MUSTARD_YELLOW, EARTHY_KHAKI, TAUPE_BROWN]

WIPI_CATEGORY_COLORS = {
    "Prioritas sangat tinggi": SAFETY_ORANGE,
    "Prioritas tinggi": MUSTARD_YELLOW,
    "Prioritas sedang": EARTHY_KHAKI,
    "Prioritas rendah": MUTED_BLUE,
    "Prioritas sangat rendah": TAUPE_BROWN,
}
KATEGORI_ORDER_ASC = list(reversed(WIPI_CATEGORY_COLORS.keys()))
KATEGORI_ORDER_DESC = list(WIPI_CATEGORY_COLORS.keys())

PAGES = ["Beranda", "Profil Wilayah", "Klasterisasi", "Analisis Gap", "Prioritas Intervensi", "Rekomendasi"]

SUMBER = ["rumah_tangga", "perkantoran", "perniagaan", "pasar", "fasilitas_publik", "lainnya", "kawasan"]
KOMPOSISI = ["Sisa Makanan", "Kayu Ranting", "Kertas/Karton", "Plastik", "Logam", "Kain", "Karet/Kulit", "Kaca", "Lainnya"]
PCA_VARS = ["WGPC", "UWR", "RR", "Organic", "Recyclable", "Household", "Handling", "Density"]
WIPI_VARS = ["UWR", "WGPC", "MG", "MP"]

PROFIL_DEF = {
    1: "Pengomposan / biodigester / pemilahan organik dari sumber",
    2: "Source segregation + MRF + penguatan pengumpulan bahan daur ulang",
    3: "Waste reduction + efisiensi + optimalisasi",
    4: "Perluasan cakupan pengumpulan + fasilitas dasar + pemilahan dari sumber",
}

KOMPOSISI_GROUP_COLORS = {
    "Sisa makanan": SAFETY_ORANGE,
    "Plastik": MUTED_BLUE,
    "Kertas/karton": MUSTARD_YELLOW,
    "Kayu/ranting": EARTHY_KHAKI,
    "Lainnya": TAUPE_BROWN,
}

# Istilah asing yang otomatis dicetak miring di seluruh teks tampilan
FOREIGN_TERMS = sorted(
    [
        "Waste Intervention Priority Index",
        "Waste Management Gap",
        "Waste composition index",
        "Management Gap",
        "Material Pressure",
        "composition index",
        "Recycling rate",
        "Source segregation",
        "Waste reduction",
        "Monitoring",
        "silhouette score",
        "elbow method",
        "P-value",
        "Household",
        "Recyclable",
        "Organic",
        "Handling",
        "Density",
        "Gap",
    ],
    key=len,
    reverse=True,
)
FOREIGN_RE = re.compile(r"(?<!\w)(" + "|".join(re.escape(t) for t in FOREIGN_TERMS) + r")(?!\w)", re.IGNORECASE)


def ital(text):
    return FOREIGN_RE.sub(lambda m: f"<i>{m.group(0)}</i>", str(text))


# Format angka gaya Indonesia (koma desimal, titik ribuan)
def id_num(x, decimals=2):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "-"
    s = f"{x:,.{decimals}f}"
    s = s.replace(",", "§").replace(".", ",").replace("§", ".")
    return s


def id_pct(x, decimals=2):
    return f"{id_num(x, decimals)}%"


def id_jt(x, decimals=2):
    return f"{id_num(x / 1_000_000, decimals)} jt"


def id_signed(x, decimals=2):
    sign = "+" if x >= 0 else "-"
    return f"{sign}{id_num(abs(x), decimals)}"


def id_sci(x, decimals=2):
    s = f"{x:.{decimals}e}"
    mant, exp = s.split("e")
    return f"{mant.replace('.', ',')}e{exp}"


def label_profil(p):
    if p == "-":
        return "Monitoring rutin"
    if p == "Darurat / Kompleks":
        return "Darurat/Kompleks"
    parts = [x.strip() for x in str(p).split(",")]
    return "Profil " + "+".join(parts)


def cluster_insight(tinggi, rendah):
    t, r = set(tinggi), set(rendah)
    if not t and r == {"Recyclable"}:
        return "Rendah pemanfaatan anorganik daur ulang"
    if t == {"WGPC"} and {"Household", "Density"} <= r:
        return "Timbulan tinggi, kepadatan rendah"
    if t == {"UWR"} and {"RR", "Handling", "MR"} <= r:
        return "Kapasitas penanganan rendah"
    if {"RR", "Recyclable", "Handling", "MR"} <= t and {"UWR", "Organic"} <= r:
        return "Kinerja pengelolaan terbaik"
    parts = []
    if t:
        parts.append(f"{', '.join(sorted(t))} tinggi")
    if r:
        parts.append(f"{', '.join(sorted(r))} rendah")
    return "; ".join(parts) if parts else "Profil mendekati rata-rata provinsi"


# Pipeline data: bersih -> indikator -> PCA -> klaster -> regresi gap -> WIPI -> rekomendasi
@st.cache_data(show_spinner="Memproses data timbulan dan pengelolaan sampah...")
def compute_wastewise(source) -> dict:
    raw = pd.read_excel(source, sheet_name="datafix", header=None)
    cols = list(raw.iloc[0, :11]) + list(raw.iloc[1, 11:])
    df = raw.iloc[2:].copy()
    df.columns = cols
    df = df.dropna(subset=["Tahun"]).reset_index(drop=True)
    for c in df.columns[2:]:
        df[c] = pd.to_numeric(df[c])
    df["Tahun"] = df["Tahun"].astype(int)
    df = df.rename(columns={
        "Kab/Kota": "KabKota",
        "Jumlah Penduduk (Jiwa)": "Population",
        "Kepadatan Penduduk per km persegi (Km2)": "Density",
    })
    df["KabKota"] = df["KabKota"].str.strip()
    df = df.sort_values(["KabKota", "Tahun"]).reset_index(drop=True)

    wg = df["jml_timbulan_tahun"]
    tot_komp = df[KOMPOSISI].sum(axis=1)
    df[KOMPOSISI] = df[KOMPOSISI].div(tot_komp, axis=0).mul(wg, axis=0)

    df["WG"] = df["jml_timbulan_tahun"]
    df["WGPC"] = df["WG"] * 1000 / df["Population"]
    df["RR"] = (df["jml_daur_ulang"] + df["jml_bahan_baku"]) / df["WG"] * 100
    df["MR"] = df["jml_kelola"] / df["WG"] * 100
    df["UWR"] = 100 - df["MR"]
    df["Household"] = df["rumah_tangga"] / df[SUMBER].sum(axis=1) * 100
    df["Handling"] = df["persen_kelola"]

    df["K_org"] = (df["Sisa Makanan"] + df["Kayu Ranting"]) / df["WG"]
    df["K_rec"] = (df["Kertas/Karton"] + df["Plastik"] + df["Logam"] + df["Kaca"]) / df["WG"]
    df["K_res"] = 1 - df["K_org"] - df["K_rec"]
    df["WCI"] = df["K_org"] + df["K_rec"]
    df["R_ratio"] = df["K_org"] / df["K_rec"]
    df["Organic"] = df["K_org"] * 100
    df["Recyclable"] = df["K_rec"] * 100
    df["Residue"] = df["K_res"] * 100
    df["MP"] = df["Organic"] + df["Recyclable"]
    df["Fokus_komposisi"] = np.where(df["R_ratio"] > 1, "Pengolahan organik", "Bank sampah / TPS3R")
    df["Label_Tahun"] = df["KabKota"] + " '" + df["Tahun"].astype(str).str.slice(-2)

    # ---- PCA ----
    Z = StandardScaler().fit_transform(df[PCA_VARS])
    pca_full = PCA(random_state=RANDOM_STATE).fit(Z)
    eig = pca_full.explained_variance_
    evr = pca_full.explained_variance_ratio_
    n_pc = max(2, int((eig > 1).sum()))

    pca = PCA(n_components=n_pc, random_state=RANDOM_STATE).fit(Z)
    scores = pca.transform(Z)
    pc_cols = [f"PC{i + 1}" for i in range(n_pc)]
    for i, c in enumerate(pc_cols):
        df[c] = scores[:, i]
    loadings = pd.DataFrame(pca.components_.T, index=PCA_VARS, columns=pc_cols)
    var_table = pd.DataFrame({
        "Komponen": pc_cols,
        "Variansi (%)": evr[:n_pc] * 100,
        "Kumulatif (%)": np.cumsum(evr[:n_pc]) * 100,
    })

    # ---- Klasterisasi ----
    ks = list(range(2, 7))
    sil_rows = []
    for k in ks:
        km_k = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20).fit(df[pc_cols].values)
        sil_rows.append({"K": k, "WCSS": km_k.inertia_, "Silhouette": silhouette_score(df[pc_cols].values, km_k.labels_)})
    sil_table = pd.DataFrame(sil_rows)
    k_opt = int(sil_table.loc[sil_table["Silhouette"].idxmax(), "K"])

    km = KMeans(n_clusters=k_opt, random_state=RANDOM_STATE, n_init=20).fit(df[pc_cols].values)
    df["Cluster"] = km.labels_ + 1

    zprof_vars = PCA_VARS + ["MR"]
    zprof = (df.groupby("Cluster")[zprof_vars].mean() - df[zprof_vars].mean()) / df[zprof_vars].std()
    cluster_summary = {}
    for k_, row in zprof.iterrows():
        tinggi = list(row[row > 0.5].index)
        rendah = list(row[row < -0.5].index)
        cluster_summary[int(k_)] = {
            "n": int((df["Cluster"] == k_).sum()),
            "tinggi": tinggi,
            "rendah": rendah,
            "insight": cluster_insight(tinggi, rendah),
        }

    # ---- Regresi & Management Gap ----
    reg_cols = pc_cols[:3] if n_pc >= 3 else pc_cols
    X_reg = sm.add_constant(df[reg_cols])
    y_reg = df["persen_kelola"]
    ols = sm.OLS(y_reg, X_reg).fit()
    df["Expected_MR"] = ols.predict(X_reg)
    df["Management_Gap"] = df["Expected_MR"] - df["persen_kelola"]
    df["MG"] = df["Management_Gap"]

    loo_pred = cross_val_predict(LinearRegression(), df[["Expected_MR"]], df["persen_kelola"], cv=LeaveOneOut())
    loocv_r2 = r2_score(df["persen_kelola"], loo_pred)
    loocv_mae = mean_absolute_error(df["persen_kelola"], loo_pred)

    residuals = ols.resid
    residuals_z = (residuals - residuals.mean()) / residuals.std()
    ks_stat, ks_p = kstest(residuals_z, "norm")
    vif = {c: variance_inflation_factor(X_reg.values, i) for i, c in enumerate(X_reg.columns) if c != "const"}

    ols_info = {
        "params": ols.params.to_dict(),
        "reg_cols": reg_cols,
        "r2": ols.rsquared,
        "r2_adj": ols.rsquared_adj,
        "fvalue": ols.fvalue,
        "f_pvalue": ols.f_pvalue,
        "n": int(ols.nobs),
        "vif": vif,
        "ks_stat": ks_stat,
        "ks_p": ks_p,
        "loocv_r2": loocv_r2,
        "loocv_mae": loocv_mae,
    }

    # ---- WIPI ----
    def mm(s):
        return (s - s.min()) / (s.max() - s.min())

    for v in WIPI_VARS:
        df[v + "_s"] = mm(df[v])
    s_cols = [v + "_s" for v in WIPI_VARS]

    pca_w = PCA(random_state=RANDOM_STATE).fit(StandardScaler().fit_transform(df[WIPI_VARS]))
    w_raw = (np.abs(pca_w.components_) * pca_w.explained_variance_ratio_[:, None]).sum(axis=0)
    w_pca = w_raw / w_raw.sum()
    weights = pd.Series(w_pca, index=WIPI_VARS)

    df["WIPI"] = 100 * (df[s_cols].values @ w_pca)
    bins = [-0.001, 20, 40, 60, 80, 100]
    df["Kategori_WIPI"] = pd.cut(df["WIPI"], bins=bins, labels=KATEGORI_ORDER_ASC)
    df["Rank_WIPI"] = df.groupby("Tahun")["WIPI"].rank(ascending=False, method="min").astype(int)

    def wipi_score(w):
        return 100 * (df[s_cols].values @ np.asarray(w))

    scenarios = {"Bobot sama rata": [0.25] * 4}
    for i, v in enumerate(WIPI_VARS):
        w = [0.2] * 4
        w[i] = 0.4
        scenarios[f"{v} dominan (40%)"] = w
    sens_rows = [{"Skenario": nama, "Spearman": spearmanr(df["WIPI"], wipi_score(w))[0]} for nama, w in scenarios.items()]
    rng = np.random.default_rng(RANDOM_STATE)
    sim = np.array([spearmanr(df["WIPI"], wipi_score(rng.dirichlet(np.ones(4))))[0] for _ in range(2000)])
    sens_rows.append({"Skenario": "Bobot acak (median, n=2000)", "Spearman": float(np.median(sim))})
    sens_table = pd.DataFrame(sens_rows)
    sens_extra = {"min": float(sim.min()), "p5": float(np.percentile(sim, 5))}

    # ---- Rekomendasi ----
    med = df[["Organic", "RR", "MG", "Recyclable", "UWR", "WGPC", "MR"]].median()

    def hi(r, v):
        return r[v] > med[v]

    def lo(r, v):
        return r[v] <= med[v]

    rules = {
        1: lambda r: hi(r, "Organic") and lo(r, "RR") and hi(r, "MG"),
        2: lambda r: hi(r, "Recyclable") and lo(r, "RR") and hi(r, "UWR"),
        3: lambda r: hi(r, "WGPC") and hi(r, "MR"),
        4: lambda r: lo(r, "WGPC") and hi(r, "MG"),
    }

    def recommend(r):
        cocok = [k for k, f in rules.items() if f(r)]
        if not cocok:
            if r["Kategori_WIPI"] in ["Prioritas tinggi", "Prioritas sangat tinggi"]:
                return pd.Series({
                    "Profil": "Darurat / Kompleks",
                    "Rekomendasi": "Optimalisasi TPS3R regional + penguatan penanganan kawasan darurat",
                })
            return pd.Series({"Profil": "-", "Rekomendasi": "Monitoring rutin dan evaluasi berkala"})
        return pd.Series({
            "Profil": ", ".join(map(str, cocok)),
            "Rekomendasi": " | ".join(PROFIL_DEF[k] for k in cocok),
        })

    df[["Profil", "Rekomendasi"]] = df.apply(recommend, axis=1)
    df["Profil_Label"] = df["Profil"].map(label_profil)

    return {
        "df": df,
        "n_pc": n_pc,
        "pc_cols": pc_cols,
        "loadings": loadings,
        "var_table": var_table,
        "sil_table": sil_table,
        "k_opt": k_opt,
        "cluster_summary": cluster_summary,
        "ols": ols_info,
        "wipi_weights": weights,
        "sensitivity": sens_table,
        "sensitivity_extra": sens_extra,
        "median_thresholds": med,
    }


# CSS
def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {BG_APP}; }}
        section[data-testid="stSidebar"] {{ background-color: {BG_CARD}; }}
        .block-container {{ padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1440px; }}

        .ww-card {{
            background-color: {BG_CARD};
            border-radius: 10px;
            padding: 1.05rem 1.25rem 1.25rem 1.25rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(191,175,160,0.10);
        }}
        .ww-kpi {{
            background-color: {BG_CARD};
            border-radius: 8px;
            padding: 0.95rem 1.05rem;
            border-left: 4px solid {EARTHY_KHAKI};
            height: 7.5rem;
            box-sizing: border-box;
            overflow: hidden;
        }}
        .ww-kpi.accent {{ border-left: 4px solid {SAFETY_ORANGE}; }}
        .ww-kpi-label {{
            color: {EARTHY_KHAKI};
            font-size: 0.80rem;
            margin-bottom: 0.35rem;
            line-height: 1.25;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .ww-kpi-value {{
            color: {SOFT_OFF_WHITE};
            font-size: clamp(1.25rem, 1.7vw, 1.8rem);
            font-weight: 700;
            line-height: 1.1;
            white-space: nowrap;
        }}
        .ww-kpi-value.accent {{ color: {SAFETY_ORANGE}; }}
        .ww-kpi-sub {{
            color: {EARTHY_KHAKI};
            font-size: 0.72rem;
            font-style: italic;
            line-height: 1.2;
            min-height: 2.4em;
            margin-top: 0.3rem;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }}

        .ww-section-header {{
            color: {SOFT_OFF_WHITE};
            font-weight: 700;
            font-size: 1.12rem;
            margin-bottom: 0.9rem;
        }}

        .ww-caption {{ color: {EARTHY_KHAKI}; font-size: 0.86rem; margin-bottom: 0.4rem; }}

        .ww-badge {{
            background-color: {BG_CARD_ALT};
            color: {SAFETY_ORANGE};
            font-weight: 700;
            border-radius: 999px;
            padding: 0.55rem 1.4rem;
            text-align: center;
            font-size: 0.85rem;
        }}

        .ww-tinggi {{ color: {SAFETY_ORANGE}; font-weight: 600; font-size: 0.85rem; }}
        .ww-rendah {{ color: {MUTED_BLUE}; font-weight: 600; font-size: 0.85rem; }}
        .ww-muted {{ color: {EARTHY_KHAKI}; font-style: italic; font-size: 0.85rem; }}
        .ww-insight {{ color: {SOFT_OFF_WHITE}; font-weight: 700; font-size: 0.9rem; margin-top: 0.7rem; }}
        .ww-heading-blue {{ color: {MUTED_BLUE}; font-weight: 700; margin-top: 0.6rem; }}
        .ww-formula {{ color: {SAFETY_ORANGE}; font-size: 1.5rem; font-weight: 700; text-align: center; margin: 0.5rem 0; }}
        .ww-note {{ color: {EARTHY_KHAKI}; font-size: 0.82rem; font-style: italic; }}
        .ww-center {{ text-align: center; }}
        .ww-bold {{ font-weight: 700; }}

        .ww-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
        .ww-table th {{
            background-color: {TAUPE_BROWN};
            color: {SOFT_OFF_WHITE};
            text-align: left;
            padding: 0.5rem 0.75rem;
            font-weight: 700;
        }}
        .ww-table td {{
            background-color: {BG_CARD};
            color: {SOFT_OFF_WHITE};
            padding: 0.55rem 0.75rem;
            border-bottom: 1px solid rgba(191,175,160,0.15);
            vertical-align: top;
        }}

        .ww-brand {{
            background-color: {SAFETY_ORANGE};
            border-radius: 10px;
            padding: 0.9rem 1rem;
            text-align: center;
            box-shadow: 0 4px 14px rgba(255,107,0,0.35);
        }}
        .ww-brand-name {{ color: {BG_APP}; font-weight: 800; font-size: 1.35rem; letter-spacing: 1.2px; }}
        .ww-sidebar-line {{ border: none; border-top: 1px solid rgba(191,175,160,0.35); margin: 0.9rem 0; }}
        .ww-sidebar-spacer {{ height: calc(100vh - 640px); min-height: 1.5rem; }}
        .ww-sidebar-footer {{ color: {EARTHY_KHAKI}; font-size: 0.78rem; line-height: 1.7; }}

        div[data-testid="stDataFrame"] {{ border-radius: 8px; overflow: hidden; }}

        div[data-testid="stPills"] button {{
            background-color: {BG_CARD_ALT} !important;
            border: none !important;
            color: {SAFETY_ORANGE} !important;
            font-weight: 700 !important;
            border-radius: 999px !important;
        }}
        div[data-testid="stPills"] button[aria-pressed="true"],
        div[data-testid="stPills"] button[aria-checked="true"] {{
            background-color: {SAFETY_ORANGE} !important;
            color: {BG_APP} !important;
        }}

        div[data-testid="stSelectbox"] > div > div {{
            background-color: {BG_CARD_ALT};
            border-radius: 999px;
            border: none;
        }}

        div[role="radiogroup"] label {{
            padding: 0.4rem 0.7rem;
            border-radius: 6px;
            margin-bottom: 0.15rem;
        }}
        div[role="radiogroup"] label:has(input:checked) {{
            background-color: {BG_CARD_ALT};
            border-left: 4px solid {SAFETY_ORANGE};
        }}

        h1, h2, h3, h4, p, span, label, div {{ color: {SOFT_OFF_WHITE}; }}
        hr {{ border-color: rgba(191,175,160,0.35); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# UI
def kpi_card(label, value, sub=None, accent=False):
    cls = "ww-kpi accent" if accent else "ww-kpi"
    vcls = "ww-kpi-value accent" if accent else "ww-kpi-value"
    sub_text = ital(sub) if sub else "&nbsp;"
    st.markdown(
        f'<div class="{cls}"><div class="ww-kpi-label">{ital(label)}</div>'
        f'<div class="{vcls}">{value}</div><div class="ww-kpi-sub">{sub_text}</div></div>',
        unsafe_allow_html=True,
    )


def section_header(text):
    st.markdown(f'<div class="ww-section-header">{ital(text)}</div>', unsafe_allow_html=True)


def caption(text):
    st.markdown(f'<div class="ww-caption">{ital(text)}</div>', unsafe_allow_html=True)


def badge(text):
    st.markdown(f'<div class="ww-badge">{text}</div>', unsafe_allow_html=True)


def html_table(df):
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{ital(html.escape(str(v)))}</td>" for v in row) + "</tr>"
        for row in df.itertuples(index=False, name=None)
    )
    st.markdown(f'<table class="ww-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>', unsafe_allow_html=True)


def style_fig(fig, height=300, showlegend=False):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_PRIMARY, size=12, family="Segoe UI, Arial, sans-serif"),
        margin=dict(l=8, r=8, t=10, b=8),
        height=height,
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=TEXT_MUTED, size=11)),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color=TEXT_MUTED, tickfont=dict(size=11))
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color=TEXT_MUTED, tickfont=dict(size=11))
    return fig


def wipi_gauge(score, kategori):
    color = WIPI_CATEGORY_COLORS.get(kategori, SAFETY_ORANGE)
    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=score,
            gauge={
                "axis": {"range": [0, 100], "tickcolor": EARTHY_KHAKI, "tickfont": {"color": EARTHY_KHAKI, "size": 10}},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 20], "color": "#292929"},
                    {"range": [20, 40], "color": "#2E2E2E"},
                    {"range": [40, 60], "color": "#333333"},
                    {"range": [60, 80], "color": "#383838"},
                    {"range": [80, 100], "color": "#3D3D3D"},
                ],
            },
        )
    )
    fig.add_annotation(x=0.5, y=0.32, text=f"<b>{id_num(score, 2)}</b>", showarrow=False, font=dict(size=36, color=SOFT_OFF_WHITE))
    fig.add_annotation(x=0.5, y=0.04, text=str(kategori), showarrow=False, font=dict(size=14, color=color))
    style_fig(fig, height=260)
    return fig


def chart_bar_config():
    return {"displayModeBar": False}


# Halaman: Beranda
def render_beranda(data):
    df = data["df"]
    years = sorted(df["Tahun"].unique().tolist())

    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        st.markdown("## Dashboard Prioritas Pengelolaan Sampah Jawa Tengah")
        st.caption("WASTEWISE - Waste Analytics System for Targeted Efficiency & Sustainable Intervention")
    with right:
        tahun = st.pills("Tahun", years, default=years[-1], selection_mode="single", required=True,
                        label_visibility="collapsed", key="tahun_beranda")
    st.divider()

    dyear = df[df["Tahun"] == tahun]
    total_timbulan = dyear["WG"].sum()
    pct_kelola = dyear["jml_kelola"].sum() / dyear["WG"].sum() * 100
    pct_tak_kelola = 100 - pct_kelola
    wgpc_avg = dyear["WGPC"].mean()
    rr_avg = dyear["RR"].mean()
    wci_avg = dyear["WCI"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        kpi_card("Total timbulan", id_jt(total_timbulan), f"ton/tahun, {tahun}", accent=True)
    with c2:
        kpi_card("Terkelola", id_pct(pct_kelola), "dari total timbulan")
    with c3:
        kpi_card("Tidak terkelola", id_pct(pct_tak_kelola), "dari total timbulan", accent=True)
    with c4:
        kpi_card("WGPC rata-rata", id_num(wgpc_avg), "kg/orang/tahun")
    with c5:
        kpi_card("Recycling rate", id_pct(rr_avg), "rata-rata provinsi", accent=True)
    with c6:
        kpi_card("WCI rata-rata", id_num(wci_avg, 3), "composition index")

    st.write("")
    col_main, col_side = st.columns([1.4, 1])

    with col_main:
        yearly = df.groupby("Tahun")[["WGPC", "RR", "UWR", "WCI"]].mean().reset_index()
        uwr_start, uwr_end = yearly["UWR"].iloc[0], yearly["UWR"].iloc[-1]
        arah = "melebar" if uwr_end > uwr_start else "menyempit"
        section_header(
            f"Kesenjangan pengelolaan {arah}: UWR {'naik' if uwr_end > uwr_start else 'turun'} dari "
            f"{id_pct(uwr_start, 1)} ({years[0]}) ke {id_pct(uwr_end, 1)} ({years[-1]})"
        )
        fig = go.Figure()
        specs = [("WGPC", 0.1, SAFETY_ORANGE, "WGPC/10"), ("RR", 1, MUTED_BLUE, "RR"),
                 ("UWR", 1, MUSTARD_YELLOW, "UWR"), ("WCI", 100, EARTHY_KHAKI, "WCI x100")]
        for col, mult, color, name in specs:
            fig.add_trace(go.Scatter(x=yearly["Tahun"], y=yearly[col] * mult, mode="lines+markers",
                                      name=name, line=dict(color=color, width=3), marker=dict(size=7)))
        fig.update_xaxes(dtick=1)
        fig.update_yaxes(range=[0, 100])
        style_fig(fig, height=250, showlegend=True)
        st.plotly_chart(fig, config=chart_bar_config())

        komp = dyear[KOMPOSISI].sum()
        grp = pd.Series({
            "Sisa makanan": komp["Sisa Makanan"],
            "Plastik": komp["Plastik"],
            "Kertas/karton": komp["Kertas/Karton"],
            "Kayu/ranting": komp["Kayu Ranting"],
            "Lainnya": komp[["Logam", "Kain", "Karet/Kulit", "Kaca", "Lainnya"]].sum(),
        })
        grp_pct = grp / grp.sum() * 100
        top_kategori = grp_pct.idxmax()
        section_header(f"{top_kategori} mendominasi {id_pct(grp_pct.max(), 1)} komposisi sampah {tahun}")
        fig2 = go.Figure(go.Pie(labels=grp.index, values=grp.values, hole=0.35,
                                 marker=dict(colors=[KOMPOSISI_GROUP_COLORS[k] for k in grp.index]),
                                 textinfo="none", sort=False))
        style_fig(fig2, height=280, showlegend=True)
        fig2.update_layout(legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02))
        st.plotly_chart(fig2, config=chart_bar_config())

    with col_side:
        top5 = dyear.sort_values("WIPI", ascending=False).head(5)
        pemimpin = top5.iloc[0]
        section_header(f"{pemimpin['KabKota']} prioritas intervensi tertinggi {tahun}, skor WIPI {id_num(pemimpin['WIPI'])}")
        fig3 = go.Figure(go.Bar(x=top5["WIPI"], y=top5["KabKota"], orientation="h",
                                 marker_color=SAFETY_ORANGE, text=top5["WIPI"].round(0).astype(int), textposition="outside"))
        fig3.update_yaxes(autorange="reversed")
        fig3.update_xaxes(range=[0, 95])
        style_fig(fig3, height=560)
        st.plotly_chart(fig3, config=chart_bar_config())


# Halaman: Profil Wilayah
def render_profil_wilayah(data):
    df = data["df"]
    years = sorted(df["Tahun"].unique().tolist())
    kota_list = sorted(df["KabKota"].unique().tolist())
    default_kota = df[df["Tahun"] == years[-1]].sort_values("WIPI", ascending=False).iloc[0]["KabKota"]

    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        title_ph = st.empty()
        subtitle_ph = st.empty()
    with right:
        tahun = st.pills("Tahun", years, default=years[-1], selection_mode="single", required=True,
                          label_visibility="collapsed", key="tahun_profil")
        kota = st.selectbox("Kabupaten/Kota", kota_list, index=kota_list.index(default_kota),
                             label_visibility="collapsed", key="kota_profil")
    st.divider()

    row = df[(df["Tahun"] == tahun) & (df["KabKota"] == kota)].iloc[0]
    title_ph.markdown("## Profil Wilayah")
    subtitle_ph.caption(f"Contoh: {kota}, peringkat prioritas #{int(row['Rank_WIPI'])} tahun {tahun}")

    dyear = df[df["Tahun"] == tahun]

    col_left, col_right = st.columns([1, 1.7])
    with col_left:
        section_header(f"{kota} berstatus {row['Kategori_WIPI']}")
        st.plotly_chart(wipi_gauge(row["WIPI"], row["Kategori_WIPI"]), config=chart_bar_config())

        prov_avg = dyear[["WGPC", "RR", "UWR"]].mean()
        cats = ["WGPC/10", "RR", "UWR"]
        kota_vals = [row["WGPC"] / 10, row["RR"], row["UWR"]]
        prov_vals = [prov_avg["WGPC"] / 10, prov_avg["RR"], prov_avg["UWR"]]
        diffs = [k - p for k, p in zip(kota_vals, prov_vals)]
        i_max = int(np.argmax(np.abs(diffs)))
        arah = "lebih tinggi" if diffs[i_max] > 0 else "lebih rendah"
        section_header(f"{kota} {arah} {id_num(abs(diffs[i_max]), 1)} poin pada {cats[i_max]} dibanding rata-rata provinsi")
        figc = go.Figure()
        figc.add_trace(go.Bar(x=cats, y=kota_vals, name=kota, marker_color=SAFETY_ORANGE))
        figc.add_trace(go.Bar(x=cats, y=prov_vals, name="Rata-rata provinsi", marker_color=MUTED_BLUE))
        style_fig(figc, height=280, showlegend=True)
        st.plotly_chart(figc, config=chart_bar_config())

    with col_right:
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi_card("WGPC", id_num(row["WGPC"]), "kg/orang/tahun")
        with k2:
            kpi_card("UWR", id_pct(row["UWR"]), "tidak terkelola", accent=True)
        with k3:
            kpi_card("RR", id_pct(row["RR"]), "recycling rate")
        with k4:
            kpi_card("WCI", id_num(row["WCI"], 2), "composition index", accent=True)

        st.write("")
        section_header("Rekomendasi intervensi")
        st.markdown(
            f'<div class="ww-card"><span class="ww-bold">{row["Profil_Label"]}</span><br/>{ital(row["Rekomendasi"])}</div>',
            unsafe_allow_html=True,
        )

        sorted_df = dyear.sort_values("WIPI", ascending=False).reset_index(drop=True)
        idx = sorted_df.index[sorted_df["KabKota"] == kota][0]
        if idx + 6 <= len(sorted_df):
            comp = sorted_df.iloc[idx + 1: idx + 6]
            comp_range = f"peringkat {idx + 2}-{idx + 6}"
        else:
            start = max(0, idx - 5)
            comp = sorted_df.iloc[start:idx]
            comp_range = f"peringkat {start + 1}-{idx}"
        section_header(f"Lima wilayah pembanding ({comp_range}, {tahun})")
        show = comp[["KabKota", "WGPC", "UWR", "RR", "WIPI"]].rename(columns={"KabKota": "Kabupaten/Kota"})
        show["WGPC"] = show["WGPC"].map(lambda v: id_num(v))
        show["UWR"] = show["UWR"].map(lambda v: id_pct(v))
        show["RR"] = show["RR"].map(lambda v: id_pct(v))
        show["WIPI"] = show["WIPI"].map(lambda v: id_num(v))
        st.dataframe(show, hide_index=True)


# Halaman: Klasterisasi
def render_klasterisasi(data):
    df = data["df"]
    sil = data["sil_table"]
    k_opt = data["k_opt"]
    loadings = data["loadings"]
    var_table = data["var_table"]
    cluster_summary = data["cluster_summary"]
    pc_cols = data["pc_cols"]

    best_sil = sil["Silhouette"].max()
    st.markdown("## Klasterisasi Wilayah")
    caption(f"K-Means pada skor {'-'.join(pc_cols)}, K={k_opt} terpilih lewat elbow method dan silhouette score")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        section_header(f"K={k_opt} paling optimal, silhouette score memuncak di {id_num(best_sil, 3)}")
        fig = go.Figure(go.Scatter(
            x=sil["K"], y=sil["Silhouette"], mode="lines+markers+text",
            line=dict(color=SAFETY_ORANGE, width=3), marker=dict(size=9),
            text=[id_num(v, 3) for v in sil["Silhouette"]], textposition="top center",
            textfont=dict(color=TEXT_MUTED, size=11),
        ))
        fig.update_xaxes(dtick=1, tickvals=sil["K"], ticktext=[f"K={k}" for k in sil["K"]])
        style_fig(fig, height=300)
        st.plotly_chart(fig, config=chart_bar_config())
    with col2:
        top_pc = var_table.iloc[0]
        section_header(f"{top_pc['Komponen']} menjelaskan {id_num(top_pc['Variansi (%)'], 1)}% variasi antarwilayah")
        rows_html = ""
        for _, r in var_table.iterrows():
            pc = r["Komponen"]
            top3 = loadings[pc].abs().sort_values(ascending=False).head(3).index
            desc = ", ".join(f"{v} ({loadings.loc[v, pc]:+.2f})" for v in top3)
            rows_html += (
                f'<div style="margin-bottom:0.9rem;">'
                f'<span style="color:{SAFETY_ORANGE};font-weight:700;">{pc} ({id_num(r["Variansi (%)"], 1)}%)</span>'
                f'<span style="color:{SOFT_OFF_WHITE};">&nbsp;&nbsp;{ital(desc)}</span></div>'
            )
        st.markdown(f'<div class="ww-card">{rows_html}</div>', unsafe_allow_html=True)

    st.write("")
    cols = st.columns(len(cluster_summary))
    for (k_, info), col in zip(sorted(cluster_summary.items()), cols):
        with col:
            tinggi = ital(", ".join(info["tinggi"])) if info["tinggi"] else "-"
            rendah = ital(", ".join(info["rendah"])) if info["rendah"] else "-"
            st.markdown(
                f"""<div class="ww-card">
                <div class="ww-bold" style="font-size:1.05rem;">Klaster {k_}</div>
                <div class="ww-muted">{info['n']} observasi</div>
                <div class="ww-tinggi">Tinggi: {tinggi}</div><br/>
                <div class="ww-rendah">Rendah: {rendah}</div>
                <div class="ww-insight">{ital(info['insight'])}</div></div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    cluster_mr = df.groupby("Cluster")["MR"].mean()
    best_k = int(cluster_mr.idxmax())
    section_header(f"Klaster {best_k} ({cluster_summary[best_k]['insight']}) menonjol di ruang komponen utama")
    fig_sc = go.Figure()
    use_3d = len(pc_cols) >= 3
    for k_ in sorted(df["Cluster"].unique()):
        sub = df[df["Cluster"] == k_]
        color = PALETTE_CATEGORICAL[(k_ - 1) % len(PALETTE_CATEGORICAL)]
        if use_3d:
            fig_sc.add_trace(go.Scatter3d(
                x=sub[pc_cols[0]], y=sub[pc_cols[1]], z=sub[pc_cols[2]], mode="markers",
                name=f"Klaster {k_}", text=sub["Label_Tahun"],
                hovertemplate="%{text}<extra>Klaster " + str(k_) + "</extra>",
                marker=dict(size=5, color=color, opacity=0.85),
            ))
        else:
            fig_sc.add_trace(go.Scatter(x=sub[pc_cols[0]], y=sub[pc_cols[1]], mode="markers",
                                         name=f"Klaster {k_}", marker=dict(size=9, color=color)))
    if use_3d:
        axis_style = dict(backgroundcolor=BG_APP, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color=TEXT_MUTED)
        fig_sc.update_layout(
            scene=dict(
                xaxis=dict(title=pc_cols[0], **axis_style),
                yaxis=dict(title=pc_cols[1], **axis_style),
                zaxis=dict(title=pc_cols[2], **axis_style),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_PRIMARY, size=12, family="Segoe UI, Arial, sans-serif"),
            margin=dict(l=0, r=0, t=10, b=0),
            height=480,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(color=TEXT_MUTED, size=11)),
        )
    else:
        fig_sc.update_xaxes(title=pc_cols[0])
        fig_sc.update_yaxes(title=pc_cols[1] if len(pc_cols) > 1 else "")
        style_fig(fig_sc, height=380, showlegend=True)
    st.plotly_chart(fig_sc, config=chart_bar_config())


# Halaman: Analisis Gap
def render_analisis_gap(data):
    df = data["df"]
    ols = data["ols"]
    reg_cols = ols["reg_cols"]

    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        st.markdown("## Analisis *Gap*")
        caption("Waste Management Gap: kinerja aktual dibanding kinerja yang diperkirakan dari karakteristik wilayah")
    with right:
        badge("2023 - 2025")
    st.divider()

    top_gap = df.sort_values("Management_Gap", ascending=False).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("R kuadrat model", id_num(ols["r2"], 3), f"Adj. R² = {id_num(ols['r2_adj'], 3)}")
    with c2:
        kpi_card("LOOCV R kuadrat", id_num(ols["loocv_r2"], 3), f"MAE = {id_num(ols['loocv_mae'], 2)} poin persentase", accent=True)
    with c3:
        kpi_card("P-value residual", id_num(ols["ks_p"], 3), f"statistik KS = {id_num(ols['ks_stat'], 4)}")
    with c4:
        kpi_card(f"Gap tertinggi {int(top_gap['Tahun'])}", id_signed(top_gap["Management_Gap"]), top_gap["KabKota"], accent=True)

    st.write("")
    p = ols["params"]
    terms = " ".join(f"{'+' if p[c] >= 0 else '-'} {id_num(abs(p[c]), 4)}({c})" for c in reg_cols if c in p)
    formula = f"MR = {id_num(p.get('const', 0), 4)} {terms}"
    max_vif = max(ols["vif"].values()) if ols["vif"] else 1.0
    st.markdown(
        f"""<div class="ww-card">
        <div class="ww-center ww-bold">Karakteristik wilayah menjelaskan {id_num(ols['r2'] * 100, 1)}% variasi kinerja pengelolaan</div>
        <div class="ww-formula">{formula}</div>
        <div class="ww-center ww-note">n = {ols['n']}, F = {id_num(ols['fvalue'], 1)}, Prob(F) = {id_sci(ols['f_pvalue'])}.
        VIF {'/'.join(reg_cols)} = {id_num(max_vif, 1)} (ortogonal, tidak ada multikolinearitas).</div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.write("")
    col1, col2 = st.columns([1, 1])
    with col1:
        top5gap = df.sort_values("Management_Gap", ascending=False).head(5)
        section_header(f"{top_gap['KabKota']} {int(top_gap['Tahun'])} catat kesenjangan pengelolaan tertinggi ({id_signed(top_gap['Management_Gap'])} poin)")
        fig = go.Figure(go.Bar(x=top5gap["Management_Gap"], y=top5gap["Label_Tahun"], orientation="h",
                                marker_color=SAFETY_ORANGE, text=top5gap["Management_Gap"].round(0).astype(int), textposition="outside"))
        fig.update_yaxes(autorange="reversed")
        style_fig(fig, height=320)
        st.plotly_chart(fig, config=chart_bar_config())
    with col2:
        section_header("Tiga dari empat asumsi regresi terpenuhi, independensi observasi berisiko")
        st.markdown(
            f"""<div class="ww-card">
            <div class="ww-heading-blue">Linearitas</div>
            <div>Terpenuhi - hubungan {reg_cols[0]} dengan persen_kelola mengikuti pola linear.</div>
            <div class="ww-heading-blue">Normalitas residual</div>
            <div>{ital(f"Terpenuhi - statistik KS {id_num(ols['ks_stat'], 4)}, p-value {id_num(ols['ks_p'], 4)} &gt; α 0,05.")}</div>
            <div class="ww-heading-blue">Multikolinearitas</div>
            <div>Tidak ada - VIF {'/'.join(reg_cols)} semuanya {id_num(max_vif, 1)}.</div>
            <div class="ww-heading-blue">Independensi observasi</div>
            <div>Berisiko - satu wilayah dapat muncul hingga 3 kali (2023/24/25) karena data panel; lihat keterbatasan.</div>
            </div>""",
            unsafe_allow_html=True,
        )


# Halaman: Prioritas Intervensi
def render_prioritas_intervensi(data):
    df = data["df"]
    weights = data["wipi_weights"]
    sens = data["sensitivity"]
    extra = data["sensitivity_extra"]
    latest = int(df["Tahun"].max())

    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        st.markdown("## Prioritas Intervensi")
        caption("Waste Intervention Priority Index (WIPI): UWR, WGPC, Management Gap, Material Pressure")
    with right:
        badge("2023 - 2025")
    st.divider()

    dyear = df[df["Tahun"] == latest]
    top10 = dyear.sort_values("WIPI", ascending=False).head(10)
    leader = top10.iloc[0]
    n_urgent = int(dyear["Kategori_WIPI"].isin(["Prioritas tinggi", "Prioritas sangat tinggi"]).sum())

    col1, col2 = st.columns([1.4, 1])
    with col1:
        section_header(f"{leader['KabKota']} pimpin peringkat prioritas {latest} dengan skor WIPI {id_num(leader['WIPI'])}")
        fig = go.Figure(go.Bar(x=top10["WIPI"], y=top10["KabKota"], orientation="h",
                                marker_color=SAFETY_ORANGE, text=top10["WIPI"].round(0).astype(int), textposition="outside"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[0, 95])
        style_fig(fig, height=430)
        st.plotly_chart(fig, config=chart_bar_config())

        section_header(f"{n_urgent} dari {len(dyear)} wilayah berstatus prioritas tinggi atau lebih di {latest}")
        counts = dyear["Kategori_WIPI"].value_counts().reindex(KATEGORI_ORDER_DESC, fill_value=0)
        fig2 = go.Figure(go.Bar(x=counts.index, y=counts.values,
                                 marker_color=[WIPI_CATEGORY_COLORS[k] for k in counts.index],
                                 text=counts.values, textposition="outside"))
        style_fig(fig2, height=260)
        st.plotly_chart(fig2, config=chart_bar_config())

    with col2:
        dominan = weights.idxmax()
        section_header(f"{dominan} berkontribusi paling besar terhadap skor WIPI ({id_pct(weights[dominan] * 100, 1)})")
        fig3 = go.Figure(go.Bar(x=list(weights.index), y=weights.values, marker_color=MUTED_BLUE,
                                 text=[id_num(v, 2) for v in weights.values], textposition="outside"))
        fig3.update_yaxes(range=[0, weights.max() * 1.35])
        style_fig(fig3, height=230)
        st.plotly_chart(fig3, config=chart_bar_config())

        min_row = sens[sens["Skenario"] != "Bobot sama rata"].sort_values("Spearman").iloc[0]
        section_header(f"Peringkat WIPI tetap stabil; korelasi terendah {id_num(min_row['Spearman'], 3)} pada skenario \"{min_row['Skenario']}\"")
        show_sens = sens.copy()
        show_sens["Spearman"] = show_sens["Spearman"].map(lambda v: id_num(v, 3))
        st.dataframe(show_sens, hide_index=True)
        st.markdown(
            f"""<div class="ww-card" style="border:1px dashed {EARTHY_KHAKI};">
            <span class="ww-note">Simulasi 2.000 bobot acak memiliki Spearman minimum {id_num(extra['min'], 3)} dan
            persentil ke-5 sebesar {id_num(extra['p5'], 3)} - urutan peringkat cukup stabil namun tidak sepenuhnya tahan
            terhadap perubahan bobot ekstrem, sehingga layak disebut sebagai batas keandalan.</span></div>""",
            unsafe_allow_html=True,
        )


# Halaman: Rekomendasi
def render_rekomendasi(data):
    df = data["df"]
    latest = int(df["Tahun"].max())
    dyear = df[df["Tahun"] == latest]

    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        st.markdown("## Rekomendasi Intervensi")
        st.caption("Matriks profil wilayah dan arahan taktis penanganan sampah")
    with right:
        badge("2023 - 2025")
    st.divider()

    n_darurat = int((dyear["Profil"] == "Darurat / Kompleks").sum())

    col1, col2 = st.columns([1, 1.5])
    with col1:
        section_header(f"{n_darurat} dari {len(dyear)} wilayah berstatus darurat/kompleks, butuh penanganan tersegmentasi")
        counts = dyear["Profil_Label"].value_counts().sort_values()
        fig = go.Figure(go.Bar(x=counts.values, y=[ital(x) for x in counts.index], orientation="h", marker_color=MUTED_BLUE,
                                text=counts.values, textposition="outside"))
        style_fig(fig, height=380)
        st.plotly_chart(fig, config=chart_bar_config())
    with col2:
        top8 = dyear.sort_values("WIPI", ascending=False).head(8)
        section_header(f"Delapan wilayah prioritas {latest} membutuhkan arahan taktis yang berbeda-beda")
        show = top8[["KabKota", "Profil_Label", "Rekomendasi"]].rename(
            columns={"KabKota": "Kabupaten/Kota", "Profil_Label": "Profil", "Rekomendasi": "Rekomendasi taktis"}
        )
        html_table(show)


# Main
st.set_page_config(page_title="WASTEWISE", page_icon=":material/recycling:", layout="wide", initial_sidebar_state="expanded")
inject_css()

with st.sidebar:
    st.markdown(
        '<div class="ww-brand"><div class="ww-brand-name">WASTEWISE</div></div>'
        '<hr class="ww-sidebar-line"/>',
        unsafe_allow_html=True,
    )
    page = st.radio("Navigasi", PAGES, label_visibility="collapsed")

    source = DEFAULT_DATA_PATH if DEFAULT_DATA_PATH.exists() else None
    if source is None:
        st.write("")
        source = st.file_uploader("Unggah Dataset_Jawa_Tengah.xlsx", type=["xlsx"])

    st.markdown(
        '<div class="ww-sidebar-spacer"></div>'
        '<hr class="ww-sidebar-line"/>'
        '<div class="ww-sidebar-footer">Oleh Kelompok 5<br/>Data 2023 - 2025<br/>Provinsi Jawa Tengah</div>',
        unsafe_allow_html=True,
    )

if source is None:
    st.error(
        "Berkas Dataset_Jawa_Tengah.xlsx tidak ditemukan. Letakkan file tersebut di folder yang sama "
        "dengan app.py, atau unggah lewat panel di sidebar."
    )
    st.stop()

data = compute_wastewise(source)

PAGE_RENDERERS = {
    "Beranda": render_beranda,
    "Profil Wilayah": render_profil_wilayah,
    "Klasterisasi": render_klasterisasi,
    "Analisis Gap": render_analisis_gap,
    "Prioritas Intervensi": render_prioritas_intervensi,
    "Rekomendasi": render_rekomendasi,
}
PAGE_RENDERERS[page](data)
