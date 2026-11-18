import html
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_samples, silhouette_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan, lilliefors, linear_reset

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_CANDIDATES = [Path(__file__).parent / "Dataset_Sampah_Jawa_Tengah.xlsx"]
GEOJSON_PATH = Path(__file__).parent / "jawa_tengah_kabkota.geojson"

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

# Skala warna heatmap (tema gelap)
SCALE_DIVERGING = [[0, MUTED_BLUE], [0.5, BG_CARD_ALT], [1, SAFETY_ORANGE]]
SCALE_SEQUENTIAL = [[0, BG_CARD_ALT], [0.5, MUSTARD_YELLOW], [1, SAFETY_ORANGE]]

WIPI_CATEGORY_COLORS = {
    "Prioritas sangat tinggi": SAFETY_ORANGE,
    "Prioritas tinggi": MUSTARD_YELLOW,
    "Prioritas sedang": EARTHY_KHAKI,
    "Prioritas rendah": MUTED_BLUE,
    "Prioritas sangat rendah": TAUPE_BROWN,
}
KATEGORI_ORDER_ASC = list(reversed(WIPI_CATEGORY_COLORS.keys()))
KATEGORI_ORDER_DESC = list(WIPI_CATEGORY_COLORS.keys())
WIPI_BINS = [-0.001, 20, 40, 60, 80, 100.001]

PAGES = ["Beranda", "Profil Wilayah", "Klasterisasi", "Analisis Gap", "Prioritas Intervensi", "Rekomendasi"]

# Variabel sesuai notebook final
PCA_VARS = ["Population", "Timbulan", "Recycle_Success", "Unmanaged_Waste", "Domestic", "Organic", "Recycle"]
PROFIL_VARS = ["Population", "WGPC", "RR", "UWR", "Domestic", "Organic", "Recycle"]
WIPI_VARS = ["WGPC", "MG", "1-RR", "UWR"]
WIPI_PROXY = {"WGPC": "Timbulan", "1-RR": "Recycle_Success", "UWR": "Unmanaged_Waste"}
WIPI_LABEL = {
    "WGPC": "Timbulan per kapita (WGPC)",
    "MG": "Management Gap (MG)",
    "1-RR": "Tidak terdaur ulang (1-RR)",
    "UWR": "Sampah ke TPA (UWR)",
}
WIPI_COMP_COLORS = {"WGPC": MUTED_BLUE, "MG": SAFETY_ORANGE, "1-RR": MUSTARD_YELLOW, "UWR": TAUPE_BROWN}

VAR_ID = {
    "Population": "penduduk",
    "WGPC": "timbulan per kapita",
    "RR": "daur ulang",
    "UWR": "sampah ke TPA",
    "Domestic": "sampah rumah tangga",
    "Organic": "sampah organik",
    "Recycle": "anorganik daur ulang",
}

KOMPOSISI_GROUP_COLORS = {
    "Organik": SAFETY_ORANGE,
    "Anorganik daur ulang": MUTED_BLUE,
    "Lainnya/residu": TAUPE_BROWN,
}

# Rekomendasi 8 kriteria (4 variabel x Tinggi/Rendah), teks lengkap mengikuti notebook
TEKS = {
    ("WGPC", "Tinggi"): "Perkuat sosialisasi & edukasi pengurangan sampah per orang: kampanye pola konsumsi rendah sampah, kebiasaan membawa wadah/tas belanja sendiri, belanja cerdas berkemasan minimal, serta pengurangan sisa makanan di rumah tangga dan rumah makan.",
    ("WGPC", "Rendah"): "Pertahankan capaian lewat literasi lingkungan berkelanjutan di sekolah, pasar, dan komunitas, disertai pemantauan berkala timbulan per kapita agar konsumsi per orang tidak meningkat.",
    ("MG", "Tinggi", "Organik"): "Percepat pengolahan organik mandiri: komposter rumah tangga, lubang biopori, takakura, eco-enzyme, dan pengomposan komunal skala RT/RW agar sampah selesai di sumber dan kapasitas kelola menyusul ekspektasi wilayah.",
    ("MG", "Tinggi", "Anorganik"): "Perkuat pemilahan anorganik di sumber serta tambah fasilitas pengolahan anorganik skala komunitas (gerai pemilahan, kemitraan pengepul dan offtaker industri).",
    ("MG", "Rendah"): "Optimalisasi efisiensi operasional (jadwal layanan, utilisasi armada, kapasitas petugas) dan replikasi praktik baik ke wilayah lain.",
    ("1-RR", "Tinggi", "Organik"): "Jadikan pengomposan terpadu (rumah tangga-komunal) serta pemanfaatan hasil olahan organik (pupuk kompos, bokashi, eco-enzyme) sebagai tuas utama peningkatan daur ulang.",
    ("1-RR", "Tinggi", "Anorganik"): "Perbanyak fasilitas pengelolaan anorganik: titik drop-off material terpilah, gerai penerimaan daur ulang, dan kemitraan dengan pengepul/offtaker industri.",
    ("1-RR", "Rendah"): "Pertahankan capaian dan perluas rantai nilai material sekunder: diversifikasi produk olahan, kemitraan industri penerima, serta insentif pemilahan berkelanjutan.",
    ("UWR", "Tinggi"): "Genjot pengurangan sampah tiga lapis: (i) aksi individu/rumah tangga, reduksi di sumber dan guna ulang; (ii) optimalisasi komunitas lokal, pengolahan mandiri lingkungan; (iii) dorongan kebijakan makro, penguatan Jakstrada, insentif reduksi, dan target pengurangan terukur.",
    ("UWR", "Rendah"): "Pertahankan capaian lewat monitoring berkala, penguatan praktik reduksi di hulu, dan evaluasi capaian pengelolaan setiap semester.",
}
TEKS_SINGKAT = {
    ("WGPC", "Tinggi"): "Edukasi pengurangan sampah per kapita",
    ("WGPC", "Rendah"): "Pertahankan capaian + literasi lingkungan",
    ("MG", "Tinggi", "Organik"): "Percepat pengolahan organik mandiri",
    ("MG", "Tinggi", "Anorganik"): "Pemilahan + fasilitas anorganik komunitas",
    ("MG", "Rendah"): "Efisiensi operasional + replikasi praktik baik",
    ("1-RR", "Tinggi", "Organik"): "Pengomposan terpadu",
    ("1-RR", "Tinggi", "Anorganik"): "Perbanyak fasilitas anorganik",
    ("1-RR", "Rendah"): "Perluas rantai nilai daur ulang",
    ("UWR", "Tinggi"): "Pengurangan sampah tiga lapis",
    ("UWR", "Rendah"): "Monitoring berkala + reduksi di hulu",
}
KRITERIA_ARTI = {
    "WGPC": ("timbulan per orang besar", "timbulan per orang kecil"),
    "MG": ("kinerja di bawah ekspektasi", "kinerja melampaui ekspektasi"),
    "1-RR": ("tingkat daur ulang rendah", "tingkat daur ulang tinggi"),
    "UWR": ("banyak sampah lolos ke TPA", "sedikit sampah lolos ke TPA"),
}
AMBANG_MINOR = 25.0  # fraksi minor >= 25% timbulan dianggap material

# Istilah asing yang otomatis dicetak miring di seluruh teks tampilan
FOREIGN_TERMS = sorted(
    [
        "Waste Intervention Priority Index",
        "Waste Management Gap",
        "Management Gap",
        "Leave-One-Region-Out",
        "Recycling rate",
        "Gradient Boosting",
        "Random Forest",
        "Elastic Net",
        "baseline",
        "Expected",
        "eco-enzyme",
        "offtaker",
        "drop-off",
        "Monitoring",
        "silhouette score",
        "elbow method",
        "P-value",
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


def id_p(p):
    return id_sci(p) if p < 0.001 else id_num(p, 3)


def kunci_teks(v, ket, dominan):
    k3 = (v, ket, dominan)
    return k3 if k3 in TEKS else (v, ket)


def label_klaster(z, n=2, thr=0.4):
    s = z.sort_values()
    atas = [v for v in s.index[::-1][:n] if z[v] > thr]
    bawah = [v for v in s.index[:n] if z[v] < -thr]
    parts = []
    if atas:
        parts.append(" & ".join(VAR_ID[v] for v in atas) + " tinggi")
    if bawah:
        parts.append(" & ".join(VAR_ID[v] for v in bawah) + " rendah")
    return "; ".join(parts).capitalize() if parts else "Profil mendekati rata-rata provinsi"


def varimax(phi, gamma=1.0, q=500, tol=1e-6):
    # Disalin apa adanya dari notebook agar loading di dashboard identik dengan laporan
    p, k = phi.shape
    r = np.eye(k)
    d = 0
    for _ in range(q):
        u, s, vt = np.linalg.svd(phi.T @ (phi ** 3 - (gamma / p) * phi @ np.diag(np.sum(phi ** 2, axis=0))))
        r = u @ vt
        d_new = np.sum(s)
        if d_new < d * (1 + tol):
            break
        d = d_new
    return phi @ r


def kmo_bartlett(Xz):
    n, p = Xz.shape
    R = np.corrcoef(Xz, rowvar=False)
    Rinv = np.linalg.inv(R)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.linalg.slogdet(R)[1]
    dof = p * (p - 1) / 2
    parsial = -Rinv / np.sqrt(np.outer(np.diag(Rinv), np.diag(Rinv)))
    np.fill_diagonal(parsial, 0)
    R0 = R - np.eye(p)
    kmo = (R0 ** 2).sum() / ((R0 ** 2).sum() + (parsial ** 2).sum())
    return float(kmo), float(chi2), float(stats.chi2.sf(chi2, dof))


def profil_kode(frame, med):
    return frame[WIPI_VARS].gt(med).apply(lambda r: "-".join("T" if x else "R" for x in r), axis=1)


# Pipeline data (mengikuti notebook final):
# audit -> PCA (Kaiser + varimax) -> model Management Gap (LORO) -> K-Means -> WIPI -> rekomendasi 8 kriteria
REQUIRED_COLS = ["Tahun", "KabKota", "Population", "Timbulan", "Kelola", "MR", "WGPC", "RR", "Recycle_Success",
                 "UWR", "Unmanaged_Waste", "Domestic", "Organic", "Recycle", "Lainnya"]


def _norm(c):
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


COL_ALIAS = {_norm(c): c for c in REQUIRED_COLS + ["Density", "Comercial", "Public"]}
COL_ALIAS.update({"kabupatenkota": "KabKota", "kabkot": "KabKota", "wilayah": "KabKota", "year": "Tahun",
                  "commercial": "Comercial", "recyclesucces": "Recycle_Success"})


class DatasetError(Exception):
    pass


def read_dataset(source):
    # Cari sheet + baris header yang memuat kolom Tahun dan KabKota (nama kolom tidak peka huruf/spasi/garis miring)
    sheets = pd.read_excel(source, sheet_name=None, header=None)
    ditemukan = []
    for nama, raw in sheets.items():
        for h in range(min(6, len(raw))):
            cols = [COL_ALIAS.get(_norm(c), str(c).strip()) for c in raw.iloc[h]]
            if "Tahun" in cols and "KabKota" in cols:
                kurang = [c for c in REQUIRED_COLS if c not in cols]
                if kurang:
                    raise DatasetError(
                        f"Sheet '{nama}' ditemukan, tetapi kolom berikut tidak ada: {', '.join(kurang)}. "
                        f"Kolom yang terbaca: {', '.join(map(str, cols))}. Pastikan dataset adalah versi yang sama "
                        f"dengan notebook (kolom Tahun, KabKota, Population, Timbulan, Kelola, dst.)."
                    )
                df = raw.iloc[h + 1:].copy()
                df.columns = cols
                return df.loc[:, ~df.columns.duplicated()]
        ditemukan.append(f"'{nama}': {', '.join(str(c) for c in raw.iloc[0].tolist()[:8])}")
    raise DatasetError("Kolom Tahun dan KabKota tidak ditemukan di sheet mana pun. Isi baris pertama tiap sheet: "
                       + " | ".join(ditemukan))


@st.cache_data(show_spinner="Memproses data timbulan dan pengelolaan sampah...")
def compute_wastewise(source) -> dict:
    df = read_dataset(source)
    df = df.dropna(subset=["Tahun", "KabKota"]).copy()
    df["KabKota"] = df["KabKota"].astype(str).str.strip()
    df["Tahun"] = df["Tahun"].astype(int)
    for c in df.columns.drop(["Tahun", "KabKota"]):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["KabKota", "Tahun"]).reset_index(drop=True)

    df["WG"] = df["Timbulan"]
    df["1-RR"] = 100 - df["RR"]
    for c in ["Organic", "Recycle", "Lainnya"]:
        df[c + "_pct"] = df[c] / df["Timbulan"] * 100
    df["Label_Tahun"] = df["KabKota"] + " '" + df["Tahun"].astype(str).str.slice(-2)

    # ---- PCA ----
    Xz = StandardScaler().fit_transform(df[PCA_VARS])
    n_obs = len(df)
    kmo, bart_chi2, bart_p = kmo_bartlett(Xz)
    pca = PCA(random_state=RANDOM_STATE).fit(Xz)
    eig, evr = pca.explained_variance_, pca.explained_variance_ratio_
    n_pc = max(2, int((eig > 1).sum()))
    pc_cols = [f"PC{i + 1}" for i in range(n_pc)]
    scores = pca.transform(Xz)
    for i, c in enumerate(pc_cols):
        df[c] = scores[:, i]
    unrot = pca.components_[:n_pc].T * np.sqrt(eig[:n_pc] * (n_obs - 1) / n_obs)
    loadings = pd.DataFrame(varimax(unrot), index=PCA_VARS, columns=pc_cols)
    var_table = pd.DataFrame({
        "Komponen": pc_cols,
        "Eigenvalue": eig[:n_pc],
        "Variansi (%)": evr[:n_pc] * 100,
        "Kumulatif (%)": np.cumsum(evr[:n_pc]) * 100,
    })
    pca_info = {"kmo": kmo, "bart_chi2": bart_chi2, "bart_p": bart_p, "n_vars": len(PCA_VARS)}

    # ---- Model Management Gap (target: Kelola, validasi Leave-One-Region-Out) ----
    reg_cols = pc_cols
    X = df[reg_cols].values
    y = df["Kelola"].values
    grp = df["KabKota"].values
    BASE = "Rata-rata (baseline)"
    kandidat = {
        BASE: DummyRegressor(strategy="mean"),
        "OLS": LinearRegression(),
        "Elastic Net": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                                       subsample=0.8, random_state=RANDOM_STATE),
    }
    p_ = X.shape[1]
    baris, pred = [], {}
    for nama, m in kandidat.items():
        r2_in = r2_score(y, m.fit(X, y).predict(X))
        mentah = cross_val_predict(m, X, y, cv=LeaveOneGroupOut(), groups=grp)
        p_cv = np.maximum(mentah, 0)
        e = y - p_cv
        pred[nama] = p_cv
        baris.append({
            "Model": nama,
            "R² dalam sampel": r2_in,
            "Adj R²": 1 - (1 - r2_in) * (n_obs - 1) / (n_obs - p_ - 1),
            "R² LORO": r2_score(y, p_cv),
            "MAE LORO": mean_absolute_error(y, p_cv),
            "RMSE LORO": float(np.sqrt(np.mean(e ** 2))),
            "WAPE (%)": np.abs(e).sum() / np.abs(y).sum() * 100,
        })
    model_table = pd.DataFrame(baris)
    final = model_table[model_table["Model"] != BASE].sort_values("RMSE LORO").iloc[0]["Model"]

    df["Kelola_expected"] = pred[final]
    df["MR_expected"] = df["Kelola_expected"] / df["Timbulan"] * 100
    df["MG"] = df["MR_expected"] - df["MR"]
    df["Management_Gap"] = df["MG"]

    X_c = sm.add_constant(df[reg_cols])
    ols = sm.OLS(df["Kelola"], X_c).fit()
    ols_cl = sm.OLS(df["Kelola"], X_c).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(grp)[0]})
    e_ols = ols.resid
    uji = pd.DataFrame({
        "Uji": ["Normalitas (Shapiro-Wilk)", "Normalitas (Lilliefors)", "Homoskedastisitas (Breusch-Pagan)",
                "Spesifikasi linear (Ramsey RESET)"],
        "p": [stats.shapiro(e_ols).pvalue, lilliefors(e_ols)[1], het_breuschpagan(e_ols, ols.model.exog)[1],
              linear_reset(ols, power=2, test_type="fitted", use_f=True).pvalue],
    })
    uji["Lolos"] = uji["p"] >= 0.05

    years = sorted(df["Tahun"].unique())
    gp = df.pivot(index="KabKota", columns="Tahun", values="MG")
    konsistensi = pd.DataFrame([
        {"Pasangan tahun": f"{a} vs {b}", "Spearman": stats.spearmanr(gp[a], gp[b])[0], "p": stats.spearmanr(gp[a], gp[b])[1]}
        for a, b in list(zip(years[:-1], years[1:])) + ([(years[0], years[-1])] if len(years) > 2 else [])
    ])

    ols_info = {
        "params": ols.params.to_dict(),
        "bse_cluster": ols_cl.bse.to_dict(),
        "p_cluster": ols_cl.pvalues.to_dict(),
        "reg_cols": reg_cols,
        "r2": ols.rsquared,
        "r2_adj": ols.rsquared_adj,
        "fvalue": ols.fvalue,
        "f_pvalue": ols.f_pvalue,
        "n": int(ols.nobs),
        "final": final,
        "baseline": BASE,
        "model_table": model_table,
        "uji": uji,
        "konsistensi": konsistensi,
    }

    # ---- Klasterisasi K-Means pada skor PC ----
    Z = df[pc_cols].values
    sil_rows = []
    for k in range(2, 9):
        km_k = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(Z)
        sil_rows.append({"K": k, "WCSS": km_k.inertia_, "Silhouette": silhouette_score(Z, km_k.labels_)})
    sil_table = pd.DataFrame(sil_rows)
    k_opt = int(sil_table.loc[sil_table["Silhouette"].idxmax(), "K"])

    km = KMeans(n_clusters=k_opt, n_init=20, random_state=RANDOM_STATE).fit(Z)
    urut = df.assign(raw=km.labels_).groupby("raw")["MR"].mean().sort_values().index
    df["Cluster"] = pd.Series(km.labels_).map({old: i + 1 for i, old in enumerate(urut)}).to_numpy()
    sil_obs = silhouette_samples(Z, df["Cluster"])

    zprof = ((df[PROFIL_VARS] - df[PROFIL_VARS].mean()) / df[PROFIL_VARS].std()).groupby(df["Cluster"]).mean()
    cluster_summary = {}
    for k_, row in zprof.iterrows():
        mask = df["Cluster"] == k_
        cluster_summary[int(k_)] = {
            "n": int(mask.sum()),
            "wilayah": int(df.loc[mask, "KabKota"].nunique()),
            "silhouette": float(sil_obs[mask.values].mean()),
            "mr": float(df.loc[mask, "MR"].mean()),
            "tinggi": list(row[row > 0.4].sort_values(ascending=False).index),
            "rendah": list(row[row < -0.4].sort_values().index),
            "insight": label_klaster(row),
        }
    anggota = df.pivot(index="KabKota", columns="Tahun", values="Cluster")
    anggota["Berpindah"] = anggota.nunique(axis=1) > 1

    # ---- WIPI ----
    comp = df[WIPI_VARS]
    compz = (comp - comp.min()) / (comp.max() - comp.min())
    basis, w_raw = [], []
    for v in WIPI_VARS:
        if v == "MG":
            w_raw.append(evr[:len(reg_cols)].sum())
            basis.append(" & ".join(reg_cols))
        else:
            j = int(loadings.loc[WIPI_PROXY[v]].abs().to_numpy().argmax())
            w_raw.append(evr[j])
            basis.append(pc_cols[j])
    w = np.array(w_raw) / np.sum(w_raw)
    weights = pd.Series(w, index=WIPI_VARS)
    weight_table = pd.DataFrame({"Komponen": WIPI_VARS, "PC dominan": basis, "Bobot": w})

    df["WIPI"] = 100 * compz.to_numpy() @ w
    for v in WIPI_VARS:
        df["Kontrib_" + v] = 100 * compz[v] * weights[v]
    df["Kategori_WIPI"] = pd.cut(df["WIPI"], bins=WIPI_BINS, labels=KATEGORI_ORDER_ASC)
    df["Rank_WIPI"] = df.groupby("Tahun")["WIPI"].rank(ascending=False, method="min").astype(int)

    def rho(wv):
        return stats.spearmanr(df["WIPI"], compz.to_numpy() @ np.asarray(wv))[0]

    sens_rows = [{"Skenario": "Bobot sama rata", "Spearman": rho([0.25] * 4)}]
    for i, v in enumerate(WIPI_VARS):
        wv = [0.2] * 4
        wv[i] = 0.4
        sens_rows.append({"Skenario": f"{v} dominan (40%)", "Spearman": rho(wv)})
    rng = np.random.default_rng(RANDOM_STATE)
    sim = np.array([rho(rng.dirichlet(np.ones(4))) for _ in range(2000)])
    sens_rows.append({"Skenario": "Bobot acak (median, n=2000)", "Spearman": float(np.median(sim))})
    sens_table = pd.DataFrame(sens_rows)
    sens_extra = {"mean": float(sim.mean()), "min": float(sim.min()), "p5": float(np.percentile(sim, 5)), "sim": sim}

    # ---- Rekomendasi 8 kriteria (basis rerata 2023-2025, ambang median antar-wilayah) ----
    agg = df.groupby("KabKota").agg(
        WGPC=("WGPC", "mean"), MG=("MG", "mean"), RR=("RR", "mean"), UWR=("UWR", "mean"),
        Organic=("Organic_pct", "mean"), Recycle=("Recycle_pct", "mean"), WIPI=("WIPI", "mean"),
        Cluster=("Cluster", lambda s: s.mode().iloc[0]),
    )
    agg["1-RR"] = 100 - agg["RR"]
    med = agg[WIPI_VARS].median()
    for v in WIPI_VARS:
        agg[v + "_ket"] = np.where(agg[v] > med[v], "Tinggi", "Rendah")
    agg["Dominan"] = np.where(agg["Organic"] >= agg["Recycle"], "Organik", "Anorganik")
    agg["Profil"] = profil_kode(agg, med)
    agg["Kategori_WIPI"] = pd.cut(agg["WIPI"], bins=WIPI_BINS, labels=KATEGORI_ORDER_ASC)
    agg["Rank_WIPI"] = agg["WIPI"].rank(ascending=False, method="min").astype(int)

    def rekomendasi(r):
        org = r["Dominan"] == "Organik"
        minor = r["Recycle"] if org else r["Organic"]
        items, singkat = [], []
        for v in WIPI_VARS:
            ket = r[v + "_ket"]
            k = kunci_teks(v, ket, r["Dominan"])
            teks = TEKS[k]
            if v == "1-RR" and ket == "Tinggi" and minor >= AMBANG_MINOR:
                jalur = "anorganik terdaur ulang" if org else "organik"
                teks += (f" Fraksi {jalur} yang signifikan (~{minor:.0f}%) juga perlu didukung jalur pengolahan pelengkap.")
            items.append({"var": v, "ket": ket, "teks": teks})
            if ket == "Tinggi":
                singkat.append(TEKS_SINGKAT[k])
        return pd.Series({
            "Rekomendasi_items": items,
            "Rekomendasi": " | ".join(singkat) if singkat else "Pertahankan capaian (seluruh kriteria rendah)",
        })

    agg[["Rekomendasi_items", "Rekomendasi"]] = agg.apply(rekomendasi, axis=1)

    # Profil per tahun (ambang median dihitung ulang tiap tahun) sebagai pembanding kondisi terkini
    df["Profil_tahun"] = ""
    for t in years:
        m_ = df["Tahun"] == t
        df.loc[m_, "Profil_tahun"] = profil_kode(df.loc[m_], df.loc[m_, WIPI_VARS].median())

    return {
        "df": df,
        "agg": agg,
        "n_pc": n_pc,
        "pc_cols": pc_cols,
        "loadings": loadings,
        "var_table": var_table,
        "pca_info": pca_info,
        "sil_table": sil_table,
        "k_opt": k_opt,
        "cluster_summary": cluster_summary,
        "zprof": zprof,
        "anggota": anggota,
        "ols": ols_info,
        "wipi_weights": weights,
        "weight_table": weight_table,
        "sensitivity": sens_table,
        "sensitivity_extra": sens_extra,
        "median_thresholds": med,
    }


@st.cache_data(show_spinner=False)
def load_geojson(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
    pct_kelola = dyear["Kelola"].sum() / dyear["WG"].sum() * 100
    pct_tak_kelola = 100 - pct_kelola
    wgpc_avg = dyear["WGPC"].mean()
    rr_avg = dyear["RR"].mean()
    org_share = dyear["Organic"].sum() / dyear["WG"].sum() * 100

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
        kpi_card("Fraksi organik", id_pct(org_share), "dari total timbulan")

    st.write("")
    col_main, col_side = st.columns([1.4, 1])

    with col_main:
        yearly = df.groupby("Tahun")[["WGPC", "RR", "UWR", "MR"]].mean().reset_index()
        uwr_start, uwr_end = yearly["UWR"].iloc[0], yearly["UWR"].iloc[-1]
        arah = "melebar" if uwr_end > uwr_start else "menyempit"
        section_header(
            f"Kesenjangan pengelolaan {arah}: UWR {'naik' if uwr_end > uwr_start else 'turun'} dari "
            f"{id_pct(uwr_start, 1)} ({years[0]}) ke {id_pct(uwr_end, 1)} ({years[-1]})"
        )
        fig = go.Figure()
        specs = [("WGPC", 0.1, SAFETY_ORANGE, "WGPC/10"), ("RR", 1, MUTED_BLUE, "RR"),
                 ("UWR", 1, MUSTARD_YELLOW, "UWR"), ("MR", 1, EARTHY_KHAKI, "MR")]
        for col, mult, color, name in specs:
            fig.add_trace(go.Scatter(x=yearly["Tahun"], y=yearly[col] * mult, mode="lines+markers",
                                      name=name, line=dict(color=color, width=3), marker=dict(size=7)))
        fig.update_xaxes(dtick=1)
        fig.update_yaxes(range=[0, 100])
        style_fig(fig, height=250, showlegend=True)
        st.plotly_chart(fig, config=chart_bar_config())

        grp = pd.Series({
            "Organik": dyear["Organic"].sum(),
            "Anorganik daur ulang": dyear["Recycle"].sum(),
            "Lainnya/residu": dyear["Lainnya"].sum(),
        })
        grp_pct = grp / grp.sum() * 100
        top_kategori = grp_pct.idxmax()
        section_header(f"Sampah {top_kategori.lower()} mendominasi {id_pct(grp_pct.max(), 1)} komposisi sampah {tahun}")
        fig2 = go.Figure(go.Pie(labels=grp.index, values=grp.values, hole=0.35,
                                 marker=dict(colors=[KOMPOSISI_GROUP_COLORS[k] for k in grp.index]),
                                 textinfo="percent", textfont=dict(color=BG_APP, size=12), sort=False))
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
        fig3.update_xaxes(range=[0, 105])
        style_fig(fig3, height=560)
        st.plotly_chart(fig3, config=chart_bar_config())


# Halaman: Profil Wilayah
def render_profil_wilayah(data):
    df = data["df"]
    agg = data["agg"]
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
    ra = agg.loc[kota]
    title_ph.markdown("## Profil Wilayah")
    subtitle_ph.caption(f"{kota}: peringkat prioritas #{int(row['Rank_WIPI'])} tahun {tahun}, klaster C{int(row['Cluster'])}")

    dyear = df[df["Tahun"] == tahun]

    col_left, col_right = st.columns([1, 1.7])
    with col_left:
        section_header(f"{kota} berstatus {row['Kategori_WIPI']}")
        st.plotly_chart(wipi_gauge(row["WIPI"], row["Kategori_WIPI"]), config=chart_bar_config())

        prov_avg = dyear[["WGPC", "RR", "UWR", "MR"]].mean()
        cats = ["WGPC/10", "RR", "UWR", "MR"]
        kota_vals = [row["WGPC"] / 10, row["RR"], row["UWR"], row["MR"]]
        prov_vals = [prov_avg["WGPC"] / 10, prov_avg["RR"], prov_avg["UWR"], prov_avg["MR"]]
        diffs = [k - p for k, p in zip(kota_vals, prov_vals)]
        i_max = int(np.argmax(np.abs(diffs)))
        arah = "lebih tinggi" if diffs[i_max] > 0 else "lebih rendah"
        section_header(f"{kota} {arah} {id_num(abs(diffs[i_max]), 1)} poin pada {cats[i_max]} dibanding rata-rata provinsi")
        figc = go.Figure()
        figc.add_trace(go.Bar(x=cats, y=kota_vals, name=kota, marker_color=SAFETY_ORANGE))
        figc.add_trace(go.Bar(x=cats, y=prov_vals, name="Rata-rata provinsi", marker_color=MUTED_BLUE))
        style_fig(figc, height=280, showlegend=True)
        st.plotly_chart(figc, config=chart_bar_config())

        kontrib = pd.Series({WIPI_LABEL[v]: row["Kontrib_" + v] for v in WIPI_VARS})
        section_header(f"Skor WIPI {tahun} paling banyak disumbang {kontrib.idxmax()}")
        figk = go.Figure(go.Bar(x=kontrib.values, y=kontrib.index, orientation="h",
                                marker_color=[WIPI_COMP_COLORS[v] for v in WIPI_VARS],
                                text=[id_num(v, 1) for v in kontrib.values], textposition="outside"))
        figk.update_yaxes(autorange="reversed")
        figk.update_xaxes(range=[0, max(kontrib.max() * 1.3, 1)])
        style_fig(figk, height=200)
        st.plotly_chart(figk, config=chart_bar_config())

    with col_right:
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi_card("WGPC", id_num(row["WGPC"]), "kg/orang/tahun")
        with k2:
            kpi_card("UWR", id_pct(row["UWR"]), "sampah masuk TPA", accent=True)
        with k3:
            kpi_card("RR", id_pct(row["RR"]), "recycling rate")
        with k4:
            kpi_card("Management Gap", id_signed(row["MG"]), "poin %, positif = di bawah ekspektasi", accent=True)

        st.write("")
        profil_t = row["Profil_tahun"]
        section_header(f"Rekomendasi intervensi: profil {ra['Profil']} (rerata 2023-2025)")
        rows_html = ""
        for it in ra["Rekomendasi_items"]:
            v, ket = it["var"], it["ket"]
            cls = "ww-tinggi" if ket == "Tinggi" else "ww-rendah"
            arti = KRITERIA_ARTI[v][0 if ket == "Tinggi" else 1]
            rows_html += (
                f'<div style="margin-bottom:0.7rem;"><span class="{cls}">{v} {ket.lower()}</span>'
                f'<span class="ww-muted">&nbsp;&nbsp;{arti}</span><br/>{ital(html.escape(it["teks"]))}</div>'
            )
        beda = "" if profil_t == ra["Profil"] else (
            f'<div class="ww-note">Profil kondisi {tahun} saja: <b>{profil_t}</b>; berbeda dari rerata, '
            f'tinjau ulang kriteria yang berubah.</div>')
        st.markdown(
            f'<div class="ww-card"><div class="ww-muted" style="margin-bottom:0.6rem;">Kode profil WGPC-MG-1RR-UWR '
            f'(T = di atas median, R = di bawah median), komposisi dominan {ra["Dominan"].lower()}</div>'
            f'{rows_html}{beda}</div>',
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
        show = comp[["KabKota", "WGPC", "UWR", "RR", "MG", "WIPI"]].rename(columns={"KabKota": "Kabupaten/Kota"})
        show["WGPC"] = show["WGPC"].map(lambda v: id_num(v))
        show["UWR"] = show["UWR"].map(lambda v: id_pct(v))
        show["RR"] = show["RR"].map(lambda v: id_pct(v))
        show["MG"] = show["MG"].map(lambda v: id_signed(v))
        show["WIPI"] = show["WIPI"].map(lambda v: id_num(v))
        st.dataframe(show, hide_index=True)


def render_peta_klaster(df, cluster_summary):
    if not GEOJSON_PATH.exists():
        caption("Peta tidak ditampilkan: letakkan jawa_tengah_kabkota.geojson di folder yang sama dengan app.py.")
        return
    geo = load_geojson(str(GEOJSON_PATH))
    semua = [f["properties"]["KabKota"] for f in geo["features"]]
    years = sorted(df["Tahun"].unique().tolist())

    head_l, head_r = st.columns([3, 1], vertical_alignment="center")
    with head_r:
        tahun = st.pills("Tahun peta", years, default=years[-1], selection_mode="single", required=True,
                         label_visibility="collapsed", key="tahun_peta")
    d = df[df["Tahun"] == tahun]
    with head_l:
        dom = int(d["Cluster"].mode().iloc[0])
        section_header(f"Sebaran spasial klaster {tahun}: C{dom} mendominasi {int((d['Cluster'] == dom).sum())} "
                       f"dari {len(d)} wilayah")

    # Poligon digambar sebagai area terisi (tanpa basemap) agar peta tetap tampil tanpa koneksi internet
    data_by = d.set_index("KabKota")
    fig = go.Figure()
    legend_done = set()
    for feat in geo["features"]:
        nama = feat["properties"]["KabKota"]
        gm = feat["geometry"]
        polys = gm["coordinates"] if gm["type"] == "MultiPolygon" else [gm["coordinates"]]
        if nama in data_by.index:
            r = data_by.loc[nama]
            k_ = int(r["Cluster"])
            color = PALETTE_CATEGORICAL[(k_ - 1) % len(PALETTE_CATEGORICAL)]
            grp = f"C{k_}"
            label = f"C{k_}: {cluster_summary[k_]['insight']}"
            hover = (f"<b>{nama}</b><br>Klaster C{k_}<br>WIPI {id_num(r['WIPI'], 1)} ({r['Kategori_WIPI']})"
                     f"<br>MR {id_pct(r['MR'], 1)}<br>UWR {id_pct(r['UWR'], 1)}")
        else:
            color, grp, label = BG_CARD_ALT, "luar", "Di luar sampel"
            hover = f"<b>{nama}</b><br>Tidak ada data"
        for poly in polys:
            ring = np.asarray(poly[0])
            if ring[:, 1].mean() > -6.1:  # lewati Kepulauan Karimunjawa agar daratan utama tampil lebih besar
                continue
            fig.add_trace(go.Scatter(
                x=ring[:, 0], y=ring[:, 1], mode="lines", fill="toself", fillcolor=color,
                line=dict(color=BG_APP, width=0.8), hoveron="fills", hoverinfo="text", text=hover,
                name=label, legendgroup=grp, showlegend=grp not in legend_done,
                legendrank=0 if grp == "luar" else int(grp[1:]),
            ))
            legend_done.add(grp)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    style_fig(fig, height=460, showlegend=True)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), hoverlabel=dict(bgcolor=BG_CARD, font=dict(color=TEXT_PRIMARY)))
    st.plotly_chart(fig, config=chart_bar_config())
    caption("Warna = klaster K-Means wilayah pada tahun terpilih; abu-abu = kabupaten/kota di luar 26 wilayah sampel. "
            "Arahkan kursor ke wilayah untuk melihat WIPI, MR, dan UWR.")
    st.write("")


# Halaman: Klasterisasi
def render_klasterisasi(data):
    df = data["df"]
    sil = data["sil_table"]
    k_opt = data["k_opt"]
    loadings = data["loadings"]
    var_table = data["var_table"]
    cluster_summary = data["cluster_summary"]
    pc_cols = data["pc_cols"]
    pinfo = data["pca_info"]
    zprof = data["zprof"]
    anggota = data["anggota"]

    best_sil = sil["Silhouette"].max()
    st.markdown("## Klasterisasi Wilayah")
    caption(f"K-Means pada skor {'-'.join(pc_cols)} (78 observasi wilayah-tahun), K={k_opt} terpilih lewat elbow method "
            f"dan silhouette score; klaster diurutkan dari MR terendah (C1)")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        section_header(f"K={k_opt} paling optimal, silhouette score memuncak di {id_num(best_sil, 3)}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sil["K"], y=sil["Silhouette"], mode="lines+markers+text", name="Silhouette",
            line=dict(color=SAFETY_ORANGE, width=3), marker=dict(size=9),
            text=[id_num(v, 3) for v in sil["Silhouette"]], textposition="top center",
            textfont=dict(color=TEXT_MUTED, size=11),
        ))
        fig.add_trace(go.Scatter(x=sil["K"], y=sil["WCSS"], mode="lines+markers", name="WCSS (elbow)", yaxis="y2",
                                 line=dict(color=EARTHY_KHAKI, width=2, dash="dot"), marker=dict(size=6)))
        fig.add_vline(x=k_opt, line=dict(color=TAUPE_BROWN, dash="dash"))
        fig.update_xaxes(dtick=1, tickvals=sil["K"], ticktext=[f"K={k}" for k in sil["K"]])
        style_fig(fig, height=300, showlegend=True)
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, color=TEXT_MUTED,
                                      tickfont=dict(size=10)))
        st.plotly_chart(fig, config=chart_bar_config())
    with col2:
        top_pc = var_table.iloc[0]
        section_header(f"{top_pc['Komponen']} menjelaskan {id_num(top_pc['Variansi (%)'], 1)}% variasi antarwilayah; "
                       f"{len(pc_cols)} komponen mencakup {id_num(var_table['Kumulatif (%)'].iloc[-1], 1)}%")
        rows_html = ""
        for _, r in var_table.iterrows():
            pc = r["Komponen"]
            dom = loadings[pc][loadings[pc].abs() >= 0.5].sort_values(key=abs, ascending=False)
            desc = ", ".join(f"{v} ({id_signed(dom[v])})" for v in dom.index) or "-"
            rows_html += (
                f'<div style="margin-bottom:0.9rem;">'
                f'<span style="color:{SAFETY_ORANGE};font-weight:700;">{pc} ({id_num(r["Variansi (%)"], 1)}%, '
                f'eigenvalue {id_num(r["Eigenvalue"], 2)})</span>'
                f'<span style="color:{SOFT_OFF_WHITE};">&nbsp;&nbsp;{html.escape(desc)}</span></div>'
            )
        rows_html += (
            f'<div class="ww-note">Loading setelah rotasi varimax, ditampilkan |loading| &ge; 0,50. '
            f'KMO = {id_num(pinfo["kmo"], 3)} (layak &ge; 0,50); Bartlett &chi;² = {id_num(pinfo["bart_chi2"], 1)}, '
            f'p = {id_sci(pinfo["bart_p"])}. Komponen dipilih dengan kriteria Kaiser (eigenvalue &gt; 1).</div>'
        )
        st.markdown(f'<div class="ww-card">{rows_html}</div>', unsafe_allow_html=True)

    st.write("")
    cols = st.columns(len(cluster_summary))
    for (k_, info), col in zip(sorted(cluster_summary.items()), cols):
        with col:
            tinggi = ", ".join(info["tinggi"]) if info["tinggi"] else "-"
            rendah = ", ".join(info["rendah"]) if info["rendah"] else "-"
            st.markdown(
                f"""<div class="ww-card">
                <div class="ww-bold" style="font-size:1.05rem;">Klaster C{k_}</div>
                <div class="ww-muted">{info['n']} observasi, {info['wilayah']} wilayah, MR rerata {id_pct(info['mr'], 1)}</div>
                <div class="ww-muted">silhouette rerata {id_num(info['silhouette'], 2)}</div><br/>
                <div class="ww-tinggi">Tinggi: {tinggi}</div><br/>
                <div class="ww-rendah">Rendah: {rendah}</div>
                <div class="ww-insight">{ital(info['insight'])}</div></div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    best_k = max(cluster_summary, key=lambda k: cluster_summary[k]["mr"])
    section_header(f"Klaster C{best_k} ({cluster_summary[best_k]['insight'].lower()}) mencatat MR rerata tertinggi")
    fig_sc = go.Figure()
    use_3d = len(pc_cols) >= 3
    for k_ in sorted(df["Cluster"].unique()):
        sub = df[df["Cluster"] == k_]
        color = PALETTE_CATEGORICAL[(k_ - 1) % len(PALETTE_CATEGORICAL)]
        if use_3d:
            fig_sc.add_trace(go.Scatter3d(
                x=sub[pc_cols[0]], y=sub[pc_cols[1]], z=sub[pc_cols[2]], mode="markers",
                name=f"Klaster C{k_}", text=sub["Label_Tahun"],
                hovertemplate="%{text}<extra>Klaster C" + str(k_) + "</extra>",
                marker=dict(size=5, color=color, opacity=0.85),
            ))
        else:
            fig_sc.add_trace(go.Scatter(
                x=sub[pc_cols[0]], y=sub[pc_cols[1]], mode="markers", name=f"Klaster C{k_}",
                text=sub["Label_Tahun"], hovertemplate="%{text}<extra>Klaster C" + str(k_) + "</extra>",
                marker=dict(size=10, color=color, opacity=0.85, line=dict(color=BG_APP, width=1)),
            ))
            fig_sc.add_trace(go.Scatter(
                x=[sub[pc_cols[0]].mean()], y=[sub[pc_cols[1]].mean()], mode="markers+text", showlegend=False,
                text=[f"C{k_}"], textposition="top center", textfont=dict(color=SOFT_OFF_WHITE, size=13),
                hoverinfo="skip", marker=dict(symbol="x", size=16, color=color, line=dict(color=SOFT_OFF_WHITE, width=1)),
            ))
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
        style_fig(fig_sc, height=420, showlegend=True)
    st.plotly_chart(fig_sc, config=chart_bar_config())

    render_peta_klaster(df, cluster_summary)

    col3, col4 = st.columns([1.5, 1])
    with col3:
        z_abs = zprof.abs().stack().idxmax()
        section_header(f"Pembeda terkuat: C{z_abs[0]} pada {z_abs[1]} ({id_signed(zprof.loc[z_abs], 2)} simpangan baku)")
        figh = go.Figure(go.Heatmap(
            z=zprof.values, x=list(zprof.columns), y=[f"C{c}" for c in zprof.index],
            colorscale=SCALE_DIVERGING, zmid=0, zmin=-2, zmax=2,
            text=[[id_num(v, 2) for v in r] for r in zprof.values], texttemplate="%{text}",
            textfont=dict(color=SOFT_OFF_WHITE, size=12), colorbar=dict(title="z", tickfont=dict(color=TEXT_MUTED)),
        ))
        figh.update_yaxes(autorange="reversed")
        style_fig(figh, height=280)
        st.plotly_chart(figh, config=chart_bar_config())
        caption("Rerata z-score tiap variabel per klaster: oranye = di atas rerata seluruh observasi, biru = di bawah.")
    with col4:
        n_stabil = int((~anggota["Berpindah"]).sum())
        section_header(f"{n_stabil} dari {len(anggota)} wilayah konsisten di klaster yang sama sepanjang periode")
        pindah = anggota[anggota["Berpindah"]].drop(columns="Berpindah")
        if len(pindah):
            show = pindah.reset_index().rename(columns={"KabKota": "Wilayah berpindah"})
            show.columns = [str(c) for c in show.columns]
            for c in show.columns[1:]:
                show[c] = show[c].map(lambda v: f"C{int(v)}")
            html_table(show)
        else:
            st.markdown('<div class="ww-card ww-muted">Tidak ada wilayah yang berpindah klaster.</div>', unsafe_allow_html=True)


# Halaman: Analisis Gap
def render_analisis_gap(data):
    df = data["df"]
    ols = data["ols"]
    reg_cols = ols["reg_cols"]
    mt = ols["model_table"]
    final = ols["final"]
    fin = mt.set_index("Model").loc[final]
    years = sorted(df["Tahun"].unique().tolist())

    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        st.markdown("## Analisis *Gap*")
        caption("Waste Management Gap: MR aktual dibanding MR yang diperkirakan dari karakteristik wilayah "
                "(MG = Expected MR - MR aktual; positif = di bawah ekspektasi)")
    with right:
        badge(f"{years[0]} - {years[-1]}")
    st.divider()

    top_gap = df.sort_values("Management_Gap", ascending=False).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("R kuadrat model OLS", id_num(ols["r2"], 3), f"Adj. R² = {id_num(ols['r2_adj'], 3)}")
    with c2:
        kpi_card("R kuadrat Leave-One-Region-Out", id_num(fin["R² LORO"], 3),
                 f"RMSE = {id_num(fin['RMSE LORO'] / 1000, 1)} ribu ton", accent=True)
    with c3:
        kpi_card("Model terpilih", final, f"RMSE LORO terendah dari {len(mt) - 1} kandidat")
    with c4:
        kpi_card(f"Gap tertinggi {int(top_gap['Tahun'])}", id_signed(top_gap["Management_Gap"]), top_gap["KabKota"], accent=True)

    st.write("")
    p = ols["params"]
    terms = " ".join(f"{'+' if p[c] >= 0 else '-'} {id_num(abs(p[c]), 1)}({c})" for c in reg_cols if c in p)
    formula = f"Kelola = {id_num(p.get('const', 0), 1)} {terms}"
    st.markdown(
        f"""<div class="ww-card">
        <div class="ww-center ww-bold">Karakteristik wilayah menjelaskan {id_num(ols['r2'] * 100, 1)}% variasi volume sampah terkelola</div>
        <div class="ww-formula">{formula}</div>
        <div class="ww-center ww-note">n = {ols['n']}, F = {id_num(ols['fvalue'], 1)}, Prob(F) = {id_sci(ols['f_pvalue'])}.
        VIF {'/'.join(reg_cols)} = 1,0 (skor PC ortogonal). Expected MR = prediksi Leave-One-Region-Out model {final} / Timbulan x 100,
        sehingga patokan suatu wilayah tidak dipengaruhi capaiannya sendiri.</div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.write("")
    col1, col2 = st.columns([1, 1])
    with col1:
        BASE = ols["baseline"]
        mt_s = mt.sort_values("RMSE LORO")
        rmse_base = mt.set_index("Model").loc[BASE, "RMSE LORO"]
        section_header(f"{final} paling akurat di luar sampel: RMSE {id_num(fin['RMSE LORO'] / 1000, 1)} ribu ton, "
                       f"{id_num((1 - fin['RMSE LORO'] / rmse_base) * 100, 0)}% lebih rendah dari baseline")
        colors = [SAFETY_ORANGE if m == final else TAUPE_BROWN if m == BASE else MUTED_BLUE for m in mt_s["Model"]]
        figm = go.Figure(go.Bar(x=mt_s["RMSE LORO"] / 1000, y=[ital(m) for m in mt_s["Model"]], orientation="h",
                                marker_color=colors, text=[id_num(v / 1000, 1) for v in mt_s["RMSE LORO"]],
                                textposition="outside"))
        figm.update_yaxes(autorange="reversed")
        figm.update_xaxes(title="RMSE Leave-One-Region-Out (ribu ton)", range=[0, mt_s["RMSE LORO"].max() / 1000 * 1.25])
        style_fig(figm, height=260)
        st.plotly_chart(figm, config=chart_bar_config())
    with col2:
        top_in = mt.sort_values("R² dalam sampel", ascending=False).iloc[0]
        if top_in["Model"] != final:
            section_header(f"{top_in['Model']} unggul dalam sampel (R² {id_num(top_in['R² dalam sampel'], 3)}) "
                           f"tetapi turun ke {id_num(top_in['R² LORO'], 3)} saat diuji ke wilayah baru")
        else:
            section_header(f"{final} unggul baik dalam sampel maupun saat diuji ke wilayah baru")
        show = mt.copy()
        for c in ["R² dalam sampel", "R² LORO"]:
            show[c] = show[c].map(lambda v: id_num(v, 3))
        show["MAE LORO"] = show["MAE LORO"].map(lambda v: id_num(v / 1000, 1) + " rb")
        show["RMSE LORO"] = show["RMSE LORO"].map(lambda v: id_num(v / 1000, 1) + " rb")
        show["WAPE (%)"] = show["WAPE (%)"].map(lambda v: id_num(v, 1))
        html_table(show.drop(columns="Adj R²"))
        st.markdown(f'<div class="ww-note" style="margin-top:0.5rem;">LORO: ketiga tahun satu wilayah dikeluarkan bersamaan '
                    f'dari data latih (Roberts dkk., 2017). Satuan error: ton/tahun.</div>', unsafe_allow_html=True)

    st.write("")
    col3, col4 = st.columns([1, 1])
    with col3:
        top5gap = df.sort_values("Management_Gap", ascending=False).head(5)
        section_header(f"{top_gap['KabKota']} {int(top_gap['Tahun'])} catat kesenjangan pengelolaan tertinggi ({id_signed(top_gap['Management_Gap'])} poin)")
        fig = go.Figure(go.Bar(x=top5gap["Management_Gap"], y=top5gap["Label_Tahun"], orientation="h",
                                marker_color=SAFETY_ORANGE, text=top5gap["Management_Gap"].round(0).astype(int), textposition="outside"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[0, top5gap["Management_Gap"].max() * 1.2])
        style_fig(fig, height=320)
        st.plotly_chart(fig, config=chart_bar_config())
    with col4:
        latest = years[-1]
        d = df[df["Tahun"] == latest]
        n_bawah = int((d["MG"] > 0).sum())
        section_header(f"{n_bawah} dari {len(d)} wilayah berada di bawah ekspektasi pada {latest}")
        figs = go.Figure()
        figs.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Aktual = Expected",
                                  line=dict(color=EARTHY_KHAKI, dash="dash", width=1)))
        for nama, m_, color in [("Di bawah ekspektasi", d["MG"] > 0, SAFETY_ORANGE),
                                ("Melampaui ekspektasi", d["MG"] <= 0, MUTED_BLUE)]:
            sub = d[m_]
            figs.add_trace(go.Scatter(
                x=sub["MR_expected"], y=sub["MR"], mode="markers", name=nama, text=sub["KabKota"],
                customdata=sub["MG"], marker=dict(color=color, size=10, line=dict(color=BG_APP, width=1)),
                hovertemplate="%{text}<br>Expected MR %{x:.1f}%<br>MR aktual %{y:.1f}%<br>MG %{customdata:+.1f} poin<extra></extra>",
            ))
        figs.update_xaxes(title="Expected MR (%)", range=[0, 105])
        figs.update_yaxes(title="MR aktual (%)", range=[0, 105])
        style_fig(figs, height=320, showlegend=True)
        st.plotly_chart(figs, config=chart_bar_config())

    st.write("")
    col5, col6 = st.columns([1, 1])
    with col5:
        uji = ols["uji"]
        n_lolos = int(uji["Lolos"].sum())
        section_header(f"{n_lolos} dari {len(uji)} uji asumsi OLS terpenuhi; dependensi antar-tahun ditangani validasi LORO")
        rows_html = ""
        for _, r in uji.iterrows():
            status = "Terpenuhi" if r["Lolos"] else "Tidak terpenuhi"
            cls = "ww-rendah" if r["Lolos"] else "ww-tinggi"
            rows_html += (f'<div class="ww-heading-blue">{r["Uji"]}</div>'
                          f'<div><span class="{cls}">{status}</span> - p-value {id_p(r["p"])} '
                          f'{"&ge;" if r["Lolos"] else "&lt;"} &alpha; 0,05</div>')
        rows_html += ('<div class="ww-heading-blue">Independensi observasi</div><div>Berisiko - satu wilayah muncul 3 kali '
                      '(data panel), sehingga ukuran sampel efektif mendekati 26; karena itu validasi dilakukan per wilayah.</div>')
        st.markdown(f'<div class="ww-card">{rows_html}</div>', unsafe_allow_html=True)
    with col6:
        kons = ols["konsistensi"]
        rho_mean = kons["Spearman"].mean()
        sifat = "persisten (struktural)" if rho_mean >= 0.5 else "berfluktuasi antar-tahun"
        section_header(f"Peringkat gap {sifat}: rerata korelasi Spearman antar-tahun {id_num(rho_mean, 2)}")
        show = kons.copy()
        show["Spearman"] = show["Spearman"].map(lambda v: id_num(v, 2))
        show["Signifikan (α 0,05)"] = kons["p"].map(lambda v: "Ya" if v < 0.05 else "Tidak")
        show["p"] = kons["p"].map(id_p)
        html_table(show)
        st.markdown('<div class="ww-note" style="margin-top:0.5rem;">Korelasi rendah berarti wilayah dengan gap besar '
                    'berganti dari tahun ke tahun; gap tahunan sebaiknya dibaca bersama rerata 2023-2025.</div>',
                    unsafe_allow_html=True)


# Halaman: Prioritas Intervensi
def render_prioritas_intervensi(data):
    df = data["df"]
    agg = data["agg"]
    weights = data["wipi_weights"]
    wtab = data["weight_table"].set_index("Komponen")
    sens = data["sensitivity"]
    extra = data["sensitivity_extra"]
    years = sorted(df["Tahun"].unique().tolist())
    latest = int(df["Tahun"].max())

    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        st.markdown("## Prioritas Intervensi")
        caption("Waste Intervention Priority Index (WIPI) = 100 x (w1 WGPC* + w2 MG* + w3 (1-RR)* + w4 UWR*), "
                "* = min-max 0-1 atas seluruh observasi")
    with right:
        badge(f"{years[0]} - {years[-1]}")
    st.divider()

    dyear = df[df["Tahun"] == latest]
    top10 = dyear.sort_values("WIPI", ascending=False).head(10)
    leader = top10.iloc[0]
    n_urgent = int(dyear["Kategori_WIPI"].isin(["Prioritas tinggi", "Prioritas sangat tinggi"]).sum())

    col1, col2 = st.columns([1.4, 1])
    with col1:
        section_header(f"{leader['KabKota']} pimpin peringkat prioritas {latest} dengan skor WIPI {id_num(leader['WIPI'])}")
        fig = go.Figure(go.Bar(x=top10["WIPI"], y=top10["KabKota"], orientation="h",
                                marker_color=[WIPI_CATEGORY_COLORS[k] for k in top10["Kategori_WIPI"]],
                                text=top10["WIPI"].round(0).astype(int), textposition="outside"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[0, 105])
        style_fig(fig, height=430)
        st.plotly_chart(fig, config=chart_bar_config())

        section_header(f"{n_urgent} dari {len(dyear)} wilayah berstatus prioritas tinggi atau lebih di {latest}")
        counts = dyear["Kategori_WIPI"].value_counts().reindex(KATEGORI_ORDER_DESC, fill_value=0)
        fig2 = go.Figure(go.Bar(x=counts.index, y=counts.values,
                                 marker_color=[WIPI_CATEGORY_COLORS[k] for k in counts.index],
                                 text=counts.values, textposition="outside"))
        fig2.update_yaxes(range=[0, counts.max() * 1.25])
        style_fig(fig2, height=260)
        st.plotly_chart(fig2, config=chart_bar_config())

    with col2:
        dominan = weights.idxmax()
        section_header(f"{dominan} berkontribusi paling besar terhadap skor WIPI ({id_pct(weights[dominan] * 100, 1)})")
        fig3 = go.Figure(go.Bar(x=list(weights.index), y=weights.values,
                                 marker_color=[WIPI_COMP_COLORS[v] for v in weights.index],
                                 text=[f"{id_num(v, 2)}<br>({wtab.loc[k, 'PC dominan']})" for k, v in weights.items()],
                                 textposition="outside"))
        fig3.update_yaxes(range=[0, weights.max() * 1.5])
        style_fig(fig3, height=230)
        st.plotly_chart(fig3, config=chart_bar_config())
        caption("Bobot = rasio varians PC dominan tiap komponen (MG memakai seluruh PC regresi), dinormalisasi berjumlah 1.")

        min_row = sens[sens["Skenario"] != "Bobot sama rata"].sort_values("Spearman").iloc[0]
        section_header(f"Peringkat WIPI cukup stabil; korelasi terendah {id_num(min_row['Spearman'], 3)} pada skenario \"{min_row['Skenario']}\"")
        show_sens = sens.copy()
        show_sens["Spearman"] = show_sens["Spearman"].map(lambda v: id_num(v, 3))
        st.dataframe(show_sens, hide_index=True)
        st.markdown(
            f"""<div class="ww-card" style="border:1px dashed {EARTHY_KHAKI};">
            <span class="ww-note">Simulasi 2.000 bobot acak (Dirichlet) menghasilkan Spearman rerata {id_num(extra['mean'], 3)},
            persentil ke-5 sebesar {id_num(extra['p5'], 3)}, dan minimum {id_num(extra['min'], 3)} - urutan peringkat cukup stabil
            namun tidak sepenuhnya tahan terhadap perubahan bobot ekstrem, sehingga layak disebut sebagai batas keandalan.</span></div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    col3, col4 = st.columns([1.4, 1])
    with col3:
        kontrib = df.groupby("KabKota")[["Kontrib_" + v for v in WIPI_VARS]].mean()
        kontrib = kontrib.loc[kontrib.sum(axis=1).sort_values(ascending=False).index]
        dom_count = kontrib.idxmax(axis=1).str.replace("Kontrib_", "").value_counts()
        section_header(f"Akar masalah berbeda: {WIPI_LABEL[dom_count.index[0]]} jadi penyumbang terbesar di "
                       f"{dom_count.iloc[0]} dari {len(kontrib)} wilayah (rerata {years[0]}-{years[-1]})")
        figd = go.Figure()
        for v in WIPI_VARS:
            figd.add_trace(go.Bar(x=kontrib["Kontrib_" + v], y=kontrib.index, orientation="h", name=WIPI_LABEL[v],
                                  marker_color=WIPI_COMP_COLORS[v],
                                  hovertemplate="%{y}<br>" + WIPI_LABEL[v] + ": %{x:.1f}<extra></extra>"))
        figd.update_layout(barmode="stack")
        figd.update_yaxes(autorange="reversed", tickfont=dict(size=10))
        figd.update_xaxes(title="Kontribusi terhadap skor WIPI")
        style_fig(figd, height=640, showlegend=True)
        st.plotly_chart(figd, config=chart_bar_config())
    with col4:
        piv = df.pivot(index="KabKota", columns="Tahun", values="WIPI").sort_values(latest, ascending=False)
        naik = (piv[latest] - piv[years[0]]).sort_values(ascending=False)
        section_header(f"{naik.index[0]} mencatat kenaikan WIPI terbesar ({id_signed(naik.iloc[0], 1)} poin, {years[0]}-{latest})")
        figw = go.Figure(go.Heatmap(
            z=piv.values, x=[str(c) for c in piv.columns], y=piv.index, colorscale=SCALE_SEQUENTIAL, zmin=0, zmax=100,
            text=[[id_num(v, 0) for v in r] for r in piv.values], texttemplate="%{text}",
            textfont=dict(color=SOFT_OFF_WHITE, size=11), colorbar=dict(title="WIPI", tickfont=dict(color=TEXT_MUTED)),
            hovertemplate="%{y} %{x}: %{z:.1f}<extra></extra>",
        ))
        figw.update_yaxes(autorange="reversed", tickfont=dict(size=10))
        style_fig(figw, height=640)
        st.plotly_chart(figw, config=chart_bar_config())


# Halaman: Rekomendasi
def render_rekomendasi(data):
    df = data["df"]
    agg = data["agg"]
    med = data["median_thresholds"]
    years = sorted(df["Tahun"].unique().tolist())
    latest = years[-1]

    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        st.markdown("## Rekomendasi Intervensi")
        st.caption("8 kriteria: WGPC, MG, 1-RR, UWR masing-masing Tinggi (T) atau Rendah (R) terhadap median antar-wilayah, "
                   f"basis rerata {years[0]}-{years[-1]}; kode profil WGPC-MG-1RR-UWR")
    with right:
        badge(f"{years[0]} - {years[-1]}")
    st.divider()

    n_full = int((agg["Profil"] == "T-T-T-T").sum())

    col1, col2 = st.columns([1, 1.5])
    with col1:
        section_header(f"{n_full} dari {len(agg)} wilayah berprofil T-T-T-T, butuh intervensi di keempat dimensi")
        counts = agg["Profil"].value_counts().sort_values()
        n_t = counts.index.str.count("T")
        warna = [SAFETY_ORANGE if t == 4 else MUSTARD_YELLOW if t == 3 else MUTED_BLUE if t >= 1 else TAUPE_BROWN for t in n_t]
        fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker_color=warna,
                                text=counts.values, textposition="outside"))
        fig.update_xaxes(range=[0, counts.max() * 1.25])
        style_fig(fig, height=420)
        st.plotly_chart(fig, config=chart_bar_config())
        caption("Oranye = 4 kriteria tinggi, kuning = 3, biru = 1-2, cokelat = seluruhnya rendah.")
    with col2:
        top8 = agg.sort_values("WIPI", ascending=False).head(8).reset_index()
        section_header(f"Delapan wilayah prioritas (rerata {years[0]}-{years[-1]}) membutuhkan arahan taktis yang berbeda-beda")
        show = pd.DataFrame({
            "Kabupaten/Kota": top8["KabKota"],
            "WIPI": top8["WIPI"].map(lambda v: id_num(v, 1)),
            "Profil": top8["Profil"].str.replace("-", "\u2011"),
            "Rekomendasi taktis (kriteria tinggi)": top8["Rekomendasi"],
        })
        html_table(show)

    st.write("")
    col3, col4 = st.columns([1, 1.5])
    with col3:
        urut = agg.sort_values("WIPI", ascending=False)
        section_header("Matriks kriteria: oranye = tinggi (perlu intervensi), biru = rendah (pertahankan)")
        zmat = np.array([[1 if urut.loc[k, v + "_ket"] == "Tinggi" else 0 for v in WIPI_VARS] for k in urut.index])
        txt = [["T" if z else "R" for z in r] for r in zmat]
        figm = go.Figure(go.Heatmap(
            z=zmat, x=WIPI_VARS, y=[f"{k} ({id_num(urut.loc[k, 'WIPI'], 0)})" for k in urut.index],
            colorscale=[[0, MUTED_BLUE], [1, SAFETY_ORANGE]], showscale=False, zmin=0, zmax=1,
            text=txt, texttemplate="%{text}", textfont=dict(color=BG_APP, size=12),
            xgap=3, ygap=3, hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
        ))
        figm.update_yaxes(autorange="reversed", tickfont=dict(size=10))
        figm.update_xaxes(side="top")
        style_fig(figm, height=700)
        st.plotly_chart(figm, config=chart_bar_config())
        caption("Ambang median: " + ", ".join(f"{v} {id_num(med[v], 2)}" for v in WIPI_VARS)
                + ". Angka di samping nama = WIPI rerata.")
    with col4:
        section_header("Delapan kriteria dan arah intervensinya")
        ref = []
        for v in WIPI_VARS:
            for ket in ["Tinggi", "Rendah"]:
                if (v, ket) in TEKS_SINGKAT:
                    arah = TEKS_SINGKAT[(v, ket)]
                else:
                    arah = f"Organik: {TEKS_SINGKAT[(v, ket, 'Organik')]}; anorganik: {TEKS_SINGKAT[(v, ket, 'Anorganik')]}"
                ref.append({"Kriteria": f"{v} {ket.lower()}", "Arti": KRITERIA_ARTI[v][0 if ket == 'Tinggi' else 1],
                            "Arah intervensi": arah})
        html_table(pd.DataFrame(ref))

        st.write("")
        d_last = df[df["Tahun"] == latest].set_index("KabKota")
        banding = pd.DataFrame({"Profil rerata": agg["Profil"], f"Profil {latest}": d_last["Profil_tahun"]})
        berubah = banding[banding["Profil rerata"] != banding[f"Profil {latest}"]]
        section_header(f"{len(berubah)} dari {len(banding)} wilayah berubah profil pada {latest}; tinjau ulang kriteria terkini")
        show = berubah.reset_index().rename(columns={"KabKota": "Kabupaten/Kota"})
        for c in show.columns[1:]:
            show[c] = show[c].str.replace("-", "\u2011")
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

    source = next((p for p in DATA_CANDIDATES if p.is_file()), None)
    if source is None:
        st.write("")
        source = st.file_uploader("Unggah Dataset_Sampah_Jawa_Tengah.xlsx", type=["xlsx"])

    st.markdown(
        '<div class="ww-sidebar-spacer"></div>'
        '<hr class="ww-sidebar-line"/>'
        '<div class="ww-sidebar-footer">Oleh Kelompok 5<br/>Data 2023 - 2025<br/>Provinsi Jawa Tengah</div>',
        unsafe_allow_html=True,
    )

if source is None:
    st.error(
        "Berkas Dataset_Sampah_Jawa_Tengah.xlsx tidak ditemukan. Letakkan file tersebut di folder yang sama "
        "dengan app.py, atau unggah lewat panel di sidebar."
    )
    st.stop()

try:
    data = compute_wastewise(source)
except DatasetError as err:
    st.error(f"Dataset tidak sesuai format. {err}")
    st.stop()

PAGE_RENDERERS = {
    "Beranda": render_beranda,
    "Profil Wilayah": render_profil_wilayah,
    "Klasterisasi": render_klasterisasi,
    "Analisis Gap": render_analisis_gap,
    "Prioritas Intervensi": render_prioritas_intervensi,
    "Rekomendasi": render_rekomendasi,
}
PAGE_RENDERERS[page](data)
