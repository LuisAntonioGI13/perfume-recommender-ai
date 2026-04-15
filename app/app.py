import re
import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# CONFIG GENERAL
# =========================================================
st.set_page_config(
    page_title="ParfumerIA",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH_MAIN = BASE_DIR / "data" / "processed" / "df_perfumes_modelo_final.csv"
DATA_PATH_AUX = BASE_DIR / "data" / "processed" / "perfumes_aux_procesado.csv"
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo.png"

# =========================================================
# MAPA DE PRECIOS POR MARCA
# Reemplázalo por tus rangos reales cuando quieras afinarlo
# =========================================================
PRICE_BY_BRAND = {
    "lattafa perfumes": (700, 1400),
    "lattafa": (700, 1400),
    "maison alhambra": (600, 1300),
    "armaf": (700, 1600),
    "afnan": (900, 1800),
    "zara": (500, 1200),
    "versace": (1500, 2800),
    "giorgio armani": (2200, 3600),
    "dior": (2200, 4200),
    "yves saint laurent": (2200, 3800),
    "chanel": (2800, 5000),
    "jean paul gaultier": (2200, 3500),
    "dolce&gabbana": (1800, 3200),
    "dolce gabbana": (1800, 3200),
    "carolina herrera": (1800, 3200),
    "rabanne": (1900, 3200),
    "hugo boss": (1800, 3200),
    "prada": (2200, 3800),
    "gucci": (2200, 3800),
    "givenchy": (1800, 3200),
    "burberry": (1800, 3200),
    "mugler": (1800, 3200),
    "guerlain": (2000, 4500),
    "tom ford": (4500, 9000),
    "xerjoff": (5000, 9000),
    "creed": (6000, 11000),
    "amouage": (5000, 9000),
    "parfums de marly": (4500, 8000),
    "maison francis kurkdjian": (5500, 9500),
    "byredo": (4500, 7500),
    "diptyque": (3500, 6000),
    "mancera": (2500, 4200),
    "montale": (2200, 3800),
    "jo malone london": (2800, 5200),
    "louis vuitton": (6000, 10000),
    "natura": (400, 1200),
    "o boticario": (500, 1300),
    "o boticário": (500, 1300),
    "french avenue": (800, 1500),
    "paris corner": (700, 1400),
    "bath & body works": (500, 1200),
}

# =========================================================
# ESTILOS
# =========================================================
st.markdown(
    """
    <style>
        :root {
            --bg-main: #060b18;
            --bg-sidebar: #1a1e2b;
            --card-bg: rgba(255,255,255,0.03);
            --card-border: rgba(255,255,255,0.08);
            --text-main: #f5f7fb;
            --text-soft: #a9b3c7;
            --accent: #ff6b6b;
            --accent-2: #7c3aed;
        }

        .stApp {
            background: linear-gradient(180deg, #040816 0%, #060b18 100%);
            color: var(--text-main);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #202433 0%, #1a1e2b 100%);
            border-right: 1px solid rgba(255,255,255,0.05);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
            max-width: 1200px;
        }

        .app-shell {
            min-height: 78vh;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .hero-wrap {
            max-width: 760px;
            margin: 2.2rem auto 1rem auto;
            text-align: left;
        }

        .hero-title {
            display: flex;
            align-items: center;
            gap: 0.9rem;
            font-size: 3rem;
            font-weight: 800;
            color: white;
            margin-bottom: 0.4rem;
            line-height: 1.1;
        }

        .hero-subtitle {
            color: var(--text-soft);
            font-size: 1.06rem;
            margin-bottom: 1.1rem;
        }

        .hero-help {
            color: #dce3f1;
            font-size: 1rem;
            margin-top: 1.2rem;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .soft-card {
            border: 1px solid var(--card-border);
            background: var(--card-bg);
            border-radius: 18px;
            padding: 1rem 1rem;
            backdrop-filter: blur(8px);
        }

        .perfume-card {
            border: 1px solid var(--card-border);
            background: rgba(255,255,255,0.025);
            border-radius: 18px;
            padding: 1rem 1rem;
            margin-bottom: 0.9rem;
        }

        .pill {
            display: inline-block;
            padding: 0.28rem 0.62rem;
            border-radius: 999px;
            font-size: 0.82rem;
            border: 1px solid rgba(255,255,255,0.12);
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
            color: #e7ebf5;
            background: rgba(255,255,255,0.03);
        }

        .section-title {
            font-size: 1.45rem;
            font-weight: 700;
            margin-bottom: 0.7rem;
            color: white;
        }

        .metric-box {
            border: 1px solid var(--card-border);
            background: rgba(255,255,255,0.025);
            border-radius: 16px;
            padding: 1rem;
            text-align: center;
        }

        .metric-label {
            color: var(--text-soft);
            font-size: 0.92rem;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            font-size: 1.5rem;
            font-weight: 800;
            color: white;
        }

        .small-muted {
            color: var(--text-soft);
            font-size: 0.9rem;
        }

        .logo-side-title {
            font-size: 1.2rem;
            font-weight: 800;
            color: white;
            line-height: 1;
        }

        .logo-side-sub {
            font-size: 0.85rem;
            color: var(--text-soft);
            margin-top: 0.15rem;
        }

        .divider-soft {
            height: 1px;
            background: rgba(255,255,255,0.06);
            margin: 0.8rem 0 1rem 0;
        }

        div[data-testid="stChatMessage"] {
            border-radius: 18px;
        }

        .buy-guide-card {
            margin-top: 0.9rem;
            border: 1px solid rgba(255,255,255,0.10);
            background: linear-gradient(180deg, rgba(255,255,255,0.045) 0%, rgba(255,255,255,0.02) 100%);
            border-radius: 16px;
            padding: 0.95rem 1rem;
        }

        .buy-guide-title {
            font-size: 0.98rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0.45rem;
        }

        .buy-guide-text {
            color: #d7deec;
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .buy-guide-list {
            margin: 0.45rem 0 0 0.9rem;
            color: #d7deec;
            font-size: 0.9rem;
        }

        .buy-guide-list li {
            margin-bottom: 0.2rem;
        }

        .buy-links {
            margin-top: 0.45rem;
            margin-bottom: 0.65rem;
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
        }

        .buy-btn {
            display: inline-block;
            padding: 0.48rem 0.85rem;
            border-radius: 12px;
            text-decoration: none !important;
            color: #f5f7fb !important;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.04);
            font-size: 0.9rem;
            font-weight: 600;
        }

        .buy-btn:hover {
            border-color: rgba(255,255,255,0.22);
            background: rgba(255,255,255,0.07);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def limpiar_texto(texto: str) -> str:
    texto = str(texto).lower()
    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(r"[^a-z0-9áéíóúñü ,|.$-]", " ", texto)
    return texto.strip()


def normalizar_marca(marca: str) -> str:
    marca = str(marca).strip().lower()
    marca = marca.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    marca = re.sub(r"\s+", " ", marca)
    return marca


def construir_contexto_uso(row):
    tags = []

    if str(row.get("office", "")).lower() in ["1", "true", "office", "sí", "si"]:
        tags += ["office", "trabajo", "oficina"]

    if str(row.get("school", "")).lower() in ["1", "true", "school", "sí", "si"]:
        tags += ["school", "escuela"]

    if str(row.get("gym", "")).lower() in ["1", "true", "gym", "sí", "si"]:
        tags += ["gym", "gimnasio"]

    if str(row.get("date", "")).lower() in ["1", "true", "date", "sí", "si"]:
        tags += ["date", "cita", "romántico", "romantico"]

    if str(row.get("party", "")).lower() in ["1", "true", "party", "sí", "si"]:
        tags += ["party", "fiesta", "salida nocturna"]

    if str(row.get("day", "")).lower() in ["1", "true", "day", "sí", "si"]:
        tags += ["day", "día", "diario"]

    if str(row.get("night", "")).lower() in ["1", "true", "night", "sí", "si"]:
        tags += ["night", "noche"]

    if str(row.get("hot_weather", "")).lower() in ["1", "true", "hot", "sí", "si"]:
        tags += ["hot weather", "calor", "clima cálido", "clima calido"]

    if str(row.get("cold_weather", "")).lower() in ["1", "true", "cold", "sí", "si"]:
        tags += ["cold weather", "frío", "frio", "clima fresco"]

    best_time = str(row.get("best_time", "")).strip()
    weather = str(row.get("weather", "")).strip()
    occasion = str(row.get("occasion", "")).strip()

    if best_time:
        tags.append(best_time)
    if weather:
        tags.append(weather)
    if occasion:
        tags.append(occasion)

    return " ".join(tags)


def asignar_familia_perfumes_score(row: pd.Series) -> str:
    accords = str(row.get("accords", "")).lower()

    familias = {
        "Sweet / Gourmand": ["vanilla", "sweet", "gourmand", "caramel", "chocolate", "lactonic", "coffee", "cacao"],
        "Floral": ["floral", "white floral", "rose", "tuberose", "powdery", "yellow floral", "violet"],
        "Woody / Amber": ["woody", "amber", "warm spicy", "leather", "tobacco", "smoky", "oud", "balsamic"],
        "Green / Earthy": ["green", "earthy", "herbal", "mossy"],
        "Fresh / Citrus": ["citrus", "aromatic", "aquatic", "fresh", "ozonic", "marine"],
    }

    scores = {}
    for familia, keywords in familias.items():
        score = sum(1 for kw in keywords if kw in accords)
        scores[familia] = score

    max_score = max(scores.values()) if scores else 0
    if max_score == 0:
        return "Other"

    prioridad = [
        "Sweet / Gourmand",
        "Floral",
        "Woody / Amber",
        "Fresh / Citrus",
        "Green / Earthy",
    ]

    candidatas = [fam for fam, score in scores.items() if score == max_score]
    for fam in prioridad:
        if fam in candidatas:
            return fam

    return "Other"


def clasificar_precio(min_price, max_price) -> str:
    if pd.isna(min_price) and pd.isna(max_price):
        return "Sin dato"

    ref = max_price if pd.notna(max_price) else min_price

    if ref <= 1200:
        return "Económico"
    if ref <= 2200:
        return "Medio"
    if ref <= 4500:
        return "Premium"
    return "Lujo"


def clasificar_anio(anio) -> str:
    if pd.isna(anio):
        return "Sin dato"
    try:
        anio = int(anio)
    except Exception:
        return "Sin dato"

    if anio >= 2020:
        return "Reciente"
    if anio >= 2010:
        return "Moderno"
    return "Clásico"


def formatear_rango_precio(min_price, max_price) -> str:
    if pd.isna(min_price) and pd.isna(max_price):
        return "Sin dato"
    if pd.notna(min_price) and pd.notna(max_price):
        return f"${int(min_price):,} - ${int(max_price):,} MXN"
    if pd.notna(max_price):
        return f"Hasta ${int(max_price):,} MXN"
    return f"Desde ${int(min_price):,} MXN"


def crear_links_compra(nombre: str, marca: str):
    query = f"{nombre} {marca} perfume"
    q = quote_plus(query)
    ml = f"https://listado.mercadolibre.com.mx/{q}"
    amazon = f"https://www.amazon.com.mx/s?k={q}"
    return ml, amazon


def tokens_nombre(nombre: str) -> set:
    stop = {"for", "women", "men", "and", "eau", "de", "parfum", "perfume"}
    tokens = re.findall(r"[a-z0-9]+", str(nombre).lower())
    return {t for t in tokens if t not in stop}


def detectar_consulta_nombre_directo(texto: str) -> bool:
    texto = texto.strip().lower()

    patrones_detalle = [
        "qué me puedes decir de",
        "que me puedes decir de",
        "háblame de",
        "hablame de",
        "dime sobre",
        "info de",
        "información de",
        "informacion de",
        "qué opinas de",
        "que opinas de",
        "cómo huele",
        "como huele",
    ]

    if any(texto.startswith(p) for p in patrones_detalle):
        return True

    palabras = texto.split()
    if len(palabras) <= 4:
        return True

    return False


def extraer_nombre_consulta(texto: str) -> str:
    texto = texto.strip()
    texto_lower = texto.lower()

    patrones = [
        "qué me puedes decir de",
        "que me puedes decir de",
        "háblame de",
        "hablame de",
        "dime sobre",
        "info de",
        "información de",
        "informacion de",
        "qué opinas de",
        "que opinas de",
        "cómo huele",
        "como huele",
    ]

    for p in patrones:
        if texto_lower.startswith(p):
            return texto[len(p):].strip()

    return texto


def detectar_modo_consulta(texto: str) -> str:
    texto = texto.lower().strip()

    patrones_similaridad = [
        "similar a",
        "parecido a",
        "parecidos a",
        "como ",
        "inspirado en",
        "dupe de",
        "alternativa a",
        "alternativas a",
        "qué se parece a",
        "que se parece a",
        "qué perfumes se parecen a",
        "que perfumes se parecen a",
    ]

    patrones_presupuesto = [
        "económico",
        "economico",
        "económicos",
        "economicos",
        "barato",
        "baratos",
        "barata",
        "baratas",
        "accesible",
        "accesibles",
        "menos de",
        "hasta ",
        "máximo",
        "maximo",
        "presupuesto",
    ]

    patrones_anio = [
        "reciente",
        "recientes",
        "moderno",
        "modernos",
        "moderna",
        "modernas",
        "clásico",
        "clasico",
        "clásicos",
        "clasicos",
        "vintage",
    ]

    for patron in patrones_similaridad:
        if patron in texto:
            return "similaridad"

    if any(p in texto for p in patrones_presupuesto) or re.search(r"\b20\d{2}\b", texto):
        return "recomendacion"

    if any(p in texto for p in patrones_anio):
        return "recomendacion"

    return "detalle_o_recomendacion"


def extraer_perfume_referencia(texto: str) -> str:
    texto = texto.strip()

    patrones = [
        r"similar a (.+)",
        r"parecido a (.+)",
        r"parecidos a (.+)",
        r"como (.+)",
        r"inspirado en (.+)",
        r"dupe de (.+)",
        r"alternativa a (.+)",
        r"alternativas a (.+)",
        r"qué se parece a (.+)",
        r"que se parece a (.+)",
        r"qué perfumes se parecen a (.+)",
        r"que perfumes se parecen a (.+)",
    ]

    texto_lower = texto.lower()
    for patron in patrones:
        match = re.search(patron, texto_lower)
        if match:
            return match.group(1).strip()

    return texto


def extraer_filtros_consulta(texto: str) -> dict:
    texto = texto.lower().strip()

    filtros = {
        "precio_max": None,
        "precio_categoria": None,
        "categoria_anio": None,
        "anio_exacto": None,
    }

    match_precio = re.search(r"(menos de|hasta|maximo|máximo)\s*\$?\s*(\d{3,5})", texto)
    if match_precio:
        filtros["precio_max"] = int(match_precio.group(2))

    if any(p in texto for p in ["económico", "economico", "económicos", "economicos", "barato", "baratos", "barata", "baratas"]):
        filtros["precio_categoria"] = "Económico"
    elif any(p in texto for p in ["medio", "gama media"]):
        filtros["precio_categoria"] = "Medio"
    elif "premium" in texto:
        filtros["precio_categoria"] = "Premium"
    elif any(p in texto for p in ["lujo", "caro", "caros", "cara", "caras"]):
        filtros["precio_categoria"] = "Lujo"

    if any(p in texto for p in ["reciente", "recientes"]):
        filtros["categoria_anio"] = "Reciente"
    elif any(p in texto for p in ["moderno", "modernos", "moderna", "modernas"]):
        filtros["categoria_anio"] = "Moderno"
    elif any(p in texto for p in ["clásico", "clasico", "clásicos", "clasicos", "vintage"]):
        filtros["categoria_anio"] = "Clásico"

    match_anio = re.search(r"\b(20\d{2}|19\d{2})\b", texto)
    if match_anio:
        filtros["anio_exacto"] = int(match_anio.group(1))

    return filtros


def aplicar_filtros_resultados(df_resultados: pd.DataFrame, filtros: dict) -> pd.DataFrame:
    if df_resultados is None or not isinstance(df_resultados, pd.DataFrame) or df_resultados.empty:
        return df_resultados

    df = df_resultados.copy()

    if filtros.get("precio_max") is not None and "precio_min_mxn" in df.columns:
        df = df[df["precio_min_mxn"].fillna(df["precio_max_mxn"]) <= filtros["precio_max"]]

    if filtros.get("precio_categoria") is not None and "precio_categoria" in df.columns:
        df = df[df["precio_categoria"] == filtros["precio_categoria"]]

    if filtros.get("categoria_anio") is not None and "categoria_anio" in df.columns:
        df = df[df["categoria_anio"] == filtros["categoria_anio"]]

    if filtros.get("anio_exacto") is not None and "anio" in df.columns:
        df = df[df["anio"] == filtros["anio_exacto"]]

    return df.reset_index(drop=True)


def traducir_texto_ia(texto: str) -> str:
    if client is None or not texto:
        return texto

    texto = str(texto).strip()
    if not texto:
        return texto

    texto_lower = f" {texto.lower()} "
    pistas_es = [" el ", " la ", " de ", " y ", " para ", " con ", " en ", " una ", " un "]
    if sum(p in texto_lower for p in pistas_es) >= 2:
        return texto

    prompt = f"""
Traduce el siguiente texto al español de forma natural y clara.
No agregues información, no resumas, solo traduce.

Texto:
{texto}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": "Eres un traductor profesional de inglés a español."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return texto


def render_buy_guide_card():
    st.markdown(
        """
        <div class="buy-guide-card">
            <div class="buy-guide-title">Recomendación de compra</div>
            <div class="buy-guide-text">
                Para mayor seguridad, prioriza vendedores con señales de confianza como:
            </div>
            <ul class="buy-guide-list">
                <li>alta reputación o vendedor verificado</li>
                <li>muchas ventas concretadas</li>
                <li>calificaciones positivas recientes</li>
                <li>etiquetas como <b>Full</b>, <b>Mercado Líder</b> o equivalentes</li>
            </ul>
            <div class="buy-guide-text">
                ParfumerIA sugiere opciones con base en perfil olfativo, valoraciones y referencia de mercado.
                La compra final y la validación del vendedor son responsabilidad del usuario.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def cargar_datos(main_path: Path, aux_path: Path) -> pd.DataFrame:
    df_main = pd.read_csv(main_path)

    try:
        df_aux = pd.read_csv(aux_path)
    except Exception:
        df_aux = None

    df_main.columns = [str(c).strip().lower() for c in df_main.columns]

    if df_aux is not None:
        df_aux.columns = [str(c).strip() for c in df_aux.columns]

        df_aux = df_aux.rename(columns={
            "Perfume": "nombre",
            "Brand": "marca",
            "Gender": "genero",
            "Year": "anio",
            "Top": "top_notes",
            "Middle": "middle_notes",
            "Base": "base_notes",
            "Rating Value": "rating",
            "Rating Count": "votes",
        })

        df_aux.columns = [str(c).strip().lower() for c in df_aux.columns]

        if "accords" not in df_aux.columns:
            accord_cols = [c for c in ["mainaccord1", "mainaccord2", "mainaccord3", "mainaccord4", "mainaccord5"] if c in df_aux.columns]
            if accord_cols:
                df_aux["accords"] = df_aux[accord_cols].fillna("").astype(str).agg(", ".join, axis=1)
                df_aux["accords"] = df_aux["accords"].str.replace(r"(,\s*)+", ", ", regex=True).str.strip(", ").str.strip()

        if "descripcion" not in df_aux.columns:
            df_aux["descripcion"] = ""

    text_cols = [
        "nombre", "descripcion", "marca", "genero",
        "top_notes", "middle_notes", "base_notes", "accords"
    ]

    for col in text_cols:
        if col not in df_main.columns:
            df_main[col] = ""
        df_main[col] = df_main[col].fillna("")

    if df_aux is not None:
        for col in text_cols:
            if col not in df_aux.columns:
                df_aux[col] = ""

        df_aux = df_aux.fillna("")

        df_main["nombre"] = df_main["nombre"].astype(str).str.strip().str.lower()
        df_aux["nombre"] = df_aux["nombre"].astype(str).str.strip().str.lower()

        df_merge = df_main.merge(df_aux, on="nombre", how="left", suffixes=("", "_aux"))

        for col in text_cols:
            col_aux = f"{col}_aux"
            if col_aux in df_merge.columns:
                df_merge[col] = np.where(
                    (df_merge[col] == "") | (df_merge[col].isna()),
                    df_merge[col_aux],
                    df_merge[col]
                )

        cols_aux = [c for c in df_merge.columns if c.endswith("_aux")]
        df_main = df_merge.drop(columns=cols_aux)

    for col in ["anio", "rating", "votes"]:
        if col in df_main.columns:
            df_main[col] = pd.to_numeric(df_main[col], errors="coerce")
        else:
            df_main[col] = np.nan

    columnas_contexto = [
        "office", "school", "gym", "date", "party", "day", "night",
        "hot_weather", "cold_weather", "best_time", "weather", "occasion"
    ]
    for col in columnas_contexto:
        if col not in df_main.columns:
            df_main[col] = ""
        df_main[col] = df_main[col].fillna("")

    if "familia_olfativa" not in df_main.columns:
        df_main["familia_olfativa"] = df_main.apply(asignar_familia_perfumes_score, axis=1)
    else:
        df_main["familia_olfativa"] = df_main["familia_olfativa"].fillna("Other")

    df_main["contexto_uso"] = df_main.apply(construir_contexto_uso, axis=1)

    df_main["texto_modelo_olfativo"] = (
        df_main["accords"].astype(str) + " " +
        df_main["accords"].astype(str) + " " +
        df_main["top_notes"].astype(str) + " " +
        df_main["middle_notes"].astype(str) + " " +
        df_main["base_notes"].astype(str) + " " +
        df_main["contexto_uso"].astype(str)
    )

    df_main["texto_modelo_olfativo"] = df_main["texto_modelo_olfativo"].apply(limpiar_texto)

    df_main["marca_normalizada"] = df_main["marca"].apply(normalizar_marca)

    df_main["precio_min_mxn"] = df_main["marca_normalizada"].map(
        lambda x: PRICE_BY_BRAND.get(x, (np.nan, np.nan))[0]
    )
    df_main["precio_max_mxn"] = df_main["marca_normalizada"].map(
        lambda x: PRICE_BY_BRAND.get(x, (np.nan, np.nan))[1]
    )
    df_main["precio_categoria"] = df_main.apply(
        lambda row: clasificar_precio(row["precio_min_mxn"], row["precio_max_mxn"]),
        axis=1
    )
    df_main["categoria_anio"] = df_main["anio"].apply(clasificar_anio)
    df_main["precio_rango_texto"] = df_main.apply(
        lambda row: formatear_rango_precio(row["precio_min_mxn"], row["precio_max_mxn"]),
        axis=1
    )

    links = df_main.apply(
        lambda row: crear_links_compra(row.get("nombre", ""), row.get("marca", "")),
        axis=1
    )
    df_main["busqueda_ml"] = links.apply(lambda x: x[0])
    df_main["busqueda_amazon"] = links.apply(lambda x: x[1])

    return df_main.reset_index(drop=True)


@st.cache_resource
def construir_motor(df: pd.DataFrame):
    corpus = df["texto_modelo_olfativo"].fillna("").astype(str)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    tfidf = vectorizer.fit_transform(corpus)
    sim_matrix = cosine_similarity(tfidf, tfidf)
    return vectorizer, tfidf, sim_matrix


def buscar_perfume(nombre_perfume: str, df: pd.DataFrame, top_n: int = 10):
    nombre_perfume = nombre_perfume.lower().strip()
    match = df[df["nombre"].str.lower().str.contains(nombre_perfume, na=False)]
    if match.empty:
        return "No encontrado"
    return match[[
        "nombre", "marca", "familia_olfativa", "rating", "votes",
        "precio_rango_texto", "precio_categoria", "anio", "categoria_anio"
    ]].head(top_n).reset_index(drop=True)


def recomendar_perfumes(
    nombre_perfume: str,
    df: pd.DataFrame,
    sim_matrix,
    top_n: int = 5,
    min_rating: float = 0.0,
    min_votes: int = 0,
    peso_sim: float = 0.80,
    peso_rating: float = 0.05,
    peso_votes: float = 0.15,
    penalizar_misma_linea: bool = True,
    excluir_misma_marca: bool = False,
):
    nombre_perfume = nombre_perfume.lower().strip()

    exactos = df[df["nombre"].str.lower().str.strip() == nombre_perfume]

    if len(exactos) == 1:
        idx = exactos.index[0]
    elif len(exactos) > 1:
        return exactos[[
            "nombre", "marca", "familia_olfativa", "rating", "votes",
            "precio_rango_texto", "precio_categoria", "anio", "categoria_anio"
        ]].reset_index(drop=True)
    else:
        match = df[df["nombre"].str.lower().str.contains(nombre_perfume, na=False)]
        if match.empty:
            return "No encontrado"
        if len(match) > 1:
            return match[[
                "nombre", "marca", "familia_olfativa", "rating", "votes",
                "precio_rango_texto", "precio_categoria", "anio", "categoria_anio"
            ]].head(10).reset_index(drop=True)
        idx = match.index[0]

    perfume_base = df.loc[idx, "nombre"]
    marca_base = df.loc[idx, "marca"]
    familia_base = df.loc[idx, "familia_olfativa"]
    tokens_base = tokens_nombre(perfume_base)

    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    votos_log = np.log1p(df["votes"].fillna(0))
    votos_max = votos_log.max() if votos_log.max() > 0 else 1

    candidatos = []

    for i, sim in sim_scores[1:]:
        if i == idx:
            continue

        fila = df.loc[i]
        misma_familia = fila["familia_olfativa"] == familia_base

        if not misma_familia:
            continue

        rating_val = fila["rating"] if pd.notnull(fila["rating"]) else 0
        votes_val = fila["votes"] if pd.notnull(fila["votes"]) else 0

        if rating_val < min_rating:
            continue
        if votes_val < min_votes:
            continue

        if excluir_misma_marca and fila["marca"] == marca_base:
            continue

        sim_ajustada = sim

        if penalizar_misma_linea:
            tokens_candidato = tokens_nombre(fila["nombre"])
            overlap = len(tokens_base & tokens_candidato)
            if overlap >= 2:
                sim_ajustada -= 0.08
            elif overlap == 1:
                sim_ajustada -= 0.03

        rating_norm = rating_val / 10 if pd.notnull(rating_val) else 0
        votes_norm = np.log1p(votes_val) / votos_max if votos_max > 0 else 0

        score_final = (
            peso_sim * sim_ajustada +
            peso_rating * rating_norm +
            peso_votes * votes_norm
        )

        candidatos.append((i, sim, sim_ajustada, score_final))

    if len(candidatos) == 0:
        respaldo = df[df["familia_olfativa"] == familia_base].copy()
        if respaldo.empty:
            return "No encontrado"

        respaldo = respaldo.sort_values(by=["rating", "votes"], ascending=[False, False]).head(top_n)
        resultados = respaldo[[
            "nombre", "marca", "familia_olfativa", "rating", "votes", "accords",
            "descripcion", "top_notes", "middle_notes", "base_notes", "genero", "anio",
            "precio_min_mxn", "precio_max_mxn", "precio_rango_texto", "precio_categoria",
            "categoria_anio", "busqueda_ml", "busqueda_amazon"
        ]].copy()
        return resultados.reset_index(drop=True)

    candidatos = sorted(candidatos, key=lambda x: x[3], reverse=True)[:top_n]
    idxs = [c[0] for c in candidatos]

    resultados = df.loc[idxs, [
        "nombre", "marca", "familia_olfativa", "rating", "votes", "accords",
        "descripcion", "top_notes", "middle_notes", "base_notes", "genero", "anio",
        "precio_min_mxn", "precio_max_mxn", "precio_rango_texto", "precio_categoria",
        "categoria_anio", "busqueda_ml", "busqueda_amazon"
    ]].copy()

    resultados["similaridad"] = [round(c[1], 3) for c in candidatos]
    resultados["sim_ajustada"] = [round(c[2], 3) for c in candidatos]
    resultados["score_final"] = [round(c[3], 3) for c in candidatos]

    return resultados.reset_index(drop=True)


def recomendar_por_texto(
    texto_usuario: str,
    df: pd.DataFrame,
    vectorizer,
    tfidf_matrix,
    top_n: int = 5,
    min_rating: float = 0.0,
    min_votes: int = 0,
    peso_sim: float = 0.80,
    peso_rating: float = 0.05,
    peso_votes: float = 0.15,
):
    texto_usuario = limpiar_texto(texto_usuario)
    query_vec = vectorizer.transform([texto_usuario])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()

    candidatos = []
    votos_log = np.log1p(df["votes"].fillna(0))
    votos_max = votos_log.max() if votos_log.max() > 0 else 1

    for i in np.argsort(sims)[::-1]:
        fila = df.iloc[i]

        rating_val = fila["rating"] if pd.notnull(fila["rating"]) else 0
        votes_val = fila["votes"] if pd.notnull(fila["votes"]) else 0

        if rating_val < min_rating:
            continue
        if votes_val < min_votes:
            continue

        if sims[i] > 0:
            rating_norm = rating_val / 10 if pd.notnull(rating_val) else 0
            votes_norm = np.log1p(votes_val) / votos_max if votos_max > 0 else 0

            score_final = (
                peso_sim * sims[i] +
                peso_rating * rating_norm +
                peso_votes * votes_norm
            )

            candidatos.append((i, sims[i], score_final))

        if len(candidatos) >= max(top_n * 4, 15):
            break

    if len(candidatos) == 0:
        respaldo = df.sort_values(by=["rating", "votes"], ascending=[False, False]).head(top_n).copy()
        return respaldo[[
            "nombre", "marca", "familia_olfativa", "rating", "votes", "accords",
            "descripcion", "top_notes", "middle_notes", "base_notes", "genero", "anio",
            "precio_min_mxn", "precio_max_mxn", "precio_rango_texto", "precio_categoria",
            "categoria_anio", "busqueda_ml", "busqueda_amazon"
        ]].reset_index(drop=True)

    candidatos = sorted(candidatos, key=lambda x: x[2], reverse=True)[:top_n]
    idxs = [c[0] for c in candidatos]

    resultados = df.iloc[idxs][[
        "nombre", "marca", "familia_olfativa", "rating", "votes", "accords",
        "descripcion", "top_notes", "middle_notes", "base_notes", "genero", "anio",
        "precio_min_mxn", "precio_max_mxn", "precio_rango_texto", "precio_categoria",
        "categoria_anio", "busqueda_ml", "busqueda_amazon"
    ]].copy()

    resultados["similaridad"] = [round(c[1], 3) for c in candidatos]
    resultados["score_final"] = [round(c[2], 3) for c in candidatos]

    return resultados.reset_index(drop=True)


def recomendaciones_por_filtros(df: pd.DataFrame, filtros: dict, top_n: int = 5):
    df_filtrado = df.copy()

    if filtros.get("precio_max") is not None:
        df_filtrado = df_filtrado[
            df_filtrado["precio_min_mxn"].fillna(df_filtrado["precio_max_mxn"]) <= filtros["precio_max"]
        ]

    if filtros.get("precio_categoria") is not None:
        df_filtrado = df_filtrado[df_filtrado["precio_categoria"] == filtros["precio_categoria"]]

    if filtros.get("categoria_anio") is not None:
        df_filtrado = df_filtrado[df_filtrado["categoria_anio"] == filtros["categoria_anio"]]

    if filtros.get("anio_exacto") is not None:
        df_filtrado = df_filtrado[df_filtrado["anio"] == filtros["anio_exacto"]]

    if df_filtrado.empty:
        return df_filtrado

    df_filtrado = df_filtrado.sort_values(
        by=["rating", "votes"],
        ascending=[False, False]
    ).head(top_n)

    return df_filtrado[[
        "nombre", "marca", "familia_olfativa", "rating", "votes", "accords",
        "descripcion", "top_notes", "middle_notes", "base_notes", "genero", "anio",
        "precio_min_mxn", "precio_max_mxn", "precio_rango_texto", "precio_categoria",
        "categoria_anio", "busqueda_ml", "busqueda_amazon"
    ]].reset_index(drop=True)


def obtener_detalle_perfume(nombre_perfume: str, df: pd.DataFrame):
    nombre_perfume = nombre_perfume.lower().strip()

    exactos = df[df["nombre"].str.lower().str.strip() == nombre_perfume]
    if len(exactos) == 1:
        return exactos.iloc[0]

    match = df[df["nombre"].str.lower().str.contains(nombre_perfume, na=False)]
    if match.empty:
        return "No encontrado"

    if len(match) > 1:
        return match[[
            "nombre", "marca", "familia_olfativa", "rating", "votes",
            "precio_rango_texto", "precio_categoria", "anio", "categoria_anio"
        ]].head(10).reset_index(drop=True)

    return match.iloc[0]


def render_cards(resultados: pd.DataFrame):
    if isinstance(resultados, str):
        st.warning(resultados)
        return

    if resultados is None or resultados.empty:
        st.info("No se encontraron resultados.")
        return

    for _, row in resultados.iterrows():
        rating_val = row["rating"] if pd.notnull(row["rating"]) else 0
        votes_val = int(row["votes"]) if pd.notnull(row["votes"]) else 0

        st.markdown("<div class='perfume-card'>", unsafe_allow_html=True)
        st.markdown(f"### {row['nombre']}")
        st.markdown(f"**Marca:** {row['marca']}")

        pills = [
            f"<span class='pill'>{row.get('familia_olfativa', 'N/D')}</span>",
        ]

        if pd.notna(row.get("anio", np.nan)):
            pills.append(f"<span class='pill'>Año: {int(row['anio'])}</span>")

        if row.get("categoria_anio", "Sin dato") != "Sin dato":
            pills.append(f"<span class='pill'>{row['categoria_anio']}</span>")

        if row.get("precio_categoria", "Sin dato") != "Sin dato":
            pills.append(f"<span class='pill'>{row['precio_categoria']}</span>")

        if "similaridad" in resultados.columns:
            pills.append(f"<span class='pill'>Sim: {row['similaridad']:.3f}</span>")
        if "score_final" in resultados.columns:
            pills.append(f"<span class='pill'>Score: {row['score_final']:.3f}</span>")

        st.markdown("".join(pills), unsafe_allow_html=True)

        precio_txt = str(row.get("precio_rango_texto", "")).strip()
        if precio_txt and precio_txt != "Sin dato":
            st.markdown(f"**Precio estimado:** {precio_txt}")

        accords = str(row.get("accords", "")).strip()
        if accords:
            st.caption(accords)

        ml = row.get("busqueda_ml", "")
        amazon = row.get("busqueda_amazon", "")

        if ml or amazon:
            links_html = "<div class='buy-links'>"
            if ml:
                links_html += f"<a class='buy-btn' href='{ml}' target='_blank'>🛒 Mercado Libre</a>"
            if amazon:
                links_html += f"<a class='buy-btn' href='{amazon}' target='_blank'>📦 Amazon</a>"
            links_html += "</div>"
            st.markdown(links_html, unsafe_allow_html=True)
            render_buy_guide_card()

        st.markdown("</div>", unsafe_allow_html=True)


def render_detalle_perfume(perfume):
    if isinstance(perfume, str):
        st.warning(perfume)
        return

    if isinstance(perfume, pd.DataFrame):
        st.info("Se encontraron varias coincidencias. Sé más específico.")
        st.dataframe(perfume, use_container_width=True)
        return

    rating_val = perfume.get("rating", np.nan)
    votes_val = perfume.get("votes", np.nan)
    anio_val = perfume.get("anio", np.nan)

    st.markdown("<div class='perfume-card'>", unsafe_allow_html=True)
    st.markdown(f"## {perfume.get('nombre', 'Sin nombre')}")
    st.markdown(f"**Marca:** {perfume.get('marca', 'N/D')}")
    st.markdown(f"**Familia olfativa:** {perfume.get('familia_olfativa', 'N/D')}")
    st.markdown(f"**Género:** {perfume.get('genero', 'N/D')}")
    st.markdown(f"**Año:** {int(anio_val) if pd.notnull(anio_val) else 'N/D'}")
    st.markdown(f"**Categoría por año:** {perfume.get('categoria_anio', 'N/D')}")
    st.markdown(f"**Precio estimado:** {perfume.get('precio_rango_texto', 'Sin dato')}")
    st.markdown(f"**Categoría de precio:** {perfume.get('precio_categoria', 'Sin dato')}")

    st.markdown("### Accords")
    st.write(perfume.get("accords", ""))

    st.markdown("### Notas de salida")
    st.write(perfume.get("top_notes", ""))

    st.markdown("### Notas medias")
    st.write(perfume.get("middle_notes", ""))

    st.markdown("### Notas de fondo")
    st.write(perfume.get("base_notes", ""))

    st.markdown("### Descripción")
    descripcion_original = perfume.get("descripcion", "")
    descripcion_traducida = traducir_texto_ia(descripcion_original)
    st.write(descripcion_traducida)

    with st.expander("Ver descripción original", expanded=False):
        st.caption(descripcion_original)

    ml = perfume.get("busqueda_ml", "")
    amazon = perfume.get("busqueda_amazon", "")

    if ml or amazon:
        st.markdown("### Dónde buscarlo")
        links_html = "<div class='buy-links'>"
        if ml:
            links_html += f"<a class='buy-btn' href='{ml}' target='_blank'>🛒 Mercado Libre</a>"
        if amazon:
            links_html += f"<a class='buy-btn' href='{amazon}' target='_blank'>📦 Amazon</a>"
        links_html += "</div>"
        st.markdown(links_html, unsafe_allow_html=True)
        render_buy_guide_card()

    st.markdown("</div>", unsafe_allow_html=True)


def generar_recomendacion_ia(
    consulta_usuario: str,
    resultados: pd.DataFrame,
    modo: str = "descripcion",
) -> str:
    if client is None:
        return "No se encontró la API key. Verifica tu archivo .env"

    if resultados is None or len(resultados) == 0:
        return "No hay resultados para analizar."

    candidatos = resultados.head(5).copy()

    perfumes_contexto = []
    for _, row in candidatos.iterrows():
        perfumes_contexto.append(
            f"- {row.get('nombre', '')} | Marca: {row.get('marca', '')} | "
            f"Familia: {row.get('familia_olfativa', '')} | "
            f"Año: {row.get('anio', '')} | Precio: {row.get('precio_rango_texto', '')} | "
            f"Categoría precio: {row.get('precio_categoria', '')} | "
            f"Accords: {row.get('accords', '')} | "
            f"Top notes: {row.get('top_notes', '')} | "
            f"Middle notes: {row.get('middle_notes', '')} | "
            f"Base notes: {row.get('base_notes', '')}"
        )

    perfumes_texto = "\n".join(perfumes_contexto)

    if modo == "similaridad":
        instruccion = (
            "El usuario busca perfumes similares a uno de referencia. "
            "Explica cuál de los candidatos sería la mejor recomendación principal y menciona hasta 2 alternativas."
        )
    else:
        instruccion = (
            "El usuario describió el tipo de perfume que busca. "
            "Explica cuál de los candidatos encaja mejor con esa descripción y menciona hasta 2 alternativas."
        )

    prompt = f"""
Consulta del usuario:
{consulta_usuario}

Candidatos:
{perfumes_texto}

Tarea:
- {instruccion}
- Responde en español.
- Sé claro, útil y natural.
- No inventes notas que no estén en los candidatos.
- Toma en cuenta estilo, ocasión, clima, precio y año cuando sean relevantes.
- Si la consulta habla de presupuesto, prioriza claramente las opciones que mejor encajen en precio.
- No afirmes que un perfume es muy parecido a otro si el perfil no lo respalda.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un asesor experto en perfumes. "
                        "Das recomendaciones precisas, útiles y fáciles de entender."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ocurrió un error al consultar la IA: {e}"



def describir_perfume_con_ia(perfume) -> str:
    if client is None:
        return "No se encontró la API key. Verifica tu archivo .env"

    prompt = f"""
Perfume: {perfume.get('nombre', '')}
Marca: {perfume.get('marca', '')}
Familia olfativa: {perfume.get('familia_olfativa', '')}
Género: {perfume.get('genero', '')}
Año: {perfume.get('anio', '')}
Categoría por año: {perfume.get('categoria_anio', '')}
Precio estimado: {perfume.get('precio_rango_texto', '')}
Categoría de precio: {perfume.get('precio_categoria', '')}
Descripción original: {perfume.get('descripcion', '')}
Accords: {perfume.get('accords', '')}
Notas de salida: {perfume.get('top_notes', '')}
Notas medias: {perfume.get('middle_notes', '')}
Notas de fondo: {perfume.get('base_notes', '')}

Tarea:
Describe este perfume en español de forma clara y atractiva.
Incluye:
- cómo huele en términos simples
- qué tipo de persona podría usarlo
- en qué clima o estación funciona mejor
- para qué ocasiones encaja
- menciona el rango de precio de forma breve cuando aporte valor
- no recomiendes alternativas a menos que el usuario lo pida
- no inventes datos fuera de la información dada
"""

    try:
        response = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un experto en perfumes que explica fragancias de forma clara, elegante y útil."
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ocurrió un error al generar la descripción con IA: {e}"


def comparar_perfumes_con_ia(perfume_1, perfume_2) -> str:
    if client is None:
        return "No se encontró la API key. Verifica tu archivo .env"

    if isinstance(perfume_1, (str, pd.DataFrame)) or isinstance(perfume_2, (str, pd.DataFrame)):
        return "No pude comparar los perfumes porque uno de los resultados no es válido."

    prompt = f"""
Vas a comparar dos perfumes usando únicamente la información proporcionada.

PERFUME 1
Nombre: {perfume_1.get('nombre', '')}
Marca: {perfume_1.get('marca', '')}
Familia olfativa: {perfume_1.get('familia_olfativa', '')}
Género: {perfume_1.get('genero', '')}
Año: {perfume_1.get('anio', '')}
Categoría por año: {perfume_1.get('categoria_anio', '')}
Precio estimado: {perfume_1.get('precio_rango_texto', '')}
Categoría de precio: {perfume_1.get('precio_categoria', '')}
Accords: {perfume_1.get('accords', '')}
Notas de salida: {perfume_1.get('top_notes', '')}
Notas medias: {perfume_1.get('middle_notes', '')}
Notas de fondo: {perfume_1.get('base_notes', '')}
Descripción original: {perfume_1.get('descripcion', '')}

PERFUME 2
Nombre: {perfume_2.get('nombre', '')}
Marca: {perfume_2.get('marca', '')}
Familia olfativa: {perfume_2.get('familia_olfativa', '')}
Género: {perfume_2.get('genero', '')}
Año: {perfume_2.get('anio', '')}
Categoría por año: {perfume_2.get('categoria_anio', '')}
Precio estimado: {perfume_2.get('precio_rango_texto', '')}
Categoría de precio: {perfume_2.get('precio_categoria', '')}
Accords: {perfume_2.get('accords', '')}
Notas de salida: {perfume_2.get('top_notes', '')}
Notas medias: {perfume_2.get('middle_notes', '')}
Notas de fondo: {perfume_2.get('base_notes', '')}
Descripción original: {perfume_2.get('descripcion', '')}

TAREA
Compara ambos perfumes en español de forma clara y útil.

Incluye:
- cómo se parecen
- diferencias principales
- cuál se siente más fresco / más dulce / más intenso, si aplica
- cuál encaja mejor para clima cálido o fresco
- cuál encaja mejor para día, noche u ocasiones sociales
- una observación breve sobre precio/valor cuando tenga sentido
- un veredicto final tipo: "elige X si buscas..." y "elige Y si buscas..."

Reglas:
- No inventes datos.
- No uses información fuera de lo dado.
- Sé claro, útil y natural.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un experto en perfumes que compara fragancias de forma clara, práctica y honesta."
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ocurrió un error al generar la comparación con IA: {e}"


# =========================================================
# CARGA
# =========================================================
try:
    df_perfumes = cargar_datos(DATA_PATH_MAIN, DATA_PATH_AUX)
    vectorizer_olf, tfidf_olf, sim_matrix_olf = construir_motor(df_perfumes)
except FileNotFoundError as e:
    st.error(f"No se encontró uno de los archivos de datos: {e}")
    st.stop()
except Exception as e:
    st.error(f"Ocurrió un error al inicializar la app: {e}")
    st.stop()

# =========================================================
# SESSION STATE
# =========================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_results" not in st.session_state:
    st.session_state.last_results = None

if "last_detail" not in st.session_state:
    st.session_state.last_detail = None

if "last_mode" not in st.session_state:
    st.session_state.last_mode = None

if "last_ai_answer" not in st.session_state:
    st.session_state.last_ai_answer = None

if "last_query" not in st.session_state:
    st.session_state.last_query = None

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    if LOGO_PATH.exists():
        col_logo, col_text = st.columns([1, 3])
        with col_logo:
            st.image(str(LOGO_PATH), width=55)
        with col_text:
            st.markdown(
                """
                <div class="logo-side-title">ParfumerIA</div>
                <div class="logo-side-sub">AI fragrance intelligence</div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.markdown("## ParfumerIA")
        st.caption("AI fragrance intelligence")

    st.markdown("<div class='divider-soft'></div>", unsafe_allow_html=True)

    st.markdown("### Secciones")
    seccion = st.radio(
        "",
        ["Chat", "Dashboard", "Comparación"],
        label_visibility="collapsed"
    )

    st.markdown("<div class='divider-soft'></div>", unsafe_allow_html=True)

    st.markdown("### Ajustes")
    top_n = st.slider("Número de resultados", 3, 10, 5)
    min_rating = st.slider("Rating mínimo", 0.0, 10.0, 0.0, 0.1)
    min_votes = st.slider("Votos mínimos", 0, 5000, 0, 50)
    penalizar_misma_linea = st.toggle("Penalizar misma línea", value=True)
    excluir_misma_marca = st.toggle("Excluir misma marca", value=False)

    st.markdown("<div class='divider-soft'></div>", unsafe_allow_html=True)

    st.markdown(f"**Perfumes cargados:** {len(df_perfumes)}")
    st.markdown(f"**Marcas:** {df_perfumes['marca'].nunique()}")
    st.markdown(f"**Familias:** {df_perfumes['familia_olfativa'].nunique()}")

# =========================================================
# VISTA CHAT
# =========================================================
if seccion == "Chat":
    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    st.markdown("<div class='hero-wrap'>", unsafe_allow_html=True)

    if LOGO_PATH.exists():
        col1, col2 = st.columns([1, 8])
        with col1:
            st.image(str(LOGO_PATH), width=70)
        with col2:
            st.markdown("<div class='hero-title'>ParfumerIA</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='hero-subtitle'>AI fragrance intelligence — recomendaciones personalizadas de perfumes</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<div class='hero-title'>ParfumerIA</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='hero-subtitle'>AI fragrance intelligence — recomendaciones personalizadas de perfumes</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class='soft-card'>
            Puedes escribir cosas como:
            <br><br>
            • <b>Quiero un perfume fresco para oficina y calor</b><br>
            • <b>Busco algo dulce para citas en clima fresco</b><br>
            • <b>Algo parecido a Ultra Male</b><br>
            • <b>Háblame de Eros Flame</b><br>
            • <b>Perfumes económicos</b><br>
            • <b>Algo por menos de 1500</b><br>
            • <b>Perfumes recientes</b>
            <div class='hero-help'>🧠 ¿En qué te puedo ayudar?</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt_usuario = st.chat_input("Escribe tu mensaje aquí...")

    if prompt_usuario:
        st.session_state.chat_history.append({"role": "user", "content": prompt_usuario})
        st.session_state.last_query = prompt_usuario

        modo = detectar_modo_consulta(prompt_usuario)
        filtros = extraer_filtros_consulta(prompt_usuario)

        with st.chat_message("user"):
            st.markdown(prompt_usuario)

        with st.chat_message("assistant"):
            with st.spinner("Analizando consulta..."):
                respuesta_ia = ""
                resultados = None
                detalle_perfume = None

                st.session_state.last_results = None
                st.session_state.last_detail = None
                st.session_state.last_mode = None

                if modo == "similaridad":
                    perfume_referencia = extraer_perfume_referencia(prompt_usuario)

                    resultados = recomendar_perfumes(
                        perfume_referencia,
                        df_perfumes,
                        sim_matrix_olf,
                        top_n=max(top_n * 2, 8),
                        min_rating=min_rating,
                        min_votes=min_votes,
                        penalizar_misma_linea=penalizar_misma_linea,
                        excluir_misma_marca=excluir_misma_marca,
                    )

                    if isinstance(resultados, pd.DataFrame) and not resultados.empty:
                        resultados = aplicar_filtros_resultados(resultados, filtros).head(top_n)

                    if isinstance(resultados, pd.DataFrame) and not resultados.empty:
                        respuesta_ia = generar_recomendacion_ia(
                            consulta_usuario=prompt_usuario,
                            resultados=resultados,
                            modo="similaridad",
                        )
                        st.session_state.last_results = resultados
                        st.session_state.last_mode = "recomendacion"
                    else:
                        respuesta_ia = (
                            "No encontré buenas coincidencias para ese perfume con esos filtros. "
                            "Prueba con otra referencia o ajusta presupuesto, año o estilo."
                        )

                elif detectar_consulta_nombre_directo(prompt_usuario) and modo != "recomendacion":
                    nombre_consulta = extraer_nombre_consulta(prompt_usuario)
                    detalle_perfume = obtener_detalle_perfume(nombre_consulta, df_perfumes)

                    if isinstance(detalle_perfume, str):
                        respuesta_ia = (
                            "No encontré ese perfume en el catálogo. "
                            "Prueba con otro nombre o escribe solo la parte principal."
                        )

                    elif isinstance(detalle_perfume, pd.DataFrame):
                        if not detalle_perfume.empty:
                            respuesta_ia = "Encontré varias coincidencias. Sé más específico con el nombre."
                            st.dataframe(detalle_perfume, use_container_width=True)
                        else:
                            respuesta_ia = (
                                "No encontré ese perfume en el catálogo. "
                                "Prueba con otro nombre o escribe solo la parte principal."
                            )
                    else:
                        respuesta_ia = describir_perfume_con_ia(detalle_perfume)
                        st.session_state.last_detail = detalle_perfume
                        st.session_state.last_mode = "detalle"

                else:
                    resultados = recomendar_por_texto(
                        prompt_usuario,
                        df_perfumes,
                        vectorizer_olf,
                        tfidf_olf,
                        top_n=max(top_n * 3, 12),
                        min_rating=min_rating,
                        min_votes=min_votes,
                    )

                    if isinstance(resultados, pd.DataFrame) and not resultados.empty:
                        resultados_filtrados = aplicar_filtros_resultados(resultados, filtros)

                        if isinstance(resultados_filtrados, pd.DataFrame) and not resultados_filtrados.empty:
                            resultados = resultados_filtrados.head(top_n)
                        elif any(v is not None for v in filtros.values()):
                            resultados = recomendaciones_por_filtros(df_perfumes, filtros, top_n=top_n)
                        else:
                            resultados = resultados.head(top_n)
                    elif any(v is not None for v in filtros.values()):
                        resultados = recomendaciones_por_filtros(df_perfumes, filtros, top_n=top_n)

                    if isinstance(resultados, pd.DataFrame) and not resultados.empty:
                        respuesta_ia = generar_recomendacion_ia(
                            consulta_usuario=prompt_usuario,
                            resultados=resultados,
                            modo="descripcion",
                        )
                        st.session_state.last_results = resultados
                        st.session_state.last_mode = "recomendacion"
                    else:
                        respuesta_ia = (
                            "No encontré coincidencias claras. "
                            "Intenta describir el estilo, ocasión, clima, presupuesto o año que buscas."
                        )

            st.markdown(respuesta_ia)

        st.session_state.chat_history.append({"role": "assistant", "content": respuesta_ia})
        st.session_state.last_ai_answer = respuesta_ia

    if (
        st.session_state.last_mode == "recomendacion"
        and st.session_state.last_results is not None
        and isinstance(st.session_state.last_results, pd.DataFrame)
        and not st.session_state.last_results.empty
    ):
        st.markdown("<div class='section-title'>Recomendaciones sugeridas</div>", unsafe_allow_html=True)
        render_cards(st.session_state.last_results)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# VISTA DASHBOARD
# =========================================================
elif seccion == "Dashboard":
    st.markdown("<div class='section-title'>Dashboard del catálogo</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='metric-box'><div class='metric-label'>Perfumes</div><div class='metric-value'>{len(df_perfumes)}</div></div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"<div class='metric-box'><div class='metric-label'>Marcas</div><div class='metric-value'>{df_perfumes['marca'].nunique()}</div></div>",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"<div class='metric-box'><div class='metric-label'>Familias</div><div class='metric-value'>{df_perfumes['familia_olfativa'].nunique()}</div></div>",
            unsafe_allow_html=True
        )
    with c4:
        avg_rating = round(df_perfumes["rating"].dropna().mean(), 2) if df_perfumes["rating"].notna().any() else 0
        st.markdown(
            f"<div class='metric-box'><div class='metric-label'>Rating promedio</div><div class='metric-value'>{avg_rating}</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("### Explorar catálogo")

    familias = ["Todas"] + sorted(df_perfumes["familia_olfativa"].dropna().unique().tolist())
    generos = ["Todos"] + sorted(df_perfumes["genero"].dropna().astype(str).unique().tolist())
    precios = ["Todas"] + sorted(df_perfumes["precio_categoria"].dropna().astype(str).unique().tolist())
    categorias_anio = ["Todas"] + sorted(df_perfumes["categoria_anio"].dropna().astype(str).unique().tolist())

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        familia_sel = st.selectbox("Familia", familias, key="familia_dashboard")
    with col2:
        genero_sel = st.selectbox("Género", generos, key="genero_dashboard")
    with col3:
        precio_sel = st.selectbox("Precio", precios, key="precio_dashboard")
    with col4:
        anio_cat_sel = st.selectbox("Categoría año", categorias_anio, key="anio_cat_dashboard")
    with col5:
        ordenar_por = st.selectbox(
            "Ordenar por",
            ["rating", "votes", "anio", "nombre", "precio_min_mxn"],
            key="orden_dashboard"
        )

    df_view = df_perfumes.copy()

    if familia_sel != "Todas":
        df_view = df_view[df_view["familia_olfativa"] == familia_sel]
    if genero_sel != "Todos":
        df_view = df_view[df_view["genero"].astype(str) == genero_sel]
    if precio_sel != "Todas":
        df_view = df_view[df_view["precio_categoria"] == precio_sel]
    if anio_cat_sel != "Todas":
        df_view = df_view[df_view["categoria_anio"] == anio_cat_sel]

    asc = ordenar_por in ["nombre", "precio_min_mxn"]
    df_view = df_view.sort_values(by=ordenar_por, ascending=asc)

    st.dataframe(
        df_view[
            [
                "nombre", "marca", "familia_olfativa", "genero", "rating", "votes",
                "anio", "categoria_anio", "precio_rango_texto", "precio_categoria", "accords"
            ]
        ].reset_index(drop=True),
        use_container_width=True,
        height=520,
    )

# =========================================================
# VISTA COMPARACIÓN
# =========================================================
elif seccion == "Comparación":
    st.markdown("<div class='section-title'>Comparación de perfumes</div>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Compara dos perfumes del catálogo lado a lado.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        perfume_1 = st.text_input("Primer perfume", placeholder="Ej. Ultra Male", key="comp_1")
    with col2:
        perfume_2 = st.text_input("Segundo perfume", placeholder="Ej. Le Beau", key="comp_2")

    if st.button("Comparar perfumes", use_container_width=True):
        p1 = obtener_detalle_perfume(perfume_1, df_perfumes) if perfume_1.strip() else "No encontrado"
        p2 = obtener_detalle_perfume(perfume_2, df_perfumes) if perfume_2.strip() else "No encontrado"

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Perfume 1")
            render_detalle_perfume(p1)

        with c2:
            st.markdown("### Perfume 2")
            render_detalle_perfume(p2)

        # Resumen comparativo objetivo
        if not isinstance(p1, (str, pd.DataFrame)) and not isinstance(p2, (str, pd.DataFrame)):
            st.markdown("### Resumen comparativo")

            comparacion_df = pd.DataFrame({
                "Aspecto": [
                    "Marca",
                    "Familia olfativa",
                    "Género",
                    "Año",
                    "Categoría por año",
                    "Precio estimado",
                    "Categoría de precio",
                    "Rating",
                    "Votes",
                ],
                p1.get("nombre", "Perfume 1"): [
                    p1.get("marca", "N/D"),
                    p1.get("familia_olfativa", "N/D"),
                    p1.get("genero", "N/D"),
                    p1.get("anio", "N/D"),
                    p1.get("categoria_anio", "N/D"),
                    p1.get("precio_rango_texto", "N/D"),
                    p1.get("precio_categoria", "N/D"),
                    round(float(p1.get("rating", np.nan)), 2) if pd.notnull(p1.get("rating", np.nan)) else "N/D",
                    int(p1.get("votes", np.nan)) if pd.notnull(p1.get("votes", np.nan)) else "N/D",
                ],
                p2.get("nombre", "Perfume 2"): [
                    p2.get("marca", "N/D"),
                    p2.get("familia_olfativa", "N/D"),
                    p2.get("genero", "N/D"),
                    p2.get("anio", "N/D"),
                    p2.get("categoria_anio", "N/D"),
                    p2.get("precio_rango_texto", "N/D"),
                    p2.get("precio_categoria", "N/D"),
                    round(float(p2.get("rating", np.nan)), 2) if pd.notnull(p2.get("rating", np.nan)) else "N/D",
                    int(p2.get("votes", np.nan)) if pd.notnull(p2.get("votes", np.nan)) else "N/D",
                ],
            })

            st.dataframe(comparacion_df, use_container_width=True, hide_index=True)
            st.markdown("### Análisis comparativo IA")
            with st.spinner("Comparando perfumes..."):
                comparacion_ia = comparar_perfumes_con_ia(p1, p2)
            st.markdown(comparacion_ia)



# =========================================================
# UTILIDADES EXTRA
# =========================================================
with st.expander("Buscar coincidencias por nombre", expanded=False):
    perfume_busqueda = st.text_input(
        "Escribe el nombre de un perfume",
        placeholder="Ej. Le Beau, Khamrah, Dior Homme Cologne",
        key="busqueda_nombre_expander",
    )

    if st.button("Mostrar coincidencias", key="btn_coincidencias_expander", use_container_width=True):
        if perfume_busqueda.strip():
            resultados_busqueda = buscar_perfume(perfume_busqueda, df_perfumes, top_n=10)
            if isinstance(resultados_busqueda, str):
                st.warning(resultados_busqueda)
            else:
                st.dataframe(resultados_busqueda, use_container_width=True)

with st.expander("Describir un perfume del catálogo", expanded=False):
    perfume_detalle_input = st.text_input(
        "Nombre del perfume a describir",
        placeholder="Ej. Khamrah, Ultra Male, Acqua di Gio",
        key="detalle_perfume_input",
    )

    if st.button("Describir perfume", key="btn_describir_perfume", use_container_width=True):
        if perfume_detalle_input.strip():
            perfume_detalle = obtener_detalle_perfume(perfume_detalle_input, df_perfumes)
            render_detalle_perfume(perfume_detalle)

            if not isinstance(perfume_detalle, (str, pd.DataFrame)):
                st.markdown("### Interpretación IA")
                with st.spinner("Generando descripción..."):
                    descripcion_ia = describir_perfume_con_ia(perfume_detalle)
                st.markdown(descripcion_ia)