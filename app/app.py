import re
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Perfume Recommender AI", page_icon="🌫️", layout="wide")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "df_perfumes_modelo_final.csv"

# =========================
# ESTILOS
# =========================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2rem;
            max-width: 1150px;
        }
        .title {
            font-size: 2.4rem;
            font-weight: 800;
            margin-top: 0.5rem;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #9aa0a6;
            margin-bottom: 1.2rem;
            font-size: 1.02rem;
        }
        .card {
            border: 1px solid rgba(120,120,120,0.22);
            border-radius: 14px;
            padding: 1rem;
            margin-bottom: 0.8rem;
            background: rgba(255,255,255,0.02);
        }
        .pill {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            font-size: 0.82rem;
            border: 1px solid rgba(120,120,120,0.25);
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        .hero-box {
            padding: 1rem 1.1rem;
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 16px;
            background: rgba(255,255,255,0.02);
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# FUNCIONES AUXILIARES
# =========================
def limpiar_texto(texto: str) -> str:
    texto = str(texto).lower()
    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(r"[^a-z0-9áéíóúñü ,|.-]", " ", texto)
    return texto.strip()


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

    max_score = max(scores.values())
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


def tokens_nombre(nombre: str) -> set:
    stop = {"for", "women", "men", "and", "eau", "de", "parfum", "perfume"}
    tokens = re.findall(r"[a-z0-9]+", str(nombre).lower())
    return {t for t in tokens if t not in stop}


@st.cache_data
def cargar_datos(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    text_cols = ["nombre", "descripcion", "marca", "genero", "top_notes", "middle_notes", "base_notes", "accords"]
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    for col in ["anio", "rating", "votes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    if "familia_olfativa" not in df.columns:
        df["familia_olfativa"] = df.apply(asignar_familia_perfumes_score, axis=1)
    else:
        df["familia_olfativa"] = df["familia_olfativa"].fillna("Other")

    df["texto_modelo_olfativo"] = (
        df["accords"].fillna("") + " " +
        df["accords"].fillna("") + " " +
        df["top_notes"].fillna("") + " " +
        df["middle_notes"].fillna("") + " " +
        df["base_notes"].fillna("")
    )

    df["texto_modelo_olfativo"] = df["texto_modelo_olfativo"].apply(limpiar_texto)

    return df.reset_index(drop=True)


@st.cache_resource
def construir_motor(df: pd.DataFrame):
    corpus = df["texto_modelo_olfativo"].fillna("").astype(str)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    tfidf = vectorizer.fit_transform(corpus)
    sim_matrix = cosine_similarity(tfidf, tfidf)
    return vectorizer, tfidf, sim_matrix


def buscar_perfume(nombre_perfume: str, df: pd.DataFrame, top_n: int = 10):
    nombre_perfume = nombre_perfume.lower().strip()
    match = df[df["nombre"].str.lower().str.contains(nombre_perfume, na=False)]
    if match.empty:
        return "No encontrado"
    return match[["nombre", "marca", "familia_olfativa", "rating", "votes"]].head(top_n).reset_index(drop=True)


def recomendar_perfumes(
    nombre_perfume: str,
    df: pd.DataFrame,
    sim_matrix,
    top_n: int = 5,
    min_rating: float = 5.0,
    min_votes: int = 200,
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
        return exactos[["nombre", "marca", "familia_olfativa", "rating", "votes"]].reset_index(drop=True)
    else:
        match = df[df["nombre"].str.lower().str.contains(nombre_perfume, na=False)]
        if match.empty:
            return "No encontrado"
        if len(match) > 1:
            return match[["nombre", "marca", "familia_olfativa", "rating", "votes"]].head(10).reset_index(drop=True)
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
        rating_ok = pd.notnull(fila["rating"]) and fila["rating"] >= min_rating
        votes_ok = pd.notnull(fila["votes"]) and fila["votes"] >= min_votes

        if not (misma_familia and rating_ok and votes_ok):
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

        rating_norm = fila["rating"] / 10 if pd.notnull(fila["rating"]) else 0
        votes_norm = np.log1p(fila["votes"]) / votos_max if pd.notnull(fila["votes"]) else 0

        score_final = (
            peso_sim * sim_ajustada +
            peso_rating * rating_norm +
            peso_votes * votes_norm
        )

        candidatos.append((i, sim, sim_ajustada, score_final))

    candidatos = sorted(candidatos, key=lambda x: x[3], reverse=True)[:top_n]
    idxs = [c[0] for c in candidatos]

    resultados = df.loc[idxs, [
        "nombre", "marca", "familia_olfativa", "rating", "votes", "accords",
        "descripcion", "top_notes", "middle_notes", "base_notes", "genero", "anio"
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
    min_rating: float = 5.0,
    min_votes: int = 200,
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

        rating_ok = pd.notnull(fila["rating"]) and fila["rating"] >= min_rating
        votes_ok = pd.notnull(fila["votes"]) and fila["votes"] >= min_votes

        if rating_ok and votes_ok:
            rating_norm = fila["rating"] / 10 if pd.notnull(fila["rating"]) else 0
            votes_norm = np.log1p(fila["votes"]) / votos_max if pd.notnull(fila["votes"]) else 0

            score_final = (
                peso_sim * sims[i] +
                peso_rating * rating_norm +
                peso_votes * votes_norm
            )

            candidatos.append((i, sims[i], score_final))

        if len(candidatos) >= top_n:
            break

    candidatos = sorted(candidatos, key=lambda x: x[2], reverse=True)[:top_n]
    idxs = [c[0] for c in candidatos]

    resultados = df.iloc[idxs][[
        "nombre", "marca", "familia_olfativa", "rating", "votes", "accords",
        "descripcion", "top_notes", "middle_notes", "base_notes", "genero", "anio"
    ]].copy()

    resultados["similaridad"] = [round(c[1], 3) for c in candidatos]
    resultados["score_final"] = [round(c[2], 3) for c in candidatos]

    return resultados.reset_index(drop=True)


def obtener_detalle_perfume(nombre_perfume: str, df: pd.DataFrame):
    nombre_perfume = nombre_perfume.lower().strip()

    exactos = df[df["nombre"].str.lower().str.strip() == nombre_perfume]
    if len(exactos) == 1:
        return exactos.iloc[0]

    match = df[df["nombre"].str.lower().str.contains(nombre_perfume, na=False)]
    if match.empty:
        return "No encontrado"

    if len(match) > 1:
        return match[["nombre", "marca", "familia_olfativa", "rating", "votes"]].head(10).reset_index(drop=True)

    return match.iloc[0]


def render_cards(resultados: pd.DataFrame):
    if isinstance(resultados, str):
        st.warning(resultados)
        return

    if resultados.empty:
        st.info("No se encontraron resultados.")
        return

    for _, row in resultados.iterrows():
        rating_val = row["rating"] if pd.notnull(row["rating"]) else 0
        votes_val = int(row["votes"]) if pd.notnull(row["votes"]) else 0

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"### {row['nombre']}")
        st.markdown(f"**Marca:** {row['marca']}")
        st.markdown(
            f"<div class='pill'>{row['familia_olfativa']}</div>"
            f"<div class='pill'>Rating: {rating_val:.2f}</div>"
            f"<div class='pill'>Votes: {votes_val}</div>",
            unsafe_allow_html=True,
        )
        if "similaridad" in resultados.columns:
            st.markdown(f"<div class='pill'>Sim: {row['similaridad']:.3f}</div>", unsafe_allow_html=True)
        if "score_final" in resultados.columns:
            st.markdown(f"<div class='pill'>Score: {row['score_final']:.3f}</div>", unsafe_allow_html=True)

        accords = str(row.get("accords", "")).strip()
        if accords:
            st.caption(accords)

        st.markdown("</div>", unsafe_allow_html=True)


def render_detalle_perfume(perfume):
    if isinstance(perfume, str):
        st.warning(perfume)
        return

    if isinstance(perfume, pd.DataFrame):
        st.info("Se encontraron varias coincidencias. Sé más específico.")
        st.dataframe(perfume, width="stretch")
        return

    rating_val = perfume.get("rating", np.nan)
    votes_val = perfume.get("votes", np.nan)
    anio_val = perfume.get("anio", np.nan)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"## {perfume.get('nombre', 'Sin nombre')}")
    st.markdown(f"**Marca:** {perfume.get('marca', 'N/D')}")
    st.markdown(f"**Familia olfativa:** {perfume.get('familia_olfativa', 'N/D')}")
    st.markdown(f"**Género:** {perfume.get('genero', 'N/D')}")
    st.markdown(f"**Año:** {anio_val if pd.notnull(anio_val) else 'N/D'}")
    st.markdown(f"**Rating:** {round(float(rating_val), 2) if pd.notnull(rating_val) else 'N/D'}")
    st.markdown(f"**Votes:** {int(votes_val) if pd.notnull(votes_val) else 'N/D'}")

    st.markdown("### Accords")
    st.write(perfume.get("accords", ""))

    st.markdown("### Notas de salida")
    st.write(perfume.get("top_notes", ""))

    st.markdown("### Notas medias")
    st.write(perfume.get("middle_notes", ""))

    st.markdown("### Notas de fondo")
    st.write(perfume.get("base_notes", ""))

    st.markdown("### Descripción original")
    st.write(perfume.get("descripcion", ""))

    st.markdown("</div>", unsafe_allow_html=True)


def generar_recomendacion_ia(
    consulta_usuario: str,
    resultados: pd.DataFrame,
    modo: str = "descripcion",
) -> str:
    if client is None:
        return "No se encontró la API key. Verifica tu archivo .env"

    if resultados is None or resultados.empty:
        return "No hay resultados para analizar."

    candidatos = resultados.head(5).copy()

    perfumes_contexto = []
    for _, row in candidatos.iterrows():
        perfumes_contexto.append(
            f"- {row.get('nombre', '')} | Marca: {row.get('marca', '')} | "
            f"Familia: {row.get('familia_olfativa', '')} | "
            f"Rating: {row.get('rating', '')} | Votes: {row.get('votes', '')} | "
            f"Accords: {row.get('accords', '')} | "
            f"Top notes: {row.get('top_notes', '')} | "
            f"Middle notes: {row.get('middle_notes', '')} | "
            f"Base notes: {row.get('base_notes', '')}"
        )

    perfumes_texto = "\n".join(perfumes_contexto)

    if modo == "perfume_base":
        instruccion = (
            "El usuario busca perfumes similares a uno de referencia. "
            "Explica cuál de los candidatos sería la mejor recomendación principal y menciona 2 alternativas."
        )
    else:
        instruccion = (
            "El usuario describió el tipo de perfume que busca. "
            "Explica cuál de los candidatos encaja mejor con esa descripción y menciona 2 alternativas."
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
- Toma en cuenta estilo, vibe, ocasión y clima de forma razonable.
- Usa este formato:

Recomendación principal:
...

¿Por qué?
...

Alternativas:
1. ...
2. ...
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
            temperature=0.7,
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
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ocurrió un error al generar la descripción con IA: {e}"


def detectar_modo_consulta(texto: str) -> str:
    texto = texto.lower().strip()

    patrones_perfume = [
        "similar a",
        "parecido a",
        "como ",
        "inspirado en",
        "dupe de",
        "alternativa a",
    ]

    for patron in patrones_perfume:
        if patron in texto:
            return "perfume_base"

    return "descripcion"


def extraer_perfume_referencia(texto: str) -> str:
    texto = texto.strip()

    patrones = [
        r"similar a (.+)",
        r"parecido a (.+)",
        r"como (.+)",
        r"inspirado en (.+)",
        r"dupe de (.+)",
        r"alternativa a (.+)",
    ]

    texto_lower = texto.lower()
    for patron in patrones:
        match = re.search(patron, texto_lower)
        if match:
            return match.group(1).strip()

    return texto


# =========================
# APP
# =========================
st.markdown("<div class='title'>Perfume Recommender AI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Tu asistente para encontrar perfumes por estilo, ocasión, clima o referencia.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='hero-box'>
        Escribe algo como:
        <br><br>
        • <b>Quiero un perfume fresco para oficina y calor</b><br>
        • <b>Busco algo dulce para citas en clima fresco</b><br>
        • <b>Algo parecido a Ultra Male</b><br>
        • <b>Quiero un perfume elegante, serio y versátil</b>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    df_perfumes = cargar_datos(DATA_PATH)
    vectorizer_olf, tfidf_olf, sim_matrix_olf = construir_motor(df_perfumes)
except FileNotFoundError:
    st.error(f"No se encontró el archivo de datos en: {DATA_PATH}")
    st.stop()
except Exception as e:
    st.error(f"Ocurrió un error al inicializar la app: {e}")
    st.stop()

with st.sidebar:
    st.header("Ajustes")
    top_n = st.slider("Número de resultados", 3, 10, 5)
    min_rating = st.slider("Rating mínimo", 0.0, 10.0, 5.0, 0.1)
    min_votes = st.slider("Votos mínimos", 0, 5000, 200, 50)
    penalizar_misma_linea = st.toggle("Penalizar misma línea", value=True)
    excluir_misma_marca = st.toggle("Excluir misma marca", value=False)

    st.divider()
    st.markdown(f"**Perfumes cargados:** {len(df_perfumes)}")
    st.markdown(f"**Familias:** {df_perfumes['familia_olfativa'].nunique()}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_results" not in st.session_state:
    st.session_state.last_results = None

if "last_ai_answer" not in st.session_state:
    st.session_state.last_ai_answer = None

if "last_query" not in st.session_state:
    st.session_state.last_query = None

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt_usuario = st.chat_input("Cuéntame qué perfume buscas...")

if prompt_usuario:
    st.session_state.chat_history.append({"role": "user", "content": prompt_usuario})
    st.session_state.last_query = prompt_usuario

    modo = detectar_modo_consulta(prompt_usuario)

    with st.chat_message("user"):
        st.markdown(prompt_usuario)

    with st.chat_message("assistant"):
        with st.spinner("Buscando recomendaciones..."):
            if modo == "perfume_base":
                perfume_referencia = extraer_perfume_referencia(prompt_usuario)
                resultados = recomendar_perfumes(
                    perfume_referencia,
                    df_perfumes,
                    sim_matrix_olf,
                    top_n=top_n,
                    min_rating=min_rating,
                    min_votes=min_votes,
                    penalizar_misma_linea=penalizar_misma_linea,
                    excluir_misma_marca=excluir_misma_marca,
                )
            else:
                resultados = recomendar_por_texto(
                    prompt_usuario,
                    df_perfumes,
                    vectorizer_olf,
                    tfidf_olf,
                    top_n=top_n,
                    min_rating=min_rating,
                    min_votes=min_votes,
                )

            if isinstance(resultados, pd.DataFrame) and not resultados.empty:
                respuesta_ia = generar_recomendacion_ia(
                    consulta_usuario=prompt_usuario,
                    resultados=resultados,
                    modo=modo,
                )
            else:
                respuesta_ia = "No encontré resultados suficientemente sólidos con los filtros actuales."

        st.markdown(respuesta_ia)

    st.session_state.chat_history.append({"role": "assistant", "content": respuesta_ia})
    st.session_state.last_results = resultados
    st.session_state.last_ai_answer = respuesta_ia

if st.session_state.last_results is not None:
    st.markdown("## Recomendaciones sugeridas")
    render_cards(st.session_state.last_results)

with st.expander("Buscar coincidencias por nombre", expanded=False):
    perfume_busqueda = st.text_input(
        "Escribe el nombre de un perfume",
        placeholder="Ej. Le Beau, Khamrah, Dior Homme Cologne",
        key="busqueda_nombre_expander",
    )

    if st.button("Mostrar coincidencias", key="btn_coincidencias_expander", width="stretch"):
        if perfume_busqueda.strip():
            resultados_busqueda = buscar_perfume(perfume_busqueda, df_perfumes, top_n=10)
            if isinstance(resultados_busqueda, str):
                st.warning(resultados_busqueda)
            else:
                st.dataframe(resultados_busqueda, width="stretch")

with st.expander("Describir un perfume del catálogo", expanded=False):
    perfume_detalle_input = st.text_input(
        "Nombre del perfume a describir",
        placeholder="Ej. Khamrah, Ultra Male, Acqua di Gio",
        key="detalle_perfume_input",
    )

    if st.button("Describir perfume", key="btn_describir_perfume", width="stretch"):
        if perfume_detalle_input.strip():
            perfume_detalle = obtener_detalle_perfume(perfume_detalle_input, df_perfumes)
            render_detalle_perfume(perfume_detalle)

            if not isinstance(perfume_detalle, (str, pd.DataFrame)):
                st.markdown("### Interpretación IA")
                with st.spinner("Generando descripción..."):
                    descripcion_ia = describir_perfume_con_ia(perfume_detalle)
                st.markdown(descripcion_ia)

with st.expander("Explorar catálogo", expanded=False):
    familias = ["Todas"] + sorted(df_perfumes["familia_olfativa"].dropna().unique().tolist())
    generos = ["Todos"] + sorted(df_perfumes["genero"].dropna().unique().tolist())

    col1, col2, col3 = st.columns(3)
    with col1:
        familia_sel = st.selectbox("Familia", familias, key="familia_catalogo")
    with col2:
        genero_sel = st.selectbox("Género", generos, key="genero_catalogo")
    with col3:
        ordenar_por = st.selectbox("Ordenar por", ["rating", "votes", "anio", "nombre"], key="orden_catalogo")

    df_view = df_perfumes.copy()

    if familia_sel != "Todas":
        df_view = df_view[df_view["familia_olfativa"] == familia_sel]
    if genero_sel != "Todos":
        df_view = df_view[df_view["genero"] == genero_sel]

    asc = ordenar_por == "nombre"
    df_view = df_view.sort_values(by=ordenar_por, ascending=asc)

    st.dataframe(
        df_view[
            ["nombre", "marca", "familia_olfativa", "genero", "rating", "votes", "anio", "accords"]
        ].reset_index(drop=True),
        width="stretch",
        height=500,
    )