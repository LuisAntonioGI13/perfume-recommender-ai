# ParfumerIA 
Sistema de recomendación de perfumes basado en similitud olfativa, NLP y clustering.

ParfumerIA permite encontrar fragancias ideales según:
- ocasión (cita, oficina, gym)
- clima (frío o calor)
- momento del día
- presupuesto

---

## Demo
*(Próximamente: link de la app en Streamlit)*

---

## Metodología

El sistema combina múltiples técnicas de ciencia de datos y NLP:

### 1. Procesamiento de texto (NLP)
- TF-IDF para vectorizar descripciones y accords
- Reducción de dimensionalidad con TruncatedSVD (SVD)

### 2. Clustering
- KMeans para segmentar perfumes en grupos olfativos
- Evaluación con:
  - Método del codo
  - Silhouette Score
  - Davies-Bouldin
  - Calinski-Harabasz

### 3. Feature Engineering

Se generaron variables clave:

#### Contexto de uso
- Día / noche
- Clima (frío / calor)
- Ocasión (date, gym, office, school, party)

#### Precio
- Rango estimado por marca
- Clasificación:
  - Económico
  - Medio
  - Alto
  - Lujo

#### Popularidad
- Ranking por marca (top 25)
- Score de popularidad normalizado
- Rating estimado

---

## Sistema de recomendación

El modelo es híbrido y combina:

1. Filtrado contextual (uso real)
2. Similitud semántica (cosine similarity)
3. Popularidad (ranking proxy)

### Score final
score_final = 0.7 * similitud + 0.3 * popularidad


Esto permite recomendar perfumes no solo similares, sino también relevantes.

---

## Dataset

- 48 marcas
- 25 perfumes por marca
- ~1200 perfumes en total

Incluye:
- descripción
- accords
- notas (top, middle, base)
- cluster olfativo
- contexto de uso
- precio estimado
- score de popularidad

---

## Tecnologías

- Python
- Pandas / NumPy
- Scikit-learn
- TF-IDF / SVD
- Cosine Similarity
- Selenium (scraping)
- Streamlit

---

## Estructura del proyecto

```bash
app/
data/
├── raw/
├── processed/

notebooks/
├── desarrollo_modelo.ipynb

scraping/
├── scrape_top_perfumes.py
├── scrape_perfume_details.py
```

---

## Cómo ejecutar


pip install -r requirements.txt
streamlit run app/app.py


---

## 🔮 Próximas mejoras

- Deploy en producción con Streamlit
- Integración con input en lenguaje natural (IA)
- Conexión con APIs de precios en tiempo real
- Sistema de recomendaciones personalizadas

---

## Autor

Luis Antonio Guerrero