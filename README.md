# ParfumerIA  
### AI Fragrance Intelligence

Sistema inteligente de recomendación de perfumes basado en similitud olfativa, NLP y análisis contextual.

ParfumerIA permite encontrar fragancias ideales según:
- descripción en lenguaje natural
- clima (calor / frío)
- momento del día (día / noche)
- ocasión (oficina, citas, gimnasio, eventos)
- presupuesto estimado

Además, integra un asistente conversacional con IA que explica, recomienda y compara perfumes.

---

## Demo

Aplicación desplegada en la nube con Streamlit Cloud, integrando un sistema de recomendación basado en NLP y un asistente conversacional con IA.

[Ver aplicación en vivo](https://lagiparfumeria-ai.streamlit.app/)

---

## Funcionalidades principales

### Chat inteligente con IA
- Interpretación de consultas en lenguaje natural
- Explicación de perfumes
- Recomendaciones personalizadas
- Comparación entre fragancias

---

### Búsqueda por perfume
- Consulta directa por nombre
- Descripción automática con IA
- Notas olfativas (top, middle, base)
- Contexto de uso recomendado

---

### Comparación de perfumes
- Comparación lado a lado
- Análisis automático con IA:
  - similitudes
  - diferencias
  - ocasiones ideales
  - clima recomendado
  - veredicto final

---

### Sistema de precios estimados
- Rango de precios por marca
- Clasificación:
  - Económico
  - Medio
  - Premium
  - Lujo
- Enlaces directos a:
  - Mercado Libre
  - Amazon

---

### Recomendaciones de compra
- Links a plataformas reales
- Sugerencias de vendedores confiables
- Nota informativa sobre responsabilidad de compra

---

## Metodología

El sistema combina múltiples técnicas de ciencia de datos y NLP:

### 1. Procesamiento de texto (NLP)
- TF-IDF para vectorización de:
  - accords
  - notas olfativas
  - contexto de uso
- Limpieza y normalización de texto
- Similaridad mediante cosine similarity

---

### 2. Sistema de recomendación

Modelo híbrido basado en:

1. Similitud olfativa (TF-IDF + cosine similarity)
2. Contexto de uso (clima, ocasión, momento)
3. Reglas heurísticas (familia olfativa, notas dominantes)

El sistema prioriza relevancia de uso, no solo similitud matemática.

---

---

### 3. Feature Engineering

#### Contexto de uso
- Día / noche  
- Clima (calor / frío)  
- Ocasión (office, gym, date, party, school)

#### Perfil olfativo
- Accords principales
- Notas (salida, corazón, fondo)
- Familia olfativa automática

#### Precio
- Estimación por marca
- Clasificación por rango

#### Año
- Clasificación:
  - Clásico
  - Moderno
  - Reciente

---

## Dataset

- ~1200 perfumes  
- 48 marcas  
- Top 25 perfumes por marca  

Incluye:
- nombre
- marca
- descripción
- accords
- notas olfativas
- familia olfativa
- contexto de uso
- año
- precio estimado

---

## Data Collection (Scraping)

El dataset fue construido mediante extracción de datos desde plataformas especializadas en perfumería, principalmente Fragrantica, utilizando un enfoque de scraping controlado.

Los scripts ubicados en la carpeta `scraping/` permiten automatizar la construcción del dataset:

- `scrape_top_perfumes.py`: obtiene los perfumes más populares por marca y genera un listado de enlaces.
- `scrape_perfume_details.py`: extrae información detallada de cada fragancia (descripción, notas, accords, etc.).

### Flujo de ejecución

1. Ejecutar `scrape_top_perfumes.py` para generar los enlaces.
2. Ejecutar `scrape_perfume_details.py` para recolectar la información completa.
3. Los datos se almacenan en formato estructurado (CSV) para su posterior procesamiento.

Este proceso se realiza de forma controlada para garantizar la calidad de los datos y evitar sobrecargar las fuentes consultadas.

---

## Tecnologías

- Python  
- Pandas / NumPy  
- Scikit-learn  
- TF-IDF  
- Cosine Similarity  
- Selenium  
- Streamlit  
- OpenAI API  

---

## Estructura del proyecto

```bash
app/
├── app.py

data/
├── raw/
├── processed/

notebooks/
├── desarrollo_modelo.ipynb

scraping/
├── scrape_top_perfumes.py
├── scrape_perfume_details.py

assets/
├── logo.png

```

## Cómo ejecutar


pip install -r requirements.txt
streamlit run app/app.py



## Autor

Luis Antonio Guerrero Ibarra
