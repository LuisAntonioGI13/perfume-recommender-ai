import time
import re
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# =========================
# CONFIGURACIÓN
# =========================
#INPUT_CSV = "links_perfumes.csv"
#INPUT_CSV = "top_marcas_ventana_nueva(top1-top5).csv"
#INPUT_CSV = "top_marcas_ventana_nueva(top6-top15).csv"
#INPUT_CSV = "top_marcas_ventana_nuevatop(top16-top30).csv"
#INPUT_CSV = "top_marcas_ventana_nueva(top31-top36).csv"
#INPUT_CSV = "top_marcas_ventana_nueva(top37-top42).csv"
INPUT_CSV = "top_marcas_ventana_nueva(top43-top48).csv"
#INPUT_CSV = "top_marcas_ventana_nueva(top49-top50).csv"


#OUTPUT_CSV = "perfumes_extraidos.csv"
#OUTPUT_CSV = "perfumes_extraidos_1.csv"
#OUTPUT_CSV = "perfumes_extraidos_2.csv"
#OUTPUT_CSV = "perfumes_extraidos_3.csv"
#OUTPUT_CSV = "perfumes_extraidos_4.csv"
#OUTPUT_CSV = "perfumes_extraidos_5.csv"
OUTPUT_CSV = "perfumes_extraidos_6.csv"
#OUTPUT_CSV = "perfumes_extraidos_7.csv"

# Si tienes chromedriver en PATH, puedes dejar esto como None
CHROMEDRIVER_PATH = None  # Ejemplo: r"C:\Users\TU_USUARIO\Downloads\chromedriver.exe"

HEADLESS = False   # True = no abre ventana visible | False = sí abre ventana
ESPERA_SEGUNDOS = 15


# =========================
# FUNCIONES AUXILIARES
# =========================
def iniciar_driver(headless=False):
    options = Options()

    if headless:
        options.add_argument("--headless=new")

    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-infobars")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    if CHROMEDRIVER_PATH:
        service = Service(CHROMEDRIVER_PATH)
        driver = webdriver.Chrome(service=service, options=options)
    else:
        driver = webdriver.Chrome(options=options)

    return driver


def safe_find_text(driver, by, value):
    """
    Intenta encontrar un elemento y devolver su texto.
    Si no existe, devuelve None.
    """
    try:
        elem = driver.find_element(by, value)
        texto = elem.text.strip()
        return texto if texto else None
    except NoSuchElementException:
        return None


def safe_find_elements_text(driver, by, value):
    """
    Devuelve lista de textos no vacíos de elementos encontrados.
    """
    try:
        elems = driver.find_elements(by, value)
        textos = []
        for elem in elems:
            txt = elem.text.strip()
            if txt:
                textos.append(txt)
        return textos
    except Exception:
        return []


def limpiar_lista_textos(lista):
    """
    Limpia textos repetidos/vacíos y devuelve lista única conservando orden.
    """
    vistos = set()
    resultado = []
    for item in lista:
        item = item.strip()
        if item and item.lower() not in vistos:
            vistos.add(item.lower())
            resultado.append(item)
    return resultado


def extraer_rating_y_votes(texto):
    """
    Intenta sacar rating y votos desde un texto grande.
    """
    rating = None
    votes = None

    if not texto:
        return rating, votes

    rating_match = re.search(r'(\d\.\d+)\s*(?:out of 5|/5)?', texto, re.IGNORECASE)
    if rating_match:
        rating = rating_match.group(1)

    votes_match = re.search(r'([\d,]+)\s+votes?', texto, re.IGNORECASE)
    if votes_match:
        votes = votes_match.group(1).replace(",", "")

    return rating, votes


def cerrar_popups_si_aparecen(driver):
    """
    Intenta cerrar popups/cookies si aparecen.
    Esto puede requerir ajustes según lo que muestre Fragrantica.
    """
    posibles_selectores = [
        (By.XPATH, "//button[contains(., 'Accept')]"),
        (By.XPATH, "//button[contains(., 'I agree')]"),
        (By.XPATH, "//button[contains(., 'Aceptar')]"),
        (By.XPATH, "//button[contains(., 'Close')]"),
        (By.XPATH, "//button[contains(., 'Cerrar')]"),
        (By.CSS_SELECTOR, "button.fc-cta-consent"),
        (By.CSS_SELECTOR, ".fc-button.fc-cta-consent"),
    ]

    for by, selector in posibles_selectores:
        try:
            boton = WebDriverWait(driver, 2).until(
                EC.element_to_be_clickable((by, selector))
            )
            boton.click()
            time.sleep(1)
            break
        except Exception:
            continue


def extraer_info_perfume(driver, url):
    """
    Abre un perfume, extrae info principal y regresa un diccionario.
    """
    data = {
        "url": url,
        "nombre": None,
        "descripcion": None,
        "marca": None,
        "anio": None,
        "genero": None,
        "top_notes": None,
        "middle_notes": None,
        "base_notes": None,
        "accords": None,
        "rating": None,
        "votes": None,
        "error": None,
    }

    try:
        driver.get(url)

        WebDriverWait(driver, ESPERA_SEGUNDOS).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(2)

        cerrar_popups_si_aparecen(driver)

        # -------- TÍTULO / NOMBRE --------
        nombre = safe_find_text(driver, By.TAG_NAME, "h1")
        data["nombre"] = nombre

        # -------- DESCRIPCIÓN --------
        descripcion = None
        posibles_xpaths_descripcion = [
            "//div[contains(@class,'main-content')]//p",
            "//div[contains(@class,'cell small-12')]//p",
            "//p[contains(., 'is a')]",
        ]

        for xpath in posibles_xpaths_descripcion:
            try:
                elems = driver.find_elements(By.XPATH, xpath)
                for elem in elems:
                    txt = elem.text.strip()
                    txt_lower = txt.lower()

                    if (
                        len(txt) > 40
                        and " by " in txt_lower
                        and ("fragrance" in txt_lower or "perfume" in txt_lower)
                    ):
                        descripcion = txt
                        break

                if descripcion:
                    break
            except Exception:
                continue

        data["descripcion"] = descripcion

        # -------- MARCA --------
        if descripcion and " by " in descripcion:
            try:
                parte = descripcion.split(" by ", 1)[1]
                marca = parte.split(" is ", 1)[0].strip()
                data["marca"] = marca
            except Exception:
                pass

        # -------- AÑO --------
        if descripcion and "launched in" in descripcion.lower():
            match = re.search(r"launched in\s+(\d{4})", descripcion, re.IGNORECASE)
            if match:
                data["anio"] = match.group(1)

        # -------- GÉNERO --------
        if descripcion:
            desc_lower = descripcion.lower()
            if "for women and men" in desc_lower:
                data["genero"] = "women and men"
            elif "for men" in desc_lower:
                data["genero"] = "men"
            elif "for women" in desc_lower:
                data["genero"] = "women"

        # -------- NOTAS --------
        if descripcion:
            top_match = re.search(r"Top notes are (.*?);", descripcion, re.IGNORECASE)
            middle_match = re.search(r"middle notes are (.*?);", descripcion, re.IGNORECASE)
            base_match = re.search(r"base notes are (.*?)(?:\.|$)", descripcion, re.IGNORECASE)

            if top_match:
                data["top_notes"] = top_match.group(1).strip()
            if middle_match:
                data["middle_notes"] = middle_match.group(1).strip()
            if base_match:
                data["base_notes"] = base_match.group(1).strip()

        # -------- ACCORDS --------
        accords = []

        posibles_selectores_accords = [
            (By.XPATH, "//*[contains(text(),'Main accords')]/following::*[self::div or self::a][position()<=10]"),
            (By.XPATH, "//div[contains(@style,'background') and string-length(normalize-space()) > 0]"),
        ]

        for by, selector in posibles_selectores_accords:
            textos = safe_find_elements_text(driver, by, selector)

            textos = [
                t for t in textos
                if len(t) < 30 and not re.search(r'\d', t)
            ]

            textos = limpiar_lista_textos(textos)

            if len(textos) >= 3:
                accords = textos[:10]
                break

        if accords:
            data["accords"] = " | ".join(accords)

        # -------- TEXTO COMPLETO DE LA PÁGINA --------
        texto_pagina = ""
        try:
            texto_pagina = driver.find_element(By.TAG_NAME, "body").text
        except Exception:
            pass


        # -------- RATING / VOTES --------
        rating, votes = extraer_rating_y_votes(texto_pagina)
        data["rating"] = rating
        data["votes"] = votes

    except TimeoutException:
        data["error"] = "Timeout al cargar la página"
    except Exception as e:
        data["error"] = str(e)

    return data


# =========================
# PROCESO PRINCIPAL
# =========================
def main():
    df_links = pd.read_csv(INPUT_CSV)

    if "link" not in df_links.columns:
        raise ValueError("El CSV debe tener una columna llamada 'link'.")

    urls = df_links["link"].dropna().astype(str).tolist()

    resultados = []

    driver = iniciar_driver(headless=HEADLESS)

    try:
        for i, url in enumerate(urls, start=1):
            print("=" * 80)
            print(f"[{i}/{len(urls)}] Procesando: {url}")

            # Abrir en nueva pestaña
            driver.execute_script("window.open('');")
            driver.switch_to.window(driver.window_handles[-1])

            info = extraer_info_perfume(driver, url)
            resultados.append(info)

            print("Nombre:", info.get("nombre"))
            print("Marca:", info.get("marca"))
            print("Año:", info.get("anio"))
            print("Género:", info.get("genero"))
            print("Accords:", info.get("accords"))
            print("Rating:", info.get("rating"))
            print("Votes:", info.get("votes"))
            print("Error:", info.get("error"))

            # Cerrar pestaña actual
            driver.close()

            # Regresar a la pestaña principal
            driver.switch_to.window(driver.window_handles[0])

            time.sleep(2)

    finally:
        driver.quit()

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nProceso terminado.")
    print(f"Archivo generado: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()