from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time
import re

MARCAS = [
    "Maison-Francis-Kurkdjian",
    "Creed",
    "Hermes",
    "Prada",
    "Gucci",
    "Lancome",
    "Natura",
    "Jo-Malone-London",
    "Montale",
    "Burberry",
    "Mugler",
    "O-Boticario",
    "Hugo-Boss",
    "Rasasi",
    "Victoria-s-Secret",
    "Kayali-Fragrances",
    "Nishane",
    "Serge-Lutens",
    "Penhaligon-s",
    "Initio-Parfums-Prives"
]
TOP_N = 25
TIEMPO_PARA_AJUSTE_MANUAL = 20
ESPERA_ENTRE_MARCAS = 8
ARCHIVO_SALIDA = "top_marcas_ventana_nueva.csv"


def crear_driver():
    options = Options()
    options.page_load_strategy = "eager"
    # options.add_argument("--headless=new")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    driver.maximize_window()
    return driver


def normalizar_marca(marca_url: str) -> str:
    return marca_url.replace("-", " ").strip()


def limpiar_nombre_desde_url(href: str) -> str:
    nombre = href.rstrip("/").split("/")[-1]
    nombre = re.sub(r"-\d+\.html$", "", nombre)
    nombre = nombre.replace("-", " ").strip()
    return nombre


def extraer_perfumes_unicos(driver, marca_slug: str, top_n: int = 25):
    xpath = f"//a[contains(@href, '/perfume/{marca_slug}/')]"
    links = driver.find_elements(By.XPATH, xpath)

    datos = []
    vistos = set()

    for link in links:
        try:
            href = link.get_attribute("href")

            if not href:
                continue
            if href in vistos:
                continue
            if f"/perfume/{marca_slug}/" not in href:
                continue

            nombre = limpiar_nombre_desde_url(href)

            vistos.add(href)

            datos.append({
                "marca": normalizar_marca(marca_slug),
                "nombre": nombre,
                "link": href
            })

            if len(datos) >= top_n:
                break

        except:
            pass

    df = pd.DataFrame(datos)

    if not df.empty:
        df = df.drop_duplicates(subset=["link"]).reset_index(drop=True)
        df["rank_marca"] = range(1, len(df) + 1)

    return df


def procesar_marca(marca_slug: str, top_n: int = 25):
    driver = crear_driver()

    try:
        url = f"https://www.fragrantica.com/designers/{marca_slug}.html"
        driver.get(url)
        time.sleep(8)

        print(f"\nAbierta marca: {marca_slug}")
        print("URL actual:", driver.current_url)
        print("Título:", driver.title)

        print(f"\n>>> Tienes {TIEMPO_PARA_AJUSTE_MANUAL} segundos para dejar visible el grid correcto.")
        time.sleep(TIEMPO_PARA_AJUSTE_MANUAL)

        df = extraer_perfumes_unicos(driver, marca_slug, top_n=top_n)

        print(f"Total extraído para {marca_slug}: {len(df)}")
        if not df.empty:
            print(df.head(10))

        return df

    finally:
        driver.quit()
        print("Ventana cerrada.")


def main():
    todos = []

    for i, marca in enumerate(MARCAS, start=1):
        print(f"\n========== Marca {i}/{len(MARCAS)} ==========")

        try:
            df_marca = procesar_marca(marca, TOP_N)

            if not df_marca.empty:
                todos.append(df_marca)

        except Exception as e:
            print(f"Error con {marca}: {e}")

        print(f"\nEsperando {ESPERA_ENTRE_MARCAS} segundos antes de la siguiente marca...")
        time.sleep(ESPERA_ENTRE_MARCAS)

    if todos:
        df_final = pd.concat(todos, ignore_index=True)
        df_final.to_csv(ARCHIVO_SALIDA, index=False, encoding="utf-8-sig")

        print(f"\nArchivo guardado: {ARCHIVO_SALIDA}")
        print(f"Total final: {len(df_final)}")
        print(df_final.head(20))
    else:
        print("No se obtuvo información.")


if __name__ == "__main__":
    main()