"""Stock en vivo desde la tienda online (WooCommerce Store API).

merci-collection.com es un WordPress con WooCommerce. WooCommerce publica una API REST
pública de solo lectura (la misma que usa la web para mostrar productos):

    GET /wp-json/wc/store/v1/products                    -> modelos publicados
    GET /wp-json/wc/store/v1/products?type=variation     -> una fila por talla

Devuelve solo información pública (nombre, SKU, categorías, precio, oferta, stock por
talla), así que NO necesita claves ni expone datos de clientes.

PEDIDOS (opcional): con claves de la API REST de WooCommerce (permiso de solo LECTURA)
en el archivo tienda.env (excluido de git), los pedidos del cliente se consultan en vivo:

    GET /wp-json/wc/v3/orders?search=<teléfono>

Sin claves, o si la tienda no responde, se usa data/pedidos.csv.

El catálogo se guarda en memoria 5 minutos. Si la tienda no responde (sin internet),
data_loader usa la última copia descargada o, si no hay, data/stock.csv.
"""
from __future__ import annotations

import base64
import html
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

TIENDA_URL = os.environ.get("HILO_TIENDA_URL", "https://merci-collection.com").rstrip("/")
USAR_API = os.environ.get("HILO_STOCK_API", "1") != "0"   # HILO_STOCK_API=0 -> solo CSV
TTL_SEGUNDOS = 300          # cada cuánto se vuelve a consultar la tienda
REINTENTO_SEGUNDOS = 60     # si falló, no reintentar en cada mensaje
TIMEOUT = 10
POR_PAGINA = 100
TZ_BOLIVIA = timezone(timedelta(hours=-4))

_lock = threading.Lock()
_cache: dict = {"df": None, "categorias": {}, "t_ok": 0.0, "t_intento": 0.0,
                "actualizado": None, "error": None, "productos": 0}


def _get(ruta: str, params: dict) -> tuple[list, dict]:
    url = f"{TIENDA_URL}/wp-json/wc/store/v1/{ruta}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Hilo-Merci/1.0",
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8")), dict(r.headers)


def _todas_las_paginas(ruta: str, params: dict) -> list:
    datos, cab = _get(ruta, {**params, "per_page": POR_PAGINA, "page": 1})
    paginas = int(cab.get("X-WP-TotalPages") or cab.get("x-wp-totalpages") or 1)
    if paginas > 1:  # el resto de páginas en paralelo
        with ThreadPoolExecutor(max_workers=4) as ex:
            for extra in ex.map(lambda n: _get(ruta, {**params, "per_page": POR_PAGINA, "page": n})[0],
                                range(2, paginas + 1)):
                datos += extra
    return datos


def _precio(prices: dict, campo: str) -> float | None:
    try:
        return int(prices[campo]) / 10 ** int(prices.get("currency_minor_unit", 2))
    except (KeyError, TypeError, ValueError):
        return None


def _cantidad(item: dict) -> float | None:
    """'5 disponibles' -> 5; agotado -> 0; sin dato -> None."""
    if not item.get("is_in_stock", True):
        return 0.0
    m = re.search(r"\d+", (item.get("stock_availability") or {}).get("text", ""))
    return float(m.group(0)) if m else (item.get("low_stock_remaining") or None)


def _texto_plano(html_texto: str | None) -> str:
    """Descripción de la ficha (HTML) -> texto plano de una línea."""
    t = re.sub(r"<[^>]+>", " ", html_texto or "")
    return re.sub(r"\s+", " ", html.unescape(t)).strip()


def _descargar() -> tuple[pd.DataFrame, dict, int]:
    from .data_loader import _extraer_ref, _extraer_talla, limpiar_nombre_modelo

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_prod = ex.submit(_todas_las_paginas, "products", {})
        f_var = ex.submit(_todas_las_paginas, "products", {"type": "variation"})
        productos, variaciones = f_prod.result(), f_var.result()
    por_id = {p["id"]: p for p in productos}
    categorias = {c["name"]: c.get("link") for p in productos for c in p.get("categories", [])}

    # Ref del modelo: SKU del producto ("MC240904"), su nombre o el SKU de una talla
    sku_talla = {}
    for v in variaciones:
        sku_talla.setdefault(v.get("parent"), v.get("sku"))
    ref_de = {pid: _extraer_ref(p.get("sku"), p.get("name"), sku_talla.get(pid))
              for pid, p in por_id.items()}

    filas = []

    def fila(producto: dict, item: dict, talla) -> dict:
        prices = item.get("prices") or {}
        normal = _precio(prices, "regular_price")
        oferta = _precio(prices, "sale_price")
        cats = [c["name"] for c in producto.get("categories", [])]
        tags = [t["name"] for t in producto.get("tags", [])]
        cantidad = _cantidad(item)
        return {
            "_ref": ref_de.get(producto["id"]),
            "_talla": talla,
            "_nombre": limpiar_nombre_modelo(producto.get("name", "")),
            "_cantidad": cantidad,
            "_hay_stock": 1.0 if item.get("is_in_stock") else 0.0,
            "_publicado": True,
            "_precio": normal,
            "_precio_oferta": oferta if item.get("on_sale") and oferta and normal and oferta < normal else None,
            "_disponible": bool(item.get("is_in_stock")) and (cantidad is None or cantidad > 0),
            "_categorias": ", ".join(cats + tags),
            "_link": producto.get("permalink"),
            "_descripcion": _texto_plano(producto.get("description") or producto.get("short_description")),
        }

    for v in variaciones:
        padre = por_id.get(v.get("parent"))
        if padre is None:  # talla de un modelo no publicado
            continue
        valor = " ".join(a.get("value", "") for a in v.get("variation_attributes", []) or []) \
            or v.get("variation", "")
        filas.append(fila(padre, v, _extraer_talla(valor, v.get("sku"), None)))
    for p in productos:
        if p.get("type") != "variable":  # producto simple: una sola fila sin talla
            filas.append(fila(p, p, None))

    df = pd.DataFrame(filas)
    if not df.empty:
        df["_talla"] = df["_talla"].astype("Int64").astype(object).where(df["_talla"].notna(), None)
    return df, categorias, len(productos)


def _actualizar() -> None:
    try:
        df, categorias, n = _descargar()
        with _lock:
            _cache.update(df=df, categorias=categorias, t_ok=time.time(), error=None, productos=n,
                          actualizado=datetime.now(TZ_BOLIVIA).strftime("%H:%M"))
    except Exception as e:  # noqa: BLE001 - sin internet, tienda caída, JSON inesperado
        with _lock:
            _cache["error"] = f"{type(e).__name__}: {e}"
    finally:
        with _lock:
            _cache["actualizando"] = False


def catalogo(forzar: bool = False) -> pd.DataFrame | None:
    """Catálogo en vivo (o la última copia buena). None si nunca se pudo descargar.

    Solo la primera descarga hace esperar. Después, cuando la copia tiene más de 5 min,
    se devuelve la copia al instante y se actualiza en segundo plano.
    """
    if not USAR_API:
        return None
    ahora = time.time()
    with _lock:
        hay_copia = _cache["df"] is not None
        fresco = hay_copia and ahora - _cache["t_ok"] < TTL_SEGUNDOS
        esperar = _cache["error"] and ahora - _cache["t_intento"] < REINTENTO_SEGUNDOS
        if _cache.get("actualizando") or (not forzar and (fresco or esperar)):
            return _cache["df"]
        _cache["t_intento"], _cache["actualizando"] = ahora, True
    if hay_copia and not forzar:
        threading.Thread(target=_actualizar, daemon=True).start()
    else:
        _actualizar()
    return _cache["df"]


def link_categoria(nombre_estilo: str) -> str | None:
    """Link de la categoría de la tienda ('Wide Leg' -> .../product-category/wide-leg/)."""
    from .data_loader import normalizar_texto

    objetivo = normalizar_texto(nombre_estilo)
    for nombre, link in _cache["categorias"].items():
        if link and objetivo and objetivo in normalizar_texto(nombre):
            return link
    return None


def estado() -> dict:
    return {
        "activa": USAR_API,
        "ok": _cache["df"] is not None,
        "actualizado": _cache["actualizado"],
        "productos": _cache["productos"],
        "error": _cache["error"],
        "url": TIENDA_URL,
    }


def limpiar_cache() -> None:
    with _lock:
        _cache.update(df=None, categorias={}, t_ok=0.0, t_intento=0.0, error=None)
    _pedidos_cache.clear()
    _prueba.update(t=0.0, ok=None, error=None)


# --------------------------------------------------------------------------
# Tabla de medidas de cada modelo (guía de tallas de la ficha del producto)
# --------------------------------------------------------------------------
# Cada ficha de merci-collection.com tiene, debajo del selector de talla, una tabla
# "Medidas de la prenda por talla, en centímetros" (Cintura, Largo, Botapié…) o el aviso
# "La tabla de medidas de este modelo todavía no está publicada". La Store API no la
# incluye, así que se lee de la página pública de cada producto (solo lectura).
# Se guarda una copia en data/medidas.json para que funcione sin internet.
ARCHIVO_MEDIDAS = Path(__file__).resolve().parent.parent / "data" / "medidas.json"
TTL_MEDIDAS = 6 * 3600
_medidas: dict = {"tablas": None, "t": 0.0, "actualizando": False, "error": None}


def _pagina(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Hilo-Merci/1.0)"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8", "replace")


def _valor_cm(v: str) -> float | None:
    m = re.search(r"\d+(?:[.,]\d+)?", v or "")
    return float(m.group(0).replace(",", ".")) if m else None


def tabla_de_html(pagina: str) -> dict[int, dict[str, float]] | None:
    """{talla: {'Cintura': 76, 'Largo': 107, 'Botapié': 30}}; {} si aún no está publicada;
    None si la página no tiene guía de tallas."""
    i = pagina.find('id="merci-guia"')
    if i < 0:
        return None
    bloque = pagina[i:i + 10000]
    tabla = re.search(r"<table.*?</table>", bloque, re.S)
    if not tabla:
        return {}
    columnas = [_texto_plano(c) for c in re.findall(r'<th scope="col">(.*?)</th>', tabla.group(0), re.S)]
    filas: dict[int, dict[str, float]] = {}
    for fila in re.findall(r"<tr>(.*?)</tr>", tabla.group(0), re.S):
        talla = re.search(r'<th scope="row">\s*(\d{2})\s*</th>', fila)
        if not talla:
            continue
        valores = [_valor_cm(_texto_plano(v)) for v in re.findall(r"<td[^>]*>(.*?)</td>", fila, re.S)]
        filas[int(talla.group(1))] = {c: v for c, v in zip(columnas[1:], valores) if v is not None}
    return filas


def _descargar_medidas() -> dict[str, dict]:
    df = catalogo()
    if df is None or df.empty:
        raise RuntimeError("sin catálogo")
    links = df.dropna(subset=["_ref", "_link"]).groupby("_ref")["_link"].first()

    def una(item):
        ref, url = item
        try:
            return ref, tabla_de_html(_pagina(url))
        except Exception:  # noqa: BLE001 - una ficha que no carga no frena al resto
            return ref, None

    with ThreadPoolExecutor(max_workers=8) as ex:
        return {ref: t for ref, t in ex.map(una, links.items()) if t is not None}


def _actualizar_medidas() -> None:
    try:
        tablas = _descargar_medidas()
        _medidas.update(tablas=tablas, t=time.time(), error=None)
        try:
            ARCHIVO_MEDIDAS.parent.mkdir(exist_ok=True)
            ARCHIVO_MEDIDAS.write_text(json.dumps({"t": _medidas["t"], "tablas": tablas},
                                                  ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    except Exception as e:  # noqa: BLE001
        _medidas["error"] = f"{type(e).__name__}: {e}"
        _medidas["t"] = max(_medidas["t"], time.time() - TTL_MEDIDAS + REINTENTO_SEGUNDOS)
    finally:
        _medidas["actualizando"] = False


def _desde_archivo() -> None:
    try:
        d = json.loads(ARCHIVO_MEDIDAS.read_text(encoding="utf-8"))
        _medidas.update(tablas={ref: {int(t): v for t, v in tabla.items()}
                                for ref, tabla in d["tablas"].items()}, t=float(d.get("t", 0)))
    except (OSError, ValueError, KeyError, AttributeError):
        pass


def medidas(esperar: bool = True) -> dict[str, dict[int, dict]] | None:
    """Tablas de medidas de todos los modelos {ref: {talla: {...}}} (None si no hay)."""
    if _medidas["tablas"] is None:
        _desde_archivo()
    viejo = time.time() - _medidas["t"] > TTL_MEDIDAS
    if USAR_API and viejo and not _medidas["actualizando"]:
        _medidas["actualizando"] = True
        if _medidas["tablas"] is None and esperar:
            _actualizar_medidas()          # primera vez: hay que esperar (~5 s)
        else:
            threading.Thread(target=_actualizar_medidas, daemon=True).start()
    tablas = _medidas["tablas"]
    if tablas is not None:  # se guarda con la Ref como texto
        return {str(k): v for k, v in tablas.items()}
    return None


def medidas_de(ref: str | None) -> dict[int, dict] | None:
    """Tabla de un modelo; {} = la tienda aún no la publicó; None = no se sabe."""
    tablas = medidas()
    if not ref or tablas is None:
        return None
    return tablas.get(str(ref))


# --------------------------------------------------------------------------
# Pedidos en vivo (API REST de WooCommerce, requiere claves de solo lectura)
# --------------------------------------------------------------------------
ARCHIVO_CLAVES = Path(__file__).resolve().parent.parent / "tienda.env"
TTL_PEDIDOS = 120  # segundos que se recuerda la consulta de un teléfono
_pedidos_cache: dict[str, tuple[float, list]] = {}
_prueba: dict = {"t": 0.0, "ok": None, "error": None}


def _claves() -> tuple[str, str] | None:
    """Claves desde variables de entorno o tienda.env (WC_CONSUMER_KEY / WC_CONSUMER_SECRET)."""
    valores = {"WC_CONSUMER_KEY": os.environ.get("WC_CONSUMER_KEY", ""),
               "WC_CONSUMER_SECRET": os.environ.get("WC_CONSUMER_SECRET", "")}
    if ARCHIVO_CLAVES.exists():
        for linea in ARCHIVO_CLAVES.read_text(encoding="utf-8-sig").splitlines():
            if "=" in linea and not linea.strip().startswith("#"):
                k, v = linea.split("=", 1)
                if k.strip() in valores and not valores[k.strip()]:
                    valores[k.strip()] = v.strip().strip('"').strip("'")
    ck, cs = valores["WC_CONSUMER_KEY"], valores["WC_CONSUMER_SECRET"]
    return (ck, cs) if ck.startswith("ck_") and cs.startswith("cs_") else None


def pedidos_configurado() -> bool:
    return _claves() is not None


def _get_admin(ruta: str, params: dict) -> list:
    ck, cs = _claves()
    base = f"{TIENDA_URL}/wp-json/wc/v3/{ruta}"
    token = base64.b64encode(f"{ck}:{cs}".encode()).decode()
    req = urllib.request.Request(f"{base}?{urllib.parse.urlencode(params)}", headers={
        "User-Agent": "Hilo-Merci/1.0", "Accept": "application/json",
        "Authorization": f"Basic {token}"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code != 401:
            raise
    # Algunos hostings borran la cabecera Authorization: claves como parámetros (HTTPS)
    q = urllib.parse.urlencode({**params, "consumer_key": ck, "consumer_secret": cs})
    req = urllib.request.Request(f"{base}?{q}", headers={"User-Agent": "Hilo-Merci/1.0",
                                                         "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def _error_legible(e: Exception) -> str:
    if isinstance(e, urllib.error.HTTPError):
        if e.code == 401:
            return "Claves rechazadas (401): revisa tienda.env"
        if e.code == 403:
            return "Sin permiso (403): la clave debe tener permiso de Lectura"
        return f"La tienda respondió HTTP {e.code}"
    return f"{type(e).__name__}: {e}"


def probar_pedidos() -> dict:
    """Comprueba (cada 5 min) que las claves funcionen. No devuelve datos de clientes."""
    if not pedidos_configurado():
        return {"configurado": False, "ok": False, "error": None}
    def probar() -> None:
        try:
            _get_admin("orders", {"per_page": 1, "_fields": "id"})
            _prueba.update(t=time.time(), ok=True, error=None)
        except Exception as e:  # noqa: BLE001
            _prueba.update(t=time.time(), ok=False, error=_error_legible(e))
        finally:
            _prueba["probando"] = False

    if time.time() - _prueba["t"] > TTL_SEGUNDOS and not _prueba.get("probando"):
        _prueba["probando"] = True
        if _prueba["ok"] is None:
            probar()                       # primera vez: se espera (~2 s)
        else:                              # después: se actualiza sin frenar la página
            threading.Thread(target=probar, daemon=True).start()
    return {"configurado": True, "ok": _prueba["ok"], "error": _prueba["error"]}


def pedidos_por_telefono(tel_normalizado: str) -> list[dict] | None:
    """Pedidos cuyo teléfono de facturación coincide. None si no hay claves o falló la API."""
    if not pedidos_configurado() or len(tel_normalizado) < 7:
        return None
    from .data_loader import normalizar_telefono

    ahora = time.time()
    if tel_normalizado in _pedidos_cache and ahora - _pedidos_cache[tel_normalizado][0] < TTL_PEDIDOS:
        return _pedidos_cache[tel_normalizado][1]
    try:
        campos = "id,number,status,date_created,total,billing,shipping,line_items"
        encontrados = _get_admin("orders", {"search": tel_normalizado, "per_page": 100,
                                            "orderby": "date", "order": "desc", "_fields": campos})
    except Exception as e:  # noqa: BLE001 - sin internet, claves malas: se usa pedidos.csv
        _prueba.update(t=time.time(), ok=False, error=_error_legible(e))
        return None
    # La búsqueda de WooCommerce es por texto: se confirma que el teléfono sea exactamente ese
    pedidos = [o for o in encontrados
               if normalizar_telefono((o.get("billing") or {}).get("phone")) == tel_normalizado]
    _pedidos_cache[tel_normalizado] = (ahora, pedidos)
    return pedidos


def limpiar_cache_pedidos() -> None:
    _pedidos_cache.clear()
    _prueba.update(t=0.0, ok=None, error=None)
