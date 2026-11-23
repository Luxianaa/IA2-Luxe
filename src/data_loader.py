"""Acceso a los datos reales de Merci Collection.

- pedidos.csv  -> export de WooCommerce (una fila por artículo de cada pedido)
- stock        -> EN VIVO desde la tienda (WooCommerce Store API, ver tienda_api.py);
                  si no hay conexión, stock.csv (export de WooCommerce) como respaldo
- couriers.csv -> oficinas centrales del courier por ciudad

Todos los CSV se leen con encoding="utf-8-sig".
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PEDIDOS_CSV = DATA_DIR / "pedidos.csv"
STOCK_CSV = DATA_DIR / "stock.csv"
COURIERS_CSV = DATA_DIR / "couriers.csv"
DATASET_CSV = DATA_DIR / "dataset.csv"
ENCODING = "utf-8-sig"

# Columnas exactas del export de WooCommerce (pedidos.csv)
COL_FECHA = "Fecha del pedido"
COL_NUM = "Número de pedido"
COL_NOMBRE = "Nombre (envío)"
COL_APELLIDO = "Apellidos (envío)"
COL_IMPORTE = "Importe total del pedido"
COL_CIUDAD = "Ciudad (envío)"
COL_TEL = "Teléfono (facturación)"
COL_ARTICULO = "Nombre del artículo"
COL_DIRECCION = "Dirección lineas 1 y 2 (envío)"
COL_ESTADO = "Estado del pedido"
COL_EMAIL = "Correo electrónico (facturación)"

# Estados que NO cuentan como compra real para "cliente_frecuente"
ESTADOS_NO_COMPRA = {"cancelado", "fallido"}
MIN_COMPRAS_FRECUENTE = 2

# Columnas de couriers.csv
COURIER_COLS = ["ciudad", "codigo", "encargado", "telefono_fijo",
                "celular_whatsapp", "direccion", "horario"]


# --------------------------------------------------------------------------
# Normalización
# --------------------------------------------------------------------------
def normalizar_texto(texto) -> str:
    """Minúsculas, sin tildes y sin signos: 'Potosí ' -> 'potosi'."""
    if texto is None or (isinstance(texto, float) and pd.isna(texto)):
        return ""
    t = unicodedata.normalize("NFKD", str(texto))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalizar_telefono(telefono) -> str:
    """Deja solo dígitos y quita el prefijo de Bolivia (+591 / 00591)."""
    if telefono is None or (isinstance(telefono, float) and pd.isna(telefono)):
        return ""
    digitos = re.sub(r"\D", "", str(telefono))
    if digitos.startswith("00591"):
        digitos = digitos[5:]
    elif digitos.startswith("591") and len(digitos) > 8:
        digitos = digitos[3:]
    return digitos


# --------------------------------------------------------------------------
# Ciudades de Bolivia
# (ciudad canónica, ciudad donde está la oficina central del courier, alias)
# --------------------------------------------------------------------------
CIUDADES_BOLIVIA: list[tuple[str, str, list[str]]] = [
    ("La Paz", "La Paz", ["la paz"]),
    ("El Alto", "La Paz", ["el alto"]),
    ("Cochabamba", "Cochabamba", ["cochabamba", "cbba", "cbb", "cocha"]),
    ("Quillacollo", "Cochabamba", ["quillacollo"]),
    ("Sacaba", "Cochabamba", ["sacaba"]),
    ("Tiquipaya", "Cochabamba", ["tiquipaya"]),
    ("Punata", "Cochabamba", ["punata"]),
    ("Cliza", "Cochabamba", ["cliza"]),
    ("Santa Cruz", "Santa Cruz", ["santa cruz de la sierra", "santa cruz", "scz", "andres ibanez"]),
    ("Yapacaní", "Santa Cruz", ["yapacani"]),
    ("San Julián", "Santa Cruz", ["san julian"]),
    ("Montero", "Santa Cruz", ["montero"]),
    ("Warnes", "Santa Cruz", ["warnes"]),
    ("Cotoca", "Santa Cruz", ["cotoca"]),
    ("Mairana", "Santa Cruz", ["mairana"]),
    ("Camiri", "Santa Cruz", ["camiri"]),
    ("Sucre", "Sucre", ["sucre"]),
    ("Oruro", "Oruro", ["oruro"]),
    # "potos" cubre registros con tilde dañada ("Potos�") en pedidos.csv
    ("Potosí", "Potosí", ["potosi", "potos"]),
    ("Tupiza", "Potosí", ["tupiza"]),
    ("Villazón", "Potosí", ["villazon"]),
    ("Uyuni", "Potosí", ["uyuni"]),
    ("Llallagua", "Potosí", ["llallagua"]),
    ("Tarija", "Tarija", ["tarija"]),
    ("Yacuiba", "Tarija", ["yacuiba"]),
    ("Bermejo", "Tarija", ["bermejo"]),
    ("Villamontes", "Tarija", ["villamontes"]),
    ("Trinidad", "Trinidad", ["trinidad"]),
    ("Riberalta", "Trinidad", ["riberalta"]),
    ("Guayaramerín", "Trinidad", ["guayaramerin"]),
    ("Cobija", "Cobija", ["cobija"]),
]

_ALIAS_CIUDAD = sorted(
    ((alias, canon, oficina) for canon, oficina, aliases in CIUDADES_BOLIVIA for alias in aliases),
    key=lambda x: -len(x[0]),  # primero los alias más largos ("santa cruz de la sierra")
)


def detectar_ciudades(texto) -> list[dict]:
    """Todas las ciudades bolivianas mencionadas, en orden de aparición."""
    norm = f" {normalizar_texto(texto)} "
    encontradas: list[tuple[int, dict]] = []
    ocupado: list[tuple[int, int]] = []
    for alias, canon, oficina in _ALIAS_CIUDAD:
        for m in re.finditer(rf"(?<=\s){re.escape(alias)}(?=\s)", norm):
            if any(a <= m.start() < b for a, b in ocupado):
                continue
            ocupado.append((m.start(), m.end()))
            encontradas.append((m.start(), {"ciudad": canon, "oficina": oficina}))
    vistas, salida = set(), []
    for _, c in sorted(encontradas, key=lambda x: x[0]):
        if c["ciudad"] not in vistas:
            vistas.add(c["ciudad"])
            salida.append(c)
    return salida


def resolver_ciudad(texto) -> dict | None:
    """Primera ciudad reconocida en el texto o None."""
    ciudades = detectar_ciudades(texto)
    return ciudades[0] if ciudades else None


# --------------------------------------------------------------------------
# Lectura de CSV con caché (se invalida si el archivo cambia)
# --------------------------------------------------------------------------
@lru_cache(maxsize=16)
def _leer_csv_cache(ruta: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(ruta, encoding=ENCODING, dtype=str, keep_default_na=False)


def leer_csv(ruta: Path) -> pd.DataFrame | None:
    ruta = Path(ruta)
    if not ruta.exists():
        return None
    return _leer_csv_cache(str(ruta), ruta.stat().st_mtime).copy()


def estado_archivos() -> dict[str, bool]:
    return {p.name: p.exists() for p in (PEDIDOS_CSV, STOCK_CSV, COURIERS_CSV, DATASET_CSV)}


# --------------------------------------------------------------------------
# Pedidos / clientes
# --------------------------------------------------------------------------
_RE_ARTICULO = re.compile(r"ref(?:erencia)?\s*[:.#]?\s*(\d{4,8})\s*-\s*(\d{2})\s*$", re.IGNORECASE)


def parsear_articulo(nombre_articulo: str) -> dict:
    """'Jean Mom fit ... Ref: 240904 - 40' -> modelo, ref y talla."""
    texto = (nombre_articulo or "").strip()
    m = _RE_ARTICULO.search(texto)
    if not m:  # algunos artículos no traen "Ref:", solo " - 38" al final
        m2 = re.search(r"\s-\s*(3[4-9]|4[0-8])\s*$", texto)
        if m2:
            return {"modelo": texto[: m2.start()].strip(" |,-–"), "ref": None,
                    "talla": int(m2.group(1)), "articulo": texto}
        return {"modelo": texto, "ref": None, "talla": None, "articulo": texto}
    modelo = texto[: m.start()].strip(" |,-–�")
    return {"modelo": modelo, "ref": m.group(1), "talla": int(m.group(2)), "articulo": texto}


@lru_cache(maxsize=4)
def _pedidos_preparados(ruta: str, mtime: float) -> pd.DataFrame:
    df = _leer_csv_cache(ruta, mtime).copy()
    df["_tel"] = df[COL_TEL].map(normalizar_telefono)
    df["_fecha"] = pd.to_datetime(df[COL_FECHA], errors="coerce")
    df["_estado"] = df[COL_ESTADO].str.strip().str.lower()
    return df


def cargar_pedidos() -> pd.DataFrame | None:
    if not PEDIDOS_CSV.exists():
        return None
    return _pedidos_preparados(str(PEDIDOS_CSV), PEDIDOS_CSV.stat().st_mtime)


# Estados de WooCommerce (slug de la API) -> como aparecen en el export
ESTADOS_WC = {"pending": "Pendiente de pago", "processing": "Procesando", "on-hold": "En espera",
              "completed": "Completado", "cancelled": "Cancelado", "refunded": "Reembolsado",
              "failed": "Fallido", "checkout-draft": "Borrador"}


def _articulo_api(item: dict) -> dict:
    """Línea de un pedido de la API -> {modelo, ref, talla, articulo}."""
    art = parsear_articulo(item.get("name", ""))
    meta = " ".join(str(m.get("display_value") or m.get("value") or "")
                    for m in item.get("meta_data", []) or []
                    if "talla" in str(m.get("display_key") or m.get("key") or "").lower())
    art["ref"] = art["ref"] or _extraer_ref(item.get("sku"), item.get("name"))
    art["talla"] = art["talla"] or _extraer_talla(meta or None, item.get("sku"), item.get("name"))
    art["modelo"] = limpiar_nombre_modelo(art["modelo"]) or art["modelo"]
    return art


def _cliente_desde_api(tel: str, pedidos: list[dict]) -> dict:
    estado = lambda o: ESTADOS_WC.get(o.get("status"), str(o.get("status", "")).replace("-", " ").capitalize())
    compras = [o for o in pedidos if normalizar_texto(estado(o)) not in ESTADOS_NO_COMPRA]
    base = sorted(compras or pedidos, key=lambda o: o.get("date_created") or "", reverse=True)
    o = base[0]
    envio, fact = o.get("shipping") or {}, o.get("billing") or {}
    ciudad_raw = envio.get("city") or fact.get("city") or ""
    ciudad = resolver_ciudad(ciudad_raw)
    nombre = f"{envio.get('first_name') or fact.get('first_name', '')} " \
             f"{envio.get('last_name') or fact.get('last_name', '')}".strip()
    return {
        "encontrado": True,
        "fuente": "api",
        "telefono": tel,
        "nombre": nombre,
        "email": fact.get("email", ""),
        "direccion": " ".join(x for x in (envio.get("address_1"), envio.get("address_2")) if x),
        "ciudad": ciudad["ciudad"] if ciudad else ciudad_raw.strip(" _"),
        "ciudad_original": ciudad_raw,
        "num_compras": len({c.get("id") for c in compras}),
        "num_pedidos_total": len(pedidos),
        "cliente_frecuente": len(compras) >= MIN_COMPRAS_FRECUENTE,
        "ultimo_pedido": {
            "numero": o.get("number") or o.get("id"),
            "fecha": (o.get("date_created") or "").replace("T", " ")[:16],
            "estado": estado(o),
            "importe": o.get("total"),
            "articulos": [_articulo_api(i) for i in o.get("line_items", [])],
        },
    }


def buscar_cliente(telefono) -> dict:
    """Busca al cliente por teléfono: pedidos en vivo de la tienda (si hay claves) o pedidos.csv."""
    from . import tienda_api

    tel = normalizar_telefono(telefono)
    vacio = {"encontrado": False, "telefono": tel}
    pedidos_api = tienda_api.pedidos_por_telefono(tel) if len(tel) >= 7 else None
    if pedidos_api:
        return _cliente_desde_api(tel, pedidos_api)
    # Sin claves, sin conexión o sin coincidencias en la API: respaldo con pedidos.csv
    df = cargar_pedidos()
    if df is None or len(tel) < 7:
        return vacio

    sub = df[df["_tel"] == tel]
    if sub.empty:
        return vacio

    compras = sub[~sub["_estado"].isin(ESTADOS_NO_COMPRA)]
    num_compras = int(compras[COL_NUM].nunique())
    base = compras if not compras.empty else sub
    base = base.sort_values(["_fecha", COL_NUM], ascending=False)
    ultimo_num = base.iloc[0][COL_NUM]
    filas = base[base[COL_NUM] == ultimo_num]
    r0 = filas.iloc[0]

    ciudad_raw = r0[COL_CIUDAD]
    ciudad = resolver_ciudad(ciudad_raw)
    articulos = [parsear_articulo(a) for a in filas[COL_ARTICULO]]

    return {
        "encontrado": True,
        "fuente": "csv",
        "telefono": tel,
        "nombre": f"{r0[COL_NOMBRE]} {r0[COL_APELLIDO]}".strip(),
        "email": r0[COL_EMAIL],
        "direccion": r0[COL_DIRECCION],
        "ciudad": ciudad["ciudad"] if ciudad else ciudad_raw.strip(" _"),
        "ciudad_original": ciudad_raw,
        "num_compras": num_compras,
        "num_pedidos_total": int(sub[COL_NUM].nunique()),
        "cliente_frecuente": num_compras >= MIN_COMPRAS_FRECUENTE,
        "ultimo_pedido": {
            "numero": ultimo_num,
            "fecha": r0[COL_FECHA],
            "estado": r0[COL_ESTADO],
            "importe": r0[COL_IMPORTE],
            "articulos": articulos,
        },
    }


# --------------------------------------------------------------------------
# Stock
# --------------------------------------------------------------------------
# Nombres de columna aceptados (normalizados con _norm_col). Se usa la primera
# coincidencia exacta y, si no hay, la primera columna que empiece igual.
# Soporta el export de productos de WooCommerce (productos "variable" + "variation").
CANDIDATAS_STOCK = {
    "ref": ["ref", "referencia", "codigo_referencia", "codigo", "cod", "sku", "ref_modelo"],
    "talla": ["talla", "tallas", "talle", "size", "valor_es_del_atributo_1", "atributo_talla"],
    "cantidad": ["stock", "cantidad", "inventario", "stock_actual", "cantidad_disponible",
                 "unidades", "inventory", "qty"],
    "hay_stock": ["existencias", "existencia", "en_stock", "in_stock", "hay_stock"],
    "precio": ["precio", "precio_bs", "precio_normal", "precio_regular", "precio_venta",
               "regular_price", "price", "pvp"],
    "precio_oferta": ["precio_oferta", "precio_rebajado", "precio_descuento", "oferta", "sale_price"],
    "oferta_desde": ["dia_en_que_empieza_el_precio_rebajado", "sale_price_dates_from"],
    "oferta_hasta": ["dia_en_que_termina_el_precio_rebajado", "sale_price_dates_to"],
    "nombre": ["nombre", "modelo", "producto", "nombre_del_articulo", "articulo",
               "descripcion", "name"],
    "tipo": ["tipo", "type"],
    "publicado": ["publicado", "published"],
    "superior": ["superior", "parent", "parent_sku"],
    "categorias": ["categorias", "categor_as", "categories", "categor"],
    "etiquetas": ["etiquetas", "tags"],
}
TIPOS_PADRE = {"variable", "grouped", "agrupado"}  # filas resumen, no son stock real
NO_PUBLICADO = {"-1", "0", "no", "false", "borrador", "draft", "private", "privado"}


def _norm_col(c: str) -> str:
    return normalizar_texto(c).replace(" ", "_")


def detectar_columnas_stock(columnas) -> dict[str, str | None]:
    norm = {_norm_col(c): c for c in columnas}
    usadas: set[str] = set()
    res: dict[str, str | None] = {}
    for campo, candidatas in CANDIDATAS_STOCK.items():
        elegida = next((norm[c] for c in candidatas if c in norm and norm[c] not in usadas), None)
        if elegida is None:
            elegida = next(
                (orig for n, orig in norm.items() if orig not in usadas
                 and any(n.startswith(c) for c in candidatas)),
                None,
            )
        if elegida is not None:
            usadas.add(elegida)
        res[campo] = elegida
    return res


def _a_numero(valor) -> float | None:
    if valor is None:
        return None
    t = str(valor).strip().lower()
    if t in {"", "nan", "none"}:
        return None
    if t in {"si", "sí", "instock", "in stock", "disponible", "hay"}:
        return 1.0
    if t in {"no", "outofstock", "out of stock", "agotado", "sin stock"}:
        return 0.0
    t = re.sub(r"[^\d,.\-]", "", t)
    if t.count(",") == 1 and t.count(".") == 0:
        t = t.replace(",", ".")
    else:
        t = t.replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


def _oferta_vigente(desde, hasta, hoy: pd.Timestamp) -> bool:
    d = pd.to_datetime(desde, errors="coerce") if str(desde).strip() else None
    h = pd.to_datetime(hasta, errors="coerce") if str(hasta).strip() else None
    if d is not None and not pd.isna(d) and hoy < d:
        return False
    if h is not None and not pd.isna(h) and hoy > h:
        return False
    return True


def _extraer_ref(*valores) -> str | None:
    """Primer código de 5-8 dígitos: SKU 'MC240904-40' -> '240904', 'id:8258' se ignora."""
    for v in valores:
        texto = str(v or "")
        if not texto.strip() or texto.strip().lower().startswith("id:"):
            continue
        m = re.search(r"ref(?:erencia)?\s*[:.#]?\s*(\d{5,8})", texto, re.I) or \
            re.search(r"(?<!\d)(\d{5,8})(?!\d)", texto)
        if m:
            return m.group(1)
    return None


def _extraer_talla(v_talla, v_ref, v_nombre) -> int | None:
    if v_talla is not None:
        tallas = re.findall(r"(?<!\d)(3[4-9]|4[0-8])(?!\d)", str(v_talla))
        if len(tallas) == 1:  # "34, 36, 38" es una lista (fila padre), no una talla
            return int(tallas[0])
    # Sin columna de talla: "240904-40" o "Jean ... Ref: 240904 - 40"
    for v in (v_ref, v_nombre):
        m = re.search(r"[-/ ]\s*(3[4-9]|4[0-8])\s*$", str(v or "").strip())
        if m:
            return int(m.group(1))
    return None


def limpiar_nombre_modelo(nombre) -> str:
    """'Jean Mom fit, ... Ref: 240904 - 40' -> 'Jean Mom fit, ...'"""
    n = re.sub(r"\s*[-–]\s*(3[4-9]|4[0-8])\s*$", "", str(nombre or ""))
    n = re.sub(r"[\s|,–-]*ref(?:erencia)?\s*[:.#]?\s*\d{5,8}.*$", "", n, flags=re.I)
    return re.sub(r"\s+", " ", n).strip(" ,|–-")


@lru_cache(maxsize=4)
def _stock_preparado(ruta: str, mtime: float, dia: str) -> tuple[pd.DataFrame, dict]:
    df = _leer_csv_cache(ruta, mtime).copy()
    cols = detectar_columnas_stock(df.columns)

    def col(campo, defecto=""):
        return df[cols[campo]] if cols[campo] else pd.Series([defecto] * len(df), index=df.index)

    df["_ref"] = [_extraer_ref(r, n, s) for r, n, s in zip(col("ref"), col("nombre"), col("superior"))]
    # Las categorías ("Wide Leg", "Mom Fit"…) están en la fila del modelo, no en cada talla
    cats = (col("categorias").fillna("").astype(str) + ", " + col("etiquetas").fillna("").astype(str))
    if cols["tipo"]:
        es_padre = df[cols["tipo"]].map(normalizar_texto).isin(TIPOS_PADRE)
        cat_de_ref = {r: c for r, c, padre in zip(df["_ref"], cats, es_padre) if padre and r}
        df["_categorias"] = [c if c.strip(", ") else cat_de_ref.get(r, "") for r, c in zip(df["_ref"], cats)]
        df = df[~es_padre].copy()
    else:
        df["_categorias"] = cats
    df["_link"] = None
    df["_descripcion"] = ""

    nombre, ref_src = col("nombre"), col("ref")
    df["_talla"] = [_extraer_talla(t, r, n) for t, r, n in
                    zip(col("talla", None), ref_src, nombre)]
    # "Jean Mom fit, ... Ref: 240904 - 40" -> "Jean Mom fit, ..."
    df["_nombre"] = [limpiar_nombre_modelo(n) for n in nombre]
    df["_cantidad"] = col("cantidad", None).map(_a_numero)
    df["_hay_stock"] = col("hay_stock", None).map(_a_numero)
    df["_publicado"] = ~col("publicado").astype(str).str.strip().str.lower().isin(NO_PUBLICADO) \
        if cols["publicado"] else True
    df["_precio"] = col("precio", None).map(_a_numero)

    hoy = pd.Timestamp(dia)
    oferta = col("precio_oferta", None).map(_a_numero)
    vigente = [_oferta_vigente(d, h, hoy) for d, h in zip(col("oferta_desde"), col("oferta_hasta"))]
    df["_precio_oferta"] = [o if v and o is not None and not pd.isna(o) and o > 0 else None
                            for o, v in zip(oferta, vigente)]

    def disponible(fila) -> bool:
        if not fila["_publicado"]:
            return False
        c, h = fila["_cantidad"], fila["_hay_stock"]
        if h is not None and not pd.isna(h) and h <= 0:
            return False
        if c is not None and not pd.isna(c):
            return c > 0
        return True  # sin columna de cantidad: que la fila exista = disponible

    df["_disponible"] = df.apply(disponible, axis=1) if len(df) else []
    return df, cols


def _cargar_stock_csv() -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    if not STOCK_CSV.exists():
        return None, None
    # Hora de Bolivia (UTC-4); el parámetro por hora refresca la vigencia de las ofertas
    hoy = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(hours=4)
    return _stock_preparado(str(STOCK_CSV), STOCK_CSV.stat().st_mtime, hoy.strftime("%Y-%m-%d %H:00"))


_COLS_STOCK = ["_ref", "_talla", "_nombre", "_cantidad", "_hay_stock", "_publicado", "_precio",
               "_precio_oferta", "_disponible", "_categorias", "_link", "_descripcion"]


def cargar_stock() -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    """Stock de la tienda en vivo; si no hay conexión, stock.csv.

    Los modelos que están en stock.csv pero ya no aparecen en la tienda se agregan
    como NO publicados (así un cambio de un modelo antiguo dice "ya no está disponible").
    """
    from . import tienda_api

    df_csv, cols = _cargar_stock_csv()
    df_api = tienda_api.catalogo()
    if df_api is None or df_api.empty:
        if df_csv is None:
            return None, None
        return df_csv, {**cols, "fuente": "csv"}
    df = df_api
    if df_csv is not None:
        antiguos = df_csv[~df_csv["_ref"].isin(set(df_api["_ref"].dropna()))][_COLS_STOCK].copy()
        if not antiguos.empty:
            antiguos["_publicado"], antiguos["_disponible"] = False, False
            df = pd.concat([df_api, antiguos], ignore_index=True)
    return df, {"fuente": "api"}


def estado_stock() -> dict:
    """De dónde sale el stock ahora mismo (para mostrarlo en la app)."""
    from . import tienda_api

    df, cols = cargar_stock()
    est = tienda_api.estado()
    return {**est, "fuente": (cols or {}).get("fuente"), "csv_existe": STOCK_CSV.exists(),
            "modelos": int(df.loc[df["_publicado"], "_ref"].nunique()) if df is not None else 0}


def _num(v):
    return None if v is None or pd.isna(v) else float(v)


def consultar_stock(ref=None, talla=None) -> dict:
    """Disponibilidad y precio de un modelo (ref) y, opcionalmente, una talla."""
    talla = int(talla) if talla not in (None, "") else None
    ref = str(ref).strip() if ref not in (None, "") else None
    res = {
        "archivo_existe": True, "fuente": None, "ref": ref, "talla": talla,
        "modelo_encontrado": False, "publicado": None, "nombre": None, "precio": None,
        "precio_oferta": None, "precios_validos": [], "tallas_disponibles": [],
        "tallas_agotadas": [], "talla_disponible": None, "cantidad": None, "columnas": None,
    }
    df, cols = cargar_stock()
    res["archivo_existe"] = df is not None
    if df is None or ref is None:
        return res
    res["columnas"] = cols
    res["fuente"] = cols.get("fuente")

    sub = df[df["_ref"] == ref]
    if sub.empty:  # búsqueda por texto en nombre como respaldo
        sub = df[df["_nombre"].astype(str).str.contains(re.escape(ref), case=False, na=False)]
    if sub.empty:
        return res

    res["modelo_encontrado"] = True
    res["publicado"] = bool(sub["_publicado"].any())
    res["nombre"] = next((n for n in sub["_nombre"] if str(n).strip()), None)
    res["link"] = next((l for l in sub["_link"] if l), None) if "_link" in sub else None
    precios = [_num(p) for p in sub["_precio"] if _num(p) is not None]
    ofertas = [_num(p) for p in sub["_precio_oferta"] if _num(p) is not None]
    res["precio"] = precios[0] if precios else None
    res["precio_oferta"] = ofertas[0] if ofertas else None
    res["precios_validos"] = sorted(set(precios) | set(ofertas))

    con_talla = sub[sub["_talla"].notna()]
    disp = {int(t) for t, d in zip(con_talla["_talla"], con_talla["_disponible"]) if d}
    todas = {int(t) for t in con_talla["_talla"]}
    res["tallas_disponibles"] = sorted(disp)
    res["tallas_agotadas"] = sorted(todas - disp)

    if talla is not None:
        fila = con_talla[con_talla["_talla"] == talla]
        if fila.empty:
            res["talla_disponible"] = False
            res["cantidad"] = 0
        else:
            f = fila.iloc[0]
            res["cantidad"] = _num(f["_cantidad"])
            res["talla_disponible"] = bool(f["_disponible"])
            if _num(f["_precio"]) is not None:
                res["precio"] = _num(f["_precio"])
            if _num(f["_precio_oferta"]) is not None:
                res["precio_oferta"] = _num(f["_precio_oferta"])
    return res


# --------------------------------------------------------------------------
# Búsqueda por estilo ("¿tienes wide leg?")
# --------------------------------------------------------------------------
# (nombre, cómo lo dice la clienta, cómo aparece en el nombre/categorías de la tienda)
ESTILOS = [
    ("Wide Leg", ["wide leg", "wide", "wideleg", "pierna ancha", "bota ancha", "ancho", "anchos",
                  "ancha", "anchas"], ["wide leg", "wide"]),
    ("Mom Fit", ["mom", "mom fit", "momfit", "mom jeans"], ["mom"]),
    ("Straight Leg", ["straight", "straight leg", "recto", "rectos", "bota recta", "pierna recta"],
     ["straight", "bota recta", "pierna recta"]),
    ("Skinny", ["skinny", "pitillo", "chupin", "chupines"], ["skinny"]),
    ("Slim Fit", ["slim", "slim fit"], ["slim"]),
    ("Flare", ["flare", "campana", "acampanado", "acampanados", "oxford"],
     ["flare", "campana", "acampanado"]),
    ("Bootcut", ["bootcut", "boot cut"], ["bootcut"]),
    ("Palazzo", ["palazzo"], ["palazzo"]),
    ("Cargo", ["cargo", "cargos"], ["cargo"]),
    ("Barrel", ["barrel"], ["barrel"]),
]


# Características que se pueden pedir junto al estilo: "wide leg de tiro alto en negro".
# (grupo, etiqueta, cómo lo dice la clienta, cómo aparece en el nombre del modelo)
CARACTERISTICAS = [
    ("tiro", "tiro alto", ["tiro alto", "cintura alta", "talle alto"], ["tiro alto"]),
    ("tiro", "tiro medio", ["tiro medio", "cintura media"], ["tiro medio"]),
    ("tiro", "tiro bajo", ["tiro bajo", "cintura baja", "talle bajo"], ["tiro bajo"]),
    ("color", "negro", ["negro", "negros", "negra", "black"], ["negro"]),
    ("color", "celeste", ["celeste", "celestes", "azul claro", "claritos", "clarito", "color claro",
                          "tono claro", "colores claros"],
     ["celeste", "azul claro", "hielo"]),
    ("color", "azul oscuro", ["azul oscuro", "color oscuro", "tono oscuro", "oscuritos", "indigo",
                              "marino", "marengo"],
     ["azul oscuro", "indigo", "marengo", "medio oscuro"]),
    ("color", "plomo", ["plomo", "gris", "plomito"], ["plomo", "gris"]),
    ("color", "hueso", ["hueso", "beige", "crema", "blanco"], ["hueso"]),
    ("tela", "stretch", ["stretch", "licrado", "licra", "elastizado", "elasticado", "que estire",
                         "que estira", "estira", "elongacion"], ["stretch", "licrado"]),
    ("tela", "rígido", ["rigido", "rigida", "sin elastizar", "que no estire", "no estira"], ["rigido"]),
    ("tela", "tela liviana", ["liviana", "liviano", "delgada", "delgadito", "fresco", "fresquito"],
     ["liviana", "ligero"]),
    ("detalle", "pretina elástica", ["pretina elastica", "elastico en la cintura", "cintura elastica"],
     ["pretina elastica", "ajuste elastico"]),
    ("detalle", "con botones", ["botones", "con boton"], ["botones", "boton"]),
    ("detalle", "con encaje", ["encaje"], ["encaje"]),
]


def _hay(texto_norm: str, frase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(frase)}(?![a-z0-9])", texto_norm) is not None


def detectar_estilos(texto) -> list[str]:
    """'tienes jeans wide leg o mom?' -> ['Wide Leg', 'Mom Fit']"""
    n = normalizar_texto(texto).replace("-", " ")
    return [nombre for nombre, alias, _ in ESTILOS if any(_hay(n, a) for a in alias)]


def detectar_caracteristicas(texto) -> list[str]:
    """'jean de tiro alto que no estire' -> ['tiro alto', 'rígido']"""
    n = normalizar_texto(texto).replace("-", " ")
    encontradas = []
    for grupo, etiqueta, alias, _ in CARACTERISTICAS:
        if any(_hay(n, a) for a in alias) and etiqueta not in encontradas:
            # "que no estire" no es "que estire"
            if etiqueta == "stretch" and any(_hay(n, a) for a in ("que no estire", "no estira")):
                continue
            encontradas.append(etiqueta)
    return encontradas


def _nombre_corto(nombre: str, max_chars: int = 60) -> str:
    partes = [p.strip() for p in str(nombre).split(",") if p.strip()]
    corto = ", ".join(partes[:2]) if partes else str(nombre)
    if len(corto) <= max_chars:
        return corto
    return corto[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:–-|") + "…"


def buscar_modelos_por_estilo(estilos: list[str], tallas: list[int] | None = None,
                              max_modelos: int | None = 5,
                              caracteristicas: list[str] | None = None,
                              todos: bool = False) -> dict:
    """Modelos publicados con stock de ese estilo y/o características (y en esas tallas).

    Dentro de un grupo basta una coincidencia (negro O celeste); entre grupos deben
    cumplirse todas (wide leg Y tiro alto Y negro). max_modelos=None -> todos.
    """
    from . import tienda_api

    caracteristicas = list(caracteristicas or [])
    res = {"estilos": list(estilos), "caracteristicas": caracteristicas,
           "tallas": list(tallas or []), "modelos": [], "total": 0, "sin_talla": [], "link": None}
    df, _ = cargar_stock()
    if df is None or not (estilos or caracteristicas or todos):
        return res
    terminos = [t for nombre, _, ts in ESTILOS if nombre in estilos for t in ts]
    grupos: dict[str, list[str]] = {}
    for grupo, etiqueta, _, ts in CARACTERISTICAS:
        if etiqueta in caracteristicas:
            grupos.setdefault(grupo, []).extend(ts)
    pub = df[df["_publicado"] & df["_ref"].notna()]
    for ref, sub in pub.groupby("_ref", sort=False):
        nombre = next((n for n in sub["_nombre"] if str(n).strip()), "")
        texto = normalizar_texto(f"{nombre} {' '.join(map(str, sub['_categorias'].fillna('')))}")
        texto = texto.replace("-", " ")
        if terminos and not any(_hay(texto, t) for t in terminos):
            continue
        if not all(any(_hay(texto, t) for t in ts) for ts in grupos.values()):
            continue
        disp = sorted({int(t) for t, d in zip(sub["_talla"], sub["_disponible"]) if d and t is not None
                       and not pd.isna(t)})
        if not disp:
            continue
        precios = [_num(p) for p in sub["_precio"] if _num(p) is not None]
        ofertas = [_num(p) for p in sub["_precio_oferta"] if _num(p) is not None]
        modelo = {"ref": ref, "nombre": _nombre_corto(nombre), "tallas_disponibles": disp,
                  "precio": precios[0] if precios else None,
                  "precio_oferta": min(ofertas) if ofertas else None,
                  "link": next((l for l in sub["_link"] if l), None),
                  "nombre_completo": str(nombre),
                  "descripcion": next((str(d) for d in sub.get("_descripcion", []) if d and str(d) != "nan"), "")}
        if tallas and not set(tallas) & set(disp):
            res["sin_talla"].append(modelo)  # es del estilo, pero no en la talla pedida
            continue
        res["modelos"].append(modelo)
    res["total"] = len(res["modelos"])
    # Primero los modelos más nuevos (Ref más alta)
    res["modelos"] = sorted(res["modelos"], key=lambda m: m["ref"], reverse=True)[:max_modelos]
    res["sin_talla"] = sorted(res["sin_talla"], key=lambda m: m["ref"], reverse=True)
    res["link"] = next((tienda_api.link_categoria(e) for e in estilos if tienda_api.link_categoria(e)),
                       None)
    return res


# --------------------------------------------------------------------------
# Medidas ("mi cintura es 78 cm, ¿qué talla soy?")
# --------------------------------------------------------------------------
# Rangos razonables (cm) para no confundir una medida con otra cosa
_RANGO_MEDIDA = {"Cintura": (50, 130), "Cadera": (70, 150), "Largo": (80, 125), "Botapié": (10, 40)}
_NOMBRE_MEDIDA = {"cintura": "Cintura", "cadera": "Cadera", "largo": "Largo",
                  "botapie": "Botapié", "bota": "Botapié", "ruedo": "Botapié"}
_PAL_MEDIDA = r"(cintura|cadera|largo|botapie|ruedo)"
_NUM_MEDIDA = r"(\d{2,3}(?:[.,]\d)?)"
_RE_MEDIDA_NUM_PRIMERO = re.compile(_NUM_MEDIDA + r"\s*(?:cm|centimetros?)?\s*(?:de\s+)?" + _PAL_MEDIDA)
_RE_MEDIDA_PAL_PRIMERO = re.compile(
    _PAL_MEDIDA + r"(?:\s*(?:es|mide|mido|son|de|tengo|seria|aprox\w*|como|unos|mas o menos|[:=]))*\s*"
    + _NUM_MEDIDA + r"(?!\s*(?:bs|anos))")


def _sin_tildes(texto) -> str:
    t = unicodedata.normalize("NFKD", str(texto or "").lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def detectar_medidas_cliente(texto) -> dict[str, float]:
    """'mi cintura es de 78 cm y 98 de cadera' -> {'Cintura': 78.0, 'Cadera': 98.0}"""
    t = _sin_tildes(texto)
    salida: dict[str, float] = {}
    usados: list[tuple[int, int]] = []
    for rx, i_num, i_pal in ((_RE_MEDIDA_NUM_PRIMERO, 1, 2), (_RE_MEDIDA_PAL_PRIMERO, 2, 1)):
        for m in rx.finditer(t):
            if any(a < m.end() and m.start() < b for a, b in usados):
                continue
            nombre = _NOMBRE_MEDIDA[m.group(i_pal)]
            valor = float(m.group(i_num).replace(",", "."))
            lo, hi = _RANGO_MEDIDA[nombre]
            if lo <= valor <= hi and nombre not in salida:
                salida[nombre] = valor
                usados.append((m.start(), m.end()))
    return salida


def talla_por_medidas(tabla: dict[int, dict] | None, medidas_cliente: dict[str, float],
                      disponibles: list[int] | None = None) -> dict | None:
    """Talla más chica cuya prenda mide al menos lo que indicó la clienta (cintura/cadera).

    La tabla de la tienda son medidas de la PRENDA; se busca la primera talla que no le
    quede chica. Devuelve {'talla', 'fila', 'disponible', 'alternativa', 'medidas_usadas'}
    o None si no se puede calcular (sin tabla o sin cintura/cadera en común).
    """
    if not tabla or not medidas_cliente:
        return None
    usadas = [m for m in ("Cintura", "Cadera") if m in medidas_cliente
              and all(m in fila for fila in tabla.values())]
    if not usadas:
        return None
    disponibles = list(disponibles or [])
    tallas = sorted(tabla)
    talla = next((t for t in tallas if all(tabla[t][m] >= medidas_cliente[m] for m in usadas)), None)
    if talla is None:
        return {"talla": None, "fila": None, "disponible": False, "alternativa": None,
                "medidas_usadas": usadas, "fuera_de_rango": True}
    alternativa = None
    if talla not in disponibles:  # la siguiente talla con stock que tampoco quede chica
        alternativa = next((t for t in tallas if t > talla and t in disponibles), None)
    return {"talla": talla, "fila": tabla[talla], "disponible": talla in disponibles,
            "alternativa": alternativa, "medidas_usadas": usadas, "fuera_de_rango": False}


def texto_medidas(fila: dict) -> str:
    """{'Cintura': 77.5, 'Largo': 105} -> 'cintura 77,5 · largo 105 cm'"""
    partes = [f"{k.lower()} {v:g}".replace(".", ",") for k, v in fila.items()]
    return " · ".join(partes) + " cm" if partes else ""


# --------------------------------------------------------------------------
# Couriers
# --------------------------------------------------------------------------
def buscar_courier(ciudad) -> dict | None:
    """Oficina central del courier para la ciudad (o su capital de departamento)."""
    if not ciudad:
        return None
    df = leer_csv(COURIERS_CSV)
    if df is None:
        return None
    info = resolver_ciudad(ciudad)
    oficina = info["oficina"] if info else str(ciudad)
    df["_n"] = df["ciudad"].map(normalizar_texto)
    fila = df[df["_n"] == normalizar_texto(oficina)]
    if fila.empty:
        return None
    datos = {c: fila.iloc[0].get(c, "") for c in COURIER_COLS}
    datos["ciudad_cliente"] = info["ciudad"] if info else str(ciudad)
    datos["es_oficina_de_otra_ciudad"] = normalizar_texto(datos["ciudad_cliente"]) != fila.iloc[0]["_n"]
    return datos
