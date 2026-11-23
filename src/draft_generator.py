"""Generador de borradores de Hilo.

1. Reúne los datos reales (pedidos.csv, stock en vivo de la tienda, couriers.csv).
2. Busca en dataset.csv las 3 respuestas reales de Merci más parecidas
   (sentence-transformers) como referencia de tono.
3. Arma el borrador con las plantillas oficiales rellenadas con datos reales.
4. Verifica de forma determinista que tallas y precios del borrador coincidan
   con el stock (alerta verde / roja / amarilla).
"""
from __future__ import annotations

import difflib
import re
from datetime import date, datetime, timedelta, timezone

from .data_loader import (
    DATASET_CSV,
    buscar_cliente,
    buscar_courier,
    buscar_modelos_por_estilo,
    cargar_stock,
    consultar_stock,
    leer_csv,
    normalizar_texto,
    talla_por_medidas,
    texto_medidas,
)
from . import tienda_api

CATALOGO_URL = "https://merci-collection.com/product-category/ver-todos/"
GUIA_TALLAS_URL = "https://merci-collection.com/guia-de-tallas/"
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TZ_BOLIVIA = timezone(timedelta(hours=-4))  # Bolivia no usa horario de verano

# --------------------------------------------------------------------------
# Plantillas oficiales de Merci (edítalas aquí si cambia el texto oficial)
# --------------------------------------------------------------------------
PLANTILLAS = {
    "saludo": "{saludo}! Gracias por comunicarte con Merci Collection! 💙👖",
    "catalogo": f"Te comparto nuestro catálogo completo 👉 {CATALOGO_URL}",
    # Respuesta oficial a "¿tienen tienda física?" / "¿tienen tienda en Cochabamba?" (2 mensajes)
    "venta_online": (
        "Nuestra venta es 100% online 😊 No tenemos un local para visitar, pero enviamos a "
        "cualquier ciudad de Bolivia con envío gratis."
    ),
    "cambio_sin_costo": (
        "Si te preocupa la talla, no hay problema: el primer cambio de talla o de modelo no "
        "tiene costo de envío 💕."
    ),
    "cambio_gratis": "Si la talla que pides no te queda bien, te hacemos un cambio de talla gratuito! 💗🎉",
    # Texto de la guía de tallas de la web (merci-collection.com/guia-de-tallas/)
    "guia_medidas": ("Cada modelo tiene su propia tabla de medidas 📏 ¿Me pasas el código o la foto "
                     "del modelo que te gustó? Te paso sus medidas, o si me dices tu medida de cintura "
                     f"te digo qué talla pedir 🤗 Guía de tallas 👉 {GUIA_TALLAS_URL}"),
    "datos_pedido": (
        "Para realizar tu pedido, ayúdame con estos datos 📝\n"
        "1️⃣ Nombre completo\n"
        "2️⃣ Dirección de entrega 🏠\n"
        "3️⃣ Ciudad 📍\n"
        "4️⃣ Email 📧\n"
        "5️⃣ Talla 📏\n"
        "6️⃣ Modelo (código o foto) 👖"
    ),
    "envio_plazos": (
        "El envío es gratis 🚚 y tarda de 2 a 3 días hábiles en llegar{destino}; "
        "el courier no trabaja fines de semana.{estimado} La entrega es a domicilio y, si "
        "prefieres, también puedes recogerlo de la oficina del courier."
    ),
    # [PLANTILLA_CAMBIO] del dataset: reemplaza por el texto oficial si es distinto
    "cambio_instrucciones": (
        "Para procesar el cambio:\n"
        "1️⃣ El jean debe estar sin uso y con sus etiquetas.\n"
        "2️⃣ Confírmame la talla nueva y tu dirección de entrega.\n"
        "3️⃣ Coordinamos el recojo con el courier y te enviamos la nueva talla.\n"
        "El primer cambio de talla es gratis 💕 (a partir del segundo cambio tiene costo de logística)."
    ),
    "confirmacion_qr": "Si todo está correcto, te envío el QR para realizar el pago 📲 y despachamos tu pedido 🚚",
}

CAMPOS_PEDIDO = ["Nombre", "Dirección", "Ciudad", "Email", "Talla", "Modelo"]
ICONOS_CAMPO = {"Nombre": "👤", "Dirección": "🏠", "Ciudad": "📍", "Email": "📧",
                "Talla": "📏", "Modelo": "👖"}
DIAS_SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def _ahora() -> datetime:
    return datetime.now(TZ_BOLIVIA)


def saludo_por_hora(ahora: datetime | None = None) -> str:
    h = (ahora or _ahora()).hour
    if 5 <= h < 12:
        return "Buenos días"
    if 12 <= h < 19:
        return "Buenas tardes"
    return "Buenas noches"


def sumar_dias_habiles(inicio: date, n: int) -> date:
    d = inicio
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def estimado_entrega(hoy: date | None = None) -> str:
    hoy = hoy or _ahora().date()
    a, b = sumar_dias_habiles(hoy, 2), sumar_dias_habiles(hoy, 3)
    return f" Si se despacha hoy, estimo que llegaría entre el {DIAS_SEMANA[a.weekday()]} " \
           f"y el {DIAS_SEMANA[b.weekday()]}."


def _fmt_precio(p) -> str:
    if p is None:
        return "?"
    return f"{int(p)}" if float(p).is_integer() else f"{p:.2f}"


def _fmt_lista(nums) -> str:
    nums = [str(n) for n in nums]
    if len(nums) <= 1:
        return "".join(nums)
    return ", ".join(nums[:-1]) + " y " + nums[-1]


def _tallas(nums) -> str:
    """'talla 42' o 'tallas 36, 40 y 42'."""
    nums = list(nums)
    return ("tallas " if len(nums) > 1 else "talla ") + _fmt_lista(nums)


# Palabras con mayúscula que son parte del nombre del modelo, no el inicio de una frase
_PALABRAS_TITULO = {"jean", "jeans", "straight", "wide", "leg", "mom", "fit", "slim", "skinny",
                    "flare", "cargo", "barrel", "bootcut", "palazzo", "tiro", "alto", "medio",
                    "bajo", "pretina", "bota", "recta", "color", "azul", "denim", "vintage"}


def _precio_modelo(m: dict) -> str:
    """'650 Bs' o '590 Bs → 531 Bs 🎉' (oferta)."""
    if m.get("precio") is None:
        return ""
    if m.get("precio_oferta") and m["precio_oferta"] < m["precio"]:
        return f"{_fmt_precio(m['precio'])} Bs → {_fmt_precio(m['precio_oferta'])} Bs 🎉"
    return f"{_fmt_precio(m['precio'])} Bs"


# --------------------------------------------------------------------------
# Extracción de datos de envío del mensaje
# --------------------------------------------------------------------------
_RE_ITEM_LISTA = re.compile(r"(?:(?<=\s)|^)([1-6])(?:️?⃣|\s*[.)\-]+)\s*")
_RE_ETIQUETA = re.compile(
    r"^\s*(nombre(\s+completo)?|direcci[oó]n(\s+de\s+(entrega|env[ií]o))?|ciudad|"
    r"e-?mail|correo(\s+electr[oó]nico)?|talla|modelo)\s*[:\-]?\s*",
    re.IGNORECASE,
)
_RE_SIMBOLOS = re.compile(r"[\U0001F000-\U0001FAFF☀-➿️⃣]")


def extraer_datos_envio(mensaje: str, entidades: dict) -> dict[str, str | None]:
    """Ordena los 6 datos de envío (Nombre, Dirección, Ciudad, Email, Talla, Modelo)."""
    datos: dict[str, str | None] = {c: None for c in CAMPOS_PEDIDO}
    texto = mensaje or ""

    partes = _RE_ITEM_LISTA.split(texto)
    # partes = [antes, num1, valor1, num2, valor2, ...]
    items = [(int(partes[i]), partes[i + 1]) for i in range(1, len(partes) - 1, 2)]
    if len(items) >= 2:
        for num, valor in items:
            limpio = _RE_SIMBOLOS.sub("", valor).strip()
            etiqueta = _RE_ETIQUETA.match(limpio)
            campo = CAMPOS_PEDIDO[num - 1]
            if etiqueta:
                e = normalizar_texto(etiqueta.group(1))
                campo = next((c for c in CAMPOS_PEDIDO if normalizar_texto(c) in e), campo)
                if e.startswith("correo"):
                    campo = "Email"
                limpio = limpio[etiqueta.end():]
            limpio = limpio.strip(" :-,;")
            if limpio and not datos[campo]:
                datos[campo] = limpio

    if not datos["Email"] and entidades.get("emails"):
        datos["Email"] = entidades["emails"][0]
    if not datos["Ciudad"] and entidades.get("ciudades"):
        datos["Ciudad"] = entidades["ciudades"][0]
    if not datos["Talla"] and entidades.get("tallas"):
        datos["Talla"] = str(entidades["tallas"][-1])
    if not datos["Modelo"] and entidades.get("referencias"):
        datos["Modelo"] = f"Ref: {entidades['referencias'][0]}"
    if not datos["Nombre"]:
        m = re.search(r"(?:me llamo|mi nombre es|nombre\s*:)\s*([A-Za-zÁÉÍÓÚÑáéíóúñ ]{3,60})", texto, re.I)
        if m:
            datos["Nombre"] = m.group(1).strip()
    if not datos["Dirección"]:
        m = re.search(r"((?:calle|av\.?|avenida|zona|barrio|urb\.?|condominio|edificio)\b[^\n@]{3,120})",
                      texto, re.I)
        if m:
            datos["Dirección"] = m.group(1).strip(" ,;")

    # Normaliza la talla a número si es posible
    if datos["Talla"]:
        m = re.search(r"(?<!\d)(3[4-9]|4[0-8])(?!\d)", datos["Talla"])
        if m:
            datos["Talla"] = m.group(1)
    return datos


# --------------------------------------------------------------------------
# Recolección de datos reales
# --------------------------------------------------------------------------
def recopilar_datos(analisis: dict, telefono: str, ref_manual: str | None = None) -> dict:
    """Busca en los CSV todo lo necesario para responder."""
    ent = analisis["entidades"]
    intenciones = set(analisis["intenciones"])
    cliente = buscar_cliente(telefono)

    articulo_previo = None
    if cliente.get("encontrado"):
        arts = [a for a in cliente["ultimo_pedido"]["articulos"] if a.get("ref")]
        # Si el mensaje menciona una Ref del pedido, ese es el artículo en cuestión
        mencionados = [a for a in arts if a["ref"] in ent["referencias"]]
        articulo_previo = (mencionados or arts or [None])[0]

    # Modelo de referencia: manual > mencionado > último comprado (solo para cambios)
    ref, origen_ref = None, None
    if ref_manual and ref_manual.strip():
        m = re.search(r"\d{4,8}", ref_manual)
        ref, origen_ref = (m.group(0) if m else ref_manual.strip()), "manual"
    elif ent["referencias"]:
        ref, origen_ref = ent["referencias"][0], "mensaje"
    elif "cambio_talla" in intenciones and articulo_previo:
        ref, origen_ref = articulo_previo["ref"], "ultimo_pedido"

    # Tallas a consultar
    tallas = list(ent["tallas"])
    talla_sugerida = None
    if "cambio_talla" in intenciones and articulo_previo and articulo_previo.get("talla"):
        anterior = articulo_previo["talla"]
        nuevas = [t for t in tallas if t != anterior]
        if nuevas:
            tallas = nuevas
        elif ent.get("ajuste_talla"):
            talla_sugerida = anterior - 2 if ent["ajuste_talla"] == "achicar" else anterior + 2
            tallas = [talla_sugerida]

    stock_modelo = consultar_stock(ref) if ref else consultar_stock(None)
    stock_tallas = {t: consultar_stock(ref, t) for t in tallas} if ref else {}

    ciudad, origen_ciudad = None, None
    if ent["ciudades"]:
        ciudad, origen_ciudad = ent["ciudades"][0], "mensaje"
    elif cliente.get("encontrado"):
        ciudad, origen_ciudad = cliente["ciudad"], "pedidos.csv"

    courier = buscar_courier(ciudad) if analisis.get("alerta_courier") and ciudad else None

    # "¿tienes wide leg?" sin un modelo concreto: buscar en la tienda los modelos de ese estilo
    # "¿Qué me recomiendas?" / "mi cintura es 78, ¿qué me sirve?" sin un modelo concreto:
    # se recomienda en generar() (usa MiniLM para comparar con las descripciones)
    medidas_cliente = ent.get("medidas_cliente") or {}
    recomendar = None
    if not ref and not ent.get("cambio_explicito") and (
            ent.get("pide_recomendacion") or medidas_cliente):
        recomendar = {"estilos": ent.get("estilos", []), "caracteristicas": ent.get("caracteristicas", []),
                      "tallas": tallas, "medidas_cliente": medidas_cliente}

    # "¿Qué medidas tiene el 240904?" / "mi cintura es 78, ¿qué talla pido?" con un modelo
    medidas = None
    if ref and (ent.get("pide_medidas") or ent.get("pide_recomendacion") or medidas_cliente):
        tabla = tienda_api.medidas_de(ref)
        disp = stock_modelo.get("tallas_disponibles") or []
        medidas = {"ref": ref, "tabla": tabla, "cliente": medidas_cliente,
                   "tallas": [t for t in tallas if t in disp] if tallas else disp,
                   "talla_por_medidas": talla_por_medidas(tabla, medidas_cliente, disp)}

    # "¿Tienen tabla de medidas?" sin modelo ni medidas: cada modelo tiene la suya
    guia_medidas = bool(ent.get("pide_medidas") and not ref and not recomendar
                        and not ent.get("cambio_explicito"))

    modelos_estilo = None
    if (ent.get("estilos") or ent.get("caracteristicas")) and not ref and not recomendar \
            and "cambio_talla" not in intenciones:
        # Con talla: TODOS los modelos que la tienen; sin talla: los 5 más nuevos
        modelos_estilo = buscar_modelos_por_estilo(
            ent.get("estilos", []), tallas, max_modelos=None if tallas else 5,
            caracteristicas=ent.get("caracteristicas", []))

    return {
        "cliente": cliente,
        "articulo_previo": articulo_previo,
        "ref": ref,
        "origen_ref": origen_ref,
        "tallas_consultadas": tallas,
        "talla_sugerida": talla_sugerida,
        "stock_modelo": stock_modelo,
        "stock_tallas": stock_tallas,
        "ciudad": ciudad,
        "origen_ciudad": origen_ciudad,
        "courier": courier,
        "modelos_estilo": modelos_estilo,
        "recomendar": recomendar,
        "recomendacion": None,   # lo llena DraftGenerator.generar()
        "medidas": medidas,
        "guia_medidas": guia_medidas,
    }


# --------------------------------------------------------------------------
# Verificación determinista contra stock.csv
# --------------------------------------------------------------------------
_RE_TALLAS_BORRADOR = re.compile(
    r"\btallas?\b[^0-9\n]{0,15}((?:(?<!\d)(?:3[4-9]|4[0-8])(?!\d)(?:\s*(?:,|y|/|-)\s*)?)+)",
    re.IGNORECASE,
)
_RE_PRECIO_1 = re.compile(r"(?<![\d.,])(\d{2,4}(?:[.,]\d{1,2})?)\s*(?:bs\b\.?|bolivianos)", re.IGNORECASE)
_RE_PRECIO_2 = re.compile(r"\bbs\.?\s*(\d{2,4}(?:[.,]\d{1,2})?)", re.IGNORECASE)
_RE_REF_BORRADOR = re.compile(r"\bref(?:erencia)?\.?\s*:?\s*(\d{5,8})", re.IGNORECASE)
_NEGACIONES = ["no tenemos", "no hay", "agotad", "sin stock", "no esta disponible",
               "no contamos", "ya no tenemos", "no la tenemos", "no lo tenemos"]
_OMITIR = ["compraste", "compro", "talla actual", "tu pedido anterior", "pediste", "llevaste"]


def _oraciones(texto: str) -> list[str]:
    # Corta en saltos de línea, !?¡¿ y en "punto/emoji + espacio + Mayúscula"
    # (no corta en "Ref: 240904" porque ':' y ',' no cierran oración).
    partes = re.split(r"[\n!?¡¿;]+|(?<=[^\w\s,;:(])\s+(?=[A-ZÁÉÍÓÚÑ])"
                      r"|,?\s+(?:pero|aunque|sin embargo)\s+", texto or "", flags=re.I)
    return [p for p in partes if p.strip()]


def verificar_borrador(texto: str, ref_por_defecto: str | None = None) -> dict:
    """Comprueba que tallas y precios del borrador coincidan con stock.csv.

    estado: "verde" (todo coincide), "rojo" (discrepancia) o "amarillo" (no verificable).
    """
    detalles: list[str] = []
    # (talla, es_negacion, refs de su línea) y (precio, refs de su línea): en una lista de
    # modelos cada línea se verifica contra SU Ref, no contra todas
    tallas_mencionadas: list[tuple[int, bool, tuple]] = []
    precios: list[tuple[float, tuple]] = []

    for linea in (texto or "").split("\n"):
        refs_linea = tuple(dict.fromkeys(_RE_REF_BORRADOR.findall(linea)))
        for oracion in _oraciones(linea):
            n = normalizar_texto(oracion)
            if any(o in n for o in _OMITIR):
                continue
            negada = any(neg in n for neg in _NEGACIONES)
            for m in _RE_TALLAS_BORRADOR.finditer(oracion):
                for t in re.findall(r"(?<!\d)(3[4-9]|4[0-8])(?!\d)", m.group(1)):
                    tallas_mencionadas.append((int(t), negada, refs_linea))
            for rx in (_RE_PRECIO_1, _RE_PRECIO_2):
                for p in rx.findall(oracion):
                    precios.append((float(p.replace(",", ".")), refs_linea))

    if not tallas_mencionadas and not precios:
        return {"estado": "verde", "detalles": ["El borrador no menciona tallas ni precios concretos."]}
    if cargar_stock()[0] is None:
        return {"estado": "amarillo",
                "detalles": ["Sin conexión con la tienda ni stock.csv: no se pueden verificar tallas ni precios."]}

    refs = list(dict.fromkeys(_RE_REF_BORRADOR.findall(texto or "")))
    if not refs and ref_por_defecto:
        refs = [ref_por_defecto]
    if not refs:
        return {"estado": "amarillo",
                "detalles": ["El borrador menciona talla/precio pero no se identificó el modelo (Ref)."]}

    rojo = False
    infos: dict[str, dict] = {}
    for r in dict.fromkeys(list(refs) + [r for *_, rs in tallas_mencionadas for r in rs]
                           + [r for _, rs in precios for r in rs]):
        infos[r] = consultar_stock(r)
        if not infos[r]["modelo_encontrado"]:
            rojo = True
            detalles.append(f"❌ La Ref {r} no existe en la tienda.")

    def stock_de(refs_mencion: tuple) -> tuple[set[int], set[float]]:
        disp, precs = set(), set()
        for r in (refs_mencion or refs):
            if infos[r]["modelo_encontrado"]:
                disp |= set(infos[r]["tallas_disponibles"])
                precs |= set(infos[r].get("precios_validos") or [])
        return disp, precs

    varias = len(infos) > 1

    def de_ref(refs_mencion: tuple) -> str:
        return f" (Ref {', '.join(refs_mencion)})" if varias and refs_mencion else ""

    for talla, negada, rs in dict.fromkeys(tallas_mencionadas):
        hay = talla in stock_de(rs)[0]
        if negada and hay:
            rojo = True
            detalles.append(f"❌ El borrador dice que la talla {talla}{de_ref(rs)} no está, pero sí hay stock.")
        elif not negada and not hay:
            rojo = True
            detalles.append(f"❌ Talla {talla}{de_ref(rs)} mencionada como disponible, pero NO tiene stock.")
        elif not varias or not rs:
            detalles.append(f"✅ Talla {talla}: coincide con el stock.")
    if varias and not rojo and tallas_mencionadas:
        detalles.append(f"✅ Tallas de los {len(infos)} modelos: coinciden con el stock.")

    precios_ok = 0
    for p, rs in dict.fromkeys(precios):
        validos_ref = stock_de(rs)[1]
        if not validos_ref:
            detalles.append(f"⚠️ Precio {_fmt_precio(p)} Bs{de_ref(rs)}: el stock no tiene precio.")
        elif any(abs(p - v) < 0.01 for v in validos_ref):
            precios_ok += 1
            if not varias or not rs:
                detalles.append(f"✅ Precio {_fmt_precio(p)} Bs: coincide con el stock.")
        else:
            rojo = True
            validos = ", ".join(_fmt_precio(v) for v in sorted(validos_ref))
            detalles.append(f"❌ Precio {_fmt_precio(p)} Bs{de_ref(rs)} no coincide con el stock ({validos} Bs).")
    if varias and precios_ok and not any("Precio" in d and "❌" in d for d in detalles):
        detalles.append("✅ Precios de cada modelo: coinciden con el stock.")

    if rojo:
        estado = "rojo"
    elif any(d.startswith("⚠️") for d in detalles):
        estado = "amarillo"
    else:
        estado = "verde"
    return {"estado": estado, "detalles": detalles}


# --------------------------------------------------------------------------
# Mensajes separados (burbujas de WhatsApp)
# --------------------------------------------------------------------------
def dividir_en_mensajes(texto: str) -> list[str]:
    """Cada párrafo (separado por una línea en blanco) es un mensaje de WhatsApp."""
    return [b.strip() for b in re.split(r"\n\s*\n", texto or "") if b.strip()]


# --------------------------------------------------------------------------
# Generador
# --------------------------------------------------------------------------
class DraftGenerator:
    def __init__(self, usar_embeddings: bool = True, modelo_embeddings: str = MODELO_EMBEDDINGS):
        df = leer_csv(DATASET_CSV)
        if df is None:
            df = leer_csv_vacio()
        self.dataset = df.fillna("")
        self.corpus = [self._texto_busqueda(c, m) for c, m in
                       zip(self.dataset.get("contexto_previo", []), self.dataset.get("mensaje_cliente", []))]
        self.modo_busqueda = "difflib"
        self.error_embeddings: str | None = None
        self._encoder = None
        self._emb = None
        self._emb_textos: dict = {}  # embeddings de descripciones de modelos (recomendación)
        if usar_embeddings and self.corpus:
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(modelo_embeddings, device="cpu")
                self._emb = self._encoder.encode(self.corpus, convert_to_tensor=True,
                                                 normalize_embeddings=True)
                self.modo_busqueda = "sentence-transformers"
            except Exception as e:  # noqa: BLE001 - sin internet / sin librería -> difflib
                self.error_embeddings = f"{type(e).__name__}: {e}"

    @staticmethod
    def _texto_busqueda(contexto: str, mensaje: str) -> str:
        contexto = (contexto or "").strip()
        return f"{mensaje} | Contexto: {contexto}" if contexto else (mensaje or "")

    # ---------------------------- búsqueda semántica ----------------------
    def buscar_similares(self, contexto: str, mensaje: str, k: int = 3) -> list[dict]:
        if not self.corpus:
            return []
        consulta = self._texto_busqueda(contexto, mensaje)
        if self.modo_busqueda == "sentence-transformers":
            from sentence_transformers import util

            q = self._encoder.encode(consulta, convert_to_tensor=True, normalize_embeddings=True)
            hits = util.semantic_search(q, self._emb, top_k=k)[0]
            pares = [(h["corpus_id"], float(h["score"])) for h in hits]
        else:
            qn = normalizar_texto(consulta)
            puntajes = [(i, difflib.SequenceMatcher(None, qn, normalizar_texto(t)).ratio())
                        for i, t in enumerate(self.corpus)]
            pares = sorted(puntajes, key=lambda x: -x[1])[:k]
        salida = []
        for i, score in pares:
            fila = self.dataset.iloc[i]
            salida.append({
                "similitud": round(score, 3),
                "contexto_previo": fila.get("contexto_previo", ""),
                "mensaje_cliente": fila.get("mensaje_cliente", ""),
                "respuesta_real": fila.get("respuesta_real", ""),
                "intenciones": fila.get("intenciones", ""),
            })
        return salida

    # ------------------------------- borrador -----------------------------
    def generar(self, analisis: dict, datos: dict, contexto: str = "", mensaje: str = "") -> dict:
        intenciones = list(analisis["intenciones"])
        ent = analisis["entidades"]
        cliente = datos["cliente"]
        bloques: list[str] = []
        notas: list[str] = []

        inicio_conversacion = not (contexto or "").strip()
        if "saludo_catalogo" in intenciones or inicio_conversacion:
            bloques.append(PLANTILLAS["saludo"].format(saludo=saludo_por_hora()))

        # "Busco wide leg talla 40": solo la lista filtrada, sin catálogo ni datos del pedido
        me = datos.get("modelos_estilo")
        # "¿Qué me recomiendas?": modelos con stock ordenados por parecido (MiniLM) con la
        # descripción real de cada modelo en la tienda
        if datos.get("recomendar") and datos.get("recomendacion") is None:
            datos["recomendacion"] = self.recomendar(mensaje, **datos["recomendar"])
        rec = datos.get("recomendacion")
        med = datos.get("medidas")
        # Pide consejo (recomendación o medidas), no está haciendo un cambio
        consejo = bool(rec or med or datos.get("guia_medidas")) and not ent.get("cambio_explicito")
        solo_lista = bool(me and me.get("tallas")) or bool(rec)

        if "saludo_catalogo" in intenciones and not solo_lista and not consejo:
            bloques.append(PLANTILLAS["catalogo"])

        # ---- ¿tienen tienda física? ----
        if "tienda_fisica" in intenciones:
            bloques.append(PLANTILLAS["venta_online"])
            bloques.append(PLANTILLAS["cambio_sin_costo"])
            if ent.get("ciudades"):
                notas.append(f"🏬 Pregunta por tienda en {', '.join(ent['ciudades'])}: "
                             "se responde que la venta es 100% online con envío gratis.")

        # ---- problema con el courier ----
        if analisis.get("alerta_courier"):
            bloques.append(self._bloque_courier(datos, notas))

        # ---- disponibilidad / precio ----
        if rec:
            bloques.append(self._bloque_recomendacion(rec, notas))
        elif "disponibilidad" in intenciones or "precio" in intenciones or datos.get("modelos_estilo"):
            # "¿Qué medidas tiene el 240904?": la tabla ya dice qué tallas hay
            if not (med and not datos["tallas_consultadas"] and "precio" not in intenciones):
                bloques.append(self._bloque_stock(intenciones, datos, notas))
        if (rec or "disponibilidad" in intenciones or "precio" in intenciones) \
                and cliente.get("cliente_frecuente"):
            notas.append(f"⭐ Cliente frecuente ({cliente['num_compras']} compras): "
                         "evalúa ofrecerle un descuento.")

        # ---- medidas de un modelo ----
        if med:
            bloques.append(self._bloque_medidas(med, notas))
        elif datos.get("guia_medidas"):
            bloques.append(PLANTILLAS["guia_medidas"])

        # ---- cambio de talla ----
        if "cambio_talla" in intenciones and not consejo:
            bloques.append(self._bloque_cambio(datos, notas))

        # ---- plazos de envío ----
        if "envio_plazos" in intenciones:
            destino = f" a {datos['ciudad']}" if datos.get("ciudad") else ""
            # Con problema de courier el pedido ya salió: no tiene sentido "si se despacha hoy"
            estimado = "" if analisis.get("alerta_courier") else estimado_entrega()
            bloques.append(PLANTILLAS["envio_plazos"].format(destino=destino, estimado=estimado))

        # ---- datos de envío ----
        solo_tienda = set(intenciones) <= {"tienda_fisica", "saludo_catalogo"} and \
            "tienda_fisica" in intenciones
        if "datos_envio" in intenciones and not solo_lista:
            bloques.append(self._bloque_datos_envio(mensaje, ent, datos, notas))
        elif ("saludo_catalogo" in intenciones or "disponibilidad" in intenciones) \
                and not solo_lista and not solo_tienda and not consejo:
            bloques.append(PLANTILLAS["datos_pedido"])

        if ({"saludo_catalogo", "disponibilidad", "precio"} & set(intenciones)
                and "cambio_talla" not in intenciones and "tienda_fisica" not in intenciones
                and not solo_lista and not consejo):
            bloques.append(PLANTILLAS["cambio_gratis"])

        borrador = "\n\n".join(b for b in bloques if b and b.strip())
        if datos.get("origen_ref") == "manual":
            notas.append(f"Modelo indicado manualmente: Ref {datos['ref']}.")
        return {
            "borrador": borrador,
            "notas": notas,
            "similares": self.buscar_similares(contexto, mensaje),
            "verificacion": verificar_borrador(borrador, datos.get("ref")),
        }

    # ---------------------------- bloques ---------------------------------
    @staticmethod
    def _bloque_courier(datos: dict, notas: list[str]) -> str:
        c = datos.get("courier")
        if not datos.get("ciudad"):
            notas.append("🚚 Alerta courier: no se conoce la ciudad del cliente.")
            return ("Lamento el inconveniente con el courier 🙏 ¿Me confirmas en qué ciudad estás "
                    "para pasarte los datos de su oficina central?")
        if not c:
            notas.append(f"🚚 Alerta courier: no hay oficina registrada en couriers.csv para {datos['ciudad']}.")
            return (f"Lamento el inconveniente con el courier 🙏 Ya mismo lo reviso y te escribo "
                    f"con los datos de contacto para {datos['ciudad']}.")
        lineas = [f"Lamento el inconveniente con el courier 🙏 Te paso los datos de su oficina "
                  f"central en {c['ciudad']}:"]
        if c.get("telefono_fijo"):
            lineas.append(f"📞 Teléfono fijo: {c['telefono_fijo']}")
        if c.get("celular_whatsapp"):
            lineas.append(f"📱 WhatsApp: {c['celular_whatsapp']}")
        if c.get("direccion"):
            lineas.append(f"📍 Dirección: {c['direccion']}")
        if c.get("horario"):
            lineas.append(f"🕒 Horario: {c['horario']}")
        if c.get("es_oficina_de_otra_ciudad"):
            notas.append(f"🚚 {c['ciudad_cliente']} no tiene oficina propia en couriers.csv; "
                         f"se usó la oficina central de {c['ciudad']}.")
        if datos.get("origen_ciudad") == "pedidos.csv":
            notas.append(f"🚚 Ciudad tomada del último pedido ({c['ciudad_cliente']}); confírmala.")
        return "\n".join(lineas)

    @staticmethod
    def _bloque_stock(intenciones: list[str], datos: dict, notas: list[str]) -> str:
        info = datos["stock_modelo"]
        ref = datos.get("ref")
        tallas = datos["tallas_consultadas"]
        pide_precio = "precio" in intenciones

        if not info["archivo_existe"]:
            notas.append("📦 Sin conexión con la tienda ni stock.csv: completa talla/precio manualmente.")
        if not ref and datos.get("modelos_estilo"):
            return DraftGenerator._bloque_estilo(datos["modelos_estilo"], notas)
        if not ref:
            notas.append("📦 No se identificó el modelo (Ref). Escríbelo en «Código de referencia» "
                         "para consultar stock y precio.")
            if tallas:
                return (f"¿Me confirmas el modelo (foto o código de referencia) para verificar la "
                        f"talla {_fmt_lista(tallas)}? 🤗")
            return "¿Me confirmas el modelo (foto o código de referencia) y la talla que buscas? 🤗"
        if not info["modelo_encontrado"]:
            if info["archivo_existe"]:
                notas.append(f"📦 La Ref {ref} no aparece en la tienda.")
            return f"Déjame verificar el modelo Ref: {ref} y te confirmo enseguida 🤗"

        if info.get("publicado") is False:
            notas.append(f"📦 La Ref {ref} NO está publicada en la web (descontinuada/privada).")
            return ("Ese modelo ya no está disponible 😔 Te comparto nuestro catálogo con los "
                    f"modelos actuales 👉 {CATALOGO_URL}")
        nombre = info.get("nombre") or "ese modelo"
        partes: list[str] = []
        if tallas:
            si = [t for t in tallas if datos["stock_tallas"][t]["talla_disponible"]]
            no = [t for t in tallas if not datos["stock_tallas"][t]["talla_disponible"]]
            if si:
                partes.append(f"¡Sí! Tenemos el modelo {nombre} (Ref: {ref}) disponible en "
                              f"{_tallas(si)} 🤩")
            if no:
                partes.append(f"Por ahora la {_tallas(no)} de ese modelo está agotada 😔")
                otras = [t for t in info["tallas_disponibles"] if t not in si]
                if otras and not si:
                    partes.append(f"Lo tenemos disponible en {_tallas(otras)}.")
        else:
            if info["tallas_disponibles"]:
                partes.append(f"¡Lo tenemos! El modelo {nombre} (Ref: {ref}) está disponible en "
                              f"{_tallas(info['tallas_disponibles'])}. ¿En qué talla lo buscas? 🤗")
            else:
                partes.append(f"Por ahora el modelo {nombre} (Ref: {ref}) está agotado 😔")

        if pide_precio or tallas:
            if info.get("precio") is not None:
                texto = f"Su precio es de {_fmt_precio(info['precio'])} Bs."
                if info.get("precio_oferta") and info["precio_oferta"] < info["precio"]:
                    texto = (f"Su precio es de {_fmt_precio(info['precio'])} Bs y ahora está en "
                             f"oferta a {_fmt_precio(info['precio_oferta'])} Bs 🎉")
                partes.append(texto)
            elif pide_precio:
                notas.append("📦 El stock no tiene precio para este modelo.")
        return " ".join(partes)

    @staticmethod
    def _bloque_estilo(res: dict, notas: list[str]) -> str:
        """'¿Tienes wide leg?' -> modelos de ese estilo/características con stock.

        Con talla ("wide leg talla 40"): TODOS los modelos con esa talla, solo Ref, nombre
        y precio. Sin talla: los 5 más nuevos con sus tallas disponibles.
        """
        estilo = " o ".join(res["estilos"]) or "jeans"
        if res.get("caracteristicas"):
            estilo += f" ({', '.join(res['caracteristicas'])})"
        link = res.get("link") or CATALOGO_URL
        tallas = res.get("tallas") or []
        en_talla = f" en {_tallas(tallas)}" if tallas else ""
        if not res["modelos"]:
            if res.get("sin_talla"):
                notas.append(f"👖 Hay {len(res['sin_talla'])} modelo(s) {estilo}, pero ninguno{en_talla}.")
                return (f"Por ahora no tenemos {estilo}{en_talla} 😔 Te comparto todos los "
                        f"modelos que tenemos 👉 {link}")
            notas.append(f"👖 No hay {estilo} con stock en la tienda.")
            return (f"Por ahora no tenemos {estilo} disponibles 😔 Te comparto nuestro "
                    f"catálogo con los modelos actuales 👉 {CATALOGO_URL}")

        lineas = []
        if tallas:
            lineas.append(f"¡Sí! Estos {estilo} tenemos{en_talla} 🤩:")
            for m in res["modelos"]:
                lineas.append(" · ".join(p for p in (f"👖 Ref {m['ref']}", m["nombre"], _precio_modelo(m)) if p))
        else:
            if res["total"] > len(res["modelos"]):
                lineas.append(f"Tenemos {res['total']} modelos {estilo}; te comparto los más nuevos:")
            elif len(res["modelos"]) == 1:
                lineas.append(f"¡Sí! Tenemos este modelo {estilo} 🤩:")
            else:
                lineas.append(f"¡Sí! Tenemos estos modelos {estilo} 🤩:")
            for m in res["modelos"]:
                partes = [f"👖 Ref {m['ref']}", m["nombre"], _precio_modelo(m), _tallas(m["tallas_disponibles"])]
                lineas.append(" · ".join(p for p in partes if p))
        if res.get("link"):
            lineas.append(f"🛍️ Todos los {' o '.join(res['estilos'])}: {link}")
        else:
            lineas.append(f"🛍️ Catálogo completo: {link}")
        if not tallas:
            lineas.append("¿Cuál te gustó más? 😍")
        notas.append(f"👖 {res['total']} modelo(s) {estilo}{en_talla} con stock en la tienda "
                     f"(se muestran {len(res['modelos'])}).")
        return "\n".join(lineas)

    # ---------------------------- recomendación ---------------------------
    def _similitud(self, consulta: str, textos: list[str]) -> list[float]:
        """Parecido semántico (MiniLM, coseno) entre el mensaje y cada texto."""
        if not textos:
            return []
        if self.modo_busqueda == "sentence-transformers":
            import numpy as np

            faltan = [t for t in textos if t not in self._emb_textos]
            if faltan:  # las descripciones se codifican una vez y se recuerdan
                for t, e in zip(faltan, self._encoder.encode(faltan, normalize_embeddings=True)):
                    self._emb_textos[t] = e
            q = self._encoder.encode(consulta, normalize_embeddings=True)
            return [float(np.dot(q, self._emb_textos[t])) for t in textos]
        qn = normalizar_texto(consulta)
        return [difflib.SequenceMatcher(None, qn, normalizar_texto(t)).ratio() for t in textos]

    def _razon(self, mensaje: str, descripcion: str, max_chars: int = 170) -> str:
        """La frase de la descripción REAL del modelo que más se parece a lo que pidió."""
        texto = re.split(r"\bDetalles\b", descripcion or "")[0]
        texto = re.sub(r"\s+([.,;:!?])", r"\1", texto)          # "personalidad ." -> "personalidad."
        frases = []
        for f in re.split(r"(?<=[.!?])\s+", texto):
            f = re.split(r"\bRef\.?:?\s*\d{5,8}\s*", f.strip())[-1]  # título con "Ref: 240933"
            # Algunas descripciones empiezan con el título pegado: "Jean Mom Fit clásico tiro
            # alto Renová tu look…" -> "Renová tu look…"
            if f.startswith("Jean "):
                m = next((m for m in re.finditer(r"\s([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)\b", f)
                          if m.group(1).lower() not in _PALABRAS_TITULO), None)
                if m:
                    f = f[m.start(1):]
            if len(f) >= 25:
                frases.append(f)
        if not frases:
            return ""
        sims = self._similitud(mensaje, frases)
        mejor = frases[max(range(len(frases)), key=lambda i: sims[i])]
        if len(mejor) > max_chars:
            mejor = mejor[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
        return mejor

    def recomendar(self, mensaje: str, estilos: list[str], caracteristicas: list[str],
                   tallas: list[int], medidas_cliente: dict, k: int = 3) -> dict:
        """Recomienda k modelos CON STOCK (nunca inventados).

        1. Filtra por lo que pidió la clienta: estilo, características, talla o, si dio sus
           medidas, la talla de cada modelo según su tabla de medidas.
        2. Ordena por similitud semántica (MiniLM) entre su mensaje y el nombre + la
           descripción del modelo en la tienda ("para un matrimonio" ~ "outfits elevados").
        3. Explica cada uno con la frase de su descripción que más se parece al mensaje.
        """
        filtro_talla = tallas if (tallas and not medidas_cliente) else None
        res = buscar_modelos_por_estilo(estilos, filtro_talla, None, caracteristicas, todos=True)
        relajado = False
        if not res["modelos"] and (estilos or caracteristicas):
            relajado = True  # no hay de ese estilo: se recomienda entre todos
            res = buscar_modelos_por_estilo([], filtro_talla, None, [], todos=True)
        candidatos, sin_tabla = res["modelos"], 0
        if medidas_cliente:
            con_talla = []
            for m in candidatos:
                tabla = tienda_api.medidas_de(m["ref"])
                tp = talla_por_medidas(tabla, medidas_cliente, m["tallas_disponibles"])
                if tp is None:
                    sin_tabla += 1
                    continue
                t = tp["talla"] if tp["disponible"] else tp["alternativa"]
                if t is not None:
                    con_talla.append({**m, "talla_rec": t, "fila_rec": tabla[t]})
            candidatos = con_talla
        textos = [f"{m['nombre_completo']}. {m['descripcion']}" for m in candidatos]
        for m, s in zip(candidatos, self._similitud(mensaje, textos)):
            m["similitud"] = round(s + (0.03 if m.get("precio_oferta") else 0.0), 3)
        candidatos.sort(key=lambda m: str(m["ref"]), reverse=True)   # empate: el más nuevo
        candidatos.sort(key=lambda m: -m["similitud"])
        elegidos = candidatos[:k]
        for m in elegidos:
            m["razon"] = self._razon(mensaje, m["descripcion"])
        return {"modelos": elegidos, "total": len(candidatos), "estilos": estilos,
                "caracteristicas": caracteristicas, "tallas": tallas if not medidas_cliente else [],
                "medidas_cliente": medidas_cliente, "relajado": relajado, "sin_tabla": sin_tabla,
                "metodo": self.modo_busqueda}

    @staticmethod
    def _bloque_recomendacion(rec: dict, notas: list[str]) -> str:
        filtro = " o ".join(rec["estilos"]) or "jeans"
        if rec.get("caracteristicas"):
            filtro += f" ({', '.join(rec['caracteristicas'])})"
        medidas = rec.get("medidas_cliente") or {}
        tallas = rec.get("tallas") or []
        if not rec["modelos"]:
            if medidas:
                notas.append(f"✨ Ningún modelo con tabla de medidas tiene stock para {medidas} "
                             f"({rec['sin_tabla']} modelos aún no tienen tabla publicada).")
                return ("Con esas medidas no encontré un modelo con stock que te quede bien 😔 "
                        "¿Me pasas el código o la foto del modelo que te gustó? Te ayudo a elegir la talla 🤗")
            notas.append("✨ No hay modelos con stock para recomendar con esos filtros.")
            return f"Por ahora no tenemos modelos disponibles en esa talla 😔 Te comparto el catálogo 👉 {CATALOGO_URL}"

        lineas = []
        if rec["relajado"]:
            lineas.append(f"Por ahora no tenemos {filtro} disponibles"
                          f"{' en tu talla' if (tallas or medidas) else ''} 😔")
        if medidas:
            lineas.append("Según tus medidas, te recomiendo estos modelos 💙:")
        elif tallas:
            lineas.append(f"Te recomiendo estos modelos en {_tallas(tallas)} 💙:")
        else:
            lineas.append("Te recomiendo estos modelos 💙:")
        for m in rec["modelos"]:
            if medidas:
                ajuste = {k: v for k, v in m["fila_rec"].items() if k in medidas}
                talla_txt = f"talla {m['talla_rec']}" + (f" ({texto_medidas(ajuste)})" if ajuste else "")
            elif tallas:
                talla_txt = ""
            else:
                talla_txt = _tallas(m["tallas_disponibles"])
            partes = (f"👖 Ref {m['ref']}", m["nombre"], _precio_modelo(m), talla_txt)
            lineas.append(" · ".join(p for p in partes if p))
            if m.get("razon"):
                lineas.append(f"✨ {m['razon']}")
        if medidas:
            lineas.append("Son medidas de la prenda, no del cuerpo: compáralas con un jean que ya te "
                          "quede bien. Si dudas entre dos tallas, el primer cambio no te cuesta el envío 💕")
        else:
            lineas.append("¿Cuál te gustó más?" + ("" if tallas else " ¿En qué talla lo buscas?") + " 😍")
        notas.append(f"✨ Recomendación: {rec['total']} modelo(s) con stock que cumplen lo pedido; "
                     f"se eligieron los {len(rec['modelos'])} más parecidos al mensaje "
                     f"({'MiniLM' if rec['metodo'] == 'sentence-transformers' else 'difflib'} sobre "
                     "las descripciones de la tienda).")
        if rec.get("sin_tabla"):
            notas.append(f"📏 {rec['sin_tabla']} modelo(s) no entraron porque aún no tienen su tabla "
                         "de medidas publicada en la web.")
        return "\n".join(lineas)

    @staticmethod
    def _bloque_medidas(med: dict, notas: list[str]) -> str:
        ref, tabla, cliente = med["ref"], med["tabla"], med.get("cliente") or {}
        if tabla is None:
            notas.append("📏 No se pudo leer la tabla de medidas de la tienda (sin conexión ni copia).")
            return f"Te paso la tabla de medidas del modelo Ref: {ref} enseguida 🤗"
        if not tabla:
            notas.append(f"📏 La Ref {ref} todavía no tiene su tabla de medidas publicada en la web.")
            return ("La tabla de medidas de este modelo todavía no está publicada 😔 Si me pasas tu "
                    "medida de cintura (o la de un jean que ya te quede bien), te digo qué talla pedir 🤗")
        lineas: list[str] = []
        mostrar = list(med.get("tallas") or [])
        tp = med.get("talla_por_medidas")
        if cliente and tp:
            medida_txt = " y ".join(f"{k.lower()} de {v:g} cm".replace(".", ",")
                                    for k, v in cliente.items() if k in tp["medidas_usadas"])
            if tp["fuera_de_rango"]:
                lineas.append(f"Con tu {medida_txt}, este modelo no tiene una talla que te quede bien 😔")
                mostrar = []
                notas.append(f"📏 La medida de la clienta supera la talla más grande de la Ref {ref}.")
            elif tp["disponible"]:
                lineas.append(f"Con tu {medida_txt}, te recomiendo la talla {tp['talla']} 🤩")
                mostrar = [tp["talla"]]
            elif tp["alternativa"]:
                lineas.append(f"Con tu {medida_txt}, la talla ideal ({tp['talla']}) está agotada 😔 "
                              f"Te recomiendo la talla {tp['alternativa']}, que sí está disponible.")
                mostrar = [tp["alternativa"]]
            else:
                lineas.append(f"Con tu {medida_txt}, la talla ideal ({tp['talla']}) está agotada 😔")
                mostrar = []
            notas.append(f"📏 Talla calculada con la tabla de medidas de la Ref {ref}: la primera talla "
                         f"cuya prenda mide al menos {medida_txt}.")
        mostrar = [t for t in mostrar if t in tabla]
        if mostrar:
            lineas.append(f"📏 Medidas de la prenda (Ref {ref}):")
            lineas += [f"📏 Talla {t}: {texto_medidas(tabla[t])}" for t in mostrar]
        elif not lineas:
            notas.append(f"📏 La Ref {ref} no tiene tallas con stock para mostrar medidas.")
            return f"Por ahora el modelo Ref: {ref} no tiene tallas disponibles 😔"
        lineas.append("Son medidas de la prenda, no del cuerpo: compáralas con un jean que ya te quede bien 💙")
        if cliente:
            lineas.append("Si dudas entre dos tallas, el primer cambio no te cuesta el envío 💕")
        return "\n".join(lineas)

    @staticmethod
    def _bloque_cambio(datos: dict, notas: list[str]) -> str:
        previo = datos.get("articulo_previo")
        ref = datos.get("ref")
        lineas: list[str] = []
        if previo:
            notas.append(f"🔁 Último pedido: {previo['modelo']} (Ref {previo['ref']}), "
                         f"talla {previo['talla']}.")
        if datos.get("talla_sugerida"):
            notas.append(f"🔁 Talla sugerida automáticamente ({datos['talla_sugerida']}) "
                         "según 'grande/chico'. Confírmala con el cliente.")

        tallas = datos["tallas_consultadas"]
        if ref and tallas and datos["stock_modelo"]["modelo_encontrado"]:
            si = [t for t in tallas if datos["stock_tallas"][t]["talla_disponible"]]
            no = [t for t in tallas if t not in si]
            if si:
                lineas.append(f"Se puede hacer el cambio sin ningún problema, tenemos el modelo "
                              f"(Ref: {ref}) en {_tallas(si)} 🙌")
            if no:
                lineas.append(f"Por ahora la {_tallas(no)} de ese modelo está agotada 😔")
                otras = datos["stock_modelo"]["tallas_disponibles"]
                if otras and not si:
                    lineas.append(f"Lo tenemos disponible en {_tallas(otras)}. "
                                  "Si gustas, puedes cambiarlo por otro modelo de nuestro "
                                  f"catálogo 👉 {CATALOGO_URL}")
                if not si:  # sin talla para cambiar: no tiene sentido mandar instrucciones aún
                    notas.append("🔁 La talla pedida está agotada: se omitieron las instrucciones de cambio.")
                    return "\n".join(lineas)
        else:
            lineas.append("¡Claro! Se puede hacer el cambio sin ningún problema 🙌")
            if not ref:
                notas.append("🔁 No se pudo identificar el modelo a cambiar (sin pedido previo ni Ref).")
            elif not datos["stock_modelo"]["archivo_existe"]:
                notas.append("🔁 Sin conexión con la tienda ni stock.csv: verifica manualmente la nueva talla.")
            elif not datos["stock_modelo"]["modelo_encontrado"]:
                notas.append(f"🔁 La Ref {ref} no aparece en la tienda: verifica la nueva talla.")
            elif not tallas:
                notas.append("🔁 El cliente no indicó la talla nueva.")
        lineas.append(PLANTILLAS["cambio_instrucciones"])
        return "\n".join(lineas)

    @staticmethod
    def _bloque_datos_envio(mensaje: str, ent: dict, datos: dict, notas: list[str]) -> str:
        if not ent.get("tiene_datos_envio"):
            return "¡Claro que sí, te ayudo con tu pedido! 🤩\n" + PLANTILLAS["datos_pedido"]

        extraidos = extraer_datos_envio(mensaje, ent)
        if not extraidos["Modelo"] and datos.get("ref"):
            extraidos["Modelo"] = f"Ref: {datos['ref']}"
        lineas = ["¡Perfecto! Confirmo los datos de tu pedido 📝"]
        for campo in CAMPOS_PEDIDO:
            valor = extraidos[campo]
            if valor:
                lineas.append(f"{ICONOS_CAMPO[campo]} {campo}: {valor}")
        faltan = [c for c in CAMPOS_PEDIDO if not extraidos[c]]
        if faltan:
            lineas.append(f"Solo me faltaría: {_fmt_lista(faltan)} 🙏")
            notas.append(f"📝 Faltan datos del pedido: {', '.join(faltan)}.")
        else:
            lineas.append(PLANTILLAS["confirmacion_qr"])
        return "\n".join(lineas)


def leer_csv_vacio():
    import pandas as pd

    return pd.DataFrame(columns=["contexto_previo", "mensaje_cliente", "respuesta_real", "intenciones"])
