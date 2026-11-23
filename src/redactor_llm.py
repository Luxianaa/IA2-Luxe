"""Redacción con un modelo de lenguaje abierto local (Qwen 2.5 vía Ollama).

Arquitectura (RAG con datos verificados):
    1. El pipeline determinista (clasificador + CSV + plantillas) produce un BORRADOR BASE
       con todos los datos correctos: tallas, precios, Ref, courier, plazos, links.
    2. Las partes que deben quedar idénticas (lista de 6 datos, contactos del courier,
       instrucciones de cambio, confirmación del pedido) se reemplazan por marcadores
       [LISTA_1], [LISTA_2]… y se reinsertan tal cual después.
    3. Se recuperan respuestas reales de Merci parecidas (búsqueda semántica) como
       ejemplos de tono (few-shot).
    4. Qwen 2.5 3B (Qwen Research License: uso académico; corre localmente con Ollama) REDACTA la parte conversacional
       con sus propias palabras, respondiendo a lo que dijo la clienta.
    5. Control de fidelidad: todo número/link del borrador base debe seguir presente y no
       debe aparecer ningún número nuevo (posible dato inventado). Además, el verificador
       de stock de draft_generator revisa tallas y precios.

No usa APIs externas: Ollama sirve el modelo en http://localhost:11434.
"""
from __future__ import annotations

import json
import os
import random
import re
import unicodedata
import urllib.error
import urllib.request
from typing import Iterator

OLLAMA_URL = os.environ.get("HILO_OLLAMA_URL", "http://localhost:11434")
MODELO_LLM = os.environ.get("HILO_LLM_MODELO", "qwen2.5:3b")
OPCIONES = {
    "temperature": 0.3,   # baja: redacta con variedad pero se ciñe al borrador base
    "top_p": 0.9,
    "num_predict": 300,   # máximo de tokens de la respuesta
    "num_ctx": 4096,
    # Si el modelo empieza a repetir partes del prompt, se corta ahí
    "stop": ["Mensaje de la clienta", "Último mensaje de Merci", "BORRADOR BASE",
             "EJEMPLOS DE", "DATOS QUE DEBEN", "CONVERSACIÓN ACTUAL", "<<<", ">>>",
             "- Clienta:", "Clienta:"],
}
MANTENER_CARGADO = "60m"  # el modelo queda en memoria 1 h (respuestas más rápidas)

SISTEMA = """Eres la vendedora de Merci Collection, una tienda boliviana de jeans que vende 100% online por WhatsApp. Escribes como una persona real: cálida, cercana, breve, con algunos emojis como 💙👖🤩💕 (sin exagerar).

REGLAS ESTRICTAS:
1. Hablas TÚ, la vendedora, a la clienta. Nunca escribas como si fueras la clienta.
2. Usa SOLO la información del BORRADOR BASE. No inventes precios, tallas, códigos, teléfonos, emails, direcciones, horarios, fechas, descuentos, promociones, requisitos ni pasos nuevos. Si algo no está en el borrador base, no existe: no lo menciones.
2b. PROHIBIDO ofrecer o sugerir cosas que no están en el borrador base: otros productos, accesorios (cinturones, bolsas, prendas), regalos, envolturas, servicios, materiales o beneficios. No hagas preguntas nuevas: solo puedes preguntar lo que ya pregunta el borrador base. No recomiendes por tu cuenta: recomienda solo los modelos y tallas que recomienda el borrador base.
3. OBLIGATORIO: menciona TODOS los precios (Bs), tallas, códigos Ref y links del borrador base, con los mismos números.
4. Si la clienta expresa una emoción o situación (un regalo, apuro, enojo, su primera compra, un evento), tu PRIMERA frase reacciona a ESO con calidez. Si no expresa nada de eso, NO inventes situaciones: empieza directo con la información.
5. Después, cuenta la información del borrador base con tus propias palabras, de forma natural y breve.
6. Los marcadores como [LISTA_1] son bloques que se insertan solos (listas, datos de contacto y textos oficiales). Escríbelos tal cual, cada uno en su propia línea, en el lugar que corresponda. No escribas listas tú.
7. NO saludes (nada de "Hola" ni "Buenos días/tardes/noches"): el saludo ya está escrito.
8. Escribe solo en español. Devuelve únicamente el mensaje para la clienta, sin títulos ni explicaciones.
9. Escribe en párrafos cortos (1 o 2 frases) separados por una línea en blanco: cada párrafo se envía como un mensaje de WhatsApp."""

_RE_PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")
_RE_SALUDO_BASE = re.compile(r"^\s*(?:Buen(?:os|as)\s+(?:d[ií]as|tardes|noches)|Buen d[ií]a)!.*$", re.I)
_RE_SALUDO_LLM = re.compile(
    # uno o más saludos seguidos: "Hola, buenas noches! 👋", "Hola de nuevo!"
    r"^\s*(?:¡?\s*(?:muy\s+)?(?:hola+(?:\s+de\s+nuevo)?|buen(?:os|as)\s+(?:d[ií]as|tardes|noches)|buen\s+d[ií]a|buenas)\b[\s,!.¡]*)+"
    r"(?:[^\w\s¡¿]{1,3}\s*)*"
    # "Gracias por comunicarte con Merci…" ya está en el saludo fijo
    r"(?:¡?\s*gracias por (?:comunicarte|contactarnos|escribirnos)[^!.\n]*[!.]+\s*(?:[^\w\s¡¿]{1,3}\s*)*)?",
    re.I,
)
_INICIOS_LISTA = ("1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "📞", "📱", "📍", "🕒",
                  "👤", "🏠", "📧", "📏", "👖", "🛍", "✨")
# Textos oficiales que se envían siempre igual (respuesta a "¿tienen tienda física?")
_FRASES_FIJAS = ("Nuestra venta es 100% online", "Si te preocupa la talla, no hay problema")


def _es_fija(linea: str) -> bool:
    t = linea.strip()
    return t.startswith(_INICIOS_LISTA) or t.startswith(_FRASES_FIJAS)


_RE_MARCADOR = re.compile(r"\[\s*LISTA[\s_-]*(\d+)\s*\]", re.I)  # [LISTA_1], [Lista 1]…


# --------------------------------------------------------------------------
# Conexión con Ollama
# --------------------------------------------------------------------------
def _get(ruta: str, timeout: float = 3) -> dict:
    with urllib.request.urlopen(f"{OLLAMA_URL}{ruta}", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def estado_ollama(modelo: str = MODELO_LLM) -> dict:
    """{'activo': bool, 'modelo_descargado': bool, 'mensaje': str}"""
    try:
        modelos = [m.get("name", "") for m in _get("/api/tags").get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return {"activo": False, "modelo_descargado": False,
                "mensaje": "Ollama no está corriendo. Ábrelo desde el menú Inicio."}
    base = modelo.split(":")[0]
    descargado = any(m == modelo or m.startswith(f"{modelo}") or
                     (":" not in modelo and m.split(":")[0] == base) for m in modelos)
    if not descargado:
        return {"activo": True, "modelo_descargado": False,
                "mensaje": f"Falta descargar el modelo: ollama pull {modelo}"}
    return {"activo": True, "modelo_descargado": True, "mensaje": f"{modelo} listo"}


# --------------------------------------------------------------------------
# Preparación del borrador base
# --------------------------------------------------------------------------
def separar_saludo(borrador_base: str) -> tuple[str, str]:
    """Separa el saludo (fijado por la hora real) del resto del borrador base."""
    bloques = (borrador_base or "").strip().split("\n\n", 1)
    if bloques and _RE_SALUDO_BASE.match(bloques[0]):
        return bloques[0].strip(), (bloques[1] if len(bloques) > 1 else "").strip()
    return "", (borrador_base or "").strip()


def extraer_bloques_fijos(texto: str) -> tuple[str, dict[str, str]]:
    """Reemplaza listas y líneas de contacto por [LISTA_n] (se reinsertan tal cual).

    Un bloque fijo = líneas seguidas que empiezan con 1️⃣…6️⃣ o íconos de datos
    (📞📱📍🕒👤🏠📧📏👖🛍), más su línea de encabezado si es corta y termina en ':' o '📝'.
    """
    lineas = (texto or "").split("\n")
    salida: list[str] = []
    bloques: dict[str, str] = {}
    i = 0
    while i < len(lineas):
        if _es_fija(lineas[i]):
            grupo = []
            # encabezado ("Para realizar tu pedido, ayúdame con estos datos 📝")
            if salida and len(salida[-1]) <= 70 and salida[-1].rstrip().endswith((":", "📝")):
                grupo.append(salida.pop())
            while i < len(lineas) and _es_fija(lineas[i]):
                grupo.append(lineas[i])
                i += 1
            clave = f"LISTA_{len(bloques) + 1}"
            bloques[clave] = "\n".join(grupo).strip()
            salida.append(f"[{clave}]")
        else:
            salida.append(lineas[i])
            i += 1
    return "\n".join(salida).strip(), bloques


_RE_DATO_CLAVE = re.compile(
    r"Ref:?\s*\d{5,8}"                                   # código de modelo
    r"|\d{2,4}(?:[.,]\d{1,2})?\s*Bs"                     # precios
    r"|tallas?\s+(?:3[4-9]|4[0-8])(?:\s*(?:,|y)\s*(?:3[4-9]|4[0-8]))*"  # tallas
    r"|https?://[^\s)>\]]+"                              # links
    r"|\d{7,8}(?:\s*-\s*\d{7,8})*"                       # teléfonos
    r"|\d+\s*a\s*\d+\s*d[ií]as\s*h[aá]biles",            # plazo de envío
    re.I,
)


def datos_obligatorios(texto: str) -> list[str]:
    """Datos concretos (fuera de los bloques fijos) que el modelo no puede omitir."""
    vistos, salida = set(), []
    resto = texto or ""
    # Las malas noticias (talla agotada, modelo no disponible) son lo primero que se pierde.
    # Sus datos ("talla 46") NO van sueltos en la lista: el modelo los copiaría como si
    # hubiera stock. Solo van dentro del aviso.
    for oracion in re.split(r"(?<=[.!?😔])\s+|\n", texto or ""):
        if re.search(r"agotad|ya no est[aá] disponible", oracion, re.I):
            salida.append(f"Avisar (con tus palabras) que: {oracion.strip()}")
            resto = resto.replace(oracion, " ")
    for m in _RE_DATO_CLAVE.finditer(resto):
        d = m.group(0).strip().rstrip(".,;")
        if d.lower() not in vistos:
            vistos.add(d.lower())
            salida.append(d)
    return salida


def _limpiar_ejemplo(texto: str, max_chars: int = 220) -> str:
    t = _RE_PLACEHOLDER.sub("", texto or "")
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) <= max_chars else t[:max_chars].rsplit(" ", 1)[0] + "…"


def construir_mensajes(mensaje: str, contexto: str, cuerpo_marcado: str,
                       similares: list[dict], max_ejemplos: int = 2) -> list[dict]:
    ejemplos = []
    for s in similares[:max_ejemplos]:
        resp = _limpiar_ejemplo(s.get("respuesta_real", ""))
        if resp:
            ejemplos.append(f"- Clienta: {_limpiar_ejemplo(s.get('mensaje_cliente', ''), 160)}\n"
                            f"  Merci: {resp}")
    bloque_ejemplos = "\n".join(ejemplos) or "(sin ejemplos)"
    obligatorios = datos_obligatorios(cuerpo_marcado)
    marcadores = _RE_MARCADOR.findall(cuerpo_marcado)
    if marcadores:
        obligatorios += [f"[LISTA_{n}] (en su propia línea)" for n in marcadores]
    bloque_obligatorios = (
        "\n\nDATOS QUE DEBEN APARECER EN TU RESPUESTA:\n" +
        "\n".join(f"- {d}" for d in obligatorios)) if obligatorios else ""
    usuario = (
        "EJEMPLOS DE RESPUESTAS REALES DE MERCI (solo para imitar el tono):\n"
        f"{bloque_ejemplos}\n\n"
        "CONVERSACIÓN ACTUAL:\n"
        f"Último mensaje de Merci: {contexto.strip() if contexto and contexto.strip() else '(inicio de la conversación)'}\n"
        f"Mensaje de la clienta: {mensaje.strip()}\n\n"
        "BORRADOR BASE (información correcta; redáctala con tus propias palabras):\n"
        f"<<<\n{cuerpo_marcado.strip()}\n>>>"
        f"{bloque_obligatorios}\n\n"
        "Escribe la respuesta final para la clienta:"
    )
    return [{"role": "system", "content": SISTEMA}, {"role": "user", "content": usuario}]


# --------------------------------------------------------------------------
# Post-procesamiento en streaming
# --------------------------------------------------------------------------
def quitar_saludo_llm(texto: str) -> str:
    """Si el modelo saludó igual, se quita (el saludo correcto ya va adelante)."""
    # Etiquetas del prompt que a veces copia: "Mensaje de Merci: …", "Respuesta: …"
    t = re.sub(r"^\s*(?:mensaje(?: de merci)?|merci|respuesta(?: final)?|vendedora)\s*:\s*", "",
               texto or "", flags=re.I)
    t = _RE_SALUDO_LLM.sub("", t, count=1).lstrip()
    t = re.sub(r"^\s*(?:mensaje(?: de merci)?|merci|respuesta(?: final)?|vendedora)\s*:\s*", "",
               t, flags=re.I)
    m = re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]", t)
    if m and t[m.start()].islower():  # "tienes nuestro…" -> "Tienes nuestro…"
        t = t[: m.start()] + t[m.start()].upper() + t[m.start() + 1:]
    return t


class _Sustituidor:
    """Reemplaza [LISTA_n] por su bloque fijo a medida que llega el texto."""

    def __init__(self, bloques: dict[str, str]):
        self.bloques, self.buf, self.usados = bloques, "", set()

    def _token(self, token: str) -> str:
        m = _RE_MARCADOR.fullmatch(token)
        if not m:
            return token
        clave = f"LISTA_{m.group(1)}"
        if clave in self.bloques and clave not in self.usados:
            self.usados.add(clave)
            return f"\n\n{self.bloques[clave]}\n\n"  # cada bloque, su propio mensaje
        return ""  # marcador repetido o inexistente: se descarta

    def feed(self, trozo: str) -> str:
        self.buf += trozo
        out = ""
        while self.buf:
            i = self.buf.find("[")
            if i == -1:
                out, self.buf = out + self.buf, ""
                break
            out, self.buf = out + self.buf[:i], self.buf[i:]
            j = self.buf.find("]")
            if j == -1:
                if len(self.buf) > 20:  # no era un marcador
                    out, self.buf = out + self.buf[0], self.buf[1:]
                    continue
                break
            out, self.buf = out + self._token(self.buf[: j + 1]), self.buf[j + 1:]
        return out

    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return out

    def faltantes(self) -> list[str]:
        return [b for k, b in self.bloques.items() if k not in self.usados]


# --------------------------------------------------------------------------
# Generación
# --------------------------------------------------------------------------
def redactar_stream(mensaje: str, contexto: str, borrador_base: str, similares: list[dict],
                    modelo: str = MODELO_LLM, timeout: float = 180,
                    variar: bool = False, quitadas: list | None = None) -> Iterator[str]:
    """Genera la respuesta token a token (para mostrarla mientras se escribe).

    El saludo ("Buenas tardes! …") no lo escribe el modelo: se toma del borrador base,
    que lo calcula con la hora real de Bolivia, y se antepone a la respuesta.
    Las frases sin respaldo en el borrador base se quitan y se anotan en `quitadas`.
    """
    saludo, cuerpo_base = separar_saludo(borrador_base)
    cuerpo_marcado, bloques = extraer_bloques_fijos(cuerpo_base or borrador_base)
    opciones = dict(OPCIONES)
    if variar:  # "Redactar otra versión": más creatividad y otra semilla
        opciones.update({"temperature": 0.6, "seed": random.randint(1, 10**9)})
    cuerpo = json.dumps({
        "model": modelo,
        "messages": construir_mensajes(mensaje, contexto, cuerpo_marcado, similares),
        "stream": True,
        "options": opciones,
        "keep_alive": MANTENER_CARGADO,
    }).encode("utf-8")
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=cuerpo,
                                 headers={"Content-Type": "application/json"})
    if saludo:
        yield saludo + "\n\n"
    sust = _Sustituidor(bloques)
    filtro = _FiltroFrases(vocabulario_respaldo(borrador_base, mensaje, contexto), quitadas)
    inicio, inicio_listo = "", False
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for linea in r:
            if not linea.strip():
                continue
            dato = json.loads(linea.decode("utf-8"))
            if dato.get("error"):
                raise RuntimeError(dato["error"])
            trozo = dato.get("message", {}).get("content", "")
            if trozo and not inicio_listo:
                # Se acumulan los primeros caracteres para quitar un saludo duplicado
                inicio += trozo
                if len(inicio) >= 40 or "\n" in inicio:
                    trozo, inicio_listo = quitar_saludo_llm(inicio), True
                else:
                    trozo = ""
            if trozo:
                salida = sust.feed(filtro.feed(trozo))
                if salida:
                    yield salida
            if dato.get("done"):
                break
    if not inicio_listo and inicio:
        yield sust.feed(filtro.feed(quitar_saludo_llm(inicio)))
    resto = sust.feed(filtro.flush()) + sust.flush()
    if resto:
        yield resto
    for bloque in sust.faltantes():  # bloques que el modelo no ubicó: van al final
        yield f"\n\n{bloque}"


def redactar(mensaje: str, contexto: str, borrador_base: str, similares: list[dict],
             modelo: str = MODELO_LLM, variar: bool = False,
             quitadas: list | None = None) -> str:
    return limpiar_salida("".join(redactar_stream(
        mensaje, contexto, borrador_base, similares, modelo, variar=variar, quitadas=quitadas)))


def precalentar(modelo: str = MODELO_LLM) -> None:
    """Carga el modelo en memoria para que la primera respuesta no tarde de más."""
    cuerpo = json.dumps({"model": modelo, "prompt": "", "keep_alive": MANTENER_CARGADO}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=cuerpo,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        r.read()


def limpiar_salida(texto: str) -> str:
    t = (texto or "").strip()
    t = re.sub(r"^(respuesta( final)?|mensaje)\s*:\s*", "", t, flags=re.I)
    t = t.strip().strip("<>").strip()
    t = re.sub(r"\[(https?://[^\]\s]+)\]", r"\1", t)  # "[https://…]" -> link limpio
    t = re.sub(r"[ \t]+\n", "\n", t)
    # Después del saludo fijo (1.er párrafo), un saludo o etiqueta repetidos se quitan:
    # "Muy buenas noches! …", "Mensaje de Merci: ¡Hola de nuevo! …"
    parrafos = re.split(r"\n\s*\n", t)
    for i in range(1, len(parrafos)):
        if not _es_fija(parrafos[i]):
            parrafos[i] = quitar_saludo_llm(parrafos[i])
    t = "\n\n".join(p for p in parrafos if p.strip())
    return re.sub(r"\n{3,}", "\n\n", t)


# --------------------------------------------------------------------------
# Filtro de frases sin respaldo (determinista)
# --------------------------------------------------------------------------
# Qwen a veces agrega frases con cosas que no existen ("¿Te envío el pantalón con un
# cinturón de cuero?"). Cada frase del modelo se compara con el vocabulario del borrador
# base + el mensaje de la clienta: si trae varias palabras de contenido que no salen de
# ahí (o es una oferta con algo nuevo), se elimina antes de mostrarse.
_IGNORAR = set("""
para pero como cuando donde porque sobre entre desde hasta hacia tambien mucho mucha muchos
muchas poco poca este esta esto estos estas esos esas aquel aquella nuestro nuestra nuestros
nuestras ellos ellas nosotros usted ustedes todo toda todos todas otro otra otros otras cada
algo alguna alguno algun nada solo sola aqui alli ahora luego despues antes entonces tanto
cual cuales quien quienes sera seria puede pueden hace hacer haya tener tengo tienes tiene
tenemos tienen estar estoy estamos estan sido eres somos mismo misma vamos dejar dejo quiero
quieres quiere puedo puedes podemos unos unas sigue siempre igual mejor tuyo tuya vale
gustaria gusto gusta gustan gustara gustado encanta encantar encantara encantado encantada
encantaria lindo linda lindos lindas hermoso hermosa hermosos bello bella precioso preciosa
perfecto perfecta genial ideal super seguro segura claro clara gracias feliz felices alegra
alegria entiendo entendemos tranquila tranquilo preocupes preocupe preocupa disculpa disculpas
lamento lamentamos sentimos siento molestia ayudar ayudo ayudarte ayudamos quedar queda quedan
quedara quedaria quedaran detalle especial momento mismo enseguida pronto rapido cualquier
consulta duda dudas avisame avisa escribeme cuenta contar contarte decir decirte dime mira aqui
modelo modelos jean jeans pantalon pantalones talla tallas pedido pedidos compra comprar envio
envios enviar enviamos merci collection disponible disponibles opcion opciones elegir escoger
favorito favorita bien buena bueno buenas buenos excelente increible primera primer bienvenida
confianza contigo necesitas necesites quieras puedas tengas listo lista saber sabes ojala
espero esperamos verte encuentres escoges eliges elijas hola tardes noches dias atender
atenderte informacion info estilo estilos alguno ninguna ninguno nuevo nueva nuevos nuevas
""".split())
_RE_OFERTA = re.compile(
    r"te gustar[ií]a|quieres que|\b(?:te|se) (?:l[oa]s? )?(?:puedo|podemos|podr[ií]a|podr[ií]amos)\b"
    r"|te ofrezco|te recomiendo|tambi[eé]n tenemos|inclu(?:imos|ye|ir)|de regalo|adicional"
    r"|agreg|añad|acompañ|complement|envuelt|envolt", re.I)
# Fin de frase: salto de línea, puntuación (+ emojis que la siguen) o justo antes de ¿ / ¡
_RE_FIN_FRASE = re.compile(r"\n|[.!?…]+(?:\s*[^\w\s\[¡¿])*\s|\s(?=[¡¿])")


def _normalizar(t: str) -> str:
    t = unicodedata.normalize("NFD", (t or "").lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def _palabras(t: str) -> list[str]:
    return [w for w in re.findall(r"[a-zñ]{4,}", _normalizar(_RE_URL.sub(" ", t or "")))
            if w not in _IGNORAR]


def vocabulario_respaldo(*textos: str) -> set[str]:
    """Raíces (4 letras) de las palabras que el modelo SÍ puede usar."""
    return {w[:4] for t in textos for w in _palabras(t)}


def frase_sin_respaldo(frase: str, respaldo: set[str]) -> list[str]:
    """Palabras nuevas de la frase si debe eliminarse; [] si está respaldada."""
    # Los bloques fijos llegan como [LISTA_n] (se insertan después de este filtro), así que
    # una frase del modelo que empieza con 👖/✨ igual se revisa
    if _RE_MARCADOR.search(frase):
        return []
    contenido = _palabras(frase)
    nuevas = [w for w in contenido if w[:4] not in respaldo]
    if not nuevas:
        return []
    if _RE_OFERTA.search(_normalizar(frase)) or _RE_OFERTA.search(frase):
        return nuevas                                   # oferta con algo que no existe
    if len(nuevas) >= 2 and len(nuevas) / len(contenido) >= 0.4:
        return nuevas                                   # frase mayormente inventada
    return []


class _FiltroFrases:
    """Deja pasar el texto frase por frase, quitando las que no tienen respaldo."""

    def __init__(self, respaldo: set[str], quitadas: list | None):
        self.respaldo, self.buf = respaldo, ""
        self.quitadas = quitadas if quitadas is not None else []

    def _decidir(self, frase: str) -> str:
        cuerpo = frase.strip()
        if cuerpo and frase_sin_respaldo(cuerpo, self.respaldo):
            self.quitadas.append(cuerpo)
            return "\n" * min(frase.count("\n"), 2)     # conserva el salto de párrafo
        return frase

    def feed(self, trozo: str) -> str:
        self.buf += trozo
        out = ""
        while True:
            m = _RE_FIN_FRASE.search(self.buf)
            if not m:
                return out
            fin = m.end()
            # Lo que sigue y aún es de esta frase: espacios y, tras puntuación, emojis ("? 🎯✨")
            while fin < len(self.buf) and (self.buf[fin].isspace() or (
                    m.group(0) != "\n" and not self.buf[fin].isalnum()
                    and self.buf[fin] not in "¡¿[")):
                fin += 1
            if fin == len(self.buf):
                return out          # puede seguir llegando texto de la misma frase
            out += self._decidir(self.buf[:fin])
            self.buf = self.buf[fin:]

    def flush(self) -> str:
        out, self.buf = (self._decidir(self.buf) if self.buf else ""), ""
        return out


# --------------------------------------------------------------------------
# Control de fidelidad (determinista)
# --------------------------------------------------------------------------
_RE_URL = re.compile(r"https?://[^\s)>\]]+|mailto:[^\s)>\]]+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_RE_NUM = re.compile(r"\d{2,}")


def _urls(t: str) -> set[str]:
    """Links y emails (un email nuevo = dato de contacto inventado)."""
    return {u.rstrip(".,;:!?").lower() for u in _RE_URL.findall(t or "")}


def _numeros(t: str) -> set[str]:
    sin_urls = _RE_URL.sub(" ", t or "")
    return set(_RE_NUM.findall(sin_urls))


def revisar_fidelidad(borrador_base: str, generado: str) -> dict:
    """Compara datos del borrador base con el texto generado.

    - faltan: números/links del base que el modelo omitió
    - nuevos: números/links que el modelo agregó (posible dato inventado)
    """
    faltan = sorted((_numeros(borrador_base) - _numeros(generado)) |
                    (_urls(borrador_base) - _urls(generado)))
    # El plazo "2 a 3 días hábiles" tiene números de un dígito: se revisa aparte
    plazo = re.compile(r"\d+\s*a\s*\d+\s*d[ií]as\s*h[aá]biles", re.I)
    if plazo.search(borrador_base or "") and not plazo.search(generado or ""):
        faltan.append("plazo de 2 a 3 días hábiles")
    nuevos = sorted((_numeros(generado) - _numeros(borrador_base)) |
                    (_urls(generado) - _urls(borrador_base)))
    otro_idioma = bool(re.search(r"[一-鿿぀-ヿ]", generado or ""))
    if nuevos or otro_idioma:
        estado = "rojo"
    elif faltan:
        estado = "amarillo"
    else:
        estado = "verde"
    return {"estado": estado, "faltan": faltan, "nuevos": nuevos, "otro_idioma": otro_idioma}
