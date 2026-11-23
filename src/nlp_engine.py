"""Motor NLP de Hilo.

- Carga BETO afinado (models/modelo_hilo_final/) en CPU para predecir las 6
  intenciones (multi-etiqueta) a partir de (contexto_previo, mensaje_cliente).
- Si el modelo todavía no existe (o torch/transformers no están instalados),
  usa un clasificador de respaldo por palabras clave.
- Extrae entidades con reglas: tallas, ciudades, códigos de referencia, email...
- Detecta problemas con el courier (alerta_courier).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .data_loader import (
    BASE_DIR,
    detectar_caracteristicas,
    detectar_ciudades,
    detectar_estilos,
    detectar_medidas_cliente,
    normalizar_texto,
)

MODEL_DIR = BASE_DIR / "models" / "modelo_hilo_final"
INTENCIONES = ["saludo_catalogo", "cambio_talla", "envio_plazos",
               "disponibilidad", "precio", "datos_envio", "tienda_fisica"]
METODOS = ("beto", "hibrido", "reglas")
UMBRAL_POR_DEFECTO = 0.5
MAX_LEN_POR_DEFECTO = 128

# --------------------------------------------------------------------------
# Reglas / palabras clave (se comparan sobre texto normalizado: sin tildes)
# --------------------------------------------------------------------------
PALABRAS_COURIER = [
    "no contestan", "no contesta", "no responde", "no responden", "no me contestan",
    "no me llaman", "no me llamaron", "no atienden", "otro numero", "recoger", "recojo",
    "oficina", "courier", "no llega", "no me llego", "no ha llegado", "aun no llega",
    "todavia no llega", "sigue sin llegar", "donde esta mi pedido", "numero de guia",
    "seguimiento",
]

PALABRAS_INTENCION = {
    "saludo_catalogo": [
        "catalogo", "que modelos", "modelos tienen", "como comprar", "como hago",
        "como hacer pedido", "como pido",
    ],
    # "¿Tienen tienda física?", "¿tienen tienda en Cochabamba?", "¿puedo ir a probarme?"
    "tienda_fisica": [
        "tienda fisica", "local", "locales", "tienen tienda", "tiene tienda", "su tienda",
        "la tienda", "donde es la tienda", "donde queda", "donde quedan", "donde estan",
        "donde se encuentran", "ubicados", "ubicacion", "ubicada", "sucursal", "sucursales",
        "showroom", "punto de venta", "puedo ir", "ir a ver", "ir a probarme", "pasar a ver",
        "acercarme", "medirme", "probarme", "en persona", "atienden en", "de que ciudad son",
        "de que departamento", "de donde son", "solo online", "solo por internet",
        "venta online", "tienda online", "solo virtual", "fisica", "por internet",
        "donde los encuentro", "donde los puedo encontrar", "donde puedo encontrar",
    ],
    "cambio_talla": [
        "cambio", "cambiar", "cambia", "queda grande", "queda chico", "queda pequeno",
        "me queda", "no me queda", "no me hace", "ajustado", "apretado", "devolver",
        "devolucion", "otra talla",
    ],
    "envio_plazos": [
        "cuando llega", "cuanto tarda", "cuanto demora", "demora", "tarda", "cuantos dias",
        "dias habiles", "en cuantos dias",
        "envio", "envian", "envios", "entrega", "delivery", "domicilio", "flota",
        "courier", "recoger", "llega", "llegaria", "llegara",
    ],
    "disponibilidad": [
        "disponible", "disponibilidad", "stock", "quedan", "este modelo", "ese modelo",
        "en talla", "en numero", "lo tienen", "la tienen", "tienen en", "tiene en",
        "hay en", "talla",
    ],
    "precio": [
        "precio", "cuanto cuesta", "cuanto sale", "cuanto vale", "cto sale", "a como",
        "costo", "cuesta", "vale", "descuento", "oferta", "promocion", "a cuanto",
        "cuanto es", "cuanto esta", "cuanto estan", "que precio", "precios", "rebaja", "rebajas",
    ],
    "datos_envio": [
        "quiero", "quisiera pedir", "quisiera adquirir", "quisiera comprar", "comprar",
        "hacer pedido", "hacer mi pedido", "realizar pedido", "realizar la compra",
        "hacer un pedido", "deseo", "me lo envia", "me envia qr", "qr", "lo quiero",
        "pedir",
    ],
}

SALUDOS = ["hola", "holaa", "holaaa", "buenas", "buenos dias", "buen dia",
           "buenas tardes", "buenas noches", "que tal"]

PALABRAS_DIRECCION = [
    "calle", "av", "avenida", "zona", "barrio", "edificio", "edif", "nro", "condominio",
    "urbanizacion", "urb", "entre", "esquina", "esq", "piso", "casa", "pasaje", "manzano",
]

PALABRAS_ACHICAR = ["queda grande", "me queda grande", "muy grande", "suelto", "holgado"]
PALABRAS_AGRANDAR = ["queda chico", "queda pequeno", "muy chico", "muy pequeno",
                     "ajustado", "apretado", "no me entra", "no me cierra"]

# "¿Qué me recomiendas?", "¿cuál me queda mejor?", "no sé cuál elegir"
RE_RECOMENDACION = re.compile(
    r"recomiend|recomendac|recomendar|sugier|sugeren|sugerir|aconsej"
    r"|(?:que|cual) (?:me )?(?:quedaria|queda mejor|sirve|conviene|favorece|va mejor)"
    r"|cual (?:es (?:el )?mejor|elijo|escojo|me llevo|me compro)|no se cual|no se que (?:modelo|talla)"
    r"|ayudame a (?:elegir|escoger)|indecisa|que modelo (?:me|para)")
# "¿Qué medidas tiene?", "¿cuánto mide de largo?", "¿qué talla soy?"
RE_PIDE_MEDIDAS = re.compile(
    r"medidas?|cuanto mide|que mide|mide de|largo|botapie|ruedo|tabla de tallas|guia de tallas"
    r"|(?:que|cual|cuál) talla (?:soy|uso|pido|seria|me corresponde|tomo|me recomiendas|deberia)")
# Palabras que sí indican un cambio (no solo "¿cuál me queda mejor?")
RE_CAMBIO_FUERTE = re.compile(r"cambi|devol|grande|chic[oa]|pequen|apretad|ajustad|suelt|"
                              r"no me entra|no me cierra|no me queda")

RE_TALLA = re.compile(r"(?<![\d.,])(3[4-9]|4[0-8])(?![\d.,%])")
RE_REFERENCIA = re.compile(r"(?<!\d)(\d{6})(?!\d)")
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
RE_LISTA_NUMERADA = re.compile(r"(?:^|\s)[1-6](?:️?⃣|\s*[.)\-]+)")
RE_PLACEHOLDER_DATOS = re.compile(r"\[(NOMBRE|DIRECCION|EMAIL)\]")


def _contiene(texto_norm: str, frase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(frase)}(?![a-z0-9])", texto_norm) is not None


def extraer_entidades(mensaje: str) -> dict:
    """Entidades básicas del mensaje mediante regex y reglas."""
    texto = mensaje or ""
    norm = normalizar_texto(texto)

    medidas = detectar_medidas_cliente(texto)  # "cintura 78 cm" no es una talla
    tallas: list[int] = []
    for m in RE_TALLA.finditer(texto):
        if int(m.group(1)) in {int(v) for v in medidas.values()}:
            continue
        resto = texto[m.end(): m.end() + 8].lower()
        if re.match(r"\s*(bs|bolivianos|anos|años|min|dias|días|hrs?)\b", resto):
            continue
        t = int(m.group(1))
        if t not in tallas:
            tallas.append(t)

    referencias = list(dict.fromkeys(RE_REFERENCIA.findall(texto)))
    emails = RE_EMAIL.findall(texto)
    lista = len(RE_LISTA_NUMERADA.findall(texto)) >= 2
    direccion = any(_contiene(norm, p) for p in PALABRAS_DIRECCION)
    placeholders = bool(RE_PLACEHOLDER_DATOS.search(texto))

    ajuste = None
    if any(_contiene(norm, p) for p in PALABRAS_ACHICAR):
        ajuste = "achicar"
    elif any(_contiene(norm, p) for p in PALABRAS_AGRANDAR):
        ajuste = "agrandar"

    return {
        "tallas": tallas,
        "ciudades": [c["ciudad"] for c in detectar_ciudades(texto)],
        "estilos": detectar_estilos(texto),  # "wide leg", "mom fit", "skinny"…
        "caracteristicas": detectar_caracteristicas(texto),  # "tiro alto", "negro", "stretch"…
        "referencias": referencias,
        "emails": emails,
        "tiene_lista_numerada": lista,
        "tiene_direccion": direccion,
        "tiene_datos_envio": bool(emails) or lista or direccion or placeholders,
        "ajuste_talla": ajuste,
        "pide_recomendacion": bool(RE_RECOMENDACION.search(norm)),
        "pide_medidas": bool(RE_PIDE_MEDIDAS.search(norm)) or bool(medidas),
        "medidas_cliente": medidas,                  # {"Cintura": 78.0, ...}
        "cambio_explicito": bool(RE_CAMBIO_FUERTE.search(norm)),
    }


def detectar_alerta_courier(mensaje: str, contexto: str = "") -> tuple[bool, list[str]]:
    norm = normalizar_texto(mensaje)
    hallazgos = [p for p in PALABRAS_COURIER if _contiene(norm, p)]
    return bool(hallazgos), hallazgos


def clasificar_por_palabras_clave(contexto: str, mensaje: str) -> dict[str, float]:
    """Clasificador de respaldo: puntaje por coincidencias de palabras clave."""
    norm = normalizar_texto(mensaje)
    ent = extraer_entidades(mensaje)
    puntos = {i: sum(_contiene(norm, p) for p in PALABRAS_INTENCION[i]) for i in INTENCIONES}

    if ent["tallas"] and not puntos["cambio_talla"] and not puntos["datos_envio"]:
        puntos["disponibilidad"] += 1
    if ent["referencias"] and not puntos["cambio_talla"]:
        puntos["disponibilidad"] += 1
    # "¿tienes wide leg?": preguntar por un estilo es preguntar qué modelos hay
    if (ent["estilos"] or ent["caracteristicas"]) and not puntos["cambio_talla"]:
        puntos["disponibilidad"] += 1
    # "¿Cuál me queda mejor?" / "¿qué talla soy?" piden consejo, no un cambio
    if (ent["pide_recomendacion"] or ent["pide_medidas"]) and not ent["cambio_explicito"]:
        puntos["cambio_talla"] = 0
        puntos["disponibilidad"] += 1
    # "quiero hacer el cambio" no es un pedido nuevo
    if puntos["cambio_talla"] and not ent["tiene_datos_envio"]:
        puntos["datos_envio"] = 0
    # "¿dónde queda la tienda? quiero ir" pregunta por el local, no manda datos de envío
    if puntos["tienda_fisica"] and not ent["emails"] and not ent["tiene_lista_numerada"]:
        puntos["datos_envio"] = 0
        ent = {**ent, "tiene_datos_envio": False}
    if ent["tiene_datos_envio"]:
        puntos["datos_envio"] += 2
    # Un saludo solo es intención si no hay otra consulta y la conversación recién empieza
    if not any(puntos.values()) and not (contexto or "").strip() \
            and any(_contiene(norm, s) for s in SALUDOS):
        puntos["saludo_catalogo"] = 1
    # En medio de una conversación, "local/tienda" sin pedir catálogo no es saludo
    if (contexto or "").strip() and puntos["saludo_catalogo"] and not _contiene(norm, "catalogo"):
        puntos["saludo_catalogo"] = 0

    tabla = {0: 0.05, 1: 0.6, 2: 0.8}
    probs = {i: tabla.get(p, 0.92) for i, p in puntos.items()}
    if max(probs.values()) < UMBRAL_POR_DEFECTO:
        defecto = "saludo_catalogo" if not (contexto or "").strip() else "disponibilidad"
        probs[defecto] = 0.5
    return probs


class NLPEngine:
    """Predice intenciones con BETO (o reglas) y extrae entidades."""

    def __init__(self, model_dir: Path | str = MODEL_DIR, forzar_fallback: bool = False):
        self.model_dir = Path(model_dir)
        self.modo = "fallback"
        self.error_carga: str | None = None
        self.etiquetas = list(INTENCIONES)
        self.umbral = UMBRAL_POR_DEFECTO
        self.max_len = MAX_LEN_POR_DEFECTO
        self.tokenizer = None
        self.model = None
        if not forzar_fallback and (self.model_dir / "config.json").exists():
            try:
                self._cargar_modelo()
                self.modo = "beto"
            except Exception as e:  # noqa: BLE001 - cualquier fallo -> respaldo
                self.error_carga = f"{type(e).__name__}: {e}"
        elif not forzar_fallback:
            self.error_carga = f"No se encontró {self.model_dir} (config.json)."

    def _cargar_modelo(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        meta_path = self.model_dir / "labels.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.etiquetas = meta.get("labels", self.etiquetas)
            self.umbral = float(meta.get("umbral", self.umbral))
            self.max_len = int(meta.get("max_len", self.max_len))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(torch.device("cpu")).eval()
        if not meta_path.exists() and self.model.config.id2label:
            n = self.model.config.num_labels
            self.etiquetas = [self.model.config.id2label[i] for i in range(n)]
        self._torch = torch

    def _predecir_beto(self, contexto: str, mensaje: str) -> dict[str, float | None]:
        """Probabilidad de BETO por intención; None si BETO no se entrenó con esa intención
        (p. ej. 'tienda_fisica' con un modelo de 6 intenciones)."""
        # Mismo formato que en train_colab.py: par (contexto_previo, mensaje_cliente)
        enc = self.tokenizer(contexto or "", mensaje or "", truncation=True,
                             max_length=self.max_len, return_tensors="pt")
        with self._torch.no_grad():
            logits = self.model(**enc).logits[0]
        probs = self._torch.sigmoid(logits).tolist()
        p = {e: float(v) for e, v in zip(self.etiquetas, probs)}
        return {i: p.get(i) for i in INTENCIONES}

    def predecir_intenciones(self, contexto: str, mensaje: str) -> dict[str, float]:
        if self.modo != "beto":
            return clasificar_por_palabras_clave(contexto, mensaje)
        reglas = clasificar_por_palabras_clave(contexto, mensaje)
        p = self._predecir_beto(contexto, mensaje)
        return {i: reglas[i] if v is None else v for i, v in p.items()}

    def intenciones_no_entrenadas(self) -> list[str]:
        """Intenciones que el BETO cargado no conoce (las decide solo la regla)."""
        return [i for i in INTENCIONES if i not in self.etiquetas] if self.modo == "beto" else []

    def metodos_disponibles(self) -> list[str]:
        return list(METODOS) if self.modo == "beto" else ["reglas"]

    def analizar(self, contexto: str, mensaje: str, metodo: str | None = None) -> dict:
        """metodo: 'beto', 'hibrido' (promedio BETO + reglas) o 'reglas'.

        Por defecto usa BETO si está cargado; si no, reglas.
        """
        p_reglas = clasificar_por_palabras_clave(contexto, mensaje)
        p_beto = self._predecir_beto(contexto, mensaje) if self.modo == "beto" else None
        if metodo not in METODOS or (metodo != "reglas" and p_beto is None):
            metodo = "beto" if p_beto is not None else "reglas"

        if metodo == "beto":
            # Intención que este BETO no conoce: decide la regla
            probs = {i: p_reglas[i] if p_beto[i] is None else p_beto[i] for i in INTENCIONES}
            umbral = self.umbral
        elif metodo == "hibrido":
            probs = {i: p_reglas[i] if p_beto[i] is None else (p_beto[i] + p_reglas[i]) / 2
                     for i in INTENCIONES}
            umbral = UMBRAL_POR_DEFECTO
        else:
            probs, umbral = p_reglas, UMBRAL_POR_DEFECTO

        detectadas = [i for i, p in sorted(probs.items(), key=lambda x: -x[1]) if p >= umbral]
        if not detectadas:  # siempre al menos una intención
            detectadas = [max(probs, key=probs.get)]
        alerta, frases = detectar_alerta_courier(mensaje, contexto)
        return {
            "intenciones": detectadas,
            "probabilidades": probs,
            "prob_beto": p_beto,
            "prob_reglas": p_reglas,
            "entidades": extraer_entidades(mensaje),
            "alerta_courier": alerta,
            "frases_courier": frases,
            "modo": metodo,
            "umbral": umbral,
        }
