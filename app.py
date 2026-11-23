"""Hilo · asistente de atención al cliente de Merci Collection (Streamlit).

Ejecutar desde la carpeta hilo-merci:
    streamlit run app.py

Pestañas:
    Responder  formulario + borradores en burbujas estilo WhatsApp
    Análisis   intenciones (BETO vs. reglas), entidades, datos, verificaciones

Velocidad: los modelos (BETO, MiniLM), la tienda y Qwen se cargan EN PARALELO y en segundo
plano al abrir la app, así la página aparece al instante. La plantilla es instantánea; Qwen
solo redacta cuando se elige la pestaña "Qwen".

Logo: assets/logo.png (o .jpg / .svg). La versión anterior de la interfaz está en app_anterior.py.
"""
from __future__ import annotations

import html
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Si MiniLM ya está descargado, no consultar internet al cargarlo (ahorra segundos)
if (Path.home() / ".cache" / "huggingface" / "hub" /
        "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd
import streamlit as st

from src import redactor_llm as llm
from src import tienda_api
from src.data_loader import estado_stock
from src.draft_generator import DraftGenerator, dividir_en_mensajes, recopilar_datos, verificar_borrador
from src.nlp_engine import INTENCIONES, NLPEngine

ASSETS = Path(__file__).resolve().parent / "assets"
LOGO = next((p for p in (ASSETS / "logo.png", ASSETS / "logo.jpg", ASSETS / "logo.jpeg",
                         ASSETS / "logo.svg") if p.exists()), None)

st.set_page_config(page_title="Luxe · Merci Collection", page_icon=str(LOGO) if LOGO else None,
                   layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ estilo (igual que las diapositivas)
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;800;900&family=JetBrains+Mono:wght@400;700&display=swap');
:root{ --paper:#ECEAE3; --ink:#0E0E0E; --green:#00A884; --lime:#C8FF3D; --red:#FF3B1F; --mute:#6B6A66;
  --sans:"Archivo","Arial Black","Helvetica Neue",Arial,sans-serif;
  --mono:"JetBrains Mono",Consolas,"Courier New",monospace; }
html, body, [class*="css"], .stApp { font-family: var(--sans); }
.stApp { background: var(--paper); }
.block-container { padding-top: 3.2rem; max-width: 1320px; }
footer, #MainMenu { visibility: hidden; }

/* encabezado */
.hilo-head { display:flex; align-items:stretch; gap:0; border:3px solid var(--ink); background:var(--paper);
  margin-bottom: 18px; box-shadow: 8px 8px 0 var(--ink); }
.hilo-head .izq { padding: 14px 22px 10px; border-right:3px solid var(--ink); }
.hilo-kicker { font: 700 12px/1 var(--mono); letter-spacing:.14em; text-transform:uppercase; color:var(--ink); }
.hilo-mark { font: 900 64px/0.9 var(--sans); letter-spacing:-.05em; color:var(--ink); margin-top:8px; }
.hilo-head .der { flex:1; position:relative; overflow:hidden; display:flex; align-items:flex-end;
  justify-content:flex-end; padding: 12px 18px; }
/* trama de pixeles tipo "umbral": puntos que se vuelven sólidos hacia la derecha */
.hilo-head .der::before { content:""; position:absolute; inset:0;
  background-image: radial-gradient(var(--ink) 1.6px, transparent 1.8px), radial-gradient(var(--green) 1.6px, transparent 1.8px);
  background-size: 8px 8px, 8px 8px; background-position: 0 0, 4px 4px;
  -webkit-mask-image: linear-gradient(90deg, transparent 0%, rgba(0,0,0,.25) 35%, #000 85%);
          mask-image: linear-gradient(90deg, transparent 0%, rgba(0,0,0,.25) 35%, #000 85%); }
.hilo-head .der span { position:relative; background:var(--ink); color:var(--lime); font:700 13px var(--mono);
  letter-spacing:.1em; text-transform:uppercase; padding:8px 12px; }

/* etiquetas, títulos */
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { font: 700 12px var(--mono) !important;
  letter-spacing:.12em; text-transform:uppercase; color: var(--ink) !important; }
h3 { font-family: var(--sans) !important; font-weight: 900 !important; text-transform: uppercase;
  letter-spacing: -.01em; font-size: 1.25rem !important; border-top: 3px solid var(--ink); padding-top: 10px !important; }
[data-testid="stCaptionContainer"] { font-family: var(--mono); font-size: 12px; color: var(--mute); }

/* campos de texto */
[data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"] {
  border-radius: 0 !important; border: 3px solid var(--ink) !important; background: #F7F6F1 !important; }
[data-testid="stTextInputRootElement"]:focus-within, [data-testid="stTextAreaRootElement"]:focus-within {
  box-shadow: 5px 5px 0 var(--green) !important; }
[data-testid="stTextInputRootElement"] *, [data-testid="stTextAreaRootElement"] * { background: transparent !important;
  border-radius: 0 !important; }
input, textarea { font-family: "Segoe UI", Arial, sans-serif !important; color: var(--ink) !important; }

/* botones */
.stButton button, [data-testid="stFormSubmitButton"] button, [data-testid="stPopover"] button {
  border-radius: 0 !important; border: 3px solid var(--ink) !important; font: 800 14px var(--sans) !important;
  text-transform: uppercase; letter-spacing: .04em; transition: transform .08s, box-shadow .08s; }
button p { font: inherit !important; color: inherit !important; }
[data-testid="stBaseButton-primaryFormSubmit"], [data-testid="stBaseButton-primary"] {
  background: var(--ink) !important; color: var(--paper) !important; box-shadow: 6px 6px 0 var(--green); }
[data-testid="stBaseButton-primaryFormSubmit"]:hover, [data-testid="stBaseButton-primary"]:hover {
  transform: translate(-2px,-2px); box-shadow: 8px 8px 0 var(--green); color: var(--lime) !important; }
[data-testid="stBaseButton-primaryFormSubmit"]:active, [data-testid="stBaseButton-primary"]:active {
  transform: translate(4px,4px); box-shadow: 0 0 0 var(--green); }
[data-testid="stBaseButton-secondary"], [data-testid="stPopover"] button {
  background: var(--paper) !important; color: var(--ink) !important; }
[data-testid="stBaseButton-secondary"]:hover, [data-testid="stPopover"] button:hover { background: var(--lime) !important; }

/* selector Plantilla | Qwen */
[data-testid="stButtonGroup"] button { border-radius: 0 !important; border: 3px solid var(--ink) !important;
  font: 700 13px var(--mono) !important; text-transform: uppercase; letter-spacing: .1em; margin-right: -3px; }
[data-testid="stBaseButton-segmented_controlActive"], [data-testid="stButtonGroup"] button[aria-checked="true"] {
  background: var(--ink) !important; color: var(--lime) !important; }
[data-testid="stButtonGroup"] button[aria-checked="false"] { background: var(--paper) !important; color: var(--ink) !important; }
[data-testid="stButtonGroup"] button:hover { background: var(--lime) !important; color: var(--ink) !important; }

/* pestañas */
[data-testid="stTabs"] [role="tablist"] { gap: 0; border-bottom: 3px solid var(--ink); box-shadow: none !important; }
[data-testid="stTab"] { font: 700 13px var(--mono) !important; text-transform: uppercase; letter-spacing: .12em;
  padding: 10px 20px !important; border: 3px solid var(--ink) !important; border-bottom: 0 !important;
  margin-right: -3px; background: var(--paper); border-radius: 0 !important; color: var(--ink); }
[data-testid="stTab"] p { font: inherit !important; color: inherit !important; }
[data-testid="stTab"][aria-selected="true"] { background: var(--ink) !important; color: var(--lime) !important; }
[data-testid="stTab"]::after, [data-testid="stTab"]::before { display: none !important; }

/* panel de chat estilo WhatsApp */
.st-key-chat { background: #EFEAE2; border: 3px solid var(--ink); box-shadow: 10px 10px 0 var(--ink);
  padding: 20px 20px 12px 20px; }
.burbuja-cliente { background: #fff; color: var(--ink); border: 2px solid var(--ink); border-radius: 4px;
  padding: 10px 14px; max-width: 82%; white-space: pre-wrap; font: 15px/1.4 "Segoe UI", Arial, sans-serif;
  margin-bottom: 12px; }
.vacio { font: 700 13px var(--mono); letter-spacing:.1em; text-transform: uppercase; color: var(--mute);
  padding: 60px 0; text-align: center; }
.st-key-chat [data-testid="stCode"] { max-width: 86%; margin-left: auto; }
.st-key-chat [data-testid="stCode"] pre, .st-key-chat [data-testid="stCode"] code {
  background: #D9FDD3 !important; color: var(--ink) !important;
  font-family: "Segoe UI", Arial, sans-serif !important; font-size: 15px !important; white-space: pre-wrap !important; }
.st-key-chat [data-testid="stCode"] pre { border: 2px solid var(--ink) !important; border-radius: 4px !important; }

/* línea de estado (como el "Verificado" de las diapositivas) */
.estado { display: flex; width: fit-content; align-items: center; gap: 9px; font: 700 12px/1.3 var(--mono);
  letter-spacing: .06em; text-transform: uppercase; padding: 8px 12px; margin: 4px 0 10px 0; border: 3px solid var(--ink); }
.estado::before { content: ""; flex: none; width: 10px; height: 10px; border-radius: 50%; }
.estado.ok { background: var(--ink); color: var(--lime); } .estado.ok::before { background: var(--lime); }
.estado.mal { background: var(--red); color: #fff; } .estado.mal::before { background: #fff; }
.estado.duda { background: var(--lime); color: var(--ink); } .estado.duda::before { background: var(--ink); }
.estado.carga { background: var(--paper); color: var(--mute); } .estado.carga::before { background: var(--mute);
  animation: parpadeo 1s infinite; }
@keyframes parpadeo { 50% { opacity: .2; } }
.dato { font-size: 0.95rem; margin: 3px 0; } .dato b { font: 700 12px var(--mono); text-transform: uppercase;
  letter-spacing: .06em; color: var(--ink); margin-right: 4px; }

/* barra lateral, barras y tablas */
[data-testid="stSidebar"] { background: #E3E0D7; border-right: 3px solid var(--ink); }
[data-testid="stProgress"] div { border-radius: 0 !important; }
[data-testid="stDataFrame"] { border: 3px solid var(--ink); }
[data-testid="stPopoverBody"] { border: 3px solid var(--ink) !important; border-radius: 0 !important; box-shadow: 8px 8px 0 var(--ink); }
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ carga en segundo plano
def _preparar_tienda() -> dict:
    est = estado_stock()          # descarga el catálogo (~7 s la primera vez)
    tienda_api.medidas(esperar=False)
    tienda_api.probar_pedidos()
    return est


def _preparar_motor() -> NLPEngine:
    m = NLPEngine()
    m.analizar("", "hola, tienen el 240904 en 38?", "hibrido")  # la 1.ª predicción es más lenta
    return m


def _preparar_generador(f_tienda) -> DraftGenerator:
    g = DraftGenerator()
    try:  # precalcula los embeddings de las descripciones: la recomendación sale al instante
        f_tienda.result()
        g.recomendar("qué me recomiendas", [], [], [], {})
    except Exception:  # noqa: BLE001 - solo es un calentamiento
        pass
    return g


def _preparar_qwen() -> bool:
    try:
        if llm.estado_ollama()["modelo_descargado"]:
            llm.precalentar()
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


class Carga:
    """Carga en segundo plano, una sola vez por servidor.

    Empieza DESPUÉS de que la página se dibujó (cargar BETO/MiniLM ocupa el procesador de
    Python y, si arranca antes, la página no aparece hasta que termina). La tienda (red) y
    Qwen (otro proceso: Ollama) cargan en paralelo; BETO y luego MiniLM, uno detrás del otro.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.F: dict | None = None

    def iniciar(self) -> dict:
        with self._lock:
            if self.F is None:
                ex = ThreadPoolExecutor(max_workers=3, thread_name_prefix="hilo")
                f_tienda = ex.submit(_preparar_tienda)
                f_qwen = ex.submit(_preparar_qwen)
                f_motor = ex.submit(_preparar_motor)
                f_gen = ex.submit(lambda: (f_motor.exception(), _preparar_generador(f_tienda))[1])
                self.F = {"tienda": f_tienda, "motor": f_motor, "generador": f_gen, "qwen": f_qwen}
        return self.F

    def hecho(self, nombre: str) -> bool:
        return self.F is not None and self.F[nombre].done()

    def esperar(self, nombre: str):
        return self.iniciar()[nombre].result()


@st.cache_resource(show_spinner=False)
def carga() -> Carga:
    return Carga()


@st.cache_data(ttl=15, show_spinner=False)
def estado_ollama_cacheado() -> dict:
    return llm.estado_ollama()


C = carga()
NOMBRES_METODO = {"beto": "BETO", "hibrido": "Híbrido (BETO + reglas)", "reglas": "Palabras clave"}
# Clasificador fijo: BETO entrenado + reglas (si BETO no cargara, analizar() usa solo las reglas)
METODO = "hibrido"


def listo(nombre: str):
    """El recurso si ya cargó (o None si sigue cargando o falló)."""
    if not C.hecho(nombre):
        return None
    try:
        return C.esperar(nombre)
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ utilidades
def sin_icono(texto: str) -> str:
    """'❌ Talla 46…' -> 'Talla 46…' (la interfaz va sin emojis)."""
    return re.sub(r"^[^\w¡¿(\"«]+", "", str(texto)).strip()


def dato(etiqueta: str, valor) -> None:
    if isinstance(valor, (list, tuple)):
        valor = ", ".join(map(str, valor)) if valor else "—"
    st.markdown(f"<div class='dato'><b>{html.escape(etiqueta)}</b> "
                f"{html.escape(str(valor if valor not in (None, '') else '—'))}</div>",
                unsafe_allow_html=True)


def linea_estado(clase: str, texto: str) -> None:
    st.markdown(f"<div class='estado {clase}'>{html.escape(texto)}</div>", unsafe_allow_html=True)


def burbujas(texto: str) -> None:
    for m in dividir_en_mensajes(texto):
        try:
            st.code(m, language=None, wrap_lines=True)
        except TypeError:  # versiones antiguas de Streamlit
            st.code(m, language=None)


def estado_borrador(texto: str, ref, base: str | None = None) -> None:
    """Una sola línea: verificado / revisar (el detalle está en la pestaña Análisis)."""
    v = verificar_borrador(texto, ref)
    f = llm.revisar_fidelidad(base, texto) if base is not None else {"estado": "verde"}
    if v["estado"] == "rojo":
        malos = [sin_icono(d) for d in v["detalles"] if d.startswith("❌")]
        linea_estado("mal", "Revisar: " + (malos[0] if malos else "no coincide con el stock"))
    elif f["estado"] == "rojo":
        nuevos = ", ".join(f.get("nuevos") or []) or "texto en otro idioma"
        linea_estado("mal", f"Revisar: datos que no están en la tienda ({nuevos})")
    elif f["estado"] == "amarillo":
        linea_estado("duda", "Falta: " + ", ".join(f["faltan"]))
    elif v["estado"] == "amarillo":
        linea_estado("duda", "Sin datos de stock para verificar")
    else:
        linea_estado("ok", "Verificado con el stock de la tienda")


def guardar_edicion(clave_widget: str, clave_texto: str) -> None:
    st.session_state[clave_texto] = st.session_state[clave_widget]


def pedir_otra_version() -> None:
    st.session_state["llm_pendiente"] = True
    st.session_state["llm_variar"] = "texto_llm" in st.session_state
    st.session_state["llm_reintentado"] = False
    for k in ("llm_aviso", "llm_error", "edit_llm", "llm_quitadas"):
        st.session_state.pop(k, None)


# ------------------------------------------------------------------ barra lateral (configuración)
motor_listo = listo("motor")
with st.sidebar:
    if LOGO:
        st.logo(str(LOGO))
    st.subheader("Configuración")
    estado_llm = estado_ollama_cacheado()
    usar_llm = st.toggle("Redactar con Qwen", value=estado_llm["modelo_descargado"],
                         disabled=not estado_llm["modelo_descargado"])

    st.subheader("Estado")
    gen_listo = listo("generador")
    estados = []
    if motor_listo:
        estados.append(("ok" if motor_listo.modo == "beto" else "duda",
                        f"BETO · {len(motor_listo.etiquetas)} intenciones" if motor_listo.modo == "beto"
                        else "Solo palabras clave"))
    else:
        estados.append(("carga", "Cargando BETO…"))
    if not estado_llm["modelo_descargado"]:
        estados.append(("mal", "Qwen no disponible"))
    else:
        estados.append(("ok", f"Qwen 2.5 · {llm.MODELO_LLM}") if C.hecho("qwen")
                       else ("carga", "Cargando Qwen…"))
    if gen_listo:
        estados.append(("ok" if gen_listo.modo_busqueda == "sentence-transformers" else "duda",
                        "MiniLM · búsqueda semántica" if gen_listo.modo_busqueda == "sentence-transformers"
                        else "Búsqueda difflib"))
    else:
        estados.append(("carga", "Cargando MiniLM…"))
    if C.hecho("tienda"):
        est_stock = estado_stock()
        est_ped = tienda_api.probar_pedidos()
        estados.append(("ok" if est_stock["fuente"] == "api" else "duda" if est_stock["fuente"] == "csv" else "mal",
                        f"Stock en vivo · {est_stock['modelos']} modelos · {est_stock['actualizado']}"
                        if est_stock["fuente"] == "api" else "Stock desde stock.csv"
                        if est_stock["fuente"] == "csv" else "Sin stock"))
        estados.append(("ok" if est_ped["ok"] else "duda",
                        "Pedidos en vivo" if est_ped["ok"] else "Pedidos desde pedidos.csv"))
    else:
        estados.append(("carga", "Conectando con la tienda…"))
    for clase, texto in estados:
        linea_estado(clase, texto)
    # Solo recarga los DATOS (tienda y pedidos); los modelos ya cargados se mantienen
    if st.button("Recargar datos de la tienda"):
        tienda_api.limpiar_cache()
        estado_stock()
        for k in list(st.session_state):
            del st.session_state[k]
        st.rerun()

# ------------------------------------------------------------------ encabezado
if LOGO:
    st.image(str(LOGO), width=90)
cargando = not (motor_listo and gen_listo and C.hecho("tienda"))
st.markdown(
    "<div class='hilo-head'><div class='izq'><div class='hilo-kicker'>Merci Collection · "
    "Asistente de respuestas</div><div class='hilo-mark'>Luxe</div></div>"
    f"<div class='der'><span>{'Preparando modelos…' if cargando else 'Listo · datos en vivo'}</span></div></div>",
    unsafe_allow_html=True)



tab_responder, tab_analisis = st.tabs(["Responder", "Análisis"])

# ------------------------------------------------------------------ pestaña Responder
with tab_responder:
    col_form, col_chat = st.columns([1, 1.35], gap="large")

    with col_form:
        with st.form("consulta", border=False):
            telefono = st.text_input("Teléfono")
            ref_manual = st.text_input("Código de referencia", placeholder="Opcional · 240904")
            contexto = st.text_area("Último mensaje de Merci", height=70, placeholder="Opcional")
            mensaje = st.text_area("Mensaje de la clienta", height=130,
                                   placeholder="Hola, ¿tienen este modelo en talla 38?")
            enviado = st.form_submit_button("Generar borrador", type="primary", width="stretch")

    if enviado and mensaje.strip():
        espera = "Preparando los modelos (solo la primera vez)…" if cargando else "Analizando…"
        with st.spinner(espera):
            motor = C.esperar("motor")
            generador = C.esperar("generador")
            analisis = motor.analizar(contexto, mensaje, METODO)
            datos = recopilar_datos(analisis, telefono, ref_manual)
            resultado = generador.generar(analisis, datos, contexto=contexto, mensaje=mensaje)
        st.session_state.update(analisis=analisis, datos=datos, resultado=resultado,
                                texto_plantilla=resultado["borrador"],
                                entrada={"mensaje": mensaje, "contexto": contexto},
                                llm_reintentado=False, llm_pendiente=False)
        for k in ("texto_llm", "llm_error", "llm_aviso", "edit_llm", "edit_plantilla", "llm_quitadas"):
            st.session_state.pop(k, None)

    with col_chat:
        with st.container(key="chat"):
            if "resultado" not in st.session_state:
                st.markdown("<div class='vacio'>El borrador aparecerá aquí</div>", unsafe_allow_html=True)
            else:
                resultado = st.session_state["resultado"]
                datos = st.session_state["datos"]
                entrada = st.session_state["entrada"]
                st.markdown(f"<div class='burbuja-cliente'>{html.escape(entrada['mensaje'])}</div>",
                            unsafe_allow_html=True)

                # La plantilla es instantánea; Qwen (~15 s) solo redacta si se elige su pestaña
                vistas = ["Plantilla", "Qwen"] if usar_llm else ["Plantilla"]
                vista = vistas[0]
                if len(vistas) > 1:
                    vista = st.segmented_control("Borrador", vistas, default=vistas[0],
                                                 key="vista", label_visibility="collapsed") or vistas[0]

                if vista == "Qwen":
                    if "texto_llm" not in st.session_state and not st.session_state.get("llm_error"):
                        st.session_state["llm_pendiente"] = True
                    if st.session_state.get("llm_pendiente"):
                        reintentar = False
                        quitadas: list[str] = []  # frases sin respaldo que el filtro eliminó
                        try:
                            texto = llm.limpiar_salida(st.write_stream(llm.redactar_stream(
                                entrada["mensaje"], entrada["contexto"], resultado["borrador"],
                                resultado["similares"], variar=st.session_state.pop("llm_variar", False),
                                quitadas=quitadas)))
                            fid = llm.revisar_fidelidad(resultado["borrador"], texto)
                            stock_rojo = verificar_borrador(texto, datos.get("ref"))["estado"] == "rojo"
                            perdio_datos = fid["estado"] == "amarillo" and any(
                                re.search(r"\d", q) for q in quitadas)
                            if (fid["estado"] == "rojo" or stock_rojo or perdio_datos) \
                                    and not st.session_state.get("llm_reintentado"):
                                # Inventó datos, perdió alguno o contradice el stock: se redacta otra vez
                                reintentar = True
                                st.session_state.update(llm_reintentado=True, llm_variar=True, llm_aviso=(
                                    "La primera versión " + ("contradecía el stock" if stock_rojo else
                                                             "agregó datos que no estaban en la base"
                                                             if fid["estado"] == "rojo" else
                                                             "omitió datos de la base")
                                    + "; se descartó y se redactó otra."))
                            else:
                                st.session_state["texto_llm"] = texto
                                st.session_state["llm_base"] = resultado["borrador"]
                                st.session_state["llm_quitadas"] = quitadas
                        except Exception as e:  # noqa: BLE001 - Ollama apagado, timeout, etc.
                            st.session_state["llm_error"] = f"{type(e).__name__}: {e}"
                        st.session_state["llm_pendiente"] = reintentar
                        st.rerun()
                    elif st.session_state.get("llm_error"):
                        linea_estado("mal", "Qwen no respondió. Usa la plantilla o reintenta.")
                        st.button("Reintentar", on_click=pedir_otra_version)
                    else:
                        texto = st.session_state["texto_llm"]
                        estado_borrador(texto, datos.get("ref"), st.session_state.get("llm_base", ""))
                        burbujas(texto)
                        b1, b2 = st.columns(2)
                        b1.button("Otra versión", on_click=pedir_otra_version, width="stretch")
                        with b2.popover("Editar", width="stretch"):
                            st.text_area("Editar", value=texto, key="edit_llm", height=320,
                                         label_visibility="collapsed", on_change=guardar_edicion,
                                         args=("edit_llm", "texto_llm"))
                else:
                    texto = st.session_state["texto_plantilla"]
                    estado_borrador(texto, datos.get("ref"))
                    burbujas(texto)
                    with st.popover("Editar", width="stretch"):
                        st.text_area("Editar", value=texto, key="edit_plantilla", height=320,
                                     label_visibility="collapsed", on_change=guardar_edicion,
                                     args=("edit_plantilla", "texto_plantilla"))

# ------------------------------------------------------------------ pestaña Análisis
with tab_analisis:
    if "resultado" not in st.session_state:
        st.markdown("<div class='vacio'>Genera un borrador para ver el análisis</div>",
                    unsafe_allow_html=True)
    else:
        analisis = st.session_state["analisis"]
        datos = st.session_state["datos"]
        resultado = st.session_state["resultado"]
        izq, der = st.columns(2, gap="large")

        with izq:
            st.subheader("Intenciones")
            st.caption(f"{NOMBRES_METODO.get(analisis['modo'], analisis['modo'])} · "
                       f"umbral {analisis['umbral']:.0%}")
            probs = analisis["probabilidades"]
            for i in sorted(INTENCIONES, key=lambda x: -probs.get(x, 0)):
                p = float(probs.get(i, 0))
                marca = "■" if i in analisis["intenciones"] else "□"
                st.progress(min(max(p, 0.0), 1.0), text=f"{marca} {i} · {p:.0%}")

            if analisis.get("prob_beto"):
                st.subheader("BETO vs. palabras clave")
                pb, pr = analisis["prob_beto"], analisis["prob_reglas"]
                st.dataframe(pd.DataFrame({
                    "Intención": INTENCIONES,
                    "BETO": [f"{pb[i]:.0%}" if pb[i] is not None else "—" for i in INTENCIONES],
                    "Palabras clave": [f"{pr[i]:.0%}" for i in INTENCIONES],
                    "Híbrido": [f"{(pb[i] + pr[i]) / 2:.0%}" if pb[i] is not None else f"{pr[i]:.0%}"
                                for i in INTENCIONES],
                }), hide_index=True, width="stretch")

            st.subheader("Entidades")
            ent = analisis["entidades"]
            dato("Tallas", ent["tallas"])
            dato("Ciudades", ent["ciudades"])
            dato("Referencias", ent["referencias"])
            dato("Estilos", ent.get("estilos"))
            dato("Características", ent.get("caracteristicas"))
            dato("Medidas de la clienta", [f"{k.lower()} {v:g} cm" for k, v in
                                          (ent.get("medidas_cliente") or {}).items()])
            dato("Pide recomendación / medidas",
                 " · ".join(x for x, s in (("recomendación", ent.get("pide_recomendacion")),
                                           ("medidas", ent.get("pide_medidas"))) if s) or "no")
            dato("Email", ent["emails"])
            dato("Datos de envío en el mensaje", "sí" if ent["tiene_datos_envio"] else "no")
            if analisis["alerta_courier"]:
                linea_estado("mal", "Alerta courier: " + ", ".join(analisis["frases_courier"]))

            st.subheader("Respuestas reales parecidas")
            for s in resultado["similares"]:
                st.caption(f"Similitud {s['similitud']:.2f} · {s['intenciones']}")
                st.markdown(f"<div class='burbuja-cliente'>{html.escape(str(s['mensaje_cliente']))}</div>",
                            unsafe_allow_html=True)
                st.markdown(f"<div class='burbuja-cliente' style='background:#D9FDD3;margin-left:auto'>"
                            f"{html.escape(str(s['respuesta_real']))}</div>", unsafe_allow_html=True)

        with der:
            st.subheader("Cliente")
            cli = datos["cliente"]
            if cli.get("encontrado"):
                dato("Nombre", cli["nombre"])
                dato("Compras", f"{cli['num_compras']}" + (" · cliente frecuente" if cli["cliente_frecuente"] else ""))
                dato("Ciudad", cli["ciudad"])
                up = cli["ultimo_pedido"]
                dato("Último pedido", f"#{up['numero']} · {up['fecha']} · {up['estado']} · {up['importe']} Bs")
                dato("Fuente", "tienda en vivo" if cli.get("fuente") == "api" else "pedidos.csv")
                st.dataframe(pd.DataFrame(up["articulos"])[["modelo", "ref", "talla"]],
                             hide_index=True, width="stretch")
            elif cli.get("telefono"):
                st.caption(f"Sin pedidos con el teléfono {cli['telefono']}.")
            else:
                st.caption("Sin teléfono.")

            sm = datos["stock_modelo"]
            if datos.get("ref"):
                st.subheader(f"Stock · Ref {datos['ref']}")
                if not sm["archivo_existe"]:
                    st.caption("Sin conexión con la tienda ni stock.csv.")
                elif not sm["modelo_encontrado"]:
                    st.caption("La referencia no aparece en la tienda.")
                else:
                    dato("Modelo", sm["nombre"])
                    dato("Precio", f"{sm['precio']:g} Bs" if sm["precio"] else "—")
                    dato("Oferta", f"{sm['precio_oferta']:g} Bs" if sm["precio_oferta"] else "—")
                    dato("Tallas disponibles", sm["tallas_disponibles"])
                    dato("Tallas agotadas", sm["tallas_agotadas"])
                    dato("Origen", f"{datos['origen_ref']} · "
                                   f"{'tienda en vivo' if sm.get('fuente') == 'api' else 'stock.csv'}")

            me = datos.get("modelos_estilo")
            if me:
                filtro = " / ".join(me["estilos"]) or "jeans"
                if me.get("caracteristicas"):
                    filtro += f" ({', '.join(me['caracteristicas'])})"
                if me["tallas"]:
                    filtro += f" · talla {', '.join(map(str, me['tallas']))}"
                st.subheader(f"Modelos · {filtro}")
                st.caption(f"{me['total']} con stock")
                if me["modelos"]:
                    st.dataframe(pd.DataFrame([{
                        "Ref": m["ref"], "Modelo": m["nombre"], "Precio": m["precio"],
                        "Oferta": m["precio_oferta"], "Tallas": ", ".join(map(str, m["tallas_disponibles"])),
                    } for m in me["modelos"]]), hide_index=True, width="stretch")

            rec = datos.get("recomendacion")
            if rec:
                criterio = [*rec["estilos"], *rec["caracteristicas"]]
                if rec["tallas"]:
                    criterio.append(f"talla {', '.join(map(str, rec['tallas']))}")
                if rec["medidas_cliente"]:
                    criterio += [f"{k.lower()} {v:g} cm" for k, v in rec["medidas_cliente"].items()]
                st.subheader("Recomendación" + (f" · {', '.join(criterio)}" if criterio else ""))
                st.caption(f"{rec['total']} modelos con stock cumplen lo pedido · ordenados por "
                           "similitud semántica (MiniLM) entre el mensaje y la descripción de la tienda"
                           + (" · no había del estilo pedido, se buscó entre todos" if rec["relajado"] else ""))
                if rec["modelos"]:
                    st.dataframe(pd.DataFrame([{
                        "Ref": m["ref"], "Modelo": m["nombre"], "Similitud": m["similitud"],
                        "Talla": m.get("talla_rec") or ", ".join(map(str, m["tallas_disponibles"])),
                        "Por qué (texto de la tienda)": m.get("razon", ""),
                    } for m in rec["modelos"]]), hide_index=True, width="stretch")

            med = datos.get("medidas")
            if med:
                st.subheader(f"Medidas · Ref {med['ref']}")
                if med["tabla"] is None:
                    st.caption("No se pudo leer la tabla de la tienda.")
                elif not med["tabla"]:
                    st.caption("La tienda todavía no publicó la tabla de este modelo.")
                else:
                    st.caption("Medidas de la prenda en cm, leídas de la ficha del producto en la web")
                    st.dataframe(pd.DataFrame([{"Talla": t, **fila} for t, fila in sorted(med["tabla"].items())]),
                                 hide_index=True, width="stretch")
                    tp = med.get("talla_por_medidas")
                    if tp and tp.get("talla"):
                        dato("Talla según sus medidas", f"{tp['talla']}"
                             + ("" if tp["disponible"] else f" (agotada → {tp['alternativa'] or 'sin alternativa'})"))

            if datos.get("courier"):
                c = datos["courier"]
                st.subheader(f"Courier · {c['ciudad']}")
                for k, etiqueta in (("encargado", "Encargado"), ("telefono_fijo", "Teléfono"),
                                    ("celular_whatsapp", "WhatsApp"), ("direccion", "Dirección"),
                                    ("horario", "Horario")):
                    dato(etiqueta, c.get(k))

            if resultado["notas"]:
                st.subheader("Notas internas")
                for n in resultado["notas"]:
                    st.markdown(f"<div class='dato'>{html.escape(sin_icono(n))}</div>",
                                unsafe_allow_html=True)

            st.subheader("Verificación")
            if st.session_state.get("llm_aviso"):
                st.caption(st.session_state["llm_aviso"])
            borradores = [("Plantilla", st.session_state.get("texto_plantilla", ""), None)]
            if "texto_llm" in st.session_state:
                borradores.insert(0, ("Qwen", st.session_state["texto_llm"], st.session_state.get("llm_base", "")))
            for nombre, texto, base in borradores:
                st.markdown(f"**{nombre}**")
                v = verificar_borrador(texto, datos.get("ref"))
                for d in v["detalles"]:
                    clase = "mal" if d.startswith("❌") else "duda" if d.startswith("⚠") else "ok"
                    linea_estado(clase, sin_icono(d))
                if base is not None:
                    f = llm.revisar_fidelidad(base, texto)
                    if f["estado"] == "verde":
                        linea_estado("ok", "Fidelidad: conserva todos los datos de la plantilla")
                    if f["nuevos"]:
                        linea_estado("mal", "Datos nuevos: " + ", ".join(f["nuevos"]))
                    if f["faltan"]:
                        linea_estado("duda", "Omitió: " + ", ".join(f["faltan"]))
                    quitadas = st.session_state.get("llm_quitadas") or []
                    if quitadas:
                        linea_estado("duda", f"Frases eliminadas por no estar en los datos ({len(quitadas)}):")
                        for q in quitadas:
                            st.markdown(f"<div class='dato'><s>{html.escape(q)}</s></div>",
                                        unsafe_allow_html=True)
                    else:
                        linea_estado("ok", "Sin frases inventadas")


# ------------------------------------------------------------------ carga en segundo plano
@st.fragment(run_every=2)
def vigilar_carga() -> None:
    """Primera vuelta: solo marca que la página ya se dibujó. Después (cada 2 s) arranca la
    carga si hace falta y, cuando todo está listo, refresca la página una vez."""
    if not st.session_state.get("_pagina_dibujada"):
        st.session_state["_pagina_dibujada"] = True
        return
    C.iniciar()
    if all(C.hecho(k) for k in ("motor", "generador", "tienda", "qwen")):
        st.rerun()


if not all(C.hecho(k) for k in ("motor", "generador", "tienda", "qwen")):
    vigilar_carga()
