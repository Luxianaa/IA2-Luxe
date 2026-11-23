"""Genera chats sintéticos para entrenar BETO (data/dataset_sintetico.csv).

Por qué: con 60 chats reales BETO aprende poco. Aquí se escriben a mano frases típicas de
las clientas de Merci (cómo preguntan por la tienda física, estilos, tallas, precios,
envíos, cambios…) y se combinan con tallas, ciudades, estilos, características y códigos
reales. Las respuestas siguen el estilo de las respuestas reales de Merci.

Reglas para que la evaluación sea honesta:
    - Los sintéticos se usan SOLO para entrenar (train_colab.py los suma al entrenamiento).
    - La validación y la validación cruzada se hacen SOLO con chats reales.

Uso:
    python generar_sinteticos.py            # crea data/dataset_sintetico.csv
    python generar_sinteticos.py --n 300    # más ejemplos (máximo según combinaciones)
"""
from __future__ import annotations

import argparse
import csv
import random
import unicodedata
from pathlib import Path

SALIDA = Path("data/dataset_sintetico.csv")

# ---------------------------------------------------------------------- vocabulario
TALLAS = [34, 36, 38, 40, 42, 44, 46]
CIUDADES = ["La Paz", "El Alto", "Cochabamba", "Cocha", "Santa Cruz", "Scz", "Sucre", "Potosí",
            "Oruro", "Tarija", "Trinidad", "Cobija", "Yacuiba", "Montero", "Quillacollo", "Riberalta"]
ESTILOS = ["wide leg", "wide-leg", "jeans anchos", "pantalón ancho", "mom fit", "mom", "straight",
           "bota recta", "jeans rectos", "skinny", "slim fit", "flare", "acampanados", "palazzo",
           "cargo", "bootcut", "barrel"]
CARACT = ["tiro alto", "de tiro alto", "tiro medio", "cintura alta", "negro", "en color negro",
          "celeste", "clarito", "azul oscuro", "stretch", "que estire", "licrado", "rígido",
          "que no estire", "de tela liviana", "con pretina elástica", "con botones", "con encaje"]
REFS = ["240904", "240919", "240933", "240935", "240936", "240939", "240941", "240943", "240944",
        "240916", "240921", "240928", "240930", "240918", "240907", "240911"]
NOMBRES = ["Ana Gutiérrez", "María López", "Carla Rojas", "Daniela Vargas", "Paola Quispe",
           "Lucía Mamani", "Fernanda Choque", "Valeria Flores", "Gabriela Suárez", "Andrea Torrez"]
CALLES = ["Calle Bolívar 245", "Av. Busch 1320", "Zona Sur calle 21 #85", "Av. América 540",
          "Barrio Equipetrol calle 4", "Calle Junín 77", "Av. Blanco Galindo km 4", "Zona Sopocachi c. Ecuador 2100"]

SALUDOS = ["Hola ", "Holaa ", "Hola buenas ", "Buenas ", "Buenas tardes ", "Buen día ",
           "Buenas noches ", "Hola buenos días ", "Holi ", "Hola, una consulta "]
COLAS = [" porfa", " por favor", " 🙏", " 😊", "??", " gracias", " 🥺", " please", ""]
CONTEXTOS = ["", "", "", "Lo tenemos, ¿en qué talla lo busca?",
             "Si la talla que pides no te queda bien, te hacemos un cambio de talla gratuito! 💗🎉",
             "Páseme sus datos y le ayudo a realizar la compra :)",
             "Buenas tardes! Gracias por comunicarte con Merci Collection! 💙👖"]

# ---------------------------------------------------------------------- respuestas (estilo Merci)
R_SALUDO = "Buenas tardes! Gracias por comunicarte con Merci Collection! 💙👖"
R_TIENDA = ("Nuestra venta es 100% online 😊 No tenemos un local para visitar, pero enviamos a "
            "cualquier ciudad de Bolivia con envío gratis. Si te preocupa la talla, no hay problema: "
            "el primer cambio de talla o de modelo no tiene costo de envío 💕.")
R_CAMBIO_GRATIS = "Si la talla que pides no te queda bien, te hacemos un cambio de talla gratuito! 💗🎉"
R_PLAZO = "El envío es gratis 🚚 y tarda de 2 a 3 días hábiles; el courier no trabaja fines de semana."
RESPUESTAS = {
    "tienda_fisica": R_TIENDA,
    "saludo_catalogo": f"{R_SALUDO} [PLANTILLA_CATALOGO] [PLANTILLA_DATOS_PEDIDO] {R_CAMBIO_GRATIS}",
    "disponibilidad": "¡Sí! Lo tenemos 🤩 [LISTA_MODELOS]",
    "precio": "Su precio es de [PRECIO] Bs 🤩",
    "envio_plazos": R_PLAZO,
    "cambio_talla": "Se puede hacer el cambio sin ningún problema 🙌 [PLANTILLA_CAMBIO]",
    "datos_envio": "¡Claro que sí! [PLANTILLA_DATOS_PEDIDO] Ayúdeme con sus datos para realizar el pedido :)",
}

# ---------------------------------------------------------------------- frases por intención
# {t}=talla {c}=ciudad {e}=estilo {f}=característica {r}=Ref
FRASES: dict[tuple[str, ...], list[str]] = {
    ("tienda_fisica",): [
        "tienen tienda física?", "tienen tienda en {c}?", "dónde queda su tienda?",
        "dónde están ubicados?", "tienen local?", "tienen algún local en {c}?", "puedo ir a probarme?",
        "hay algún lugar donde pueda ir a medirme?", "de qué ciudad son?", "de qué departamento son ustedes?",
        "atienden en persona?", "tienen sucursal en {c}?", "cuál es la dirección de la tienda?",
        "quisiera ir a ver los modelos, dónde están?", "están en {c}?", "tienen punto de venta en {c}?",
        "me puedo acercar a su tienda?", "solo venden por internet?", "es solo online?",
        "puedo pasar a medirme antes de comprar?", "en qué zona está su tienda?", "tienen tienda en el centro?",
        "tienen showroom?", "dónde los puedo encontrar en {c}?", "no tienen tienda para ir a probarme?",
        "se puede ir a su local a ver?", "atienden en algún mall?", "tienen tienda o son online nomás?",
        "hay tienda donde ver los jeans en persona?", "quiero probarme antes, tienen local en {c}?",
        "ustedes de dónde son?", "dónde están ubicados en {c}?", "tienen dirección física?",
        "puedo ir hoy a su tienda?", "a qué hora abre su tienda?", "cerca de dónde está la tienda?",
        "tienen tienda en {c} o solo hacen envíos?", "dónde es su tienda?", "tienen local para ir a medirme?",
    ],
    ("disponibilidad",): [
        "busco pantalón {e} talla {t}", "tienen {e} en talla {t}?", "hay {e} en {t}?", "tienen {e}?",
        "quiero ver modelos {e}", "qué {e} tienen en {t}?", "tienen jeans {f}?", "tienen algo {f} en talla {t}?",
        "busco un jean {f}", "algún {e} {f} en {t}?", "tienen {e} {f}?", "me muestra los {e} que tenga en {t}",
        "hay {e} talla {t}? soy {t}", "tienen tallas grandes? uso {t}", "tienen en {t}?", "el {r} hay en {t}?",
        "este modelo en {t} tienen?", "está disponible en {t}?", "todavía tienen el {r}?",
        "quedan del {r} en talla {t}?", "qué tallas tienen del {r}?", "el {r} es {f}?",
        "ese jean estira o es rígido?", "es de tiro alto o medio?", "la tela es gruesa o delgada?",
        "qué colores tienen en {e}?", "tienen {e} en color negro?", "cuál me recomiendas {f}?",
        "en número {t} hay?", "Este en {t}", "Me interesa este modelo en talla {t}", "hay en mi talla? soy {t}",
        "busco jeans {e} {f}", "tienen {e} {f} talla {t}?", "algún modelo {f}?", "lo tienen en {t}?",
    ],
    ("precio",): [
        "cuánto cuesta?", "a cómo está?", "a cuánto?", "precio por favor", "cuánto sale el {r}?",
        "qué precio tiene el {e}?", "cto sale", "cuánto es?", "precio del jean de la foto",
        "a cómo están los {e}?", "tienen descuento?", "está en oferta?", "cuánto vale ese?",
        "el precio es el mismo en todas las tallas?", "tienen promoción por 2?", "hay rebajas?",
        "ese modelo cuánto?", "a cómo el {r}?", "precio?", "y el precio?", "cuánto está el de la foto?",
    ],
    ("envio_plazos",): [
        "cuándo me llegaría a {c}?", "cuánto tarda el envío a {c}?", "en cuántos días llega?",
        "hacen envíos a {c}?", "envían a {c}?", "el envío tiene costo?", "cuánto cuesta el envío a {c}?",
        "si pido hoy cuándo llega?", "lo necesito para el viernes, llegaría?", "es a domicilio?",
        "puedo recogerlo de la oficina del courier?", "mandan por flota?", "cuándo me lo despachan?",
        "ya pagué, cuándo lo envían?", "llega a provincia? estoy en {c}", "envían a {c} o solo a capitales?",
        "trabajan con courier?", "cuánto demora a {c}?", "llegaría antes del sábado?", "hacen delivery?",
        "cuándo llega mi pedido?", "ya hice el pago, cuándo me lo mandan?", "para cuándo estaría llegando?",
    ],
    ("cambio_talla",): [
        "me quedó grande, lo puedo cambiar?", "me queda chico, cómo hago el cambio?",
        "quiero cambiar por una talla más", "si no me queda lo puedo cambiar?", "el cambio tiene costo?",
        "puedo cambiar de modelo?", "me llegó y me aprieta, quiero el {t}", "cómo es el cambio de talla?",
        "quiero cambiar el {r} por talla {t}", "me queda suelto de cintura, se puede cambiar?",
        "el cambio es gratis?", "ya hice un cambio, el segundo cuánto cuesta?", "no me entra, quiero devolverlo",
        "me queda pequeño, me lo cambian por {t}?", "cuánto tiempo tengo para cambiarlo?",
        "puedo cambiarlo por otro modelo más caro?", "me llegó el jean pero me queda grande 😩",
        "acabo de recibirlo y no me cierra, qué hago?",
    ],
    ("datos_envio",): [
        "quiero este", "lo quiero en talla {t}", "quiero hacer mi pedido", "me envía el QR?",
        "quiero comprar el {r} en {t}", "me lo llevo", "quisiera pedir 2 jeans", "ya, lo quiero",
        "deseo comprar este para {c}", "páseme el qr", "quisiera adquirir este pantalón en talla {t}",
        "1. {n} 2. {d} 3. {c} 4. {m} 5. Talla {t} 6. Ref {r}", "Mi nombre es {n}, vivo en {d}, {c}. Talla {t}",
        "{n}\n{d}\n{c}\n{m}\ntalla {t}", "Nombre: {n} Dirección: {d} Ciudad: {c} Email: {m} Talla: {t}",
        "quiero el {r} talla {t} para {c}", "me gustaría realizar la compra", "lo pido, cómo pago?",
    ],
    ("saludo_catalogo",): [
        "hola", "holaa buenas", "tienen catálogo?", "me pasa su catálogo", "qué modelos tienen?",
        "quisiera ver sus modelos", "cómo puedo comprar?", "buenas, quiero ver los jeans",
        "hola, vi su publicación en tiktok", "me pasas fotos de los modelos?", "info", "quiero información",
        "tienen página web?", "quisiera saber cómo comprar", "su catálogo", "buenas noches tiene catálogo",
        "hola vi sus jeans en instagram", "qué jeans tienen?",
    ],
    # ---- combinaciones (varias intenciones en el mismo mensaje)
    ("tienda_fisica", "envio_plazos"): [
        "no tienen tienda en {c}? cuánto tarda en llegar?", "son de {c}? cuánto demora el envío?",
        "dónde están? o envían a {c}?", "tienen local o solo envíos? estoy en {c}, cuánto tarda?",
    ],
    ("tienda_fisica", "cambio_talla"): [
        "como no tienen tienda, si no me queda lo puedo cambiar?", "no puedo probármelo? y si me queda mal lo cambian?",
        "no tienen local para medirme? y si no me queda?",
    ],
    ("saludo_catalogo", "tienda_fisica"): [
        "hola, tienen tienda física o catálogo?", "dónde están ubicados? me pasa su catálogo",
        "tienen tienda? o me pasa el catálogo",
    ],
    ("disponibilidad", "tienda_fisica"): [
        "tienen tienda en {c}? busco un {e} en {t}", "puedo ir a ver los {e}? dónde están?",
    ],
    ("disponibilidad", "precio"): [
        "tienen {e} en {t}? a cuánto?", "el {r} en {t} hay? precio?", "precio talla {t}",
        "este modelo en talla {t} y a cómo está?", "hay {e} {f} en {t}? cuánto cuestan?",
        "cuánto cuesta el {e} y en qué tallas hay?",
    ],
    ("disponibilidad", "datos_envio"): [
        "hay en {t}? lo quiero", "si tienen el {r} en {t} lo compro", "tienen en {t}? quiero hacer el pedido",
    ],
    ("envio_plazos", "precio"): [
        "cuánto cuesta y cuánto tarda a {c}?", "precio y cuánto demora el envío a {c}?",
    ],
    ("cambio_talla", "disponibilidad"): [
        "me queda chico, tienen en {t} para cambiar?", "quiero cambiar mi jean, hay el {r} en {t}?",
    ],
    ("disponibilidad", "saludo_catalogo"): [
        "hola tienen catálogo? busco {e}", "me pasa el catálogo de {e}?",
    ],
}


# ---------------------------------------------------------------------- ruido realista
def sin_tildes(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def ensuciar(texto: str, rnd: random.Random, intencion: tuple[str, ...]) -> str:
    """Como escriben de verdad: sin tildes, minúsculas, sin '¿', saludos y colas."""
    t = texto
    if "saludo_catalogo" not in intencion and rnd.random() < 0.35:
        t = rnd.choice(SALUDOS) + t
    if rnd.random() < 0.25:
        t = t + rnd.choice(COLAS)
    if rnd.random() < 0.45:
        t = sin_tildes(t)
    if rnd.random() < 0.5:
        t = t.lower()
    elif rnd.random() < 0.6:
        t = t[:1].upper() + t[1:]
    if rnd.random() < 0.15:
        t = t.replace("?", "")
    return t.strip()


def rellenar(frase: str, rnd: random.Random) -> str:
    return frase.format(t=rnd.choice(TALLAS), c=rnd.choice(CIUDADES), e=rnd.choice(ESTILOS),
                        f=rnd.choice(CARACT), r=rnd.choice(REFS), n=rnd.choice(NOMBRES),
                        d=rnd.choice(CALLES), m=f"cliente{rnd.randint(10, 99)}@gmail.com")


def respuesta(intenciones: tuple[str, ...]) -> str:
    return " ".join(RESPUESTAS[i] for i in intenciones)


def generar(n_objetivo: int, seed: int = 42) -> list[dict]:
    rnd = random.Random(seed)
    filas, vistos = [], set()
    # Cada frase se usa al menos una vez; luego se repiten con otros rellenos hasta n_objetivo
    plantillas = [(ints, f) for ints, fs in FRASES.items() for f in fs]
    intentos = 0
    while len(filas) < n_objetivo and intentos < n_objetivo * 20:
        ints, frase = plantillas[len(filas) % len(plantillas)] if len(filas) < len(plantillas) \
            else rnd.choice(plantillas)
        intentos += 1
        msg = ensuciar(rellenar(frase, rnd), rnd, ints)
        if msg.lower() in vistos:
            continue
        vistos.add(msg.lower())
        filas.append({
            "contexto_previo": rnd.choice(CONTEXTOS) if ints == ("disponibilidad",) else "",
            "mensaje_cliente": msg,
            "respuesta_real": respuesta(ints),
            "intenciones": ",".join(ints),
            "origen": "sintetico",
        })
    return filas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--salida", default=str(SALIDA))
    args = ap.parse_args()
    filas = generar(args.n, args.seed)
    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    conteo: dict[str, int] = {}
    for fila in filas:
        for i in fila["intenciones"].split(","):
            conteo[i] = conteo.get(i, 0) + 1
    print(f"✅ {len(filas)} chats sintéticos en {salida}")
    print("Por intención:", dict(sorted(conteo.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
