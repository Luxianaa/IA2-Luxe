# Hilo · Asistente NLP de atención al cliente para Merci Collection

Merci Collection es una tienda boliviana de jeans que vende 100% online por WhatsApp. Cada día
llegan mensajes como *"¿el 240904 en 38 hay? ¿a cómo está?"*, *"me queda grande, quiero
cambiarlo"* o *"¿tienen tienda en Cochabamba?"*. Responder bien exige mirar el stock, el
pedido anterior de la clienta, el courier de su ciudad y las respuestas oficiales de la tienda.

**Hilo lee el mensaje, entiende qué necesita la clienta, busca los datos reales y le propone a
la vendedora un borrador listo para copiar en WhatsApp.** La vendedora siempre revisa y envía:
Hilo asiste, no responde solo.

Todo corre en la laptop, con modelos abiertos y **sin APIs de pago** (ni Claude, ni GPT).

---

## 1. Resumen: qué se usa, para qué y por qué

| Parte | Qué se usa | Para qué | Por qué esto y no otra cosa |
|---|---|---|---|
| Entender la intención | **BETO** (BERT en español, `dccuchile/bert-base-spanish-wwm-cased`) afinado con *fine-tuning* multi-etiqueta | Saber qué pide la clienta (7 intenciones; un mensaje puede tener varias) | Preentrenado en español (entiende "a cómo está", "me queda grande"). Ver [§4.2](#42-clasificador-de-intenciones-beto--reglas--híbrido) |
| Red de seguridad | **Palabras clave** (reglas) + **híbrido** (promedio BETO + reglas) | Que una frase rara no haga fallar todo | Con 60 chats reales BETO solo no es confiable; el híbrido fue el mejor en frases nuevas (88%) |
| Datos extra para entrenar | **240 chats sintéticos** (`generar_sinteticos.py`) | Enseñarle a BETO formas de hablar que no estaban en los 60 chats | Duplicó el acierto de BETO en frases nuevas (37% → 76%) |
| Sacar datos del mensaje | **Expresiones regulares y diccionarios** | Tallas, Ref, ciudades, estilos, color, tiro, medidas, email | Son datos exactos (un número, un código): una regla no se equivoca y no necesita entrenamiento |
| Datos del negocio | **API de WooCommerce** (Store API pública + REST de pedidos con clave de solo lectura) y **fichas de producto** | Stock por talla, precios, ofertas, pedidos de la clienta, tablas de medidas | Datos en vivo de la tienda real; con respaldo en CSV sin internet |
| Respuestas parecidas y recomendación | **MiniLM multilingüe** (`paraphrase-multilingual-MiniLM-L12-v2`, búsqueda semántica) | Encontrar respuestas reales de Merci parecidas; ordenar modelos por parecido con lo que pide la clienta | Entiende sinónimos ("matrimonio" ~ "outfits elevados"); en frases nuevas acierta 67% vs. 41% de difflib |
| Borrador base | **Plantillas oficiales** de Merci + datos reales | Una respuesta siempre correcta | Determinista: el mismo dato siempre da el mismo texto |
| Redacción (opcional) | **Qwen 2.5 3B** local con **Ollama** | Reescribir la parte conversacional con otras palabras | Corre sin internet y sin costo. Ver [§5](#5-plantilla-vs-qwen-cuál-es-la-diferencia) |
| Verificación | **Controles deterministas** (fidelidad, stock y filtro de frases) | Que ninguna respuesta lleve un dato inventado | Un modelo de lenguaje puede inventar; el código no |
| Interfaz | **Streamlit** | Formulario + chat estilo WhatsApp + pestaña de análisis | Una app web en Python puro, en pocas líneas |
| Entrenamiento | **Google Colab** (GPU gratis) | Afinar BETO | En la laptop (CPU) tardaría horas |

---

## 2. Flujo completo (de un mensaje a la respuesta)

```
Mensaje de la clienta (+ último mensaje de Merci, teléfono y Ref si la vendedora los tiene)
│
├─1. ENTENDER ─────────────────────────────────────────────────────────────────────────
│   ├─ Intenciones: BETO v2 ─┐
│   │                         ├─► Híbrido = promedio de probabilidades ─► umbral 0.5
│   ├─ Palabras clave ───────┘    (siempre al menos 1 intención)
│   ├─ Entidades (regex): tallas, Ref, ciudades, estilos, color/tiro/tela, email,
│   │                     medidas ("cintura 78 cm"), ¿pide recomendación?, ¿pide medidas?
│   └─ Alerta courier ("no me llega", "el courier no contesta")
│
├─2. RECUPERAR DATOS REALES ────────────────────────────────────────────────────────────
│   ├─ Teléfono  ─► pedidos de la tienda en vivo (o pedidos.csv): compras, último pedido
│   ├─ Ref/talla ─► stock en vivo (WooCommerce Store API, o stock.csv)
│   ├─ Estilo    ─► modelos con stock de ese estilo/color/tiro y talla
│   ├─ Medidas   ─► tabla de medidas de la ficha del producto (data/medidas.json)
│   ├─ Ciudad    ─► oficina del courier (couriers.csv)
│   └─ Mensaje   ─► MiniLM: 3 respuestas reales de Merci más parecidas (ejemplos de tono)
│
├─3. BORRADOR BASE (PLANTILLA) ────────────────────────────────────────────────────────
│   Plantillas oficiales + datos reales; si pide consejo: recomendación (MiniLM sobre las
│   descripciones de la tienda) o talla según su medida.  ◄── esto es la opción "Plantilla"
│
├─4. REDACCIÓN CON QWEN (opcional) ────────────────────────────────────────────────────
│   Listas, contactos, links y textos oficiales se reemplazan por [LISTA_n] y se insertan
│   tal cual después. Qwen solo reescribe las frases sueltas.  ◄── esto es la opción "Qwen"
│
├─5. VERIFICAR (código, no IA) ─────────────────────────────────────────────────────────
│   ├─ Filtro de frases: borra cada frase de Qwen con cosas que no están en los datos
│   ├─ Fidelidad: ningún número/link/email nuevo ni omitido respecto de la plantilla
│   ├─ Stock: cada talla y precio del texto se compara con la tienda, línea por línea
│   └─ Si la fidelidad o el stock fallan ─► se descarta y Qwen redacta otra vez (1 vez)
│
└─6. HUMANO ───────────────────────────────────────────────────────────────────────────
    La vendedora ve el borrador en burbujas (una por mensaje de WhatsApp), con una línea
    verde/roja de verificación. Copia, edita o pide otra versión.
```

**Ejemplo real.** Mensaje: *"Hola, el 240904 en talla 38 hay? a cómo está? es para regalo de mi
hermana"*, teléfono de una clienta.

1. Intenciones: `disponibilidad` + `precio`. Entidades: Ref 240904, talla 38.
2. La tienda dice que la talla 38 tiene stock, cuesta 590 Bs y está en oferta a 531 Bs. Los
   pedidos dicen que es clienta frecuente (12 compras): queda una nota interna para la vendedora.
3. Plantilla: *"¡Sí! Tenemos el modelo Jean Mom fit… (Ref: 240904) disponible en talla 38 🤩 Su
   precio es de 590 Bs y ahora está en oferta a 531 Bs 🎉"* + lista de 6 datos + cambio gratis.
4. El verificador confirma la talla 38 y el precio contra la tienda: línea verde.

---

## 3. Los datos

| Fuente | Qué tiene | Uso |
|---|---|---|
| `data/dataset.csv` | **60 chats reales** de WhatsApp (mensaje, contexto, respuesta de Merci, intenciones etiquetadas a mano) | Entrenar y validar BETO; ejemplos de tono |
| `data/dataset_sintetico.csv` | **240 chats sintéticos** | Solo entrenamiento |
| Tienda en vivo (WooCommerce) | 34 modelos publicados, stock por talla, precios, ofertas, categorías, descripciones | Disponibilidad, precios, estilos, recomendación |
| Fichas de producto (web) | Tabla de medidas por talla (cintura, cadera, largo, botapié). 21 de 34 modelos la tienen | Medidas y talla según medidas |
| Pedidos (API con clave de solo lectura, o `pedidos.csv`) | Compras de cada teléfono | Cliente frecuente, último pedido (cambios), ciudad |
| `data/couriers.csv` | Oficina central del courier por ciudad | Alerta courier |

Distribución de los 60 chats reales: envío y plazos 12, datos de envío 11, disponibilidad 8,
saludo/catálogo 6, precio 6, cambio de talla 6, tienda física 3, y 11 con dos intenciones.

**Por qué datos sintéticos:** 60 chats son muy pocos para un modelo de 110 millones de
parámetros, y faltaban casos que ocurren seguido ("¿tienen tienda física?", "busco wide leg talla
40"). `generar_sinteticos.py` combina frases típicas de las clientas (con y sin tildes,
"porfa", emojis, saludos) con tallas, ciudades, estilos y Ref reales, y las respuestas siguen el
estilo de Merci. **Se usan solo para entrenar**: validar con datos sintéticos inflaría el resultado.

---

## 4. Cada componente en detalle

### 4.1 Las 7 intenciones

| Intención | Ejemplo | Qué responde Hilo |
|---|---|---|
| `saludo_catalogo` | "Hola, ¿tienen catálogo?" | saludo según la hora de Bolivia + catálogo + 6 datos del pedido |
| `tienda_fisica` | "¿Tienen tienda en Cochabamba?" | texto oficial: venta 100% online + primer cambio sin costo |
| `disponibilidad` | "¿El 240904 en 38?", "busco wide leg talla 40" | stock real; por estilo + talla, solo la lista filtrada |
| `precio` | "¿A cómo está?" | precio y oferta vigente |
| `envio_plazos` | "¿Cuándo llega a Sucre?" | 2 a 3 días hábiles + fecha estimada |
| `cambio_talla` | "Me queda grande" | último pedido de la clienta, talla sugerida, pasos del cambio |
| `datos_envio` | "Lo quiero", o la lista de datos | pide o confirma los 6 datos y ofrece el QR |

Es **multi-etiqueta**: *"¿cuánto cuesta y cuándo llega a Sucre?"* = `precio` + `envio_plazos`.

### 4.2 Clasificador de intenciones: BETO + reglas = híbrido

**Qué es el fine-tuning.** BETO ya "sabe español" (fue preentrenado con millones de textos). Se le
agrega una capa de salida con 7 neuronas (una por intención, con sigmoide) y se reentrena con
los chats etiquetados para que aprenda **esta** tarea. Detalles (`train_colab.py`):

- 48 chats reales + 240 sintéticos = 288 para entrenar; **12 chats reales** para validar (nunca vistos)
- 20 épocas como máximo, *early stopping* (paciencia 5, se guarda la mejor época), lote 8,
  tasa de aprendizaje 3e-5 con *warmup* del 10%, AdamW
- Pérdida `BCEWithLogitsLoss` con `pos_weight` (las intenciones poco frecuentes pesan más)
- Entrada: par (último mensaje de Merci, mensaje de la clienta), 128 tokens
- Umbral 0.5; si ninguna intención lo supera, se toma la más probable

**Por qué BETO y no otros:**

| Alternativa | Por qué no |
|---|---|
| TF-IDF + regresión logística | Solo cuenta palabras: no generaliza a frases que no vio ("¿están en Montero?" = tienda física). Con 60 chats, poco mejor que las reglas |
| mBERT / XLM-R (multilingües) | Reparten su capacidad entre 100 idiomas; BETO está entrenado solo en español |
| RoBERTa-BNE / MarIA | Alternativa comparable (mismo tamaño) que no se probó; BETO es el modelo en español más usado y documentado |
| GPT / Claude por API | Costo por mensaje, datos de clientas enviados a terceros, y el proyecto debía usar modelos abiertos y propios |
| Qwen para clasificar | Varios segundos por mensaje en CPU, no aprende de nuestros datos etiquetados y no da probabilidades calibradas por intención |

**Por qué el híbrido:** BETO generaliza mejor, pero se equivoca con frases muy distintas a su
entrenamiento. Las reglas no fallan cuando hay palabras clave claras ("precio", "catálogo"). El
promedio de ambas probabilidades fue lo mejor en frases nuevas (ver §6.1) y es el único método
que usa la app; en la pestaña Análisis se ven las probabilidades de BETO, de las reglas y del híbrido.

### 4.3 Entidades (reglas)

Tallas 34–48, Ref de 6 dígitos, email, ciudades y provincias de Bolivia (con alias:
"scz", "santa cruz de la sierra"), estilos (wide leg/anchos, mom, recto, skinny, flare/campana,
cargo…), características (tiro alto/medio, negro, celeste, rígido, stretch), medidas de la
clienta ("mi cintura es de 78 cm" → Cintura 78; ese 78 no se confunde con una talla), si pide
recomendación ("¿qué me recomiendas?", "no sé cuál elegir") o medidas ("¿qué talla soy?").

**Por qué reglas y no un modelo de NER:** son valores exactos que luego se buscan en la
tienda. Un NER necesitaría cientos de mensajes etiquetados palabra por palabra y podría
"equivocarse" en un número; una expresión regular no.

### 4.4 Datos reales de la tienda (WooCommerce)

- **Stock:** `GET /wp-json/wc/store/v1/products` y `?type=variation` (API pública, sin claves y
  sin datos de clientes). Copia en memoria por 5 minutos, actualizada en segundo plano.
- **Pedidos:** `GET /wp-json/wc/v3/orders?search=<teléfono>` con una clave de **solo lectura**
  guardada en `tienda.env` (excluido de git). Se confirma que el teléfono coincida exactamente.
- **Tablas de medidas:** la API no las trae, así que se leen de la ficha pública de cada producto
  (cada 6 h; copia en `data/medidas.json` para usarla sin internet).
- **Sin internet:** todo pasa solo a `stock.csv`, `pedidos.csv` y la copia de medidas.

**Por qué la API y no solo los CSV:** los CSV se desactualizan (una talla agotada ayer se
seguiría ofreciendo). **Por qué no la API para las medidas:** WooCommerce no las expone; están
en el HTML de la ficha.

### 4.5 Búsqueda semántica (MiniLM)

Convierte cada texto en un vector de 384 números que representa su significado; dos textos
parecidos tienen vectores cercanos (similitud coseno). Se usa para dos cosas:

1. **Ejemplos de tono:** las 3 respuestas reales de Merci más parecidas al mensaje (se le pasan a
   Qwen y se muestran en Análisis).
2. **Recomendación:** comparar el mensaje con la descripción real de cada modelo.

**Por qué MiniLM multilingüe:** es pequeño (118 M parámetros, ~0.1 s en CPU), funciona en
español y está entrenado justamente para medir parecido entre frases. Modelos más grandes (e5,
BGE) son más lentos en CPU. `difflib` (parecido de letras) queda como respaldo: funciona bien
con frases casi iguales, pero no entiende sinónimos (ver §6.2).

### 4.6 Recomendación y medidas

1. **Filtra** los modelos **con stock** por lo que pidió la clienta: estilo, color, tiro, talla
   o, si dio su medida, la talla que le corresponde en **cada** modelo según su tabla.
2. **Ordena** con MiniLM: compara su mensaje con el nombre y la descripción de la tienda.
3. **Explica** cada modelo con la frase de su descripción que más se parece al mensaje (✨).

**Talla según medidas:** la primera talla cuya **prenda** mide al menos lo que dijo la clienta
(cintura/cadera); si está agotada, la siguiente con stock. La web aclara que son medidas de la
prenda, no del cuerpo, y Hilo lo dice. Si el modelo no tiene tabla publicada, pide la medida
en vez de inventar.

**Por qué así:** no hay historial de "qué compró quién después de qué" para un sistema de
recomendación colaborativo. Este método (basado en contenido) usa solo datos que existen.

### 4.7 Plantillas (el borrador base)

Textos oficiales de Merci (saludo, catálogo, 6 datos, cambio gratis, venta online, plazos,
pasos del cambio) rellenados con los datos reales. Están en `src/draft_generator.py`
(`PLANTILLAS`). Es la respuesta **segura**: siempre igual, siempre con los datos correctos.

### 4.8 Redacción con Qwen 2.5 3B (Ollama)

Qwen recibe: reglas estrictas (no inventar, no saludar, no ofrecer nada que no esté en la
plantilla), la plantilla con las listas reemplazadas por `[LISTA_n]`, la lista de datos que no
puede omitir y 2 respuestas reales de Merci como ejemplo de tono. Temperatura 0.3 (0.6 en
"Otra versión").

**Por qué Qwen 2.5:** buen español para su tamaño, sigue bien instrucciones y hay una versión
de 3B que corre en CPU. **Por qué 3B y no 7B** (laptop i7-1165G7 sin GPU, 6 escenarios):

| Modelo | Sin errores | Tiempo medio | Observación |
|---|---|---|---|
| **Qwen 2.5 3B** (elegido) | 6/6 tras ajustes | ~18 s | sigue mejor las reglas en esta CPU |
| Qwen 2.5 7B | 4/6 | ~38 s (máx. 53 s) | inventó una oficina de courier y un producto |

**Por qué Ollama:** descarga el modelo ya comprimido (4 bits, ~2 GB) y lo sirve en
`localhost:11434`. Con `transformers` el modelo pesaría ~6 GB y sería más lento en CPU.

**Licencia (importante):** Qwen 2.5 **3B** se publica con la *Qwen Research License*, que
permite investigación y uso académico pero **no uso comercial**. Para usarlo en la tienda real
habría que pasar a Qwen 2.5 7B (Apache 2.0, necesita GPU) u otro modelo con licencia abierta.

**Llama 3.2 3B o Gemma 2 2B:** son alternativas razonables del mismo tamaño que **no se
compararon** en este proyecto (solo se comparó Qwen 3B vs. 7B). Con el diseño actual (datos
fijos + verificadores), cambiar de modelo cambiaría el estilo, no la exactitud de los datos.

### 4.9 Verificadores (lo que impide inventar)

| Control | Qué revisa | Si falla |
|---|---|---|
| **Filtro de frases** | Cada frase de Qwen: si ofrece algo nuevo ("te gustaría…", "también tenemos…") o la mayoría de sus palabras no aparece en la plantilla ni en el mensaje | La frase se borra antes de mostrarse (se ve tachada en Análisis) |
| **Fidelidad** | Números, links y emails: ninguno nuevo y ninguno omitido respecto de la plantilla | Rojo/amarillo → se redacta otra vez automáticamente |
| **Stock** | Cada talla y precio del texto contra la tienda, línea por línea con su Ref | Rojo → se redacta otra vez; si sigue, la vendedora lo ve en rojo |

Son **código determinista, no IA**: no pueden "convencerse" de que un dato inventado es real.

### 4.10 Interfaz (Streamlit)

- **Responder:** formulario (teléfono, Ref, último mensaje de Merci, mensaje de la clienta) y
  chat estilo WhatsApp con burbujas, selector **Qwen | Plantilla**, línea de verificación,
  "Otra versión" y "Editar".
- **Análisis:** intenciones, tabla BETO vs. reglas, entidades, cliente, stock, recomendación,
  medidas, notas internas y verificación.

**Por qué Streamlit y no Gradio, Flask o React:** permite armar la app con Python en pocas
líneas, que es el lenguaje de los modelos. Gradio sirve para demos de un solo modelo, y
Flask/React exigirían escribir un frontend completo.

---

## 5. Plantilla vs. Qwen: ¿cuál es la diferencia?

**Plantilla** = el borrador base: plantillas oficiales + datos reales. Sin modelo de lenguaje.
**Qwen** = la misma plantilla, reescrita por un modelo generativo para que suene menos a
plantilla y reaccione a lo que dijo la clienta.

**Las dos usan la misma clasificación (BETO), los mismos datos y la misma recomendación.** Lo
único que cambia es quién escribe las frases sueltas: el código o Qwen.

### Por qué se parecen tanto (medido)

17 escenarios del guion × 2 versiones = 34 respuestas de Qwen comparadas con su plantilla:

| Medida | Resultado |
|---|---|
| Parte de la plantilla que son bloques fijos (listas, contactos, textos oficiales), que Qwen no toca | **66%** |
| Respuestas de Qwen prácticamente idénticas a la plantilla (>98%) | **17 de 34 (50%)** |
| Parecido medio entre Qwen y la plantilla | **0.90** (1 = idéntico) |
| Stock correcto | **34/34** |
| Fidelidad: sin datos nuevos ni omitidos | 30/34 (3 omitieron un dato; 1 inventó números → la app redacta otra vez) |
| Tiempo medio | **15 s** (la plantilla es instantánea) |

Son casi iguales **a propósito**: en las primeras pruebas Qwen inventaba datos (una oficina de
courier, "varias tiendas de Merci en Cochabamba", un modelo "Allure Slim" en talla 28). Para
evitarlo se fijaron las listas, se bajó la temperatura, se prohibió ofrecer cosas nuevas y se
agregó el filtro de frases. Cuantas más reglas, más se parece a la plantilla. Cuando Qwen sí
cambia algo, a menudo **reordena o resume** ("Tallas disponibles: 36, 38…") y a veces omite una
frase útil.

### Entonces, ¿para qué están las dos?

| | Plantilla | Qwen |
|---|---|---|
| Exactitud de los datos | Siempre correcta | Correcta gracias a los verificadores |
| Tono | Fijo (el oficial de Merci) | Varía; "Otra versión" da otra redacción |
| Velocidad | Instantánea | ~15 s |
| Riesgo | Ninguno | Puede inventar o quitar información (se detecta y se corrige) |
| Sin Ollama / PC lenta | Funciona | No disponible |

- **Plantilla es la respuesta principal.** Es la que conviene usar en el día a día.
- **Qwen es la parte de IA generativa del proyecto**, y su valor está en lo que muestra: un
  modelo generativo, solo, inventa; con los datos fijos y los verificadores, su salida es
  segura. Las frases que inventó y que el sistema borró se ven en Análisis → Verificación.
- **Con esta laptop y este tamaño de modelo, Qwen aporta poco al texto.** Para que aporte de
  verdad haría falta un modelo más grande con GPU o afinar Qwen (LoRA) con las respuestas
  reales de Merci (ver §8).

---

## 6. Resultados y análisis de cada modelo

> **Qué es cada métrica.** *Acierto exacto*: % de mensajes con **todas** sus intenciones bien (ni
> una de más ni una de menos). *F1*: equilibrio entre no inventar intenciones (precisión) y no
> perderlas (cobertura), de 0 a 1. *Micro*: cuenta todos los aciertos juntos. *Macro*: promedia
> cada intención por igual (castiga fallar en las poco frecuentes).
> Reproducible con `.venv\Scripts\python.exe evaluar_modelos.py` (~1 min en la laptop).

### 6.1 Clasificador de intenciones

**a) Validación: 12 chats reales que BETO v2 no vio al entrenar**

| Método | Acierto exacto | F1 micro | F1 macro |
|---|---|---|---|
| Palabras clave | 83% | 0.909 | 0.903 |
| **BETO v2** | **92%** | **0.966** | **0.967** |
| Híbrido | 83% | 0.933 | 0.943 |

El F1 micro de 0.966 es el mismo que reportó Colab al entrenar (0.9655). Colab mostró un F1
macro de 0.83 porque contó `cambio_talla` con 0 (no había ningún chat de esa intención entre los
12); aquí el macro solo promedia las intenciones presentes.

**b) Entrenamiento: los 48 chats reales que BETO v2 sí vio**

| Método | Acierto exacto | F1 micro |
|---|---|---|
| Palabras clave | 62% | 0.763 |
| BETO v2 | 96% | 0.982 |
| Híbrido | 96% | 0.972 |

**c) 49 frases nuevas** (`pruebas/frases_nuevas.csv`): escritas para esta prueba con otras
palabras, ninguna está en el entrenamiento. Es la medida más cercana a "mensajes de mañana".

| Método | Acierto exacto | F1 micro | F1 macro |
|---|---|---|---|
| Palabras clave | 71% | 0.803 | 0.813 |
| BETO v1 (48 chats reales, sin sintéticos) | 37% | 0.504 | 0.503 |
| BETO v2 (48 reales + 240 sintéticos) | 76% | 0.842 | 0.833 |
| **Híbrido v2 (por defecto en la app)** | **88%** | **0.912** | **0.909** |

F1 por intención en las frases nuevas:

| Intención | Palabras clave | BETO v1 | BETO v2 | Híbrido |
|---|---|---|---|---|
| saludo_catalogo | 0.70 | 0.40 | 0.67 | **0.92** |
| cambio_talla | 0.73 | 0.56 | 0.78 | **0.93** |
| envio_plazos | **0.93** | 0.44 | 0.89 | 0.89 |
| disponibilidad | 0.72 | 0.53 | **0.95** | **0.95** |
| precio | 0.88 | 0.33 | 0.88 | **0.94** |
| datos_envio | **0.88** | 0.40 | 0.83 | 0.83 |
| tienda_fisica | 0.86 | 0.86 | 0.84 | **0.89** |

**Análisis:**

1. **BETO aprendió.** De 37% (v1) a 76% (v2) en frases nuevas: los datos sintéticos le enseñaron
   formas de hablar que no estaban en los 60 chats. Antes, BETO v1 no le ganaba a las reglas
   (validación cruzada 0.72 vs. 0.80, ver abajo).
2. **Hay sobreajuste.** BETO v2 acierta 96% en lo que vio y 76% en frases nuevas. Es esperable con
   tan pocos datos reales: memoriza parte de los ejemplos.
3. **La validación de 12 chats es optimista.** El 92% de BETO v2 se basa en 12 mensajes (un error
   más = −8 puntos); la cifra de las frases nuevas es más representativa.
4. **El híbrido es el más robusto (88%).** BETO y las reglas se equivocan en mensajes distintos;
   al promediar, uno corrige al otro. Por eso es el método por defecto.
5. **Errores que quedan** (híbrido): "me recomendó una amiga, ¿qué jeans manejan?" →
   tienda_fisica; "¿tienen alguna promo si llevo dos?" → cambio_talla; "me envías el QR para
   pagar?" y "¿cómo pago? por transferencia o QR?" → envío (en vez de datos de envío). Los pagos y
   las promociones casi no aparecen en los datos de entrenamiento.
6. **Ojo:** las 49 frases se escribieron para esta prueba (no son chats reales) conociendo las
   reglas, así que la cifra de las palabras clave puede estar algo inflada. La prueba ideal son
   chats reales nuevos etiquetados a mano.

**BETO v1 · validación cruzada 5-fold con los 60 chats reales** (`evaluar_cv.py`, antes de los
sintéticos):

| Clasificador | F1 micro | F1 macro | Acierto exacto |
|---|---|---|---|
| Palabras clave | 0.797 ± 0.089 | 0.819 ± 0.091 | 68% |
| BETO v1 | 0.722 ± 0.065 | 0.729 ± 0.065 | 57% |

Durante el entrenamiento de v1, el F1 de validación subió de 0.28 (época 1) a 0.71: aprendía,
pero 48 chats no alcanzaban para superar a las reglas. Esa fue la razón para generar datos
sintéticos y usar el híbrido.

### 6.2 Búsqueda semántica (MiniLM)

¿La respuesta real recuperada es de la misma intención que el mensaje?

| Prueba | MiniLM | difflib | Al azar |
|---|---|---|---|
| Entre los 60 chats reales (dejando fuera el propio) | 57% | 63% | 20% |
| **49 frases nuevas → chat real más parecido** | **67%** | 41% | — |

Con frases **casi iguales** (dentro del dataset), comparar letras (difflib) funciona igual o algo
mejor: la diferencia es de 4 chats. Con frases **nuevas** MiniLM es muy superior (+26 puntos),
porque compara significado y no letras. Ese es el caso real: las clientas escriben distinto
cada vez.

### 6.3 Qwen 2.5 3B (redacción)

| Medida | Resultado |
|---|---|
| Respuestas evaluadas (3 rondas de prueba) | 80 |
| Stock correcto en el texto final | 100% (en la ronda de 34: 34/34) |
| Fidelidad (sin números/links nuevos ni omitidos), ronda de 34 | 88% (30/34); el resto se corrige con el reintento automático |
| Frases eliminadas por el filtro | 12 en 80 respuestas: **5 con datos inventados** ("varias tiendas de Merci Collection" en Bolivia, "puedo ayudarte a ir a una tienda cercana", "el modelo Allure Slim en talla 28", "la que está en Cochabamba", "los datos de una tienda que vende Merci") y 7 de relleno o repetidas ("Te agradezco tu interés…", "¿Te gustaría ver algunos?") |
| Frases con datos reales eliminadas por error | 0 |
| Respuestas casi idénticas a la plantilla | 50% |
| Tiempo medio por respuesta (CPU) | ~15 s |

**Análisis:** Qwen **no se entrenó** en este proyecto (se usa tal cual, con instrucciones y
ejemplos: *few-shot*), así que no tiene un "porcentaje de aprendizaje". Lo que se mide es si es
**seguro**: con los verificadores, ningún dato inventado llega a la vendedora sin una alerta.
El filtro es estricto: prefiere borrar una frase amable a dejar pasar una inventada.

### 6.4 Recomendación y medidas

No hay respuestas "correctas" etiquetadas para medir un porcentaje de acierto (haría falta
saber qué compró cada clienta después de la recomendación). Lo verificable:

- **Siempre con stock:** las tallas y precios recomendados pasaron el verificador de stock en el
  100% de las pruebas.
- **Cobertura de medidas:** 21 de 34 modelos tienen tabla publicada. Para los otros 13 Hilo
  pide la medida de cintura en vez de inventar una talla.
- **Detección de medidas:** "mi cintura es de 78 cm y 98 de cadera", "mido 80 de cintura",
  "cintura: 77,5" se reconocen; "talla 38 cintura alta" y "cintura 40" (fuera de rango) no se
  toman como medida.

---

## 7. Cómo iniciar y reproducir

**Usar la app:** doble clic en `iniciar_hilo.bat` (abre `http://localhost:8501`; si Hilo ya está
abierto en otra ventana, avisa en vez de abrir una copia vieja).

**Instalación desde cero** (PowerShell, en la carpeta `hilo-merci`):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
winget install Ollama.Ollama        # opcional: solo para la opción Qwen
ollama pull qwen2.5:3b
.\.venv\Scripts\python.exe -m streamlit run app.py
```

| Script | Qué hace | Dónde se corre |
|---|---|---|
| `generar_sinteticos.py` | Genera los 240 chats sintéticos | laptop |
| `train_colab.py` | Fine-tuning de BETO (reales + sintéticos) | Google Colab (GPU) |
| `evaluar_cv.py` | Validación cruzada 5-fold de BETO vs. reglas | Google Colab (GPU) |
| `evaluar_modelos.py` | Todas las métricas de §6.1 y §6.2 | laptop (~1 min) |

El modelo entrenado va en `models/modelo_hilo_final/` (BETO v1 queda en
`models/modelo_hilo_6intenciones/` para comparar).

| Archivo | Qué hace |
|---|---|
| `app.py` | Interfaz Streamlit |
| `src/nlp_engine.py` | Intenciones (BETO / híbrido / reglas), entidades, alerta courier |
| `src/data_loader.py` | Pedidos, stock, estilos, medidas, couriers |
| `src/tienda_api.py` | Conexión con la tienda (stock, pedidos, tablas de medidas) |
| `src/draft_generator.py` | Plantillas, búsqueda semántica, recomendación, verificador de stock |
| `src/redactor_llm.py` | Qwen: prompt, bloques fijos, filtro de frases, fidelidad |

---

## 8. Limitaciones y trabajo futuro

1. **Más chats reales etiquetados** (150–200): la tienda tiene miles de conversaciones. Es lo
   que más mejoraría a BETO y la confiabilidad de las métricas.
2. **Validación cruzada de BETO v2** (sintéticos solo en entrenamiento) para tener una cifra con
   margen de error, como la de v1.
3. **Afinar Qwen con LoRA** sobre las respuestas reales de Merci, para que redacte con su tono sin
   depender de reglas tan estrictas, o usar un modelo mayor con GPU. Así Qwen aportaría más que
   la plantilla.
4. **Licencia:** cambiar Qwen 2.5 3B por un modelo con licencia comercial antes de usarlo en la
   tienda real.
5. Pagos y promociones casi no aparecen en los datos: agregar ejemplos.
6. Si la clienta solo manda una foto, la vendedora escribe la Ref (no hay reconocimiento de imagen).
7. Responder directo por la API de WhatsApp Business (hoy la vendedora copia y pega).

## 9. Privacidad

No se suben al repositorio (ver `.gitignore`):

- `tienda.env`: claves de la API de WooCommerce. Para usar la tienda en vivo, copia
  `tienda.env.ejemplo` como `tienda.env` y pega una clave de **solo lectura**.
- `data/pedidos.csv`: pedidos con nombres, teléfonos, direcciones y correos de clientas. Sin este
  archivo la app funciona igual (consulta los pedidos por la API o los omite).
- El guion de la demo y los modelos entrenados (`model.safetensors` pesa ~440 MB). Para tener a
  BETO, corre `train_colab.py` en Google Colab y copia el resultado en `models/modelo_hilo_final/`;
  sin el modelo, la app usa el clasificador de palabras clave.

Los chats de `data/dataset.csv` están anonimizados (`[NOMBRE]`, `[TEL]`, `[DIRECCION]`,
`[EMAIL]`). Los modelos corren en la laptop: ningún mensaje de clienta sale a servicios externos.
