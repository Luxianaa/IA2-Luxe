"""Evaluación de los modelos de Hilo (corre en la laptop, sin GPU, en ~1 minuto).

No entrena nada: usa los modelos ya entrenados y mide.

1. Clasificador de intenciones — palabras clave vs. BETO v1 vs. BETO v2 vs. híbrido, en:
   a) los 12 chats reales de VALIDACIÓN (la misma división que train_colab.py: semilla 42),
   b) los 48 chats reales de ENTRENAMIENTO (para ver cuánto "memorizó"),
   c) 49 FRASES NUEVAS (pruebas/frases_nuevas.csv) que ningún modelo vio al entrenar.
2. Búsqueda semántica — MiniLM vs. difflib: ¿la respuesta real más parecida que recupera es de
   la misma intención? (dejando fuera el propio chat).

Uso:  .venv\\Scripts\\python.exe evaluar_modelos.py
"""
from __future__ import annotations

import difflib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.data_loader import normalizar_texto
from src.nlp_engine import INTENCIONES, NLPEngine

RAIZ = Path(__file__).resolve().parent
V1 = RAIZ / "models" / "modelo_hilo_6intenciones"
V2 = RAIZ / "models" / "modelo_hilo_final"


def etiquetas(texto: str) -> list[str]:
    return [t.strip() for t in str(texto).split(",") if t.strip() in INTENCIONES]


def matriz(listas: list[list[str]]) -> np.ndarray:
    return np.array([[int(i in l) for i in INTENCIONES] for l in listas])


def medir(nombre: str, reales: list[list[str]], predichas: list[list[str]]) -> dict:
    Y, P = matriz(reales), matriz(predichas)
    presentes = Y.sum(axis=0) > 0  # macro solo sobre intenciones que aparecen en el conjunto
    return {
        "metodo": nombre,
        "acierto_exacto": float(np.mean([set(a) == set(b) for a, b in zip(reales, predichas)])),
        "f1_micro": f1_score(Y, P, average="micro", zero_division=0),
        "f1_macro": f1_score(Y[:, presentes], P[:, presentes], average="macro", zero_division=0),
        "por_intencion": {i: f1_score(Y[:, k], P[:, k], zero_division=0)
                          for k, i in enumerate(INTENCIONES) if presentes[k]},
    }


def predecir(motor: NLPEngine, metodo: str, contextos, mensajes) -> list[list[str]]:
    return [motor.analizar(c, m, metodo)["intenciones"] for c, m in zip(contextos, mensajes)]


def tabla(titulo: str, resultados: list[dict], n: int) -> None:
    print(f"\n=== {titulo} (n={n}) ===")
    print(f"{'método':34s} {'exacto':>7s} {'F1 micro':>9s} {'F1 macro':>9s}")
    for r in resultados:
        print(f"{r['metodo']:34s} {r['acierto_exacto']:7.0%} {r['f1_micro']:9.3f} {r['f1_macro']:9.3f}")
    print("F1 por intención:")
    intenciones = list(resultados[0]["por_intencion"])
    print(f"{'':18s}" + "".join(f"{r['metodo'][:14]:>16s}" for r in resultados))
    for i in intenciones:
        print(f"{i:18s}" + "".join(f"{r['por_intencion'][i]:16.2f}" for r in resultados))


def main() -> None:
    reales = pd.read_csv(RAIZ / "data" / "dataset.csv", encoding="utf-8-sig", dtype=str,
                         keep_default_na=False)
    reales["_tags"] = reales["intenciones"].map(etiquetas)
    reales = reales[reales["_tags"].map(len) > 0].reset_index(drop=True)
    # Misma división que train_colab.py (estratificada por la primera etiqueta, semilla 42)
    idx = np.arange(len(reales))
    try:
        tr, va = train_test_split(idx, test_size=0.2, random_state=42,
                                  stratify=reales["_tags"].map(lambda t: t[0]))
    except ValueError:  # una clase con 1 solo chat: train_colab.py divide sin estratificar
        tr, va = train_test_split(idx, test_size=0.2, random_state=42)
    nuevas = pd.read_csv(RAIZ / "pruebas" / "frases_nuevas.csv", dtype=str, keep_default_na=False)
    nuevas["_tags"] = nuevas["intenciones"].map(etiquetas)
    nuevas["contexto_previo"] = ""

    v2 = NLPEngine(V2)
    v1 = NLPEngine(V1) if V1.exists() else None
    conjuntos = {
        "Validación: chats reales no vistos": reales.iloc[va],
        "Entrenamiento: chats reales ya vistos": reales.iloc[tr],
        "Frases nuevas (pruebas/frases_nuevas.csv)": nuevas,
    }
    for titulo, df in conjuntos.items():
        ctx, msg, y = list(df["contexto_previo"]), list(df["mensaje_cliente"]), list(df["_tags"])
        res = [medir("Palabras clave (reglas)", y, predecir(v2, "reglas", ctx, msg))]
        # BETO v1 se validó con otra división (antes de reetiquetar 3 chats): parte de estos
        # chats reales los vio al entrenar, así que solo se compara en las frases nuevas
        if v1 is not None and titulo.startswith("Frases"):
            res.append(medir("BETO v1 (48 reales, 6 int.)", y, predecir(v1, "beto", ctx, msg)))
        res.append(medir("BETO v2 (+240 sintéticos)", y, predecir(v2, "beto", ctx, msg)))
        res.append(medir("Híbrido v2 (BETO+reglas)", y, predecir(v2, "hibrido", ctx, msg)))
        tabla(titulo, res, len(df))
        if titulo.startswith("Frases"):
            print("\nErrores del híbrido en frases nuevas:")
            for m, real, p in zip(msg, y, predecir(v2, "hibrido", ctx, msg)):
                if set(real) != set(p):
                    print(f"  - {m!r}: real={real} predicho={p}")

    # ---------------- búsqueda semántica (leave-one-out) ----------------
    from sentence_transformers import SentenceTransformer

    textos = [f"{m} | Contexto: {c}" if c.strip() else m
              for c, m in zip(reales["contexto_previo"], reales["mensaje_cliente"])]
    enc = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
    E = enc.encode(textos, normalize_embeddings=True)
    S = E @ E.T
    np.fill_diagonal(S, -1)
    tags = list(reales["_tags"])

    def acierto(vecinos):
        return np.mean([bool(set(tags[i]) & set(tags[j])) for i, j in enumerate(vecinos)])

    mini = acierto(S.argmax(axis=1))
    norm = [normalizar_texto(t) for t in textos]
    dif = acierto([max((j for j in range(len(norm)) if j != i),
                       key=lambda j: difflib.SequenceMatcher(None, norm[i], norm[j]).ratio())
                   for i in range(len(norm))])
    azar = np.mean([np.mean([bool(set(tags[i]) & set(tags[j])) for j in range(len(tags)) if j != i])
                    for i in range(len(tags))])
    print(f"\n=== Búsqueda de respuestas parecidas (60 chats reales, dejando fuera el propio) ===")
    print(f"El chat recuperado comparte intención: MiniLM {mini:.0%} · difflib {dif:.0%} · al azar {azar:.0%}")

    # Frases nuevas -> chat real más parecido (formas de hablar que no están en el dataset)
    tn = list(nuevas["_tags"])
    En = enc.encode(list(nuevas["mensaje_cliente"]), normalize_embeddings=True)
    mini_n = np.mean([bool(set(tn[i]) & set(tags[j])) for i, j in enumerate((En @ E.T).argmax(axis=1))])
    nn = [normalizar_texto(t) for t in nuevas["mensaje_cliente"]]
    dif_n = np.mean([bool(set(tn[i]) & set(tags[max(range(len(norm)), key=lambda j: difflib.SequenceMatcher(
        None, nn[i], norm[j]).ratio())])) for i in range(len(nn))])
    print(f"Frases nuevas -> chat real recuperado comparte intención: MiniLM {mini_n:.0%} · difflib {dif_n:.0%}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
