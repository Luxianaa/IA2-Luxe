"""Validación cruzada (k-fold): BETO vs. reglas por palabras clave.

¿Qué hace?
    Divide los chats de data/dataset.csv en K grupos (por defecto 5). En cada vuelta
    reserva un grupo para evaluar y usa los demás para entrenar. Así cada chat se
    evalúa exactamente una vez y se obtiene el PROMEDIO ± DESVIACIÓN del F1, que es
    mucho más confiable que una sola división train/val cuando hay pocos datos.

    Los dos métodos se evalúan en los MISMOS grupos, con el mismo umbral y la misma
    regla de "al menos una intención" que usa la app, para que la comparación sea justa.

Uso:
    Local, solo reglas (no entrena nada, tarda segundos):
        python evaluar_cv.py --solo-reglas

    Google Colab con GPU (BETO + reglas, ~5-10 min con 60 chats):
        !pip install -q transformers scikit-learn pandas
        !python evaluar_cv.py

Salida: carpeta resultados_cv/ con resumen.json, por_fold.csv y por_clase.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer

from src.nlp_engine import INTENCIONES, NLPEngine

MODELO_BASE = "dccuchile/bert-base-spanish-wwm-cased"


def argumentos():
    p = argparse.ArgumentParser(description="Validación cruzada BETO vs. reglas")
    p.add_argument("--dataset", default="data/dataset.csv")
    p.add_argument("--salida", default="resultados_cv")
    p.add_argument("--k", type=int, default=5, help="número de grupos (folds)")
    p.add_argument("--solo-reglas", dest="solo_reglas", action="store_true",
                   help="evalúa solo el clasificador por palabras clave (sin entrenar BETO)")
    p.add_argument("--modelo_base", default=MODELO_BASE)
    # Sin parada temprana: dentro de cada vuelta no hay datos extra para elegir la época
    # sin mirar el grupo de prueba, así que se entrena un número fijo de épocas.
    p.add_argument("--epocas", type=int, default=15)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--umbral", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--paciencia", type=int, default=999)  # requerido por entrenar_modelo
    return p.parse_known_args()[0]


def cargar_dataset(ruta: str) -> pd.DataFrame:
    df = pd.read_csv(ruta, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df["_tags"] = [[t.strip() for t in str(f).split(",") if t.strip() in INTENCIONES]
                   for f in df["intenciones"]]
    return df[df["_tags"].map(len) > 0].reset_index(drop=True)


def crear_folds(df: pd.DataFrame, k: int, seed: int):
    """Grupos estratificados por la primera intención (si se puede)."""
    primera = df["_tags"].map(lambda t: t[0])
    if primera.value_counts().min() >= k:
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(df, primera)), "estratificado"
    return list(KFold(k, shuffle=True, random_state=seed).split(df)), "aleatorio"


def predecir_reglas(df: pd.DataFrame, ids) -> np.ndarray:
    motor = NLPEngine(forzar_fallback=True)
    pred = np.zeros((len(ids), len(INTENCIONES)), dtype=int)
    for fila, i in enumerate(ids):
        r = motor.analizar(df.at[i, "contexto_previo"], df.at[i, "mensaje_cliente"])
        for intencion in r["intenciones"]:
            pred[fila, INTENCIONES.index(intencion)] = 1
    return pred


def metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "por_clase": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "acierto_exacto": float((y_true == y_pred).all(axis=1).mean()),
    }


def main():
    args = argumentos()
    df = cargar_dataset(args.dataset)
    Y = MultiLabelBinarizer(classes=INTENCIONES).fit_transform(df["_tags"])
    folds, tipo = crear_folds(df, args.k, args.seed)
    print(f"Chats: {len(df)} | {args.k} grupos ({tipo}) de ~{len(df) // args.k} chats cada uno\n")

    metodos = ["reglas"] if args.solo_reglas else ["reglas", "beto"]
    if "beto" in metodos:
        import torch
        import train_colab as tc

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Dispositivo: {device}")
        if device.type != "cuda":
            print("⚠️  Sin GPU será MUY lento. En Colab: Entorno de ejecución > Cambiar tipo > GPU.\n")

    filas_fold, preds_totales = [], {m: np.zeros_like(Y) for m in metodos}
    for n, (tr, te) in enumerate(folds, start=1):
        print(f"── Grupo {n}/{args.k}: entrena con {len(tr)}, evalúa con {len(te)} ──")
        for metodo in metodos:
            t0 = time.time()
            if metodo == "reglas":
                pred = predecir_reglas(df, te)
            else:
                tc.fijar_semilla(args.seed + n)
                model, tok = tc.entrenar_modelo(df, Y, tr, None, args, device, verbose=False)
                probs, _ = tc.predecir(model, tc.crear_loader(df, Y, te, tok, args), device)
                pred = tc.binarizar(probs, args.umbral)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            preds_totales[metodo][te] = pred
            m = metricas(Y[te], pred)
            filas_fold.append({"grupo": n, "metodo": metodo, "f1_micro": m["f1_micro"],
                               "f1_macro": m["f1_macro"], "acierto_exacto": m["acierto_exacto"],
                               "segundos": round(time.time() - t0, 1)})
            print(f"   {metodo:<7} F1 micro {m['f1_micro']:.3f} | F1 macro {m['f1_macro']:.3f}"
                  f" | acierto exacto {m['acierto_exacto']:.0%}")

    por_fold = pd.DataFrame(filas_fold)
    resumen = {}
    print("\n================ RESUMEN (promedio ± desviación entre grupos) ================")
    for metodo in metodos:
        d = por_fold[por_fold.metodo == metodo]
        resumen[metodo] = {
            c: {"media": round(d[c].mean(), 4), "desv": round(d[c].std(ddof=1), 4)}
            for c in ("f1_micro", "f1_macro", "acierto_exacto")
        }
        r = resumen[metodo]
        print(f"{metodo:<7} F1 micro {r['f1_micro']['media']:.3f} ± {r['f1_micro']['desv']:.3f}"
              f" | F1 macro {r['f1_macro']['media']:.3f} ± {r['f1_macro']['desv']:.3f}"
              f" | acierto exacto {r['acierto_exacto']['media']:.0%}")

    # F1 por intención calculado con las predicciones de todos los grupos juntas
    por_clase = pd.DataFrame(
        {metodo: f1_score(Y, preds_totales[metodo], average=None, zero_division=0)
         for metodo in metodos}, index=INTENCIONES)
    por_clase.insert(0, "ejemplos", Y.sum(axis=0))
    print("\nF1 por intención:\n" + por_clase.round(3).to_string())

    if len(metodos) == 2:
        a = por_fold[por_fold.metodo == "beto"]["f1_micro"].to_numpy()
        b = por_fold[por_fold.metodo == "reglas"]["f1_micro"].to_numpy()
        dif = a - b
        gana = int((dif > 0).sum())
        print(f"\nBETO − reglas (F1 micro): {dif.mean():+.3f} en promedio; "
              f"BETO gana en {gana} de {len(dif)} grupos.")
        if abs(dif.mean()) < dif.std(ddof=1):
            print("➡️  La diferencia es menor que la variación entre grupos: con estos datos "
                  "NO se puede afirmar que un método sea mejor. Hacen falta más chats etiquetados.")
        else:
            mejor = "BETO rinde" if dif.mean() > 0 else "las reglas rinden"
            print(f"➡️  La diferencia supera la variación entre grupos: {mejor} mejor de forma consistente.")
        resumen["diferencia_beto_menos_reglas"] = {"media": round(dif.mean(), 4),
                                                   "desv": round(dif.std(ddof=1), 4),
                                                   "grupos_gana_beto": gana}

    print("\nℹ️  Ojo: las palabras clave se escribieron mirando este mismo dataset, así que su "
          "puntaje es optimista. BETO, en cambio, nunca ve el grupo con el que se le evalúa.")

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    por_fold.to_csv(salida / "por_fold.csv", index=False, encoding="utf-8-sig")
    por_clase.to_csv(salida / "por_clase.csv", encoding="utf-8-sig")
    resumen["config"] = {"k": args.k, "tipo_division": tipo, "chats": len(df), "umbral": args.umbral,
                         **({} if args.solo_reglas else {"epocas": args.epocas, "lr": args.lr,
                                                         "batch": args.batch, "modelo_base": args.modelo_base})}
    (salida / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Resultados guardados en {salida.resolve()}")


if __name__ == "__main__":
    main()
