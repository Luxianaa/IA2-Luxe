"""Fine-tuning de BETO para las intenciones de Hilo (multi-etiqueta).

⚠️ Ejecutar en Google Colab con GPU (Entorno de ejecución > Cambiar tipo > T4 GPU).
   NO está pensado para correr en la máquina local.

Pasos en Colab:
    1. Sube y descomprime hilo_colab.zip (trae este script, src/ y data/dataset.csv):
       !unzip -o hilo_colab.zip -d hilo-merci
       %cd hilo-merci
    2. !pip install -q transformers scikit-learn pandas
    3. !python train_colab.py
    4. Descarga modelo_hilo_final.zip (panel de archivos) y descomprímelo en
       hilo-merci/models/modelo_hilo_final/

Entrada : data/dataset.csv  (chats reales: contexto_previo, mensaje_cliente, respuesta_real, intenciones)
          data/dataset_sintetico.csv (opcional, de generar_sinteticos.py: SOLO para entrenar;
          la validación usa únicamente chats reales)
Salida  : models/modelo_hilo_final/  (modelo, tokenizer, labels.json, mlb.pkl)

Las funciones cargar_datos / entrenar_modelo / predecir las reutiliza evaluar_cv.py.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

MODELO_BASE = "dccuchile/bert-base-spanish-wwm-cased"
INTENCIONES = ["saludo_catalogo", "cambio_talla", "envio_plazos",
               "disponibilidad", "precio", "datos_envio", "tienda_fisica"]


def argumentos():
    p = argparse.ArgumentParser(description="Fine-tuning BETO multi-etiqueta para Hilo")
    p.add_argument("--dataset", default="data/dataset.csv")
    p.add_argument("--sinteticos", default="data/dataset_sintetico.csv",
                   help="chats sintéticos: SOLO se suman al entrenamiento ('' para no usarlos)")
    p.add_argument("--salida", default="models/modelo_hilo_final")
    p.add_argument("--modelo_base", default=MODELO_BASE)
    p.add_argument("--epocas", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--umbral", type=float, default=0.5)
    p.add_argument("--paciencia", type=int, default=5, help="épocas sin mejora antes de parar")
    p.add_argument("--seed", type=int, default=42)
    # parse_known_args: tolera los argumentos extra que inyecta Colab/Jupyter
    return p.parse_known_args()[0]


def fijar_semilla(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def buscar_dataset(ruta: str) -> Path:
    for c in (Path(ruta), Path("dataset.csv"), Path("/content/dataset.csv"),
              Path("/content/data/dataset.csv")):
        if c.exists():
            return c
    raise FileNotFoundError(f"No se encontró {ruta}. Súbelo a Colab dentro de data/.")


def cargar_datos(ruta: Path) -> pd.DataFrame:
    """Lee dataset.csv y deja en df['_tags'] la lista de intenciones válidas."""
    df = pd.read_csv(ruta, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    faltan = {"contexto_previo", "mensaje_cliente", "intenciones"} - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en dataset.csv: {faltan}")

    etiquetas = []
    for fila in df["intenciones"]:
        tags = [t.strip() for t in str(fila).split(",") if t.strip()]
        desconocidas = [t for t in tags if t not in INTENCIONES]
        if desconocidas:
            print(f"⚠️  Etiquetas ignoradas (no son de las {len(INTENCIONES)} definitivas): {desconocidas}")
        etiquetas.append([t for t in tags if t in INTENCIONES])
    df["_tags"] = etiquetas
    vacias = df["_tags"].map(len) == 0
    if vacias.any():
        print(f"⚠️  Se descartan {int(vacias.sum())} filas sin intención válida.")
        df = df[~vacias].reset_index(drop=True)
    return df


class ChatsDataset(Dataset):
    def __init__(self, contextos, mensajes, y, tokenizer, max_len):
        # Mismo formato que src/nlp_engine.py: par (contexto_previo, mensaje_cliente)
        self.enc = tokenizer(list(contextos), list(mensajes), truncation=True,
                             max_length=max_len, padding="max_length", return_tensors="pt")
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.y[i]
        return item


def crear_loader(df, Y, ids, tokenizer, args, shuffle=False) -> DataLoader:
    ds = ChatsDataset(df.loc[ids, "contexto_previo"], df.loc[ids, "mensaje_cliente"],
                      Y[ids], tokenizer, args.max_len)
    return DataLoader(ds, batch_size=args.batch, shuffle=shuffle)


@torch.no_grad()
def predecir(model, loader, device):
    model.eval()
    probs, reales = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        probs.append(torch.sigmoid(logits).cpu().numpy())
        reales.append(labels.numpy())
    return np.vstack(probs), np.vstack(reales)


def binarizar(probs: np.ndarray, umbral: float) -> np.ndarray:
    """Umbral + 'al menos una intención' (igual que NLPEngine.analizar en la app)."""
    pred = (probs >= umbral).astype(int)
    vacias = pred.sum(axis=1) == 0
    pred[vacias, probs[vacias].argmax(axis=1)] = 1
    return pred


def entrenar_modelo(df, Y, tr_idx, va_idx, args, device, verbose=True):
    """Afina BETO con las filas tr_idx.

    Si va_idx no es None, evalúa cada época sobre esas filas, se queda con la mejor
    época (F1 micro) y aplica parada temprana. Si es None, entrena todas las épocas.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.modelo_base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.modelo_base,
        num_labels=len(INTENCIONES),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(INTENCIONES)),
        label2id={l: i for i, l in enumerate(INTENCIONES)},
    ).to(device)

    train_loader = crear_loader(df, Y, tr_idx, tokenizer, args, shuffle=True)
    val_loader = crear_loader(df, Y, va_idx, tokenizer, args) if va_idx is not None else None

    # BCEWithLogitsLoss con pos_weight para compensar clases poco frecuentes
    pos = Y[tr_idx].sum(axis=0)
    neg = len(tr_idx) - pos
    pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, 10.0), dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    pasos = len(train_loader) * args.epocas
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * pasos), pasos)

    mejor_f1, mejor_estado, sin_mejora = -1.0, None, 0
    for epoca in range(1, args.epocas + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = loss_fn(model(**batch).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total += loss.item()
        linea = f"  Época {epoca:02d} | loss {total / len(train_loader):.4f}"

        if val_loader is not None:
            probs, reales = predecir(model, val_loader, device)
            pred = (probs >= args.umbral).astype(int)
            f1_mi = f1_score(reales, pred, average="micro", zero_division=0)
            f1_ma = f1_score(reales, pred, average="macro", zero_division=0)
            linea += f" | F1 micro {f1_mi:.4f} | F1 macro {f1_ma:.4f}"
            if f1_mi > mejor_f1:
                mejor_f1, sin_mejora = f1_mi, 0
                mejor_estado = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                sin_mejora += 1
        if verbose:
            print(linea)
        if val_loader is not None and sin_mejora >= args.paciencia:
            print(f"  Parada temprana en la época {epoca}.")
            break

    if mejor_estado is not None:
        model.load_state_dict(mejor_estado)
    return model, tokenizer


def main():
    args = argumentos()
    fijar_semilla(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type != "cuda":
        print("⚠️  No hay GPU. En Colab: Entorno de ejecución > Cambiar tipo > GPU.")

    reales = cargar_datos(buscar_dataset(args.dataset))
    reales["_origen"] = "real"
    n_reales = len(reales)

    # División estratificada por la primera etiqueta (si alguna clase es muy rara, sin estratificar)
    idx = np.arange(n_reales)
    primera = reales["_tags"].map(lambda t: t[0])
    try:
        tr_idx, va_idx = train_test_split(idx, test_size=args.val_size,
                                          random_state=args.seed, stratify=primera)
    except ValueError:
        tr_idx, va_idx = train_test_split(idx, test_size=args.val_size, random_state=args.seed)

    # Sintéticos: se agregan SOLO al entrenamiento. La validación son únicamente chats reales.
    ruta_sint = next((c for c in (Path(args.sinteticos), Path("dataset_sintetico.csv"),
                                  Path("/content/data/dataset_sintetico.csv"))
                      if args.sinteticos and c.exists()), None)
    if ruta_sint is not None:
        sint = cargar_datos(ruta_sint)
        sint["_origen"] = "sintetico"
        df = pd.concat([reales, sint], ignore_index=True)
        tr_idx = np.concatenate([tr_idx, np.arange(n_reales, len(df))])
        print(f"Chats reales: {n_reales} · sintéticos (solo entrenamiento): {len(sint)}")
    else:
        df = reales
        print(f"Chats reales: {n_reales} (sin sintéticos)")

    mlb = MultiLabelBinarizer(classes=INTENCIONES)
    Y = mlb.fit_transform(df["_tags"])
    print(f"Entrenamiento: {len(tr_idx)} · validación (solo reales): {len(va_idx)}")
    print("Distribución total:", dict(zip(INTENCIONES, Y.sum(axis=0).tolist())))

    model, tokenizer = entrenar_modelo(df, Y, tr_idx, va_idx, args, device)

    probs, reales = predecir(model, crear_loader(df, Y, va_idx, tokenizer, args), device)
    pred = (probs >= args.umbral).astype(int)
    f1_mi = f1_score(reales, pred, average="micro", zero_division=0)
    f1_ma = f1_score(reales, pred, average="macro", zero_division=0)
    reporte = classification_report(reales, pred, target_names=INTENCIONES, zero_division=0)

    print("\n================ RESULTADOS (validación) ================")
    print(f"F1 Micro: {f1_mi:.4f}")
    print(f"F1 Macro: {f1_ma:.4f}")
    print("\nReporte por clase:\n" + reporte)
    print("ℹ️  Con pocos datos esta cifra depende mucho del azar: usa evaluar_cv.py "
          "para una estimación más robusta.")

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    model.to("cpu").save_pretrained(salida)
    tokenizer.save_pretrained(salida)
    (salida / "labels.json").write_text(json.dumps({
        "labels": INTENCIONES,
        "umbral": args.umbral,
        "max_len": args.max_len,
        "modelo_base": args.modelo_base,
        "metricas_validacion": {"f1_micro": round(f1_mi, 4), "f1_macro": round(f1_ma, 4)},
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_sinteticos": int(len(df) - n_reales),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(salida / "mlb.pkl", "wb") as f:
        pickle.dump(mlb, f)
    (salida / "reporte_validacion.txt").write_text(
        f"F1 Micro: {f1_mi:.4f}\nF1 Macro: {f1_ma:.4f}\n\n{reporte}", encoding="utf-8")
    print(f"\n✅ Modelo guardado en {salida.resolve()}")

    zip_path = shutil.make_archive("modelo_hilo_final", "zip", root_dir=salida)
    print(f"📦 Zip listo: {zip_path}")
    try:
        from google.colab import files  # type: ignore

        files.download(zip_path)
    except Exception:  # noqa: BLE001 - fuera de Colab o ejecutado con !python
        print("Descarga el zip desde el panel de archivos de Colab (icono de carpeta).")


if __name__ == "__main__":
    main()
