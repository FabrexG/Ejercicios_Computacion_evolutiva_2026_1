"""
Optimización de Hiperparámetros con CMA-ES
para clasificador de humor en español (RoBERTuito)

Requiere: pip install cma
CMA-ES (Hansen, 2016) — estado del arte en optimización black-box.
Busca simultáneamente: LR, dropout, weight_decay, warmup_ratio, capas del clasificador.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import cma
import json
import warnings
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from pysentimiento.preprocessing import preprocess_tweet

warnings.filterwarnings("ignore")

# ─── Reproducibilidad ────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ─── Config Global ────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 100
BATCH_SIZE = 16
# Para CMA-ES usamos epochs reducidos para acelerar evaluación
FAST_EPOCHS = 4
FULL_EPOCHS = 10
DATA_PATH = "./Datasets/dataset_humor_train.json"
MODEL_NAME = "pysentimiento/robertuito-base-cased"

print(f"Dispositivo: {DEVICE}")

# ─── Tokenizer (carga única) ──────────────────────────────────────────────────

print("Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ─── Dataset ─────────────────────────────────────────────────────────────────

class HumorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = preprocess_tweet(str(self.texts[idx]), lang="es")
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# ─── Modelo ───────────────────────────────────────────────────────────────────

class HumorModel(nn.Module):
    """
    RoBERTuito con Mean+Max pooling y clasificador configurable.
    dropout1, dropout2 y C1, C2 son hiperparámetros a optimizar.
    """
    def __init__(self, C1=512, C2=128, dropout1=0.4, dropout2=0.25):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(MODEL_NAME)
        input_dim = 768 * 2  # mean + max pooling

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, C1),
            nn.BatchNorm1d(C1),
            nn.GELU(),
            nn.Dropout(dropout1),
            nn.Linear(C1, C2),
            nn.BatchNorm1d(C2),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(C2, 2),
        )

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state

        mask_exp = attention_mask.unsqueeze(-1).expand(h.size()).float()
        mean_pooled = torch.sum(h * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)

        h_cloned = h.clone()
        h_cloned[mask_exp == 0] = -1e9
        max_pooled = torch.max(h_cloned, 1)[0]

        concat = torch.cat((mean_pooled, max_pooled), 1)
        return self.classifier(concat)

# ─── Entrenamiento rápido para CMA-ES ────────────────────────────────────────

def train_and_evaluate(params: dict, X_tr, y_tr, X_val, y_val, epochs=FAST_EPOCHS) -> float:
    """
    Entrena el modelo con los parámetros dados y retorna F1 macro en validación.
    Retorna negativo (CMA-ES minimiza).
    """
    set_seed(42)

    lr         = params["lr"]
    dropout1   = params["dropout1"]
    dropout2   = params["dropout2"]
    weight_dec = params["weight_decay"]
    warmup_r   = params["warmup_ratio"]
    C1         = params["C1"]
    C2         = params["C2"]

    train_ds = HumorDataset(X_tr, y_tr, tokenizer)
    val_ds   = HumorDataset(X_val, y_val, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = HumorModel(C1=C1, C2=C2, dropout1=dropout1, dropout2=dropout2).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_dec)
    total_steps = len(train_dl) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_r),
        num_training_steps=total_steps,
    )
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        for batch in train_dl:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            tgts = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(ids, mask)
            loss = loss_fn(logits, tgts)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        preds, reals = [], []
        with torch.no_grad():
            for batch in val_dl:
                ids  = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                tgts = batch["labels"].to(DEVICE)
                logits = model(ids, mask)
                _, p = torch.max(logits, dim=1)
                preds.extend(p.cpu().tolist())
                reals.extend(tgts.cpu().tolist())

        f1 = f1_score(reals, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1

    del model
    torch.cuda.empty_cache()
    return best_f1

# ─── Decodificador de vector CMA-ES → parámetros ─────────────────────────────

def decode_params(x: np.ndarray) -> dict:
    """
    CMA-ES trabaja en espacio continuo. Mapeamos cada dimensión a rangos válidos.
    x tiene 7 dimensiones, todas ≈ [0, 1] tras sigmoid.
    """
    # Sigmoid para contener en [0,1]
    s = 1 / (1 + np.exp(-x))

    params = {
        # LR: [5e-6, 5e-5] en escala log
        "lr": float(np.exp(np.log(5e-6) + s[0] * (np.log(5e-5) - np.log(5e-6)))),
        # Dropout capas
        "dropout1": float(0.1 + s[1] * 0.5),   # [0.1, 0.6]
        "dropout2": float(0.05 + s[2] * 0.4),  # [0.05, 0.45]
        # Weight decay
        "weight_decay": float(np.exp(np.log(1e-4) + s[3] * (np.log(1e-1) - np.log(1e-4)))),
        # Warmup ratio
        "warmup_ratio": float(0.0 + s[4] * 0.2),  # [0.0, 0.2]
        # Dimensiones del clasificador (discretizadas)
        "C1": int(256 + s[5] * (4096 - 256)),
        "C2": int(64  + s[6] * (1024 - 64)),
    }
    return params

# ─── Cargar Datos ─────────────────────────────────────────────────────────────

print("Cargando datos...")
df = pd.read_json(DATA_PATH, lines=True)
textos   = df["text"].values
etiquetas = df["klass"].values

X_tr, X_val, y_tr, y_val = train_test_split(
    textos, etiquetas, test_size=0.15, stratify=etiquetas, random_state=42
)

print(f"Train: {len(X_tr)} | Val: {len(X_val)}")

# ─── Función Objetivo para CMA-ES ────────────────────────────────────────────

eval_count = [0]
best_global = {"f1": 0.0, "params": None}

def objective(x: np.ndarray) -> float:
    """CMA-ES minimiza → retornamos -F1."""
    params = decode_params(x)
    eval_count[0] += 1

    print(f"\n[Eval {eval_count[0]}] Probando parámetros:")
    for k, v in params.items():
        if k == "lr":
            print(f"  {k}: {v:.2e}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    try:
        f1 = train_and_evaluate(params, X_tr, y_tr, X_val, y_val, epochs=FAST_EPOCHS)
        print(f"  → F1 Macro: {f1:.4f}")
    except RuntimeError as e:
        print(f"  ✗ Error (OOM o similar): {e}")
        return 1.0  # Penalización máxima

    if f1 > best_global["f1"]:
        best_global["f1"] = f1
        best_global["params"] = params.copy()
        print(f"  ★ Nuevo mejor: {f1:.4f}")
        # Guardar checkpoint
        with open("cmaes_best_params.json", "w") as f_out:
            json.dump(best_global, f_out, indent=2)

    return -f1  # Negativo porque CMA-ES minimiza

# ─── Ejecutar CMA-ES ──────────────────────────────────────────────────────────

print("\n" + "="*60)
print("INICIANDO OPTIMIZACIÓN CMA-ES")
print("="*60)

# Punto de inicio: valores neutros (x=0 → sigmoid=0.5 → punto medio de rangos)
x0 = np.zeros(7)

# sigma0: desviación estándar inicial. 0.5 es conservador y estable.
sigma0 = 0.5

# Opciones de CMA-ES
options = {
    "maxiter":         30,    # Iteraciones máximas (cada una evalúa ~lambda=10 candidatos)
    "popsize":         8,     # Tamaño de población (lambda). Más = más robusto, más costoso.
    "tolx":            1e-4,  # Tolerancia en cambio de x
    "tolfun":          1e-4,  # Tolerancia en cambio de función
    "verbose":         1,
    "verb_log":        0,
    "CMA_stds":        [1.0] * 7,  # Escala inicial por dimensión
    # Bounds suaves (la sigmoid ya contiene, pero CMA-ES las respeta igual)
    "bounds":          [[-4]*7, [4]*7],
}

es = cma.CMAEvolutionStrategy(x0, sigma0, options)

print(f"\nPresupuesto total estimado: ~{options['maxiter'] * options['popsize']} evaluaciones")
print(f"Épocas por evaluación: {FAST_EPOCHS}")
print("="*60 + "\n")

es.optimize(objective)

# ─── Resultados ───────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("OPTIMIZACIÓN FINALIZADA")
print("="*60)
print(f"\nMejor F1 (eval rápida, {FAST_EPOCHS} epochs): {best_global['f1']:.4f}")
print("\nMejores parámetros:")
for k, v in best_global["params"].items():
    if k == "lr":
        print(f"  {k}: {v:.2e}")
    elif isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

# ─── Reentrenamiento final con mejores parámetros ────────────────────────────

print(f"\n{'='*60}")
print(f"REENTRENAMIENTO FINAL ({FULL_EPOCHS} epochs con mejores parámetros)")
print("="*60)

best_params = best_global["params"]
f1_final = train_and_evaluate(best_params, X_tr, y_tr, X_val, y_val, epochs=FULL_EPOCHS)
print(f"\nF1 Final (entrenamiento completo): {f1_final:.4f}")

# Guardar resultados finales
best_global["f1_full_training"] = f1_final
with open("cmaes_best_params.json", "w") as f_out:
    json.dump(best_global, f_out, indent=2)

print("\nParámetros guardados en: cmaes_best_params.json")
print("Usa estos parámetros en ensemble_humor.py para el mejor rendimiento.")
