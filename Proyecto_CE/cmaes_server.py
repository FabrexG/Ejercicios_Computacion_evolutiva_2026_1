"""
CMA-ES OPTIMIZADO PARA RoBERTuito
Humor Detection en Español

OPTIMIZACIONES:
- AMP
- Dynamic Padding
- CLS + Mean Pooling
- Arquitectura pequeña
- Early pruning
- Gradient checkpointing
- torch.compile()
- Search space reducido
- DataLoader optimizado
- Menos overfitting
- Mucho más rápido
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import json
import cma

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)

from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from pysentimiento.preprocessing import preprocess_tweet

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "pysentimiento/robertuito-base-cased"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Dispositivo: {DEVICE}")

FAST_EPOCHS = 4
FULL_EPOCHS = 20

BATCH_SIZE = 32

DATA_PATH = "./Datasets/dataset_humor_train.json"

# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================================
# TOKENIZER
# ============================================================

print("Cargando tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt"
)

# ============================================================
# DATASET
# ============================================================

class HumorDataset(Dataset):

    def __init__(
        self,
        texts,
        labels=None,
        max_len=64
    ):

        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = preprocess_tweet(
            str(self.texts[idx]),
            lang="es"
        )

        enc = tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True
        )

        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"]
        }

        if self.labels is not None:
            item["labels"] = int(self.labels[idx])

        return item

# ============================================================
# MODELO
# ============================================================

class HumorModel(nn.Module):

    def __init__(
        self,
        C1=512,
        C2=128,
        drop1=0.2,
        drop2=0.1
    ):

        super().__init__()

        self.roberta = AutoModel.from_pretrained(
            MODEL_NAME,
            attn_implementation="sdpa"
        )

        self.roberta.gradient_checkpointing_enable()

        hidden_size = 768

        # CLS + Mean
        input_dim = hidden_size * 2

        self.classifier = nn.Sequential(

            nn.Linear(input_dim, C1),

            nn.LayerNorm(C1),
            nn.GELU(),
            nn.Dropout(drop1),

            nn.Linear(C1, C2),

            nn.LayerNorm(C2),
            nn.GELU(),
            nn.Dropout(drop2),

            nn.Linear(C2, 2)
        )

    def mean_pooling(
        self,
        hidden_states,
        attention_mask
    ):

        mask = attention_mask.unsqueeze(-1).expand(
            hidden_states.size()
        ).float()

        summed = torch.sum(
            hidden_states * mask,
            dim=1
        )

        summed_mask = torch.clamp(
            mask.sum(dim=1),
            min=1e-9
        )

        return summed / summed_mask

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state

        cls_embedding = hidden_states[:, 0]

        mean_embedding = self.mean_pooling(
            hidden_states,
            attention_mask
        )

        x = torch.cat(
            [cls_embedding, mean_embedding],
            dim=1
        )

        logits = self.classifier(x)

        return logits

# ============================================================
# DATOS
# ============================================================

print("Cargando dataset...")

df = pd.read_json(
    DATA_PATH,
    lines=True
)

texts = df["text"].values
labels = df["klass"].values

X_train, X_val, y_train, y_val = train_test_split(
    texts,
    labels,
    test_size=0.15,
    stratify=labels,
    random_state=42
)

print(f"Train: {len(X_train)}")
print(f"Val: {len(X_val)}")

# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_and_evaluate(
    params,
    epochs=FAST_EPOCHS
):

    set_seed(42)

    lr = params["lr"]
    wd = params["weight_decay"]
    warmup = params["warmup_ratio"]

    drop1 = params["drop1"]
    drop2 = params["drop2"]

    C1 = params["C1"]
    C2 = params["C2"]

    max_len = params["max_len"]

    train_dataset = HumorDataset(
        X_train,
        y_train,
        max_len=max_len
    )

    val_dataset = HumorDataset(
        X_val,
        y_val,
        max_len=max_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True
    )

    model = HumorModel(
        C1=C1,
        C2=C2,
        drop1=drop1,
        drop2=drop2
    )

    model = torch.compile(model)

    model = model.to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd
    )

    total_steps = len(train_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup),
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler('cuda')

    best_f1 = 0.0

    # ========================================================
    # TRAIN
    # ========================================================

    for epoch in range(epochs):

        model.train()

        for batch in train_loader:

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                loss = loss_fn(
                    logits,
                    labels
                )

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        preds = []
        reals = []

        with torch.no_grad():

            for batch in val_loader:

                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                p = torch.argmax(
                    logits,
                    dim=1
                )

                preds.extend(
                    p.cpu().numpy()
                )

                reals.extend(
                    labels.cpu().numpy()
                )

        f1 = f1_score(
            reals,
            preds,
            average="macro"
        )

        if f1 > best_f1:
            best_f1 = f1

        # ====================================================
        # EARLY PRUNING
        # ====================================================

        if epoch >= 1 and f1 < 0.55:

            del model
            torch.cuda.empty_cache()
            gc.collect()

            return 0.0

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return best_f1

# ============================================================
# DECODER
# ============================================================

def decode_params(x):

    s = 1 / (1 + np.exp(-x))

    params = {

        # Learning rate
        "lr": float(
            np.exp(
                np.log(1e-5)
                + s[0] * (
                    np.log(5e-5)
                    - np.log(1e-5)
                )
            )
        ),

        # Weight decay
        "weight_decay": float(
            np.exp(
                np.log(1e-4)
                + s[1] * (
                    np.log(5e-2)
                    - np.log(1e-4)
                )
            )
        ),

        # Warmup
        "warmup_ratio": float(
            0.02 + s[2] * 0.18
        ),

        # Dropout 1
        "drop1": float(
            0.05 + s[3] * 0.35
        ),

        # Dropout 2
        "drop2": float(
            0.05 + s[4] * 0.25
        ),

        # Hidden layer 1
        "C1": int(
            128 + s[5] * (768 - 128)
        ),

        # Hidden layer 2
        "C2": int(
            32 + s[6] * (256 - 32)
        ),

        # MAX_LEN
        "max_len": int(
            32 + s[7] * (96 - 32)
        )
    }

    return params

# ============================================================
# CMA-ES
# ============================================================

eval_count = [0]

best_global = {
    "f1": 0.0,
    "params": None
}

def objective(x):

    params = decode_params(x)

    eval_count[0] += 1

    print("\n" + "="*60)
    print(f"EVALUACIÓN {eval_count[0]}")
    print("="*60)

    for k, v in params.items():

        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    try:

        f1 = train_and_evaluate(
            params,
            epochs=FAST_EPOCHS
        )

        print(f"\nF1: {f1:.4f}")

    except RuntimeError as e:

        print(f"\nERROR: {e}")

        return 1.0

    if f1 > best_global["f1"]:

        best_global["f1"] = f1
        best_global["params"] = params.copy()

        print("\nNUEVO MEJOR MODELO")

        with open(
            "best_cmaes_params.json",
            "w"
        ) as f:

            json.dump(
                best_global,
                f,
                indent=2
            )

    return -f1

# ============================================================
# RUN CMA-ES
# ============================================================

print("\n")
print("="*70)
print("INICIANDO CMA-ES")
print("="*70)

x0 = np.zeros(8)

sigma0 = 0.5

options = {

    "maxiter": 20,

    "popsize": 6,

    "bounds": [
        [-4] * 8,
        [4] * 8
    ],

    "verb_disp": 1,
}

es = cma.CMAEvolutionStrategy(
    x0,
    sigma0,
    options
)

es.optimize(objective)

# ============================================================
# RESULTADOS
# ============================================================

print("\n")
print("="*70)
print("OPTIMIZACIÓN FINALIZADA")
print("="*70)

print(f"\nMEJOR F1: {best_global['f1']:.4f}")

print("\nMEJORES PARÁMETROS:\n")

for k, v in best_global["params"].items():

    if isinstance(v, float):
        print(f"{k}: {v:.6f}")
    else:
        print(f"{k}: {v}")

# ============================================================
# FINAL TRAIN
# ============================================================

print("\n")
print("="*70)
print("ENTRENAMIENTO FINAL")
print("="*70)

final_f1 = train_and_evaluate(
    best_global["params"],
    epochs=FULL_EPOCHS
)

print(f"\nF1 FINAL: {final_f1:.4f}")

best_global["final_f1"] = final_f1

with open(
    "best_cmaes_params.json",
    "w"
) as f:

    json.dump(
        best_global,
        f,
        indent=2
    )

print("\nParámetros guardados.")