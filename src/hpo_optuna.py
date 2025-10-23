import argparse, os, copy, math, warnings
from pathlib import Path
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import SuccessiveHalvingPruner
import torch
import torch.nn as nn
import sys
# Asegura que el directorio del archivo (src/) esté en sys.path para importar train_mnv3.py
sys.path.append(str(Path(__file__).resolve().parent))
import train_mnv3 as m   # usa make_dataloaders, build_model, validate_groupwise, set_seed

# Suprime algunos warnings de usuario para un log más limpio
warnings.filterwarnings("ignore", category=UserWarning)

def train_one_trial(trial, data_dir, img_size, workers):
    """
    Ejecuta un único trial de Optuna:
    - Muestra hiperparámetros (lr, wd, batch, accum).
    - Entrena en dos fases: HEAD (solo clasificador) y FT (fine-tuning total).
    - Devuelve el mejor F1 en validación y guarda atributos del trial (umbral, f1, épocas).
    """
    # ---- espacio de búsqueda (definición de hiperparámetros que Optuna optimiza)
    lr     = trial.suggest_float('lr', 1e-4, 3e-3, log=True)         # learning rate (escala log)
    wd     = trial.suggest_float('wd', 1e-6, 5e-4, log=True)         # weight decay (regularización L2)
    batch  = trial.suggest_categorical('batch', [8, 12, 16])         # tamaño de batch
    accum  = trial.suggest_categorical('accum', [1, 2, 4])           # acumulación de gradiente

    # épocas cortas para HPO (rápidas para evaluar muchos trials)
    epochs_head = 1
    epochs_ft   = 6
    patience    = 2   # paciencia para early stopping simple en FT

    # ---- dataloaders (reutiliza tu implementación con sampler por grupo)
    # Devuelve: train_dl, val_dl, test_dl (no se usa aquí), y conteo por clase para pos_weight
    train_dl, val_dl, _, class_counts = m.make_dataloaders(data_dir, img_size, batch, workers)
    pos_weight = (class_counts[0] / class_counts[1]) if (1 in class_counts and class_counts[1]>0) else 1.0

    # Selección de dispositivo y creación del modelo (MobileNetV3 de train_mnv3.build_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = m.build_model().to(device).to(memory_format=torch.channels_last)  # channels_last optimiza en GPU

    # AMP GradScaler si hay CUDA; pérdida binaria con pos_weight; optimizador AdamW; scheduler Cosine
    scaler     = torch.amp.GradScaler('cuda') if device.type=='cuda' else None
    criterion  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs_head+epochs_ft))

    def run_epoch(loader, train=True, accumulation=1):
        """
        Bucle de una época: soporte de AMP y acumulación de gradiente.
        - train=True: modo entrenamiento (backprop/step).
        - train=False: solo forward para validación.
        """
        if train: model.train()
        else: model.eval()
        tot, steps = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for xb, yb, _ in loader:
            xb = xb.to(device).to(memory_format=torch.channels_last)
            yb = yb.float().unsqueeze(1).to(device)
            # autocast habilitado solo si hay GPU; reduce costo de cómputo en FP16
            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                logits = model(xb)
                loss = criterion(logits, yb) / accumulation
            if train:
                # Backprop con soporte de AMP (si CUDA) o FP32
                if scaler:
                    scaler.scale(loss).backward()
                    if (steps+1) % accumulation == 0:
                        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                else:
                    loss.backward()
                    if (steps+1) % accumulation == 0:
                        optimizer.step(); optimizer.zero_grad(set_to_none=True)
            tot += loss.item() * accumulation
            steps += 1
        return tot / max(1, steps)

    # Variables para seguimiento del mejor modelo/umbral/métrica y paciencia
    best_f1, best_thr, best_state, wait = -1.0, 0.5, None, 0
    global_ep = 0  # contador global de épocas (HEAD + FT), usado para trial.report

    # ---- fase HEAD (entrena solo la "cabeza" / clasificador)
    for p in model.parameters(): p.requires_grad = False
    for p in model.get_classifier().parameters(): p.requires_grad = True
    for _ in range(epochs_head):
        _ = run_epoch(train_dl, train=True, accumulation=accum)
        # Validación agrupada por caso (usa validate_groupwise del módulo m)
        f1, thr, _metrics = m.validate_groupwise(model, val_dl, device)
        scheduler.step()
        global_ep += 1
        # Reporta a Optuna el valor objetivo (F1) para permitir pruning
        trial.report(f1, step=global_ep)
        # Actualiza mejor estado si mejora F1
        if f1 > best_f1: best_f1, best_thr, best_state, wait = f1, thr, copy.deepcopy(model.state_dict()), 0
        else: wait += 1
        # Pruning (termina el trial si no va prometedor)
        if trial.should_prune(): raise optuna.TrialPruned()

    # ---- fase FT (fine-tuning de TODA la red)
    for p in model.parameters(): p.requires_grad = True
    for _ in range(epochs_ft):
        _ = run_epoch(train_dl, train=True, accumulation=accum)
        f1, thr, _metrics = m.validate_groupwise(model, val_dl, device)
        scheduler.step()
        global_ep += 1
        trial.report(f1, step=global_ep)
        if f1 > best_f1: best_f1, best_thr, best_state, wait = f1, thr, copy.deepcopy(model.state_dict()), 0
        else: wait += 1
        if wait >= patience: break                 # early stopping simple si no mejora
        if trial.should_prune(): raise optuna.TrialPruned()

    # guarda info útil del trial como atributos para inspección posterior
    trial.set_user_attr('best_thr', float(best_thr))
    trial.set_user_attr('best_f1',  float(best_f1))
    trial.set_user_attr('epochs_run', global_ep)
    return best_f1  # Optuna maximiza este valor

def main():
    # Parser CLI: rutas, tamaño de imagen, workers, #trials, nombre de estudio y storage (sqlite)
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='./data')
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--workers',  type=int, default=2)
    ap.add_argument('--trials',   type=int, default=10)
    ap.add_argument('--study_name', type=str, default='mnv3_hpo')
    ap.add_argument('--storage', type=str, default='sqlite:///outputs/optuna_study.db')  # persiste resultados
    args = ap.parse_args()

    # Crea carpeta de outputs y fija seed global (para reproducibilidad de dataloaders, etc.)
    Path('outputs').mkdir(exist_ok=True)
    m.set_seed(42)

    # Define sampler (TPE multivariado) y pruner (Successive Halving ~ ASHA) para acelerar HPO
    sampler = TPESampler(seed=42, multivariate=True, group=True)
    pruner  = SuccessiveHalvingPruner(min_resource=2, reduction_factor=2)  # ASHA-like

    # Crea/recupera estudio de Optuna en SQLite (permite continuar sesiones y comparar resultados)
    study = optuna.create_study(direction='maximize',
                                sampler=sampler, pruner=pruner,
                                study_name=args.study_name,
                                storage=args.storage, load_if_exists=True)

    # Función objetivo que llama a train_one_trial y maneja OOM como pruning limpio
    def objective(trial):
        try:
            return train_one_trial(trial, Path(args.data_dir), args.img_size, args.workers)
        except RuntimeError as e:
            # Manejo limpio de OOM → prunea el trial para no abortar el estudio completo
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()
            raise

    # Corre la optimización de n_trials con barra de progreso
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    # Resumen del mejor trial
    print('\n=== BEST TRIAL ===')
    print('value (F1):', study.best_value)
    print('params:', study.best_params)
    print('attrs:', study.best_trial.user_attrs)

    # Sugerencia de comando final con mejores hiperparámetros (para entrenamiento completo)
    bp = study.best_params
    batch = bp.get('batch', 16); accum = bp.get('accum', 2)
    lr    = bp.get('lr', 1e-3);  wd    = bp.get('wd', 1e-4)
    print('\nEjecuta entrenamiento final con:')
    print(f'python .\\src\\train_mnv3.py --data_dir .\\data --img_size {args.img_size} --batch {batch} --accum {accum} '
          f'--epochs_head 2 --epochs_ft 15 --patience 5 --lr {lr} --wd {wd} --workers {args.workers} --outfile sample_submission.CSV')

if __name__ == '__main__':
    # Punto de entrada: inicia el estudio de Optuna
    main()
