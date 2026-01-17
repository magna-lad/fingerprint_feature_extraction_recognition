import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import os
import random
import matplotlib.pyplot as plt
import gc 
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# IMPORTS
from graph_minutiae import GraphMinutiae
from load_save import load_users_dictionary
from xgboost_feature_extractor import create_feature_vector_for_pair
from cnn_model import DeeperCNN, FingerprintTextureDataset, EarlyStopping



SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# --- CONFIGURATION ---
NUM_CNN_MODELS = 3      # Ensemble Size
BATCH_SIZE = 64
EPOCHS = 35             # Slightly reduced epochs since we train 3 models
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_FILE = '/kaggle/working/fingerprint_feature_extraction_recognition/biometric_cache/processed_data.pkl'

OUTPUT_DIR = "."        # In Kaggle, this maps to /kaggle/working/

def calculate_eer(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


def prepare_xgb_features(pairs, desc):
    """Extracts geometric graph features for XGBoost."""
    X, y = [], []
    for g1, g2, label in tqdm(pairs, desc=desc):
        X.append(create_feature_vector_for_pair(g1, g2))
        y.append(label)
    return np.array(X), np.array(y)

def train_one_cnn_model(model_idx, train_loader, val_loader):
    """Helper function to train a single CNN instance."""
    print(f"\n--- Training CNN Ensemble Member {model_idx+1}/{NUM_CNN_MODELS} ---")
    
    model = DeeperCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=EPOCHS, steps_per_epoch=len(train_loader))
    
    save_path = os.path.join(OUTPUT_DIR, f'cnn_v{model_idx}.pth')
    stopper = EarlyStopping(patience=8, path=save_path)
    
    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        
        # TQDM bar
        train_bar = tqdm(train_loader, desc=f"M{model_idx+1} Ep {epoch+1}/{EPOCHS}", leave=False)
        for img1, img2, label in train_bar:
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            logits = model(img1, img2)
            loss = criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # stopping exploding gradients
            optimizer.step()
            scheduler.step()
            t_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Validation
        model.eval()
        v_loss = 0
        preds, labels_list = [], []
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
                logits = model(img1, img2)
                v_loss += criterion(logits, label).item()
                preds.extend(torch.sigmoid(logits).cpu().numpy())
                labels_list.extend(label.cpu().numpy())
        
        avg_v_loss = v_loss / len(val_loader)
        v_auc = auc(roc_curve(labels_list, preds)[0], roc_curve(labels_list, preds)[1])
        
        print(f"  M{model_idx+1} Ep {epoch+1}: Val Loss {avg_v_loss:.4f} | Val AUC {v_auc:.4f}")
        
        stopper(avg_v_loss, model)
        if stopper.early_stop:
            print("  > Early Stopping.")
            break
    
    # Clean up GPU memory
    del model
    del optimizer
    torch.cuda.empty_cache()
    gc.collect()
    return save_path

def run_hybrid_system():
    print("="*60); print(" KAGGLE ENSEMBLE PIPELINE (3x CNN + XGB) "); print("="*60)
    
    # [1] LOAD DATA
    print("\n[1] Preparing Data...")
    users = load_users_dictionary(DATA_FILE, True) 
    if users is None: return

    analyzer = GraphMinutiae(users)
    analyzer.graph_maker()
    all_pairs = analyzer.create_graph_pairs(num_impostors_per_genuine=3)
    
    # Split
    train_u, val_u, test_u = analyzer.get_user_splits(train_ratio=0.70, val_ratio=0.15) 
    train_pairs, val_pairs, test_pairs = analyzer.split_pairs_by_user(all_pairs, train_u, val_u, test_u)
    
    if len(val_pairs) == 0: print("CRITICAL ERROR: Val set empty."); return

    # [2] TRAIN XGBOOST
    print("\n[2] Training XGBoost (Geometry Branch)...")
    X_train, y_train = prepare_xgb_features(train_pairs, "Extract Train")
    X_val, y_val = prepare_xgb_features(val_pairs, "Extract Val")
    X_test, y_test = prepare_xgb_features(test_pairs, "Extract Test")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    xgb_model = xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.02, 
                                  eval_metric='logloss', early_stopping_rounds=50, n_jobs=-1,random_state=SEED)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Store XGBoost Predictions
    xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
    xgb_test_probs = xgb_model.predict_proba(X_test)[:, 1]
    print(f"   > XGBoost Best Iteration: {xgb_model.best_iteration}")
    
    # [3] TRAIN CNN ENSEMBLE
    print(f"\n[3] Training CNN Ensemble ({NUM_CNN_MODELS} Models)...")
    
    # Setup Data Loaders
    train_ds = FingerprintTextureDataset(train_pairs, GraphMinutiae.find_true_core, augment=True)
    val_ds = FingerprintTextureDataset(val_pairs, GraphMinutiae.find_true_core, augment=False)
    test_ds = FingerprintTextureDataset(test_pairs, GraphMinutiae.find_true_core, augment=False)
    
    class_counts = np.bincount(y_train)
    weights = 1. / class_counts
    sample_weights = [weights[int(t)] for t in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model_paths = []
    # Train Loop for Ensemble
    for i in range(NUM_CNN_MODELS):
        path = train_one_cnn_model(i, train_loader, val_loader)
        model_paths.append(path)
        
    # [4] ENSEMBLE INFERENCE
    print("\n[4] Running Ensemble Inference...")
    
    def get_ensemble_probs(loader, desc):
        # Accumulate probabilities from all models
        accumulated_probs = np.zeros(len(loader.dataset))
        
        for idx, path in enumerate(model_paths):
            print(f"   > Inference using Model {idx+1}...")
            model = DeeperCNN().to(DEVICE)
            model.load_state_dict(torch.load(path))
            model.eval()
            
            model_probs = []
            with torch.no_grad():
                for img1, img2, _ in tqdm(loader, desc=f"M{idx+1} {desc}", leave=False):
                    logits = model(img1.to(DEVICE), img2.to(DEVICE))
                    model_probs.extend(torch.sigmoid(logits).cpu().numpy())
            
            accumulated_probs += np.array(model_probs)
            
            # Memory Cleanup
            del model; torch.cuda.empty_cache(); gc.collect()
            
        return accumulated_probs / NUM_CNN_MODELS  # Average them

    ensemble_val_probs = get_ensemble_probs(val_loader, "Val")
    ensemble_test_probs = get_ensemble_probs(test_loader, "Test")
    
    # [5] OPTIMIZE FUSION
    print("\n[5] Optimizing Fusion Weights...")
    best_val_auc = 0
    best_alpha = 0.5
    for alpha in np.linspace(0, 1, 101):
        mix = (alpha * ensemble_val_probs) + ((1 - alpha) * xgb_val_probs)
        fpr, tpr, _ = roc_curve(y_val, mix)
        sc = auc(fpr, tpr)
        if sc > best_val_auc:
            best_val_auc = sc
            best_alpha = alpha
            
    print(f"  > Best Alpha: {best_alpha:.2f} (CNN Ensemble Weight)")
    print(f"  > Best Val AUC: {best_val_auc:.4f}")
    
    # [6] FINAL EVALUATION & MULTI-PLOT
    print("\n[6] PLOTTING & SAVING RESULTS")
    
    # Calculate curves for all 3
    # 1. XGB Only
    fpr_x, tpr_x, _ = roc_curve(y_test, xgb_test_probs)
    auc_x = auc(fpr_x, tpr_x)
    
    # 2. CNN Ensemble Only
    fpr_c, tpr_c, _ = roc_curve(y_test, ensemble_test_probs)
    auc_c = auc(fpr_c, tpr_c)
    
    # 3. Hybrid
    final_probs = (best_alpha * ensemble_test_probs) + ((1 - best_alpha) * xgb_test_probs)
    fpr_h, tpr_h, _ = roc_curve(y_test, final_probs)
    auc_h = auc(fpr_h, tpr_h)
    # Calculate and print EER
    final_eer = calculate_eer(y_test, final_probs)
    
    print("-" * 40)
    print(f"XGBoost Only AUC:     {auc_x:.4f}")
    print(f"CNN Ensemble AUC:     {auc_c:.4f}")
    print(f"FINAL HYBRID AUC:     {auc_h:.4f}")
    print(f"FINAL HYBRID EER:     {final_eer:.4f}")
    print("-" * 40)
    
    try:
        plt.close('all')
        plt.figure(figsize=(10,8))
        
        # Plot 1: XGB
        plt.plot(fpr_x, tpr_x, label=f'XGBoost (AUC={auc_x:.4f})', 
                 color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        # Plot 2: CNN Ensemble
        plt.plot(fpr_c, tpr_c, label=f'CNN Ensemble (AUC={auc_c:.4f})', 
                 color='blue', linestyle='-.', alpha=0.8, linewidth=2)
        
        # Plot 3: Hybrid (Bold)
        plt.plot(fpr_h, tpr_h, label=f'Hybrid Fusion (AUC={auc_h:.4f})', 
                 color='purple', linestyle='-', linewidth=3)
        
        plt.plot([0,1],[0,1],'k--', alpha=0.5)
        plt.title(f"Performance Comparison (Hybrid $\\alpha={best_alpha:.2f}$)")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(OUTPUT_DIR, 'kaggle_final_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[SUCCESS] Graph saved to: {save_path}")
            
    except Exception as e:
        print(f"\n[ERROR] Plotting failed: {e}")

if __name__ == '__main__':
    run_hybrid_system()