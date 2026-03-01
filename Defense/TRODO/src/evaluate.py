import torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from .attacks.pgd_idscore import PGD
import numpy as np
import gc
from .id_scores.msp import get_msp

def mean_id_score_diff(model,
                       dataloader,
                       device=None,
                       verbose=False,
                       eps=1/255):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    attack_eps = eps
    attack_steps = 10
    attack_alpha = 2.5 * attack_eps / attack_steps
    
    attack = PGD(model, eps=attack_eps, steps=attack_steps, alpha=attack_alpha)
    before_attack_scores = []
    after_attack_scores = []
    
    for data, targets in dataloader:
        data = data.to(device)
        
        before_attack = get_msp(model, data)
        
        data = attack(data, targets)
        
        after_attack = get_msp(model, data)
        
        before_attack_scores += before_attack.detach().cpu().numpy().tolist()
        after_attack_scores += after_attack.detach().cpu().numpy().tolist()
        
        torch.cuda.empty_cache()
        gc.collect()
        
    before_attack_scores = np.asarray(before_attack_scores)
    after_attack_scores = np.asarray(after_attack_scores)
    
    if verbose:
        print("Before:", np.mean(before_attack_scores))
        print("After:", np.mean(after_attack_scores))
    
    return np.mean(after_attack_scores) - np.mean(before_attack_scores)

def get_models_scores(model_dataset,
                        model_score_function,
                        progress,
                        verbose=False,
                        live=True,
                        strict=False):
    
    labels = []
    scores = []

    tq = range(len(model_dataset))
    if progress is False:
        tq = tqdm(tq)
    
    if live:
        seen_labels = set()
    failed_models = 0
    
    for i in tq:
        try:
            model, label = model_dataset[i]

            score = model_score_function(model)
            if progress:
                print(f'No. {i}, Label: {label}, Score: {score}')
            
            scores.append(score)
            labels.append(label)
            if live:
                if verbose:
                    print("Label:", label, "Score:", score)
                seen_labels.add(label)
                
                if len(seen_labels) > 1:
                    print("Current:", roc_auc_score(labels, scores))
        except Exception as e:
            if strict:
                raise e
            failed_models += 1
            print(f"The following error occured during the evaluation of a model: {str(e)}")
            print("Skipping this model")
    print("No. of failed models:", failed_models)
    return scores, labels

def get_results(scores, labels):
    return roc_auc_score(labels, scores)


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec

def evaluate_modelset(model_dataset,
                      signature_function,
                      signature_function_kwargs={},
                      get_dataloader_func=None,
                      verbose=False,
                      strict=False,
                      progress=False):
    
    def model_score_function(model):
        dataloader = get_dataloader_func()
        return signature_function(model,
                                 dataloader,
                                 **signature_function_kwargs)
    
    scores, labels = get_models_scores(model_dataset,
                                       model_score_function,
                                       progress,
                                       strict=strict,
                                       verbose=verbose)

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    clean_diff, troj_diff = [], []
    for idx in range(len(model_dataset)):
        try:
            label = labels[idx]
            diff  = scores[idx]
            if label == 0:
                clean_diff.append(diff)
            else:
                troj_diff.append(diff)
        except Exception:
            continue
    
    acc_matrix = np.vstack([clean_diff, troj_diff])
    acc_matrix[0] = np.sort(acc_matrix[0])
    acc_matrix[1] = np.sort(acc_matrix[1])

    color1 = (170/255, 220/255, 224/255)
    color2 = (30/255, 70/255, 155/255)
    custom_cmap = LinearSegmentedColormap.from_list('custom_blue_cmap', [color1, color2])

    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['font.weight'] = 'bold'

    fig = plt.figure(figsize=(10, 10))
    main_right = 0.94
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2, 8],
                           hspace=0.20, left=0.10, right=main_right)

    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(acc_matrix, cmap=custom_cmap, aspect='auto')
    ax0.set_yticks([0, 1])
    ax0.set_yticklabels(['Clean', 'Trojan'], fontweight='bold')
    ax0.set_xticks(np.arange(acc_matrix.shape[1]))
    ax0.set_xticklabels([f'M{i}' for i in range(acc_matrix.shape[1])],
                        fontweight='bold')
    ax0.set_ylabel('Model Type', fontweight='bold', fontsize=11)

    for i in range(2):
        for j in range(acc_matrix.shape[1]):
            val = acc_matrix[i, j]
            ax0.text(j, i, f'{val:.3f}', ha='center', va='center',
                    color='w', fontsize=9, fontweight='bold')

    ax1 = fig.add_subplot(gs[1])
    blue_deep = '#376795'
    ax1.plot(fpr, tpr, color=blue_deep, lw=2.3,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('False Positive Rate', fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontweight='bold')
    ax1.grid(True, color='#ccd9e6', lw=0.6, alpha=0.8)
    for spine in ax1.spines.values():
        spine.set_color('black'); spine.set_linewidth(2)
    ax1.legend(loc="lower right")

    gap_w   = 0.03
    cbar_w  = 0.020 
    roc_right_edge = 0.94
    cbar_left = roc_right_edge - cbar_w
    heatmap_bottom = ax0.get_position().y0
    heatmap_height = ax0.get_position().height
    cax = fig.add_axes([cbar_left, heatmap_bottom, cbar_w, heatmap_height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('ΔMSP', fontweight='bold')
    cbar.ax.tick_params(labelsize=10, width=2)
    cbar.outline.set_linewidth(2)

    heatmap_left   = 0.10
    heatmap_width  = cbar_left - gap_w - heatmap_left
    ax0.set_position([heatmap_left, heatmap_bottom,
                     heatmap_width, heatmap_height])

    roc_left  = heatmap_left
    roc_width = roc_right_edge - roc_left
    roc_bottom = ax1.get_position().y0
    roc_height = ax1.get_position().height
    ax1.set_position([roc_left, roc_bottom, roc_width, roc_height])
    ax0.axhline(y=0.5, color='black', linewidth=1.5)

    plt.savefig('roc_heatmap_combo.png', dpi=450, bbox_inches='tight')
    return get_results(scores, labels)