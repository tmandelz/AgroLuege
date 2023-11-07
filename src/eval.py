import sys
sys.path.append("src")
sys.path.append("src/models")
import torch.optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_cm
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

level_hierarchy  =pd.read_csv("./Level_hierarchy.csv").drop(columns=["LNF_code","Unnamed: 0","GT"]).drop_duplicates()

def test(model, model_gt, dataloader, level=3):
    model.eval()

    logprobabilities = list()
    targets_list = list()
    gt_instance_list = list()
    logprobabilities_refined = list()
    for iteration, data in tqdm(enumerate(dataloader)):
        n +=1
        if level==1:
            inputs, _, targets, _, gt_instance = data
        elif level ==2:
            inputs, _, _, targets, gt_instance = data
        else:
            inputs, targets, _, _, gt_instance = data

        del data

        if torch.cuda.is_available():
            inputs = inputs.cuda()

        
        y = targets.numpy()
        y_i = gt_instance.cpu().detach().numpy()

        z3, z1, z2 = model.forward(inputs)
        z3_refined = model_gt([z1.detach(), z2.detach(), z3.detach()])

        if type(z3_refined) == tuple:
            z3_refined = z3_refined[0]
            
        z1 = z1.cpu().detach().numpy()
        z2 = z2.cpu().detach().numpy()
        z3 = z3.cpu().detach().numpy()
        z3_refined = z3_refined.cpu().detach().numpy()
        
        targets_list.append(y)
        gt_instance_list.append(y_i)

        if level==1:
            logprobabilities.append(z1)
        elif level ==2:
            logprobabilities.append(z2)
        else:
            logprobabilities.append(z3)

        logprobabilities_refined.append(z3_refined)
    return np.vstack(logprobabilities), np.concatenate(targets_list), np.vstack(gt_instance_list), np.vstack(logprobabilities_refined)

def plot_fields(targets,predictions,level_hierarchy=level_hierarchy,n_samples=8):
    random_fields = np.random.choice(list(range(0,targets.shape[0])),size=n_samples)
    data_list = np.vstack((targets[random_fields],predictions[random_fields]))
    # Create a colormap that spans the range of unique numbers
    all_unique_numbers = np.unique(data_list)
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_unique_numbers)))
    color_map = {num: colors[i] for i, num in enumerate(sorted(all_unique_numbers))}

    fig, axes = plt.subplots(2, n_samples, figsize=(20, 4))
    axes_flat = axes.flatten()

    for i, data in enumerate(data_list):
        # Create the heatmap for the current data array without a color bar
        used_colors = list(map(color_map.get, np.unique(data)) )
        sns.heatmap(data, cmap=used_colors, cbar=False, ax=axes_flat[i])
        if i < n_samples:
            title_text = "Target"
            number_field = i
        else:
            title_text = "Prediction"
            number_field = i-n_samples
        axes_flat[i].set_title(f'{title_text} field {number_field+1}',fontsize = 8)
        axes_flat[i].set_xticks([])
        axes_flat[i].set_yticks([])
        axes_flat[i].set_xticklabels([])
        axes_flat[i].set_yticklabels([])



    number_name_dict = level_hierarchy.set_index("level3").loc[:,"level3-name"].to_dict()
    legend_handles = [mpatches.Patch(color=color_map[value], label=number_name_dict.get(value, 'Unknown')) for value in all_unique_numbers]

    # Add the legend to the last axis or figure
    fig.legend(handles=legend_handles, title='Crop Classes', loc='center right', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.9, 1])


def confusion_matrix_to_accuraccies(confusion_matrix):

    confusion_matrix = confusion_matrix.astype(float)
    # sum(0) <- predicted sum(1) ground truth

    total = np.sum(confusion_matrix)
    n_classes, _ = confusion_matrix.shape
    overall_accuracy = np.sum(np.diag(confusion_matrix)) / total

    # calculate Cohen Kappa (https://en.wikipedia.org/wiki/Cohen%27s_kappa)
    N = total
    p0 = np.sum(np.diag(confusion_matrix)) / N
    pc = np.sum(np.sum(confusion_matrix, axis=0) * np.sum(confusion_matrix, axis=1)) / N ** 2
    kappa = (p0 - pc) / (1 - pc)

    recall = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + 1e-12)
    precision = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=0) + 1e-12)
    f1 = (2 * precision * recall) / ((precision + recall) + 1e-12)
    # Per class accuracy
    cl_acc = np.diag(confusion_matrix) / (confusion_matrix.sum(1) + 1e-12)
    
    return overall_accuracy, kappa, precision.mean(), recall.mean(), f1.mean(), cl_acc


def create_confusion_matrix(targets,predictions,label_int,label_name,level=3):
    if level == 3:
        figure_size = 30
    elif level == 2:
        figure_size = 15
    elif level == 1:
        figure_size = 6
    else:
        raise ValueError

    fig = plt.figure(figsize=(figure_size,figure_size),dpi=200)
    cm = sklearn_cm(targets, predictions, labels=label_int)
    df_cm = pd.DataFrame(cm, index = label_name,
                    columns = label_name)
    sns.heatmap(df_cm,annot=True,cbar=False)
    plt.xticks(rotation = 45)

    # drop missing target labels for calculations
    labels = np.unique(targets)
    labels = labels.tolist()
    cm_for_calculation = sklearn_cm(targets, predictions, labels=labels)
    return cm_for_calculation

def evaluate_fieldwise(model, model_gt, dataset,epoch,n_epochs, batchsize=1, workers=8, viz=False, fold_num=5, level=3,
                        ignore_undefined_classes=False,level_hierarchy = level_hierarchy):
    model.eval()
    model_gt.eval()

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batchsize, num_workers=workers,shuffle=True)
    logprobabilites, targets, gt_instance, logprobabilites_refined = test(model, model_gt, dataloader, level)
    predictions = logprobabilites.argmax(1)
    predictions_refined = logprobabilites_refined.argmax(1)
    predictions = predictions.flatten()
    targets = targets.flatten()
    gt_instance = gt_instance.flatten()
    predictions_refined = predictions_refined.flatten()

    # Ignore unknown class class_id=0
    if viz:
        valid_crop_samples = targets != 9999999999
    elif level == 2 and ignore_undefined_classes:
        valid_crop_samples = (targets != 0) * (targets != 7) * (targets != 9) * (targets != 12)
    elif level == 2:

        targets[(targets == 7)] = 12
        targets[(targets == 9)] = 12
        predictions[(predictions == 7)] = 12
        predictions[(predictions == 9)] = 12
        valid_crop_samples = (targets != 0) * (targets != 7) * (targets != 9)
    else:
        valid_crop_samples = targets != 0

    targets_wo_unknown = targets[valid_crop_samples]
    predictions_wo_unknown = predictions[valid_crop_samples]
    gt_instance_wo_unknown = gt_instance[valid_crop_samples]
    predictions_refined_wo_unknown = predictions_refined[valid_crop_samples]

    labels = np.unique(targets_wo_unknown)
    
    if level == 3:
        unique_labels = level_hierarchy["level3"].values
        label_names = level_hierarchy["level3-name"].values
        confusion_matrix = create_confusion_matrix(targets_wo_unknown,predictions_refined_wo_unknown,unique_labels,label_names,level = 3)
        if n_epochs-1 == epoch:
            wandb.log({f"confusion matrix_bevor_field_majority_level_{level}":wandb.Image("./wandb/bug_wandb_lv3.png")})
            plt.close()
            plot_fields(targets.reshape(-1, 24, 24),predictions_refined.reshape(-1, 24, 24))
            wandb.log({"Example Fields bevor field majority": wandb.Image(plt)})
        plt.close()
    else:
        level_2_1= level_hierarchy.loc[:,[f"level{level}",f"level{level}-name"]].sort_values(by=f"level{level}").drop_duplicates()
        unique_labels = level_2_1[f"level{level}"].values
        label_names = level_2_1[f"level{level}-name"].values
        confusion_matrix = create_confusion_matrix(targets_wo_unknown,predictions_refined_wo_unknown,unique_labels,label_names,level= level)
        wandb.log({f"confusion matrix_bevor_field_majority_level_{level}":wandb.Image(plt)})
        plt.close()
    overall_accuracy, kappa, precision, recall, f1, cl_acc = confusion_matrix_to_accuraccies(confusion_matrix)
    log_wandb = dict(zip(label_names + f"_level_bevor_field_majority_{level}",cl_acc))
    log_wandb |= {"epoch":epoch,
               f"overall_accuracy_bevor_field_majority_level_{level}":overall_accuracy,
               f"kappa_bevor_field_majority_level_{level}":kappa,
               f"precision_bevor_field_majority_level_{level}":precision,
               f"recall_bevor_field_majority_level_{level}":recall,
               f"f1_bevor_field_majority_level_{level}":f1,
               f"perclassacc_bevor_field_majority_level_{level}":cl_acc}
    wandb.log(log_wandb)
    
    prediction_wo_fieldwise = np.zeros_like(targets_wo_unknown)
    prediction_wo_fieldwise_refined = np.zeros_like(targets_wo_unknown)
    num_field = np.unique(gt_instance_wo_unknown).shape[0]
    target_field = np.ones(num_field) * 8888
    prediction_field = np.ones(num_field) * 9999

    count = 0
    for i in np.unique(gt_instance_wo_unknown).tolist():
        field_indexes = gt_instance_wo_unknown == i

        pred = predictions_wo_unknown[field_indexes]
        pred = np.bincount(pred)
        pred = np.argmax(pred)
        prediction_wo_fieldwise[field_indexes] = pred
        prediction_field[count] = pred

        pred = predictions_refined_wo_unknown[field_indexes]
        pred = np.bincount(pred)
        pred = np.argmax(pred)
        prediction_wo_fieldwise_refined[field_indexes] = pred

        target = targets_wo_unknown[field_indexes]
        target = np.bincount(target)
        target = np.argmax(target)
        target_field[count] = target
        count += 1
    if level == 3:
        confusion_matrix = create_confusion_matrix(targets_wo_unknown,prediction_wo_fieldwise_refined,unique_labels,label_names,level=3)
        if n_epochs-1 == epoch:
            wandb.log({f"confusion matrix_level_{level}":wandb.Image(plt),"epoch":epoch})
        plt.close()
            
    else:
        confusion_matrix = create_confusion_matrix(targets_wo_unknown,prediction_wo_fieldwise,unique_labels,label_names,level=level)
        wandb.log({f"confusion matrix_level_{level}":wandb.Image(plt),"epoch":epoch})
        plt.close()
    overall_accuracy, kappa, precision, recall, f1, cl_acc = confusion_matrix_to_accuraccies(confusion_matrix)
    log_wandb = dict(zip(label_names + f"_level_{level}",cl_acc))
    pix_accuracy = np.sum( prediction_wo_fieldwise_refined==targets_wo_unknown ) / prediction_wo_fieldwise_refined.shape[0]
    log_wandb |={"epoch":epoch,
               f"overall_accuracy_level_{level}":overall_accuracy,
               f"kappa_level_{level}":kappa,
               f"precision_level_{level}":precision,
               f"recall_level_{level}":recall,
               f"f1_level_{level}":f1,
               f"perclassacc_level_{level}":cl_acc,
               f"Pixel Accuracy_level_{level}":pix_accuracy}
    wandb.log(log_wandb)

    # Save for the visulization
    if viz:
        prediction_wo_fieldwise = prediction_wo_fieldwise.reshape(-1, 24, 24)
        targets = targets.reshape(-1, 24, 24)

        if level == 3:
            np.savez('./result/msSTAR_ch_analysis4_level_' + str(
                level) + '_fold_' + str(fold_num), targets=targets,
                     predictions_refined=prediction_wo_fieldwise_refined, cm=confusion_matrix,
                     predictions=predictions_refined_wo_unknown)
        else:
            np.savez('./result/msSTAR_ch_analysis4_level_' + str(
                level) + '_fold_' + str(fold_num), targets=targets, predictions=prediction_wo_fieldwise,
                     cm=confusion_matrix)

    return pix_accuracy