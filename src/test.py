
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch 
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os

# Metrics
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer


#Delong ROC
from src.delong import compute_midrank, delong_roc_variance
from scipy.stats import norm

#calibration curve
#from sklearn.calibration import calibration_curve

# store the results in a csv file
import pandas as pd
import time


def calculate_auc_ci_delong(y_true, y_prob, alpha=0.95):
    auc_value, auc_variance = delong_roc_variance(y_true, y_prob)  # Get AUC and variance
    auc_std = np.sqrt(auc_variance)  # Directly take sqrt since it's a scalar
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)  # Quantiles
    ci = norm.ppf(lower_upper_q, loc=auc_value, scale=auc_std)  # Compute confidence interval
    ci = np.clip(ci, 0, 1)  # Ensure bounds are within [0,1]
    return auc_value, ci




# Generate Saliency Maps
def get_saliency_map(model, input_image):
    model.eval()
    input_image.requires_grad_()
    output = model(input_image)
    max_idx = output.argmax()
    output[0, max_idx].backward()
    saliency_map, _ = torch.max(input_image.grad.data.abs(),dim=1)
    #saliency_map = input_image.grad.data.abs().max(1)[0]
    return saliency_map

def test_model(y_test, y_pred, y_prob=None):
    """
    Evaluates the model on the training and test data respectively
    1. Predictions on test data
    2. Classification report
    3. Confusion matrix
    4. ROC curve

    Inputs:
    y_test: numpy array with test labels
    y_pred: numpy array with predicted test labels
    """

    # Generate a timestamp for uniqueness
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f'y_{timestamp}.csv'  # Unique filename

    print(y_test, y_prob)

    # Create DataFrame based on shape of y_prob
    if len(y_prob.shape) > 1:
        results_df = pd.DataFrame({f'y_test_{i}': y_test[:, i] for i in range(y_test.shape[1])})
        results_df = pd.concat([results_df, pd.DataFrame({f'y_prob_{i}': y_prob[:, i] for i in range(y_prob.shape[1])})], axis=1)
    else:
        results_df = pd.DataFrame({'y_test': y_test[:, 1], 'y_pred': y_prob})

    # Save results with unique filename
    results_df.to_csv(filename, index=False)
    print(f'Saved results to {filename}')


    # Initialize AUC values to prevent UnboundLocalError
    # auc_score, lower_ci, upper_ci = 0.5, 0.0, 1.0  
    
    # Check if the output is a single class or multi-class
    plot_matrix = False
    if y_pred.shape[1] < 102:
        plot_matrix = True
        
    # Check if the output is a single class or multi-class
    if y_pred.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

    
    
    # Confusion matrix
    # Create a confusion matrix of the test predictions
    if plot_matrix:
        cm = confusion_matrix(y_test, y_pred)
        # create heatmap
        # Set the size of the plot
        fig, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=ax)
        # Set plot labels
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        # Display plot
        plt.show()

    #create ROC curve
    fig, ax = plt.subplots(figsize=(15, 15))

    # binarize the labels
    label_binarizer = LabelBinarizer().fit(y_test) 
    y_onehot_test = label_binarizer.transform(y_test)
    y_onehot_pred = label_binarizer.transform(y_pred)
    
    # Check if the output is a single class
    if (y_onehot_pred.shape[1] < 2):
        fpr, tpr, _ = roc_curve(y_test,  y_pred)

        # Calculate AUC and confidence interval if probabilities are provided
        if y_prob is not None:
            auc_score, ci = calculate_auc_ci_delong(y_test, y_prob)
            lower_ci, upper_ci = ci
        else:
            auc_score, lower_ci, upper_ci = 0.5, 0.0, 1.0  # Default values if no probabilities

        #create ROC curve
        #plt.plot(fpr,tpr)
        if y_prob is not None: # If probabilities are provided, plot the ROC curve using probabilities
            RocCurveDisplay.from_predictions(
                    y_test,
                    y_prob,
                    name=f"ROC curve",
                    color='aqua',
                    ax=ax,
                )
        else:
            RocCurveDisplay.from_predictions(
                    y_test,
                    y_pred,
                    name=f"ROC curve",
                    color='aqua',
                    ax=ax,
                )
        plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
        plt.title(f'ROC Curve (AUC = {auc_score:.2f}, 95% CI = [{lower_ci:.2f}, {upper_ci:.2f}])')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        
    else: # If the output is multi-class, plot the ROC curve for each class
        from itertools import cycle
        colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "green", "yellow", "purple", "pink", "brown", "black"])
        if y_prob is None:
            for class_id, color in zip(range(len(label_binarizer.classes_)), colors):
                RocCurveDisplay.from_predictions(
                    y_onehot_test[:, class_id],
                    y_onehot_pred[:, class_id],
                    name=f"ROC curve for {label_binarizer.classes_[class_id]}",
                    color=color,
                    ax=ax,
                )
        else: # If probabilities are provided, plot the ROC curve for each class using probabilities
            for class_id, color in zip(range(len(label_binarizer.classes_)), colors):
                RocCurveDisplay.from_predictions(
                    y_onehot_test[:, class_id],
                    y_prob[:, class_id],
                    name=f"ROC curve for {label_binarizer.classes_[class_id]}",
                    color=color,
                    ax=ax,
                )
                # Calculate AUC and confidence intervals for each class using DeLong's method
                auc_score, ci = calculate_auc_ci_delong(y_onehot_test[:, class_id], y_prob[:, class_id])
                lower_ci, upper_ci = ci
                print(f'Class {label_binarizer.classes_[class_id]}: AUC = {auc_score:.2f}, 95% CI = [{lower_ci:.2f}, {upper_ci:.2f}]')


        plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
        plt.show()
    
    # Create a classification report of the test predictions
    cr = classification_report(y_test, y_pred)
    # print classification report
    print(cr)
    
    # Calculate the Matthews correlation coefficient
    mcc = matthews_corrcoef(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Use weighted average for multi-class precision
    recall = recall_score(y_test, y_pred, average='weighted')  # Use weighted average for multi-class recall
    f1 = f1_score(y_test, y_pred, average='weighted')  # Use weighted average for multi-class F1-score

    return accuracy, precision, recall, f1, mcc


def test(model, test_dataloader, saliency=True, device='cpu', save=False):

    model.to(device)
    model.eval()

    output_size = test_dataloader.dataset.labels.shape[1]
    num_classes = 2 if test_dataloader.dataset.labels.shape[1] == 1 else test_dataloader.dataset.labels.shape[1]

    eval_images_per_class = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        y_true, y_pred = [], []
        for batch in tqdm(test_dataloader, total=len(test_dataloader), disable=False):
            image, labels =  batch['image'].to(device), batch['labels'].to(device)

            outputs = model(image)

            if (output_size == 1):
                preds = torch.sigmoid(outputs)
            else:
                preds = torch.softmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            # Get 5 images per class for saliency maps
            for i in range(num_classes):
                if len(eval_images_per_class[i]) < 5:
                    if (output_size == 1):
                        eval_images_per_class[i] += [img for i, img in enumerate(image) if labels[i] == i]
                    else:
                        eval_images_per_class[i] += [img for i, img in enumerate(image) if np.argmax(labels[i].cpu().numpy()) == i]
                    
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        if (output_size == 1):
            y_pred_one_hot = (y_pred > 0.5).astype(int)
        else:
            predicted_class_indices = np.argmax(y_pred, axis=1)
            # Convert the predicted class indices to one-hot encoding
            y_pred_one_hot = np.eye(y_pred.shape[1])[predicted_class_indices]
            
        # If the output size is 2, then we just need the probabilities of the positive class
        if (output_size == 2):
            y_pred = y_pred[:, 1]
        
        test_model(y_true, y_pred_one_hot, y_pred)
    
    if saliency:
        if save:
            os.makedirs('saliency_maps', exist_ok=True)
        
        print('#' * 50, f' Saliency Maps ', '#' * 50)
        print('')

        # Select some evaluation images to generate saliency maps
        eval_images = []
        for img_class in eval_images_per_class.keys():
            eval_images = eval_images_per_class[img_class][:5]

            print(f'Class {img_class}:')
            i = 0
            for eval_image in eval_images:
                eval_image = eval_image.unsqueeze(0)  # Add batch dimension
                saliency_map = get_saliency_map(model, eval_image)

                # Plot original image and saliency map side by side
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(eval_image[0].permute(1, 2, 0).detach().cpu().numpy())
                plt.title(f'Original Image (Class {img_class})')
                
                plt.subplot(1, 2, 2)
                plt.imshow(saliency_map[0].detach().cpu().numpy(), cmap=plt.cm.hot)
                plt.title('Saliency Map')
                
                plt.tight_layout()
                if save:
                    plt.savefig(f'saliency_maps/saliency_map_class_{img_class}_image_{i}.pdf')
                    i+=1
                    
                plt.show()
