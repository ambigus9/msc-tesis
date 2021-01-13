import os
import gc
import csv
import yaml
from tensorflow.keras.backend import clear_session
import matplotlib.pyplot as plt

def read_yaml(yml_path):
    
    with open(yml_path) as f:
        # use safe_load instead load
        dataMap = yaml.safe_load(f)
    return dataMap

def reset_keras():
    # Reset Keras Session
    clear_session()
    print(gc.collect())

def plot_cm_seaborn(y_true, y_pred, labels, kfold, iteracion, architecture, pipeline):

    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm,labels,labels)

    plt.figure(figsize=(10,6))
    sns.heatmap(cm_df, annot=True)

    ID = pipeline['id']
    save_fig_name = f"exp_{ID:02d}_cm_{kfold:02d}_{iteracion:02d}_{architecture}.png"
    save_fig_cm = os.path.join(pipeline["save_fig_path"] , 'conf' , save_fig_name)

    plt.savefig(save_fig_cm)
    plt.clf()

def calculate_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)
    return cm

def accuracy_by_class(cm, labels):
    #Normalize confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.round(cm, 2)
    
    #Getting diagonal
    cm_dg = cm.diagonal()
    return cm_dg

def plot_confusion_matrix(cm, labels, kfold, iteracion, architecture, pipeline):

    from sklearn.metrics import confusion_matrix
    import plotly.figure_factory as ff
    import numpy as np
    
    #cm = confusion_matrix(y_true, y_pred)
    
    # normalize confusion matrix
    if pipeline["cm_normalize"]:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 2)

    z = cm

    # invert z idx values
    z = z[::-1]

    x = labels
    y = labels
    y = x[::-1].copy()

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='blues')

    # add title
    fig.update_layout(title_text=f'Confusion matrix {kfold:02d}_{iteracion:02d}_{architecture}')

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            #textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    #fig.update_layout(margin=dict(t=400, l=400))
    fig.update_layout(margin=dict(t=150))

    # add colorbar
    fig['data'][0]['showscale'] = True

    ID = pipeline['id']
    save_fig_name = f"exp_{ID:02d}_cm_{kfold:02d}_{iteracion:02d}_{architecture}.html"
    save_fig_cm = os.path.join(pipeline["save_fig_path"] , 'conf' , save_fig_name)
    print(save_fig_cm)
    fig.write_html(save_fig_cm)

def save_plots(history, kfold, iteracion, architecture, pipeline):
    #os.makedirs(pipeline["save_fig_path"], exist_ok=True)
    ID = pipeline['id']

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy - {}'.format(architecture))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    save_fig_name = f"exp_{ID:02d}_accu_{kfold:02d}_{iteracion:02d}_{architecture}.png"
    save_fig_accu = os.path.join(pipeline["save_fig_path"] , 'accu' , save_fig_name)
    plt.savefig(save_fig_accu)
    plt.clf()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss - {}'.format(architecture))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    save_fig_name = f"exp_{ID:02d}_loss_{kfold:02d}_{iteracion:02d}_{architecture}.png"
    save_fig_loss = os.path.join(pipeline["save_fig_path"] , 'loss' , save_fig_name)
    plt.savefig(save_fig_loss)
    plt.clf()
    
def save_logs(logs, log_type, pipeline):
    ID = pipeline['id']
    save_path = pipeline['save_path_logs']
    if not os.path.exists(save_path):
        os.makedirs(save_path,exist_ok=True)
    filename = f'{save_path}exp_{ID:02d}_{log_type}.csv'
    file = open(filename, "a")
    writer = csv.writer(file, delimiter = ",")
    for l in [logs[-1]]:
        writer.writerow(l)
    file.close()
