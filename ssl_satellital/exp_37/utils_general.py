import os
import gc
import csv
import yaml
import numpy as np
from tensorflow.keras.backend import clear_session
from sklearn.metrics import confusion_matrix
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

    #from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm,labels,labels)

    plt.figure(figsize=(10,6))
    sns.heatmap(cm_df, annot=True)

    ID = pipeline['id']
    save_fig_name = f"exp_{ID:02d}_cm_{kfold:02d}_{iteracion:02d}_{architecture}.png"
    save_fig_cm = os.path.join(pipeline["save_path_fig"] , 'conf' , save_fig_name)

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
    import plotly.express as px
    
    ID = pipeline['id']

    for file_format in [".html",".svg"]:

        if file_format == '.html':
            cm_labels = labels
            cm_color_scale = px.colors.sequential.Plasma

        if file_format == '.svg':
            cm_labels = None
            cm_color_scale = px.colors.sequential.Plasma
            
        fig = px.imshow(cm,
                        labels=dict(x="Predicted Class", y="True Class", color="Probability (%)"),
                        x=cm_labels,
                        y=cm_labels,
                        color_continuous_scale=cm_color_scale
                    )

        save_fig_name = f"exp_{ID:02d}_cm_{kfold:02d}_{iteracion:02d}_{architecture}{file_format}"
        save_fig_cm = os.path.join(pipeline["save_path_fig"] , 'conf' , save_fig_name)
        print(save_fig_cm)
        
        if file_format == '.html':
            # add title
            fig.update_layout(title_text=f'Confusion matrix {kfold:02d}_{iteracion:02d}_{architecture}')
            fig.write_html(save_fig_cm)
        
        if file_format == '.svg':
            fig.update_layout(coloraxis_showscale=False)
            fig.write_image(save_fig_cm)

def save_confusion_matrix(cm, labels, kfold, iteracion, architecture, pipeline):
    import numpy as np
    import pandas as pd

    ID = pipeline['id']
    save_cm_data_name = f"exp_{ID:02d}_cm_{kfold:02d}_{iteracion:02d}_{architecture}.pkl"
    save_cm_data_name = os.path.join( pipeline["save_path_stats"] , 'conf' , save_cm_data_name )
    print(save_cm_data_name)

    cm_data = {
        "cm": cm,
        "classnames": labels
    }

    df_cm_data = pd.DataFrame([cm_data])
    df_cm_data.to_pickle(save_cm_data_name)

def save_plots(history, kfold, iteracion, architecture, pipeline):

    ID = pipeline['id']

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy - {}'.format(architecture))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    save_fig_name = f"exp_{ID:02d}_accu_{kfold:02d}_{iteracion:02d}_{architecture}.png"
    save_fig_accu = os.path.join(pipeline["save_path_fig"] , 'accu' , save_fig_name)
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
    save_fig_loss = os.path.join(pipeline["save_path_fig"] , 'loss' , save_fig_name)
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
