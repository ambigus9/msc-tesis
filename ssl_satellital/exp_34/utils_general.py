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

def reset_keras():
    clear_session()
    print(gc.collect())
