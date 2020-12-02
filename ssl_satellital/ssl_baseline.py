"MSC Thesis"

#import gc
import os
#import csv
import time
import random
import tensorflow
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.model_selection import StratifiedKFold

from utils_data import process_dataset
#from utils_data import get_dataset
from utils_data import dividir_lotes
from utils_preprocess import dividir_balanceado2
from utils_general import save_logs

#from ssl_train import get_model
from ssl_train import training
from ssl_eval import evaluate_cotrain
from ssl_label import labeling
from ssl_stats import label_stats

SEED = 8128
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

random.seed(SEED)
np.random.seed(SEED)
tensorflow.random.set_random_seed(SEED)

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(gpus[0], True)

#df_train, df_val, df_test1, df_test2 = get_data(archivos, csvs)

#from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing import image

## Preparar dataset
def ssl_global(model_zoo=model_zoo, pipeline=pipeline):

    datos = {}
    models_info = {}
    df_train, df_val, df_test = process_dataset(pipeline)

    # Medir tiempo de ejecucion
    #import time
    start = time.time()
    fold = dividir_balanceado2(df_train,4)

    for kfold in range(1):

        if dataset == 'gleasson':
            #import pandas as pd
            df_train_58         = pd.DataFrame([fold[kfold][0],fold[kfold][2]]).T
            df_train_58.columns = [x_col_name,y_col_name]

            df_val           = pd.DataFrame([fold[kfold][1],fold[kfold][3]]).T
            df_val.columns   = [x_col_name,y_col_name]

            fold1            = dividir_balanceado2(df_train_58,4)
            df_train         = pd.DataFrame([fold1[0][1],fold1[0][3]]).T
            df_train.columns = [x_col_name,y_col_name]

            df_train.to_csv('data/train.csv',index=False)
            df_val.to_csv('data/val.csv',index=False)
            df_test.to_csv('data/test.csv',index=False)

            df_U         = pd.DataFrame([fold1[0][0],fold1[0][2]]).T
            df_U.columns = [x_col_name,y_col_name]
            EL,LC        = [],[]
            #datos["EL"] = []
            #datos["LC"] = []

            print("train :",len(df_train))
            print("val   :",len(df_val))
            print("u     :",len(df_U))

            # Segmentación de U en lotes para etiquetar
            batch_set=list(dividir_lotes(df_U, numero_lotes))
            #for i in range(len(batch_set)):
            for i in enumerate(batch_set):
                print(len(batch_set[i].iloc[:,0].values.tolist()))

        datos['df_train'] = df_train
        datos['df_val'] = df_val
        datos['df_test'] = df_test

        for iteracion in range(numero_lotes*1):

            #import random
            random.seed(SEED)
            np.random.seed(SEED)
            tensorflow.random.set_random_seed(SEED)

            print("\n######################")
            print("K-FOLD {} - ITERACION {}".format(kfold,iteracion))
            print("######################\n")

            if iteracion == 0:
                etapa = 'train'
            else:
                etapa = 'train_EL'

            print(pipeline["path_label_stats"]+str(pipeline["id"])+'_'+str(iteracion)+'.pickle')

            for model in model_zoo:

                model_memory , model_performance = training(kfold,etapa,datos,model,train_epochs,batch_epochs,iteracion,models_info,pipeline)

                models_info[model] = {
                    'model_memory': model_memory,
                    'model_performance': model_performance['val_acc']
                }

            #import pandas as pd
            df_temp = pd.DataFrame(models_info).T
            top_models = df_temp.sort_values('model_performance', ascending=False)
            top_models = top_models.reset_index()['index'].values.tolist()[:3]

            mod_top1, arch_top1 = models_info[ top_models[0] ]['model_memory'] , top_models[0]
            mod_top2, arch_top2 = models_info[ top_models[1] ]['model_memory'] , top_models[1]
            mod_top3, arch_top3 = models_info[ top_models[2] ]['model_memory'] , top_models[2]

            if dataset == 'gleasson':
                print("\nCo-train1: \n", evaluate_cotrain(mod_top1,mod_top2,mod_top3,arch_top1,arch_top2,arch_top3,'gleasson-patologo1',datos,etapa,kfold,iteracion,pipeline,models_info))
                print("\nCo-train2: \n", evaluate_cotrain(mod_top1,mod_top2,mod_top3,arch_top1,arch_top2,arch_top3,'gleasson-patologo2',datos,etapa,kfold,iteracion,pipeline,models_info))

            if semi_method == 'supervised':
                break

            if iteracion < numero_lotes:

                df_batchset = batch_set[iteracion]
                df_batchset.columns = [x_col_name,y_col_name]
                df_batchset[y_col_name] = '0'
            else:
                if  iteracion == numero_lotes:
                    df_LC = pd.DataFrame(LC)
                    batch_set_LC=list(dividir_lotes(df_LC, numero_lotes))
                    #for i in range(len(batch_set_LC)):
                    for i in enumerate(batch_set_LC):
                        print(len(batch_set_LC[i].iloc[:,0].values.tolist()))
                    LC = []

                df_batchset = pd.DataFrame([batch_set_LC[int(iteracion-numero_lotes)].iloc[:,0].values.tolist()]).T
                df_batchset.columns = [x_col_name]
                df_batchset[y_col_name] = '0'

            datos['df_batchset'] = df_batchset

            EL, LC, EL_iter, LC_iter = labeling(etapa, mod_top1, mod_top2, mod_top3, arch_top1, arch_top2, arch_top3, EL, LC, datos, pipeline, iteracion, models_info)
            #logs_label.append([kfold,iteracion,arch_top1,arch_top2,arch_top3,len(EL_iter),len(LC_iter)])
            #save_logs(logs_label,'label',pipeline)

            #df_EL = pd.DataFrame(EL, columns=[x_col_name, y_col_name, 'arch_scores'])
            #df_LC = pd.DataFrame(LC, columns=[x_col_name, y_col_name, 'arch_scores'])

            df_EL = pd.DataFrame(EL_iter, columns=[x_col_name, y_col_name, 'arch_scores']) # EXP30
            df_LC = pd.DataFrame(LC_iter, columns=[x_col_name, y_col_name, 'arch_scores']) # EXP30

            df_label_stats = label_stats(df_EL, df_LC)
            print(df_label_stats)
            df_label_stats.to_pickle(pipeline["path_label_stats"]+str(pipeline["id"])+'_'+str(iteracion)+'.pickle')

            #df_train_EL = pd.concat([df_train,df_EL.iloc[:,:2]])
            df_train_EL = df_EL.iloc[:,:2].copy() # EXP30
            #print(df_train)
            print("df_train_EL")
            print(df_train_EL)
            #print(df_EL.iloc[:,:2])
            #print(df_train_EL)
            datos['df_train_EL'] = df_train_EL

            df_EL_stats = df_label_stats["df_EL_stats"]["df"]
            df_LC_stats = df_label_stats["df_LC_stats"]["df"]

            df_U_iter = pd.concat([df_EL_stats,df_LC_stats], ignore_index=True)
            #df_U_iter.describe()["arch_scores_mean"]["25%"]
            #df_U_iter = pd.concat([df_EL,df_LC], ignore_index=True)
            ssl_th = df_U_iter.describe()["arch_scores_mean"]["mean"]
            #EXP 33
            #print("df_U_describe")
            #print(f"MEAN U_{iteracion}: {ssl_th}")
            #print(df_U_iter.describe())
            #ssl_th = df_U_iter.describe()["arch_scores_mean"]["25%"]
            #print(f"MEAN U_{iteracion}: {ssl_th}")
            #print(f" P25 U_{iteracion}: {ssl_th}")

            #print(f"NUEVO UMBRAL PARA SSL: {ssl_th}")
            #pipeline["ssl_threshold"] = ssl_th

            logs_label.append([kfold,iteracion,arch_top1,arch_top2,arch_top3,len(EL_iter),len(LC_iter),ssl_th])
            save_logs(logs_label,'label',pipeline)
            #reset_keras()
            #models_info = []
    end = time.time()
    print(end - start)

pipeline = {}

server = 'bivl2ab'
dataset = 'satellital'
dataset_base = ''
metodo = 'semi-supervisado'

csvs = f'/home/miguel/{dataset}/dataset/tma_info/'
archivos = f'/home/miguel/{dataset}/'
ruta = f'/home/miguel/{dataset}/'

pipeline['save_path_model'] = '/home/miguel/satellital/models/v7/'

x_col_name = 'patch_name'
y_col_name = 'grade_'

dataset = 'satellital'
ruta_base = 'home/miguel/satellital'
dataset_base = 'NWPU-RESISC45'

if dataset == 'satellital':
    pipeline['dataset_base'] = 'NWPU-RESISC45'


    pipeline['weights'] = 'imagenet'
    pipeline['img_height'] = 299
    pipeline['img_width'] = 299
    pipeline["augmenting_factor"] = 1.5
    pipeline["class_num"] = 2
    pipeline["batch_size"] = 16
    pipeline['save_model'] = False
    pipeline['save_path_logs'] = 'logs/'
    pipeline['id'] = 32
    pipeline["path_label_stats"] = "results/data_label_stats_"
    pipeline["ssl_threshold"] = 0.90
    pipeline['stage_config'] = {
        0: {
            'LR': 1e-5,
            'layer_percent': 1
        },
        1: {
            'LR': 1e-5,
            'layer_percent': 0.7
        },
        2: {
            'LR': 1e-5,
            'layer_percent': 0.5
        },
        3: {
            'LR': 1e-6,
            'layer_percent': 0.3
        },
        4: {
            'LR': 1e-6,
            'layer_percent': 0.1
        }
    }

#EL,LC,test_cotraining,predicciones = [],[],[],[]
test_cotraining,predicciones = [],[]
logs,logs_time,logs_label = [], [], []

data_aumentation = True
early_stopping = True
semi_method = 'co-training-multi'
modalidad = 'rapido'
version = pipeline["id"]
porcentaje='10%'
numero_lotes = 5
label_active = False

if modalidad == 'ultra-fast':
    train_epochs = 1
    batch_epochs = 1

if modalidad == 'ultra':
    train_epochs = 5
    batch_epochs = 5

if modalidad == 'rapido':
    train_epochs = 10
    batch_epochs = 10

if modalidad == 'medio':
    train_epochs = 20
    batch_epochs = 20

if modalidad == 'lento':
    train_epochs = 30
    batch_epochs = 30

logs.append(["kfold","iteracion","arquitectura","val_loss","val_accu",
"test_loss","test_accu"])
logs_time.append(["kfold","iteracion","arquitectura","training_time"])
logs_label.append(["kfold","iteracion","arquitectura","EL","LC"])

save_logs(logs,'train',pipeline)
save_logs(logs_time,'time',pipeline)
save_logs(logs_label,'label',pipeline)

model_zoo = ['ResNet50','Xception','DenseNet169','InceptionV4','DenseNet121']
ssl_global(model_zoo , pipeline)
