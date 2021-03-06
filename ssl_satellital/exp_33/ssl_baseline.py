"MSC Thesis"

import os
import random
import traceback
import tensorflow
import pandas as pd
import numpy as np

from utils_data import get_dataset
from utils_data import dividir_lotes
from utils_data import split_train_test
from utils_data import get_Fold

from utils_general import save_logs
from utils_general import read_yaml

from ssl_train import training
from ssl_eval import evaluate_cotrain
from ssl_label import labeling
from ssl_stats import label_stats

#loading global configuration
pipeline = read_yaml('ssl_baseline.yml')

SEED = 8128
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(pipeline["gpu"])
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

random.seed(SEED)
np.random.seed(SEED)
tensorflow.random.set_random_seed(SEED)

if pipeline["gpu"] >= 0:
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    tensorflow.config.experimental.set_memory_growth(gpus[0], True)

# TO DO -> USE UNIVERSAL DICTS
logs,logs_time,logs_label = [], [], []

## Preparar dataset
def ssl_global(model_zoo, pipeline):

    numero_lotes = 5
    semi_method = 'co-training-multi'

    datos = {}
    models_info = {}
    #test_cotraining,predicciones = [],[]
    #logs,logs_time,logs_label = [], [], []

    datos["df_base"] = get_dataset(pipeline)
    datos = split_train_test(datos, pipeline)

    # Medir tiempo de ejecucion
    import time
    start = time.time()

    for kfold in range(1):
        for iteracion in range(numero_lotes*1):

            print("\n######################")
            print("K-FOLD {} - ITERACION {}".format(kfold,iteracion))
            print("######################\n")

            datos = get_Fold(kfold, datos, pipeline)
            
            if iteracion == 0:
                etapa = 'train'
            else:
                etapa = 'train_EL'

            print(pipeline["path_label_stats"]+str(pipeline["id"])+'_'+str(iteracion)+'.pickle')

            for model in model_zoo:

                model_memory , model_performance = training(kfold,etapa,datos,model,iteracion,models_info,pipeline)

                models_info[model] = {
                    'model_memory': model_memory,
                    'model_performance': model_performance['val_acc']
                }

            df_temp = pd.DataFrame(models_info).T
            top_models = df_temp.sort_values('model_performance', ascending=False)
            top_models = top_models.reset_index()['index'].values.tolist()[:3]

            mod_top1, arch_top1 = models_info[ top_models[0] ]['model_memory'] , top_models[0]
            mod_top2, arch_top2 = models_info[ top_models[1] ]['model_memory'] , top_models[1]
            mod_top3, arch_top3 = models_info[ top_models[2] ]['model_memory'] , top_models[2]
            
            print("\n")
            print("Co-train: ", evaluate_cotrain(mod_top1,mod_top2,mod_top3,arch_top1,
                                                    arch_top2,arch_top3,datos,etapa,kfold,
                                                    iteracion,pipeline,models_info,logs))
            print("\n")

            if semi_method == 'supervised':
                break

            if iteracion < numero_lotes:

                df_batchset = datos["batch_set"][iteracion]
                df_batchset.columns = [pipeline["x_col_name"],pipeline["y_col_name"]]
                df_batchset[pipeline["y_col_name"]] = '0'
            else:
                if  iteracion == numero_lotes:
                    df_LC = pd.DataFrame(pipeline["LC"])
                    batch_set_LC=list(dividir_lotes(df_LC, numero_lotes))

                    for i in enumerate(batch_set_LC):
                        print(len(batch_set_LC[i].iloc[:,0].values.tolist()))
                    pipeline["LC"] = []

                df_batchset = pd.DataFrame([batch_set_LC[int(iteracion-numero_lotes)].iloc[:,0].values.tolist()]).T
                df_batchset.columns = [pipeline["x_col_name"]]
                df_batchset[pipeline["y_col_name"]] = '0'

            datos['df_batchset'] = df_batchset

            print("LABELING ...")
            datos, EL_iter, LC_iter = labeling(etapa, mod_top1, mod_top2, mod_top3, 
                                                arch_top1, arch_top2, arch_top3, 
                                                datos, pipeline, iteracion, models_info)
            print("OK - LABELING")
            #logs_label.append([kfold,iteracion,arch_top1,arch_top2,arch_top3,len(EL_iter),len(LC_iter)])
            #save_logs(logs_label,'label',pipeline)

            #df_EL = pd.DataFrame(EL, columns=[pipeline["x_col_name"], pipeline["y_col_name"], 'arch_scores'])
            #df_LC = pd.DataFrame(LC, columns=[pipeline["x_col_name"], pipeline["y_col_name"], 'arch_scores'])
            print("EL_iter", len(EL_iter))
            print("LC_iter", len(LC_iter))
            #df_EL = pd.DataFrame(EL_iter, columns=[ pipeline["x_col_name"], pipeline["y_col_name"], 'arch_scores' ]) # EXP30
            #df_LC = pd.DataFrame(LC_iter, columns=[ pipeline["x_col_name"], pipeline["y_col_name"], 'arch_scores' ]) # EXP30

            df_EL = pd.DataFrame(datos["EL"], columns=[ pipeline["x_col_name"], pipeline["y_col_name"], 'arch_scores' ])
            df_LC = pd.DataFrame(datos["LC"], columns=[ pipeline["x_col_name"], pipeline["y_col_name"], 'arch_scores' ])

            os.makedirs(pipeline["path_label_stats"].split('/')[0], exist_ok=True)
            
            df_EL.to_pickle(pipeline["path_label_stats"]+str(pipeline["id"])+'_'+str(iteracion)+'_EL.pickle')
            df_LC.to_pickle(pipeline["path_label_stats"]+str(pipeline["id"])+'_'+str(iteracion)+'_LC.pickle')

            df_label_stats = label_stats(df_EL, df_LC, pipeline)
            #print(df_label_stats)
            df_label_stats.to_pickle(pipeline["path_label_stats"]+str(pipeline["id"])+'_'+str(iteracion)+'.pickle')

            df_train_EL = pd.concat([datos["df_train"],df_EL.iloc[:,:2]]) # USANDO MUESTRAS TRAIN Y EL
            #df_train_EL = df_EL.iloc[:,:2].copy() # EXP30 # UNICAMENTE USANDO MUESTRAS EL
            #print(df_train)
            #print("df_train_EL")
            #print(df_train_EL)
            #print(df_EL.iloc[:,:2])
            #print(df_train_EL)
            datos['df_train_EL'] = df_train_EL

            try:
                print("AUTO-ESTIMATING OF SSL THRESHOLD ...")
                df_EL_stats = df_label_stats["df_EL_stats"]["df"]
                df_LC_stats = df_label_stats["df_LC_stats"]["df"]

                df_U_iter = pd.concat([df_EL_stats,df_LC_stats], ignore_index=True)

                ssl_th = df_U_iter.describe()["arch_scores_mean"]["mean"]
                pipeline["ssl_threshold"] = ssl_th
                print("NEW SSL THRESHOLD: ", ssl_th)
            except:
                print("ERROR - AUTO-ESTIMATING SSL THRESHOLD")
                ssl_th = pipeline["ssl_threshold"]
                traceback.print_exc()
            
            #df_U_iter.describe()["arch_scores_mean"]["25%"]
            #df_U_iter = pd.concat([df_EL,df_LC], ignore_index=True)
                
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

print(pipeline)

pipeline['save_path_model'] = '/home/miguel/satellital/models/v7/'

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

pipeline["modality_config"] = {
    "ultra-fast": {
        "train_epochs": 1,
        "batch_epochs": 1
    },
    "ultra": {
        "train_epochs": 5,
        "batch_epochs": 5
    },
    "fast": {
        "train_epochs": 10,
        "batch_epochs": 10
    },
    "medium": {
        "train_epochs": 20,
        "batch_epochs": 20
    },
    "slow": {
        "train_epochs": 30,
        "batch_epochs": 30
    }
}

logs.append(["kfold","iteracion","arquitectura","val_loss","val_accu",
"test_loss","test_accu"])
logs_time.append(["kfold","iteracion","arquitectura","training_time"])
logs_label.append(["kfold","iteracion","arquitectura","EL","LC"])

save_logs(logs,'train',pipeline)
save_logs(logs_time,'time',pipeline)
save_logs(logs_label,'label',pipeline)

models = ['ResNet50','Xception','DenseNet169','InceptionV4','DenseNet121']
ssl_global(model_zoo=models, pipeline=pipeline)
