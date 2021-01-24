"MSC Thesis"

import os
import random

from utils_general import read_yaml
from utils_general import reset_keras
from utils_general import save_logs

import argparse
parser     = argparse.ArgumentParser(description='')
parser.add_argument('--yml', type=str, default=None, help='path to YAML config')
parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use')
args = parser.parse_args()

print(args)

#loading global configuration
#pipeline = read_yaml('ssl_baseline_nwpu.yml')

pipeline = read_yaml(args.yml)

SEED = 8128
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow
import numpy as np
import pandas as pd

random.seed(SEED)
np.random.seed(SEED)
tensorflow.random.set_random_seed(SEED)

if int(args.gpu) >= 0:
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    tensorflow.config.experimental.set_memory_growth(gpus[0], True)

from utils_data import get_dataset
from utils_data import dividir_lotes
from utils_data import split_train_test
from utils_data import get_Fold

from ssl_train import training
from ssl_eval import evaluate_cotrain
from ssl_eval import classification_metrics
from ssl_label import labeling
from ssl_stats import label_stats

# TO DO -> USE UNIVERSAL DICTS
logs,logs_time,logs_label,logs_infer_time = [], [], [], []

## Preparar dataset
def ssl_global(model_zoo, pipeline):

    datos = {}
    datos["df_base"] = get_dataset(pipeline)
    datos = split_train_test(datos, pipeline)

    # Medir tiempo de ejecucion
    import time
    start = time.time()

    split_kfold = pipeline["split_kfold"]

    for kfold in range(split_kfold):

        models_info = {}
        datos = get_Fold(kfold, datos, pipeline)
        
        numero_lotes = len(datos["batch_set"])

        for iteracion in range(numero_lotes*1):
            
            kfold_info = f"K-FOLD {kfold}/{split_kfold} - ITERACION {iteracion}/{numero_lotes}"
            print("\n")
            print("#"*len(kfold_info))
            print(kfold_info)
            print("#"*len(kfold_info))
            print("\n")
            
            if iteracion == 0:
                etapa = 'train'
            else:
                etapa = 'train_EL'

            print(pipeline["save_path_stats"]+str(pipeline["id"])+'_'+str(iteracion)+'.pickle')

            for model in model_zoo:

                model_memory , model_performance = training(kfold,etapa,datos,model,iteracion,models_info,classification_metrics,pipeline)

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
            
            # Medir tiempo de ejecucion
            import time
            start = time.time()

            print("EVALUATING CO-TRAINING ...")
            print("\n")
            print("Co-train: ", evaluate_cotrain(mod_top1,mod_top2,mod_top3,arch_top1,
                                                    arch_top2,arch_top3,datos,etapa,kfold,
                                                    iteracion,pipeline,models_info,logs))
            print("\n")
            print("OK - EVALUATING CO-TRAINING")

            end = time.time()
            infer_time = end - start
            # SAVE INFER_TIME BY DF_TEST BY ITERATION AND ARCH

            print(infer_time, len(datos["df_test"]))

            logs_infer_time = []
            logs_infer_time.append([kfold, iteracion, 'co-train', infer_time, len(datos["df_test"])])
            save_logs(logs_infer_time, 'infer_time', pipeline)

            print(f"GETTING BATCH_SET OF ITERATION {iteracion}...")

            df_batchset = datos["batch_set"][iteracion]
            df_batchset.columns = [pipeline["x_col_name"],pipeline["y_col_name"]]
            df_batchset[pipeline["y_col_name"]] = '0'

            datos['df_batchset'] = df_batchset

            print("LABELING ...")

            datos, EL_iter, LC_iter = labeling(etapa, mod_top1, mod_top2, mod_top3, 
                                                arch_top1, arch_top2, arch_top3, 
                                                datos, pipeline, iteracion, models_info)
            print("OK - LABELING")
            print("EL_iter", len(EL_iter))
            print("LC_iter", len(LC_iter))

            df_EL = pd.DataFrame(datos["EL"], columns=[ pipeline["x_col_name"], pipeline["y_col_name"], 'arch_scores' ])
            df_LC = pd.DataFrame(datos["LC"], columns=[ pipeline["x_col_name"], pipeline["y_col_name"], 'arch_scores' ])
            
            df_EL.to_pickle(pipeline["save_path_stats"]+'exp_'+str(pipeline["id"])+'_'+str(iteracion)+'_EL.pickle')
            df_LC.to_pickle(pipeline["save_path_stats"]+'exp_'+str(pipeline["id"])+'_'+str(iteracion)+'_LC.pickle')

            df_label_stats = label_stats(df_EL, df_LC, pipeline)
            df_label_stats.to_pickle(pipeline["save_path_stats"]+str(pipeline["id"])+'_'+str(iteracion)+'.pickle')

            df_train_EL = pd.concat([datos["df_train"],df_EL.iloc[:,:2]])
            datos['df_train_EL'] = df_train_EL

            ssl_th = pipeline["ssl_threshold"]
            
            logs_label.append([kfold,iteracion,arch_top1,arch_top2,arch_top3,len(EL_iter),len(LC_iter),ssl_th])
            save_logs(logs_label,'label',pipeline)
            reset_keras()

            #re-assigning SEED
            random.seed(SEED)
            np.random.seed(SEED)
            tensorflow.random.set_random_seed(SEED)

    end = time.time()
    print(end - start)

print(pipeline)

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

# UPDATE PIPELINE
pipeline["save_path_fig"] = os.path.join( pipeline["save_path_results"], pipeline["dataset_base"], pipeline["save_path_fig"] )
pipeline["save_path_stats"] = os.path.join( pipeline["save_path_results"], pipeline["dataset_base"], pipeline["save_path_stats"] )
pipeline["save_path_logs"] = os.path.join( pipeline["save_path_results"], pipeline["dataset_base"], pipeline["save_path_logs"] )

# CREATING SCHEMA
plot_accu = os.path.join(pipeline["save_path_fig"], 'accu')
plot_loss = os.path.join(pipeline["save_path_fig"], 'loss')
plot_conf = os.path.join(pipeline["save_path_fig"], 'conf')
stats_conf = os.path.join(pipeline["save_path_stats"], 'conf')
stats_label = os.path.join(pipeline["save_path_stats"], 'label')
logs_path = os.path.join(pipeline["save_path_logs"])

os.makedirs( stats_conf , exist_ok=True)
os.makedirs( stats_label , exist_ok=True)
os.makedirs( plot_accu , exist_ok=True)
os.makedirs( plot_loss , exist_ok=True)
os.makedirs( plot_conf , exist_ok=True)
os.makedirs( logs_path , exist_ok=True)

logs.append(["kfold","iteracion","arquitectura","val_loss","val_accu",
"test_loss","test_accu","test_precision","test_recall","test_f1score","support"])
logs_time.append(["kfold","iteracion","arquitectura","training_time"])
logs_label.append(["kfold","iteracion","arquitectura","EL","LC"])
logs_infer_time.append(["kfold","iteracion","arquitectura","infer_time","num_test_samples"])

save_logs(logs,'train', pipeline)
save_logs(logs_time,'time', pipeline)
save_logs(logs_label,'label', pipeline)
save_logs(logs_infer_time,'infer_time', pipeline)

models = ['ResNet152','InceptionV3','InceptionV4']
ssl_global(model_zoo=models, pipeline=pipeline)
