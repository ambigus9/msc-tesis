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

#FROZE SEED BY STAGE
SEED = 42

pipeline["seed_value"] = SEED

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

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
cotrain_list,label_list, datos_total = [], [], []

## Preparar dataset
def ssl_global(model_zoo, pipeline):

    datos = {}
    datos["df_base"] = get_dataset(pipeline)
    datos = split_train_test(datos, pipeline)

    # Medir tiempo de ejecucion
    import time
    start = time.time()

    split_kfold = pipeline["split_kfold"]
    num_kfold = pipeline["num_kfold"]

    for kfold in range(num_kfold):

        models_info = {}
        datos = get_Fold(kfold, datos, pipeline)

        datos_by_fold = {
            "kfold": kfold,
            "datos": datos
        }

        datos_total.append(datos_by_fold)
        df_datos = pd.DataFrame(datos_total)
        datos_path = pipeline["save_path_stats"] + 'exp_'+str(pipeline["id"])+'_'+str(kfold)+'_data.pkl'
        df_datos.to_pickle(datos_path)
        
        numero_lotes = len(datos["batch_set"])

        #datos["batch_set"][0]

        for iteracion in range(numero_lotes*1):
            
            kfold_info = f"K-FOLD {kfold}/{num_kfold} - ITERACION {iteracion}/{numero_lotes}"
            print("\n")
            print("#"*len(kfold_info))
            print(kfold_info)
            print("#"*len(kfold_info))
            print("\n")
            
            print("\n")
            print(f"CLASS DISTRIBUTION - BATCH_SET {iteracion}")
            print( datos["batch_set"][iteracion].groupby(pipeline["y_col_name"]).count() )
            print(f"OK - CLASS DISTRIBUTION - BATCH_SET {iteracion}")
            print("\n")

            if iteracion == 0:
                etapa = 'train'
            else:
                etapa = 'train_EL'

            #print(pipeline["save_path_stats"]+str(pipeline["id"])+'_'+str(iteracion)+'.pkl')

            for model in model_zoo:
                
                #print("##########")
                #print("AUG_FACTOR - CURRENT: ", pipeline["stage_config"][iteracion]["aug_factor"])
                #pipeline["aug_factor"] = pipeline["stage_config"][iteracion]["aug_factor"]
                print("AUG_FACTOR: ", pipeline["aug_factor"])
                
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
            
            #if pipeline['save_model']:
            #   mod_top1 = load_model(mod_top1, compile=True)
            #    mod_top2 = load_model(mod_top2, compile=True)
            #    mod_top3 = load_model(mod_top3, compile=True)
                

            # Medir tiempo de ejecucion
            import time
            start = time.time()

            print("EVALUATING CO-TRAINING ...")
            print("\n")
            #print("Co-train: ", evaluate_cotrain(mod_top1,mod_top2,mod_top3,arch_top1,
            #                                        arch_top2,arch_top3,datos,etapa,kfold,
            #                                        iteracion,pipeline,models_info,logs))

            cotrain_acc, cotrain_infer_dfs = evaluate_cotrain(mod_top1,mod_top2,mod_top3,arch_top1,
                                                    arch_top2,arch_top3,datos,etapa,kfold,
                                                    iteracion,pipeline,models_info,logs)

            print("Co-train: ", cotrain_acc)
            df_cotrain_info = {
                    "kfold": kfold,
                    "iteracion" : iteracion,
                    "df_arch1" : cotrain_infer_dfs[0],
                    "df_arch2" : cotrain_infer_dfs[1],
                    "df_arch3" : cotrain_infer_dfs[2]
            }

            cotrain_list.append(df_cotrain_info)
            df_cotrain_list = pd.DataFrame(cotrain_list)
            #print(df_cotrain_list)

            infer_pkl = pipeline["save_path_stats"] + 'exp_'+str(pipeline["id"])+'_'+str(iteracion)+'_cotrain_eval.pkl'
            
            print("SAVING COTRAIN EVAL PICKLE")
            df_cotrain_list.to_pickle(infer_pkl)
            print("OK - SAVING COTRAIN EVAL PICKLE")

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

            datos, EL_iter, LC_iter , label_infer_df= labeling(etapa, mod_top1, mod_top2, mod_top3, 
                                                arch_top1, arch_top2, arch_top3, 
                                                datos, pipeline, iteracion, models_info)

            df_label_info = {
                    "kfold": kfold,
                    "iteracion" : iteracion,
                    "df_arch1" : label_infer_df[0],
                    "df_arch2" : label_infer_df[1],
                    "df_arch3" : label_infer_df[2]
            }

            label_list.append(df_label_info)
            df_label_list = pd.DataFrame(label_list)
            #print(df_label_list)

            label_pkl = pipeline["save_path_stats"] + 'exp_'+str(pipeline["id"])+'_'+str(iteracion)+'_labeling.pkl'
            
            print("SAVING LABEL PICKLE")
            df_label_list.to_pickle(label_pkl)
            print("OK - SAVING LABEL PICKLE")

            print("OK - LABELING")
            print("EL_iter", len(EL_iter))
            print("LC_iter", len(LC_iter))

            df_EL = pd.DataFrame(datos["EL"], columns=[ pipeline["x_col_name"], pipeline["y_col_name"], 'arch_scores' ])
            df_LC = pd.DataFrame(datos["LC"], columns=[ pipeline["x_col_name"], pipeline["y_col_name"], 'arch_scores' ])
            
            df_EL.to_pickle(pipeline["save_path_stats"]+'exp_'+str(pipeline["id"])+'_'+str(iteracion)+'_EL.pickle')
            df_LC.to_pickle(pipeline["save_path_stats"]+'exp_'+str(pipeline["id"])+'_'+str(iteracion)+'_LC.pickle')

            df_label_stats = label_stats(df_EL, df_LC, pipeline)
            #df_label_stats.to_pickle(pipeline["save_path_stats"]+'exp_'+str(pipeline["id"])+'_'+str(iteracion)+'.pickle')

            df_label_stats.to_pickle( pipeline["save_path_stats"] + 'exp_'+str(pipeline["id"])+'_'+str(iteracion)+'_stats.pickle' )
            

            df_train_EL = pd.concat([datos["df_train"],df_EL.iloc[:,:2]])
            datos['df_train_EL'] = df_train_EL

            ssl_th = pipeline["ssl_threshold"]
            
            logs_label.append([kfold,iteracion,arch_top1,arch_top2,arch_top3,len(EL_iter),len(LC_iter),ssl_th])
            save_logs(logs_label,'label',pipeline)

            reset_keras(pipeline)

            #if pipeline["restart_weights"]:
            #    reset_keras()

                #random.seed(SEED)
                #np.random.seed(SEED)
                #tensorflow.random.set_random_seed(SEED)

    end = time.time()
    print(end - start)

pipeline["modality_config"] = {
    "ultra-fast": {
        "train_epochs": 1,
        "batch_epochs": 1
    },
    "ultra": {
        "train_epochs": 3,
        "batch_epochs": 3
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

pipeline['stage_config'] = {
    0: {
        'LR': 1e-5,
        'layer_percent': 1,
        'aug_factor': 5,
        "train_epochs": 30,
    },
    1: {
        'LR': 1e-5,
        'layer_percent': 0.7,
        'aug_factor': 5,
        "train_epochs": 30,
    },
    2: {
        'LR': 1e-5,
        'layer_percent': 0.5,
        'aug_factor': 5,
        "train_epochs": 30,
    },
    3: {
        'LR': 1e-5,
        'layer_percent': 0.3,
        'aug_factor': 5,
        'train_epochs': 30,
    },
    4: {
        'LR': 1e-5,
        'layer_percent': 0.1,
        'aug_factor': 5,
        "train_epochs": 30,
    }
}

server = pipeline["server"]

if server == "colab":
    save_path_results = pipeline["save_path_results"]
    print(f"Path to save results initial: {save_path_results} on {server}")
    pipeline["save_path_results"] = os.path.join( "/content/drive/MyDrive/msc-miguel/satellital/", pipeline["save_path_results"] )

#elif server == "bivl2ab":
#    pipeline["save_path_results"] = os.path.join( "/content/", pipeline["save_path_results"] )

# UPDATE PIPELINE
pipeline["save_path_results"] = os.path.join( pipeline["save_path_results"], f'exp_{pipeline["id"]}' )

print(f"Path to save results final: {save_path_results} on {server}")

pipeline["save_path_fig"] = os.path.join( pipeline["save_path_results"], pipeline["dataset_base"], pipeline["save_path_fig"] )
pipeline["save_path_stats"] = os.path.join( pipeline["save_path_results"], pipeline["dataset_base"], pipeline["save_path_stats"] )
pipeline["save_path_logs"] = os.path.join( pipeline["save_path_results"], pipeline["dataset_base"], pipeline["save_path_logs"] )
pipeline["save_path_models"] = os.path.join(pipeline["save_path_results"], pipeline["dataset_base"], pipeline["save_path_models"])
pipeline["save_path_data"] = os.path.join(pipeline["save_path_results"], pipeline["dataset_base"] , pipeline["save_path_data"] )

# GETTING GPU FROM ARGS
pipeline["gpu"] = args.gpu

#print(pipeline)

# CREATING SCHEMA
plot_accu = os.path.join(pipeline["save_path_fig"], 'accu')
plot_loss = os.path.join(pipeline["save_path_fig"], 'loss')
plot_conf = os.path.join(pipeline["save_path_fig"], 'conf')
stats_conf = os.path.join(pipeline["save_path_stats"], 'conf')
stats_label = os.path.join(pipeline["save_path_stats"], 'label')
logs_path = pipeline["save_path_logs"]
models_path = pipeline["save_path_models"]
data_path = pipeline["save_path_data"]

os.makedirs( stats_conf , exist_ok=True)
os.makedirs( stats_label , exist_ok=True)
os.makedirs( plot_accu , exist_ok=True)
os.makedirs( plot_loss , exist_ok=True)
os.makedirs( plot_conf , exist_ok=True)
os.makedirs( logs_path , exist_ok=True)
os.makedirs( models_path, exist_ok=True)
os.makedirs( data_path, exist_ok=True)

logs.append(["kfold","iteracion","arquitectura","val_loss","val_accu",
"test_loss","test_accu","test_precision","test_recall","test_f1score","support"])
logs_time.append(["kfold","iteracion","arquitectura","training_time"])
logs_label.append(["kfold","iteracion","arquitectura","EL","LC"])
logs_infer_time.append(["kfold","iteracion","arquitectura","infer_time","num_test_samples"])

save_logs(logs,'train', pipeline)
save_logs(logs_time,'time', pipeline)
save_logs(logs_label,'label', pipeline)
save_logs(logs_infer_time,'infer_time', pipeline)

#models = ['InceptionV4','ResNet152','InceptionV3',]
models = ['ResNet152','InceptionV4','InceptionV3']
#models = ['ResNet152']
#models = ['InceptionV3']
#models = ['InceptionV4']
ssl_global(model_zoo=models, pipeline=pipeline)
