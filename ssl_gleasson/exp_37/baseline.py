"MSC Thesis"

import os
import random

from shutil import copyfile
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
from ssl_label import labeling, labeling_v2
from ssl_stats import label_stats

# TO DO -> USE UNIVERSAL DICTS
logs,logs_time,logs_label,logs_infer_time = [], [], [], []
cotrain_list,label_list, datos_total = [], [], []

## Preparar dataset
def ssl_global(model_zoo, pipeline):

    #datos = {}
    datos = get_dataset(pipeline)

    #print(datos)
    #return True

    #datos = split_train_test(datos, pipeline)

    #return True
    # Medir tiempo de ejecucion
    import time
    start = time.time()

    #split_kfold = pipeline["split_kfold"]
    #num_kfold = pipeline["num_kfold"]
    method = pipeline["method"]
    
    #for kfold in range(num_kfold):

    models_info = {}

    if method == "semi-supervised":
        datos = get_Fold(kfold, datos, pipeline)

    #return True

    #datos_by_fold = {
    #    "kfold": kfold,
    #    "datos": datos
    #}

    #datos_total.append(datos_by_fold)
    #df_datos = pd.DataFrame(datos_total)
    #datos_path = pipeline["save_path_stats"] + 'exp_'+str(pipeline["id"])+'_'+str(kfold)+'_data.pkl'
    #df_datos.to_pickle(datos_path)

    if method == "supervised":
        kfold = 0
        total_stages = 1#pipeline["train_epochs"]
    elif pipeline["labeling_method"] == 'decision' and method == "semi-supervised":
        total_stages = len(datos["batch_set"])
    elif pipeline["labeling_method"] == 'democratic' and method == "semi-supervised":
        total_stages = pipeline["labeling_stages"]
    else:
        pass

    for iteracion in range(total_stages*1):
        
        #kfold_info = f"K-FOLD {kfold}/{num_kfold} - ITERACION {iteracion}/{total_stages}"
        #print("\n")
        #print("#"*len(kfold_info))
        #print(kfold_info)
        #print("#"*len(kfold_info))
        #print("\n")

        info = f"METHOD - {method} - ITERATION {iteracion}/{total_stages}"
                
        if method == "semi-supervised":
            print("\n")
            print(f"CLASS DISTRIBUTION - BATCH_SET {iteracion}")

            if len(datos["LC"]) > 0:
                U_set = pd.DataFrame(datos["LC"], columns=[ pipeline["x_col_name"], pipeline["y_col_name"], 'arch_scores' ])
                #print("LABELING LOW CONFIDENCE SAMPLES (LC)")
                print( U_set.groupby(pipeline["y_col_name"]).count() )
                #print("OK - LABELING LOW CONFIDENCE SAMPLES (LC)")
            else:
                U_set = datos['U']
                #print("LABELING UNLABELED SAMPLES (U)")
                print( U_set.groupby(pipeline["y_col_name"]).count() )
                #print("OK - LABELING UNLABELED SAMPLES (U)")

            #print( datos["batch_set"][iteracion].groupby(pipeline["y_col_name"]).count() )
            print(f"OK - CLASS DISTRIBUTION - BATCH_SET {iteracion}")
            print("\n")

        if iteracion == 0 or method == "supervised":
            etapa = 'train'
        else:
            etapa = 'train_EL'

        for model in model_zoo:
            
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
        
        # Medir tiempo de ejecucion
        import time
        start = time.time()

        print("EVALUATING CO-TRAINING ...")
        print("\n")

        cotrain_acc1, cotrain_infer_dfs1 = evaluate_cotrain(mod_top1,mod_top2,mod_top3,arch_top1,
                                                arch_top2,arch_top3,datos,etapa,kfold,
                                                iteracion,pipeline,models_info,'patologo1',logs)

        print("Co-train - Patologo 1: ", cotrain_acc1)

        cotrain_acc2, cotrain_infer_dfs2 = evaluate_cotrain(mod_top1,mod_top2,mod_top3,arch_top1,
                                                arch_top2,arch_top3,datos,etapa,kfold,
                                                iteracion,pipeline,models_info,'patologo2',logs)

        print("Co-train - Patologo 2: ", cotrain_acc2)

        df_cotrain_info = {
                "kfold": kfold,
                "iteracion" : iteracion,
                "patologo1": {
                    "df_arch1" : cotrain_infer_dfs1[0],
                    "df_arch2" : cotrain_infer_dfs1[1],
                    "df_arch3" : cotrain_infer_dfs1[2]
                },
                "patologo2": {
                    "df_arch1" : cotrain_infer_dfs2[0],
                    "df_arch2" : cotrain_infer_dfs2[1],
                    "df_arch3" : cotrain_infer_dfs2[2]
                },
        }

        cotrain_list.append(df_cotrain_info)
        df_cotrain_list = pd.DataFrame(cotrain_list)

        infer_pkl = pipeline["save_path_stats"] + 'exp_'+str(pipeline["id"])+'_'+str(iteracion)+'_cotrain_eval.pkl'
        
        print("SAVING COTRAIN EVAL PICKLE")
        df_cotrain_list.to_pickle(infer_pkl)
        print("OK - SAVING COTRAIN EVAL PICKLE")

        print("\n")
        print("OK - EVALUATING CO-TRAINING")

        end = time.time()
        infer_time = end - start

        # SAVE INFER_TIME BY DF_TEST BY ITERATION AND ARCH
        print(infer_time, len(datos["df_test1"]))

        logs_infer_time = []
        logs_infer_time.append([kfold, iteracion, 'co-train1', infer_time, len(datos["df_test1"])])
        save_logs(logs_infer_time, 'infer_time', pipeline)

        if method == "supervised":
            print(f"SUPERVISED METHOD COMPLETED FOR ITERATION: {iteracion}")
            #reset_keras(pipeline)
            continue

        print(f"GETTING BATCH_SET OF ITERATION {iteracion}...")
        print("LABELING ...")

        if pipeline["labeling_method"] == "decision":
            datos, EL_iter, LC_iter, EL_accu, LC_accu, label_infer_df = labeling(etapa, mod_top1, mod_top2, mod_top3, 
                                                arch_top1, arch_top2, arch_top3, 
                                                datos, pipeline, iteracion, models_info)
        elif pipeline["labeling_method"] == "democratic": 
            datos, EL_iter, LC_iter, EL_accu, LC_accu, label_infer_df = labeling_v2(etapa, mod_top1, mod_top2, mod_top3, 
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
        df_label_stats.to_pickle( pipeline["save_path_stats"] + 'exp_'+str(pipeline["id"])+'_'+str(iteracion)+'_stats.pickle' )
        

        df_train_EL = pd.concat([datos["df_train"],df_EL.iloc[:,:2]])
        datos['df_train_EL'] = df_train_EL

        ssl_th = pipeline["ssl_threshold"]
        
        logs_label.append([kfold, iteracion, arch_top1, arch_top2, arch_top3,
                            len(EL_iter), len(LC_iter), EL_accu, LC_accu, ssl_th])
        save_logs(logs_label,'label',pipeline)

        reset_keras(pipeline)

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

# UPDATE PIPELINE
pipeline["save_path_results"] = os.path.join( pipeline["save_path_results"], f'exp_{pipeline["id"]}' )

pipeline["save_path_fig"] = os.path.join( pipeline["save_path_results"], pipeline["dataset_base"], pipeline["save_path_fig"] )
pipeline["save_path_stats"] = os.path.join( pipeline["save_path_results"], pipeline["dataset_base"], pipeline["save_path_stats"] )
pipeline["save_path_logs"] = os.path.join( pipeline["save_path_results"], pipeline["dataset_base"], pipeline["save_path_logs"] )
pipeline["save_path_models"] = os.path.join(pipeline["save_path_results"], pipeline["dataset_base"], pipeline["save_path_models"])
pipeline["save_path_data"] = os.path.join(pipeline["save_path_results"], pipeline["dataset_base"] , pipeline["save_path_data"] )

# GETTING GPU FROM ARGS
pipeline["gpu"] = args.gpu

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

# BACKUP CONFIG YML
yml_config = str(args.yml)
yml_config_path = os.path.join( pipeline["save_path_results"] , yml_config.split('/')[-1] )
copyfile(yml_config, yml_config_path)

logs.append(["kfold","iteracion","arquitectura","val_loss","val_accu",
"test1_loss","test1_accu","test1_precision","test1_recall","test1_f1score",
"test2_loss","test2_accu","test2_precision","test2_recall","test2_f1score"])
logs_time.append(["kfold","iteracion","arquitectura","training_time"])
logs_label.append(["kfold", "iteracion", "arquitectura", "EL", "LC", "EL_accu", "LC_accu", "ssl_th"])
logs_infer_time.append(["kfold","iteracion","arquitectura","infer_time","num_test_samples"])

save_logs(logs,'train', pipeline)
save_logs(logs_time,'time', pipeline)
save_logs(logs_label,'label', pipeline)
save_logs(logs_infer_time,'infer_time', pipeline)

#models = ['ResNet152','InceptionV4','InceptionV3']
models = ['ResNet50','Xception','DenseNet169','InceptionV4','DenseNet121']
ssl_global(model_zoo=models, pipeline=pipeline)
