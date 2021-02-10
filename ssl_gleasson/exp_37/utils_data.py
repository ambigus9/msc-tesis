"Utils data"

import os
import pandas as pd
#from utils_preprocess import dividir_balanceado2
from sklearn.model_selection import StratifiedKFold

def load_csv_gleasson(patches, pipeline):

    ruta_base = pipeline["ruta_base"]
    csvs = os.path.join(ruta_base, 'tma_info')

    df_concat = pd.DataFrame(columns=[pipeline["x_col_name"], pipeline["y_col_name"]])
    ruta_base = pipeline["ruta_base"]

    for i in range(len(patches)):
        csv_path = os.path.join( csvs , patches[i]+'_pgleason_scores.csv' )
        df = pd.read_csv(csv_path,sep='\t',dtype=str)
        df['patch_name']=df['patch_name'].apply(lambda x:x.replace('/content/gdrive/My Drive/gleason_CNN-master/dataset', ruta_base))    
        df_concat = df_concat.append(df, ignore_index=True)
    
    return df_concat

def get_dataset(pipeline):
    """
    Get dataset as DataFrame format
    Args:
        dataset_base (str): Dataset to process
        pipeline (dict): General config

    Returns:
        raw_data (dict): DataFrames containing train and test samples
    """

    datos = {}
    dataset_base = pipeline["dataset_base"]
    ruta_base = pipeline["ruta_base"]

    csvs = os.path.join(ruta_base, 'tma_info')
    
    df_train = load_csv_gleasson(["ZT111","ZT199","ZT204"], pipeline)
    df_val = load_csv_gleasson(["ZT76"], pipeline)
    
    # unión de patches de train y validación
    df_train = pd.concat([df_train, df_val]).reset_index().drop('index',axis=1)
    if pipeline["balance_downsampling"]:
        df_train = balancear_downsampling(df_train, pipeline).copy()

    # Only grades 1 and 2
    df_train = df_train[(df_train[pipeline["y_col_name"]] != '0') & (df_train[pipeline["y_col_name"]] != '3')]

    # patches de test1 y test2
    csv_path = os.path.join( csvs , "ZT80_pgleason_scores.csv" )
    df_test = pd.read_csv(csv_path,sep='\t',dtype=str)
    df_test['patch_name1'] = df_test['patch_name1'].apply(lambda x:x.replace('/content/gdrive/My Drive/gleason_CNN-master/dataset', ruta_base))    
    df_test['patch_name2'] = df_test['patch_name2'].apply(lambda x:x.replace('/content/gdrive/My Drive/gleason_CNN-master/dataset', ruta_base))

    # Only grades 1 and 2
    df_test = df_test[(df_test['grade_1'] != '0') & (df_test['grade_1'] != '3') & (df_test['grade_2'] != '0') & (df_test['grade_2'] != '3')].copy()
    
    # Si se desea utilizar las muestras en las que los patólogos coinciden
    #df = df[(df['grade_1'] != '0') & (df['grade_1'] != '3') & (df['grade_2'] != '0') & (df['grade_2'] != '3') & (df['grade_1'] == df['grade_2'])].copy()

    df_test1 = df_test[['patch_name1','grade_1']]
    df_test2 = df_test[['patch_name2','grade_2']]

    #df.columns = pipeline["x_col_name"] , pipeline["y_col_name"]

    datos["df_train"] = df_train
    datos["df_test1"] = df_test1
    datos["df_test2"] = df_test2

    print("## TRAIN ##")
    print(df_train.groupby(pipeline["y_col_name"]).count())
    print("TOTAL TRAIN: ",len(df_train))
    print("\n")
    print("## TEST1 ##")
    print(df_test1.groupby(pipeline["y_col_name"]+'1').count())
    print("TOTAL TEST1: ",len(df_test1))
    print("\n")
    print("## TEST2 ##")
    print(df_test2.groupby(pipeline["y_col_name"]+'2').count())
    print("TOTAL TEST2: ",len(df_test2))
    print("\n")

    return datos

def dividir_lotes(lista, divisiones):
    k, m = divmod(len(lista), divisiones)
    return (lista[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(divisiones))

def balancear_downsampling(df, pipeline):
    df.columns = ["filename","classname"]
    g = df.groupby("classname")
    df_temp = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
    df_temp.columns = [pipeline["x_col_name"], pipeline["y_col_name"]]
    df_temp = df_temp.reset_index().drop(["classname","level_1"],axis=1)
    df_temp.columns = [pipeline["x_col_name"], pipeline["y_col_name"]]
    return df_temp

def dividir_balanceado2(df,fragmentos):
    X = df.iloc[:,0].values
    y = df.iloc[:,1].values
    kf = StratifiedKFold(n_splits=fragmentos)
    kf.get_n_splits(X)

    fold = []

    print(kf)

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        fold.append([X_train,X_test,y_train,y_test])
    return fold

def get_Fold(kfold, datos, pipeline):

    fold = datos["kfold_total"]

    df_train_base = pd.DataFrame([fold[kfold][0],fold[kfold][2]]).T
    df_train_base.columns = [pipeline["x_col_name"],pipeline["y_col_name"]]

    df_val = pd.DataFrame([fold[kfold][1],fold[kfold][3]]).T
    df_val.columns = [pipeline["x_col_name"],pipeline["y_col_name"]]

    total_train = len(datos["df_train"])
    total_val = len(df_val)
    total_grand = total_train + len(datos["df_test1"])

    ratio_local_train = round((len(df_train_base)/total_train)*100, 2)
    ratio_local_val = round((total_val/total_train)*100, 2)

    ratio_global_train = round((len(df_train_base)/total_grand)*100, 2)
    ratio_global_val = round((total_val/total_grand)*100, 2)

    print(f"TRAIN BASE {ratio_local_train}% (LOCAL) , {ratio_global_train}% (GLOBAL)")
    print(f"  VAL BASE {ratio_local_val}% (LOCAL) , {ratio_global_val}% (GLOBAL)")

    df_test1 = datos["df_test1"]
    df_test2 = datos["df_test2"]

    total_test1 = len(df_test1)
    total_test2 = len(df_test2)

    ratio_global_test1 = round((total_test1/total_grand)*100, 2)
    ratio_global_test2 = round((len(df_test2)/total_grand)*100, 2)

    #print(f"  TEST1 {ratio_global_test1}% (GLOBAL)")
    #print(f"  TEST2 {ratio_global_test2}% (GLOBAL)")

    # Segmentación del 60% de train en 10% train y 50% Unlabeled
    sub_fold = dividir_balanceado2(df_train_base,6)
    df_train = pd.DataFrame([sub_fold[0][1],sub_fold[0][3]]).T
    df_train.columns = [pipeline["x_col_name"],pipeline["y_col_name"]]

    total_train_init = len(df_train)

    ratio_local_train_init = round((len(df_train)/total_train)*100, 2)
    ratio_global_train_init = round((len(df_train)/total_grand)*100, 2)
    print(f"TRAIN INIT {ratio_local_train_init}% (LOCAL) , {ratio_global_train_init}% (GLOBAL)")

    df_U = pd.DataFrame([sub_fold[0][0],sub_fold[0][2]]).T
    df_U.columns = [pipeline["x_col_name"],pipeline["y_col_name"]]

    # Shuffle Unlabeled Samples
    df_U = df_U.sample(frac=1).reset_index(drop=True)
    
    total_U = len(df_U)
    ratio_global_U = round((total_U/total_grand)*100, 2)

    EL,LC = [],[]

    # saving csv data
    #save_csv_train = os.path.join(pipeline["save_path_data"], f'exp_{pipeline["id"]}_kfold_{kfold}_train.csv')
    #save_csv_val = os.path.join(pipeline["save_path_data"], f'exp_{pipeline["id"]}_kfold_{kfold}_val.csv')
    #save_csv_test = os.path.join(pipeline["save_path_data"], f'exp_{pipeline["id"]}_kfold_{kfold}_test.csv')

    #df_train.to_csv(save_csv_train,index=False)
    #df_val.to_csv(save_csv_val,index=False)
    #df_test.to_csv(save_csv_test,index=False)
    total_samples = total_train_init + total_val
    total_samples = total_samples + total_U
    total_samples = total_samples + total_test1
    #total_samples = total_samples + total_test2

    ratio_global_total = ratio_global_train_init + ratio_global_val
    ratio_global_total = ratio_global_total + ratio_global_U
    ratio_global_total = ratio_global_total + ratio_global_test1
    #ratio_global_total = ratio_global_total + ratio_global_test2

    print("\n")
    print(f"  TRAIN  {total_train_init} {ratio_global_train_init}% (GLOBAL)")
    print(f"    VAL  {total_val} {ratio_global_val}% (GLOBAL)")
    print(f"  TEST1  {total_test1} {ratio_global_test1}%  (GLOBAL)")
    #print(f"  TEST2  {total_test2} {ratio_global_test2}%  (GLOBAL)")
    print(f"      U  {total_U} {ratio_global_U}% (GLOBAL)")
    print(f"--------------------------------------------")
    print(f"  TOTAL {total_samples} {ratio_global_total}% (GLOBAL)")
    print("\n")

    # Segmentación de U en lotes para etiquetar
    batch_set = list(dividir_lotes(df_U, pipeline["batch_size_u"]))
    for i in range(len(batch_set)):
        total_batch_U = len(batch_set[i])
        ratio_global_batch_U = round((total_batch_U/total_grand)*100, 2)
        print(f"BATCH_U {total_batch_U} {ratio_global_batch_U}% (GLOBAL)")

    datos["df_train"] = df_train
    datos["df_val"] = df_val
    datos["batch_set"] = batch_set
    datos["EL"] = EL
    datos["LC"] = LC
    datos["U"] = df_U

    return datos

def split_train_test(datos, pipeline):

    # Segmentacion 80% train base y 20% val
    fold_base = dividir_balanceado2(datos["df_train"], pipeline["split_train_test"])
    datos["kfold_total"] = fold_base

    return datos