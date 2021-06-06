"Utils data"

import os
import pandas as pd
#from utils_preprocess import dividir_balanceado2
from sklearn.model_selection import StratifiedKFold

def get_dataset(pipeline):
    """
    Get dataset as DataFrame format
    Args:
        dataset_base (str): Dataset to process
        pipeline (dict): General config

    Returns:
        df (object): DataFrame of imgs paths with label
    """

    dataset_base = pipeline["dataset_base"]

    server = pipeline["server"]
    if server == "bivl2ab":
        base_path = pipeline["path_bivl2ab"]
    elif server == "colab":
        base_path = pipeline["path_colab"]
    else:
        pass

    if dataset_base == 'ucmerced':
        L = '/{}/{}/Images'.format(base_path,dataset_base)

    if dataset_base == 'whu_rs19' or dataset_base == 'AID':
        L = '/{}/{}/'.format(base_path,dataset_base)

    if dataset_base == 'NWPU-RESISC45':
        #L = '/{}/{}/'.format(base_path,dataset_base)
        L = os.path.join(base_path,dataset_base)

    imagenes, clases = [],[]

    for root, _, files in os.walk(os.path.abspath(L)) :
        for file in files:
            if file.endswith(".tif") or file.endswith(".jpg"):
                clases.append(str(os.path.basename(os.path.normpath(root))))
                imagenes.append(os.path.join(root, file))

    df=pd.DataFrame([imagenes,clases]).T
    df.columns = pipeline["x_col_name"] , pipeline["y_col_name"]
    return df
def validar_existencia(df, pipeline):
    imgs = df[pipeline["x_col_name"]].values.tolist()
    num = 0
    #for i in range(len(imgs)):
    for i in enumerate(imgs):
        if not os.path.exists(imgs[i]):
            print(num, imgs[i])
            num+=1
    if num == 0:
        return True
    else:
        return False

def dividir_lotes(lista, divisiones):
    k, m = divmod(len(lista), divisiones)
    return (lista[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(divisiones))

def balancear_downsampling(df, pipeline):
    df.columns = pipeline["x_col_name"], pipeline["y_col_name"]
    g = df.groupby(pipeline["y_col_name"])
    df_temp=g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
    df_temp.columns = pipeline["y_col_name"]
    df_temp = df_temp.reset_index().drop([pipeline["y_col_name"],'level_1'],axis=1)
    df_temp.columns=[pipeline["x_col_name"], pipeline["y_col_name"]]
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

    df_train = pd.DataFrame([fold[kfold][0],fold[kfold][2]]).T
    df_train.columns = [pipeline["x_col_name"],pipeline["y_col_name"]]

    df_val = pd.DataFrame([fold[kfold][1],fold[kfold][3]]).T
    df_val.columns = [pipeline["x_col_name"],pipeline["y_col_name"]]

    df_test = datos["df_test"]

    init_train = pipeline["init_train"]

    if init_train == "10%":
        print(f"Entrenamiento inicial con {init_train} de las muestras")
        # Segmentación del 60% de train en 10% train y 50% Unlabeled
        sub_fold = dividir_balanceado2(df_train,6)
    elif init_train == "5%":
        print(f"Entrenamiento inicial con {init_train} de las muestras")
        # Segmentación del 60% de train en 5% train y 55% Unlabeled
        sub_fold = dividir_balanceado2(df_train,12)
    df_train = pd.DataFrame([sub_fold[0][1],sub_fold[0][3]]).T
    df_train.columns = [pipeline["x_col_name"],pipeline["y_col_name"]]

    df_U = pd.DataFrame([sub_fold[0][0],sub_fold[0][2]]).T
    df_U.columns = [pipeline["x_col_name"],pipeline["y_col_name"]]

    # Shuffle Unlabeled Samples
    df_U = df_U.sample(frac=1).reset_index(drop=True)
    EL,LC = [],[]

    # saving csv data
    save_csv_train = os.path.join(pipeline["save_path_data"], f'exp_{pipeline["id"]}_kfold_{kfold}_train.csv')
    save_csv_val = os.path.join(pipeline["save_path_data"], f'exp_{pipeline["id"]}_kfold_{kfold}_val.csv')
    save_csv_test = os.path.join(pipeline["save_path_data"], f'exp_{pipeline["id"]}_kfold_{kfold}_test.csv')

    df_train.to_csv(save_csv_train,index=False)
    df_val.to_csv(save_csv_val,index=False)
    df_test.to_csv(save_csv_test,index=False)

    print("\n")
    print("  train :", len(df_train))
    print("    val :", len(df_val))
    print("   test :", len(df_test))
    print("      u :", len(df_U))

    # Segmentación de U en lotes para etiquetar
    batch_set = list(dividir_lotes(df_U, pipeline["batch_size_u"]))
    for i in range(len(batch_set)):
        print("batch_u :", len( batch_set[i] ) )

    datos["df_train"] = df_train
    datos["df_val"] = df_val
    datos["batch_set"] = batch_set
    datos["EL"] = EL
    datos["LC"] = LC

    return datos

def split_train_test(datos, pipeline):

    # Segmentacion 80% train y 20% test
    fold_base = dividir_balanceado2(datos["df_base"], pipeline["split_train_test"])

    df_train_base = pd.DataFrame([fold_base[0][0],fold_base[0][2]]).T
    df_train_base.columns = [pipeline["x_col_name"],pipeline["y_col_name"]]

    df_test = pd.DataFrame([fold_base[0][1],fold_base[0][3]]).T
    df_test.columns = [pipeline["x_col_name"],pipeline["y_col_name"]]

    # Segmentacion del 80% de train en K-folds=4
    fold_total = dividir_balanceado2(df_train_base, pipeline["split_kfold"])

    datos["df_test"] = df_test
    datos["kfold_total"] = fold_total

    return datos