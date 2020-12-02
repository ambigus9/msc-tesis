"Utils data"

import os
import pandas as pd

#from sklearn.model_selection import StratifiedKFold

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

    if dataset_base == 'ucmerced':
        L = '/{}/{}/Images'.format(pipeline["ruta_base"],dataset_base)

    if dataset_base == 'whu_rs19' or dataset_base == 'AID':
        L = '/{}/{}/'.format(pipeline["ruta_base"],dataset_base)

    if dataset_base == 'NWPU-RESISC45':
        L = '/{}/{}/'.format(pipeline["ruta_base"],dataset_base)

    imagenes, clases = [],[]

    for root, _, files in os.walk(os.path.abspath(L)) :
        for file in files:
            if file.endswith(".tif") or file.endswith(".jpg"):
                clases.append(str(os.path.basename(os.path.normpath(root))))
                imagenes.append(os.path.join(root, file))

    df=pd.DataFrame([imagenes,clases]).T
    df.columns = pipeline["x_col_name"] , pipeline["y_col_name"]
    return df

def process_dataset(pipeline):
    df = get_dataset(pipeline)
    # SPLIT ON SET #ACA VOY
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()
    print(df)
    return df_train, df_val, df_test

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
