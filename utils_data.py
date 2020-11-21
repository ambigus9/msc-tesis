
import os
import pandas as pd

def procesar_dataset(dataset_base, dataset, pipeline):
    if dataset == 'satellital':
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
        df.columns = pipeline["x_col_name"] , pipeline["y_col_name"] #["imagen","clase"]
        #x_col_name, y_col_name = "imagen","clase"

    return df

def get_data(archivos, csvs, pipeline):
    df = procesar_dataset(dataset_base,dataset, pipeline)
    df # SPLIT ON SET
    #return df_train, df_val, df_test1, df_test2

def descargar_dataset(dataset_base,dataset):
    if dataset == 'satellital':
        if dataset_base == 'ucmerced':
            pass
            #!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1--zKmpmsX7cfmdtBfz0ixrQp3w7TYQGF' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1--zKmpmsX7cfmdtBfz0ixrQp3w7TYQGF" -O ucmerced.zip && rm -rf /tmp/cookies.txt
            #!unzip -o ucmerced.zip
        if dataset_base == 'whu_rs19':
            pass
            #!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-6GM-Ctf1eSODLrfmS3RHGgJsxglcBsY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-6GM-Ctf1eSODLrfmS3RHGgJsxglcBsY" -O whu_rs19.zip && rm -rf /tmp/cookies.txt
            #!unzip -o whu_rs19.zip
        if dataset_base == 'AID':
            pass
            #!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1v9Jjh6VGzIyR7RAmhwTa3p7i-5e47gOv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1v9Jjh6VGzIyR7RAmhwTa3p7i-5e47gOv" -O aid.zip && rm -rf /tmp/cookies.txt
            #!unzip -o aid.zip
        if dataset_base == 'nwpu_resisc45':
            pass
            #!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-0cYXOMJTwoz23ZGbeUD4DvpGkEdfvZd' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-0cYXOMJTwoz23ZGbeUD4DvpGkEdfvZd" -O nwpu_resisc45.zip && rm -rf /tmp/cookies.txt
            #!unzip -o nwpu_resisc45.zip

def validar_existencia(df):
    parches_ = df[x_col_name].values.tolist()
    num = 0
    for i in range(len(parches_)):
        if not os.path.exists(parches_[i]):
            print(num, parches_[i])
            num+=1

def dividir_lotes(lista, divisiones):
    k, m = divmod(len(lista), divisiones)
    return (lista[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(divisiones))

