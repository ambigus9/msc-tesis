import os
import gc
import csv
from tensorflow.keras.backend import clear_session

def guardar_logs(ruta,lista):
    #guardar lista de listas en csv crear csv
    import csv
    import os
    if metodo == 'semi-supervisado':
        archivo = "{}logs/logs_{}_{}_{}_{}_{}_{}.csv".format(ruta,dataset,dataset_base,porcentaje,version,modalidad,str(pipeline["ssl_threshold"]).replace('0.',''))
    if metodo == 'supervisado':
        archivo = "{}logs/logs_{}_{}_{}_{}.csv".format(ruta,dataset,dataset_base,version,modalidad)

    file = open(archivo, "a")
    writer = csv.writer(file, delimiter = ",")
    for l in lista:
        writer.writerow(l)
    file.close()


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