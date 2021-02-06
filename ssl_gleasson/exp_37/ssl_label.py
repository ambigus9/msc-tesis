"SSL Labeling"

from ml_generators import generadores
from ssl_eval import evaluar

def labeling(etapa, modelo1, modelo2, modelo3, arquitectura1, arquitectura2, arquitectura3, datos, pipeline, iteracion, models_info):
    """
    Labels samples using models
    Args:
        etapa (str): Word 1 to compare
        modelo1 (bytes): Top1 Model
        modelo2 (bytes): Top2 Model
        modelo3 (bytes): Top3 Model
        arquitectura1 (str): Top1 Arch
        arquitectura2 (str): Top2 Arch
        arquitectura3 (str): Top3 Arch
        EL (list): Enlarge labeled samples
        LC (list): Low confidence samples
        datos (df): Dataframe with data
        pipeline (dict): General config
        iteracion (int): Semi-supervised Stage
        models_info (list): Info about models

    Returns:
        EL (list): Enlarge labeled samples updated
        LC (list): Low confidence samples updated
        EL_iter ():
        LC_iter ():

    """
    etiquetados_EL = 0
    etiquetados_LC = 0
    EL_iter = []
    LC_iter = []

    predicciones = []
    predicciones_logs = []

    train_generator_arch1,batchset_generator_arch1,STEP_SIZE_BATCH_arch1=generadores(etapa, arquitectura1, datos, pipeline, True, iteracion, models_info)
    train_generator_arch2,batchset_generator_arch2,STEP_SIZE_BATCH_arch2=generadores(etapa, arquitectura2, datos, pipeline, True, iteracion, models_info)
    train_generator_arch3,batchset_generator_arch3,STEP_SIZE_BATCH_arch3=generadores(etapa, arquitectura3, datos, pipeline, True, iteracion, models_info)

    df1 = evaluar(modelo1,train_generator_arch1, batchset_generator_arch1, STEP_SIZE_BATCH_arch1)
    df2 = evaluar(modelo2,train_generator_arch2, batchset_generator_arch2, STEP_SIZE_BATCH_arch2)
    df3 = evaluar(modelo3,train_generator_arch3, batchset_generator_arch3, STEP_SIZE_BATCH_arch3)

    for i in range(len(df1)):

        arch_scores = {}
        arch_scores[arquitectura1] = df1['Max_Probability'][i]
        arch_scores[arquitectura2] = df2['Max_Probability'][i]
        arch_scores[arquitectura3] = df3['Max_Probability'][i]

        c1 = (df1['Predictions'][i] == df2['Predictions'][i])
        c2 = (df1['Predictions'][i] == df3['Predictions'][i])
        c3 = (df2['Predictions'][i] == df3['Predictions'][i])

        p1 = (df1['Max_Probability'][i] > pipeline["ssl_threshold"])
        p2 = (df2['Max_Probability'][i] > pipeline["ssl_threshold"])
        p3 = (df3['Max_Probability'][i] > pipeline["ssl_threshold"])

        if c1 and c2 and p1 and p2 and p3:
            datos["EL"].append([df1['Filename'][i], df1['Predictions'][i], arch_scores])
            predicciones.append([df1['Filename'][i],df1['Predictions'][i]])
            selected = df1['Predictions'][i]
            prob_selected = df1["Max_Probability"][i]
            predicciones_logs.append([df1['Filename'][i],selected,prob_selected,"EL",
                                    df1['Predictions'][i],df1['Max_Probability'][i],
                                    df2["Predictions"][i],df2['Max_Probability'][i],
                                    df3["Predictions"][i],df3['Max_Probability'][i]])
            EL_iter.append([df1['Filename'][i], df1['Predictions'][i], arch_scores])
            etiquetados_EL += 1
        else:
            datos["LC"].append([df1['Filename'][i], df1['Predictions'][i], arch_scores])
            predicciones.append([df1['Filename'][i],df1['Predictions'][i]])
            selected = df1['Predictions'][i]
            prob_selected = df1["Max_Probability"][i]
            predicciones_logs.append([df1['Filename'][i],selected,prob_selected,"LC",
                                    df1['Predictions'][i],df1['Max_Probability'][i],
                                    df2["Predictions"][i],df2['Max_Probability'][i],
                                    df3["Predictions"][i],df3['Max_Probability'][i]])
            LC_iter.append([df1['Filename'][i], df1['Predictions'][i], arch_scores])
            etiquetados_LC += 1

    print('etiquetados EL {} LC {}'.format(etiquetados_EL, etiquetados_LC))

    import pandas as pd

    df_logs = pd.DataFrame(predicciones_logs,
                columns=["y_true","y_pred","prob_pred","type","pred1","prob1","pred2","prob2","pred3","prob3"])

    df_logs["y_true"] = df_logs['y_true'].apply(lambda x:x.split('/')[-2])

    print("LABELING LOGS")

    print(df_logs)


    df_EL_logs = df_logs[(df_logs["type"] == "EL")].copy()
    df_LC_logs = df_logs[(df_logs["type"] == "LC")].copy()

    y_true = df_EL_logs['y_true'].values.tolist()
    y_pred = df_EL_logs['y_pred'].values.tolist()

    from sklearn.metrics import accuracy_score
    EL_accu = accuracy_score(y_true,y_pred)
    print("EL_ACCU: ",EL_accu)

    y_true = df_LC_logs['y_true'].values.tolist()
    y_pred = df_LC_logs['y_pred'].values.tolist()

    from sklearn.metrics import accuracy_score
    LC_accu = accuracy_score(y_true,y_pred)
    print("LC_ACCU: ",LC_accu)

    return datos, EL_iter, LC_iter, [df1, df2, df3]

def labeling_v2(etapa, modelo1, modelo2, modelo3, arquitectura1, arquitectura2, arquitectura3, datos, pipeline, iteracion, models_info):

    predicciones = []
    predicciones_logs = []

    etiquetados_EL = 0
    etiquetados_LC = 0
    EL_iter = []
    LC_iter = []

    train_generator_arch1,batchset_generator_arch1,STEP_SIZE_BATCH_arch1=generadores(etapa, arquitectura1, datos, pipeline, True, iteracion, models_info)
    train_generator_arch2,batchset_generator_arch2,STEP_SIZE_BATCH_arch2=generadores(etapa, arquitectura2, datos, pipeline, True, iteracion, models_info)
    train_generator_arch3,batchset_generator_arch3,STEP_SIZE_BATCH_arch3=generadores(etapa, arquitectura3, datos, pipeline, True, iteracion, models_info)

    df1 = evaluar(modelo1,train_generator_arch1, batchset_generator_arch1, STEP_SIZE_BATCH_arch1)
    df2 = evaluar(modelo2,train_generator_arch2, batchset_generator_arch2, STEP_SIZE_BATCH_arch2)
    df3 = evaluar(modelo3,train_generator_arch3, batchset_generator_arch3, STEP_SIZE_BATCH_arch3)

    for i in range(len(df1)):

        arch_scores = {}
        arch_scores[arquitectura1] = df1['Max_Probability'][i]
        arch_scores[arquitectura2] = df2['Max_Probability'][i]
        arch_scores[arquitectura3] = df3['Max_Probability'][i]

        c1 = (df1['Predictions'][i] == df2['Predictions'][i])
        c2 = (df1['Predictions'][i] == df3['Predictions'][i])
        c3 = (df2['Predictions'][i] == df3['Predictions'][i])

        if c1 and c2:
            datos["EL"].append([df1['Filename'][i], df1['Predictions'][i], arch_scores])
            predicciones.append([df1['Filename'][i],df1['Predictions'][i]])
            selected = df1['Predictions'][i]
            prob_selected = df1["Max_Probability"][i]
            predicciones_logs.append([df1['Filename'][i],selected,prob_selected,"EL",
                                    df1['Predictions'][i],df1['Max_Probability'][i],
                                    df2["Predictions"][i],df2['Max_Probability'][i],
                                    df3["Predictions"][i],df3['Max_Probability'][i]])
            EL_iter.append([df1['Filename'][i], df1['Predictions'][i], arch_scores])
            etiquetados_EL += 1
        elif c1 or c2:
            datos["EL"].append([df1['Filename'][i], df1['Predictions'][i], arch_scores])
            predicciones.append([df1['Filename'][i],df1['Predictions'][i]])
            selected = df1['Predictions'][i]
            prob_selected = df1["Max_Probability"][i]
            predicciones_logs.append([df1['Filename'][i],selected,prob_selected,"EL",
                                    df1['Predictions'][i],df1['Max_Probability'][i],
                                    df2["Predictions"][i],df2['Max_Probability'][i],
                                    df3["Predictions"][i],df3['Max_Probability'][i]])
            EL_iter.append([df1['Filename'][i], df1['Predictions'][i], arch_scores])
            etiquetados_EL += 1
        elif c3:
            datos["EL"].append([df1['Filename'][i], df1['Predictions'][i], arch_scores])
            predicciones.append([df2['Filename'][i],df2['Predictions'][i]])
            selected = df2['Predictions'][i]
            prob_selected = df2["Max_Probability"][i]
            predicciones_logs.append([df1['Filename'][i],selected,prob_selected,"EL",
                                    df1['Predictions'][i],df1['Max_Probability'][i],
                                    df2["Predictions"][i],df2['Max_Probability'][i],
                                    df3["Predictions"][i],df3['Max_Probability'][i]])
            EL_iter.append([df1['Filename'][i], df1['Predictions'][i], arch_scores])
            etiquetados_EL += 1
            
        else:
            datos["LC"].append([df1['Filename'][i], df1['Predictions'][i], arch_scores])
            LC_iter.append([df1['Filename'][i], df1['Predictions'][i], arch_scores])
            etiquetados_LC += 1
            
            predicciones.append([df1['Filename'][i],df1['Predictions'][i]])
            selected = df1['Predictions'][i]
            prob_selected = df1["Max_Probability"][i]
            predicciones_logs.append([df1['Filename'][i],selected,prob_selected,"LC",
                                    df1['Predictions'][i],df1['Max_Probability'][i],
                                    df2["Predictions"][i],df2['Max_Probability'][i],
                                    df3["Predictions"][i],df3['Max_Probability'][i]])

    print('etiquetados EL {} LC {}'.format(etiquetados_EL, etiquetados_LC))

    import pandas as pd

    df_logs = pd.DataFrame(predicciones_logs,
                columns=["y_true","y_pred","prob_pred","type","pred1","prob1","pred2","prob2","pred3","prob3"])

    df_logs["y_true"] = df_logs['y_true'].apply(lambda x:x.split('/')[-2])

    print("LABELING LOGS")

    print(df_logs)


    df_EL_logs = df_logs[(df_logs["type"] == "EL")].copy()
    df_LC_logs = df_logs[(df_logs["type"] == "LC")].copy()

    y_true = df_EL_logs['y_true'].values.tolist()
    y_pred = df_EL_logs['y_pred'].values.tolist()

    from sklearn.metrics import accuracy_score
    EL_accu = accuracy_score(y_true,y_pred)
    print("EL_ACCU: ",EL_accu)

    y_true = df_LC_logs['y_true'].values.tolist()
    y_pred = df_LC_logs['y_pred'].values.tolist()

    from sklearn.metrics import accuracy_score
    LC_accu = accuracy_score(y_true,y_pred)
    print("LC_ACCU: ",LC_accu)

    return datos, EL_iter, LC_iter, [df1, df2, df3]
