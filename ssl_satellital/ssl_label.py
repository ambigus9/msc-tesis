"SSL Labeling"

from ml_generators import generadores
from ssl_eval import evaluar

def labeling(etapa,modelo1,modelo2,modelo3,arquitectura1,arquitectura2,arquitectura3,EL,LC,datos,pipeline,iteracion,models_info):
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

    train_generator_arch1,batchset_generator_arch1,STEP_SIZE_BATCH_arch1=generadores(etapa,arquitectura1,datos,pipeline,True,None,iteracion,models_info)
    train_generator_arch2,batchset_generator_arch2,STEP_SIZE_BATCH_arch2=generadores(etapa,arquitectura2,datos,pipeline,True,None,iteracion,models_info)
    train_generator_arch3,batchset_generator_arch3,STEP_SIZE_BATCH_arch3=generadores(etapa,arquitectura3,datos,pipeline,True,None,iteracion,models_info)

    df1=evaluar(modelo1,train_generator_arch1,batchset_generator_arch1,STEP_SIZE_BATCH_arch1)
    df2=evaluar(modelo2,train_generator_arch2,batchset_generator_arch2,STEP_SIZE_BATCH_arch2)
    df3=evaluar(modelo3,train_generator_arch3,batchset_generator_arch3,STEP_SIZE_BATCH_arch3)

    for i in range(len(df1)):

        arch_scores = {}
        arch_scores[arquitectura1] = df1['Max_Probability'][i]
        arch_scores[arquitectura2] = df2['Max_Probability'][i]
        arch_scores[arquitectura3] = df3['Max_Probability'][i]

        c1 = (df1['Predictions'][i] == df2['Predictions'][i]) and  (df1['Max_Probability'][i] > pipeline["ssl_threshold"]) and (df2['Max_Probability'][i] > pipeline["ssl_threshold"])
        c2 = (df1['Predictions'][i] == df3['Predictions'][i]) and  (df1['Max_Probability'][i] > pipeline["ssl_threshold"]) and (df3['Max_Probability'][i] > pipeline["ssl_threshold"])
        c3 = (df2['Predictions'][i] == df3['Predictions'][i]) and  (df2['Max_Probability'][i] > pipeline["ssl_threshold"]) and (df3['Max_Probability'][i] > pipeline["ssl_threshold"])

        if c1 and c2 and c3:
            EL.append([df1['Filename'][i], df1['Predictions'][i], arch_scores])
            EL_iter.append([df1['Filename'][i], df1['Predictions'][i], arch_scores])
            etiquetados_EL += 1
        else:
            LC.append([df1['Filename'][i], df1['Predictions'][i], arch_scores])
            LC_iter.append([df1['Filename'][i], df1['Predictions'][i], arch_scores])
            etiquetados_LC += 1

    print('etiquetados EL {} LC {}'.format(etiquetados_EL, etiquetados_LC))

    return EL, LC, EL_iter, LC_iter
