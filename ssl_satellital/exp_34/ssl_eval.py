import pandas as pd
import numpy as np
from ml_generators import generadores
from utils_general import save_logs
from utils_general import plot_confusion_matrix
from utils_general import calculate_confusion_matrix
from utils_general import save_confusion_matrix
#from utils_general import accuracy_by_class


from sklearn.metrics import precision_recall_fscore_support

def evaluate_cotrain(modelo1,modelo2,modelo3,
                    arquitectura1,arquitectura2,arquitectura3,
                    datos, etapa, kfold, iteracion,
                    pipeline, models_info, logs):

    train_generator_arch1,test_generator_arch1,STEP_SIZE_TEST_arch1=generadores(etapa,arquitectura1,datos,pipeline,False,iteracion,models_info)
    train_generator_arch2,test_generator_arch2,STEP_SIZE_TEST_arch2=generadores(etapa,arquitectura2,datos,pipeline,False,iteracion,models_info)
    train_generator_arch3,test_generator_arch3,STEP_SIZE_TEST_arch3=generadores(etapa,arquitectura3,datos,pipeline,False,iteracion,models_info)

    df1=evaluar(modelo1,train_generator_arch1,test_generator_arch1,STEP_SIZE_TEST_arch1)
    df2=evaluar(modelo2,train_generator_arch2,test_generator_arch2,STEP_SIZE_TEST_arch2)
    df3=evaluar(modelo3,train_generator_arch3,test_generator_arch3,STEP_SIZE_TEST_arch3)

    predicciones = []
    for i in range(len(df1)):

        c1 = (df1['Predictions'][i] == df2['Predictions'][i])
        c2 = (df1['Predictions'][i] == df3['Predictions'][i])

        if c1 and c2:
            predicciones.append([df1['Filename'][i],df1['Predictions'][i]])
        else:
            probabilidades = np.array([df1['Max_Probability'][i],df2['Max_Probability'][i],df3['Max_Probability'][i]])
            indice_prob_max = probabilidades.argmax()

            clases = np.array([df1['Predictions'][i],df2['Predictions'][i],df3['Predictions'][i]])
            real = np.array([df1['Filename'][i],df2['Filename'][i],df3['Filename'][i]])
            
            predicciones.append([real[indice_prob_max],clases[indice_prob_max]])

    results = pd.DataFrame(predicciones,columns=["filename","predictions"])

    results['filename'] = results['filename'].apply(lambda x:x.split('/')[-2])
    y_true = results['filename'].values.tolist()
    y_pred = results['predictions'].values.tolist()

    labels_arch1 = (train_generator_arch1.class_indices)

    print("LABELS CO-TRAIN")
    print([*labels_arch1])

    architecture = 'co-train'

    class_metrics = precision_recall_fscore_support(y_true, y_pred, average=pipeline["metrics"])
    
    cm = calculate_confusion_matrix(y_true, y_pred)

    # normalize confusion matrix
    if pipeline["cm_normalize"]:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 2)
    
    save_confusion_matrix(cm, pipeline)

    #plot_confusion_matrix(cm, [*labels_arch1], kfold, iteracion, architecture, pipeline)
    #acc_cls = accuracy_by_class(cm, [*labels_arch1])
    #print("ACCURACY BY CLASS")
    #print(acc_cls)
    #print("LEN ACCURACY BY CLASS")
    #print(len(acc_cls))
    # SAVE ACC_CLS
    #logs_accBycls = []
    #logs_accBycls.append([kfold,iteracion,architecture,acc_cls])
    #save_logs(logs_accBycls, 'accBycls', pipeline)

    #plot_confusion_matrix(y_true, y_pred, [*labels_arch1], kfold, iteracion, architecture, pipeline)
    
    from sklearn.metrics import accuracy_score
    co_train_accu = accuracy_score(y_true,y_pred)

    logs.append([kfold,iteracion,architecture,None,None,None,co_train_accu,
    class_metrics[0],class_metrics[1],class_metrics[2],class_metrics[3]])

    print(f"Co-train Accuracy: {co_train_accu}")
    print(f"Co-train Precision: {class_metrics[0]}")
    print(f"Co-train Recall: {class_metrics[1]}")
    print(f"Co-train F1-Score: {class_metrics[2]}")
    print(f"Co-train Support: {class_metrics[3]}")

    save_logs(logs,'train',pipeline)
    return co_train_accu

def evaluar(modelo, train_generator, test_generator, test_steps):

    pred=modelo.predict(test_generator,steps=test_steps,verbose=0)

    predicted_class_indices=np.argmax(pred,axis=1)
    predicted_class_probab=np.max(pred,axis=1)

    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    filenames = test_generator.filenames
    diferencia = len(predictions)-len(filenames)

    if len(predictions)-len(filenames) == 0:
        results=pd.DataFrame({"Filename":filenames,
                            "Predictions":predictions,
                            "Max_Probability":predicted_class_probab})
    else:
        results=pd.DataFrame({"Filename":filenames[:diferencia],
                            "Predictions":predictions,
                          "Max_Probability":predicted_class_probab})
    return results

def classification_metrics(model, train_generator, test_generator, test_steps, 
                            kfold, iteracion, architecture, pipeline):

    df_pred = evaluar(model, train_generator, test_generator, test_steps)

    prediction_names = df_pred['Filename'].apply(lambda x:x.split('/')[-2])
    y_true = prediction_names.values.tolist()
    y_pred = df_pred['Predictions'].values.tolist()

    labels = (train_generator.class_indices)

    print("LABELS")
    print([*labels])

    class_metrics = precision_recall_fscore_support(y_true, y_pred, average=pipeline["metrics"])
    cm = calculate_confusion_matrix(y_true, y_pred)
    #plot_confusion_matrix(cm, [*labels], kfold, iteracion, architecture, pipeline)

    # normalize confusion matrix
    if pipeline["cm_normalize"]:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 2)
    
    save_confusion_matrix(cm, pipeline)
    
    #acc_cls = accuracy_by_class(cm, [*labels])
    #print("ACCURACY BY CLASS")
    #print(acc_cls)
    #print("LEN ACCURACY BY CLASS")
    #print(len(acc_cls))
    # SAVE ACC_CLS
    #logs_accBycls = []
    #logs_accBycls.append([kfold,iteracion,architecture,acc_cls])
    #save_logs(logs_accBycls, 'accBycls', pipeline)

    print(class_metrics)
    return class_metrics
