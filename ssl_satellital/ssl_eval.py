
def evaluate_cotrain(modelo1,modelo2,modelo3,
                    arquitectura1,arquitectura2,arquitectura3,
                    dataset_base, datos, etapa, kfold, iteracion,
                    pipeline,models_info):

    train_generator_arch1,test1_generator_arch1,STEP_SIZE_TEST1_arch1=generadores(etapa,arquitectura1,datos,pipeline,False,dataset_base,iteracion,models_info)
    train_generator_arch2,test1_generator_arch2,STEP_SIZE_TEST1_arch2=generadores(etapa,arquitectura2,datos,pipeline,False,dataset_base,iteracion,models_info)
    train_generator_arch3,test1_generator_arch3,STEP_SIZE_TEST1_arch3=generadores(etapa,arquitectura3,datos,pipeline,False,dataset_base,iteracion,models_info)

    df1=evaluar(modelo1,train_generator_arch1,test1_generator_arch1,STEP_SIZE_TEST1_arch1)
    df2=evaluar(modelo2,train_generator_arch2,test1_generator_arch2,STEP_SIZE_TEST1_arch2)
    df3=evaluar(modelo3,train_generator_arch3,test1_generator_arch3,STEP_SIZE_TEST1_arch3)

    predicciones = []
    for i in range(len(df1)):

        c1 = (df1['Predictions'][i] == df2['Predictions'][i])
        c2 = (df1['Predictions'][i] == df3['Predictions'][i])
        c3 = (df2['Predictions'][i] == df3['Predictions'][i])

        if c1 and c2 and c3:
            predicciones.append([df1['Filename'][i],df1['Predictions'][i]])
        else:
            probabilidades = np.array([df1['Max_Probability'][i],df2['Max_Probability'][i],df3['Max_Probability'][i]])
            indice_prob_max = probabilidades.argmax()

            clases = np.array([df1['Predictions'][i],df2['Predictions'][i],df3['Predictions'][i]])
            indice_clas_max = clases.argmax()

            real = np.array([df1['Filename'][i],df2['Filename'][i],df3['Filename'][i]])
            predicciones.append([real[indice_prob_max],clases[indice_clas_max]])

    results = pd.DataFrame(predicciones,columns=["filename","predictions"])

    results['filename'] = results['filename'].apply(lambda x:x.split('/')[-1].split('_')[-1][0])
    y_true = results['filename'].values.tolist()
    y_pred = results['predictions'].values.tolist()

    from sklearn.metrics import accuracy_score
    co_train_accu = accuracy_score(y_pred,y_true)
    co_train_label = 'co-train'

    logs.append([kfold,iteracion,co_train_label,None,None,None,co_train_accu])
    save_logs(logs,'train',pipeline)
    return co_train_accu

def evaluar(modelo,train_generator,test_generator,STEP_SIZE_TEST):
    pred=modelo.predict(test_generator,steps=STEP_SIZE_TEST,verbose=0)

    predicted_class_indices=np.argmax(pred,axis=1)
    predicted_class_probab=np.max(pred,axis=1)

    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    filenames=test_generator.filenames
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
