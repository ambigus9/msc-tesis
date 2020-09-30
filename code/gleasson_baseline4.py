# pip install pandas scikit-learn Pillow
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

SEED = 8128

os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import pandas as pd
import numpy as np
import csv

from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

import tensorflow
  # new flag present in tf 2.0+
# # Agregar semilla
import random
random.seed(SEED)
np.random.seed(SEED)
tensorflow.random.set_random_seed(SEED)

#USAR SOLO MEMORIA NECESARIA
gpus = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(gpus[0], True)


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers
#from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
#from sklearn.metrics import plot_confusion_matrix

from utils.train import get_model

import gc
from tensorflow.keras.models import Model
from tensorflow.keras.backend import clear_session

def reset_keras():
    # Reset Keras Session
    clear_session()
    print(gc.collect())

def save_logs(logs,log_type,pipeline):
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

def get_data(archivos, csvs):

    parches   = ["ZT111","ZT199","ZT204"]
    df_concat = pd.DataFrame(columns=["patch_name","grade_"])

    # parches de entrenamiento
    for i in range(len(parches)):
        df=pd.read_csv(csvs+parches[i]+'_pgleason_scores.csv',sep='\t',dtype=str)
        df['patch_name']=df['patch_name'].apply(lambda x:x.replace('/content/gdrive/My Drive/gleason_CNN-master/',archivos))    
        df_concat = df_concat.append(df, ignore_index=True)

    df_train=df_concat.copy()

    # parches de validación
    parches=["ZT76"]
    df_concat = pd.DataFrame(columns=["patch_name","grade_"])

    for i in range(len(parches)):
        df=pd.read_csv(csvs+parches[i]+'_pgleason_scores.csv',sep='\t',dtype=str)
        df['patch_name']=df['patch_name'].apply(lambda x:x.replace('/content/gdrive/My Drive/gleason_CNN-master/',archivos))    
        df_concat = df_concat.append(df, ignore_index=True)

    df_val=df_concat.copy()
    df_val=df_val[(df_val['grade_'] != '0') & (df_val['grade_'] != '3')]

    # unión de parches de train y validación
    df_train=pd.concat([df_train,df_val]).reset_index().drop('index',axis=1)

    df_train=df_train[(df_train['grade_'] != '0') & (df_train['grade_'] != '3')]

    # parches de test1 y test2
    parches=["ZT80"]
    df=pd.read_csv(csvs+parches[0]+'_pgleason_scores.csv',sep='\t',dtype=str)
    df['patch_name1']=df['patch_name1'].apply(lambda x:x.replace('/content/gdrive/My Drive/gleason_CNN-master/',archivos))    
    df['patch_name2']=df['patch_name2'].apply(lambda x:x.replace('/content/gdrive/My Drive/gleason_CNN-master/',archivos))

    df = df[(df['grade_1'] != '0') & (df['grade_1'] != '3') & (df['grade_2'] != '0') & (df['grade_2'] != '3')].copy()
    # Si se desea utilizar las muestras en las que los patólogos coinciden
    #df = df[(df['grade_1'] != '0') & (df['grade_1'] != '3') & (df['grade_2'] != '0') & (df['grade_2'] != '3') & (df['grade_1'] == df['grade_2'])].copy()

    df_test1=df[['patch_name1','grade_1']]
    df_test2=df[['patch_name2','grade_2']]

    df_train = balancear_downsampling(df_train)

    print('=== DISTRIBUCION DATASETSET ===')
    print(df_train.groupby('grade_').count())
    print(df_val.groupby('grade_').count())
    print(df_test1.groupby('grade_1').count())
    print(df_test2.groupby('grade_2').count())

    return df_train, df_val, df_test1, df_test2

#df_train, df_val, df_test1, df_test2 = get_data(archivos, csvs)

def balancear_downsampling(df):
    df.columns=["archivo","clase"]
    g = df.groupby('clase')
    df_=g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
    df_.columns = ['archivos','clases']
    df_ = df_.reset_index().drop(['clase','level_1'],axis=1)
    df_.columns=[x_col_name,y_col_name]
    return df_

def dividir_balanceado2(df,fragmentos):
    X = df.iloc[:,0].values
    y = df.iloc[:,1].values
    kf = StratifiedKFold(n_splits=fragmentos)
    
    kf.get_n_splits(X)

    fold = []

    #print(kf)

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        fold.append([X_train,X_test,y_train,y_test])
    return fold

def transfer_learning(base_model, num_clases):

    if dataset == 'gleasson':
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        #x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_clases, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers[:]:
            layer.trainable = True
    
    return model

def entrenamiento(kfold,etapa,datos,arquitectura,LR,train_epochs,batch_epochs,early_stopping,iteracion,batch_size,pipeline):
    
    import time
    start_model = time.time()
    
    # AGREGAR PIPELINE CONFIG DICT AL FLUJO
    base_model, preprocess_input = get_model(arquitectura,pipeline)
    model_performance = {}

    if dataset == 'gleasson':
        datagen = ImageDataGenerator(
                                    preprocessing_function=preprocess_input,
                                    rotation_range=40,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.01,
                                    zoom_range=[0.9, 1.25],
                                    horizontal_flip=True,
                                    vertical_flip=False,
                                    fill_mode='reflect',
                                    data_format='channels_last'
                                )

    if etapa=='train':
        train_generator = datagen.flow_from_dataframe(
                         dataframe=datos['df_train'], 
                         x_col=x_col_name, 
                         y_col=y_col_name, 
                         target_size=(pipeline['img_height'],pipeline['img_width']),
                         class_mode='categorical', 
                         batch_size=batch_size,
                         seed=42,
                         shuffle=True)
        
    if etapa=='train_EL':
        train_generator = datagen.flow_from_dataframe(
                         dataframe=datos['df_train_EL'], 
                         x_col=x_col_name, 
                         y_col=y_col_name, 
                         target_size=(pipeline['img_height'],pipeline['img_width']),
                         class_mode='categorical', 
                         batch_size=batch_size,
                         seed=42,
                         shuffle=True)
    
    if len(datos['df_val']) > 0:
        val_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
    
        valid_generator=val_datagen.flow_from_dataframe(
                        dataframe=datos['df_val'],
                        x_col=x_col_name,
                        y_col=y_col_name,
                        batch_size=batch_size,
                        seed=42,
                        shuffle=True,
                        class_mode="categorical",
                        target_size=(pipeline['img_height'],pipeline['img_width']))
    
    test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

    if dataset=='gleasson':
        test1_generator=test_datagen.flow_from_dataframe(
                      dataframe=datos['df_test1'],
                      x_col="patch_name1",
                      y_col="grade_1",
                      batch_size=batch_size,
                      seed=42,
                      shuffle=False,
                      class_mode="categorical",
                      target_size=(pipeline['img_height'],pipeline['img_width']))

        test2_generator=test_datagen.flow_from_dataframe(
                    dataframe=datos['df_test2'],
                    x_col="patch_name2",
                    y_col="grade_2",
                    batch_size=batch_size,
                    seed=42,
                    shuffle=False,
                    class_mode="categorical",
                    target_size=(pipeline['img_height'],pipeline['img_width']))
    
    if etapa == 'train' or etapa == 'train_EL':
        finetune_model = transfer_learning(base_model,clases)

    #entrenar modelo
    from tensorflow.keras.optimizers import SGD, Adam, RMSprop

    if etapa == 'train':
        NUM_EPOCHS = train_epochs
        num_train_images = len(datos['df_train'])*augmenting_factor
        datos_entrenamiento = datos['df_train'].copy()
    if etapa == 'train_EL':
        NUM_EPOCHS = batch_epochs 
        num_train_images = len(datos['df_train_EL'])*augmenting_factor
        datos_entrenamiento = datos['df_train_EL'].copy()
    
    STEP_SIZE_TRAIN=num_train_images//train_generator.batch_size
    if len(datos['df_val'])>0:
        STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    STEP_SIZE_TEST1=test1_generator.n//test1_generator.batch_size

    if dataset == 'gleasson': 
        STEP_SIZE_TEST2=test2_generator.n//test2_generator.batch_size
    

    if len(datos['df_val'])>0:
        generator_seguimiento = valid_generator
        pasos_seguimiento = STEP_SIZE_VALID
    else:
        generator_seguimiento = test1_generator
        pasos_seguimiento = STEP_SIZE_TEST1
    
    if clases>3:
        metrics = ['accuracy']#, tf.keras.metrics.AUC(name='auc', multi_label=True)]
        loss='categorical_crossentropy'
        peso_clases = {}
        total = datos_entrenamiento.shape[0]
        #classes = datos_entrenamiento.groupby(y_col_name).count().index.values
        weights = (total/datos_entrenamiento.groupby(y_col_name).count().values)/4
        # rx normal:0, covid:1, pneumonia:2
        peso_clases = {0:weights[0][0], 1:weights[1][0], 2:weights[2][0], 3:weights[3][0]}

        #print(datos_entrenamiento.groupby(y_col_name))
        print(datos_entrenamiento.groupby(y_col_name).count())
        print(datos_entrenamiento.groupby(y_col_name).count().values)
        max_class_num = np.argmax(datos_entrenamiento.groupby(y_col_name).count().values)
        print('id_maximo',max_class_num)
        
        peso_clases = {}
        class_num = datos_entrenamiento.groupby(y_col_name).count().values
        print('num_maximo',class_num[max_class_num])
        weights = class_num[max_class_num] / datos_entrenamiento.groupby(y_col_name).count().values
        print(weights)
        peso_clases = {0:weights[0][0], 1:weights[1][0], 2:weights[2][0], 3:weights[3][0]}
        print(peso_clases)

    if clases==3:
        metrics = ['accuracy']#, tf.keras.metrics.AUC(name='auc', multi_label=True)]
        # calcular pesos de cada clase
        total = datos_entrenamiento.shape[0]
        #classes = datos_entrenamiento.groupby(y_col_name).count().index.values
        weights = (total/datos_entrenamiento.groupby(y_col_name).count().values)/3
        # rx normal:0, covid:1, pneumonia:2
        peso_clases = {0:weights[0][0], 1:weights[1][0], 2:weights[2][0]}
    elif clases==2:
        metrics = ['accuracy']#, tf.keras.metrics.AUC(name='auc', multi_label=True)]
        # calcular pesos de cada clase
        total = datos_entrenamiento.shape[0]
        #classes = datos_entrenamiento.groupby(y_col_name).count().index.values
        weights = (total/datos_entrenamiento.groupby(y_col_name).count().values)/2
        # ct covid:0, no covid:1
        peso_clases = {0:weights[0][0], 1:weights[1][0]}
        loss='binary_crossentropy' 
    
    adam = Adam(lr=LR)
    finetune_model.compile(adam, loss=loss, metrics=metrics)
    early = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1,restore_best_weights=True)

    # Inicializar con semilla en tensorflow y numpy
    # import random
    # random.seed(8128)
    # np.random.seed(8128)
    # tf.random.set_seed(8128)

    if len(peso_clases)>0:
        print("\n")
        print("ESTOY USANDO PESADO DE CLASES!")      
        print(peso_clases)
        print("##############################")
        print("\n")
        history = finetune_model.fit(
                    train_generator,
                    epochs=NUM_EPOCHS, workers=1, 
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=generator_seguimiento,
                    validation_steps=pasos_seguimiento,
                    callbacks=[early],
                    #verbose=2,
                    verbose=1,
                    class_weight=peso_clases)
    else:
        history = finetune_model.fit(
                    train_generator,
                    epochs=NUM_EPOCHS, workers=1, 
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=generator_seguimiento,
                    validation_steps=pasos_seguimiento,
                    #verbose=2,
                    verbose=1,
                    callbacks=[early])

    if len(datos['df_val']) > 0:
        score1=finetune_model.evaluate(valid_generator,verbose=0,steps=STEP_SIZE_VALID)
        
    score2=finetune_model.evaluate(test1_generator,verbose=0,steps=STEP_SIZE_TEST1)

    #Confution Matrix and Classification Report
    #logits = finetune_model.predict(test1_generator, steps=df_test1.shape[0] // batch_size+1)
    #y_pred_class = np.argmax(logits, axis=1)
    #predicted_class_probab=np.max(logits,axis=1)
    #print('Confusion Matrix')
    #print(confusion_matrix(test1_generator.classes, y_pred_class))
    #print('Classification Report')

    if dataset == 'gleasson':
        score3=finetune_model.evaluate_generator(generator=test2_generator,verbose=1,steps=STEP_SIZE_TEST2)

    if len(datos['df_val']) > 0:
        print("Val  Loss      : ", score1[0])
        print("Val  Accuracy  : ", score1[1])
        print("\n")
    print("Test1 Loss     : ", score2[0])
    print("Test1 Accuracy : ", score2[1])
    print("\n")
    if dataset == 'gleasson':
        print("Test2 Loss     : ", score3[0])
        print("Test2 Accuracy : ", score3[1])
        logs.append([kfold,iteracion,arquitectura,score1[0],score1[1],score2[0],score2[1],score3[0],score3[1]])
    
    #guardar_logs(ruta,[logs[-1]])
    save_logs(logs,'train',pipeline)

    # Plot training & validation accuracy values
    #print(history.history)
    #print('--- Val acc ---')
    #print(history.history['val_acc'])
    plt.plot(history.history['acc'])
    if len(datos['df_val'])>0: 
        plt.plot(history.history['val_acc'])
    plt.title('Model accuracy - {}'.format(arquitectura))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    if len(datos['df_val'])>0:
        plt.plot(history.history['val_loss'])
    plt.title('Model loss - {}'.format(arquitectura))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    end_model = time.time()
    time_training = end_model - start_model
    print(f"training {arquitectura}",time_training)
    
    if pipeline['save_model']:
        save_path_model = os.path.join( pipeline['save_path_model'] , f'{kfold}_{iteracion}_{arquitectura}.h5' )
        finetune_model.save(save_path_model)
        model_performance['val_acc'] = score1[1]
        return save_path_model , model_performance
    
    logs_time.append([kfold,iteracion,arquitectura,time_training])
    save_logs(logs_time,'time',pipeline)

    #pipeline['logs_model']
    #time_training
    model_performance['val_acc'] = score1[1]
    model_performance['test1_acc'] = score2[1]
    model_performance['test2_acc'] = score3[1]
    return finetune_model , model_performance

def evaluate_cotrain(modelo1,modelo2,modelo3,arquitectura1,arquitectura2,arquitectura3,dataset_base,datos,etapa,kfold,iteracion,pipeline):

    train_generator_arch1,test1_generator_arch1,STEP_SIZE_TEST1_arch1=generadores(etapa,arquitectura1,datos,pipeline,False,dataset_base)
    train_generator_arch2,test1_generator_arch2,STEP_SIZE_TEST1_arch2=generadores(etapa,arquitectura2,datos,pipeline,False,dataset_base)
    train_generator_arch3,test1_generator_arch3,STEP_SIZE_TEST1_arch3=generadores(etapa,arquitectura3,datos,pipeline,False,dataset_base)

    df1=evaluar(modelo1,train_generator_arch1,test1_generator_arch1,STEP_SIZE_TEST1_arch1)
    df2=evaluar(modelo2,train_generator_arch2,test1_generator_arch2,STEP_SIZE_TEST1_arch2)
    df3=evaluar(modelo3,train_generator_arch3,test1_generator_arch3,STEP_SIZE_TEST1_arch3)

    predicciones = []
    for i in range(len(df1)):
        if ( df1['Predictions'][i] == df2['Predictions'][i] ) and ( df1['Predictions'][i] == df3['Predictions'][i] ):
            predicciones.append([df1['Filename'][i],df1['Predictions'][i]])
        else:
            probabilidades = np.array([df1['Max_Probability'][i],df2['Max_Probability'][i],df3['Max_Probability'][i]])
            indice_prob_max = probabilidades.argmax()

            clases = np.array([df1['Predictions'][i],df2['Predictions'][i],df3['Predictions'][i]])
            #indice_clas_max = clases.argmax() (relevant variable)

            real = np.array([df1['Filename'][i],df2['Filename'][i],df3['Filename'][i]])
            predicciones.append([real[indice_prob_max],clases[indice_prob_max]])

    results = pd.DataFrame(predicciones,columns=["filename","predictions"])

    if dataset == 'gleasson': 
        results['filename'] = results['filename'].apply(lambda x:x.split('/')[-1].split('_')[-1][0])
        y_true = results['filename'].values.tolist()
        y_pred = results['predictions'].values.tolist()

    from sklearn.metrics import accuracy_score
    co_train_accu = accuracy_score(y_pred,y_true)
    logs.append([kfold,iteracion,'co-train',None,None,None,co_train_accu])
    #guardar_logs(ruta,[logs[-1]])
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

def generadores(etapa,arquitectura,datos,pipeline,label_active,dataset_base):

    #print("Arquitectura {} en iteracion {}".format(arquitectura,iteracion))
    
    _ , preprocess_input = get_model(arquitectura, pipeline)

    if dataset == 'gleasson':
        datagen = ImageDataGenerator(
                                    preprocessing_function=preprocess_input,
                                    rotation_range=40,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.01,
                                    zoom_range=[0.9, 1.25],
                                    horizontal_flip=True,
                                    vertical_flip=False,
                                    fill_mode='reflect',
                                    data_format='channels_last'
                                )

    if etapa=='train':
        train_generator = datagen.flow_from_dataframe(
                         dataframe=datos['df_train'], 
                         x_col=x_col_name, 
                         y_col=y_col_name, 
                         target_size=(pipeline['img_height'],pipeline['img_width']),
                         class_mode='categorical', 
                         batch_size=batch_size,
                         seed=42,
                         shuffle=True)
        
    if etapa=='train_EL':
        train_generator = datagen.flow_from_dataframe(
                         dataframe=datos['df_train_EL'], 
                         x_col=x_col_name, 
                         y_col=y_col_name, 
                         target_size=(pipeline['img_height'],pipeline['img_width']),
                         class_mode='categorical', 
                         batch_size=batch_size,
                         seed=42,
                         shuffle=True)
    
    #if len(datos['df_val'])>0:
    #    val_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)   
    #    valid_generator=val_datagen.flow_from_dataframe(
    #                    dataframe=datos['df_val'],
    #                    x_col=x_col_name,
    #                    y_col=y_col_name,
    #                    batch_size=batch_size,
    #                    seed=42,
    #                    shuffle=True,
    #                    class_mode="categorical",
    #                    target_size=(pipeline['img_height'],pipeline['img_width']))
    
    test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
    
    if dataset == 'gleasson':
        test1_generator=test_datagen.flow_from_dataframe(
                      dataframe=datos['df_test1'],
                      x_col="patch_name1",
                      y_col="grade_1",
                      batch_size=batch_size,
                      seed=42,
                      shuffle=False,
                      class_mode="categorical",
                      target_size=(pipeline['img_height'],pipeline['img_width']))

        test2_generator=test_datagen.flow_from_dataframe(
                      dataframe=datos['df_test2'],
                      x_col="patch_name2",
                      y_col="grade_2",
                      batch_size=batch_size,
                      seed=42,
                      shuffle=False,
                      class_mode="categorical",
                      target_size=(pipeline['img_height'],pipeline['img_width']))
    
    #STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    #if len(datos['df_val']):
    #    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    STEP_SIZE_TEST1=test1_generator.n//test1_generator.batch_size

    if label_active:
        batchset_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

        batchset_generator=batchset_datagen.flow_from_dataframe(
                      dataframe=datos['df_batchset'],
                      x_col=x_col_name,
                      y_col=y_col_name,
                      batch_size=batch_size,
                      seed=42,
                      shuffle=False,
                      class_mode="categorical",
                      target_size=(pipeline['img_height'],pipeline['img_width']))

        STEP_SIZE_BATCH=batchset_generator.n//batchset_generator.batch_size

        return train_generator,batchset_generator,STEP_SIZE_BATCH

    if dataset_base == 'gleasson-patologo2':
        STEP_SIZE_TEST2=test2_generator.n//test2_generator.batch_size
        return train_generator,test2_generator,STEP_SIZE_TEST2
    
    if dataset_base == 'gleasson-patologo1':
        return train_generator,test1_generator,STEP_SIZE_TEST1

def labeling(etapa,modelo1,modelo2,modelo3,arquitectura1,arquitectura2,arquitectura3,EL,LC,datos,pipeline):
    etiquetados_EL = 0
    etiquetados_LC = 0
    
    train_generator_arch1,batchset_generator_arch1,STEP_SIZE_BATCH_arch1=generadores(etapa,arquitectura1,datos,pipeline,True,None)
    train_generator_arch2,batchset_generator_arch2,STEP_SIZE_BATCH_arch2=generadores(etapa,arquitectura2,datos,pipeline,True,None)
    train_generator_arch3,batchset_generator_arch3,STEP_SIZE_BATCH_arch3=generadores(etapa,arquitectura3,datos,pipeline,True,None)

    df1=evaluar(modelo1,train_generator_arch1,batchset_generator_arch1,STEP_SIZE_BATCH_arch1)
    df2=evaluar(modelo2,train_generator_arch2,batchset_generator_arch2,STEP_SIZE_BATCH_arch2)
    df3=evaluar(modelo3,train_generator_arch3,batchset_generator_arch3,STEP_SIZE_BATCH_arch3)

    for i in range(len(df1)):
        if (df1['Predictions'][i] == df2['Predictions'][i]) and (df1['Predictions'][i] == df3['Predictions'][i]) and (df1['Max_Probability'][i]>=confianza) and (df2['Max_Probability'][i]>=confianza) and (df3['Max_Probability'][i]>=confianza):
            EL.append([df1['Filename'][i],df1['Predictions'][i]])
            etiquetados_EL += 1
        else:
            LC.append([df1['Filename'][i],df1['Predictions'][i]])
            etiquetados_LC += 1

    print('etiquetados EL {} LC {}'.format(etiquetados_EL, etiquetados_LC))
    return EL,LC

def guardar_logs(ruta,lista):
    #save_logs(logs,pipeline)
    #guardar lista de listas en csv crear csv
    import csv
    import os
    if metodo == 'semi-supervisado':
        archivo = "{}logs/logs_{}_{}_{}_{}_{}_{}.csv".format(ruta,dataset,dataset_base,porcentaje,version,modalidad,str(confianza).replace('0.',''))
    if metodo == 'supervisado':
        archivo = "{}logs/logs_{}_{}_{}_{}.csv".format(ruta,dataset,dataset_base,version,modalidad)
    #carpeta = ruta+'logs/'

    file = open(archivo, "a")
    writer = csv.writer(file, delimiter = ",")
    for l in lista:
        writer.writerow(l)
    file.close()

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

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

## Preparar dataset
def ssl_global( archivos, model_zoo, csvs, pipeline ):

    datos = {}
    models_info = []
    df_train, df_val, df_test1, df_test2 = get_data(archivos, csvs)

    # Medir tiempo de ejecucion
    import time
    start = time.time()

    if dataset == 'gleasson':
        # Segmentacion 58.5% train y 19.5% val
        fold = dividir_balanceado2(df_train,4)

        # Extracción de muestras de test1
        #X_test1_=df_test1.iloc[:,0].values.tolist()
        #y_test1_=df_test1.iloc[:,1].values.tolist()

        # Extracción de muestras de test2
        #X_test2_=df_test2.iloc[:,0].values.tolist()
        #y_test2_=df_test2.iloc[:,1].values.tolist()

    #for kfold in range(4):
    for kfold in range(1):

        if dataset == 'gleasson':
            df_train_58         = pd.DataFrame([fold[kfold][0],fold[kfold][2]]).T
            df_train_58.columns = [x_col_name,y_col_name]

            df_val           = pd.DataFrame([fold[kfold][1],fold[kfold][3]]).T
            df_val.columns   = [x_col_name,y_col_name]

            fold1            = dividir_balanceado2(df_train_58,4)
            df_train         = pd.DataFrame([fold1[0][1],fold1[0][3]]).T
            df_train.columns = [x_col_name,y_col_name]

            df_train.to_csv('data/train.csv',index=False)
            df_val.to_csv('data/val.csv',index=False)
            df_test1.to_csv('data/test1.csv',index=False)
            df_test2.to_csv('data/test2.csv',index=False)

            # Extracción de muestras de train
            #X_train_=df_train.iloc[:,0].values.tolist()
            #y_train_=df_train.iloc[:,1].values.tolist()

            df_U         = pd.DataFrame([fold1[0][0],fold1[0][2]]).T
            df_U.columns = [x_col_name,y_col_name]
            EL,LC        = [],[]

            print("train :",len(df_train))
            print("val   :",len(df_val))
            print("u     :",len(df_U))

            # Segmentación de U en lotes para etiquetar
            batch_set=list(dividir_lotes(df_U, numero_lotes))
            for i in range(len(batch_set)):
                print(len(batch_set[i].iloc[:,0].values.tolist()))
            
        datos['df_train'] = df_train
        datos['df_val'] = df_val
        datos['df_test1'] = df_test1
        datos['df_test2'] = df_test2

        for iteracion in range(numero_lotes*1):

            import random
            random.seed(SEED)
            np.random.seed(SEED)
            tensorflow.random.set_random_seed(SEED)


            # # Agregar semilla
            #import random
            #random.seed(8128)
            #np.random.seed(8128)
            #tensorflow.random.set_random_seed(8128)

            #from numpy import seed
            #seed(8128)
            #from tensorflow import set_random_seed
            #set_random_seed(8128)

            # Agregar semilla
            #import random
            #random.seed(42)
            #np.random.seed(42)
            #tf.random.set_seed(42)

            print("\n######################")
            print("K-FOLD {} - ITERACION {}".format(kfold,iteracion))
            print("######################\n")

            if iteracion == 0:
                #modeloA = None
                #modeloB = None
                #modeloC = None
                etapa = 'train'
            else:
                #modeloA = mod_tmpA
                #modeloB = mod_tmpB
                #modeloC = mod_tmpC
                etapa = 'train_EL'

            #mod_tmpA = entrenamiento(kfold,etapa,datos,model_zoo[0],0.00001,train_epochs,batch_epochs,early_stopping,iteracion,batch_size,pipeline)
            #mod_tmpB = entrenamiento(kfold,etapa,datos,model_zoo[1],0.00001,train_epochs,batch_epochs,early_stopping,iteracion,batch_size,pipeline)
            #mod_tmpC = entrenamiento(kfold,etapa,datos,model_zoo[2],0.00001,train_epochs,batch_epochs,early_stopping,iteracion,batch_size,pipeline)
            #mod_tmpD = entrenamiento(kfold,etapa,datos,model_zoo[3],0.00001,train_epochs,batch_epochs,early_stopping,iteracion,batch_size,pipeline)
            #mod_tmpE = entrenamiento(kfold,etapa,datos,model_zoo[4],0.00001,train_epochs,batch_epochs,early_stopping,iteracion,batch_size,pipeline)
            #mod_tmpF = entrenamiento(kfold,etapa,datos,model_zoo[5],0.00001,train_epochs,batch_epochs,early_stopping,iteracion,batch_size,pipeline)
            #mod_tmpG = entrenamiento(kfold,etapa,datos,model_zoo[6],0.00001,train_epochs,batch_epochs,early_stopping,iteracion,batch_size,pipeline)
            #mod_tmpH = entrenamiento(kfold,etapa,datos,model_zoo[7],0.00001,train_epochs,batch_epochs,early_stopping,iteracion,batch_size,pipeline)
            #mod_tmpI = entrenamiento(kfold,etapa,datos,model_zoo[8],0.00001,train_epochs,batch_epochs,early_stopping,iteracion,batch_size,pipeline)
            #mod_tmpJ = entrenamiento(kfold,etapa,datos,model_zoo[9],0.00001,train_epochs,batch_epochs,early_stopping,iteracion,batch_size,pipeline)

            for model in model_zoo:
                model_memory , model_performance = entrenamiento(kfold,etapa,datos,model,0.00001,train_epochs,batch_epochs,early_stopping,iteracion,batch_size,pipeline)            
                #models_info.append([model_memory , model, model_performance['val_acc']])
                models_info.append([model_memory , model, model_performance['test1_acc']])
            
            #if semi_method == 'supervised':
            #    break

            #df_models_performance = pd.DataFrame(models_info, columns=['model_path','model_architecture','model_val_acc'])
            #top3_models = df_models_performance.sort_values('model_val_acc',ascending=False).iloc[:3,:].reset_index().drop('index',axis=1)
            df_models_performance = pd.DataFrame(models_info, columns=['model_path','model_architecture','model_test1_acc'])
            top3_models = df_models_performance.sort_values('model_test1_acc',ascending=False).iloc[:3,:].reset_index().drop('index',axis=1)

            mod_top1, arch_top1 = top3_models.loc[0,'model_path'],top3_models.loc[0,'model_architecture']
            mod_top2, arch_top2 = top3_models.loc[1,'model_path'],top3_models.loc[1,'model_architecture']
            mod_top3, arch_top3 = top3_models.loc[2,'model_path'],top3_models.loc[2,'model_architecture']

            #df_models_performance = pd.DataFrame(models_info, columns=['model_path','model_architecture','model_val_acc'])
            #top3_models = df_models_performance.sort_values('model_val_acc',ascending=False).iloc[:,:2]

            #mod_top1, arch_top1 = load_model(top3_models.loc[0,'model_path']) , top3_models.loc[0,'model_architecture']
            #mod_top2, arch_top2 = load_model(top3_models.loc[1,'model_path']) , top3_models.loc[1,'model_architecture']
            #mod_top3, arch_top3 = load_model(top3_models.loc[2,'model_path']) , top3_models.loc[2,'model_architecture']

            if dataset == 'gleasson':
                print("\nCo-train1: \n",evaluate_cotrain(mod_top1,mod_top2,mod_top3,arch_top1,arch_top2,arch_top3,'gleasson-patologo1',datos,etapa,kfold,iteracion,pipeline))
                print("\nCo-train2: \n",evaluate_cotrain(mod_top1,mod_top2,mod_top3,arch_top1,arch_top2,arch_top3,'gleasson-patologo2',datos,etapa,kfold,iteracion,pipeline))

            if semi_method == 'supervised':
                break

            if iteracion < numero_lotes:
                
                df_batchset = batch_set[iteracion]
                df_batchset.columns = [x_col_name,y_col_name]
                df_batchset[y_col_name] = '0'
                #LC_backup = LC.copy()
            else:
                if  iteracion == numero_lotes:
                    df_LC = pd.DataFrame(LC)
                    batch_set_LC=list(dividir_lotes(df_LC, numero_lotes))
                    for i in range(len(batch_set_LC)):
                        print(len(batch_set_LC[i].iloc[:,0].values.tolist()))
                    LC = []
                
                df_batchset = pd.DataFrame([batch_set_LC[int(iteracion-numero_lotes)].iloc[:,0].values.tolist()]).T
                df_batchset.columns = [x_col_name]
                df_batchset[y_col_name] = '0'

            datos['df_batchset'] = df_batchset
            
            #label_active = True
            EL,LC = labeling(etapa,mod_top1,mod_top2,mod_top3,arch_top1,arch_top2,arch_top3,EL,LC,datos,pipeline)
            logs_label.append([kfold,iteracion,arch_top1,arch_top2,arch_top3,len(EL),len(LC)])
            save_logs(logs_label,'label',pipeline)
            #label_active = False
            df_EL = pd.DataFrame(EL,columns=[x_col_name,y_col_name])
            df_train_EL = pd.concat([df_train,df_EL])
            datos['df_train_EL'] = df_train_EL
            reset_keras()
            models_info = []
        
    end = time.time()
    print(end - start)

pipeline = {}

server = 'bivl2ab'
dataset = 'gleasson'
dataset_base = 'hardvard'
metodo = 'semi-supervisado'

csvs = '/home/miguel/gleasson/dataset/tma_info/'
archivos = '/home/miguel/gleasson/'
ruta = '/home/miguel/gleasson/'

pipeline['save_path_model'] = '/home/miguel/gleasson/models/v3/'

x_col_name = 'patch_name'
y_col_name = 'grade_'

if dataset == 'gleasson':
    pipeline['weights'] = 'imagenet'
    #pipeline['weights'] = None
    pipeline['img_height'] = 299
    pipeline['img_width'] = 299
    #HEIGHT = 299
    #WIDTH = 299
    augmenting_factor = 1.5
    clases = 4
    batch_size = 16
    confianza = 0.90
    
    pipeline['save_model'] = False
    pipeline['save_path_logs'] = 'logs/'
    pipeline['id'] = 11

EL,LC,test_cotraining,predicciones = [],[],[],[]
logs,logs_time,logs_label = [], [], []

data_aumentation = True
early_stopping = True
semi_method = 'co-training-multi'
#semi_method = 'supervised'
LR = 0.00001
#peso_clases = {}
modalidad = 'lento'
#configuracion =  { 
#   dataset='covid19', modalidad='rapido',
#   dataset='satellital', modalidad='medio',
# }
#version = version_automatica(ruta)
version = pipeline["id"]
porcentaje='10%'
numero_lotes = 5
label_active = False

if modalidad == 'ultra-fast':
    train_epochs = 1
    batch_epochs = 1

if modalidad == 'ultra':
    train_epochs = 5
    batch_epochs = 5

if modalidad == 'rapido':
    train_epochs = 10
    batch_epochs = 10

if modalidad == 'medio':
    train_epochs = 20
    batch_epochs = 20

if modalidad == 'lento':
    train_epochs = 30
    batch_epochs = 30

if dataset == 'gleasson':
    logs.append(["kfold","iteracion","arquitectura","val_loss","val_accu",
    "test1_loss","test1_accu","test2_loss","test2_accu"])
    logs_time.append(["kfold","iteracion","arquitectura","training_time"])
    logs_label.append(["kfold","iteracion","arquitectura","EL","LC"])

#ruta = ''
#os.makedirs('logs', exist_ok=True)
#os.makedirs(pipeline['save_path_model'] , exist_ok=True)
#guardar_logs(ruta,[logs[-1]])

save_logs(logs,'train',pipeline)
save_logs(logs_time,'time',pipeline)
save_logs(logs_label,'label',pipeline)

#model_zoo = [ 'InceptionV3', 'InceptionV4', 
#'ResNet50', 'ResNet101', 'ResNet152', 
#'DenseNet121', 'Dense169', 'Dense201', 
#'NASNetLarge', 'Xception']

#ACA VOY
#model_zoo = [ 'InceptionV3', 'InceptionV4', 
#'ResNet50', 'ResNet101', 
#'DenseNet121', 'DenseNet169', 
#'Xception' ]

#model_zoo = ['ResNet50','Xception','DenseNet169','InceptionV4','DenseNet121']
model_zoo = ['Xception','ResNet101','DenseNet169']

ssl_global( archivos, model_zoo , csvs, pipeline )