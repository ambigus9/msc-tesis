
def entrenamiento(  kfold,etapa,datos,arquitectura,train_epochs,
                    batch_epochs,early_stopping,iteracion,models_info,pipeline  ):
    import time
    start_model = time.time()
    base_model, preprocess_input = get_model(arquitectura, iteracion, models_info, pipeline)
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
                         batch_size=pipeline["batch_size"],
                         seed=42,
                         shuffle=True)

    if etapa=='train_EL':
        train_generator = datagen.flow_from_dataframe(
                         dataframe=datos['df_train_EL'],
                         x_col=x_col_name,
                         y_col=y_col_name,
                         target_size=(pipeline['img_height'],pipeline['img_width']),
                         class_mode='categorical',
                         batch_size=pipeline["batch_size"],
                         seed=42,
                         shuffle=True)

    if len(datos['df_val']) > 0:
        val_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

        valid_generator=val_datagen.flow_from_dataframe(
                        dataframe=datos['df_val'],
                        x_col=x_col_name,
                        y_col=y_col_name,
                        batch_size=pipeline["batch_size"],
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
                      batch_size=pipeline["batch_size"],
                      seed=42,
                      shuffle=False,
                      class_mode="categorical",
                      target_size=(pipeline['img_height'],pipeline['img_width']))

        test2_generator=test_datagen.flow_from_dataframe(
                    dataframe=datos['df_test2'],
                    x_col="patch_name2",
                    y_col="grade_2",
                    batch_size=pipeline["batch_size"],
                    seed=42,
                    shuffle=False,
                    class_mode="categorical",
                    target_size=(pipeline['img_height'], pipeline['img_width']))

    if etapa == 'train' or etapa == 'train_EL':
        finetune_model = transfer_learning_soft(
                                                base_model,
                                                pipeline["class_num"],
                                                pipeline["stage_config"][iteracion])

    #entrenar modelo
    from tensorflow.keras.optimizers import SGD, Adam, RMSprop

    if etapa == 'train':
        NUM_EPOCHS = train_epochs
        num_train_images = len(datos['df_train'])*pipeline["augmenting_factor"]
        datos_entrenamiento = datos['df_train'].copy()
    if etapa == 'train_EL':
        NUM_EPOCHS = batch_epochs
        num_train_images = len(datos['df_train_EL'])*pipeline["augmenting_factor"]
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

    if pipeline["class_num"]>3:
        metrics = ['accuracy']
        loss='categorical_crossentropy'
        peso_clases = {}
        total = datos_entrenamiento.shape[0]
        weights = (total/datos_entrenamiento.groupby(y_col_name).count().values)/4
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

    if pipeline["class_num"]==3:
        metrics = ['accuracy']
        # calcular pesos de cada clase
        total = datos_entrenamiento.shape[0]
        weights = (total/datos_entrenamiento.groupby(y_col_name).count().values)/3
        peso_clases = {0:weights[0][0], 1:weights[1][0], 2:weights[2][0]}
    elif pipeline["class_num"]==2:
        metrics = ['accuracy']
        # calcular pesos de cada clase
        total = datos_entrenamiento.shape[0]
        weights = (total/datos_entrenamiento.groupby(y_col_name).count().values)/2
        peso_clases = {0:weights[0][0], 1:weights[1][0]}
        loss='binary_crossentropy'

    adam = Adam(lr=pipeline["stage_config"][iteracion]['LR'])
    finetune_model.compile(adam, loss=loss, metrics=metrics)
    early = EarlyStopping(
                            monitor='val_loss',
                            min_delta=1e-3,
                            patience=5,
                            verbose=1,
                            restore_best_weights=True)

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
                    verbose=1,
                    class_weight=peso_clases)
    else:
        history = finetune_model.fit(
                    train_generator,
                    epochs=NUM_EPOCHS, workers=1,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=generator_seguimiento,
                    validation_steps=pasos_seguimiento,
                    verbose=1,
                    callbacks=[early])

    if len(datos['df_val']) > 0:
        score1=finetune_model.evaluate(valid_generator,verbose=0,steps=STEP_SIZE_VALID)

    score2=finetune_model.evaluate(test1_generator,verbose=0,steps=STEP_SIZE_TEST1)

    if dataset == 'gleasson':
        score3=finetune_model.evaluate_generator(generator=test2_generator,
        verbose=1,
        steps=STEP_SIZE_TEST2)

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
        logs.append([
            kfold,iteracion,arquitectura,
            score1[0],score1[1],
            score2[0],score2[1],
            score3[0],score3[1]])

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
        save_path_model = os.path.join(
            pipeline['save_path_model'],
            f'{kfold}_{iteracion}_{arquitectura}.h5')
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

def get_model(architecture, iteracion, models_info, pipeline):

    print("="*len(architecture))
    print(architecture)
    print("="*len(architecture))

    if iteracion > 0:
        base_model = models_info[architecture]['model_memory']

    if architecture == 'InceptionV3':
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        if iteracion == 0:
            base_model = InceptionV3(weights=pipeline['weights'],include_top=False,input_shape=(pipeline['img_height'], pipeline['img_width'], 3))
    if architecture == 'InceptionV4':
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
        if iteracion == 0:
            base_model = InceptionResNetV2(weights=pipeline['weights'],include_top=False,input_shape=(pipeline['img_height'], pipeline['img_width'], 3))
    if architecture == 'ResNet50':
        from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
        if iteracion == 0:
            base_model = ResNet50(weights=pipeline['weights'],include_top=False,input_shape=(pipeline['img_height'], pipeline['img_width'], 3))
    if architecture == 'ResNet101':
        from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
        if iteracion == 0:
            base_model = ResNet101(weights=pipeline['weights'],include_top=False,input_shape=(pipeline['img_height'], pipeline['img_width'], 3))
    if architecture == 'ResNet152':
        from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
        if iteracion == 0:
            base_model = ResNet152(weights=pipeline['weights'],include_top=False,input_shape=(pipeline['img_height'], pipeline['img_width'], 3))
    if architecture == 'DenseNet121':
        from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
        if iteracion == 0:
            base_model = DenseNet121(weights=pipeline['weights'],include_top=False,input_shape=(pipeline['img_height'], pipeline['img_width'], 3))
    if architecture == 'DenseNet169':
        from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input
        if iteracion == 0:
            base_model = DenseNet169(weights=pipeline['weights'],include_top=False,input_shape=(pipeline['img_height'], pipeline['img_width'], 3))
    if architecture == 'DenseNet201': 
        from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
        if iteracion == 0:
            base_model = DenseNet201(weights=pipeline['weights'],include_top=False,input_shape=(pipeline['img_height'], pipeline['img_width'], 3))
    if architecture == 'NASNetLarge': 
        from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
        if iteracion == 0:
            base_model = NASNetLarge(weights=pipeline['weights'],include_top=False,input_shape=(pipeline['img_height'], pipeline['img_width'], 3))
    if architecture == 'Xception':
        from tensorflow.keras.applications.xception import Xception, preprocess_input
        if iteracion == 0:
            base_model = Xception(weights=pipeline['weights'], include_top=False, input_shape=(pipeline['img_height'], pipeline['img_width'], 3))

    return base_model, preprocess_input
