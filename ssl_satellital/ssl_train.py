"SSL train"

import os
import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from utils_general import save_plots
from utils_general import save_logs

from ml_strategy import transfer_learning_classic
from ml_strategy import transfer_learning_soft

def training(kfold, etapa, datos, architecture, iteracion, models_info, pipeline):

    start_model = time.time()
    base_model, preprocess_input = get_model(architecture, iteracion, models_info, pipeline)
    model_performance = {}

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
                         x_col=pipeline["x_col_name"],
                         y_col=pipeline["y_col_name"],
                         target_size=(pipeline['img_height'],pipeline['img_width']),
                         class_mode='categorical',
                         batch_size=pipeline["batch_size"],
                         seed=42,
                         shuffle=True)

    if etapa=='train_EL':
        train_generator = datagen.flow_from_dataframe(
                         dataframe=datos['df_train_EL'],
                         x_col=pipeline["x_col_name"],
                         y_col=pipeline["y_col_name"],
                         target_size=(pipeline['img_height'],pipeline['img_width']),
                         class_mode='categorical',
                         batch_size=pipeline["batch_size"],
                         seed=42,
                         shuffle=True)

    if len(datos['df_val']) > 0:
        val_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

        valid_generator=val_datagen.flow_from_dataframe(
                        dataframe=datos['df_val'],
                        x_col=pipeline["x_col_name"],
                        y_col=pipeline["y_col_name"],
                        batch_size=pipeline["batch_size"],
                        seed=42,
                        shuffle=True,
                        class_mode="categorical",
                        target_size=(pipeline['img_height'],pipeline['img_width']))

    test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

    test_generator=test_datagen.flow_from_dataframe(
                      dataframe=datos['df_test'],
                      x_col=pipeline["x_col_name"],
                      y_col=pipeline["y_col_name"],
                      batch_size=pipeline["batch_size"],
                      seed=42,
                      shuffle=False,
                      class_mode="categorical",
                      target_size=(pipeline['img_height'],pipeline['img_width']))

    #if etapa == 'train' or etapa == 'train_EL':
    if etapa == 'train' and pipeline["transfer_learning"] == "classic":
        finetune_model = transfer_learning_classic(base_model,
                                                len( datos["df_train"]["y_col_name"].unique() ))
    elif pipeline["transfer_learning"] == "soft":
        finetune_model = transfer_learning_soft(base_model,
                                                len( datos["df_train"]["y_col_name"].unique() ),
                                                pipeline["stage_config"][iteracion])
    else:
        finetune_model = base_model
    

    if etapa == 'train':
        NUM_EPOCHS = pipeline["modality_config"][pipeline["modality"]]["train_epochs"]
        num_train_images = len(datos['df_train'])*pipeline["aug_factor"]
        #datos_entrenamiento = datos['df_train'].copy()
    if etapa == 'train_EL':
        NUM_EPOCHS = pipeline["modality_config"][pipeline["modality"]]["batch_epochs"]
        num_train_images = len(datos['df_train_EL'])*pipeline["aug_factor"]
        #datos_entrenamiento = datos['df_train_EL'].copy()

    STEP_SIZE_TRAIN=num_train_images//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

    metrics = ['accuracy']
    loss='categorical_crossentropy'

    if pipeline["transfer_learning"] == "soft":
        LR = pipeline['LR']
    else:
        LR = pipeline["stage_config"][iteracion]['LR']
    
    adam = Adam(lr=float(LR))
    finetune_model.compile(adam, loss=loss, metrics=metrics)

    early = EarlyStopping(monitor='val_loss',
                        min_delta=1e-3,
                        patience=5,
                        verbose=1,
                        restore_best_weights=True)

    history = finetune_model.fit(train_generator,
                epochs=NUM_EPOCHS,
                workers=1,
                steps_per_epoch=STEP_SIZE_TRAIN,
                validation_data=valid_generator,
                validation_steps=STEP_SIZE_VALID,
                verbose=1,
                callbacks=[early])

    val_score=finetune_model.evaluate(valid_generator,verbose=0,steps=STEP_SIZE_VALID)
    test_score=finetune_model.evaluate(test_generator,verbose=0,steps=STEP_SIZE_TEST)

    print("Val  Loss : ", val_score[0])
    print("Test Loss : ", test_score[0])
    print("Val  Accuracy : ", val_score[1])
    print("Test Accuracy : ", test_score[1])

    end_model = time.time()
    time_training = end_model - start_model
    print(f"training {architecture}",time_training)

    logs = []
    logs.append([kfold,iteracion,architecture,val_score[0],val_score[1],
            test_score[0],test_score[1]])

    logs_time = []
    logs_time.append([kfold,iteracion,architecture,time_training])

    save_logs(logs,'train',pipeline)
    save_logs(logs_time,'time',pipeline)
    save_plots(history, kfold, iteracion, architecture, pipeline)

    model_performance['val_acc'] = val_score[1]
    model_performance['test_acc'] = test_score[1]

    if pipeline['save_model']:
        save_path_model = os.path.join(
            pipeline['save_path_model'],
            f'{kfold}_{iteracion}_{architecture}.h5')
        finetune_model.save(save_path_model)
        model_performance['val_acc'] = val_score[1]
        return save_path_model , model_performance

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
