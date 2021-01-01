"SSL train"

import os
import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

#from ssl_eval import classification_metrics

from utils_general import save_plots
from utils_general import save_logs

from ml_strategy import transfer_learning_classic
from ml_strategy import transfer_learning_soft

def training(kfold, etapa, datos, architecture, iteracion, models_info, classification_metrics, pipeline):

    start_model = time.time()
    base_model, preprocess_input = get_model(architecture, iteracion, models_info, pipeline)
    model_performance = {}

    datagen = ImageDataGenerator(
                                    preprocessing_function=preprocess_input,
                                    rotation_range=90,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                )

    if etapa=='train':
        print("CREATING GENERATOR FOR TRAIN FROM SSL_TRAIN")
        train_generator = datagen.flow_from_dataframe(
                         dataframe=datos['df_train'],
                         x_col=pipeline["x_col_name"],
                         y_col=pipeline["y_col_name"],
                         target_size=(pipeline['img_height'],pipeline['img_width']),
                         class_mode='categorical',
                         batch_size=pipeline["batch_size"],
                         seed=42,
                         shuffle=True)
        print("OK - CREATING GENERATOR FOR TRAIN FROM SSL_TRAIN")

    if etapa=='train_EL':
        print("CREATING GENERATOR FOR TRAIN FROM SSL_TRAIN")
        train_generator = datagen.flow_from_dataframe(
                         dataframe=datos['df_train_EL'],
                         x_col=pipeline["x_col_name"],
                         y_col=pipeline["y_col_name"],
                         target_size=(pipeline['img_height'],pipeline['img_width']),
                         class_mode='categorical',
                         batch_size=pipeline["batch_size"],
                         seed=42,
                         shuffle=True)
        print("OK - CREATING GENERATOR FOR TRAIN_EL FROM SSL_TRAIN")

    if len(datos['df_val']) > 0:
        print("CREATING GENERATOR FOR VAL FROM SSL_TRAIN")
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

        print("OK - CREATING GENERATOR FOR VAL FROM SSL_TRAIN")

    print("CREATING GENERATOR FOR TEST FROM SSL_TRAIN")
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
    print("OK - CREATING GENERATOR FOR TEST FROM SSL_TRAIN")

    num_classes = len( datos["df_train"][ pipeline["y_col_name"] ].unique() )
    print("NUM CLASSES", num_classes)

    if pipeline["transfer_learning"] == "classic":
        if pipeline["restart_weights"]:
            print("TRANSFER LEARNING - CLASSIC + YES RESTART WEIGHTS")
            finetune_model = transfer_learning_classic( base_model, num_classes )
            print("OK - TRANSFER LEARNING - CLASSIC + YES RESTART WEIGHTS")
        else:
            if etapa == 'train':
                print("TRANSFER LEARNING - TRAIN + CLASSIC")
                finetune_model = transfer_learning_classic( base_model, num_classes )
                print("OK - TRANSFER LEARNING - TRAIN + CLASSIC")
            elif etapa == 'train_EL':
                print("TRANSFER LEARNING - TRAIN_EL + CLASSIC + NO RESTART WEIGHTS")
                finetune_model = base_model
                print("OK - TRANSFER LEARNING - TRAIN_EL + CLASSIC + NO RESTART WEIGHTS")
    elif pipeline["transfer_learning"] == "soft":
        print("TRANSFER LEARNING - TRAIN + SOFT")
        finetune_model = transfer_learning_soft( base_model, num_classes,
                                                 pipeline["stage_config"][iteracion] )
        print("OK - TRANSFER LEARNING - TRAIN + SOFT")    

    if etapa == 'train':
        NUM_EPOCHS = pipeline["modality_config"][pipeline["modality"]]["train_epochs"]
        num_train_images = len(datos['df_train'])*pipeline["aug_factor"]
    if etapa == 'train_EL':
        NUM_EPOCHS = pipeline["modality_config"][pipeline["modality"]]["batch_epochs"]
        num_train_images = len(datos['df_train_EL'])*pipeline["aug_factor"]

    STEP_SIZE_TRAIN=num_train_images//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

    metrics = ['accuracy']
    loss='categorical_crossentropy'

    if pipeline["transfer_learning"] == "classic":
        LR = pipeline['learning_rate']
    elif pipeline["transfer_learning"] == "soft":
        LR = pipeline["stage_config"][iteracion]['LR']
    
    print(f"LEARNING RATE: {LR}")
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

    class_metrics = classification_metrics(finetune_model, train_generator, test_generator, STEP_SIZE_TEST)

    print("Val  Loss : ", val_score[0])
    print("Test Loss : ", test_score[0])
    print("Val  Accuracy : ", val_score[1])
    print("Test Accuracy : ", test_score[1])

    print(f"Test Precision: {class_metrics[0]}")
    print(f"Test Recall: {class_metrics[1]}")
    print(f"Test F1-Score: {class_metrics[2]}")
    print(f"Test Support: {class_metrics[3]}")

    end_model = time.time()
    time_training = end_model - start_model
    print(f"training time of - {architecture}",time_training)

    logs = []
    logs.append([kfold,iteracion,architecture,val_score[0],val_score[1],
            test_score[0],test_score[1],class_metrics[0],class_metrics[1],
            class_metrics[2],class_metrics[3]])

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

def get_preprocess_function(architecture):
    
    if architecture == 'InceptionV3':
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    
    if architecture == 'InceptionV4':
        from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

    if architecture == 'ResNet152':
        from tensorflow.keras.applications.resnet import preprocess_input

    return preprocess_input    

def get_model(architecture, iteracion, models_info, pipeline):

    print("="*len(architecture))
    print(architecture)
    print("="*len(architecture))

    if iteracion > 0 and not pipeline["restart_weights"]:
        print("USING MODELS FROM MEMORY")
        base_model = models_info[architecture]['model_memory']
        print("OK - USING MODELS FROM MEMORY")

    if architecture == 'InceptionV3':
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

        if iteracion == 0 or pipeline["restart_weights"]:
            base_model = InceptionV3(weights=pipeline['weights'],include_top=False,input_shape=(pipeline['img_height'], pipeline['img_width'], 3))
            print(f"OK - RESTARTING WEIGHTS FROM IMAGENET FOR {architecture}")

    if architecture == 'InceptionV4':
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

        if iteracion == 0 or pipeline["restart_weights"]:
            base_model = InceptionResNetV2(weights=pipeline['weights'],include_top=False,input_shape=(pipeline['img_height'], pipeline['img_width'], 3))
            print(f"OK - RESTARTING WEIGHTS FROM IMAGENET FOR {architecture}")

    if architecture == 'ResNet50':
        from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
        if iteracion == 0 or pipeline["restart_weights"]:
            base_model = ResNet50(weights=pipeline['weights'],include_top=False,input_shape=(pipeline['img_height'], pipeline['img_width'], 3))
    if architecture == 'ResNet101':
        from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
        if iteracion == 0 or pipeline["restart_weights"]:
            base_model = ResNet101(weights=pipeline['weights'],include_top=False,input_shape=(pipeline['img_height'], pipeline['img_width'], 3))
    
    if architecture == 'ResNet152':
        from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
        if iteracion == 0 or pipeline["restart_weights"]:
            base_model = ResNet152(weights=pipeline['weights'],include_top=False,input_shape=(pipeline['img_height'], pipeline['img_width'], 3))
            print(f"OK - RESTARTING WEIGHTS FROM IMAGENET FOR {architecture}")

    if architecture == 'DenseNet121':
        from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
        if iteracion == 0 or pipeline["restart_weights"]:
            base_model = DenseNet121(weights=pipeline['weights'],include_top=False,input_shape=(pipeline['img_height'], pipeline['img_width'], 3))
    if architecture == 'DenseNet169':
        from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input
        if iteracion == 0 or pipeline["restart_weights"]:
            base_model = DenseNet169(weights=pipeline['weights'],include_top=False,input_shape=(pipeline['img_height'], pipeline['img_width'], 3))
    if architecture == 'DenseNet201': 
        from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
        if iteracion == 0 or pipeline["restart_weights"]:
            base_model = DenseNet201(weights=pipeline['weights'],include_top=False,input_shape=(pipeline['img_height'], pipeline['img_width'], 3))
    if architecture == 'NASNetLarge': 
        from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
        if iteracion == 0 or pipeline["restart_weights"]:
            base_model = NASNetLarge(weights=pipeline['weights'],include_top=False,input_shape=(pipeline['img_height'], pipeline['img_width'], 3))
    if architecture == 'Xception':
        from tensorflow.keras.applications.xception import Xception, preprocess_input
        if iteracion == 0 or pipeline["restart_weights"]:
            base_model = Xception(weights=pipeline['weights'], include_top=False, input_shape=(pipeline['img_height'], pipeline['img_width'], 3))

    return base_model, preprocess_input
