"SSL generators"

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ssl_train import get_model
from ssl_train import get_preprocess_function

def generadores(etapa, architecture, datos, pipeline, label_active, iteracion, models_info):

    preprocess_function = get_preprocess_function(architecture)

    if not pipeline["aug_stages"]:
        print("USING TRANSFORMATIONS FROM ML_GENERATORS")
        datagen = ImageDataGenerator(
                                        preprocessing_function=preprocess_function,
                                        rotation_range=360,
                                        zoom_range=[0.1,0.5],
                                        brightness_range=[0.1,0.5],
                                        shear_range=0.2,
                                        fill_mode='nearest',
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                    )
        print(datagen)
        print("OK - USING TRANSFORMATIONS FROM ML_GENERATORS")

    if iteracion == 0 and pipeline["aug_stages"]:
        datagen = ImageDataGenerator(
                                        preprocessing_function=preprocess_function,
                                        rotation_range=360,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                    )
    elif iteracion == 1 and pipeline["aug_stages"]:
        datagen = ImageDataGenerator(
                                        preprocessing_function=preprocess_function,
                                        rotation_range=360,
                                        zoom_range=[0.1,0.5],
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                    )
    elif iteracion >= 2 and pipeline["aug_stages"]:
        datagen = ImageDataGenerator(
                                        preprocessing_function=preprocess_function,
                                        rotation_range=360,
                                        brightness_range=[0.1,0.5],
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                    )

    if etapa=='train':
        print("CREATING GENERATOR FOR TRAIN FROM GENERATORS")
        train_generator = datagen.flow_from_dataframe(
                         dataframe=datos['df_train'],
                         x_col=pipeline["x_col_name"],
                         y_col=pipeline["y_col_name"],
                         target_size=(pipeline['img_height'],pipeline['img_width']),
                         class_mode='categorical',
                         batch_size=pipeline["batch_size"],
                         seed=42,
                         shuffle=True)
        print("OK - CREATING GENERATOR FOR TRAIN FROM GENERATORS")

    if etapa=='train_EL':
        print("CREATING GENERATOR FOR TRAIN_EL FROM GENERATORS")
        train_generator = datagen.flow_from_dataframe(
                         dataframe=datos['df_train_EL'],
                         x_col=pipeline["x_col_name"],
                         y_col=pipeline["y_col_name"],
                         target_size=(pipeline['img_height'],pipeline['img_width']),
                         class_mode='categorical',
                         batch_size=pipeline["batch_size"],
                         seed=42,
                         shuffle=True)
        print("OK - CREATING GENERATOR FOR TRAIN_EL FROM GENERATORS")

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_function)

    test_generator=test_datagen.flow_from_dataframe(
                    dataframe=datos['df_test'],
                    x_col=pipeline["x_col_name"],
                    y_col=pipeline["y_col_name"],
                    batch_size=pipeline["batch_size"],
                    seed=42,
                    shuffle=False,
                    class_mode="categorical",
                    target_size=(pipeline['img_height'],pipeline['img_width']))

    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

    if label_active:
        print("LABEL ACTIVE FROM GENERATORS ...")
        batchset_datagen = ImageDataGenerator(preprocessing_function=preprocess_function)

        batchset_generator=batchset_datagen.flow_from_dataframe(
                      dataframe=datos['df_batchset'],
                      x_col=pipeline["x_col_name"],
                      y_col=pipeline["y_col_name"],
                      batch_size=pipeline["batch_size"],
                      seed=42,
                      shuffle=False,
                      class_mode="categorical",
                      target_size=(pipeline['img_height'],pipeline['img_width']))

        STEP_SIZE_BATCH=batchset_generator.n//batchset_generator.batch_size
        print("OK - LABEL ACTIVE FROM GENERATORS ...")
        return train_generator, batchset_generator, STEP_SIZE_BATCH

    return train_generator, test_generator, STEP_SIZE_TEST
