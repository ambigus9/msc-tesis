"SSL generators"

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ssl_train import get_model
from ssl_train import get_preprocess_function

def generadores(etapa, architecture, datos, pipeline, label_active, iteracion, models_info):

    preprocess_function = get_preprocess_function(architecture)

    datagen = ImageDataGenerator(
                                    preprocessing_function=preprocess_function,
                                    rotation_range=90,
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
