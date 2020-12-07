"SSL generators"

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ssl_train import get_model

def generadores(etapa, architecture, datos, pipeline, label_active, iteracion, models_info):

    _ , preprocess_input = get_model(architecture, iteracion, models_info, pipeline)

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

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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
        batchset_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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

        return train_generator, batchset_generator, STEP_SIZE_BATCH

    return train_generator, test_generator, STEP_SIZE_TEST
