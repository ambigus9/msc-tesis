def generadores(etapa,architecture,datos,pipeline,label_active,dataset_base,iteracion, models_info):

    _ , preprocess_input = get_model(architecture, iteracion, models_info, pipeline)

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

    test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

    if dataset == 'gleasson':
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
                      target_size=(pipeline['img_height'],pipeline['img_width']))

    STEP_SIZE_TEST1=test1_generator.n//test1_generator.batch_size

    if label_active:
        batchset_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

        batchset_generator=batchset_datagen.flow_from_dataframe(
                      dataframe=datos['df_batchset'],
                      x_col=x_col_name,
                      y_col=y_col_name,
                      batch_size=pipeline["batch_size"],
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

