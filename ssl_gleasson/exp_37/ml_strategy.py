"ML Strategy"

#from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, AveragePooling2D, Flatten

def transfer_learning_classic(base_model, num_clases, pipeline):

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    #x = AveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3, seed=pipeline["seed_value"])(x)
    #x = Dense(512, activation='relu', kernel_initializer="glorot_uniform")(x)
    #x = Dropout(0.3, seed=pipeline["seed_value"])(x)

    predictions = Dense(num_clases, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers[:]:
        layer.trainable = True
    return model

def transfer_learning_soft(base_model, num_clases, pipeline):

    if pipeline['layer_percent'] == 1:
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        #x = Dense(512, activation='relu')(x) # INCEPTIONV4
        x = Dense(256, activation='relu')(x) # INCEPTIONV4
        #x = Dense(512, activation='relu')(x) # INCEPTIONV4
        x = Dropout(0.3)(x) # INCEPTIONV4
        #x = Dense(256, activation='relu')(x) # RESNET152
        #x = Dropout(0.5)(x) # RESNET152
        predictions = Dense(num_clases, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers[:]:
            layer.trainable = True

        return model

    if pipeline['layer_percent'] > 0 and pipeline['layer_percent'] < 1:
        split_layer = int( len(base_model.layers[:])*(1-pipeline['layer_percent']) )
        print('estoy usando TL suave! ->',  len(base_model.layers[:]),
                                            split_layer,
                                            len(base_model.layers[:split_layer]))
        for layer in base_model.layers[:split_layer]:
            layer.trainable = False
        for layer in base_model.layers[split_layer:]:
            layer.trainable = True
        return base_model
