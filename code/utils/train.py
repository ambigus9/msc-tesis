def get_model(architecture, iteracion, models_info, pipeline): 

    #print("\n")
    print("="*len(architecture))
    print(architecture)
    print("="*len(architecture))
    #print("\n")

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
            base_model = Xception(weights=pipeline['weights'],include_top=False,input_shape=(pipeline['img_height'], pipeline['img_width'], 3))

    return base_model, preprocess_input