from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
import numpy as np
import os
import pandas as pd

base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1000').output)

def normalizar(datos):
  scaler = StandardScaler()
  return scaler.fit_transform(datos)

def training(X_train,y_train):
  caracteristicas_train = deep_extractor(X_train)
  X_train = normalizar( np.array(caracteristicas_train) )
  # Entrenar SVM
  clf = SVC(probability=True, kernel='linear', C=1.0)
  clf.fit(X_train, y_train)
  return clf

def deep_extractor(imagenes):
  caracteristicas = []
  for i in tqdm(imagenes):
    img = image.load_img( i , target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    fc1000 = model.predict(x)
    caracteristicas.append(fc1000[0])
  return caracteristicas

def labeling(clasificador,batch_set,EL,LC):
  caracteristicas_batch_set = normalizar( np.array( deep_extractor(batch_set) ) )
  y_pred = clasificador.predict(caracteristicas_batch_set)
  y_pred_proba = clasificador.predict_proba(caracteristicas_batch_set)
  return y_pred, y_pred_proba

def test(clasificador,X_test,y_test):
  caracteristicas_test = deep_extractor(X_test)
  X_test = normalizar( np.array(caracteristicas_test) )
  print("Score: ",clasificador.score(X_test,y_test))
  pass

def split_dataset(L,metodo):
  clases = list()
  imagenes = list()
  for root, dirs, files in os.walk(os.path.abspath("/content/gdrive/My Drive/msc-miguel/datasets/"+L+"/Images")) :
      for file in files:
          if file.endswith(".tif"):
            clases.append(str(os.path.basename(os.path.normpath(root))))
            imagenes.append(os.path.join(root, file))

  # Dividir train, val y test
  if metodo=='supervisado':
    X_train, X_test, y_train, y_test = train_test_split(imagenes, clases, test_size=0.2, random_state=1)
    X_train, X_val, y_train , y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    
    print("train: ",len(X_train))
    print("  val: ",len(X_val))
    print(" test: ",len(X_test))
    print("total: ",len(X_train)+len(X_val)+len(X_test))

    return X_train, X_val, X_test, y_train, y_val, y_test
    
  if metodo=='semi-supervisado':
    L_img, U_img, L_class, U_class = train_test_split(imagenes, clases, test_size=0.5, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(L_img, L_class, test_size=0.4, random_state=1)
    X_train, X_val, y_train , y_val = train_test_split(X_train, y_train, test_size=0.66, random_state=1)

    U_img_ = U_img[:int(len(U_img)*porcentaje)]
    U_class_ = U_class[:int(len(U_class)*porcentaje)]

    print('\n')
    print("train: ",len(X_train))
    print("  val: ",len(X_val))
    print(" test: ",len(X_test))
    print("U_img: ",len(U_img_))
    print("total etiquetado: ",len(X_train)+len(X_val)+len(X_test))
    print("total no-etiquetado: ",len(U_img))
    print("total global: ",len(X_train)+len(X_val)+len(X_test)+len(U_img))
  
    return X_train, X_val, X_test, y_train, y_val, y_test, U_img_, U_class_
  
def return_batch_set(U_img,batch_size):
  return np.split(np.array(U_img), int(1/batch_size))

def update_training(X_train,y_train,EL):
  [X_train.append(i[0]) for i in EL]
  [y_train.append(i[1]) for i in EL]
  caracteristicas_train = deep_extractor(X_train)
  X_train = normalizar( np.array(caracteristicas_train) )
  # Entrenar SVM
  clf = SVC(probability=True, kernel='linear', C=1.0)
  clf.fit(X_train, y_train)
  return clf

EL = list()
LC = list()
count = 0

L = 'ucmerced'
batch_size = 0.2
porcentaje = 0.1

X_train, X_val, X_test, y_train, y_val, y_test, U_img, U_class = split_dataset(L,'semi-supervisado')

clasificador = training(X_train,y_train)
print('\n')
test(X_test,y_test)

batch_set = return_batch_set(U_img,batch_size)[count]
prediccion,probabilidad = labeling(clasificador,batch_set,EL,LC)

mejores,todos = [],[]

# agregar a funcion labeling
for i in range(len(probabilidad)):
  todos.append([max(probabilidad[i]),prediccion[i]])
  if max(probabilidad[i]) > 0.4:
    mejores.append([max(probabilidad[i]),prediccion[i],batch_set[i]])
    EL.append((batch_set[i],prediccion[i]))
  else:
    LC.append(batch_set[i])

df = pd.DataFrame(todos)
df.describe()

clasificador = training(X_train,y_train)
print('\n')
test(X_test,y_test)


while batch_size*count < 0.2:
  count += 1
  clasificador = training(X_train,y_train,X_test,y_test)
  batch_set = return_batch_set(U_img,batch_size)[count]
  print(count,len(batch_set))