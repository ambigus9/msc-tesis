from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from keras.preprocessing import image
from keras.models import Model
import numpy as np
import os
import pandas as pd
import subprocess

red='resnet50'
#red='vgg16'

if red=='resnet50':
  from keras.applications.resnet50 import ResNet50
  from keras.applications.resnet50 import preprocess_input
  base_model = ResNet50(weights='imagenet')
  model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
  
if red=='vgg16':
  from keras.applications.vgg16 import VGG16
  from keras.applications.vgg16 import preprocess_input
  base_model = VGG16(weights='imagenet')
  model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

def training(X_train,y_train):
  caracteristicas_train = deep_extractor(X_train)
  X_train = np.array(caracteristicas_train)
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
    fully_connected = model.predict(x)
    caracteristicas.append(fully_connected[0])
  return caracteristicas

def labeling(clasificador,batch_set,EL,LC):
  caracteristicas_batch_set = np.array( deep_extractor(batch_set) )
  y_pred = clasificador.predict(caracteristicas_batch_set)
  y_pred_proba = clasificador.predict_proba(caracteristicas_batch_set)
  for i in range(len(y_pred_proba)):
    if max(y_pred_proba[i]) >= 0.6:
      EL.append((batch_set[i],y_pred[i]))
    else:
      LC.append(batch_set[i])
  pass

def test(X_test,y_test):
  caracteristicas_test = deep_extractor(X_test)
  X_test = np.array(caracteristicas_test)
  return X_test,y_test

def test_score(clasificador,X_test,y_test):
  return clasificador.score(X_test,y_test)

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
    X_train, X_test, y_train, y_test = train_test_split(imagenes, clases, test_size=0.2)
    X_train, X_val, y_train , y_val = train_test_split(X_train, y_train, test_size=0.25)
    
    print("train: ",len(X_train))
    print("  val: ",len(X_val))
    print(" test: ",len(X_test))
    print("total: ",len(X_train)+len(X_val)+len(X_test))

    return X_train, X_val, X_test, y_train, y_val, y_test
    
  if metodo=='semi-supervisado':
    L_img, U_img, L_class, U_class = train_test_split(imagenes, clases, test_size=0.5)
    X_train, X_test, y_train, y_test = train_test_split(L_img, L_class, test_size=0.4)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.6666666)

    U_img = U_img[:int(len(U_img)*porcentaje)]
    U_class_ = U_class[:int(len(U_class)*porcentaje)]

    print('\n')
    print("train: ",len(X_train))
    print("  val: ",len(X_val))
    print(" test: ",len(X_test))
    print("U_img: ",len(U_img))
    print("total etiquetado: ",len(X_train)+len(X_val)+len(X_test))
    print("total no-etiquetado: ",len(U_img))
    print("total global: ",len(X_train)+len(X_val)+len(X_test)+len(U_img))
  
    return X_train, X_val, X_test, y_train, y_val, y_test, U_img, U_class
  
def return_batch_set(U_img,batch_size):
  return np.split(np.array(U_img), int(1/batch_size))

def update_training(X_train,y_train,EL):
  [X_train.append(i[0]) for i in EL]
  [y_train.append(i[1]) for i in EL]
  caracteristicas_train = deep_extractor(X_train)
  X_train = np.array(caracteristicas_train)
  # Entrenar SVM
  clf = SVC(probability=True, kernel='linear', C=1.0)
  clf.fit(X_train, y_train)
  return clf

metodo='semi-supervisado'
L = 'ucmerced'
porcentaje=1

# Supervised
X_train, X_val, X_test, y_train, y_val, y_test, U_img_, U_class_ = split_dataset(L,metodo)
clasificador = training(X_train,y_train)
X_test, y_test = test(X_test,y_test)
print('\n')
print('Supervised_score_'+red+': ',test_score(clasificador,X_test,y_test))

# Self-training
metodo='semi-supervisado'
L = 'ucmerced'
batch_size = 0.1
porcentaje = 0.2

def self_training(L,batch_size,porcentaje):
  EL,LC,iteraciones=[],[],[]
  count = 0

  X_train, X_val, X_test, y_train, y_val, y_test, U_img, U_class = split_dataset(L,metodo)

  clasificador = training(X_train,y_train)
  batch_set = return_batch_set(U_img,batch_size)[count]
  labeling(clasificador,batch_set,EL,LC)
  X_test, y_test = test(X_test,y_test)
  print(test_score(clasificador,X_test,y_test))

  while (batch_size+0.1)*count < 0.8:
    count += 1
    clasificador = update_training(X_train,y_train,EL)
    batch_set = return_batch_set(U_img,batch_size)[count]
    labeling(clasificador,batch_set,EL,LC)
    print(test_score(clasificador,X_test,y_test))

  while len(EL)/8 <  len(LC):
    clasificador = update_training(X_train,y_train,EL)
    print("EL: ",len(EL), "LC: ",len(LC))
    len_EL = len(EL)
    batch_set = LC
    LC = []
    labeling(clasificador,batch_set,EL,LC)
    print(test_score(clasificador,X_test,y_test))
    if len(EL) == len_EL:
      iteraciones.append(len_EL)
    else:
      iteraciones = []
    if len(iteraciones) == 5 and np.mean(iteraciones)==len_EL:
      break

self_training(L,batch_size,porcentaje)