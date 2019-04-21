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

base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1000').output)


def normalizar(datos):
  scaler = StandardScaler()
  return scaler.fit_transform(datos)

def training(X_train,y_train,X_test,y_test):
  # Extraer Deep Features X_train
  caracteristicas_train = []
  for i in tqdm(X_train):
    img = image.load_img( i , target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    fc1000 = model.predict(x)
    caracteristicas_train.append(fc1000[0])
  
  X_train = normalizar( np.array(caracteristicas_train) )
    
  # Extraer Deep Features X_test
  caracteristicas_test = []
  for i in tqdm(X_test):
    img = image.load_img( i , target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    fc1000 = model.predict(x)
    caracteristicas_test.append(fc1000[0])
  
  X_test = normalizar( np.array(caracteristicas_test) )
    
  # Entrenar SVM
  clf = SVC(probability=True, kernel='linear', C=1.0)
  clf.fit(X_train, y_train)
  
  y_pred = clf.predict(X_train)
  print("Puntaje: ",clf.score(X_test,y_test))
  return clf

def split_dataset(L,porcentaje,metodo):
  clases = list()
  imagenes = list()
  for root, dirs, files in os.walk(os.path.abspath("/content/gdrive/My Drive/msc-miguel/datasets/"+L+"/Images")) :
      for file in files:
          if file.endswith(".tif"):
            clases.append(str(os.path.basename(os.path.normpath(root))))
            imagenes.append(os.path.join(root, file))

  # Dividir L y U
  L_img = imagenes[:int(len(clases)*0.5)]
  L_class = clases[:int(len(clases)*0.5)]
  U_img = imagenes[int(len(clases)*0.5):int(len(clases)*(0.5+porcentaje))]
  print("L_len : ",len(L_img))
  print("U_len : ",len(U_img))
  
  # Dividir train, val y test
  if metodo=='supervisado':
    X_train, X_test, y_train, y_test = train_test_split(L_img, L_class, test_size=0.2, random_state=1)
    X_train, X_val, y_train , y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    
    print("train: ",len(X_train))
    print("  val: ",len(X_val))
    print(" test: ",len(X_test))
    print("total: ",len(X_train)+len(X_val)+len(X_test))
    #return X_train, X_test, y_train, y_test, U_img
    return X_train, X_val, X_test, y_train, y_val, y_test, U_img
    
  if metodo=='semi-supervisado': 
    X_train, X_test, y_train, y_test = train_test_split(L_img, L_class, test_size=0.4, random_state=1)
    X_train, X_val, y_train , y_val = train_test_split(X_train, y_train, test_size=0.66, random_state=1)
  
    print("train: ",len(X_train))
    print("  val: ",len(X_val))
    print(" test: ",len(X_test))
    print("total: ",len(X_train)+len(X_val)+len(X_test))
  
  return X_train, X_test, y_train, y_test, U_img
  
def return_batch_set(U_img,batch_size):
  return np.split(np.array(U_img), int(1/batch_size))


EL = list()
LC = list()
count = 0

L = 'ucmerced'
metodo = 'semi-supervisado'
batch_size = 0.2
porcentaje = 0.1

#X_train, X_val, X_test, y_train, y_val, y_test, U_img = split_dataset(L,porcentaje,metodo)
X_train, X_test, y_train, y_test, U_img = split_dataset(L,porcentaje,metodo)

while batch_size*count < 0.2:
  clasificador = training(X_train,y_train,X_test,y_test)
  print(clasificador)
  batch_set = return_batch_set(U_img,batch_size)[count]
  count += 1
  print(count,len(batch_set))