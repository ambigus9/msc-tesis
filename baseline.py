import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
  
  print("X_train: ",X_train)
  
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
  
  print("X_test: ",X_test)
  
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
  L_img = imagenes[:int(len(clases)*porcentaje)]
  L_class = clases[:int(len(clases)*porcentaje)]
  U_img = imagenes[int(len(clases)*porcentaje):]
  print("L_len : ",len(L_img))
  print("U_len : ",len(U_img))
  
  # Dividir train, val y test
  if metodo=='supervisado':
    X_train, X_test, y_train, y_test = train_test_split(L_img, L_class, test_size=0.2, random_state=1)
    #X_train, X_val, X_test = np.split(L_img, [int(.6 * len(L_img)), int(.8 * len(L_img))])
    #y_train, y_val, y_test = np.split(L_class, [int(.6 * len(L_class)), int(.8 * len(L_class))])
    
    print("train_len: ",len(X_train))
    #print("val___len: ",len(X_val))
    print("test__len: ",len(X_test))
    #print("total    : ",len(X_train)+len(X_val)+len(X_test))
    return X_train, X_test, y_train, y_test, U_img
    #return X_train, X_val, X_test, y_train, y_val, y_test, U_img
    
  if metodo=='semi-supervisado': 
    #X_train, X_val, X_test = np.split(L_img, [int(.2 * len(L_img)), int(.6 * len(L_img))])
    #y_train, y_val, y_test = np.split(L_class, [int(.2 * len(L_class)), int(.6 * len(L_class))])
  
    print("train_len: ",len(X_train))
    print("val___len: ",len(X_val))
    print("test__len: ",len(X_test))
    print("total    : ",len(X_train)+len(X_val)+len(X_test))
  
  return X_train, X_test, y_train, y_test, U_img
  
def return_batch_set(U_img,batch_size):
  return np.split(np.array(U_img), int(1/batch_size))