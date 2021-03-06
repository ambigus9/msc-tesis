def balancear_downsampling(df):
    df.columns=["archivo","clase"]
    g = df.groupby('clase')
    df_=g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
    df_.columns = ['archivos','clases']
    df_ = df_.reset_index().drop(['clase','level_1'],axis=1)
    df_.columns=[x_col_name,y_col_name]
    return df_

def dividir_balanceado2(df,fragmentos):
    X = df.iloc[:,0].values
    y = df.iloc[:,1].values
    kf = StratifiedKFold(n_splits=fragmentos)
    kf.get_n_splits(X)

    fold = []

    #print(kf)

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        fold.append([X_train,X_test,y_train,y_test])
    return fold
