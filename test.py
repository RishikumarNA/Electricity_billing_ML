def values(inv1,day1,month1,year1,hour1,min1):
    import sys
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import sklearn
    import seaborn as sns


    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from pandas.plotting import scatter_matrix
    from sklearn.preprocessing import LabelEncoder


    import warnings
    warnings.filterwarnings("ignore")

    pd.set_option('display.max_columns',None)
    pd.set_option('display.max_rows',None)
    pd.options.display.precision=3

    generation_data = pd.read_csv('./Plant_2_Generation_Data.csv')
    weather_data = pd.read_csv('./Plant_2_Weather_Sensor_Data.csv')
    generation_data.sample(5).style.set_properties(
        **{
            'background-color': 'OliveDrab',
            'color': 'white',
            'border-color': 'darkblack'
        })
    weather_data.sample(5).style.set_properties(
        **{
            'background-color': 'pink',
            'color': 'Black',
            'border-color': 'darkblack'
        })

    generation_data['DATE_TIME'] = pd.to_datetime(generation_data['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')
    weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')
    df_solar = pd.merge(generation_data.drop(columns = ['PLANT_ID']), weather_data.drop(columns = ['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')
    #df_solar.sample(5).style.background_gradient(cmap='cool')

    df_solar["DATE"] = pd.to_datetime(df_solar["DATE_TIME"]).dt.date
    df_solar["TIME"] = pd.to_datetime(df_solar["DATE_TIME"]).dt.time
    df_solar['DAY'] = pd.to_datetime(df_solar['DATE_TIME']).dt.day
    df_solar['MONTH'] = pd.to_datetime(df_solar['DATE_TIME']).dt.month
    df_solar['YEAR'] = pd.to_datetime(df_solar['DATE_TIME']).dt.year


    # add hours and minutes for ml models
    df_solar['HOURS'] = pd.to_datetime(df_solar['TIME'],format='%H:%M:%S').dt.hour
    df_solar['MINUTES'] = pd.to_datetime(df_solar['TIME'],format='%H:%M:%S').dt.minute
    df_solar['TOTAL MINUTES PASS'] = df_solar['MINUTES'] + df_solar['HOURS']*60

    # add date as string column
    df_solar["DATE_STRING"] = df_solar["DATE"].astype(str) # add column with date as string
    df_solar["HOURS"] = df_solar["HOURS"].astype(str)
    df_solar["TIME"] = df_solar["TIME"].astype(str)

    #df_solar.tail(1)
    #df_solar.info()
    #df_solar.isnull().sum()

    encoder = LabelEncoder()
    df_solar['SOURCE_KEY_NUMBER'] = encoder.fit_transform(df_solar['SOURCE_KEY'])
    #df_solar.tail(22)

    # Convert 'DATE_TIME' column to datetime dtype
    df_solar['DATE_TIME'] = pd.to_datetime(df_solar['DATE_TIME'])

    # Create a mask for the missing values (0) in DC_POWER and AC_POWER columns
    mask = (df_solar['DC_POWER'] == 0) & (df_solar['AC_POWER'] == 0)

    # Calculate group-wise means for DC_POWER and AC_POWER based on DATE and TIME
    group_means = df_solar[~mask].groupby(['DATE', 'TIME'])[['DC_POWER', 'AC_POWER']].mean()

    # Update the missing values based on the group means using 'fillna' method
    df_solar[['DC_POWER', 'AC_POWER']] = df_solar[['DC_POWER', 'AC_POWER']].mask(mask).fillna(
        df_solar.groupby(['DATE', 'TIME'])[['DC_POWER', 'AC_POWER']].transform('mean')
    )

    #df_solar.head()

    # Read the dataset

    # Convert 'DATE_TIME' column to datetime format
    df_solar['DATE_TIME'] = pd.to_datetime(df_solar['DATE_TIME'])

    # Extract the time component from 'DATE_TIME'
    df_solar['TIME'] = df_solar['DATE_TIME'].dt.time

    # Calculate the average AC_POWER for each unique SOURCE_KEY_NUMBER and time
    df_solar['AVERAGE_AC_POWER'] = df_solar.groupby(['SOURCE_KEY_NUMBER', 'TIME','YEAR'])['AC_POWER'].transform('mean')

    # Print the updated dataset with the average AC_POWER column
    #df_solar.head()

    df_solar['PRICE'] = df_solar.apply(
        lambda row: 0 if row['AC_POWER'] == 0 or row['AVERAGE_AC_POWER'] == 0
        else (row['AVERAGE_AC_POWER'] - row['AC_POWER']),axis=1
    )

    #df_solar.head()

    columns_to_drop = ['DATE_TIME', 'SOURCE_KEY','DATE','TIME','DATE_STRING']
    df_solar.drop(columns_to_drop, axis=1, inplace=True)

    X = df_solar[['SOURCE_KEY_NUMBER','DAY','MONTH','YEAR','HOURS','MINUTES']]
    y = df_solar['PRICE']
    #df_solar.head()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)


    #print(y_train.dtype)
    #print(y_train)

    seed = 8
    scoring = 'accuracy'

    models = []
    #models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
    #models.append(('LR',LinearRegression))
    models.append(('CART', DecisionTreeClassifier()))
    #models.append(('NB', GaussianNB()))


    # Evaluate each model in turn
    results = []
    names = []

    for name, model in models:
        kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        #print(msg)
        
    for name, model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        #print(name)
        #print(accuracy_score(y_test, predictions))
        #print(classification_report(y_test, predictions))
        
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    #print(accuracy)

    example = np.array([[inv1,day1,month1,year1,hour1,min1]])
    example = example.reshape(len(example), -1)
    prediction = clf.predict(example)
    return prediction


