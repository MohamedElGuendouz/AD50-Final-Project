#%%
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn import metrics



def readWeather():
    filename = 'Data/Arpajon_weather.csv'
    df = pd.read_csv(filename)
    df['date_time']= pd.to_datetime(df['date_time'])
    df['year'] = df['date_time'].dt.isocalendar().year
    df['week'] = df['date_time'].dt.isocalendar().week
    return df

def readDiarrhee():
    filename = 'Data/diarrhee_aigues.csv'
    df = pd.read_csv(filename)
    #Select only Ile de France
    df = df[df['geo_insee']==11]
    df['week'] = df['week'].astype(str)
    df['year'] = df['week'].str[0:4].astype(int)
    df['week'] = df['week'].str[4:6].astype(int)
    df = df[['week','year','inc']]
    df = df.rename(columns={"inc": "inc_diarrhee"})
    return df

def readGrippes():
    filename = 'Data/symptomes_grippaux.csv'
    df = pd.read_csv(filename)
    #Select only Ile de France
    df = df[df['geo_insee']==11]
    # make string version of original column, call it 'col'
    df['week'] = df['week'].astype(str)
    df['year'] = df['week'].str[0:4].astype(int)
    df['week'] = df['week'].str[4:6].astype(int)
    df = df[['week','year','inc']]
    df = df.rename(columns={"inc": "inc_grippe"})
    return df

def readVaricelle():
    filename = 'Data/cas_varicelles.csv'
    df = pd.read_csv(filename)
    #Select only Ile de France
    df = df[df['geo_insee']==11]
    df['week'] = df['week'].astype(str)
    print(type(df['week']))
    df['year'] = df['week'].str[0:4].astype(int)
    df['week'] = df['week'].str[4:6].astype(int)
    df = df[['week','year','inc']]
    df = df.rename(columns={"inc": "inc_varicelle"})
    return df

def readInterventions():
    filename = 'Data/interventions-hebdo-2010-2017.csv'
    df = pd.read_csv(filename, sep=";")
    # remove info on type of intervention and city
    df = df.drop(columns=['ope_code_insee','ope_categorie','ope_code_postal','ope_nom_commune'], axis=1)
    # group and count number of operation by week
    df = df.groupby(['ope_annee','ope_semaine'], as_index=False).sum()
    df = df.rename(columns={"ope_annee": "year", "ope_semaine": "week"})
    return df

def readVacances():
    filename = 'Data/vacances.csv'
    df = pd.read_csv(filename)
    #remove zone a, zone c and nom
    df = df.drop(columns=['vacances_zone_a','vacances_zone_b','nom_vacances'], axis=1)
    return df

def readFeries():
    filename = 'Data/jours_feries_metropole.csv'
    df = pd.read_csv(filename)
    df = df.drop(columns=['annee','zone','nom_jour_ferie'], axis=1)
    df['ferie'] = True
    return df


def readVacancesFeries():
    df1 = readVacances()
    df2 = readFeries()
    #Create new dataframe with merged vacances and feries
    df = df1.merge(df2, how='outer')
    df['ferie'].fillna(False, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.week
    df['year'] = df['date'].dt.year
    df = df.drop(columns=['date'], axis=1)
    df = df.groupby(['year','week'], as_index=False).sum()
    return df

def mergeAll(dfIntervention, dfVaricelle, dfWeather, dfDiarrhee, dfGrippe, dfVacFerie):
    df = pd.merge(dfIntervention, dfVaricelle, how='left', on=['year', 'week'])
    df = pd.merge(df, dfWeather, how='left', on=['year', 'week'])
    df = pd.merge(df, dfDiarrhee, how='left', on=['year', 'week'])
    df = pd.merge(df, dfGrippe, how='left', on=['year', 'week'])
    df = pd.merge(df, dfVacFerie, how='left', on=['year', 'week'])
    return df

def mergeWithoutWeather(dfIntervention, dfVaricelle, dfDiarrhee, dfGrippe, dfVacFerie):
    df = pd.merge(dfVacFerie, dfVaricelle, how='left', on=['year', 'week'])
    df = pd.merge(df, dfDiarrhee, how='left', on=['year', 'week'])
    df = pd.merge(df, dfGrippe, how='left', on=['year', 'week'])
    df = pd.merge(df, dfIntervention, how='left', on=['year', 'week'])
    df = df[df.week != 53]
    return df

dfI = readInterventions()
dfVF = readVacancesFeries()
dfV = readVaricelle()
dfG = readGrippes()
dfD = readDiarrhee()

dfW = readWeather()

#Faux car weather n'a pas de colonne year and week mais une ligne par jour
#df = mergeAll(dfI, dfV, dfW, dfD, dfG, dfVF)
#df.to_csv('Data/data_merged.csv', index=False,compression=compression_opts)  

df2 = mergeWithoutWeather(dfI,dfV,dfD,dfG,dfVF)
df2.to_csv('Data/data_merged_without_weather.csv', index=False)

#LINEAR REGRESSION
features = ['year','week','vacances_zone_c','ferie','inc_varicelle','inc_diarrhee','inc_grippe']
X=df2.loc[:,features].values
y=df2.loc[:,['nb_ope']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(y_test, predictions)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
# %%
