#%%
import pandas as pd

def readVaricelle():
    filename = 'Data/cas_varicelles.csv'
    df = pd.read_csv(filename)
    #Select only Ile de France
    df = df[df['geo_insee']==11]
    df['week'] = df['week'].astype(str)
    print(type(df['week']))
    df['year'] = df['week'].str[0:4]
    df['week'] = df['week'].str[4:6]
    return df

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
    df['year'] = df['week'].str[0:4]
    df['week'] = df['week'].str[4:6]
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
    df['date']= pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.isocalendar().year
    df['week'] = df['date'].dt.isocalendar().week
    return df

def readFeries():
    filename = 'Data/jours_feries_metropole.csv'
    df = pd.read_csv(filename)
    df = df.drop(columns=['annee','zone','nom_jour_ferie'], axis=1)
    df['ferie'] = True
    df['date']= pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.isocalendar().year
    df['week'] = df['date'].dt.isocalendar().week
    return df

def readGrippes():
    filename = 'Data/symptomes_grippaux.csv'
    df = pd.read_csv(filename)
    #Select only Ile de France
    df = df[df['geo_insee']==11]
    # make string version of original column, call it 'col'
    df['week'] = df['week'].astype(str)
    df['year'] = df['week'].str[0:4]
    df['week'] = df['week'].str[4:6]
    return df

def readVacancesFeries():
    df1 = readVacances()
    df2 = readFeries()
    #Create new dataframe with merged vacances and feries
    df = df1.merge(df2, how='outer')
    df['ferie'].fillna(False, inplace=True)
    return df

print(readVacancesFeries())
print(readVaricelle())
"""
print(readFeries())
print(readDiarrhee())
print(readWeather())
print(readInterventions())
#print(readVacances())
print(readGrippes())
#print(readFeries())"""

# %%
