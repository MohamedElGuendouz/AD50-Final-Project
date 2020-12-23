#%%
import pandas as pd

def readVaricelle():
    filename = 'Data/cas_varicelles.csv'
    df = pd.read_csv(filename)
    #Select only Ile de France
    df = df[df['geo_insee']==11]
    return df

def readWeather():
    filename = 'Data/Arpajon_weather.csv'
    df = pd.read_csv(filename)
    return df

def readDiarrhee():
    filename = 'Data/diarrhee_aigues.csv'
    df = pd.read_csv(filename)
    #Select only Ile de France
    df = df[df['geo_insee']==11]
    return df

def readInterventions():
    filename = 'Data/interventions-hebdo-2010-2017.csv'
    df = pd.read_csv(filename, sep=";")
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

def readGrippes():
    filename = 'Data/symptomes_grippaux.csv'
    df = pd.read_csv(filename)
    #Select only Ile de France
    df = df[df['geo_insee']==11]
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
print(readVacances())
print(readGrippes())
print(readFeries())"""

# %%
