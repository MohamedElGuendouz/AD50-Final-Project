#%%
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn import metrics


#Read weather csv and return synthetic dataframe
def readWeather():
    #Read csv and convert into dataframe
    filename = 'Data/Arpajon_weather.csv'
    df = pd.read_csv(filename)
    #Convert the column date_time in date format
    df['date_time']= pd.to_datetime(df['date_time'])
    #Create and get year and week
    df['year'] = df['date_time'].dt.isocalendar().year
    df['week'] = df['date_time'].dt.isocalendar().week
    #Create dataframe with groupby function and average on rows
    df = df.groupby(['year','week'], as_index=False).mean()
    return df

#Read diarrhee csv and return synthetic dataframe
def readDiarrhee():
    #Read csv and convert into dataframe
    filename = 'Data/diarrhee_aigues.csv'
    df = pd.read_csv(filename)
    #Select only Ile de France
    df = df[df['geo_insee']==11]
    #Create and get year and week from string
    df['week'] = df['week'].astype(str)
    df['year'] = df['week'].str[0:4].astype(int)
    df['week'] = df['week'].str[4:6].astype(int)
    #Get only week, year and inc from dataframe (the other parameters and derivated from inc)
    df = df[['week','year','inc']]
    #Rename column inc in order to merge it later
    df = df.rename(columns={"inc": "inc_diarrhee"})
    return df


def readGrippes():
    filename = 'Data/symptomes_grippaux.csv'
    df = pd.read_csv(filename)
    df = df[df['geo_insee']==11]
    df['week'] = df['week'].astype(str)
    df['year'] = df['week'].str[0:4].astype(int)
    df['week'] = df['week'].str[4:6].astype(int)
    df = df[['week','year','inc']]
    df = df.rename(columns={"inc": "inc_grippe"})
    return df

def readVaricelle():
    filename = 'Data/cas_varicelles.csv'
    df = pd.read_csv(filename)
    df = df[df['geo_insee']==11]
    df['week'] = df['week'].astype(str)
    df['year'] = df['week'].str[0:4].astype(int)
    df['week'] = df['week'].str[4:6].astype(int)
    df = df[['week','year','inc']]
    df = df.rename(columns={"inc": "inc_varicelle"})
    return df

#Read interventions csv and return a synthetic dataframe
def readInterventions():
    #Read csv and convert into dataframe
    filename = 'Data/interventions-hebdo-2010-2017.csv'
    df = pd.read_csv(filename, sep=";")
    #Remove columns not usefull like type of intervention and name of city
    df = df.drop(columns=['ope_code_insee','ope_categorie','ope_code_postal','ope_nom_commune'], axis=1)
    #Group and count number of operation by week
    df = df.groupby(['ope_annee','ope_semaine'], as_index=False).sum()
    #Rename columns in order to be consistent with other files
    df = df.rename(columns={"ope_annee": "year", "ope_semaine": "week"})
    return df

#Read vacances csv and return a synthetic dataframe
def readVacances():
    filename = 'Data/vacances.csv'
    df = pd.read_csv(filename)
    #Remove zone a, zone b and name 
    df = df.drop(columns=['vacances_zone_a','vacances_zone_b','nom_vacances'], axis=1)
    return df

#Read feries csv and return a synthetic dataframe
def readFeries():
    filename = 'Data/jours_feries_metropole.csv'
    df = pd.read_csv(filename)
    #Remove annee, zone and name
    df = df.drop(columns=['annee','zone','nom_jour_ferie'], axis=1)
    #Create a new column ferie which equals True always because all rows displayed are feries
    df['ferie'] = True
    return df

#Read vacances and feries to create a subdataframe
def readVacancesFeries():
    df1 = readVacances()
    df2 = readFeries()
    #Create new dataframe with merged vacances and feries
    df = df1.merge(df2, how='outer')
    #Replace nan values by False
    df['ferie'].fillna(False, inplace=True)
    #Convert date column to date format
    df['date'] = pd.to_datetime(df['date'])
    #Create and get week and year column
    df['week'] = df['date'].dt.week
    df['year'] = df['date'].dt.year
    #Remove the date column
    df = df.drop(columns=['date'], axis=1)
    #Group by year and week and count values (= sum becasue true=1 and false=0)
    df = df.groupby(['year','week'], as_index=False).sum()
    return df

#Main function that merge all dataframes on year and week
def mergeAll(dfIntervention, dfVaricelle, dfWeather, dfDiarrhee, dfGrippe, dfVacFerie):
    df = pd.merge(dfVacFerie, dfVaricelle, how='left', on=['year', 'week'])
    df = pd.merge(df, dfDiarrhee, how='left', on=['year', 'week'])
    df = pd.merge(df, dfGrippe, how='left', on=['year', 'week'])
    df = pd.merge(df, dfWeather, how='left', on=['year', 'week'])
    df = pd.merge(df, dfIntervention, how='left', on=['year', 'week'])
    #Remove week 53 because it is not complete or empty
    df = df[df.week != 53]
    return df


dfI = readInterventions()
dfVF = readVacancesFeries()
dfV = readVaricelle()
dfG = readGrippes()
dfD = readDiarrhee()
dfW = readWeather()

df = mergeAll(dfI, dfV, dfW, dfD, dfG, dfVF)
#Convert the dataframe into csv file
df.to_csv('Data/data_merged.csv', index=False)  

