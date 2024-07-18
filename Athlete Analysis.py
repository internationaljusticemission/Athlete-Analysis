#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 08:55:45 2023

@author: adampayne
"""

# Athlete Data SOQL
# SELECT Donor_Advised_Fund__c,Donor_Advised_Fund__r.Name,Owner_IJM_Team__c,Owner.Name,Donor_Confirmed_DAF_Ownership__c,Family_Foundation__c,Family_Foundation__r.Name,Id,Name,npe01__One2OneContact__r.FirstName,npe01__One2OneContact__r.LastName,npe01__LifetimeDonationHistory_Amount__c,npe01__LifetimeDonationHistory_Number__c,npo02__OppAmount2YearsAgo__c,npo02__OppAmountLastYear__c,npo02__OppAmountThisYear__c,Total_Gifts_Three_Years_Ago__c,Total_Gifts_Four_Years_Ago__c,Total_Gifts_Five_Years_Ago__c,Family_Foundation__r.npo02__OppAmount2YearsAgo__c,Family_Foundation__r.npo02__OppAmountLastYear__c,Family_Foundation__r.npo02__OppAmountThisYear__c,Family_Foundation__r.Total_Gifts_Three_years_Ago__c,Family_Foundation__r.Total_Gifts_Four_Years_Ago__c,Family_Foundation__r.Total_Gifts_Five_Years_Ago__c,Program__r.Name,CreatedDate,Family_Foundation__r.npo02__FirstCloseDate__c,npo02__FirstCloseDate__c FROM Account WHERE Program__r.Name LIKE '%Athlete%' ORDER BY Family_Foundation__c ASC NULLS LAST

# Athlete Tasks SOQL
# SELECT AccountId,Account__c,Action_Subtype__c,Action_Type__c,Activity_Type__c,Subject,Assigned_IJM_Team__c,CreatedDate,Date_Completed__c,Id,Type,WhoId, Account__r.Program__r.Name,ActivityDate,CompletedDateTime FROM Task WHERE Account__r.Program__r.Name LIKE '%Athlete%'

# Athlete Donations SOQL
# SELECT AccountId,Amount,CloseDate,Id,Program__r.Name FROM Opportunity WHERE Program__r.Name LIKE '%Athlete%' AND StageName = 'Closed Won' AND RecordTypeId IN ('0121J000001DaT4QAK', '012o0000001AMWGAA4') AND IsParentTransaction__c = False AND IRS_Credit_Amount_FF__c > 0 AND CurrencyISOCode = 'USD'


import pandas as pd
from pathlib2 import Path

home = str(Path.home())


athlete_data = pd.read_csv('{}/Desktop/{}'.format(home, 'Athlete Data.csv'))

athlete_tasks = pd.read_csv('{}/Desktop/{}'.format(home, 'Athlete Tasks.csv'))

athlete_donor_search = pd.read_csv('{}/Desktop/{}'.format(home, 'Athlete Screening Data.csv'))

athlete_contracts = pd.read_excel('{}/Desktop/{}'.format(home, 'athlete_contracts.xlsx'))


def league_correction(row):
    if 'nfl' in row['league']:
        return 'Athlete - Football'
    elif 'mlb' in row['league']:
        return 'Athlete - Baseball'
    elif 'nba' in row['league']:
        return 'Athlete - Basketball'
    elif 'nhl' in row['league']:
        return 'Athlete - Hockey'
    else:
        return row['league']

athlete_contracts['league'] = athlete_contracts.apply(league_correction, axis=1)



def free_agent_year(row):
     return row['contract_years'].split('FA: ')[1][:-1]


athlete_contracts['Free Agency Year'] = athlete_contracts.apply(free_agent_year, axis = 1).astype(int)


athlete_tasks['Date_Completed__c'] = pd.to_datetime(athlete_tasks['Date_Completed__c'])
athlete_tasks['ActivityDate'] = pd.to_datetime(athlete_tasks['ActivityDate'])
athlete_tasks['CompletedDateTime'] = pd.to_datetime(athlete_tasks['CompletedDateTime'])
athlete_tasks['CompletedDateTime'] = pd.to_datetime(athlete_tasks['CompletedDateTime'].dt.date)


athlete_tasks['Activity Date'] = athlete_tasks[['Date_Completed__c','ActivityDate','CompletedDateTime']].min(axis=1, skipna=True)

athlete_tasks = athlete_tasks.sort_values('Activity Date', ascending=True)



athlete_data['Donations This Year'] = athlete_data['Family_Foundation__r.npo02__OppAmountThisYear__c'] + athlete_data['npo02__OppAmountThisYear__c']
    
athlete_data['Donations Last Year'] = athlete_data['Family_Foundation__r.npo02__OppAmountLastYear__c'] + athlete_data['npo02__OppAmountLastYear__c']

athlete_data['Donations 2 Years Ago'] = athlete_data['Family_Foundation__r.npo02__OppAmount2YearsAgo__c'] + athlete_data['npo02__OppAmount2YearsAgo__c']

athlete_data['Donations 3 Years Ago'] = athlete_data['Family_Foundation__r.Total_Gifts_Three_Years_Ago__c'] + athlete_data['Total_Gifts_Three_Years_Ago__c']

athlete_data['Donations 4 Years Ago'] = athlete_data['Family_Foundation__r.Total_Gifts_Four_Years_Ago__c'] + athlete_data['Total_Gifts_Four_Years_Ago__c']

athlete_data['Donations 5 Years Ago'] = athlete_data['Family_Foundation__r.Total_Gifts_Five_Years_Ago__c'] + athlete_data['Total_Gifts_Five_Years_Ago__c']


def donated(row):
    if pd.isnull(row['npo02__FirstCloseDate__c']) == False:
        return 1
    else:
        return 0

def daf_flag(row):
    if pd.isnull(row['Donor_Advised_Fund__c'])==False:
        return 1
    else:
        return 0
    
def family_foundation_flag(row):
    if pd.isnull(row['Family_Foundation__c'])==False:
        return 1
    else:
        return 0
    
athlete_data['Has Donated'] = athlete_data.apply(donated, axis = 1)
athlete_data['Has DAF'] = athlete_data.apply(daf_flag,axis=1)
athlete_data['Has Family Foundation'] = athlete_data.apply(family_foundation_flag,axis=1)


athlete_data['npo02__FirstCloseDate__c'] = pd.to_datetime(athlete_data['npo02__FirstCloseDate__c'])
athlete_data['Family_Foundation__r.npo02__FirstCloseDate__c'] = pd.to_datetime(athlete_data['Family_Foundation__r.npo02__FirstCloseDate__c'])
athlete_data['First Gift Date'] = athlete_data[['npo02__FirstCloseDate__c','Family_Foundation__r.npo02__FirstCloseDate__c']].min(axis=1, skipna=True)


    
athlete_tasks = athlete_tasks[
    (athlete_tasks['Action_Subtype__c'] == 'Personal Communication') |
    (athlete_tasks['Action_Subtype__c'] == 'Personal Event Invitation') |
    (athlete_tasks['Activity_Type__c'] == 'Face-to-Face Meeting') |
    (athlete_tasks['Activity_Type__c'] == 'Meeting') |
    (athlete_tasks['Activity_Type__c'] == 'Phone Call') |
    (athlete_tasks['Activity_Type__c'] == 'Phone Call - Inbound') |
    (athlete_tasks['Activity_Type__c'] == 'Phone Call - Outbound') |
    athlete_tasks['Action_Type__c'].str.contains('SP: ')
]

athlete_tasks = athlete_tasks.groupby('AccountId').size().reset_index(name='Number of Interactions')

athlete_data['Athlete Name'] = athlete_data['npe01__One2OneContact__r.FirstName'] + ' ' + athlete_data['npe01__One2OneContact__r.LastName'] 

athlete_data = pd.merge(athlete_data, athlete_contracts, how='left', left_on=['Athlete Name','Program__r.Name'], right_on = ['name','league'])

athlete_data = athlete_data[['Id',
 'Name',
 'Owner_IJM_Team__c',
 'Owner.Name',
 'Program__r.Name',
 'Has Donated',
 'First Gift Date',
 'Donations This Year',
 'Donations Last Year',
 'Donations 2 Years Ago',
 'Donations 3 Years Ago',
 'Donations 4 Years Ago',
 'Donations 5 Years Ago',
 'Has DAF',
 'Has Family Foundation',
 'Athlete Name',
 'name',
 'position',
 'contract_years',
 'Free Agency Year',
 'age',
 'contract_length',
 'total_amount',
 'yearly_amount',
 'signing_bonus',
 'league']] 
    
athlete_donor_search = athlete_donor_search[['DS Rating',
                                             'Quality Score',
                                             'Client ID',
                                             'SP-First',
                                             'SP-Middle',
                                             'SP-Last',
                                             'User1',
                                             'Total Of Likely Matches',
                                             '# Of Gift Matches',
                                             'Foundation',
                                             'NonProfit',
                                             'Wealth-Based Capacity',
                                             'Classic Quality Score',
                                             'Religion Count',
                                             'Society Benefit Count']]
    
athlete_data = pd.merge(athlete_data, athlete_donor_search, how = 'left', left_on = 'Id', right_on = 'Client ID')

athlete_data = pd.merge(athlete_data, athlete_tasks, how = 'left', left_on = 'Id', right_on = 'AccountId').sort_values(by='Quality Score', ascending=False)

athlete_data.drop_duplicates(subset='Id', keep='first', inplace=True)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = athlete_data[pd.isnull(athlete_data['Classic Quality Score']) == False]
data = data[[ 'Has DAF',
 'Has Family Foundation',
 'Has Donated',
 'Number of Interactions',
 'Free Agency Year',
 'age',
 'contract_length',
 'signing_bonus',
 'league',
 'Program__r.Name',
 'Owner.Name',
 'Name',
 'Owner_IJM_Team__c',
 'Classic Quality Score',
 'Id',
 'Foundation',
 'NonProfit',
 'Religion Count',
 'Society Benefit Count']]

data['league'] = data['league'].fillna('Athlete')


def foundation_flag(row):
    if row['Foundation']=='Yes':
        return 1
    elif row['Foundation']=='Maybe':
        return 0.5
    else:
        return 0

def nonprofit_flag(row):
    if row['NonProfit']=='Yes':
        return 1
    elif row['NonProfit']=='Maybe':
        return 0.5
    else:
        return 0


data['Foundation'] = data.apply(foundation_flag, axis = 1)
data['NonProfit'] = data.apply(nonprofit_flag, axis = 1)

# One-hot encode the 'category' column
one_hot_encoded = pd.get_dummies(data['league'])

# Concatenate the one-hot encoded columns with the original dataframe
data = pd.concat([data, one_hot_encoded], axis=1)
data.drop(columns={'age'}, inplace = True)
data = data.fillna(0)


# Split the data into training and testing sets
X = data.drop(['Has Donated','Program__r.Name','Name', 'Owner.Name','Owner_IJM_Team__c','league','Athlete','Id','Free Agency Year', 'contract_length', 'signing_bonus', 'Has DAF','Has Family Foundation'], axis=1)
y = data['Has Donated']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)

# Train the lookalike model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")


# Use the model to predict lookalikes
lookalike_data = data[data['Has Donated']==0]
lookalike_data.drop(columns=['Has Donated','Program__r.Name','Name', 'Owner.Name','Owner_IJM_Team__c','Id','Athlete','league','Free Agency Year', 'contract_length', 'signing_bonus','Has DAF','Has Family Foundation'], inplace = True)
lookalike_predictions = model.predict(lookalike_data)
lookalike_predictions = pd.DataFrame(lookalike_predictions)

data = data[data['Has Donated']==0].reset_index()

data = pd.concat([data, lookalike_predictions], axis=1)

data.drop(columns={'index'}, inplace=True)

data.rename(columns={0:'Likely To Give'}, inplace=True)

data.to_csv('{}/Desktop/{}'.format(home,'Top Priority Athletes.csv'))












    
    
    
    