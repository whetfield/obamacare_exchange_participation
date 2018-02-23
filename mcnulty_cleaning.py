#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:20:36 2018

@author: whetfield
Cleaning and loading Dataframes of healthcare data for McNulty
"""
import pandas as pd
import numpy as np

def create_df_plans_offered_for_year (df_servicearea):
    """
    Pass the raw dataframe from ServiceArea.csv and return
    a dataframe with entries by Issuer, Year, County and number
    of plans offered in county in those years
    """
    #Dont want to include dental only plans in analysis
    df_servicearea = df_servicearea[df_servicearea.DentalOnlyPlan != 'Yes']
    
    
    df_plans_offered = df_servicearea.groupby(['BusinessYear','County','StateCode','IssuerId'],
                                              as_index=False)\
                                              .size()\
                                              .reset_index(name='Issuer_Plans_Offered_County')
                                              
    return df_plans_offered



def create_median_rate_dataframe (rate_csv_path):
    """
    Take the string filepath to a rate.csv and return a dataframe with the median
    individual rate for the issuer in state / year.  Excludes any individual rates
    below $10,000 
    """
    df_rates = pd.read_csv(rate_csv_path)
    
    
    #No flags for dental plans in rate workbook, assuming need to be above $50 monthly premium
    #Assume anything above $10,000 per month is outlier
    
    
    df_median_rate = df_rates[(df_rates['IndividualRate'] < 10000) & \
                              (df_rates['IndividualRate'] > 25)].\
    groupby(['BusinessYear','StateCode','IssuerId'], as_index=False)\
    ['IndividualRate'].median()
    
    return df_median_rate



def make_total_plans_in_county_all_issuers(df_plans_offered):
    """
    Make column of total plans offered in the county that year.  Make county
    level metric of Population per number of plans offered in the county.
    
    Maker column of Total Issuers in County.  Make county level metric of number
    of Population per number of Issuers in County
    """
    
    
    df_plans_offered.loc[: ,'Countywide_Plans_Current_Year'] = df_plans_offered.groupby(['County',
                    'BusinessYear'])['Issuer_Plans_Offered_County'].transform(sum)
    
    #15 records have an issue with the population
    df_plans_offered = df_plans_offered.loc[df_plans_offered.County_Population != 'n/a',:]
    
    df_plans_offered.loc[:,'Population_per_All_County_Plans'] = df_plans_offered.County_Population \
                                                        / df_plans_offered.Countywide_Plans_Current_Year
    
    
    df_plans_offered.loc[: ,'Countywide_Issuers_Current_Year'] = df_plans_offered.groupby(['County',
                    'BusinessYear'])['Issuer_Plans_Offered_County'].transform('count')
        
    df_plans_offered.loc[:,'Population_per_All_County_Issuers'] = df_plans_offered.County_Population \
                                                        / df_plans_offered.Countywide_Issuers_Current_Year   
    
    return df_plans_offered




def make_county_pops_names (df_plans_offered, df_fips):
    """
    Take df_plans_offered and df_fips, which has population and county names
    by fips code, create lists for county population and county name that correspond
    to df_plans_offered.  Return df_plans_offered with appropriate columns added.
    """

    population_list = []
    county_name_list = []
    
    for _ , row in df_plans_offered.iterrows():
        df_single_county = df_fips.loc[df_fips.FIPS == row.County,['Population','County']]
        df_single_county.reset_index(inplace = True)
    
        if df_single_county.shape[0] == 0:
            population_list.append('n/a')
            county_name_list.append('n/a')
            continue
    
        population_list.append(int(df_single_county.loc[0,'Population']))
        county_name_list.append(str(df_single_county.loc[0,'County']))
    
    
    df_plans_offered['County_Population'] = pd.Series(population_list)
    df_plans_offered['County_Name'] = pd.Series(county_name_list)
    
    
    
    return df_plans_offered



def plans_offered_prior_year (df_plans_offered, first_year):
    """
    Take dataframe in format of df_plans_offered and return  list
    corresponding to each row that gives number of plans offered prior year for
    issuer in that county.  "n/a" in list if year is first_year in the 
    BusinessYear column of df_plans_offered
    """
    
    plans_offered_prior_year = []
    for _ , row in df_plans_offered.iterrows():
        if row.BusinessYear == first_year:
            plans_offered_prior_year.append("n/a")
        else:
            mask = (df_plans_offered.County == row.County) & \
                   (df_plans_offered.BusinessYear == (row.BusinessYear - 1)) &\
                   (df_plans_offered.IssuerId == row.IssuerId)
                    
            prior_year_offering = df_plans_offered[mask]
            
            #if no rows in shape of next_year offering, then no plans were offered
            if prior_year_offering.shape[0] == 0:
                plans_offered_prior_year.append(0)
                continue
                
            else:
                plans_offered_prior_year.append(int(prior_year_offering.Issuer_Plans_Offered_County))
                
    return plans_offered_prior_year


def plans_offered_next_year (df_plans_offered, last_year):
    """
    Take dataframe in format of df_plans_offered and return  list
    corresponding to each row that gives number of plans offered next year for
    issuer in that county.  "n/a" in list if year is last_year in the 
    BusinessYear column of df_plans_offered
    """
    
    plans_offered_next_year = []
    for _ , row in df_plans_offered.iterrows():
        if row.BusinessYear == last_year:
            plans_offered_next_year.append("n/a")
        else:
            mask = (df_plans_offered.County == row.County) & \
                   (df_plans_offered.BusinessYear == (row.BusinessYear + 1)) &\
                   (df_plans_offered.IssuerId == row.IssuerId)
                    
            next_year_offering = df_plans_offered[mask]
            
            #if no rows in shape of next_year offering, then no plans were offered
            if next_year_offering.shape[0] == 0:
                plans_offered_next_year.append(0)
                continue
                
            else:
                plans_offered_next_year.append(int(next_year_offering.Issuer_Plans_Offered_County))
                
    return plans_offered_next_year


def add_county_percentage_change_columns_to_df(df_plans_offered):
    """
    Takes df_plans_offered with columns for issuer plans in next and current
    years already in place, creates a percentage change column
    """
    
    #Create column for percentage change in issuer plans from prior year to current year
    df_plans_offered['Issuer_%_Change_Plans_From_Prior_Year'] = 0
                                                                
    
    
    mask = (df_plans_offered["Issuer_Plans_Prior_Year_County"] != 'n/a') & \
            (df_plans_offered["Issuer_Plans_Prior_Year_County"] != 0)
    
    #set base values    
    df_plans_offered.loc[mask,'Issuer_%_Change_Plans_From_Prior_Year'] = df_plans_offered.loc[mask,'Issuer_Plans_Offered_County']\
                                                                / df_plans_offered.loc[mask, "Issuer_Plans_Prior_Year_County"] - 1
    
    #first year in the data set, then we don't have prior year data so no growth rate
    df_plans_offered.loc[df_plans_offered["Issuer_Plans_Prior_Year_County"] == 'n/a',
                         'Issuer_%_Change_Plans_From_Prior_Year'] = 'n/a'
    
    #If entered county in current year, ie, zero value prior year, put large value                     
    df_plans_offered.loc[df_plans_offered["Issuer_Plans_Prior_Year_County"] == 0,
                         'Issuer_%_Change_Plans_From_Prior_Year'] = 100000
                         
                         
    return df_plans_offered


def add_plans_offered_and_exit_county_target_columns_to_df(df_plans_offered,
                                                           plans_offered_next_year,
                                                           plans_offered_prior_year):
    
    """
    Take dataframe in format of df_plans_offered, add plans_offered_next_year list
    as a column, and add a target column for if the plan exited the county 
    completely
    """
    
    
    df_plans_offered['Issuer_Plans_Next_Year_County'] = pd.Series(plans_offered_next_year)
    df_plans_offered['Issuer_Plans_Prior_Year_County'] = pd.Series(plans_offered_prior_year)
        
    df_plans_offered['Exit_County_Next_Year'] = 0
    df_plans_offered.loc[df_plans_offered["Issuer_Plans_Next_Year_County"] == 0,
                         'Exit_County_Next_Year'] = 1
    df_plans_offered.loc[df_plans_offered["Issuer_Plans_Next_Year_County"] == 'n/a',
                         'Exit_County_Next_Year'] = 'n/a'
                         
    return df_plans_offered


def add_issuer_median_rate_offered_in_state(df_plans_offered, df_rate):
    """
    Create a list with the same length of the number of rows in the main
    df_plans_offered dataframe.  List has the median rate for the rows issuer
    in the state of the rows county.  df_rate is a dataframe built from rates.csv
    and the create_median_rate_dataframe function in this file.
    
    Insert list into date_frame as new column.  Also creates new column with average
    Median Rate for Entire State over all issuers
    """
    
    median_rate_in_state = []
    for _ , row in df_plans_offered.iterrows():
        
        mask = (df_rate.StateCode == row.StateCode) & \
               (df_rate.BusinessYear == row.BusinessYear) & \
               (df_rate.IssuerId == row.IssuerId)
                    
        median_rate = df_rate[mask]
        median_rate.reset_index(inplace = True)
            
        #if no rows in shape of median_rate, then median_rate for issuer missing
        if median_rate.shape[0] == 0:
            median_rate_in_state.append('n/a')
            continue
                
        else:
            median_rate_in_state.append(int(median_rate.loc[0,'IndividualRate']))
                
    
    df_plans_offered.loc[ : ,'Issuer_Median_Rate_in_State'] = pd.Series(median_rate_in_state)
    
    df_plans_offered = df_plans_offered.loc[df_plans_offered.Issuer_Median_Rate_in_State != 'n/a', :]
    
    df_plans_offered = df_plans_offered.loc[df_plans_offered.Issuer_Median_Rate_in_State.notnull(),:]
    
    df_plans_offered.loc[:,'Average_Median_Rate_in_State'] = df_plans_offered.groupby(['StateCode',
                    'BusinessYear'])['Issuer_Median_Rate_in_State'].transform(lambda x: x.mean())
        
    return df_plans_offered
    
    
    
def median_rate_prior_year (df_plans_offered, first_year):
    """
    Take dataframe in format of df_plans_offered with rate date inserted and return  list
    corresponding to each row that gives .  "n/a" in list if year is first_year in the 
    BusinessYear column of df_plans_offered
    """
    
    median_rate_issuer_prior_year = []
    average_median_state_prior_year = []
    for _ , row in df_plans_offered.iterrows():
        if row.BusinessYear == first_year:
            median_rate_issuer_prior_year.append("first year")
            average_median_state_prior_year.append("first year")
        else:
            mask = (df_plans_offered.County == row.County) & \
                   (df_plans_offered.BusinessYear == (row.BusinessYear - 1)) &\
                   (df_plans_offered.IssuerId == row.IssuerId)
                    
            prior_year_offering = df_plans_offered[mask]
            
            #if no rows in shape of next_year offering, then no plans were offered
            if prior_year_offering.shape[0] == 0:
                median_rate_issuer_prior_year.append("n/a")
                average_median_state_prior_year.append("n/a")
                continue
                
            else:
                median_rate_issuer_prior_year.append(int(prior_year_offering.Issuer_Median_Rate_in_State))
                average_median_state_prior_year.append(int(prior_year_offering.Average_Median_Rate_in_State))
                
    
    df_plans_offered.loc[ : ,'Issuer_Prior_Year_Median_Rate_in_State'] = \
    pd.Series(median_rate_issuer_prior_year)
    
    df_plans_offered.loc[ : ,'State_Prior_Year_Average_Median_Rate'] = \
    pd.Series(average_median_state_prior_year)
    
    
    
    return df_plans_offered 


def median_rate_prior_year2 (df_plans_offered, first_year):
    """
    Take dataframe in format of df_plans_offered with rate date inserted and return  list
    corresponding to each row that gives .  "n/a" in list if year is first_year in the 
    BusinessYear column of df_plans_offered
    """
    
    median_rate_issuer_prior_year = []
    average_median_state_prior_year = []
    
    state_annual_avg = df_plans_offered.groupby(['StateCode', 'BusinessYear'],\
                                                as_index = False )['Issuer_Median_Rate_in_State'].aggregate(lambda x: x.mean())
    
    
    for _ , row in df_plans_offered.iterrows():
        
        
        #if 2014, no proir year
        if row.BusinessYear == first_year:
            median_rate_issuer_prior_year.append("first year")
            average_median_state_prior_year.append("first year")
        
        else:
            
           #first if / else deals with  
            if (row.BusinessYear - 1) in state_annual_avg[state_annual_avg.StateCode == row.StateCode].BusinessYear.values:
                
                mask = (state_annual_avg.StateCode == row.StateCode) & \
                   (state_annual_avg.BusinessYear == (row.BusinessYear - 1))
                
            
                state_value = float(state_annual_avg[mask].Issuer_Median_Rate_in_State)
                
                average_median_state_prior_year.append(state_value)
                
            else:
                average_median_state_prior_year.append("state data not available")
            
            
            mask = (df_plans_offered.County == row.County) & \
                   (df_plans_offered.BusinessYear == (row.BusinessYear - 1)) &\
                   (df_plans_offered.IssuerId == row.IssuerId)
                    
            prior_year_offering = df_plans_offered[mask]
            
            #if no rows in shape of next_year offering, then no plans were offered
            if prior_year_offering.shape[0] == 0:
                median_rate_issuer_prior_year.append("n/a")

                continue
                
            else:
                median_rate_issuer_prior_year.append(int(prior_year_offering.Issuer_Median_Rate_in_State))
                
    
    df_plans_offered.loc[ : ,'Issuer_Prior_Year_Median_Rate_in_State'] = \
    pd.Series(median_rate_issuer_prior_year)
    
    df_plans_offered.loc[ : ,'State_Prior_Year_Average_Median_Rate'] = \
    pd.Series(average_median_state_prior_year)

    
    return df_plans_offered



def add_rate_percentage_change_columns_to_df(df_plans_offered):
    """
    Takes df_plans_offered with columns in place for prior year average_rate in state
    for issuer and and average median rate in prior year to create a percentage change
    from prior year for both the issuer median in the state and the average median
    overall in the state
    """
    
    
    #Create column for percentage change in median_rate from prior year
    #Crate column for percentage change in average median rate for state
    df_plans_offered['Issuer_Percent_Change_Rate_From_Prior_Year'] = 0
    df_plans_offered['State_Percent_Change_Rate_From_Prior_Year'] = 0                                                           
    
    
    #set mask for issuer level 
    
    mask = (df_plans_offered['Issuer_Prior_Year_Median_Rate_in_State'] != 'n/a') & \
            (df_plans_offered['Issuer_Prior_Year_Median_Rate_in_State'] != 'first year')
         
    
    
    #set base values of percentage change at ISsuer Kevek     
    df_plans_offered.loc[mask,"Issuer_Percent_Change_Rate_From_Prior_Year"]\
    = df_plans_offered.loc[mask,'Issuer_Median_Rate_in_State']\
    / df_plans_offered.loc[mask, 'Issuer_Prior_Year_Median_Rate_in_State'] - 1
    
    mask = (df_plans_offered['State_Prior_Year_Average_Median_Rate'] != 'n/a') & \
            (df_plans_offered['State_Prior_Year_Average_Median_Rate'] != "state data not available") & \
            (df_plans_offered['State_Prior_Year_Average_Median_Rate'] != 'first year') 
               
    df_plans_offered.loc[mask,"State_Percent_Change_Rate_From_Prior_Year"]\
    = (df_plans_offered.loc[mask,'Average_Median_Rate_in_State']\
    / df_plans_offered.loc[mask, 'State_Prior_Year_Average_Median_Rate'] - 1) * 100
    
    
    #first year in the data set, then we don't have prior year data so no growth rate
    #2014 will be dropped in the main analysis 
    df_plans_offered.loc[df_plans_offered['Issuer_Prior_Year_Median_Rate_in_State'] == 'first year',
                         'Issuer_Percent_Change_Rate_From_Prior_Year'] = 'first year'
    
    df_plans_offered.loc[df_plans_offered['State_Prior_Year_Average_Median_Rate'] == "state data not available",
                         'State_Percent_Change_Rate_From_Prior_Year'] = "state data not available"
      
    df_plans_offered.loc[df_plans_offered['State_Prior_Year_Average_Median_Rate'] == 'first year',
                         'State_Percent_Change_Rate_From_Prior_Year'] = 'first year'
                         
    #If entered county in current year, ie, put large negative value                    
    df_plans_offered.loc[df_plans_offered['Issuer_Prior_Year_Median_Rate_in_State'] == 'n/a',
                         'Issuer_Percent_Change_Rate_From_Prior_Year'] = -100000
    
    df_plans_offered.loc[df_plans_offered['State_Prior_Year_Average_Median_Rate'] == 'n/a',
                         'State_Percent_Change_Rate_From_Prior_Year'] = -100000
                         
    return df_plans_offered


def add_medicaid_expansion (df_plans_offered):
    """
    Add to main dataframe categorical variable of 1 if Medicaid was expanded in state
    """
    
    #Initialize column
    
    df_plans_offered['Expansion'] = 0
    
    medicaid_expansion_list = ['PA','MI','OH', 'IL', 'IN','OR','IA','NJ','AR',
                           'MT','ND','KY','AZ', 'NM','NV','WV','NH','AK']
    
    df_plans_offered.loc[df_plans_offered['StateCode'].isin(medicaid_expansion_list),
                         "Expansion"] = 1
                         
    return df_plans_offered


def add_first_year_in_county_flag (df_plans_offered):
    """
    Take formatted dataferame and add flag for first year in market
    """
    
    df_plans_offered['First_Year_in_County'] = 0
    df_plans_offered.loc[df_plans_offered['Issuer_%_Change_Plans_From_Prior_Year'] == 100000.000000, 
                    'First_Year_in_County'] = 1
                         
    return df_plans_offered
    
    
    