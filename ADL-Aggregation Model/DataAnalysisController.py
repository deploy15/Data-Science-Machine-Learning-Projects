# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 08:46:43 2017

@author: adeshina

The aim of this file is to serves as a point where all the classes will be called and the integration interface to another system
"""
import GetMonitoredData as gmd  # Import the class to get all the data from the repository
#import SleepingActivitiesDataExtraction as _sleeping_ade # Import all the function in SleepingActivitiesDataExtraction
#import MovementActivitiesDataExtraction as made # Import all the functions in MovementActivitiesDataExtraction
#import SleepingActivity as sa # Import all the function in SleepingActivity.py
import ADLMovementPlaceActivity as adl_mvt_place # function that retun ADL
import ADLSummary as adl_activity_summary # This function is majorly to retun the activity summary
import MLPredictionsModels as ml # Class where ML training & Testing is being done
from pytz import timezone
from datetime import datetime, timedelta,date, time
import time
import pandas as pd
import scipy.stats as ss
from flask import Flask
import theano
import tensorflow
import keras
import json
import datetime
import math


def get_dataset(house_id,start_time,end_time,table_name,db_name,tz_from,tz_to,tz_api_key,local_tz):
    
    #activate this when testing for production release
    get_monitored_data = gmd.GetMonitoredData(house_id,start_time,end_time,table_name,db_name,tz_from,tz_to,tz_api_key,timezone(local_tz))
    #Create the object of the class GetMonitoredData
    
   
    #Class initialization values
    gui_id = get_monitored_data.gw_euid
    db_name = get_monitored_data.db_name
    table_name = get_monitored_data.table_name
    start_datetime = get_monitored_data.start_time
    end_datetime = get_monitored_data.stop_time
    from_tz = get_monitored_data.from_tz
    to_tz = get_monitored_data.to_tz
    api_key = get_monitored_data.tz_account_api_key
    local_tz = get_monitored_data.local_tz
    
    # building SQL Query
    db_name = str(db_name)
    #print (db_name)
    SQL_Query = "select * from "+ db_name + "."+ str(table_name) + " where gw_euid=" + "'"+ str(gui_id) + "'"+ " and ts<=%s and ts>=%s allow filtering" 
    #print ("SQL QUERY - >",SQL_Query)
    #Get the data out of the database
    data_df = get_monitored_data.connect_to_db(db_name,SQL_Query,end_datetime,start_datetime)
    
    
    #print (data_df.head())
    
    #Transpose the data to the local timeon
    #data_df.ts =pd.DataFrame( gmd.time_rearrangement(data_df,"Africa/Accra","America/Vancouver","T5EIC058WQ3L",local_tz))
    
    
    #The correct Timezone Calling
    #print(type(data_df))
    #print ("Before TimeZone Data Frame",data_df.head(10))
    #data_df.ts =pd.DataFrame( gmd.time_rearrangement(data_df,from_tz,to_tz,api_key,local_tz)) # correct timezone converter
    #print ("After TimeZone  Data Frame",data_df.head(10))
    
    
    #process time to date time format
    '''
    data_df.ts = pd.data_df(data_df.ts)# convert from string to datetime
    data_df.index = data_df.ts # turn ts  column into index
    #print (dataset)
    todays_dataset = data_df[start_datetime]  #extract data for a specific date; easy because is now a time series
    '''
    return data_df # This return dataset


# THis function is basically to save us from typing all the parameters needed to access the dataset in our repository
#All parameter can be changes in get_dataset and we will just call get_dataset_quickly to get the data out
#This return the raw dataset
def get_dataset_quickly(gw_uid,start_date,end_date,table_to_select,database_to_use,from_tz,to_tz,api_key,local_tz):
    
    #data_df = get_dataset("000D6F000C362FB5", "2017-06-01 00:00:00.000+0000", "2017-06-30 23:59:59.000+0000", "user_mvts","recare_adl","Africa/Accra","America/Vancouver", "T5EIC058WQ3L", "US/Pacific")
    #print("Database To USE",database_to_use)
    data_df = get_dataset(gw_uid,start_date,end_date,table_to_select,database_to_use,from_tz,to_tz,api_key,"US/Pacific")
    
       
    return data_df #This return the raw dataset
    



    

#get the sleeping pattern
def get_sleep_timing():
    # This is te list of returned value
    #listSleepTime,listOffBedTimeAndMostPlaceAfterOffBed,list_of_time_wakeup_inbetween_sleep,wakeup_inbetween_sleep_counter,longest_time_awake_between_sleep 
    return _sleeping_ade.Q1_Q2(get_dataset_quickly())
        
    
    
#get 
    
#get the sleepdurations and timing after predictions
def get_sleep_computation():
    inbed_df_new = _sleeping_ade.get_in_bed_data(get_dataset_quickly())
    offbed_df_new = _sleeping_ade.get_off_bed_data(get_dataset_quickly())
    sleepDurationParameters = _sleeping_ade.Q3(inbed_df_new,offbed_df_new) 
    sleepAggregation = _sleeping_ade.get_sleep_duration_aggregation(sleepDurationParameters)
    SD = _sleeping_ade.clean_sleep_duration(sleepAggregation)
    ST = _sleeping_ade.sleep_time_extraction(sleepDurationParameters)
    
    # return value are scale_comp_predict_duration, mode_res_duration,scale_comp_predict_timing, mode_res_timing 
    return _sleeping_ade.sleep_scale_computation(ST,SD,"") 

# This extract the date that an activity will be extracted for
def get_startdate_to_extract_activity(start_date):
    split_date = start_date.split(" ")
    return split_date[0]
    

# This extract the date that an activity will be extracted for
def get_enddate_to_extract_activity(end_date):
    split_date = end_date.split(" ")[0]
    return  int(split_date.split("-")[2])
    
#This return the list of mvt activity of concern
def get_list_of_mvt_activity():
    mvt_sensor_id = [1,2,5,6,18,19,20]
    return mvt_sensor_id
    

#This return the list of place activity of concern
def get_list_of_mvt_activity():
    place_sensor_id = [4,5,29,46]
    return mvt_sensor_id

#Get the next day date
def get_next_day_date():
    nextday_date =  datetime.date.today() + datetime.timedelta(days=1)
    #print tomorrow's date in YYYY-MM-DD format
    formatted_result =  datetime.datetime.strftime(nextday_date,'%Y-%m-%d') 
    return formatted_result

#Get the next day from current date
#The current date parameter is a string
def get_next_day_date_based_on_current_date(current_date):
    split_date = current_date.split("-")
    
    new_date = split_date[0] + "-"+split_date[1]+ "-"+ str(int(split_date[2])+1)
    return new_date

#Get the Daily activity for movement and place
#current_date = adl_mvt_place.get_todays_date() # is working
#current_date = '2017-06-025' 
#activity_id = 1 # This is the sensor ID
#activity_type = 'mvt' # Activity Type can either be 'mvt' or 'place'
#This will get specific activity
def get_adl(current_date,activity_id,activity_type,from_tz,to_tz,api_key,local_tz):
    print("Data Extraction and Conversion to local Timezone")
    dataset =  adl_mvt_place.get_dataset_based_on_date(current_date,from_tz,to_tz,api_key,local_tz)
    print ("Daily Computation Based on Activity ID")
    daily_activity = adl_mvt_place.get_daily_in_total_minute(dataset,current_date, activity_id,activity_type) # This activity only receive movement sensor id
    print (daily_activity)
    



#This function will get the entire daily activity dataset
def get_adl_dataset(todays_date,from_tz,to_tz,api_key,local_tz):
    return adl_mvt_place.get_dataset_based_on_date(todays_date,from_tz,to_tz,api_key,local_tz)
    



#Testing ADL Computation
#"000D6F000C362FB5", "2017-06-01 00:00:00.000+0000", "2017-06-30 23:59:59.000+0000", "user_mvts","recare_adl","Africa/Accra","America/Vancouver", "T5EIC058WQ3L", "US/Pacific")
#("000D6F000C362FB5", "2017-06-01 00:00:00.000+0000", "2017-06-30 23:59:59.000+0000", "user_mvts","recare_adl","Africa/Accra","America/Vancouver", "T5EIC058WQ3L", "US/Pacific")
dic_activity = {}
gw_uid = "000D6F000C362FB5"
start_date ="2017-06-01 00:00:00.000+0000"
end_date = "2017-06-30 23:59:59.000+0000"
table_to_select = "user_mvts"
database_to_use = "recare_adl"
from_tz = "Africa/Accra"
to_tz = "America/Vancouver"
api_key = "T5EIC058WQ3L"
local_tz = timezone("US/Pacific")




#assumptions
sleep_assumption_time_night = 20  #This is the assumed hour when the night time starts (Start Sleeping for the current day)
sleep_assumption_time_morning = 10  #This is the assumed hour when the morning time stop (Stop sleeping for the next day)
sleep_assumption_duration = 7  #This is not being used now but is the assumed sleeping duration
#daily activity parameter
current_date = '2017-06-02' 
nextday__date = '2017-06-13'
activity_id = 5 # This is the sensor ID
activity_type = 'place' # Activity Type can either be 'mvt' or 'place'

#fetch data
fetch_data , status,dataset_size = get_dataset_quickly(gw_uid,start_date,end_date,table_to_select,database_to_use,from_tz,to_tz,api_key,local_tz)    # just to test




if(status == True):
    print("Connection to DB Successful with return dataset size of "+ str(dataset_size))
    #Timezone Parameter
    
    #current_month = '2017-06-'
    #end_day = get_enddate_to_extract_activity(end_date)
    
    #print (get_adl(current_date,activity_id,activity_type,from_tz,to_tz,api_key,local_tz))

else:
    print ("Unsuccessful Connection")# We can write a script to send mail to the admin on this failure
#fetch_data = fetch_data.head(20)
#data_df = get_dataset_quickly()
#print(data_df)

print (get_startdate_to_extract_activity(start_date))
print(get_enddate_to_extract_activity(end_date))
print ("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
#print (dic_activity)
print ("&&&&&&&&&&&&&&&&&&&&TRAINING AND FITING LSTM &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
print (fetch_data.head())
print(type(fetch_data))










#print(get_sleep_timing())

def convert_to_timeseries(data_df,start_datetime):
    todays_dataset = pd.DataFrame() #creates a new dataframe that's empty
    #data_df.ts = pd.data_df(data_df.ts)# convert from string to datetime
    data_df.index = data_df.ts # turn ts  column into index
    print (data_df)
    print ("Panda Version",pd.__version__)
    todays_dataset = data_df[start_datetime]  #extract data for a specific date; easy because is now a time series
    return todays_dataset

    

print ("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
#print (fetch_data.head(10))
#print (type(fetch_data))
##activity_id = 6 # This is the sensor ID
##print (convert_to_timeseries(fetch_data,current_date))




#Testing the daily activity summary
print ("**************************** PROCESSING CURRENT DATE ADL ****************************************************")
adl_dataset = get_adl_dataset(current_date,from_tz,to_tz,api_key,local_tz) #dataset of today

print ("**************************** PROCESSING NEXT DATE ADL ****************************************************")
adl_dataset_nextday = get_adl_dataset(nextday__date,from_tz,to_tz,api_key,local_tz) #dataset of nextday
#print (type(adl_dataset), len(adl_dataset.index), len(adl_dataset_nextday.index))
print ("Next day:", get_next_day_date())
#print (adl_dataset)
#print (adl_dataset_nextday)








#test_data = get_dataset_quickly().head(10)    # just to test
#print (fetch_data)
print ("*************************** *SLEEPING ACTIVITY SUMMARY ****************************************************")
print (adl_activity_summary.daily_sleeping_summary(adl_dataset,adl_dataset_nextday,sleep_assumption_time_night,sleep_assumption_duration,sleep_assumption_time_morning))
print("")






print ("*************************** *MOVEMENT ACTIVITY SUMMARY ****************************************************")
mvt_hourly = adl_activity_summary.hourly_daily_movement_activity(adl_dataset)
print(mvt_hourly)






















#print(fetch_data)
#when when threshold is is less or equal to the activity, it might be the person is just passing by



#listSleepTime,listOffBedTimeAndMostPlaceAfterOffBed,list_of_time_wakeup_inbetween_sleep,wakeup_inbetween_sleep_counter = get_sleep_timing()

'''

print ("****************Just Testing the dynamic sleep duration *********************************")
SD_X = [4,5,6,3,3,7,2,7,6,2,2,7,10]
print(_sleeping_ade.dynamic_sleep_duration(SD_X))


print (get_sleep_computation())



washroom_threshold_time = 30 # The washroom time
print ("Wash Room Time is",washroom_threshold_time)

print  ("################################################################################")
print (fetch_data)
print (made.washroom_movement_exception(fetch_data,washroom_threshold_time))



def f(x):
    return ss.norm.cdf(x,loc = 18, scale=0.09)
xx = ss.norm(loc = 18, scale=0.09)
print (xx)


print (f(17.8))


#print (ade.get_data_whem_in_bedroom(data_df))

#print (data_df)
#a,b,c,d,e = ade.Q1_Q2(data_df)
#print (a,d,b,d,e)


print (listSleepTime)
print (listOffBedTimeAndMostPlaceAfterOffBed)
print (list_of_time_wakeup_inbetween_sleep)
print (wakeup_inbetween_sleep_counter)












print (get_sleep_computation())
'''


################################## TESTING  ADL Using By Getting Several Daily Activity #########################
current_date = '2017-06' 
nextday__date = '2017-06-13'
activity_id = 5 # This is the sensor ID
activity_type = 'mvt' # Activity Type can either be 'mvt' or 'place'


#******************************** TESTING SLEEPING PREDICTION BY COLLECTING SEVERAL DAILY ACTIVITY *******************************************


# Data Cleaning section
#This function returns the dataframe object of extracted daily activity
def get_dataframe_of_daily_activity(daily_activity_date,daily_activity_list):
    # convert the arrays to panda dataframe object
    df_sleep_duration = pd.DataFrame({'sleep_duration':daily_activity_list})
    df_sleep_date =  pd.DataFrame({'date':daily_activity_date})
    df_new = pd.concat([df_sleep_date,df_sleep_duration], axis=1)
    return df_new
    
    
#print (get_adl(current_date,activity_id,activity_type,from_tz,to_tz,api_key,local_tz))

daily_sleep_duration_list = [] # This store the daily sleep duration
daily_date = [] # This stores the daily date
for i in range(1,10):
    current_date_new = current_date + "-"+str(i) # Get the current date
    next_date = get_next_day_date_based_on_current_date(current_date_new) # get the next day from the current date
    
    adl_dataset = get_adl_dataset(current_date_new,from_tz,to_tz,api_key,local_tz) #dataset for current date
    adl_dataset_nextday = get_adl_dataset(next_date,from_tz,to_tz,api_key,local_tz) #dataset of nextday
    #print ("Current Date:",current_date_new, "Next Date:", get_next_day_date_based_on_current_date(current_date_new))
    result = adl_activity_summary.only_daily_sleeping_activity (adl_dataset,adl_dataset_nextday,sleep_assumption_time_night,sleep_assumption_duration,sleep_assumption_time_morning)
    
    daily_result = json.loads(result) # Load the json returned records
    appended_result_sleep_duration =  str(math.ceil(daily_result["sleeping-duration-at-night"]))
    
    daily_sleep_duration_list.append(appended_result_sleep_duration) 
    daily_date.append(current_date_new)
    
    
    print (result)
    print (daily_result["sleeping-duration-at-night"])


print (daily_sleep_duration_list)
print (daily_date)







# ML Prediction Section
#ml_obj = ml.MLPredictiveModel()
new_df = get_dataframe_of_daily_activity(daily_date,daily_sleep_duration_list) # Cleaned Dataframe object for the sleep activity
print (new_df)
#print (ml_obj.model_activity_prediction(new_df) )



print (ml.sensor_id_activity_prediction(fetch_data,activity_type))

print (ml.each_adl_activity_prediction(new_df,"sleep_duration")) # This is for sleep duration prediction



# Testing Anomalies
#adl_data_train = [9,6,5,4,6,5,4,6,5,4,3,20,50]
#adl_data_test = [10]

adl_data_train = daily_sleep_duration_list
adl_data_test = daily_sleep_duration_list[len(daily_sleep_duration_list)-1]
result = ml.is_anomaly(adl_data_train,adl_data_test) 

print("Anomaly Result is->", result)         
    







#print(_sleeping_ade.Q1_Q2(fetch_data))

#print(made.outside_movement(fetch_data,0)) # geth outside mocement activities






'''
# THis is a working Code

maximum_movement_threshold = 300
livingroom_result = made.living_room_movement(fetch_data,maximum_movement_threshold) #get living room movement activities
#print (livingroom_result)
kitchen_result = made.kitchen_movement(fetch_data,maximum_movement_threshold) # get kitchen movement activities
#print (kitchen_result)
washroom_result = made.washroom_movement(fetch_data,maximum_movement_threshold) # get washroom movement activities
#print (washroom_result)
outside_result = made.outside_movement(fetch_data,maximum_movement_threshold) # geth outside mocement activities
#print (outside_result)



app = Flask(__name__)

@app.route("/kitchen")
def GetKitchenActivity():
    return kitchen_result

@app.route("/washroom")
def GetWashroomActivity():
    
    return washroom_result
    
@app.route("/livingroom")
def LivingRoomActivity():
    return livingroom_result


@app.route("/outside")
def GetOutsideActivity():
    return outside_result
    
    
@app.route("/sleeping")
def GetSleepingActivity():
    return "Hello, Sleeping!"
    
@app.route("/sedentary")
def GetSedentaryActivity():
    return "Hello, Sendary!"
    

@app.route("/sleepcomputation")
def GetSleepingActivities():
    return get_sleep_computation()
    
@app.route('/hello', methods=['POST'])
def hello():
    return ("Good is faithful")
    #return("Hello {}!".format(username))
    
    

@app.route("/")
def Welcome():
    return "Welcome to JSON RESTful Service for Responsive Home Monitoring!"
    
#@app.route('/users/<string:username>')

    

if __name__ == '__main__':
    
    app.run(port=8098, debug=True)
    
'''