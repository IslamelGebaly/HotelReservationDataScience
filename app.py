"""
    StreamLit Application for applying the classifcation model in order to classify user input data
"""
#Importing Libraries
import streamlit as st
import joblib as jb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
from imblearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, f_regression, RFE
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

#Title Element
st.title("Hotel Reservation Status Predicition")
st.write('Enter sample data for prediction:')

#Getting the Data from the User from UI Elements
sample = {}
#No. of Adults
no_of_adults = st.number_input("Number of adults:", min_value=0, step=1)
sample["no_of_adults"] = [no_of_adults]

#No. of Children
no_of_children = st.number_input('Number of children', min_value=0, step=1)
sample["no_of_children"] = [no_of_children]

#No. of Weekend Nights
no_of_weekend_nights = st.number_input("Number of Weekend Nights", min_value=0, step=1)
sample["no_of_weekend_nights"] = [no_of_weekend_nights]

#No. of Week Nights
no_of_week_nights = st.number_input("Number of Week Nights", min_value=0, step=1)
sample["no_of_week_nights"] = [no_of_week_nights]

#Type of Meal Plan
meal_plan_options = ['Meal Plan 1', 'Not Selected', 'Meal Plan 2', 'Meal Plan 3']
type_of_meal_plan = st.selectbox("Type of Meal Plan", meal_plan_options)
sample["type_of_meal_plan"] = [type_of_meal_plan]

#Required Parking Space
parking_space = [0, 1]
required_parking_space = st.selectbox("Required Parking Space", parking_space)
sample["required_car_parking_space"] = [required_parking_space]

#Days before Arrival
lead_time = st.number_input("Lead Time", min_value=0, step=1)
sample["lead_time"] = [lead_time]

#Market Segment Type
market_segments = ['Offline', 'Online', 'Corporate', 'Aviation', 'Complementary']
market_segment_type = st.selectbox("Market Segment Type", market_segments)
sample["market_segment_type"] = [market_segment_type]

#If the User is a Repeated Guest
repeated_guest_options = [0, 1]
repeated_guest = st.selectbox("Repeated Guest", repeated_guest_options)
sample["repeated_guest"] = [repeated_guest]

#Arrival Date Values
date = st.date_input("Pick a date:")
arrival_day = date.day
arrival_month = date.month
arrival_year = date.year

sample["arrival_year"] = [arrival_year]
sample["arrival_month"] = [arrival_year]
sample["arrival_date"] = [arrival_day]

#Room Type
room_type_options = ['Room_Type 1', 'Room_Type 4', 'Room_Type 2', 'Room_Type 6','Room_Type 5', 'Room_Type 7', 'Room_Type 3']
room_type = st.selectbox("Room_Type", room_type_options)
sample["room_type_reserved"] = [room_type]

#No of Previous Cancellations
no_of_previous_cancellations = st.number_input("Number of Previous Cancellations", min_value=0, step=1)
sample["no_of_previous_cancellations"] = [no_of_previous_cancellations]

#No of bookings not canceled
no_of_previous_bookings_not_canceled = st.number_input("Number of Previous Bookings Not Canceled", min_value=0, step=1)
sample["no_of_previous_bookings_not_canceled"] = [no_of_previous_bookings_not_canceled]

#Average Price per Room
avg_price_per_room = st.number_input("Avg Price per Room", min_value=0, step=1)
sample["avg_price_per_room"] = [avg_price_per_room]

#No of Special Requests
no_of_special_requests = st.number_input("No of Special Requests")
sample["no_of_special_requests"] = [no_of_special_requests]

def make_prediction(pipeline, sample):
    """
        This function's role is to get the user input data, turn them into 
        a dataframe and then return the prediction using the pipeline
    """
    #String form of Target classes
    #0 : Not_Canceled
    #1: Canceled
    classes = ["Not_Canceled", "Canceled"]

    #Transform the User Data into a Dataframe
    user_data = pd.DataFrame.from_dict(sample)
    #Features Created with Feature Engineering Added to the DataFrame
    user_data["no_of_members"] = user_data["no_of_adults"] + user_data["no_of_children"]
    user_data["no_of_nights"] = user_data["no_of_weekend_nights"] + user_data["no_of_week_nights"]

    pred = pipeline.predict(user_data)
    return classes[pred[0]]


def main():
    """
        This is the main function
    """
    #Loading the pipeline from the joblib file
    pipeline = jb.load("pipeline.joblib")

    #If the predict button is pressed then the make_prediction function will be triggered
    #and will return a prediction
    if st.button("Predict"):
        #st.write('Sample Data:', user_input)
        pred = make_prediction(pipeline, sample)
        st.write("Result: ", pred)


if __name__ == '__main__':
    main()