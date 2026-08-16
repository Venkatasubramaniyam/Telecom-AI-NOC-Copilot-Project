import pandas as pd

def load_alarm_data():
    return pd.read_csv("data/alarms.csv")

def load_incident_data():
    return pd.read_csv("data/incidents.csv")