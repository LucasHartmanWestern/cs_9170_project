import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Constants --- #

RAW_HEADERS = [
    "Timestamp",
    "Activity ID",
    "Heart Rate",

    "Hand Sensor - Temperature",
    "Hand Sensor - Accelerometer - X",
    "Hand Sensor - Accelerometer - Y",
    "Hand Sensor - Accelerometer - Z",
    "Hand Sensor - Accelerometer 2 - X (Remove)",
    "Hand Sensor - Accelerometer 2 - Y (Remove)",
    "Hand Sensor - Accelerometer 2 - Z (Remove)",
    "Hand Sensor - Gyroscope - X",
    "Hand Sensor - Gyroscope - Y",
    "Hand Sensor - Gyroscope - Z",
    "Hand Sensor - Magnetometer - X",
    "Hand Sensor - Magnetometer - Y",
    "Hand Sensor - Magnetometer - Z",
    "Hand Sensor - Orientation 1 (Remove)",
    "Hand Sensor - Orientation 2 (Remove)",
    "Hand Sensor - Orientation 3 (Remove)",
    "Hand Sensor - Orientation 4 (Remove)",

    "Chest Sensor - Temperature",
    "Chest Sensor - Accelerometer - X",
    "Chest Sensor - Accelerometer - Y",
    "Chest Sensor - Accelerometer - Z",
    "Chest Sensor - Accelerometer 2 - X (Remove)",
    "Chest Sensor - Accelerometer 2 - Y (Remove)",
    "Chest Sensor - Accelerometer 2 - Z (Remove)",
    "Chest Sensor - Gyroscope - X",
    "Chest Sensor - Gyroscope - Y",
    "Chest Sensor - Gyroscope - Z",
    "Chest Sensor - Magnetometer - X",
    "Chest Sensor - Magnetometer - Y",
    "Chest Sensor - Magnetometer - Z",
    "Chest Sensor - Orientation 1 (Remove)",
    "Chest Sensor - Orientation 2 (Remove)",
    "Chest Sensor - Orientation 3 (Remove)",
    "Chest Sensor - Orientation 4 (Remove)",

    "Ankle Sensor - Temperature",
    "Ankle Sensor - Accelerometer - X",
    "Ankle Sensor - Accelerometer - Y",
    "Ankle Sensor - Accelerometer - Z",
    "Ankle Sensor - Accelerometer 2 - X (Remove)",
    "Ankle Sensor - Accelerometer 2 - Y (Remove)",
    "Ankle Sensor - Accelerometer 2 - Z (Remove)",
    "Ankle Sensor - Gyroscope - X",
    "Ankle Sensor - Gyroscope - Y",
    "Ankle Sensor - Gyroscope - Z",
    "Ankle Sensor - Magnetometer - X",
    "Ankle Sensor - Magnetometer - Y",
    "Ankle Sensor - Magnetometer - Z",
    "Ankle Sensor - Orientation 1 (Remove)",
    "Ankle Sensor - Orientation 2 (Remove)",
    "Ankle Sensor - Orientation 3 (Remove)",
    "Ankle Sensor - Orientation 4 (Remove)",
]
SUBJECT_INFO_HEADERS = [
    "Sex",
    "Age",
    "Height",
    "Weight",
    "Resting HR",
    "Max HR",
    "Dominant Hand",
]
SUBJECT_INFO_DICT = {
    101: {"Sex": "Male", "Age": 27,"Height": 182,"Weight": 83,"Resting HR": 75, "Max HR": 193, "Dominant Hand": "Right"},
    102: {"Sex": "Female", "Age": 25,"Height": 169,"Weight": 78,"Resting HR": 74, "Max HR": 195, "Dominant Hand": "Right"},
    103: {"Sex": "Male", "Age": 31,"Height": 187,"Weight": 92,"Resting HR": 68, "Max HR": 189, "Dominant Hand": "Right"},
    104: {"Sex": "Male", "Age": 24,"Height": 194,"Weight": 95,"Resting HR": 58, "Max HR": 196, "Dominant Hand": "Right"},
    105: {"Sex": "Male", "Age": 26,"Height": 180,"Weight": 73,"Resting HR": 70, "Max HR": 194, "Dominant Hand": "Right"},
    106: {"Sex": "Male", "Age": 26,"Height": 183,"Weight": 69,"Resting HR": 60, "Max HR": 194, "Dominant Hand": "Right"},
    107: {"Sex": "Male", "Age": 23,"Height": 173,"Weight": 86,"Resting HR": 60, "Max HR": 197, "Dominant Hand": "Right"},
    108: {"Sex": "Male", "Age": 32,"Height": 179,"Weight": 87,"Resting HR": 66, "Max HR": 188, "Dominant Hand": "Left"},
    109: {"Sex": "Male", "Age": 31,"Height": 168,"Weight": 65,"Resting HR": 54, "Max HR": 189, "Dominant Hand": "Right"},
}
VALID_ACTIVITIES = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 24]
VALID_SUBJECTS = [101, 102, 103, 104, 105, 106, 107, 109]

# --- Information Getter Methods --- #

def get_activity(activity_id):
    if activity_id == 0:
        return "Other (transient activities)"
    elif activity_id == 1:
        return "Lying"
    elif activity_id == 2:
        return "Sitting"
    elif activity_id == 3:
        return "Standing"
    elif activity_id == 4:
        return "Walking"
    elif activity_id == 5:
        return "Running"
    elif activity_id == 6:
        return "Cycling"
    elif activity_id == 7:
        return "Nordic Walking"
    elif activity_id == 9:
        return "Watching TV"
    elif activity_id == 10:
        return "Computer Work"
    elif activity_id == 11:
        return "Car Driving"
    elif activity_id == 12:
        return "Ascending Stairs"
    elif activity_id == 13:
        return "Descending Stairs"
    elif activity_id == 16:
        return "Vaccum Cleaning"
    elif activity_id == 17:
        return "Ironing"
    elif activity_id == 18:
        return "Folding Laundry"
    elif activity_id == 19:
        return "House Cleaning"
    elif activity_id == 20:
        return "Playing Soccer"
    elif activity_id == 24:
        return "Rope Jumping"
    else:
        print(f"Invalid activity ID: {activity_id}")
        return ""

def get_column_units(column_name):
    if "Timestamp" in column_name:
        return "s"
    elif "Heart Rate" in column_name:
        return "bpm"
    elif "Temperature" in column_name:
        return "C"
    elif "Accelerometer" in column_name:
        return "m/s^2"
    elif "Gyroscope" in column_name:
        return "rad/s"
    elif "Magnetometer" in column_name:
        return "µT"
    elif "Age" in column_name:
        return "years"
    elif "Height" in column_name:
        return "cm"
    elif "Weight" in column_name:
        return "kg"
    elif "HR" in column_name:
        return "bpm"
    else:
        print(f"Invalid column name: {column_name}")
        return ""

# --- Preprocessing Methods --- #

def preprocess_subject(subject_id):
    # Load the raw data file
    df = pd.read_csv(os.path.join("data", "protocol", f"subject{subject_id}.dat"), sep=" ", names=RAW_HEADERS)

    # Remove invalid data columns
    for col in df.columns:
        if "Remove" in col:
            df = df.drop(col, axis=1)

    # Remove rows with transient activities
    df = df[df["Activity ID"] != 0]

    # Remove rows with missing Heart Rate data (this will reduce timestep frequency) 
    df = df[df["Heart Rate"].notna()]

    # Remove all NaN values
    df.dropna(inplace=True)

    # Convert timestamps into sequential integers for each activity
    for activity_id in VALID_ACTIVITIES:
        temp_df = df[df["Activity ID"] == activity_id].copy()
        if temp_df.shape[0] > 0:
            temp_df.reset_index(drop=True, inplace=True)
            df.loc[df["Activity ID"] == activity_id, "Timestamp"] = temp_df.index
    df["Timestamp"] = df["Timestamp"].astype(int)

    # Add subject information to the dataframe
    for subj_info_header in SUBJECT_INFO_HEADERS:
        df[subj_info_header] = SUBJECT_INFO_DICT[subject_id][subj_info_header]

    # Add subject ID to the dataframe
    df["Subject ID"] = subject_id

    return df

def preprocess_all_data():
    '''
    Preprocess all data from all subjects and save to a single CSV file.
    Rows with NA heart rate data are removed.
    This effectively reduces the timestep frequency to match the frequency of the heart rate monitor.
    All rows with transient activities (activity ID = 0) are also removed.
    All rows with left-handed subjects are removed.
    All rows with NaN values are removed.
    The Timestamp column is converted into sequential integers.
    '''

    # Preprocess all subjects
    df = pd.DataFrame()
    for subject_id in tqdm(range(101, 110), desc="Preprocessing subjects"):
        if subject_id == 108:
            continue # skip subject 108 due to left-dominant hand
        df = pd.concat([df, preprocess_subject(subject_id)])

    # Remove all rows with left-handed subjects so that all rows are right-handed
    df = df[df["Dominant Hand"] == "Right"]
    df = df.drop(columns=["Dominant Hand"])

    # Ensure that each activity only has 1000 sequential rows
    df = df[df["Activity ID"] != 5] # has minimum row count of 318
    df = df[df["Activity ID"] != 12] # has minimum row count of 945
    df = df[df["Activity ID"] != 24] # has minimum row count of 24
    df = df[df["Timestamp"] < 1000]

     # One-hot-encode column for Sex
    df = pd.get_dummies(df, columns=["Sex"], prefix_sep=" - ", dtype=np.int8)

    # Save to CSV
    print(f"Number of columns: {df.shape[1]}, Number of rows: {df.shape[0]}")
    print("Saving to CSV...")
    df.to_csv(os.path.join("data", "formatted_data.csv"), index=False)

    print("preprocess_all_data() completed.")

# --- Load Data Methods --- #

def load_preprocessed_dataset(verbose=True, drop_subject_id=False):
    '''
    Load the preprocessed dataset from the CSV file.
    The Timestamp column is at index 0 and the Activity ID column (target feature) is at index 1.
    '''

    df = pd.read_csv(os.path.join("data", "formatted_data.csv"))

    if drop_subject_id:
        df = df.drop(columns=["Subject ID"])

    if verbose:
        print("---- Preprocessed Dataset Info ----")
        print(f"Number of rows: {df.shape[0]}")
        print(f"Number of columns: {df.shape[1]}")
        print("-----------------------------------")

    return df

def get_xy_from_data(df, target_features: list[str]):
    '''
    Get the X and y data from the dataframe.
    The X data is all columns except the target features.
    The y data is the target features.
    '''
    y = df[target_features]
    x = df.drop(columns=target_features)
    return x, y

# --- Main method for DEBUGGING --- #

if __name__ == "__main__":

    # ----- Preprocess data to create formatted csv file ----- #
    start_time = time.time()
    preprocess_all_data()
    end_time = time.time()
    print(f"Completed. Time taken: {end_time - start_time} seconds")

