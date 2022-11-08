import time
import random
import pandas as pd
import numpy as np
from random import randrange
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

dataset_path = './days_dataset_with_level_new.csv'
data_for_drop = ['Level', 'shortwave_radiation (W/m²)', 'direct_radiation (W/m²)', 'diffuse_radiation (W/m²)',
                 'direct_normal_irradiance (W/m²)', 'soil_temperature_7_to_28cm (°C)',
                 'soil_temperature_28_to_100cm (°C)', 'soil_temperature_100_to_255cm (°C)',
                 'soil_moisture_100_to_255cm (m³/m³)', 'soil_moisture_28_to_100cm (m³/m³)']
train_number = 1000
interval = randrange(3, 18)
model = RandomForestRegressor(random_state=1)
model.set_params(n_estimators=175)

df = pd.read_csv(dataset_path)
df = df.fillna(0)
y = df.Level
X = df.drop(data_for_drop, axis=1)
features = np.array(X.columns)
features_list = list(features)
frame_array = []
code_running_timer = time.time()


def score(rand_features):
    X = df.drop(data_for_drop, axis=1)
    X = X.drop(rand_features, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.75)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    pred_mae = mean_absolute_error(pred, y_test)
    selected_features = "'" + "', '".join(X.columns) + "'"
    length = len(X.columns)
    return [score, selected_features, length, pred_mae]


for x in range(train_number):
    rand_features = random.sample(features_list, interval)
    scores_array = score(rand_features)
    scores = [scores_array[3], scores_array[0], "[" + scores_array[1] + "]", scores_array[2]]
    frame_array.append(scores)

frame = pd.DataFrame(data=frame_array, columns=['mae', 'score', 'features', 'count_of_features'])
frame.to_csv('find_best_combination_of_features.csv', index=False)
print("--- %s seconds ---" % (time.time() - code_running_timer))