import json
import pandas as pd
import os

json_files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.json')]

for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)

    df_data = {}

    for emotion, models in data.items():
        for model, metrics in models.items():
            for metric, value in metrics.items():
                column_name = (model, metric)
                if column_name not in df_data:
                    df_data[column_name] = {}
                df_data[column_name][emotion] = value
    df = pd.DataFrame(df_data)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['Model', 'Metric'])
    csv_file = os.path.splitext(json_file)[0] + '.csv'
    df.to_csv(csv_file)
