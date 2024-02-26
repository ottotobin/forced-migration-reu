"""
Authors: Lina Laghzaoui, Grace Magny-Fokam, Toby Otto, Rich Pihlstrom

Description:
    This is a small program that prints out the features tested in the parameter
    tuning process in order of importance for each emotion in the selected model.

Model: the name of the model that you want to run feature selection for

USAGE:
    python3 feature_select.py -m <muse, GloVe, bert2>
"""   
import pandas as pd
import argparse
from sklearn.feature_selection import SelectKBest, f_classif

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", nargs=1, type=str, 
                        help="The name of the model you want to test", required=True)
    args = parser.parse_args()

    model_name = args.model[0]
    filename = f"output/paramTuning/{model_name}.csv"

    emotions = ["anger", "fear", "sadness", "disgust", "joy", "anger-disgust"]
    df = pd.read_csv(filename)

    feature_cols = ['resample', 'epochs', 'lr', 'batch_size','threshold']

    def do_the_thing(X, y):
        order_list=[]
        for i in range(len(feature_cols)):
            num_features_to_select = i+1   # Set the number of top features you want to select
            selector = SelectKBest(score_func=f_classif, k=num_features_to_select)
            X_new = selector.fit_transform(X, y)

            selected_feature_names = X.columns[selector.get_support()]
            for val in selected_feature_names:
                if val not in order_list:
                    order_list.append(val)
            
        return(order_list)
    
    def print_output(label, order_list):
        out_str = f"{label}: "
        for val in order_list:
            out_str += val
            if val != order_list[-1]:
                out_str += ", "
        print(out_str)

    X = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std()
    for emotion in emotions:
        y = df[f"{emotion}_accuracy"]
        print_output(emotion, do_the_thing(X,y))

    big_df = pd.DataFrame()
    for emotion in emotions:
        concat_df = df[feature_cols].copy()
        concat_df["accuracy"] = df[f"{emotion}_accuracy"]

        if big_df.empty:
            big_df = concat_df
        else:
            big_df = pd.concat([big_df, concat_df])
    
    X = (big_df[feature_cols] - big_df[feature_cols].mean()) / big_df[feature_cols].std()
    y = big_df["accuracy"]
    print_output("Combined",do_the_thing(X,y))

if __name__ == "__main__":
    main()