import pandas as pd
import os

DATASET_PATH_MENTIONS_PREDICTIONS =  "../Datasets/dataset_mentions_predictions/"
PARTIES = ["SPD", "CDU_CSU", "GRUENE", "FDP", "AFD", "LINKE"]
DATASET_MENTIONS_PATH = "../Datasets/dataset_mentions/"
DATASET_POLITICIANS_PATH = "../Datasets/dataset_politicians/all_tweets_predicted_bert93.csv"

def load_politicians_dataset():
    df = pd.read_csv(DATASET_PATH)
    
    # Umbenennen einer Spalte
    df = df.rename(columns={"Embedded_text": "text", "UserName": "source_account", "Partei": "source_party", "sentiment_prediction": "sentiment", "Timestamp": "date", "Image link": "photos"})
    
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['source_account'] = df['source_account'].str.lstrip('@')
    
    # Löschen einer Spalte durch ihren Namen
    df = df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"])
    
    # Mapping der Parteinamen
    party_mapping = {
        'AfD': 'AFD',
        'CDU/CSU': 'CDU_CSU',
        'DieLinke': 'LINKE',
        'Gruene': 'GRUENE',
        'FDP': 'FDP',
        'SPD': 'SPD'
    }
    
    df['source_party'] = df['source_party'].map(party_mapping)
    df['tweet_id'] = df['Tweet URL'].str.extract(r'status/(\d+)')
    
    return df

def load_mentions_dataset():
    df = pd.DataFrame({})

    for party in PARTIES:
        for subdir, _, files in os.walk(DATASET_PATH_MENTIONS_PREDICTIONS + party):
            for file in files:
                if file.endswith('.csv') and subdir[len(DATASET_PATH_MENTIONS_PREDICTIONS):] in PARTIES:
                    # Get username of CSV file
                    username = file[:-4]
                
                    # Read CSV file as pandas dataframe
                    df_acc_data = pd.read_csv(DATASET_MENTIONS_PATH + party + "/" + file)
                    ids = df_acc_data["id"].values
                    df_acc_data = df_acc_data[["tweet", "source_party", "source_account", "date"]].reset_index().drop(columns='index')
                
                    df_pred = pd.read_csv(DATASET_PATH_MENTIONS_PREDICTIONS + party + "/" + file)
                    df_pred = df_pred[df_pred["id"].isin(ids)][["pred"]].reset_index().drop(columns='index')
                
                    matched_df = pd.concat([df_acc_data, df_pred], axis=1)
                    matched_df = matched_df.rename(columns={"pred": "sentiment", "tweet": "text"})
                
                    df = pd.concat([df, matched_df], axis=0)

    return df.reset_index().drop(columns='index')