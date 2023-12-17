import random
import pandas as pd

# Set the path for the data
import_path = r'..\..\Data'
export_path = r'data\video'

# Read a parquet dataset
df_raw = pd.read_parquet(f'{import_path}\\0000.parquet')
df_sample = df_raw.sample(30, random_state=30).reset_index(drop=True)
df_sample.drop(columns=['Start_timestamp','End_timestamp','UMT-SIM'], inplace=True)
df_sample.rename(columns={'YoutubeID':'VideoID'}, inplace=True)

# Generate a list of nodes and a list of sentiment
repeated_nodes = ['node1'] * 15 + ['node2'] * 15

# Generate a list of sentiment
sentiment_node1 = ['good'] * 4 + ['bad'] * 10 + ['neutral'] * 1
sentiment_node2 = ['good'] * 5 + ['bad'] * 5 + ['neutral'] * 5
sentiment = sentiment_node1 + sentiment_node2

# Randomly shuffle the nodes and sentiment
# import random
# random.shuffle(repeated_nodes)
# random.shuffle(repeated_sentiment)

# Create a dataframe with the nodes and sentiment
df_nodes = pd.DataFrame({'Node': repeated_nodes,'Sentiment': sentiment})

# Merge the two dataframes, reorder the columns and sort by Nodes
df = pd.merge(df_sample, df_nodes, left_index=True, right_index=True)
col_order = ['Node','Sentiment','VideoID','Caption']
df = df.reindex(columns=col_order)

# Export the training data to csv files
df.query('Node == "node1"').iloc[:9,:].to_csv(f'{export_path}\\node1_train.csv', index=False)
df.query('Node == "node2"').iloc[:9,:].to_csv(f'{export_path}\\node2_train.csv', index=False)

# Export test data to csv files
df.query('Node == "node1"').iloc[12:15,[0,2,3]].to_csv(f'{export_path}\\node1_test.csv', index=False)
df.query('Node == "node2"').iloc[12:15,[0,2,3]].to_csv(f'{export_path}\\node2_test.csv', index=False)

# Export the incremental learning data to csv files
df.query('Node == "node1"').iloc[:10,:].to_csv(f'{export_path}\\node1_incremental_1.csv', index=False)
df.query('Node == "node1"').iloc[:11,:].to_csv(f'{export_path}\\node1_incremental_2.csv', index=False)
df.query('Node == "node1"').iloc[:12,:].to_csv(f'{export_path}\\node1_incremental_3.csv', index=False)
df.query('Node == "node2"').iloc[:10,:].to_csv(f'{export_path}\\node2_incremental_1.csv', index=False)
df.query('Node == "node2"').iloc[:11,:].to_csv(f'{export_path}\\node2_incremental_2.csv', index=False)
df.query('Node == "node2"').iloc[:12,:].to_csv(f'{export_path}\\node2_incremental_3.csv', index=False)