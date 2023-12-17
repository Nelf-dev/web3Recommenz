import pandas as pd

# Set the path for the data
import_path = r'..\..\Data' # This folder is not in the repo due to the size of the raw dataset
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

# Create a dataframe with the nodes and sentiment
df_nodes = pd.DataFrame({'Node': repeated_nodes,'Sentiment': sentiment})

# Merge the two dataframes, reorder the columns and sort by Nodes
df = pd.merge(df_sample, df_nodes, left_index=True, right_index=True)
col_order = ['Node','Sentiment','VideoID','Caption']
df = df.reindex(columns=col_order)

# Create dataset with training and test data
df_n1 = pd.concat([df.query('Node == "node1"').iloc[:9,:], df.query('Node == "node1"').iloc[12:15,:]])
df_n2 = pd.concat([df.query('Node == "node2"').iloc[:9,:], df.query('Node == "node2"').iloc[12:15,:]])

# Drop values from Sentiment column for rows 12:14
df_n1.loc[12:14, 'Sentiment'] = ''
df_n2.loc[12:14, 'Sentiment'] = ''

# Export data to csv files
df_n1.to_csv(f'{export_path}\\node1_data.csv', index=False)
df_n2.to_csv(f'{export_path}\\node2_data.csv', index=False)

# Export the incremental learning data to csv files
df.query('Node == "node1"').iloc[:10,:].to_csv(f'{export_path}\\node1_incremental_1.csv', index=False)
df.query('Node == "node1"').iloc[:11,:].to_csv(f'{export_path}\\node1_incremental_2.csv', index=False)
df.query('Node == "node1"').iloc[:12,:].to_csv(f'{export_path}\\node1_incremental_3.csv', index=False)
df.query('Node == "node2"').iloc[:10,:].to_csv(f'{export_path}\\node2_incremental_1.csv', index=False)
df.query('Node == "node2"').iloc[:11,:].to_csv(f'{export_path}\\node2_incremental_2.csv', index=False)
df.query('Node == "node2"').iloc[:12,:].to_csv(f'{export_path}\\node2_incremental_3.csv', index=False)