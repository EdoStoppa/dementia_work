import pandas as pd

# Load data
old_data = pd.read_csv('data/partialData.csv', skipinitialspace=True)
new_data = pd.read_csv('data/newLinguistic.csv', skipinitialspace=True)
liwc = pd.read_csv('data/LIWC.csv').drop(['utterance', 'Segment'], axis=1)

##############    LIWC    ##############
# Fix index
liwc['file'] = liwc['file'].map(lambda x: x.split('_')[0])
# Fix Columns
liwc.columns = ['REGTRYID'] + [e.replace('(', '').replace(')', '').replace('-', '') for e in liwc.columns.to_list()][1:]
# Force type int to WC
liwc['WC'] = liwc['WC'].astype(int)

##############    Other Data    ##############
# Preprocess the ids contained in new data
new_data['REGTRYID'] = new_data['REGTRYID'].map(lambda x: x.split('_')[0])
# Force correct ordering
old_data = old_data.sort_values(by='REGTRYID')
new_data = new_data.sort_values(by='REGTRYID')
# Use only columns related to "participant"
new_cols = [e.replace('(', '').replace(')', '').replace('-', '') for e in new_data.columns.to_list() if 'participant' in e]
new_cols = ['REGTRYID'] + [''.join(e.split()) for e in new_cols]
# Drop all columns that are not related to "participant"
new_data = new_data.drop([e for e in new_data.columns.to_list() if 'participant' not in e][1:], axis=1)
# Set the new processed column names
new_data.columns = new_cols
# Substitute the line for id 3722
idx_old = old_data.REGTRYID[old_data.REGTRYID == 3722].index.tolist()[-1]
idx_new = new_data.REGTRYID[new_data.REGTRYID.astype(int) == 3722].index.tolist()[-1]
new_data.iloc[idx_new] = old_data.drop(old_data.columns.to_list()[48:], axis=1).iloc[idx_old]
# Force the features to be int
for e in new_data.columns.to_list()[:-6]:
    new_data[e] = new_data[e].astype(int)

##############    Save the dataset    ##############
# Add new linguistic data
old_data = old_data.drop(old_data.columns.to_list()[:48], axis=1)
final = pd.concat([new_data, old_data], axis=1)
# Set the correct indexes
final.set_index('REGTRYID', inplace=True)
liwc.set_index('REGTRYID', inplace=True)
# Force indexes to be int
final.index = final.index.astype(int)
liwc.index = liwc.index.astype(int)
# Insert new data
final.update(liwc)
# Force feature 'WC' to be int
final['WC'] = final['WC'].astype(int)
# Dump full dataset
final.to_csv('data/fullData.csv')
