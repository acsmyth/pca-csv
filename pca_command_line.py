import csv
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


parser = ArgumentParser(prog='pca-csv', description='Quickly execute a principal component analysis on a CSV.')
parser.add_argument('csv_path', help='path to the CSV to process')
parser.add_argument('target_variable', help='name of the column you want to predict')
parser.add_argument('-e', '--exclude', default=[], help='list of column names to exclude from the computation')
args = parser.parse_args()


with open(args.csv_path, 'r') as csvfile:
  dialect = csv.Sniffer().sniff(csvfile.read(), delimiters=';,')

df = pd.read_csv(args.csv_path, dialect=dialect)

target = args.target_variable
possible_target_values = list(set(df[target]))

features = list(df.columns)

for column_name, dtype in zip(df.columns, df.dtypes):
  def drop(cl_name):
    df.drop(cl_name, axis=1, inplace=True)
    features.remove(cl_name)
  
  BOLD = '\033[1m'
  YELLOW = '\033[93m'
  END = '\033[0m'

  if column_name in args.exclude:
    features.remove(column_name)
    print(f'Ignoring column {BOLD}{column_name}{END} because it is manually excluded.')
  elif column_name.lower() == 'id':
    features.remove(column_name)
    print(f'Ignoring column {BOLD}{column_name}{END} because it appears to be a row identifier column.')
  elif column_name == target:
    features.remove(column_name)
    print(f'Ignoring column {BOLD}{column_name}{END} because it is the target column.')
  elif not pd.api.types.is_numeric_dtype(dtype):
    drop(column_name)
    print(f'{YELLOW}Warning{END}: Automatically ignoring column {BOLD}{column_name}{END} because {BOLD}{dtype}{END} is a non-numeric data type.')

data = df.loc[:, features].values

# TODO add warning for this
if df.isnull().values.any():
  df.fillna(0.0, inplace=True)
  print('Warning: Found NaN values in input, defaulting to 0')

data = StandardScaler().fit_transform(data)
if np.isnan(data).any():
  print('Warning: Found NaN values after normalizing, defaulting to 0')
  data[np.isnan(data)] = 0


pca = PCA(n_components=2)
principal_components = pca.fit_transform(data)
principal_df = pd.DataFrame(data = principal_components, columns = ['principal component 1', 'principal component 2'])
final_df = pd.concat([principal_df, df[[target]]], axis = 1)

fig = plt.figure(figsize = (12,12))
ax = fig.add_subplot(1,1,1) 
ax.set_title('2-Component PCA', fontsize = 30)

plt.xticks([], [])
plt.yticks([], [])

# Make it a 4-quadrant
min_x = min(final_df['principal component 1'])
max_x = max(final_df['principal component 1'])

min_y = min(final_df['principal component 2'])
max_y = max(final_df['principal component 2'])



ax.text(min_x, 0, '- pc1', fontsize = 15)
ax.text(max_x, 0, '+ pc1', fontsize = 15)

ax.text(0, min_y, '- pc2', fontsize = 15)
ax.text(0, max_y, '+ pc2', fontsize = 15)


# TODO - should do automatically by checking for very close values and shifting them a bit with some noise


for t in possible_target_values:
  row_indices_to_keep = final_df[target] == t

  ax.scatter(final_df.loc[row_indices_to_keep, 'principal component 1']
            , final_df.loc[row_indices_to_keep, 'principal component 2']
            , s = 20)

# TODO feature where if only a single one per row_indices_to_keep, then just label each of them individually instead
# of using a legend. only use a legend if multiple values per thing.
ax.legend(possible_target_values)

ax.grid()

ax.plot([min_x, max_x], [0, 0], linewidth=2, color='gray')
ax.plot([0, 0], [min_y, max_y], linewidth=2, color='gray')


print()
print('% Variance Explained:')
print(pca.explained_variance_ratio_)
print()
print('Component Vectors:')
print(pca.components_.round(2))
print()


# PCA Explained
for component in range(len(pca.components_)):
  print(f'Component {component+1}:')
  for feature, weight in sorted(zip(features, pca.components_[component]), key=lambda x: abs(x[1]), reverse=True):
    print(f'{feature}: {weight.round(2)}')
  print()

plt.show()
