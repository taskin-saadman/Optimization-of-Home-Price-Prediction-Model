import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

home_data = pd.read_csv('home_data.csv')
# target object
y = home_data.SalePrice
# choose features
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# splitting into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# function for getting mae on validation data for different leaf nodes
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# list of some candidate leaf nodes
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

'''FINDING THE BEST TREE SIZE'''
mae_dict = {} # Create an empty dictionary to store the leaf nodes and corresponding MAE

# populating mae_dict
for nodes in candidate_max_leaf_nodes:
    mae_dict[nodes] = get_mae(nodes, train_X, val_X, train_y, val_y)

# initializing variables to track the smallest value and corresponding key
smallest_key = None
smallest_value = float('inf')  # set to infinity so any value is smaller

# finding the smallest value in the dictionary
for key, val in mae_dict.items():
    if val < smallest_value:
        smallest_value = val
        smallest_key = key

# storing the best tree size
best_tree_size = smallest_key

# create a new model with the best tree size
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)

# fit the final model
'''Here we can use all of the data.
We do this because we have already found the best tree size and we want to use all of the data to train the model.'''
final_model.fit(X, y)

print("PREDICTED PRICES")
print(final_model.predict(X.head()).astype(int)) # convert to integer
print("ACTUAL PRICES")
print(y.head())
