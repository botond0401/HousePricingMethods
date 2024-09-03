# %%
#imports
import matplotlib.pyplot as plt
import seaborn as sns
from load_data import DataLoader

# %%
# load in training data
dataloader = DataLoader()
dataloader.load_data('data/train.csv')
df = dataloader.data
df.head()

# %%
# Create a histogram for all numerical variables
df_num = df.drop(columns='SalePrice').select_dtypes(include = ['float64', 'int64'])
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
df_cat = df.drop(columns='SalePrice').select_dtypes(include = ['object'])

# %%
# Iterate through each categorical column and create a bar plot
fig, axes = plt.subplots(nrows=len(df_cat.columns), ncols=1, figsize=(16, 200))
fig.tight_layout(pad=5.0)
for ax, col in zip(axes, df_cat.columns):
    df_cat[col].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f'Distribution of {col}', fontsize=12)
    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
plt.show()

# %%
# plot the distribution of the target variable
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['SalePrice'])
plt.title('Boxplot of column_name')
plt.xlabel('Values')
plt.show()

# %%
#get rid of outliers
sale_price_threshold = 500000
df = df[df['SalePrice'] < sale_price_threshold]
