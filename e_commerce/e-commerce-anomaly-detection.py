#!/usr/bin/env python
# coding: utf-8
# # Data download

# +
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lux 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('./data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# -

customers = pd.read_csv("./data/olist_customers_dataset.csv")
sellers = pd.read_csv("./data/olist_sellers_dataset.csv")
reviews = pd.read_csv("./data/olist_order_reviews_dataset.csv")
items = pd.read_csv("./data/olist_order_items_dataset.csv")
products = pd.read_csv("./data/olist_products_dataset.csv")
geolocation = pd.read_csv("./data/olist_geolocation_dataset.csv")
category_name_translation = pd.read_csv("./data/product_category_name_translation.csv")
orders = pd.read_csv("./data/olist_orders_dataset.csv")
order_payments = pd.read_csv("./data/olist_order_payments_dataset.csv")


# ## Merge datasets

datasets = [customers, sellers, reviews, items, products, geolocation, category_name_translation, orders, order_payments]

df = orders.merge(items, on="order_id").merge(products, on = "product_id").merge(sellers, on="seller_id").merge(customers, on="customer_id")

df.info()

# # Data cleaning

# +
#date_cols = [order_delivered_customer_date", "order_estimated_delivery_date", "order_purchase_timestamp", "order_delivered_customer_date"]

df["order_delivered_carrier_date"] = pd.to_datetime(df["order_delivered_carrier_date"])
df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])
df['order_approved_at'] = pd.to_datetime(df['order_approved_at'])
df['shipping_limit_date'] = pd.to_datetime(df['shipping_limit_date'])

df['expected_delivery_timedelta'] = pd.to_numeric(df['order_estimated_delivery_date']- df["order_purchase_timestamp"])
df['delivery_time'] = pd.to_numeric(df["order_delivered_customer_date"] - df['order_purchase_timestamp'])

df['product_volume'] = df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]

df["same_city"] = df["customer_city"] == df["seller_city"]
df["same_state"] = df["customer_state"] == df["seller_state"]

# +
#df.drop(["expected_delivery_timedelta"], axis = 1, inplace=True)
# -

df

df.info()

# ## Detecting outliers in numerical data



# +
unique_orders_count = df.groupby(["customer_id"])["order_id"].count() # Count unique orders
nof_moest_popular_sales = df.groupby(["customer_id"])["seller_id"].agg(lambda x:x.value_counts()[0]) # Number of orders from most popular sellers
max_sale = df.groupby(["customer_id"])["price"].max() # Most money spent on single order
median_sale = df.groupby(["customer_id"])["price"].median() # Median money spent on orders
sum_sale = df.groupby(["customer_id"])["price"].sum() # Median money spent on orders

median_volume = df.groupby(["customer_id"])["product_volume"].median() # Median volume of products in orders
max_volume = df.groupby(["customer_id"])["product_volume"].max() # Median money spent on orders

average_delivery_time = df.groupby(["customer_id"])["delivery_time"].max() # Average delivery time
average_expected_delivery_time = df.groupby(["customer_id"])["expected_delivery_timedelta"].max() # Average delivery time


unique_orders_count.name = "unique_orders_count"
nof_moest_popular_sales.name = "nof_moest_popular_sales"
max_sale.name = "max_sale"
median_sale.name = "median_sale"
sum_sale.name = "sum_sale"
median_volume.name = "median_volume"
average_delivery_time.name = "average_delivery_time"
average_expected_delivery_time.name = "average_expected_delivery_time"
# -

customers2 = customers.set_index("customer_id")

customers2

df2 = customers2.join([unique_orders_count, nof_moest_popular_sales, max_sale, median_sale, sum_sale, median_volume, average_delivery_time,average_expected_delivery_time],  how="outer")

df2[df2["unique_orders_count"] > 1]

cat_attributes = ["customer_city", "customer_state"]
num_attributes = ["unique_orders_count", "nof_moest_popular_sales", "max_sale", "median_sale", "sum_sale", "median_volume", "average_delivery_time","average_expected_delivery_time"]

df2['average_expected_delivery_time'] = df2['average_expected_delivery_time']
df2['average_delivery_time'] = df2['average_delivery_time']

df2.fillna(df2.mean(), inplace=True)

df2.info()

df2.groupby("customer_city")["customer_state"].value_counts()

groupby_col="customer_city"


other_countes = df2.groupby(groupby_col).count().sort_values('customer_state', ascending=False).iloc[21:,:].reset_index()[groupby_col].array

df2.loc[df2.customer_city.isin(other_countes), "customer_city"] = "other"

df2.customer_city.nunique()

# +
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


pipeline = ColumnTransformer([
        ('num', StandardScaler(), num_attributes),
        ('cat', OneHotEncoder(), cat_attributes),
])
df_prepared = pipeline.fit_transform(df2)
df_prepared
# -

df3 = df_prepared.toarray()

len(num_attributes)

import lux
df3

# # HDBSCAN

import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=600, min_samples=80, cluster_selection_epsilon=0.6)

#clusterer.fit(df3[np.random.choice(df3.shape[0],50000, replace=False),:])
clusterer.fit(df3)

max(clusterer.labels_)

import seaborn as sns
sns.histplot(clusterer.labels_)

# # UMAP

import umap

import umap.plot

mapper = umap.UMAP(densmap=True).fit(df3)

umap.plot.points(mapper)

# ## HDBSCAN viz

umap.plot.points(mapper, labels=clusterer.labels_)

mapper2 = umap.UMAP(densmap=True).fit(df3[np.random.choice(df3.shape[0], 10000, replace=False)])


umap.plot.points(mapper2)

df3.data

df3.shape

df3

# # KMEANS

# +
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
model = KMeans()

visualizer = KElbowVisualizer(model, k=(10,30), timings=True)
visualizer.fit(df3)
visualizer.show()
# -

X = df3
kmeans = KMeans(n_clusters=21, random_state=0, copy_x=False, precompute_distances=False).fit(X)

kmeans.labels_

sns.histplot(kmeans.labels_, bins=max(kmeans.labels_))

# ## Kmeans viz

umap.plot.points(mapper, labels=kmeans.labels_)

points = mapper.embedding_

points[:,0]

df2

umap.plot.connectivity(mapper, show_points=True, labels=kmeans.labels_)

# + active=""
# umap.plot.connectivity(mapper, edge_bundling='hammer')
# -

import seaborn as sns

points

df2["x"] = points[:,0]
df2["y"] = points[:,1]

df2['y']

df2

countries = df2.groupby('customer_city').count().sort_values('customer_state', ascending=False).iloc[0:10,:].reset_index()['customer_city'].array
sns.relplot(
    data = df2.loc[df2['customer_city'].isin(countries)],
    x = "x",
    y = "y",
    hue = 'customer_city',
    height = 12,
    s=200)

df2.columns

smapper = umap.UMAP().fit_transform(X, kmeans.labels_)



# +
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(20,10))

sns.scatterplot(x=smapper[:,0], y=smapper[:,1], hue=kmeans.labels_, ax=ax, palette=sns.color_palette("icefire", 21))

# +
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(20,10))

sns.scatterplot(x=mapper.embedding_[:,0], y=mapper.embedding_[:,1], hue=kmeans.labels_, ax=ax, palette=sns.color_palette("icefire", as_cmap=True))
# -

df2['x1'] = smapper[:,0]
df2['y1'] = smapper[:,1]

# +
groupby_col="customer_city"

countries = df2.groupby(groupby_col).count().sort_values('customer_state', ascending=False).iloc[0:20,:].reset_index()[groupby_col].array

sns.relplot(
    data = df2.loc[df2[groupby_col].isin(countries)],
    x = "x1",
    y = "y1",
    hue = groupby_col,
    height = 12,
    s=200)
# -

df2.loc[df2[groupby_col].isin(countries)]

outlier_scores = sklearn.neighbors.LocalOutlierFactor(contamination=0.001428).fit_predict(mapper.embedding_)

import sklearn

outlier_scores

df2['outlier'] = outlier_scores

df2.describe()

sns.relplot(
    data = df2,
    x = "x",
    y = "y",
    hue = "outlier",
    height = 12,
    s=200)

df.info(0)



df4 = df.merge(df2, on="customer_unique_id")

sns.pointplot(x="order_purchase_timestamp", y="product_width_cm", data=df4.sample(100), hue='outlier')

sns.boxplot(x="price", data=df.sample(1000))

sns.boxplot(x="product_volume", data=df.sample(1000))

# # Analizing clusters

df2["cluster"] = kmeans.labels_

df2.columns

sns.violinplot(data=df2,x="cluster", y="max_sale")

# For all float columns
cols = df2.columns[df2.dtypes == "float64"]

import matplotlib.pyplot as plt
from pylab import *

nrow = 2
ncol = 4
fig, axs = plt.subplots(nrow, ncol, figsize=(40,20))
for i, ax in enumerate(fig.axes):
    ax.set_ylabel(str(cols[i]))
    sns.violinplot(data=df2,x="cluster", y=cols[i], ax=ax)

df2.groupby("cluster").mean()

df2.mean()

# +
# Author: YousefGh
# Source: https://github.com/YousefGh/kmeans-feature-importance

from sklearn.cluster import KMeans
import numpy as np


class KMeansInterp(KMeans):
    def __init__(self, ordered_feature_names, feature_importance_method='wcss_min', **kwargs):
        super(KMeansInterp, self).__init__(**kwargs)
        self.feature_importance_method = feature_importance_method
        self.ordered_feature_names = ordered_feature_names
        
    def fit(self, X, y=None, sample_weight=None):
        super().fit(X=X, y=y, sample_weight=sample_weight)
        
        if not len(self.ordered_feature_names) == self.n_features_in_:
            raise Exception(f"Model is fitted on {self.n_features_in_} but ordered_feature_names = {len(self.ordered_feature_names)}")
        
        if self.feature_importance_method == "wcss_min":
            self.feature_importances_ = self.get_feature_imp_wcss_min()
        elif self.feature_importance_method == "unsup2sup":
            self.feature_importances_ = self.get_feature_imp_unsup2sup(X)
        else: 
            raise Exception(f" {self.feature_importance_method}"+\
            "is not available. Please choose from  ['wcss_min' , 'unsup2sup']")
        
        return self
        
    def get_feature_imp_wcss_min(self):
        labels = self.n_clusters
        centroids = self.cluster_centers_
        centroids = np.vectorize(lambda x: np.abs(x))(centroids)
        sorted_centroid_features_idx = centroids.argsort(axis=1)[:,::-1]

        cluster_feature_weights = {}
        for label, centroid in zip(range(labels), sorted_centroid_features_idx):
            ordered_cluster_feature_weights = centroids[label][sorted_centroid_features_idx[label]]
            ordered_cluster_features = [self.ordered_feature_names[feature] for feature in centroid]
            cluster_feature_weights[label] = list(zip(ordered_cluster_features, 
                                                      ordered_cluster_feature_weights))
        
        return cluster_feature_weights
    
    def get_feature_imp_unsup2sup(self, X):
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError as IE:
            print(IE.__class__.__name__ + ": " + IE.message)
            raise Exception("Please install scikit-learn. " + 
                            "'unsup2sup' method requires using a classifier"+ 
                            "and depends on 'sklearn.ensemble.RandomForestClassifier'")
        
        cluster_feature_weights = {}
        for label in range(self.n_clusters):
            binary_enc = np.vectorize(lambda x: 1 if x == label else 0)(self.labels_)
            clf = RandomForestClassifier()
            clf.fit(X, binary_enc)

            sorted_feature_weight_idxes = np.argsort(clf.feature_importances_)[::-1]
            ordered_cluster_features = np.take_along_axis(
                np.array(self.ordered_feature_names), 
                sorted_feature_weight_idxes, 
                axis=0)
            ordered_cluster_feature_weights = np.take_along_axis(
                np.array(clf.feature_importances_), 
                sorted_feature_weight_idxes, 
                axis=0)
            cluster_feature_weights[label] = list(zip(ordered_cluster_features, 
                                                      ordered_cluster_feature_weights))
        return cluster_feature_weights


# -

# # KMeans Interpret

pipeline.transformers_[1][1].get_feature_names()



interpreter = KMeansInterp(n_clusters=21, random_state=0, copy_x=False, ordered_feature_names=np.concatenate((cols.to_numpy(), pipeline.transformers_[1][1].get_feature_names()))).fit(X)

for cluster_label, feature_weights in interpreter.feature_importances_.items():    
    df_feature_weight = pd.DataFrame(feature_weights[:15], columns=["Feature", "Weight"])
    fig, ax = plt.subplots(figsize=(14,6))
    sns.barplot(x="Feature", y="Weight", data=df_feature_weight)
    plt.xticks(rotation=-45, ha="left");
    ax.tick_params(axis='both', which='major', labelsize=22)
    plt.title(f'Highest Weight Features in Cluster {cluster_label}', fontsize='xx-large')
    plt.xlabel('Feature', fontsize=18)
    plt.ylabel('Weight', fontsize=18)

    plt.show();
    
    print('\n\n')


