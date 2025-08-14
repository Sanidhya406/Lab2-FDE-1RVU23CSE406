import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Ingestion
sales_df = pd.read_csv("Data Engineering/Lab-2/data_warehouse/processed_sales_data.csv")

# 2. Feature Engineering
customer_stats = sales_df.groupby('customer_id').agg(
    total_purchase_amount=('sale_price', 'sum'),
    purchase_frequency=('sale_price', 'count'),
    avg_transaction_value=('sale_price', 'mean')
).reset_index()

# 3. Preprocessing
scaler = StandardScaler()
features = ['total_purchase_amount', 'purchase_frequency', 'avg_transaction_value']
customer_stats_scaled = scaler.fit_transform(customer_stats[features])

# 4. VIP Classification (K-Means Clustering)
kmeans = KMeans(n_clusters=2, random_state=42)
customer_stats['vip_label'] = kmeans.fit_predict(customer_stats_scaled)

# Optional: Assign VIP to cluster with higher total purchase amount
vip_cluster = customer_stats.groupby('vip_label')['total_purchase_amount'].mean().idxmax()
customer_stats['VIP_status'] = customer_stats['vip_label'].apply(lambda x: 'VIP' if x == vip_cluster else 'Non-VIP')
customer_stats = customer_stats.drop(columns=['vip_label'])

# 5. Enrich Original Dataset
enriched_df = pd.merge(sales_df, customer_stats[['customer_id', 'VIP_status']], on='customer_id', how='left')

# 6. Reverse ETL: Export enriched data
enriched_df.to_csv("Data Engineering/Lab-2/data_warehouse/enriched_sales_data.csv", index=False)
print("Enriched data with VIP status exported successfully.")