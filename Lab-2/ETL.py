

# -------------------------------
# Stage 1: Business Analyst Task
# -------------------------------
# Business Question: What are the top 5 products by revenue in the last quarter, and how does customer sentiment vary for these products?
# Required data points: product_id, sale_price, quantity, sale_date, customer_id, sentiment_score

# -------------------------------
# Stage 2: Data Engineer Task
# -------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure directory structure
os.makedirs("data_warehouse", exist_ok=True)

# -------------------------------
# Stage 1: Business Analyst Task
# -------------------------------
# Business Question: What are the top 5 products by revenue in the last quarter, 
# and how does customer sentiment vary for these products?
# Required data points: product_id, sale_price, quantity, sale_date, customer_id, sentiment_score

# -------------------------------
# Stage 2: Data Engineer Task
# -------------------------------

# 1. Ingestion
sales_df = pd.read_csv("Data Engineering/Lab-2/raw_data/sale_price.csv")
feedback_df = pd.read_json("Data Engineering/Lab-2/raw_data/customer_feedback.json")

# 2. Cleansing

# Remove '$' and convert sale_price to float
sales_df['sale_price'] = sales_df['sale_price'].replace(r'[\$,]', '', regex=True).astype(float)

# Fill missing quantity with 1 (default)
sales_df['quantity'] = sales_df['quantity'].fillna(1).astype(int)

# Standardize date format
sales_df['sale_date'] = pd.to_datetime(sales_df['sale_date'], errors='coerce', dayfirst=True)
feedback_df['review_date'] = pd.to_datetime(feedback_df['review_date'], errors='coerce')

# 3. Transformation

# Compute total revenue
sales_df['total_revenue'] = sales_df['sale_price'] * sales_df['quantity']

# Keep the latest feedback entry per (product_id, customer_id)
feedback_df = feedback_df.sort_values('review_date').drop_duplicates(
    subset=['product_id', 'customer_id'], keep='last'
)

# Merge sales and feedback
merged_df = pd.merge(sales_df, feedback_df, on=['product_id', 'customer_id'], how='left')

# Remove rows where sale_price is 0 or review_date is missing
merged_df = merged_df[(merged_df['sale_price'] > 0) & (merged_df['review_date'].notna())]

# 4. Loading to warehouse
processed_path = "Data Engineering/Lab-2/data_warehouse/processed_sales_data.csv"
merged_df.to_csv(processed_path, index=False)

print(f"Processed data saved to {processed_path}")








