import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------
# STEP 1: Load & Clean
# ------------------------------------
df = pd.read_csv(r"C:\Users\dell\Documents\customer.csv")
df['Description'].fillna("Unknown Product", inplace=True)

# Split known vs guest
known = df[df['CustomerID'].notna()].copy()
guest = df[df['CustomerID'].isna()].copy()

print(f"✅ Known customers: {known.shape[0]}")
print(f"✅ Guest customers: {guest.shape[0]}")
print('-' * 100)

# ------------------------------------
# STEP 2: Reference Date
# ------------------------------------
known['InvoiceDate'] = pd.to_datetime(known['InvoiceDate'], dayfirst=True)
guest['InvoiceDate'] = pd.to_datetime(guest['InvoiceDate'], dayfirst=True)
ref_date = known['InvoiceDate'].max() + pd.Timedelta(days=1)

# ------------------------------------
# STEP 3: RFM for Known Customers
# ------------------------------------
known['TotalPrice'] = known['Quantity'] * known['UnitPrice']

rfm = known.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (ref_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Merge country info
customer_country = known[['CustomerID', 'Country']].drop_duplicates()
rfm = rfm.merge(customer_country, on='CustomerID', how='left')

# Scale known RFM
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# KMeans Clustering for Known Customers
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

segment_map = {
    0: 'Champions',
    1: 'Loyal',
    2: 'Potential',
    3: 'At Risk'
}
rfm['Segment'] = rfm['Cluster'].map(segment_map)

# ------------------------------------
# STEP 4: RFM for Guest Customers
# ------------------------------------
guest['TotalPrice'] = guest['Quantity'] * guest['UnitPrice']

guest_rfm = guest.groupby('InvoiceNo').agg({
    'InvoiceDate': lambda x: (ref_date - x.max()).days,
    'Quantity': 'sum',
    'TotalPrice': 'sum'
}).reset_index()

# Rename columns
guest_rfm.columns = ['InvoiceNo', 'Recency', 'Frequency', 'Monetary']

# Scale guest data
guest_scaled = scaler.fit_transform(guest_rfm[['Recency', 'Frequency', 'Monetary']])

# Elbow Plot for Guests
inertias = []
K = range(2, 12)
for k in K:
    guest_kmeans = KMeans(n_clusters=k, random_state=42)
    guest_kmeans.fit(guest_scaled)
    inertias.append(guest_kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K, inertias, marker='o', linestyle='--', color='teal')
plt.title('Elbow Method for Guest Segments')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Final clustering for Guests (choose k=3 based on elbow)
guest_kmeans = KMeans(n_clusters=3, random_state=42)
guest_rfm['Cluster'] = guest_kmeans.fit_predict(guest_scaled)

# Label guest clusters
guest_labels = {
    0: 'Low-Value Guests',
    1: 'One-Time Big Spenders',
    2: 'Recent High-Spenders'
}
guest_rfm['Segment'] = guest_rfm['Cluster'].map(guest_labels)

# ------------------------------------
# STEP 5: Segment Summary by Country
# ------------------------------------
country_segment_summary = (
    rfm.groupby(['Country', 'Segment'])
       .size()
       .unstack(fill_value=0)
       .sort_values('Champions', ascending=False)
)

# ------------------------------------
# STEP 6: Visualizations
# ------------------------------------
plt.figure(figsize=(12, 6))
sns.countplot(data=rfm, x='Segment', order=rfm['Segment'].value_counts().index, palette='Set2')
plt.title("Known Customer Segments")
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(data=guest_rfm, x='Segment', order=guest_rfm['Segment'].value_counts().index, palette='coolwarm')
plt.title("Guest Invoice Segments")
plt.show()

plt.figure(figsize=(12, 6))
sns.heatmap(country_segment_summary, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Customer Segments by Country')
plt.xlabel('Segment')
plt.ylabel('Country')
plt.tight_layout()
plt.show()

# ------------------------------------
# STEP 7: Export Results
# ------------------------------------
rfm.to_csv("known_customer_segments.csv", index=False)
guest_rfm.to_csv("guest_invoice_segments.csv", index=False)

print("✅ All segments exported:")
print("📁 known_customer_segments.csv")
print("📁 guest_invoice_segments.csv")
