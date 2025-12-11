import pandas as pd
import numpy as np

def compute_country_metrics(df):
    if 'country' not in df.columns:
        raise ValueError("No 'country' column found.")
    if 'net_sales' not in df.columns:
        df['net_sales'] = df.get('total_sale_value', 0)
    df_sales = df[df['net_sales'] > 0].copy()
    if df_sales.empty:
        agg = df.groupby('country', as_index=False)['net_sales'].sum().rename(columns={'net_sales':'total_sales'})
        agg['tx_count'] = 0
    else:
        agg = df_sales.groupby('country', as_index=False)['net_sales'].sum().rename(columns={'net_sales':'total_sales'})
        tx = df_sales.groupby('country', as_index=False)['total_sale_value'].count().rename(columns={'total_sale_value':'tx_count'})
        agg = agg.merge(tx, on='country', how='left').fillna({'tx_count':0})
    agg = agg.sort_values('total_sales', ascending=False).reset_index(drop=True)
    return agg

def compute_product_metrics(df):
    if 'product' not in df.columns:
        raise ValueError("No 'product' column found.")
    if 'net_sales' not in df.columns:
        df['net_sales'] = df.get('total_sale_value', 0)
    df_sales = df[df['net_sales'] > 0].copy()
    if df_sales.empty:
        agg = df.groupby('product', as_index=False)['net_sales'].sum().rename(columns={'net_sales':'total_sales'})
    else:
        agg = df_sales.groupby('product', as_index=False)['net_sales'].sum().rename(columns={'net_sales':'total_sales'})
    if agg['total_sales'].max() - agg['total_sales'].min() > 0:
        agg['product_score'] = (agg['total_sales'] - agg['total_sales'].min()) / (agg['total_sales'].max() - agg['total_sales'].min())
    else:
        agg['product_score'] = 0.0
    agg = agg.sort_values('total_sales', ascending=False).reset_index(drop=True)
    return agg

def get_best_country(df_country_metrics, top_n=1, filters=None):
    dfc = df_country_metrics.copy()
    if filters:
        for k, v in filters.items():
            if k in dfc.columns:
                dfc = dfc[dfc[k] == v]
    return dfc.head(top_n)

def get_best_product(df_product_metrics, top_n=1, filters=None):
    dfp = df_product_metrics.copy()
    if filters:
        for k, v in filters.items():
            if k in dfp.columns:
                dfp = dfp[dfp[k] == v]
    return dfp.head(top_n)
