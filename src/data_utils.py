import pandas as pd
import numpy as np
import os

COLUMN_MAP = {
    'invoice': ['InvoiceNo', 'Invoice', 'invoice', 'Invoice Number'],
    'stock_code': ['StockCode', 'Stock Code', 'stockcode'],
    'product': ['Description', 'Product', 'Item', 'description'],
    'quantity': ['Quantity', 'Qty', 'quantity'],
    'invoice_date': ['InvoiceDate', 'Invoice Date', 'Date'],
    'unit_price': ['UnitPrice', 'Unit Price', 'Price', 'unit_price'],
    'customer_id': ['CustomerID', 'Customer ID', 'customer_id'],
    'country': ['Country', 'Nation', 'country'],
    'total_price': ['TotalPrice', 'Total Price', 'total', 'total_price']
}

def find_col(df, candidates):
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lc = {col.lower(): col for col in cols}
    for c in candidates:
        if c.lower() in lc:
            return lc[c.lower()]
    return None

def detect_columns(df):
    detected = {}
    for k, candidates in COLUMN_MAP.items():
        found = find_col(df, candidates)
        if found:
            detected[k] = found
    return detected

def load_data(path="data/Retail.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected dataset at {path}. Place your CSV there.")
    try:
        df = pd.read_csv(path, encoding='utf-8', low_memory=False)
    except Exception:
        df = pd.read_excel(path)
    return df

def preprocess(df):
    df = df.copy()
    detected = detect_columns(df)

    # rename detected columns to short names
    rename_map = {}
    for std, orig in detected.items():
        rename_map[orig] = std
    if rename_map:
        df = df.rename(columns=rename_map)

    # Trim strings
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip()

    # lowercase product/country for matching
    for txt in ['product', 'country']:
        if txt in df.columns:
            df[txt] = df[txt].astype(str).str.lower()

    # convert numeric columns safely
    for col in ['quantity', 'unit_price', 'total_price']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # compute total_sale_value
    if 'total_price' in df.columns:
        df['total_sale_value'] = df['total_price'].fillna(0)
    elif 'quantity' in df.columns and 'unit_price' in df.columns:
        df['total_sale_value'] = df['quantity'].fillna(0) * df['unit_price'].fillna(0)
    else:
        df['total_sale_value'] = 0

    # parse invoice_date if present
    if 'invoice_date' in df.columns:
        df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')

    # Fill missing quantity
    if 'quantity' in df.columns:
        df['quantity'] = df['quantity'].fillna(0)

    #  NEW: mark returns and create net_sales 
    df['is_return'] = False
    if 'quantity' in df.columns:
        try:
            df['is_return'] = df['quantity'].astype(float) < 0
        except Exception:
            df['is_return'] = False

    # net_sales: use total_sale_value
    df['net_sales'] = df['total_sale_value'].fillna(0)

    # Add a sales_only flag for positive net sales
    df['sales_only'] = df['net_sales'] > 0

    return df

def quick_inspect(df, n=5):
    print("Rows:", df.shape[0], "Cols:", df.shape[1])
    print("Detected standard columns present:", [c for c in ['invoice','stock_code','product','quantity','unit_price','total_price','country','customer_id','invoice_date'] if c in df.columns])
    return df.head(n)
