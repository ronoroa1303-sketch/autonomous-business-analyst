import sqlite3
import os
import pandas as pd

def get_connection():
    """
    Create and return a connection to the SQLite database.
    Also loads the dataset if it doesn't exist yet.
    """
    db_path = os.path.join(os.path.dirname(__file__), "ecommerce.db")
    
    conn = sqlite3.connect(db_path)
    
    load_data_if_not_exists(conn)
    
    return conn

def load_data_if_not_exists(conn):
    
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='orders';
    """)
    table_exists = cursor.fetchone()
    
    if not table_exists:
        print("Loading dataset into database...")
        
        dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset")
        
        csv_files = {
            "olist_orders_dataset.csv": "orders",
            "olist_order_items_dataset.csv": "order_items",
            "olist_order_payments_dataset.csv": "payments",
            "olist_products_dataset.csv": "products",
            "olist_customers_dataset.csv": "customers",
            "olist_sellers_dataset.csv": "sellers",
            "olist_geolocation_dataset.csv": "geolocation",
            "olist_order_reviews_dataset.csv": "reviews",
            "product_category_name_translation.csv": "category_translation"
        }
        
        # Load each CSV file
        for csv_file, table_name in csv_files.items():
            csv_path = os.path.join(dataset_path, csv_file)
            
            # Check if file exists
            if os.path.exists(csv_path):
                print(f"Loading {csv_file} into {table_name} table...")
                
                
                df = pd.read_csv(csv_path)
                
                
                df.to_sql(table_name, conn, if_exists='fail', index=False)
                
                print(f"Loaded {len(df)} rows into {table_name}")
            else:
                print(f"Warning: {csv_file} not found in {dataset_path}")
        
        # Commit the changes
        conn.commit()
        print("Dataset loading completed!")
    else:
        print("Database tables already exist, skipping data load.")