import pandas as pd
import numpy as np
from faker import Faker
import random
import os

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

def generate_csvs(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Suppliers
    suppliers = []
    print("Generating suppliers...")
    for i in range(50):
        suppliers.append({
            "supplier_id": f"S_{i}",
            "name": fake.company(),
            "country": fake.country(),
            "reliability_score": round(random.uniform(0.5, 1.0), 2),
            "active_since": fake.year()
        })
    df_supp = pd.DataFrame(suppliers)
    df_supp.to_csv(f"{output_dir}/suppliers.csv", index=False)
    
    # 2. Warehouses
    warehouses = []
    print("Generating warehouses...")
    for i in range(10):
        warehouses.append({
            "warehouse_id": f"W_{i}",
            "location": fake.city(),
            "capacity": random.randint(1000, 10000),
            "current_load": round(random.uniform(0.1, 0.9), 2)
        })
    df_wh = pd.DataFrame(warehouses)
    df_wh.to_csv(f"{output_dir}/warehouses.csv", index=False)
    
    # 3. Products
    products = []
    print("Generating products...")
    categories = ["Electronics", "Apparel", "Home", "Automotive", "Toys"]
    for i in range(200):
        products.append({
            "product_id": f"P_{i}",
            "name": fake.word().capitalize() + " " + fake.word().capitalize(),
            "category": random.choice(categories),
            "supplier_id": random.choice(df_supp["supplier_id"])
        })
    df_prod = pd.DataFrame(products)
    df_prod.to_csv(f"{output_dir}/products.csv", index=False)
    
    # 4. Shipments
    shipments = []
    print("Generating shipments...")
    weather_conds = ["Clear", "Rain", "Snow", "Storm"]
    for i in range(5000):
        prod = random.choice(products)
        supp_id = prod["supplier_id"]
        supp_rel = float(df_supp[df_supp["supplier_id"] == supp_id]["reliability_score"].values[0])
        warehouse = random.choice(warehouses)
        weather = random.choice(weather_conds)
        
        # Calculate delay probability based on features
        base_delay_prob = 0.1
        if weather in ["Snow", "Storm"]:
            base_delay_prob += 0.3
        if supp_rel < 0.7:
            base_delay_prob += 0.2
        if warehouse["current_load"] > 0.8:
            base_delay_prob += 0.1
            
        delayed = random.random() < base_delay_prob
        
        shipments.append({
            "shipment_id": f"SH_{i}",
            "product_id": prod["product_id"],
            "warehouse_id": warehouse["warehouse_id"],
            "dispatch_date": fake.date_this_year(),
            "weather_conditions": weather,
            "delayed": int(delayed)
        })
    df_ship = pd.DataFrame(shipments)
    df_ship.to_csv(f"{output_dir}/shipments.csv", index=False)
    print(f"Synthetic data generated in '{output_dir}/' directory.")

if __name__ == "__main__":
    generate_csvs()
