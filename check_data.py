import pandas as pd
from sqlalchemy import create_engine, text
from atlas_ml.config import Config

cfg = Config()
engine = create_engine(
    f'postgresql://{cfg.database.user}:{cfg.database.password}@'
    f'{cfg.database.host}:{cfg.database.port}/{cfg.database.database}'
)

with engine.connect() as conn:
    # Check n_trips_logistic distribution
    print("=== N_TRIPS_LOGISTIC DISTRIBUTION (TOP 20) ===")
    df = pd.read_sql(text("""
        SELECT n_trips_logistic, COUNT(*) as count 
        FROM app.sodd_loads_filtered 
        GROUP BY n_trips_logistic 
        ORDER BY n_trips_logistic DESC 
        LIMIT 20
    """), conn)
    print(df.to_string(index=False))
    
    # Check stats
    print("\n=== N_TRIPS_LOGISTIC STATISTICS ===")
    stats = pd.read_sql(text("""
        SELECT 
            MIN(n_trips_logistic) as min,
            MAX(n_trips_logistic) as max,
            AVG(n_trips_logistic) as mean,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY n_trips_logistic) as median,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY n_trips_logistic) as p95,
            STDDEV(n_trips_logistic) as std
        FROM app.sodd_loads_filtered
    """), conn)
    print(stats.to_string(index=False))
    
    # Check specific OD pairs
    print("\n=== SPECIFIC OD PAIRS ===")
    pairs = pd.read_sql(text("""
        SELECT origin_id, destination_id, n_trips_logistic
        FROM app.sodd_loads_filtered
        WHERE (origin_id = 8205 AND destination_id = 8019)
           OR (origin_id = 8029 AND destination_id = 17013)
           OR (origin_id = 8279 AND destination_id = 8019)
        LIMIT 10
    """), conn)
    print(pairs.to_string(index=False))
