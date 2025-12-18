import sys
import pandas as pd
import numpy as np
# import xgboost as xgb 
import time as tm

# Placeholder model class for trained model
class DummyModel:
    def predict(self, feature_vector):
        return {
            "prop_participated": 0, 
            "is_prop_buy": 0, 
            "prop_price": 0.0,
            "prop_qty": 0.0
        }

class MarketPredictor:
    def __init__(self, date, lob_file, model_path=None):
        self.date = date
        self.lob_file = lob_file
        self.output_file = f"LOB/LOB_Predicted_{date}.csv"
        # self.model.load_model(model_path) if model_path else None
        self.model = DummyModel() 
        
    def run(self):
        print(f"Loading LOB file: {self.lob_file}")
        df = pd.read_csv(self.lob_file)
        
        prop_cols = ['prop_participated', 'is_prop_buy', 'prop_price', 'prop_qty']
        prev_prop_state = df.loc[0, prop_cols].to_dict()
        
        results = []
        
        results.append(df.iloc[0].to_dict())
        
        print("Starting Prediction Loop...")
        start_time = tm.time()
      
        count = 0
        N = len(df)
        
        for idx in range(1, N): 
            row = df.iloc[idx].copy() 
            # prev_row = results[-1].copy()
            prev_row = df.iloc[idx-1].copy() 
            prediction = self.model.predict(prev_row)
            
            row['prop_participated'] = prediction['prop_participated']
            row['is_prop_buy'] = prediction['is_prop_buy']
            row['prop_price'] = prediction['prop_price']
            row['prop_qty'] = prediction['prop_qty']
            
            results.append(row.to_dict())
            
            count += 1
            if count % 5000 == 0:
                print(f"Processed period {idx}/{N}")
                
        elapsed = tm.time() - start_time
        print(f"Prediction Complete. Elapsed: {elapsed:.2f}s")
        
        # Save Output
        print(f"Saving to {self.output_file}...")
        pd.DataFrame(results).to_csv(self.output_file, index=False)
        print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simulator.py <DATE_DDMMYYYY>")
        sys.exit(1)
        
    date = sys.argv[1]
    # Assuming LOB file exists
    lob_file = f"LOB/LOB_{date}.csv"
    
    predictor = MarketPredictor(date, lob_file)
    predictor.run()
