import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SyntheticTimeSeriesLoader:
    def __init__(self, ratio_non_drifting=0.5, n_features=2):
        self.q = ratio_non_drifting
        self.n_features = n_features
            
    def take(self, size):
        t = np.arange(size)
        
        # Trigger drift exactly halfway through
        os_flag = (t >= size // 2).astype(int)
        
        # Assign populations: 0 = stationary, 1 = drifting
        z = np.random.choice([0, 1], size=size, p=[self.q, 1 - self.q])
        ds = z.copy() 
        
        X = np.zeros((size, self.n_features))
        
        for i in range(size):
            if z[i] == 0:
                # Stationary Subpopulation: Stays at -2.0 forever
                X[i] = np.random.normal(loc=-2.0, scale=1.0, size=self.n_features)
            else:
                # Drifting Subpopulation: Shifts from 2.0 to 6.0 at halfway mark
                if os_flag[i] == 0:
                    X[i] = np.random.normal(loc=2.0, scale=1.0, size=self.n_features) # Before
                else:
                    X[i] = np.random.normal(loc=6.0, scale=1.0, size=self.n_features) # After
                    
        return X, os_flag, ds, z

print("⚙️ Generating Synthetic Time-Series Data...")
loader = SyntheticTimeSeriesLoader(ratio_non_drifting=0.5, n_features=2)

# Generate 1000 data points
X, os_flag, ds, z = loader.take(1000)

# 1. Save to a CSV so you can read it
df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
df['Time_Step'] = np.arange(1000)
df['Observation_State'] = os_flag       # 0 = Before Drift Event, 1 = After Drift Event
df['Is_Drifting_Population'] = ds       # 0 = Stationary Point, 1 = Drifting Point


df.to_csv(f"data/synthetic_drift_data.csv", index=False)
print("✅ Data saved to 'synthetic_drift_data.csv'")

# 2. Plot the data to visualize the drift
plt.figure(figsize=(14, 6))

# Plot Feature 1 over time
plt.scatter(df['Time_Step'][df['Is_Drifting_Population'] == 0], 
            df['Feature_1'][df['Is_Drifting_Population'] == 0], 
            alpha=0.5, label='Stationary Population (Stays ~ -2.0)', color='blue', s=15)

plt.scatter(df['Time_Step'][df['Is_Drifting_Population'] == 1], 
            df['Feature_1'][df['Is_Drifting_Population'] == 1], 
            alpha=0.5, label='Drifting Population (Shifts from 2.0 -> 6.0)', color='red', s=15)

# Draw a line where the drift occurs
plt.axvline(x=500, color='black', linestyle='--', linewidth=2, label='Drift Trigger (Time=500)')

plt.title('Synthetic Concept Drift Over Time (Feature 1)')
plt.xlabel('Time Step')
plt.ylabel('Feature Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()