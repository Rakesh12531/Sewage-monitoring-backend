"""
ULTIMATE FIX - Adds Diverse Normal Data + Strong Risk 1 Examples
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

print("="*70)
print("  🤖 ULTIMATE FIX - FINAL AI TRAINER")
print("="*70)

# Load your normal data
df = pd.read_csv('sensor_readings_rows.csv')
print(f"\n✅ Loaded {len(df)} normal records")

np.random.seed(42)

# CRITICAL FIX: Add more diverse NORMAL data with varying water levels
print("\n🔧 Creating diverse NORMAL scenarios (Risk 0)...")
normal_variations = []

# Keep original normal data
normal_data = df[['temperature', 'mq135_ppm', 'mq4_ppm', 'water_level', 'flow_rate', 'risk_level']].copy()

# Add normal data with water levels between 0-69% (below threshold)
for _ in range(200):
    normal_variations.append({
        'temperature': np.random.uniform(10, 16),       # Normal temp range
        'mq135_ppm': np.random.uniform(500, 3000),      # Normal MQ135 range
        'mq4_ppm': np.random.uniform(0.5, 2.5),         # Normal MQ4 range
        'water_level': np.random.uniform(0, 69),        # BELOW 70% threshold
        'flow_rate': np.random.uniform(0, 39),          # BELOW 40 threshold
        'risk_level': 0
    })

print(f"✅ Added {len(normal_variations)} diverse normal samples")

# RISK 1: ONLY water level high (like your real case)
print("\n🔧 Creating Risk 1 scenarios - WATER ONLY HIGH...")
risk1_water_only = []

for _ in range(150):  # LOTS of water-only samples
    risk1_water_only.append({
        'temperature': np.random.uniform(10, 16),       # Normal
        'mq135_ppm': np.random.uniform(500, 3000),      # Normal
        'mq4_ppm': np.random.uniform(0.5, 2.5),         # Normal
        'water_level': np.random.uniform(70, 99),       # HIGH (70-99%)
        'flow_rate': np.random.uniform(0, 15),          # Normal/low
        'risk_level': 1
    })

print(f"✅ Created {len(risk1_water_only)} water-only Risk 1 samples")

# RISK 1: Other single-sensor scenarios
risk1_other = []

# Only temp high
for _ in range(25):
    risk1_other.append({
        'temperature': np.random.uniform(41, 55),
        'mq135_ppm': np.random.uniform(500, 3000),
        'mq4_ppm': np.random.uniform(0.5, 2.5),
        'water_level': np.random.uniform(0, 69),
        'flow_rate': np.random.uniform(0, 39),
        'risk_level': 1
    })

# Only MQ135 high
for _ in range(25):
    risk1_other.append({
        'temperature': np.random.uniform(10, 16),
        'mq135_ppm': np.random.uniform(12500, 20000),
        'mq4_ppm': np.random.uniform(0.5, 2.5),
        'water_level': np.random.uniform(0, 69),
        'flow_rate': np.random.uniform(0, 39),
        'risk_level': 1
    })

# Only MQ4 high
for _ in range(15):
    risk1_other.append({
        'temperature': np.random.uniform(10, 16),
        'mq135_ppm': np.random.uniform(500, 3000),
        'mq4_ppm': np.random.uniform(3.5, 8.0),
        'water_level': np.random.uniform(0, 69),
        'flow_rate': np.random.uniform(0, 39),
        'risk_level': 1
    })

# Only flow high
for _ in range(15):
    risk1_other.append({
        'temperature': np.random.uniform(10, 16),
        'mq135_ppm': np.random.uniform(500, 3000),
        'mq4_ppm': np.random.uniform(0.5, 2.5),
        'water_level': np.random.uniform(0, 69),
        'flow_rate': np.random.uniform(41, 65),
        'risk_level': 1
    })

print(f"✅ Created {len(risk1_other)} other Risk 1 samples")

# RISK 2: Two sensors high
risk2_data = []
print("\n🔧 Creating Risk 2 scenarios...")

for _ in range(20):
    # Temp + Water
    risk2_data.append({
        'temperature': np.random.uniform(41, 60),
        'mq135_ppm': np.random.uniform(500, 3000),
        'mq4_ppm': np.random.uniform(0.5, 2.5),
        'water_level': np.random.uniform(70, 90),
        'flow_rate': np.random.uniform(0, 39),
        'risk_level': 2
    })

for _ in range(20):
    # MQ135 + Water
    risk2_data.append({
        'temperature': np.random.uniform(10, 16),
        'mq135_ppm': np.random.uniform(12500, 22000),
        'mq4_ppm': np.random.uniform(0.5, 2.5),
        'water_level': np.random.uniform(70, 90),
        'flow_rate': np.random.uniform(0, 39),
        'risk_level': 2
    })

for _ in range(20):
    # Water + Flow
    risk2_data.append({
        'temperature': np.random.uniform(10, 16),
        'mq135_ppm': np.random.uniform(500, 3000),
        'mq4_ppm': np.random.uniform(0.5, 2.5),
        'water_level': np.random.uniform(70, 90),
        'flow_rate': np.random.uniform(41, 70),
        'risk_level': 2
    })

for _ in range(20):
    # Temp + MQ135
    risk2_data.append({
        'temperature': np.random.uniform(41, 60),
        'mq135_ppm': np.random.uniform(12500, 22000),
        'mq4_ppm': np.random.uniform(0.5, 2.5),
        'water_level': np.random.uniform(0, 69),
        'flow_rate': np.random.uniform(0, 39),
        'risk_level': 2
    })

print(f"✅ Created {len(risk2_data)} Risk 2 samples")

# RISK 3: Multiple sensors critical
risk3_data = []
print("\n🔧 Creating Risk 3 scenarios...")
for _ in range(50):
    risk3_data.append({
        'temperature': np.random.uniform(45, 65),
        'mq135_ppm': np.random.uniform(15000, 25000),
        'mq4_ppm': np.random.uniform(5.0, 12.0),
        'water_level': np.random.uniform(85, 100),
        'flow_rate': np.random.uniform(50, 80),
        'risk_level': 3
    })

print(f"✅ Created {len(risk3_data)} Risk 3 samples")

# Combine ALL data
all_data = pd.concat([
    normal_data,
    pd.DataFrame(normal_variations),
    pd.DataFrame(risk1_water_only),
    pd.DataFrame(risk1_other),
    pd.DataFrame(risk2_data),
    pd.DataFrame(risk3_data)
], ignore_index=True)

print(f"\n{'='*70}")
print("📊 FINAL ENHANCED DATASET")
print('='*70)
print(f"Total: {len(all_data)} samples\n")
for risk in sorted(all_data['risk_level'].unique()):
    count = (all_data['risk_level'] == risk).sum()
    pct = (count / len(all_data)) * 100
    bar = '█' * int(pct / 2)
    print(f"Risk {risk}: {count:4d} ({pct:5.1f}%) {bar}")

# Train model
X = all_data[['temperature', 'mq135_ppm', 'mq4_ppm', 'water_level', 'flow_rate']].values
y = all_data['risk_level'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n{'='*70}")
print("🌲 TRAINING ENHANCED RANDOM FOREST")
print('='*70)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

train_acc = model.score(X_train_scaled, y_train) * 100
test_acc = model.score(X_test_scaled, y_test) * 100

print(f"\n📈 Training Accuracy: {train_acc:.2f}%")
print(f"📈 Testing Accuracy:  {test_acc:.2f}%")

# Feature importance
features = ['Temperature', 'MQ135', 'MQ4', 'Water Level', 'Flow Rate']
importance = model.feature_importances_
print(f"\n{'='*70}")
print("🔍 FEATURE IMPORTANCE")
print('='*70)
for feat, imp in sorted(zip(features, importance), key=lambda x: x[1], reverse=True):
    bar = '█' * int(imp * 60)
    print(f"{feat:.<20} {imp:.4f} {bar}")

# Save
joblib.dump(model, 'risk_model.pkl')
joblib.dump(scaler, 'risk_scaler.pkl')
print(f"\n💾 Models saved!")

# TEST WITH YOUR EXACT DATA
print(f"\n{'='*70}")
print("🧪 FINAL TEST WITH YOUR REAL DATA")
print('='*70)

test_cases = [
    {
        'name': 'Your Real Data (Water=85.14%)',
        'data': [12.92, 1278.51, 1.51, 85.14, 0.0],
        'expected': 1
    },
    {
        'name': 'All Normal',
        'data': [12.0, 1000, 1.3, 50, 0],
        'expected': 0
    },
    {
        'name': 'Water at 69% (just below threshold)',
        'data': [12.0, 1000, 1.3, 69, 0],
        'expected': 0
    },
    {
        'name': 'Water at 70% (at threshold)',
        'data': [12.0, 1000, 1.3, 70, 0],
        'expected': 1
    },
]

for test in test_cases:
    test_input = np.array([test['data']])
    test_scaled = scaler.transform(test_input)
    prediction = model.predict(test_scaled)[0]
    proba = model.predict_proba(test_scaled)[0]
    
    print(f"\n{test['name']}:")
    print(f"  Input: {test['data']}")
    print(f"  🤖 Predicted: Risk {prediction}")
    print(f"  🎯 Expected: Risk {test['expected']}")
    print(f"  {'✅ CORRECT' if prediction == test['expected'] else '❌ WRONG'}")
    print(f"  📊 Confidence: {max(proba)*100:.1f}%")

print(f"\n{'='*70}")
print("✅ TRAINING COMPLETE - MODEL READY!")
print('='*70)

input("\nPress Enter to exit...")
