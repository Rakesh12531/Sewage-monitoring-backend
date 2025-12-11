"""
COMPLETE SEWAGE MONITORING SYSTEM - PRODUCTION READY
Uses existing Supabase tables + MQTT + AI + Web Dashboard
"""

import paho.mqtt.client as mqtt
import json
from supabase import create_client
from flask import Flask, jsonify, render_template_string
from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import threading
import time

# ==================== CONFIGURATION ====================
SUPABASE_URL = "https://ritbhmmrsdfmugfmndth.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJpdGJobW1yc2RmbXVnZm1uZHRoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjUzNzY4NTYsImV4cCI6MjA4MDk1Mjg1Nn0.QrMR9ETJWTzfivHsjffRImtfXofUH_K7OCEVWjOBz28"

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "sewage/sensor/data"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
app = Flask(__name__)

# ==================== MQTT HANDLER ====================
class MQTTBridge:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.latest_data = {}
        self.message_count = 0
    
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("✅ MQTT Connected")
            client.subscribe(MQTT_TOPIC)
            print(f"📡 Listening to: {MQTT_TOPIC}")
        else:
            print(f"❌ MQTT Failed: {rc}")
    
    def on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            self.latest_data = data
            self.message_count += 1
            
            # Store in Supabase
            supabase.table("sensor_readings").insert({
                "temperature": data.get("temp"),
                "mq135_ppm": data.get("mq135_ppm"),
                "mq4_ppm": data.get("mq4_ppm"),
                "water_level": data.get("waterLevel"),
                "flow_rate": data.get("flow"),
                "risk_level": data.get("risk"),
                "warmed_up": data.get("warmedUp") == "true"
            }).execute()
            
            # Create alert if needed
            if data.get("risk", 0) >= 2:
                self.create_alert(data.get("risk"))
            
            print(f"✅ [{datetime.now().strftime('%H:%M:%S')}] #{self.message_count} | Risk: {data.get('risk')} | Water: {data.get('waterLevel'):.1f}%")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def create_alert(self, risk_level):
        """Create alert in database"""
        messages = {
            1: "⚠️ Warning: 1 sensor exceeded threshold",
            2: "🚨 Alert: 2 sensors exceeded threshold",
            3: "🆘 CRITICAL: 3+ sensors exceeded - Immediate action!"
        }
        
        try:
            supabase.table("alerts").insert({
                "alert_type": "THRESHOLD_EXCEEDED",
                "severity": risk_level,
                "message": messages.get(risk_level, "Unknown risk"),
                "resolved": False
            }).execute()
        except:
            pass
    
    def start(self):
        self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
        self.client.loop_start()

mqtt_bridge = MQTTBridge()

# ==================== AI RISK PREDICTOR ====================
class RiskPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_file = 'risk_model.pkl'
        self.scaler_file = 'risk_scaler.pkl'
    
    def train(self):
        """Train AI model on historical data"""
        print("\n🤖 TRAINING AI MODEL...")
        print("="*60)
        
        # Fetch data
        response = supabase.table("sensor_readings")\
            .select("*")\
            .eq("warmed_up", True)\
            .order("timestamp", desc=True)\
            .limit(500)\
            .execute()
        
        if len(response.data) < 50:
            print(f"❌ Need 50+ records. Currently have: {len(response.data)}")
            print("💡 Let the system run for 10-15 minutes to collect data")
            return False
        
        df = pd.DataFrame(response.data)
        print(f"✅ Loaded {len(df)} records")
        
        # Prepare features
        X = df[['temperature', 'mq135_ppm', 'mq4_ppm', 'water_level', 'flow_rate']].fillna(0)
        y = df['risk_level']
        
        print(f"\n📊 Risk Distribution:")
        for risk in sorted(y.unique()):
            count = (y == risk).sum()
            print(f"  Risk {risk}: {count} records ({count/len(y)*100:.1f}%)")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        
        # Evaluate
        accuracy = self.model.score(X_scaled, y)
        print(f"\n✅ Model Trained!")
        print(f"📈 Training Accuracy: {accuracy*100:.1f}%")
        
        # Feature importance
        features = ['Temperature', 'MQ135', 'MQ4', 'Water Level', 'Flow Rate']
        importance = self.model.feature_importances_
        print(f"\n🔍 Feature Importance:")
        for feat, imp in sorted(zip(features, importance), key=lambda x: x[1], reverse=True):
            bar = '█' * int(imp * 50)
            print(f"  {feat:.<20} {imp:.3f} {bar}")
        
        # Save model
        joblib.dump(self.model, self.model_file)
        joblib.dump(self.scaler, self.scaler_file)
        print(f"\n💾 Model saved to {self.model_file}")
        
        return True
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            self.model = joblib.load(self.model_file)
            self.scaler = joblib.load(self.scaler_file)
            return True
        except FileNotFoundError:
            return False
    
    def predict(self):
        """Predict risk for latest sensor data"""
        # Load model if not loaded
        if not self.model:
            if not self.load_model():
                return {"error": "Model not trained yet. Train first!"}
        
        # Get latest reading
        response = supabase.table("sensor_readings")\
            .select("*")\
            .order("timestamp", desc=True)\
            .limit(1)\
            .execute()
        
        if not response.data:
            return {"error": "No sensor data available"}
        
        latest = response.data[0]
        
        # Prepare features
        features = [[
            latest['temperature'],
            latest['mq135_ppm'],
            latest['mq4_ppm'],
            latest['water_level'],
            latest['flow_rate']
        ]]
        
        # Predict
        features_scaled = self.scaler.transform(features)
        predicted_risk = int(self.model.predict(features_scaled)[0])
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = float(max(probabilities))
        
        # Store prediction
        try:
            supabase.table("overflow_predictions").insert({
                "predicted_water_level": latest['water_level'],
                "prediction_time_minutes": 5,
                "confidence_score": confidence,
                "anomaly_detected": predicted_risk > latest['risk_level'],
                "anomaly_type": "AI_RISK_MISMATCH" if predicted_risk != latest['risk_level'] else None
            }).execute()
        except:
            pass
        
        return {
            "predicted_risk": predicted_risk,
            "actual_risk": latest['risk_level'],
            "confidence": confidence,
            "probabilities": {f"Risk_{i}": float(p) for i, p in enumerate(probabilities)},
            "match": predicted_risk == latest['risk_level']
        }

ai_predictor = RiskPredictor()

# ==================== WEB DASHBOARD ====================
@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/latest')
def api_latest():
    response = supabase.table("sensor_readings")\
        .select("*")\
        .order("timestamp", desc=True)\
        .limit(1)\
        .execute()
    return jsonify(response.data[0] if response.data else {})

@app.route('/api/history')
def api_history():
    response = supabase.table("sensor_readings")\
        .select("*")\
        .order("timestamp", desc=True)\
        .limit(100)\
        .execute()
    return jsonify(response.data)

@app.route('/api/alerts')
def api_alerts():
    response = supabase.table("alerts")\
        .select("*")\
        .eq("resolved", False)\
        .order("timestamp", desc=True)\
        .limit(10)\
        .execute()
    return jsonify(response.data)

@app.route('/api/predict')
def api_predict():
    result = ai_predictor.predict()
    return jsonify(result)

@app.route('/api/stats')
def api_stats():
    response = supabase.table("sensor_readings")\
        .select("*")\
        .order("timestamp", desc=True)\
        .limit(100)\
        .execute()
    
    if not response.data:
        return jsonify({"total": 0})
    
    df = pd.DataFrame(response.data)
    return jsonify({
        "total_records": len(df),
        "avg_temp": float(df['temperature'].mean()),
        "avg_water_level": float(df['water_level'].mean()),
        "max_risk": int(df['risk_level'].max()),
        "warmed_up_pct": float((df['warmed_up'].sum() / len(df)) * 100)
    })

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Smart Sewage Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            padding: 20px;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { 
            text-align: center; 
            margin-bottom: 30px; 
            font-size: 2.5em; 
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin-bottom: 30px;
        }
        .card {
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
            border: 1px solid rgba(255,255,255,0.3);
            transition: transform 0.3s;
        }
        .card:hover { transform: translateY(-5px); }
        .card h3 { 
            margin-bottom: 15px; 
            font-size: 1.1em; 
            opacity: 0.9; 
            display: flex;
            align-items: center;
        }
        .value { 
            font-size: 3em; 
            font-weight: bold; 
            margin: 10px 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .unit { font-size: 0.9em; opacity: 0.8; }
        .risk-0 { color: #4ade80; }
        .risk-1 { color: #fbbf24; }
        .risk-2 { color: #fb923c; }
        .risk-3 { color: #ef4444; }
        .btn-group {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin: 30px 0;
            flex-wrap: wrap;
        }
        .btn {
            background: rgba(255,255,255,0.25);
            border: 2px solid rgba(255,255,255,0.4);
            padding: 15px 35px;
            border-radius: 10px;
            color: white;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .btn:hover { 
            background: rgba(255,255,255,0.35); 
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        #ai-result {
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            padding: 30px;
            margin: 20px 0;
            text-align: center;
            font-size: 1.3em;
            line-height: 1.8;
            min-height: 100px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .timestamp { 
            text-align: center; 
            opacity: 0.7; 
            margin: 20px 0;
            font-size: 0.95em;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .stat-card {
            background: rgba(0,0,0,0.2);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .alerts {
            background: rgba(255,0,0,0.2);
            border-left: 4px solid #ef4444;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .live-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #4ade80;
            border-radius: 50%;
            margin-right: 10px;
            animation: pulse 2s infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚰 Smart Sewage Monitoring System</h1>
        
        <div class="grid">
            <div class="card">
                <h3><span class="live-indicator"></span>🌡️ Temperature</h3>
                <div class="value" id="temp">--</div>
                <div class="unit">°C</div>
            </div>
            
            <div class="card">
                <h3>💨 MQ135 (NH3/CO2)</h3>
                <div class="value" id="mq135">--</div>
                <div class="unit">PPM</div>
            </div>
            
            <div class="card">
                <h3>🔥 MQ4 (Methane)</h3>
                <div class="value" id="mq4">--</div>
                <div class="unit">PPM</div>
            </div>
            
            <div class="card">
                <h3>💧 Water Level</h3>
                <div class="value" id="water">--</div>
                <div class="unit">%</div>
            </div>
            
            <div class="card">
                <h3>🌊 Flow Rate</h3>
                <div class="value" id="flow">--</div>
                <div class="unit">LPM</div>
            </div>
            
            <div class="card">
                <h3>⚠️ Risk Level</h3>
                <div class="value" id="risk">--</div>
                <div class="unit">0-3 Scale</div>
            </div>
        </div>
        
        <div class="btn-group">
            <button class="btn" onclick="predictRisk()">🤖 AI Risk Prediction</button>
            <button class="btn" onclick="loadStats()">📊 View Statistics</button>
            <button class="btn" onclick="checkAlerts()">🚨 Check Alerts</button>
        </div>
        
        <div id="ai-result">Click "AI Risk Prediction" to analyze current conditions</div>
        
        <div id="alerts-container"></div>
        
        <div class="timestamp" id="timestamp">Waiting for data...</div>
    </div>
    
    <script>
        function updateData() {
            fetch('/api/latest')
                .then(r => r.json())
                .then(data => {
                    if (!data.id) return;
                    
                    document.getElementById('temp').textContent = (data.temperature || 0).toFixed(1);
                    document.getElementById('mq135').textContent = (data.mq135_ppm || 0).toFixed(0);
                    document.getElementById('mq4').textContent = (data.mq4_ppm || 0).toFixed(2);
                    document.getElementById('water').textContent = (data.water_level || 0).toFixed(1);
                    document.getElementById('flow').textContent = (data.flow_rate || 0).toFixed(1);
                    
                    const risk = data.risk_level || 0;
                    const riskEl = document.getElementById('risk');
                    riskEl.textContent = risk;
                    riskEl.className = 'value risk-' + risk;
                    
                    const time = new Date(data.timestamp);
                    document.getElementById('timestamp').textContent = 
                        '🕐 Last updated: ' + time.toLocaleString() + 
                        (data.warmed_up ? ' ✅ Sensors Ready' : ' ⏳ Warming up...');
                });
        }
        
        function predictRisk() {
            document.getElementById('ai-result').innerHTML = '🔄 Analyzing...';
            
            fetch('/api/predict')
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('ai-result').innerHTML = 
                            '⚠️ ' + data.error;
                        return;
                    }
                    
                    const match = data.match ? '✅ MATCH' : '⚠️ MISMATCH';
                    const confidence = (data.confidence * 100).toFixed(1);
                    
                    let html = `
                        <div>
                            <div style="font-size:1.5em; margin-bottom:15px;">
                                🤖 AI Prediction: <span class="risk-${data.predicted_risk}">Risk ${data.predicted_risk}</span>
                            </div>
                            <div>
                                🎯 Actual (ESP32): Risk ${data.actual_risk} | ${match}
                            </div>
                            <div style="margin-top:10px;">
                                📊 Confidence: ${confidence}%
                            </div>
                        </div>
                    `;
                    
                    document.getElementById('ai-result').innerHTML = html;
                });
        }
        
        function loadStats() {
            fetch('/api/stats')
                .then(r => r.json())
                .then(data => {
                    const html = `
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div>📊 Total Records</div>
                                <div class="stat-value">${data.total_records || 0}</div>
                            </div>
                            <div class="stat-card">
                                <div>🌡️ Avg Temperature</div>
                                <div class="stat-value">${(data.avg_temp || 0).toFixed(1)}°C</div>
                            </div>
                            <div class="stat-card">
                                <div>💧 Avg Water Level</div>
                                <div class="stat-value">${(data.avg_water_level || 0).toFixed(1)}%</div>
                            </div>
                            <div class="stat-card">
                                <div>⚠️ Max Risk Seen</div>
                                <div class="stat-value risk-${data.max_risk || 0}">${data.max_risk || 0}</div>
                            </div>
                        </div>
                    `;
                    document.getElementById('ai-result').innerHTML = html;
                });
        }
        
        function checkAlerts() {
            fetch('/api/alerts')
                .then(r => r.json())
                .then(alerts => {
                    if (alerts.length === 0) {
                        document.getElementById('ai-result').innerHTML = 
                            '✅ No active alerts. System operating normally.';
                        return;
                    }
                    
                    let html = '<div class="alerts"><h3>🚨 Active Alerts</h3>';
                    alerts.forEach(alert => {
                        const time = new Date(alert.timestamp).toLocaleTimeString();
                        html += `
                            <div style="margin:10px 0; padding:10px; background:rgba(0,0,0,0.2); border-radius:5px;">
                                <strong>Severity ${alert.severity}</strong> - ${time}<br>
                                ${alert.message}
                            </div>
                        `;
                    });
                    html += '</div>';
                    document.getElementById('ai-result').innerHTML = html;
                });
        }
        
        // Auto-update every 3 seconds
        updateData();
        setInterval(updateData, 3000);
    </script>
</body>
</html>
"""

# ==================== FLASK SERVER ====================
def run_flask():
    print("🌐 Starting web server...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# ==================== MAIN PROGRAM ====================
def main():
    print("="*70)
    print("  🚰 SMART SEWAGE MONITORING SYSTEM - FINAL VERSION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check Supabase connection
    print("🔌 Checking Supabase connection...")
    try:
        test = supabase.table("sensor_readings").select("id").limit(1).execute()
        print("✅ Supabase Connected Successfully!\n")
    except Exception as e:
        print(f"❌ Supabase Error: {e}")
        print("\n⚠️ Check your SUPABASE_URL and SUPABASE_KEY!")
        input("Press Enter to exit...")
        return
    
    # Start MQTT Bridge
    print("📡 Starting MQTT Bridge...")
    mqtt_bridge.start()
    time.sleep(2)
    print("✅ MQTT Bridge Running\n")
    
    # Start Flask in background
    print("🌐 Starting Web Dashboard...")
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    time.sleep(2)
    print("✅ Web Dashboard Running\n")
    
    print("="*70)
    print("✅ SYSTEM FULLY OPERATIONAL!")
    print("="*70)
    print(f"\n📊 Dashboard URL: http://localhost:5000")
    print(f"📡 MQTT Topic: {MQTT_TOPIC}")
    print(f"💡 Ensure ESP32 is running and publishing data\n")
    
    # Check current data count
    try:
        response = supabase.table("sensor_readings").select("*").execute()
        record_count = len(response.data)
        print(f"📊 Current database records: {record_count}\n")
    except:
        record_count = 0
        print("📊 Database is empty - waiting for data...\n")
    
    # Show menu
    print("="*70)
    print("TRAINING OPTIONS:")
    print("="*70)
    print("1. Train AI Model (requires 50+ sensor readings)")
    print("2. Just Monitor (collect data and view dashboard)")
    print("="*70)
    
    choice = input("\n👉 Select option (1 or 2): ").strip()
    
    if choice == "1":
        print("\n" + "="*70)
        print("🤖 STARTING AI MODEL TRAINING")
        print("="*70)
        success = ai_predictor.train()
        
        if success:
            print("\n" + "="*70)
            print("✅ TRAINING COMPLETE!")
            print("="*70)
            print("\n📊 Next Steps:")
            print("   1. Open: http://localhost:5000")
            print("   2. Click: 'AI Risk Prediction' button")
            print("   3. View AI predictions in real-time\n")
        else:
            print("\n" + "="*70)
            print("⚠️  TRAINING FAILED - Not Enough Data")
            print("="*70)
            print("\n💡 Solution:")
            print("   1. Let ESP32 run for 10-15 minutes")
            print("   2. Restart this program")
            print("   3. Try training again\n")
    
    elif choice == "2":
        print("\n" + "="*70)
        print("✅ MONITORING MODE ACTIVE")
        print("="*70)
        print("\n📊 Dashboard: http://localhost:5000")
        print("💾 Data is being saved to Supabase automatically")
        print("⏳ Collect 50+ readings, then restart and select option 1\n")
    
    else:
        print("\n❌ Invalid option. Defaulting to Monitoring Mode...")
        print("\n📊 Dashboard: http://localhost:5000\n")
    
    print("="*70)
    print("⌨️  Press Ctrl+C to stop the system")
    print("="*70)
    print()
    
    # Keep program running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("👋 SHUTTING DOWN SYSTEM")
        print("="*70)
        print("\nGoodbye! System stopped successfully.\n")

if __name__ == "__main__":
    main()
