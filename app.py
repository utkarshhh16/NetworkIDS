from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import pickle
import numpy as np
import pandas as pd
import threading
import os
import time
from datetime import datetime
import logging
from pathlib import Path
import subprocess
import sys

# Create folders
Path('logs').mkdir(exist_ok=True)
Path('templates').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)
Path('static/uploads').mkdir(parents=True, exist_ok=True)
Path('pcap_files').mkdir(exist_ok=True)

logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cyber-ai-live'
socketio = SocketIO(app, cors_allowed_origins="*")

MODEL = None
SCALER = None
FEATURE_NAMES = None
CAPTURE_ACTIVE = False
CICFLOW_PROCESS = None
RECENT_EVENTS = []
STATS = {'attacks': 0, 'benign': 0, 'total': 0}

# ============================================
# LOAD MODELS
# ============================================

def load_models():
    global MODEL, SCALER, FEATURE_NAMES
    
    try:
        MODEL = pickle.load(open('models/rf_intrusion_model.pkl', 'rb'))
        SCALER = pickle.load(open('models/scaler.pkl', 'rb'))
        FEATURE_NAMES = pickle.load(open('models/feature_names.pkl', 'rb'))
        
        print("✓ Model loaded")
        print(f"✓ Scaler loaded")
        print(f"✓ {len(FEATURE_NAMES)} features loaded")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

# ============================================
# PREDICTION
# ============================================

def predict_flow(flow_dict):
    """Make prediction"""
    try:
        features = []
        for feat in FEATURE_NAMES:
            features.append(float(flow_dict.get(feat, 0)))
        
        features = np.array(features).reshape(1, -1)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        features_scaled = SCALER.transform(features)
        prediction = MODEL.predict(features_scaled)[0]
        confidence = max(MODEL.predict_proba(features_scaled)[0]) * 100
        
        return {
            'prediction': int(prediction),
            'label': 'BENIGN' if prediction == 0 else 'ATTACK',
            'confidence': float(confidence)
        }
    except Exception as e:
        return None

# ============================================
# CICFLOWMETER LIVE CAPTURE
# ============================================

def cicflow_live_capture(interface):
    """Live capture with CICFlowMeter - continuously process flows"""
    global CAPTURE_ACTIVE, CICFLOW_PROCESS, STATS, RECENT_EVENTS
    
    print(f"🔴 Starting CICFlowMeter live capture on {interface}")
    
    try:
        # Output file - update continuously
        output_csv = f"static/uploads/cicflow_live_{int(time.time())}.csv"
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        
        # Command: cicflowmeter -i interface -c -u output.csv
        cmd = ['cicflowmeter', '-i', interface, '-c', '-u', output_csv]
        
        print(f"Running: {' '.join(cmd)}")
        
        # Start cicflowmeter process
        CICFLOW_PROCESS = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        print("✓ CICFlowMeter started - monitoring for flows...")
        socketio.emit('capture_started', {'status': 'Live capturing - generate traffic!'})
        
        last_lines_count = 0
        
        # Monitor CSV file for updates
        while CAPTURE_ACTIVE:
            try:
                time.sleep(1)
                
                # Check if CSV file exists and has content
                if os.path.exists(output_csv) and os.path.getsize(output_csv) > 100:
                    
                    # Read CSV
                    df = pd.read_csv(output_csv)
                    
                    # Process only new rows
                    if len(df) > last_lines_count:
                        new_rows = df.iloc[last_lines_count:]
                        
                        for idx, row in new_rows.iterrows():
                            try:
                                flow_dict = row.to_dict()
                                pred = predict_flow(flow_dict)
                                
                                if pred:
                                    event = {
                                        'timestamp': datetime.now().isoformat(),
                                        'source_ip': str(flow_dict.get('Src IP', 'N/A')),
                                        'dest_ip': str(flow_dict.get('Dst IP', 'N/A')),
                                        'dest_port': int(float(flow_dict.get('Destination Port', 0))),
                                        'protocol': int(float(flow_dict.get('Protocol', 0))),
                                        'prediction': pred['label'],
                                        'confidence': round(pred['confidence'], 2)
                                    }
                                    
                                    RECENT_EVENTS.append(event)
                                    if len(RECENT_EVENTS) > 100:
                                        RECENT_EVENTS.pop(0)
                                    
                                    if pred['label'] == 'ATTACK':
                                        STATS['attacks'] += 1
                                        print(f"🚨 ATTACK: {event['source_ip']} → {event['dest_ip']}:{event['dest_port']}")
                                    else:
                                        STATS['benign'] += 1
                                    
                                    STATS['total'] += 1
                                    
                                    # Emit to dashboard
                                    socketio.emit('live_prediction', event)
                                    socketio.emit('stats', STATS)
                            
                            except Exception as e:
                                logging.error(f"Row error: {e}")
                        
                        last_lines_count = len(df)
                        print(f"📊 Flows analyzed: {STATS['total']} | Attacks: {STATS['attacks']}")
            
            except Exception as e:
                logging.error(f"Monitor error: {e}")
        
        # Stop capture
        print("⏹ Stopping capture...")
        CICFLOW_PROCESS.terminate()
        
        try:
            CICFLOW_PROCESS.wait(timeout=5)
        except subprocess.TimeoutExpired:
            CICFLOW_PROCESS.kill()
        
        print("✓ Capture stopped")
    
    except Exception as e:
        print(f"✗ Capture error: {e}")
        socketio.emit('error', {'message': str(e)})

def cicflow_pcap_process(pcap_file):
    """Process PCAP with CICFlowMeter"""
    global STATS, RECENT_EVENTS
    
    try:
        print(f"Processing PCAP: {pcap_file}")
        socketio.emit('processing_update', {'status': 'Running CICFlowMeter...', 'progress': 30})
        
        output_csv = f"static/uploads/cicflow_pcap_{int(time.time())}.csv"
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        
        cmd = ['cicflowmeter', '-f', pcap_file, '-c', '-u', output_csv]
        
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print(f"Return code: {result.returncode}")
        
        time.sleep(2)
        
        if os.path.exists(output_csv):
            socketio.emit('processing_update', {'status': 'Analyzing flows...', 'progress': 70})
            
            df = pd.read_csv(output_csv)
            print(f"✓ Processing {len(df)} flows")
            
            for idx, row in df.iterrows():
                try:
                    flow_dict = row.to_dict()
                    pred = predict_flow(flow_dict)
                    
                    if pred:
                        event = {
                            'timestamp': datetime.now().isoformat(),
                            'source_ip': str(flow_dict.get('Src IP', 'N/A')),
                            'dest_ip': str(flow_dict.get('Dst IP', 'N/A')),
                            'dest_port': int(float(flow_dict.get('Destination Port', 0))),
                            'protocol': int(float(flow_dict.get('Protocol', 0))),
                            'prediction': pred['label'],
                            'confidence': round(pred['confidence'], 2)
                        }
                        
                        RECENT_EVENTS.append(event)
                        if len(RECENT_EVENTS) > 100:
                            RECENT_EVENTS.pop(0)
                        
                        if pred['label'] == 'ATTACK':
                            STATS['attacks'] += 1
                        else:
                            STATS['benign'] += 1
                        
                        STATS['total'] += 1
                        socketio.emit('live_prediction', event)
                
                except Exception as e:
                    pass
            
            return True
        
        return False
    
    except Exception as e:
        print(f"✗ PCAP error: {e}")
        return False

# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    return render_template('dashboard_cicflow.html')

@app.route('/api/start-live', methods=['POST'])
def start_live():
    global CAPTURE_ACTIVE
    
    if CAPTURE_ACTIVE:
        return jsonify({'error': 'Already capturing'}), 400
    
    try:
        CAPTURE_ACTIVE = True
        interface = 'Wi-Fi'
        
        print("\n" + "="*70)
        print("⚠️  IMPORTANT: Generate network traffic NOW!")
        print("   Open another terminal and run:")
        print("   ping 8.8.8.8 -t")
        print("   (Keep it running while capturing)")
        print("="*70 + "\n")
        
        thread = threading.Thread(
            target=cicflow_live_capture,
            args=(interface,),
            daemon=True
        )
        thread.start()
        
        return jsonify({'status': 'Started - Generate traffic!'}), 200
    
    except Exception as e:
        CAPTURE_ACTIVE = False
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-live', methods=['POST'])
def stop_live():
    global CAPTURE_ACTIVE, CICFLOW_PROCESS
    
    CAPTURE_ACTIVE = False
    
    if CICFLOW_PROCESS:
        CICFLOW_PROCESS.terminate()
    
    socketio.emit('capture_stopped', {'status': 'Stopped'})
    return jsonify({'status': 'Stopped'}), 200

@app.route('/api/upload-pcap', methods=['POST'])
def upload_pcap():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    
    if not file.filename.endswith('.pcap'):
        return jsonify({'error': 'Only .pcap files'}), 400
    
    try:
        filename = f"upload_{int(time.time())}.pcap"
        filepath = os.path.abspath(os.path.join('pcap_files', filename))
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file.save(filepath)
        
        thread = threading.Thread(
            target=process_pcap_bg,
            args=(filepath,),
            daemon=True
        )
        thread.start()
        
        return jsonify({'message': 'Processing...'}), 202
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_pcap_bg(pcap_file):
    global STATS
    
    try:
        STATS = {'attacks': 0, 'benign': 0, 'total': 0}
        
        success = cicflow_pcap_process(pcap_file)
        
        if success:
            socketio.emit('processing_update', {'status': 'Complete!', 'progress': 100})
            socketio.emit('processing_complete', {
                'total_flows': STATS['total'],
                'attacks': STATS['attacks'],
                'benign': STATS['benign']
            })
    
    except Exception as e:
        socketio.emit('processing_error', {'error': str(e)})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify(STATS)

@app.route('/api/events', methods=['GET'])
def get_events():
    limit = request.args.get('limit', 50, type=int)
    return jsonify(RECENT_EVENTS[-limit:])

@socketio.on('connect')
def handle_connect():
    emit('stats', STATS)

# ============================================
# STARTUP
# ============================================

if __name__ == '__main__':
    if load_models():
        print("\n" + "="*70)
        print("✓ CyberAI IDS - CICFlowMeter Live Capture")
        print("✓ Ready for live traffic analysis")
        print("="*70 + "\n")
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    else:
        print("✗ Failed to load models")
