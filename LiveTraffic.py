import collections

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import pickle
import joblib
import numpy as np
import pandas as pd
import threading
import os
import time
from datetime import datetime
import logging
from pathlib import Path
import sys

from scapy.sendrecv import sniff
from scapy.utils import rdpcap

if sys.version_info >= (3, 0):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from scapy.layers.inet import IP, TCP, UDP, ICMPfrom
        collections
        import defaultdict

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
app.config['SECRET_KEY'] = 'cyber-ai-working'
socketio = SocketIO(app, cors_allowed_origins="*")

MODEL = None
SCALER = None
LABEL_ENCODER = None
FEATURE_NAMES = None
CAPTURE_ACTIVE = False
RECENT_EVENTS = []
STATS = {'attacks': 0, 'benign': 0, 'total': 0}

# ============================================
# LOAD MODELS
# ============================================

def load_models():
    global MODEL, SCALER, LABEL_ENCODER, FEATURE_NAMES
    
    try:
        # Load using joblib (same as your dashboard)
        MODEL = joblib.load('models/rf_intrusion_model.pkl')
        SCALER = joblib.load('models/scaler.pkl')
        LABEL_ENCODER = joblib.load('models/label_encoder.pkl')
        
        # Try to load feature names, if not available, get from scaler
        try:
            FEATURE_NAMES = joblib.load('models/feature_names.pkl')
        except:
            # If feature_names.pkl doesn't exist, try to get from scaler
            if hasattr(SCALER, 'feature_names_in_'):
                FEATURE_NAMES = list(SCALER.feature_names_in_)
            else:
                print("⚠ Warning: feature_names.pkl not found. Features must match training order!")
                FEATURE_NAMES = None
        
        print("✓ Model loaded")
        print(f"✓ Scaler loaded")
        print(f"✓ Label Encoder loaded with classes: {LABEL_ENCODER.classes_}")
        if FEATURE_NAMES:
            print(f"✓ {len(FEATURE_NAMES)} features loaded")
        return True
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False

# ============================================
# FEATURE EXTRACTION FROM PACKETS
# ============================================

class FlowAnalyzer:
    """Extract network flow features matching your training data"""
    
    def __init__(self):
        self.flows = defaultdict(lambda: {
            'forward_packets': [],
            'backward_packets': [],
            'forward_bytes': 0,
            'backward_bytes': 0,
            'start_time': None,
            'last_seen': None,
            'flags': {'FIN': 0, 'SYN': 0, 'RST': 0, 'PSH': 0, 'ACK': 0, 'URG': 0, 'CWE': 0, 'ECE': 0},
            'fwd_flags': {'FIN': 0, 'SYN': 0, 'RST': 0, 'PSH': 0, 'ACK': 0, 'URG': 0, 'CWE': 0, 'ECE': 0},
            'bwd_flags': {'FIN': 0, 'SYN': 0, 'RST': 0, 'PSH': 0, 'ACK': 0, 'URG': 0, 'CWE': 0, 'ECE': 0},
            'protocol': 0,
            'src_ip': '',
            'dst_ip': '',
            'src_port': 0,
            'dst_port': 0,
            'fwd_header_length': 0,
            'bwd_header_length': 0
        })
    
    def get_flow_key(self, src_ip, dst_ip, src_port, dst_port, protocol):
        """Create bidirectional flow key"""
        # Sort to ensure same flow regardless of direction
        if src_ip < dst_ip:
            return f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
        else:
            return f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
    
    def extract_tcp_flags(self, tcp_layer):
        """Extract TCP flags"""
        flags = {
            'FIN': int(tcp_layer.flags.F),
            'SYN': int(tcp_layer.flags.S),
            'RST': int(tcp_layer.flags.R),
            'PSH': int(tcp_layer.flags.P),
            'ACK': int(tcp_layer.flags.A),
            'URG': int(tcp_layer.flags.U),
            'CWE': int(tcp_layer.flags.C) if hasattr(tcp_layer.flags, 'C') else 0,
            'ECE': int(tcp_layer.flags.E) if hasattr(tcp_layer.flags, 'E') else 0
        }
        return flags
    
    def update_flow(self, pkt):
        """Update flow statistics from packet"""
        try:
            if IP not in pkt:
                return None
            
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            protocol = pkt[IP].proto
            timestamp = float(pkt.time) if hasattr(pkt, 'time') else time.time()
            
            src_port = 0
            dst_port = 0
            header_length = len(pkt[IP])
            packet_length = len(pkt)
            
            # Get ports and flags
            flags = {}
            if TCP in pkt:
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
                flags = self.extract_tcp_flags(pkt[TCP])
                header_length = pkt[TCP].dataofs * 4
            elif UDP in pkt:
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport
            
            flow_key = self.get_flow_key(src_ip, dst_ip, src_port, dst_port, protocol)
            flow = self.flows[flow_key]
            
            # Initialize flow
            if flow['start_time'] is None:
                flow['start_time'] = timestamp
                flow['src_ip'] = src_ip
                flow['dst_ip'] = dst_ip
                flow['src_port'] = src_port
                flow['dst_port'] = dst_port
                flow['protocol'] = protocol
            
            flow['last_seen'] = timestamp
            
            # Determine direction (forward = src to dst, backward = dst to src)
            is_forward = (src_ip == flow['src_ip'] and dst_ip == flow['dst_ip'])
            
            if is_forward:
                flow['forward_packets'].append({
                    'length': packet_length,
                    'timestamp': timestamp,
                    'header_length': header_length
                })
                flow['forward_bytes'] += packet_length
                flow['fwd_header_length'] += header_length
                if flags:
                    for flag, val in flags.items():
                        flow['fwd_flags'][flag] += val
                        flow['flags'][flag] += val
            else:
                flow['backward_packets'].append({
                    'length': packet_length,
                    'timestamp': timestamp,
                    'header_length': header_length
                })
                flow['backward_bytes'] += packet_length
                flow['bwd_header_length'] += header_length
                if flags:
                    for flag, val in flags.items():
                        flow['bwd_flags'][flag] += val
                        flow['flags'][flag] += val
            
            return flow_key
            
        except Exception as e:
            logging.error(f"Flow update error: {e}")
            return None
    
    def compute_features(self, flow_key):
        """Compute all features for the flow matching training data"""
        flow = self.flows[flow_key]
        
        # Basic counts
        fwd_packets = len(flow['forward_packets'])
        bwd_packets = len(flow['backward_packets'])
        total_packets = fwd_packets + bwd_packets
        
        if total_packets == 0:
            return None
        
        # Timing
        duration = (flow['last_seen'] - flow['start_time']) * 1_000_000  # microseconds
        duration = max(duration, 1)  # Avoid division by zero
        
        # Packet lengths
        fwd_lengths = [p['length'] for p in flow['forward_packets']]
        bwd_lengths = [p['length'] for p in flow['backward_packets']]
        all_lengths = fwd_lengths + bwd_lengths
        
        # Calculate statistics
        fwd_length_total = sum(fwd_lengths) if fwd_lengths else 0
        bwd_length_total = sum(bwd_lengths) if bwd_lengths else 0
        total_length = fwd_length_total + bwd_length_total
        
        fwd_length_mean = np.mean(fwd_lengths) if fwd_lengths else 0
        fwd_length_std = np.std(fwd_lengths) if len(fwd_lengths) > 1 else 0
        fwd_length_max = max(fwd_lengths) if fwd_lengths else 0
        fwd_length_min = min(fwd_lengths) if fwd_lengths else 0
        
        bwd_length_mean = np.mean(bwd_lengths) if bwd_lengths else 0
        bwd_length_std = np.std(bwd_lengths) if len(bwd_lengths) > 1 else 0
        bwd_length_max = max(bwd_lengths) if bwd_lengths else 0
        bwd_length_min = min(bwd_lengths) if bwd_lengths else 0
        
        # Flow metrics
        flow_bytes_s = (total_length / duration) * 1_000_000 if duration > 0 else 0
        flow_packets_s = (total_packets / duration) * 1_000_000 if duration > 0 else 0
        
        # Inter-arrival times
        fwd_iat = []
        if len(flow['forward_packets']) > 1:
            for i in range(1, len(flow['forward_packets'])):
                iat = (flow['forward_packets'][i]['timestamp'] - 
                       flow['forward_packets'][i-1]['timestamp']) * 1_000_000
                fwd_iat.append(iat)
        
        bwd_iat = []
        if len(flow['backward_packets']) > 1:
            for i in range(1, len(flow['backward_packets'])):
                iat = (flow['backward_packets'][i]['timestamp'] - 
                       flow['backward_packets'][i-1]['timestamp']) * 1_000_000
                bwd_iat.append(iat)
        
        # Build feature dictionary (common CICIDS features)
        features = {
            'Destination Port': flow['dst_port'],
            'Flow Duration': duration,
            'Total Fwd Packets': fwd_packets,
            'Total Backward Packets': bwd_packets,
            'Total Length of Fwd Packets': fwd_length_total,
            'Total Length of Bwd Packets': bwd_length_total,
            'Fwd Packet Length Max': fwd_length_max,
            'Fwd Packet Length Min': fwd_length_min,
            'Fwd Packet Length Mean': fwd_length_mean,
            'Fwd Packet Length Std': fwd_length_std,
            'Bwd Packet Length Max': bwd_length_max,
            'Bwd Packet Length Min': bwd_length_min,
            'Bwd Packet Length Mean': bwd_length_mean,
            'Bwd Packet Length Std': bwd_length_std,
            'Flow Bytes/s': flow_bytes_s,
            'Flow Packets/s': flow_packets_s,
            'Flow IAT Mean': np.mean(fwd_iat + bwd_iat) if (fwd_iat + bwd_iat) else 0,
            'Flow IAT Std': np.std(fwd_iat + bwd_iat) if len(fwd_iat + bwd_iat) > 1 else 0,
            'Flow IAT Max': max(fwd_iat + bwd_iat) if (fwd_iat + bwd_iat) else 0,
            'Flow IAT Min': min(fwd_iat + bwd_iat) if (fwd_iat + bwd_iat) else 0,
            'Fwd IAT Total': sum(fwd_iat) if fwd_iat else 0,
            'Fwd IAT Mean': np.mean(fwd_iat) if fwd_iat else 0,
            'Fwd IAT Std': np.std(fwd_iat) if len(fwd_iat) > 1 else 0,
            'Fwd IAT Max': max(fwd_iat) if fwd_iat else 0,
            'Fwd IAT Min': min(fwd_iat) if fwd_iat else 0,
            'Bwd IAT Total': sum(bwd_iat) if bwd_iat else 0,
            'Bwd IAT Mean': np.mean(bwd_iat) if bwd_iat else 0,
            'Bwd IAT Std': np.std(bwd_iat) if len(bwd_iat) > 1 else 0,
            'Bwd IAT Max': max(bwd_iat) if bwd_iat else 0,
            'Bwd IAT Min': min(bwd_iat) if bwd_iat else 0,
            'Fwd PSH Flags': flow['fwd_flags']['PSH'],
            'Bwd PSH Flags': flow['bwd_flags']['PSH'],
            'Fwd URG Flags': flow['fwd_flags']['URG'],
            'Bwd URG Flags': flow['bwd_flags']['URG'],
            'Fwd Header Length': flow['fwd_header_length'],
            'Bwd Header Length': flow['bwd_header_length'],
            'Fwd Packets/s': (fwd_packets / duration) * 1_000_000 if duration > 0 else 0,
            'Bwd Packets/s': (bwd_packets / duration) * 1_000_000 if duration > 0 else 0,
            'Min Packet Length': min(all_lengths) if all_lengths else 0,
            'Max Packet Length': max(all_lengths) if all_lengths else 0,
            'Packet Length Mean': np.mean(all_lengths) if all_lengths else 0,
            'Packet Length Std': np.std(all_lengths) if len(all_lengths) > 1 else 0,
            'Packet Length Variance': np.var(all_lengths) if len(all_lengths) > 1 else 0,
            'FIN Flag Count': flow['flags']['FIN'],
            'SYN Flag Count': flow['flags']['SYN'],
            'RST Flag Count': flow['flags']['RST'],
            'PSH Flag Count': flow['flags']['PSH'],
            'ACK Flag Count': flow['flags']['ACK'],
            'URG Flag Count': flow['flags']['URG'],
            'CWE Flag Count': flow['flags']['CWE'],
            'ECE Flag Count': flow['flags']['ECE'],
            'Down/Up Ratio': bwd_packets / fwd_packets if fwd_packets > 0 else 0,
            'Average Packet Size': total_length / total_packets if total_packets > 0 else 0,
            'Avg Fwd Segment Size': fwd_length_mean,
            'Avg Bwd Segment Size': bwd_length_mean,
            'Fwd Header Length.1': flow['fwd_header_length'],  # Some datasets have duplicate columns
            'Fwd Avg Bytes/Bulk': fwd_length_total / fwd_packets if fwd_packets > 0 else 0,
            'Fwd Avg Packets/Bulk': fwd_packets,
            'Fwd Avg Bulk Rate': flow_bytes_s if fwd_packets > 0 else 0,
            'Bwd Avg Bytes/Bulk': bwd_length_total / bwd_packets if bwd_packets > 0 else 0,
            'Bwd Avg Packets/Bulk': bwd_packets,
            'Bwd Avg Bulk Rate': flow_bytes_s if bwd_packets > 0 else 0,
            'Subflow Fwd Packets': fwd_packets,
            'Subflow Fwd Bytes': fwd_length_total,
            'Subflow Bwd Packets': bwd_packets,
            'Subflow Bwd Bytes': bwd_length_total,
            'Init_Win_bytes_forward': 0,  # Would need raw TCP window sizes
            'Init_Win_bytes_backward': 0,
            'act_data_pkt_fwd': fwd_packets,
            'min_seg_size_forward': fwd_length_min if fwd_length_min > 0 else 0,
            'Active Mean': 0,  # Would need detailed timing analysis
            'Active Std': 0,
            'Active Max': 0,
            'Active Min': 0,
            'Idle Mean': 0,
            'Idle Std': 0,
            'Idle Max': 0,
            'Idle Min': 0,
            'Protocol': flow['protocol']
        }
        
        return features

# ============================================
# PREDICTION WITH REAL MODEL
# ============================================

def predict_flow(features_dict):
    """Predict using actual trained model"""
    try:
        # If we have feature names from training, ensure proper order
        if FEATURE_NAMES:
            features_ordered = [features_dict.get(feat, 0) for feat in FEATURE_NAMES]
        else:
            # Use features as-is (risky if order doesn't match)
            features_ordered = list(features_dict.values())
        
        # Convert to numpy array
        features_array = np.array(features_ordered).reshape(1, -1)
        
        # Handle invalid values
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        features_scaled = SCALER.transform(features_array)
        
        # Predict
        prediction_encoded = MODEL.predict(features_scaled)[0]
        probabilities = MODEL.predict_proba(features_scaled)[0]
        
        # Decode label
        prediction_label = LABEL_ENCODER.inverse_transform([prediction_encoded])[0]
        confidence = float(max(probabilities))
        
        # Determine if attack or benign
        is_attack = prediction_label.lower() != 'benign' and prediction_label.lower() != 'normal'
        
        return {
            'prediction': int(prediction_encoded),
            'label': prediction_label,
            'is_attack': is_attack,
            'confidence': confidence,
            'probabilities': {
                label: float(prob) 
                for label, prob in zip(LABEL_ENCODER.classes_, probabilities)
            }
        }
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return None

# ============================================
# LIVE CAPTURE WITH REAL FEATURES
# ============================================

def scapy_live_capture(interface):
    """Capture packets and extract real features"""
    global CAPTURE_ACTIVE, STATS, RECENT_EVENTS
    
    print(f"🔴 Capturing on: {interface}")
    
    analyzer = FlowAnalyzer()
    packet_count = [0]
    
    def packet_callback(pkt):
        try:
            packet_count[0] += 1
            
            flow_key = analyzer.update_flow(pkt)
            
            if flow_key is None:
                return
            
            flow = analyzer.flows[flow_key]
            total_packets = len(flow['forward_packets']) + len(flow['backward_packets'])
            
            # Process every 50 packets per flow
            if total_packets >= 50 and total_packets % 50 == 0:
                features = analyzer.compute_features(flow_key)
                
                if features is None:
                    return
                
                pred = predict_flow(features)
                
                if pred:
                    event = {
                        'timestamp': datetime.now().isoformat(),
                        'source_ip': flow['src_ip'],
                        'dest_ip': flow['dst_ip'],
                        'dest_port': flow['dst_port'],
                        'protocol': 'TCP' if flow['protocol'] == 6 else 'UDP' if flow['protocol'] == 17 else 'OTHER',
                        'packets': total_packets,
                        'bytes': flow['forward_bytes'] + flow['backward_bytes'],
                        'prediction': pred['label'],
                        'confidence': pred['confidence'],
                        'is_attack': pred['is_attack']
                    }
                    
                    RECENT_EVENTS.append(event)
                    if len(RECENT_EVENTS) > 100:
                        RECENT_EVENTS.pop(0)
                    
                    if pred['is_attack']:
                        STATS['attacks'] += 1
                        print(f"🚨 {pred['label']}: {flow['src_ip']} → {flow['dst_ip']}:{flow['dst_port']} ({pred['confidence']:.2%})")
                    else:
                        STATS['benign'] += 1
                    
                    STATS['total'] += 1
                    socketio.emit('live_prediction', event)
            
            if packet_count[0] % 100 == 0:
                socketio.emit('stats', STATS)
                print(f"Packets: {packet_count[0]}, Flows: {STATS['total']}")
        
        except Exception as e:
            logging.error(f"Packet callback error: {e}")
    
    try:
        sniff(
            iface=interface,
            prn=packet_callback,
            store=False,
            stop_filter=lambda x: not CAPTURE_ACTIVE,
            filter="ip"
        )
    except PermissionError:
        print("✗ Need Admin privileges!")
        socketio.emit('error', {'message': 'Need Admin/sudo privileges'})
    except Exception as e:
        print(f"✗ Capture error: {e}")
        socketio.emit('error', {'message': str(e)})

# ============================================
# PCAP PROCESSING WITH REAL FEATURES
# ============================================

def process_pcap_bg(pcap_file):
    """Process PCAP file with real feature extraction"""
    global STATS, RECENT_EVENTS
    
    try:
        print(f"Processing: {pcap_file}")
        socketio.emit('processing_update', {'status': 'Loading PCAP...', 'progress': 20})
        
        packets = rdpcap(pcap_file)
        analyzer = FlowAnalyzer()
        
        socketio.emit('processing_update', {'status': 'Extracting flows...', 'progress': 40})
        
        # Process all packets
        for i, pkt in enumerate(packets):
            analyzer.update_flow(pkt)
            
            if i % 1000 == 0:
                socketio.emit('processing_update', {
                    'status': f'Processed {i}/{len(packets)} packets...',
                    'progress': 40 + int((i / len(packets)) * 40)
                })
        
        socketio.emit('processing_update', {'status': 'Computing features...', 'progress': 80})
        
        # Analyze all flows
        flow_keys = list(analyzer.flows.keys())
        for idx, flow_key in enumerate(flow_keys[:2000]):  # Limit to 2000 flows
            try:
                features = analyzer.compute_features(flow_key)
                
                if features is None:
                    continue
                
                pred = predict_flow(features)
                
                if pred:
                    flow = analyzer.flows[flow_key]
                    
                    if pred['is_attack']:
                        STATS['attacks'] += 1
                    else:
                        STATS['benign'] += 1
                    STATS['total'] += 1
                    
                    event = {
                        'timestamp': datetime.now().isoformat(),
                        'source_ip': flow['src_ip'],
                        'dest_ip': flow['dst_ip'],
                        'dest_port': flow['dst_port'],
                        'protocol': 'TCP' if flow['protocol'] == 6 else 'UDP' if flow['protocol'] == 17 else 'OTHER',
                        'packets': len(flow['forward_packets']) + len(flow['backward_packets']),
                        'bytes': flow['forward_bytes'] + flow['backward_bytes'],
                        'prediction': pred['label'],
                        'confidence': pred['confidence'],
                        'is_attack': pred['is_attack']
                    }
                    
                    RECENT_EVENTS.append(event)
                    if len(RECENT_EVENTS) > 100:
                        RECENT_EVENTS.pop(0)
                    
                    socketio.emit('live_prediction', event)
                
                if idx % 50 == 0:
                    socketio.emit('processing_update', {
                        'status': f'Analyzing flow {idx}/{len(flow_keys)}...',
                        'progress': 80 + int((idx / min(len(flow_keys), 2000)) * 15)
                    })
            
            except Exception as e:
                logging.error(f"Flow analysis error: {e}")
                continue
        
        socketio.emit('processing_update', {'status': 'Complete!', 'progress': 100})
        socketio.emit('processing_complete', {
            'total_flows': STATS['total'],
            'attacks': STATS['attacks'],
            'benign': STATS['benign']
        })
        
        print(f"✓ Processed {STATS['total']} flows: {STATS['attacks']} attacks, {STATS['benign']} benign")
    
    except Exception as e:
        logging.error(f"PCAP processing error: {e}")
        socketio.emit('processing_error', {'error': str(e)})

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
        interface = 'Wi-Fi'  # Change based on your network adapter
        
        print(f"Starting capture on {interface}...")
        thread = threading.Thread(target=scapy_live_capture, args=(interface,), daemon=True)
        thread.start()
        
        socketio.emit('capture_started', {'status': 'Capturing'})
        return jsonify({'status': f'Started on {interface}'}), 200
    
    except Exception as e:
        CAPTURE_ACTIVE = False
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-live', methods=['POST'])
def stop_live():
    global CAPTURE_ACTIVE
    
    CAPTURE_ACTIVE = False
    socketio.emit('capture_stopped', {'status': 'Stopped'})
    return jsonify({'status': 'Stopped'}), 200

@app.route('/api/upload-pcap', methods=['POST'])
def upload_pcap():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    
    if not file.filename.endswith(('.pcap', '.pcapng')):
        return jsonify({'error': 'Only .pcap files'}), 400
    
    try:
        filename = f"upload_{datetime.now().timestamp()}.pcap"
        filepath = os.path.join('pcap_files', filename)
        file.save(filepath)
        
        thread = threading.Thread(target=lambda: process_pcap_bg(filepath), daemon=True)
        thread.start()
        
        return jsonify({'message': 'Processing...'}), 202
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify(STATS)

@app.route('/api/events', methods=['GET'])
def get_events():
    limit = request.args.get('limit', 50, type=int)
    return jsonify(RECENT_EVENTS[-limit:])

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Return model information"""
    return jsonify({
        'classes': list(LABEL_ENCODER.classes_) if LABEL_ENCODER else [],
        'n_features': len(FEATURE_NAMES) if FEATURE_NAMES else 'Unknown',
        'feature_names': FEATURE_NAMES[:10] if FEATURE_NAMES else []  # First 10 features
    })

@socketio.on('connect')
def handle_connect():
    emit('stats', STATS)

# ============================================
# STARTUP
# ============================================

if __name__ == '__main__':
    if load_models():
        print("\n" + "="*70)
        print("✓ CyberAI IDS Ready!")
        print("✓ Real-Time Feature Extraction")
        print("✓ Using Your Trained Model")
        print("="*70 + "\n")
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    else:
        print("Failed to load models")