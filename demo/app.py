"""Streamlit demo for edge audio processing."""

import streamlit as st
import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import time
import json
from typing import Dict, Any, Optional

# Set page config
st.set_page_config(
    page_title="Edge Audio Processing Demo",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add disclaimer
st.warning("""
**DISCLAIMER**: This is a research/educational demonstration. 
NOT FOR SAFETY-CRITICAL USE. This system is designed for learning and experimentation purposes only.
""")

# Title and description
st.title("🎵 Edge Audio Processing Demo")
st.markdown("""
This demo showcases real-time audio event detection optimized for edge devices. 
The system can classify audio events like claps, glass breaks, and background noise using lightweight neural networks.
""")

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Model Type",
    ["cnn", "tiny", "transformer"],
    help="Select the neural network architecture"
)

# Audio parameters
sample_rate = st.sidebar.slider("Sample Rate", 8000, 48000, 16000)
duration = st.sidebar.slider("Audio Duration (seconds)", 0.5, 3.0, 1.0)
n_mfcc = st.sidebar.slider("MFCC Coefficients", 8, 20, 13)

# Demo mode selection
demo_mode = st.sidebar.radio(
    "Demo Mode",
    ["Synthetic Audio", "Upload Audio", "Real-time Simulation"],
    help="Choose how to generate audio for testing"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'audio_processor' not in st.session_state:
    st.session_state.audio_processor = None
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []
if 'latency_history' not in st.session_state:
    st.session_state.latency_history = []

# Load model (simplified for demo)
@st.cache_resource
def load_model(model_type: str, n_mfcc: int = 13):
    """Load the audio event detection model."""
    try:
        # Import here to avoid issues if modules aren't available
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        
        from src.models.audio_models import create_model
        from src.utils.audio import AudioProcessor
        
        model = create_model(
            model_type=model_type,
            input_length=100,
            n_mfcc=n_mfcc,
            n_classes=3
        )
        
        audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            max_length=100
        )
        
        return model, audio_processor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load model
if st.session_state.model is None or st.session_state.audio_processor is None:
    with st.spinner("Loading model..."):
        model, audio_processor = load_model(model_type, n_mfcc)
        if model is not None:
            st.session_state.model = model
            st.session_state.audio_processor = audio_processor
            st.success("Model loaded successfully!")
        else:
            st.error("Failed to load model")
            st.stop()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Audio Input")
    
    if demo_mode == "Synthetic Audio":
        st.subheader("Generate Synthetic Audio")
        
        # Event type selection
        event_type = st.selectbox(
            "Event Type",
            ["clap", "glass_break", "noise"],
            help="Select the type of audio event to generate"
        )
        
        # Noise level
        noise_level = st.slider("Background Noise Level", 0.0, 0.5, 0.1)
        
        # Generate button
        if st.button("Generate Audio", type="primary"):
            with st.spinner("Generating audio..."):
                # Generate synthetic audio
                audio = st.session_state.audio_processor.synthesize_audio_event(
                    event_type=event_type,
                    duration=duration,
                    noise_level=noise_level
                )
                
                # Store in session state
                st.session_state.current_audio = audio
                st.session_state.current_label = event_type
                
                st.success(f"Generated {event_type} audio!")
                
    elif demo_mode == "Upload Audio":
        st.subheader("Upload Audio File")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'm4a'],
            help="Upload an audio file for analysis"
        )
        
        if uploaded_file is not None:
            try:
                # Load audio file
                audio, sr = librosa.load(uploaded_file, sr=sample_rate)
                
                # Resize to target duration
                target_length = int(duration * sample_rate)
                if len(audio) > target_length:
                    audio = audio[:target_length]
                elif len(audio) < target_length:
                    audio = np.pad(audio, (0, target_length - len(audio)))
                
                st.session_state.current_audio = audio
                st.session_state.current_label = "unknown"
                
                st.success("Audio file loaded successfully!")
                
            except Exception as e:
                st.error(f"Error loading audio file: {e}")
                
    elif demo_mode == "Real-time Simulation":
        st.subheader("Real-time Audio Simulation")
        
        if st.button("Start Simulation", type="primary"):
            # Simulate real-time audio processing
            st.info("Simulating real-time audio processing...")
            
            # Generate random audio events
            event_types = ["clap", "glass_break", "noise"]
            event_type = np.random.choice(event_types)
            
            audio = st.session_state.audio_processor.synthesize_audio_event(
                event_type=event_type,
                duration=duration
            )
            
            st.session_state.current_audio = audio
            st.session_state.current_label = event_type

with col2:
    st.header("Audio Analysis")
    
    if 'current_audio' in st.session_state:
        # Display audio waveform
        st.subheader("Audio Waveform")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=st.session_state.current_audio,
            mode='lines',
            name='Audio Signal'
        ))
        fig.update_layout(
            title="Audio Waveform",
            xaxis_title="Time (samples)",
            yaxis_title="Amplitude",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Play audio
        st.audio(st.session_state.current_audio, sample_rate=sample_rate)
        
        # Run inference
        if st.button("Analyze Audio", type="primary"):
            with st.spinner("Running inference..."):
                # Preprocess audio
                features = st.session_state.audio_processor.preprocess_audio(
                    st.session_state.current_audio
                )
                
                # Convert to tensor
                input_tensor = torch.FloatTensor(features).unsqueeze(0)
                
                # Run inference
                start_time = time.time()
                with torch.no_grad():
                    outputs = st.session_state.model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(outputs, dim=1).item()
                inference_time = time.time() - start_time
                
                # Class names
                class_names = ["clap", "glass_break", "noise"]
                predicted_label = class_names[predicted_class]
                
                # Store results
                st.session_state.predictions_history.append({
                    'predicted': predicted_label,
                    'actual': st.session_state.current_label,
                    'probabilities': probabilities.numpy()[0],
                    'inference_time': inference_time
                })
                st.session_state.latency_history.append(inference_time)
                
                # Display results
                st.subheader("Analysis Results")
                
                # Prediction
                col_pred, col_time = st.columns([2, 1])
                with col_pred:
                    st.metric("Predicted Class", predicted_label)
                with col_time:
                    st.metric("Inference Time", f"{inference_time*1000:.2f} ms")
                
                # Confidence scores
                st.subheader("Confidence Scores")
                
                conf_df = {
                    'Class': class_names,
                    'Probability': probabilities.numpy()[0]
                }
                
                fig_conf = px.bar(
                    x=conf_df['Class'],
                    y=conf_df['Probability'],
                    title="Class Probabilities",
                    color=conf_df['Probability'],
                    color_continuous_scale='viridis'
                )
                fig_conf.update_layout(height=400)
                st.plotly_chart(fig_conf, use_container_width=True)
                
                # Show actual vs predicted
                if st.session_state.current_label != "unknown":
                    if predicted_label == st.session_state.current_label:
                        st.success(f"✅ Correct prediction: {predicted_label}")
                    else:
                        st.error(f"❌ Incorrect prediction. Actual: {st.session_state.current_label}, Predicted: {predicted_label}")

# Performance metrics
if st.session_state.predictions_history:
    st.header("Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Accuracy
        if any(p['actual'] != 'unknown' for p in st.session_state.predictions_history):
            correct = sum(1 for p in st.session_state.predictions_history 
                         if p['actual'] != 'unknown' and p['predicted'] == p['actual'])
            total = sum(1 for p in st.session_state.predictions_history 
                       if p['actual'] != 'unknown')
            accuracy = correct / total if total > 0 else 0
            st.metric("Accuracy", f"{accuracy:.2%}")
    
    with col2:
        # Average latency
        avg_latency = np.mean(st.session_state.latency_history) * 1000
        st.metric("Avg Latency", f"{avg_latency:.2f} ms")
    
    with col3:
        # Total predictions
        st.metric("Total Predictions", len(st.session_state.predictions_history))
    
    # Latency over time
    if len(st.session_state.latency_history) > 1:
        st.subheader("Latency Over Time")
        
        fig_latency = go.Figure()
        fig_latency.add_trace(go.Scatter(
            y=[t * 1000 for t in st.session_state.latency_history],
            mode='lines+markers',
            name='Inference Latency (ms)'
        ))
        fig_latency.update_layout(
            title="Inference Latency Over Time",
            xaxis_title="Prediction Number",
            yaxis_title="Latency (ms)",
            height=300
        )
        st.plotly_chart(fig_latency, use_container_width=True)
    
    # Prediction history
    st.subheader("Prediction History")
    
    history_data = []
    for i, pred in enumerate(st.session_state.predictions_history):
        history_data.append({
            'Prediction': i + 1,
            'Predicted': pred['predicted'],
            'Actual': pred['actual'],
            'Latency (ms)': f"{pred['inference_time']*1000:.2f}"
        })
    
    st.dataframe(history_data, use_container_width=True)

# Model information
st.sidebar.header("Model Information")
st.sidebar.info(f"""
**Model Type**: {model_type.upper()}
**Parameters**: {sum(p.numel() for p in st.session_state.model.parameters()):,}
**Input Shape**: (1, 100, {n_mfcc})
**Classes**: clap, glass_break, noise
""")

# Clear history button
if st.sidebar.button("Clear History"):
    st.session_state.predictions_history = []
    st.session_state.latency_history = []
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
**Edge Audio Processing Demo** - A research/educational demonstration of real-time audio event detection optimized for edge devices.
""")
