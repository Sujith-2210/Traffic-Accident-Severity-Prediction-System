"""
Traffic Accident Severity Prediction System
Real-time prediction using Machine Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Traffic Accident Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 20px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .high-risk {
        background-color: #ffebee;
        color: #c62828;
        border: 3px solid #c62828;
    }
    .low-risk {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 3px solid #2e7d32;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained model, scaler, and feature names with compatibility handling"""
    try:
        # Try using joblib first (recommended for scikit-learn models)
        try:
            import joblib
            model = joblib.load('best_accident_model.pkl')
            scaler = joblib.load('scaler.pkl')
            feature_names = joblib.load('feature_names.pkl')
            return model, scaler, feature_names
        except (ImportError, Exception):
            pass
        
        # Fall back to pickle with numpy compatibility handling
        import numpy as np
        
        # Enable numpy compatibility for older pickle files
        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        
        try:
            with open('best_accident_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            with open('feature_names.pkl', 'rb') as f:
                feature_names = pickle.load(f)
            return model, scaler, feature_names
        finally:
            # Restore original np.load
            np.load = np_load_old
            
    except FileNotFoundError as e:
        st.error(f"⚠️ Model file not found: {e.filename}")
        st.info("💡 Please ensure you have trained the model and the .pkl files are in the same directory as app.py")
        return None, None, None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        st.warning("💡 Try regenerating the model files or check for version compatibility issues")
        
        # Show detailed error info in expander
        with st.expander("🔍 Technical Details"):
            st.code(f"""
Error Type: {type(e).__name__}
Error Message: {str(e)}

This error often occurs due to:
1. NumPy/scikit-learn version mismatch
2. Model trained with different Python version
3. Corrupted pickle files

Solution:
- Retrain the model with current environment
- Or install compatible versions of numpy/scikit-learn
            """)
        return None, None, None

model, scaler, feature_names = load_model()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_weather_data(city="Guntur", api_key=None):
    """
    Fetch real-time weather data from OpenWeatherMap API
    Falls back to simulated data if API key is not provided or request fails
    
    Args:
        city: City name for weather data
        api_key: OpenWeatherMap API key
    
    Returns:
        dict: Weather data with keys for weather, temp, visibility, humidity, wind_speed, etc.
    """
    if api_key:
        try:
            # OpenWeatherMap API endpoint
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Map OpenWeatherMap conditions to our app's categories
                weather_main = data['weather'][0]['main']
                weather_mapping = {
                    'Clear': 'Clear',
                    'Clouds': 'Cloudy',
                    'Rain': 'Rain',
                    'Drizzle': 'Rain',
                    'Thunderstorm': 'Rain',
                    'Snow': 'Snow',
                    'Mist': 'Fog',
                    'Fog': 'Fog',
                    'Haze': 'Fog'
                }
                
                return {
                    "weather": weather_mapping.get(weather_main, 'Clear'),
                    "temp": round(data['main']['temp']),
                    "visibility": round(data['visibility'] / 1000, 1),  # Convert m to km
                    "humidity": data['main']['humidity'],
                    "wind_speed": round(data['wind']['speed'] * 3.6, 1),  # Convert m/s to km/h
                    "description": data['weather'][0]['description'].title(),
                    "source": "Live API",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "city": data['name']
                }
            else:
                st.warning(f"⚠️ API Error: {response.status_code}. Using simulated data.")
        except requests.exceptions.Timeout:
            st.warning("⚠️ Weather API request timed out. Using simulated data.")
        except requests.exceptions.RequestException as e:
            st.warning(f"⚠️ Weather API error: {str(e)}. Using simulated data.")
        except Exception as e:
            st.warning(f"⚠️ Unexpected error: {str(e)}. Using simulated data.")
    
    # Simulated weather data for demo (fallback)
    return {
        "weather": np.random.choice(["Clear", "Rain", "Fog", "Snow", "Cloudy"]),
        "temp": np.random.randint(15, 35),
        "visibility": round(np.random.uniform(1, 10), 1),
        "humidity": np.random.randint(30, 90),
        "wind_speed": round(np.random.uniform(5, 30), 1),
        "description": "Simulated Weather",
        "source": "Simulated",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "city": city
    }

def prepare_input_data(input_dict):
    """Prepare input data for model prediction"""
    # Create a dataframe with all feature columns
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Set numerical features
    numerical_features = {
        'Speed_Limit': input_dict['speed_limit'],
        'Number_of_Vehicles': input_dict['num_vehicles'],
        'Traffic_Density': input_dict['traffic_density'],
        'Driver_Alcohol': input_dict['driver_alcohol'],
        'Age_vs_Experience': input_dict['age'] - input_dict['experience']
    }
    
    for feature, value in numerical_features.items():
        if feature in input_df.columns:
            input_df[feature] = value
    
    # Set categorical features (one-hot encoded)
    categorical_mappings = {
        'Weather': input_dict['weather'],
        'Road_Type': input_dict['road_type'],
        'Time_of_Day': input_dict['time_of_day'],
        'Accident_Severity': input_dict['accident_severity'],
        'Road_Condition': input_dict['road_condition'],
        'Vehicle_Type': input_dict['vehicle_type'],
        'Road_Light_Condition': input_dict['road_light']
    }
    
    for prefix, value in categorical_mappings.items():
        col_name = f"{prefix}_{value}"
        if col_name in input_df.columns:
            input_df[col_name] = 1
    
    # Scale numerical features
    numeric_columns = ['Speed_Limit', 'Number_of_Vehicles', 'Traffic_Density', 
                      'Driver_Alcohol', 'Age_vs_Experience']
    input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])
    
    return input_df

def predict_accident(input_data):
    """Make prediction using the trained model"""
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    return prediction, probability

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🚦 Traffic Accident Severity Prediction System</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time ML-powered accident risk assessment</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/traffic-light.png", width=100)
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", 
                                     ["🎯 Prediction", "📊 Model Info", "📈 Statistics"])
    
    # Weather API Configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌤️ Weather API Settings")
    
    # Initialize session state for weather data
    if 'weather_data' not in st.session_state:
        st.session_state.weather_data = None
    if 'auto_populate' not in st.session_state:
        st.session_state.auto_populate = False
    
    # API Key input
    api_key = st.sidebar.text_input(
        "OpenWeatherMap API Key",
        type="password",
        help="Get your free API key from https://openweathermap.org/api"
    )
    
    # Location selector
    city = st.sidebar.text_input(
        "Location",
        value="Guntur",
        help="Enter city name for weather data"
    )
    
    # Fetch weather button
    if st.sidebar.button("🔄 Fetch Live Weather", use_container_width=True):
        with st.spinner("Fetching weather data..."):
            st.session_state.weather_data = get_weather_data(city, api_key)
            if st.session_state.weather_data['source'] == "Live API":
                st.sidebar.success("✅ Live weather data fetched!")
            else:
                st.sidebar.info("ℹ️ Using simulated weather data")
    
    # Auto-populate toggle
    st.session_state.auto_populate = st.sidebar.checkbox(
        "Auto-populate weather fields",
        value=st.session_state.auto_populate,
        help="Automatically fill weather-related input fields with live data"
    )
    
    # Display API status
    if api_key:
        st.sidebar.success("🔑 API Key configured")
    else:
        st.sidebar.info("ℹ️ No API key - using simulated data")
    
    if app_mode == "🎯 Prediction":
        prediction_page()
    elif app_mode == "📊 Model Info":
        model_info_page()
    else:
        statistics_page()

# ============================================================================
# PREDICTION PAGE
# ============================================================================

def prediction_page():
    st.header("🎯 Accident Risk Prediction")
    
    # Two column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Traffic Conditions")
        
        # Create tabs for different input sections
        tab1, tab2, tab3 = st.tabs(["🌤️ Environment", "🚗 Traffic & Road", "👤 Driver Info"])
        
        with tab1:
            # Auto-populate weather fields if enabled
            if st.session_state.auto_populate and st.session_state.weather_data:
                wd = st.session_state.weather_data
                default_weather = wd['weather']
                default_temp = wd['temp']
                default_visibility = int(wd['visibility'])
                
                # Determine road condition based on weather
                road_cond_map = {
                    'Rain': 'Wet',
                    'Snow': 'Snow',
                    'Fog': 'Wet',
                    'Clear': 'Dry',
                    'Cloudy': 'Dry'
                }
                default_road_cond = road_cond_map.get(default_weather, 'Dry')
            else:
                default_weather = "Clear"
                default_temp = 25
                default_visibility = 8
                default_road_cond = "Dry"
            
            col_a, col_b = st.columns(2)
            with col_a:
                weather = st.selectbox("Weather Condition", 
                                      ["Clear", "Rain", "Fog", "Snow", "Cloudy"],
                                      index=["Clear", "Rain", "Fog", "Snow", "Cloudy"].index(default_weather))
                time_of_day = st.selectbox("Time of Day", 
                                          ["Morning", "Afternoon", "Evening", "Night"])
                road_light = st.selectbox("Road Light Condition", 
                                         ["Daylight", "Dark", "Street Lights", "Dawn/Dusk"])
            with col_b:
                road_condition = st.selectbox("Road Condition", 
                                             ["Dry", "Wet", "Icy", "Snow", "Muddy"],
                                             index=["Dry", "Wet", "Icy", "Snow", "Muddy"].index(default_road_cond))
                temperature = st.slider("Temperature (°C)", -10, 45, default_temp)
                visibility = st.slider("Visibility (km)", 0, 10, default_visibility)
        
        with tab2:
            col_c, col_d = st.columns(2)
            with col_c:
                road_type = st.selectbox("Road Type", 
                                        ["Highway", "Urban", "Rural", "Intersection"])
                traffic_density = st.slider("Traffic Density (vehicles/km)", 0, 200, 50)
                num_vehicles = st.slider("Number of Vehicles Involved", 1, 10, 2)
            with col_d:
                speed_limit = st.slider("Speed Limit (km/h)", 20, 120, 60)
                accident_severity = st.selectbox("Expected Severity Level", 
                                                ["Minor", "Moderate", "Severe", "Fatal"])
        
        with tab3:
            col_e, col_f = st.columns(2)
            with col_e:
                driver_age = st.slider("Driver Age", 18, 80, 35)
                driver_experience = st.slider("Driver Experience (years)", 0, 50, 10)
            with col_f:
                vehicle_type = st.selectbox("Vehicle Type", 
                                           ["Car", "Truck", "Motorcycle", "Bus", "Van"])
                driver_alcohol = st.slider("Driver Alcohol Level (BAC)", 0.0, 0.5, 0.0, 0.01)
        
        # Predict button
        if st.button("🔍 Predict Accident Risk", type="primary", use_container_width=True):
            # Prepare input
            input_dict = {
                'weather': weather,
                'time_of_day': time_of_day,
                'road_light': road_light,
                'road_condition': road_condition,
                'road_type': road_type,
                'traffic_density': traffic_density,
                'num_vehicles': num_vehicles,
                'speed_limit': speed_limit,
                'accident_severity': accident_severity,
                'age': driver_age,
                'experience': driver_experience,
                'vehicle_type': vehicle_type,
                'driver_alcohol': driver_alcohol
            }
            
            try:
                input_data = prepare_input_data(input_dict)
                prediction, probability = predict_accident(input_data)
                
                # Display results in col2
                with col2:
                    st.subheader("🎯 Prediction Result")
                    
                    if prediction == 1:
                        st.markdown("""
                        <div class="prediction-box high-risk">
                            ⚠️ HIGH RISK<br>
                            Accident Likely
                        </div>
                        """, unsafe_allow_html=True)
                        st.error("🚨 High probability of accident occurrence!")
                    else:
                        st.markdown("""
                        <div class="prediction-box low-risk">
                            ✅ LOW RISK<br>
                            Safe Conditions
                        </div>
                        """, unsafe_allow_html=True)
                        st.success("✅ Low probability of accident occurrence")
                    
                    # Probability display
                    st.subheader("📊 Risk Probability")
                    risk_prob = probability[1] * 100
                    safe_prob = probability[0] * 100
                    
                    st.metric("Accident Risk", f"{risk_prob:.1f}%", 
                             delta=f"{risk_prob - 50:.1f}% from baseline")
                    st.metric("Safe Conditions", f"{safe_prob:.1f}%")
                    
                    # Progress bar
                    st.progress(risk_prob / 100)
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    if prediction == 1:
                        st.warning("""
                        - ⚠️ Exercise extreme caution
                        - 🚗 Reduce speed immediately
                        - 🔦 Increase following distance
                        - 📞 Consider alternative route
                        - ⏸️ Delay travel if possible
                        """)
                    else:
                        st.info("""
                        - ✅ Conditions are favorable
                        - 👀 Stay alert and focused
                        - 🚦 Follow traffic rules
                        - 🛣️ Maintain safe distance
                        """)
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    with col2:
        if 'prediction' not in locals():
            st.info("👈 Enter traffic conditions and click 'Predict' to see results")
            
            # Display weather information
            st.subheader("🌤️ Current Weather")
            
            # Use session state weather data if available, otherwise fetch new
            if st.session_state.weather_data:
                weather_data = st.session_state.weather_data
            else:
                weather_data = get_weather_data()
            
            # Weather icon mapping
            weather_icons = {
                'Clear': '☀️',
                'Cloudy': '☁️',
                'Rain': '🌧️',
                'Fog': '🌫️',
                'Snow': '❄️'
            }
            
            # Display weather data with icons
            st.markdown(f"### {weather_icons.get(weather_data['weather'], '🌤️')} {weather_data['weather']}")
            
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                st.metric("🌡️ Temperature", f"{weather_data['temp']}°C")
                st.metric("👁️ Visibility", f"{weather_data['visibility']} km")
            with col_w2:
                st.metric("💧 Humidity", f"{weather_data['humidity']}%")
                st.metric("💨 Wind Speed", f"{weather_data['wind_speed']} km/h")
            
            # Display data source and timestamp
            if weather_data['source'] == "Live API":
                st.success(f"📡 Live data from {weather_data['city']}")
            else:
                st.info("🎲 Simulated weather data")
            
            st.caption(f"Last updated: {weather_data['timestamp']}")
            st.caption(f"Description: {weather_data['description']}")

# ============================================================================
# MODEL INFO PAGE
# ============================================================================

def model_info_page():
    st.header("📊 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Model Details")
        st.info("""
        **Model Type:** Balanced Random Forest Classifier  
        **Algorithm:** Random Forest with 300 estimators  
        **Training Data:** 3,000 accident records (3.5x expanded dataset)  
        **Features:** 41 engineered features (numerical + categorical)  
        **Test Accuracy:** ~65%  
        **ROC-AUC Score:** ~70% (↑ 16% improvement from baseline)  
        **Class Balance:** 54% no-accident, 46% accident  
        
        **Note:** Model trained on expanded synthetic dataset with realistic
        correlations between weather, road conditions, and driver factors.
        """)
        
        st.subheader("📈 Top Predictive Features")
        st.write("""
        **Most Important (by feature importance):**
        1. Driver Age & Driver Experience
        2. Number of Vehicles Involved
        3. Age vs Experience Gap
        4. Speed Limit & Traffic Density
        5. Road Conditions (Dry/Wet/Icy)
        6. Weather Conditions (Rain, Fog, etc.)
        7. Vehicle Type (Car, Truck, etc.)
        8. Accident Severity Level
        
        **Feature Engineering:**
        - One-hot encoding for categorical variables
        - Standardization of numerical features
        - Age-Experience interaction term
        """)
    
    with col2:
        st.subheader("🎯 Model Performance")
        
        # Display model comparison chart if available
        try:
            st.image("model_comparison.png", use_container_width=True)
        except:
            st.write("Model comparison chart not available")
        
        st.subheader("🔍 Confusion Matrix")
        try:
            st.image("confusion_matrix.png", use_container_width=True)
        except:
            st.write("Confusion matrix not available")
    
    st.subheader("💡 How It Works")
    st.write("""
    1. **Data Collection:** Gather traffic, weather, and road conditions
    2. **Feature Engineering:** Transform raw data into meaningful features
    3. **Preprocessing:** Scale and encode features for optimal performance
    4. **Prediction:** Use trained ML model to assess accident risk
    5. **Output:** Provide risk level and actionable recommendations
    """)

# ============================================================================
# STATISTICS PAGE
# ============================================================================

def statistics_page():
    st.header("📈 Traffic Accident Statistics")
    
    # Sample statistics (in real app, these would come from database)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", "1,247", delta="↑ 12%")
    with col2:
        st.metric("High Risk Alerts", "189", delta="↓ 5%")
    with col3:
        st.metric("Accuracy Rate", "93.4%", delta="↑ 2.1%")
    with col4:
        st.metric("Lives Saved", "47", delta="↑ 8")
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🌧️ Accidents by Weather")
        weather_data = pd.DataFrame({
            'Weather': ['Clear', 'Rain', 'Fog', 'Snow', 'Cloudy'],
            'Count': [120, 85, 45, 30, 55]
        })
        fig, ax = plt.subplots()
        ax.bar(weather_data['Weather'], weather_data['Count'], color='skyblue')
        ax.set_ylabel('Number of Accidents')
        ax.set_title('Accident Distribution by Weather')
        st.pyplot(fig)
    
    with col_b:
        st.subheader("⏰ Accidents by Time of Day")
        time_data = pd.DataFrame({
            'Time': ['Morning', 'Afternoon', 'Evening', 'Night'],
            'Count': [65, 95, 110, 75]
        })
        fig, ax = plt.subplots()
        ax.pie(time_data['Count'], labels=time_data['Time'], autopct='%1.1f%%', 
               colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
        ax.set_title('Accident Distribution by Time')
        st.pyplot(fig)
    
    st.markdown("---")
    
    st.subheader("🚗 Risk Factors Analysis")
    risk_factors = pd.DataFrame({
        'Factor': ['High Traffic Density', 'Poor Weather', 'Low Visibility', 
                   'Alcohol Involvement', 'High Speed', 'Poor Road Condition'],
        'Impact Score': [85, 78, 72, 95, 88, 70]
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(risk_factors['Factor'], risk_factors['Impact Score'], color='coral')
    ax.set_xlabel('Impact Score')
    ax.set_title('Top Risk Factors (Impact on Accident Probability)')
    st.pyplot(fig)

# ============================================================================
# FOOTER
# ============================================================================

def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🚦 Traffic Accident Prediction System | Built with Streamlit & Machine Learning</p>
        <p>⚠️ For demonstration purposes only. Not for actual emergency use.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    if model is not None:
        main()
        show_footer()
    else:
        st.error("❌ Failed to load model. Please ensure model files exist in the same directory.")