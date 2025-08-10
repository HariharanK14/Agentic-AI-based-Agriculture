# combined_app.py
import streamlit as st
import pandas as pd
import numpy as np
import folium
import joblib
import os
import time
import json
import logging
import traceback

# --- Core ML/Data Libraries ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from folium.plugins import MarkerCluster

# --- LLM & Agent Libraries ---
import google.generativeai as genai # Direct Gemini usage
from phi.agent import Agent, Toolkit
from phi.llm.google import Gemini as PhiGemini # Gemini for phi-agent
from phi.model.groq import Groq
from phi.workflow import Workflow, RunResponse, RunEvent
from phi.storage.workflow.sqlite import SqlWorkflowStorage
from phi.utils.log import logger as phi_logger # Rename phi's logger

# --- Utilities ---
from dotenv import load_dotenv
import warnings

# --- TensorFlow (Optional Import) ---
try:
    from tensorflow import keras
    from keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    st.warning("TensorFlow/Keras not found. Crop Suitability Map feature will be disabled. Install with: `pip install tensorflow`")
    TF_AVAILABLE = False
    def load_model(path): return None

# ==============================================================================
# Configuration & Setup
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="Integrated Agricultural AI Assistant",
    page_icon="🧑‍🌾"
)

warnings.filterwarnings("ignore", category=UserWarning, module='folium')
warnings.filterwarnings("ignore", category=FutureWarning) # Ignore some numpy/pandas warnings

# --- Basic Logging Setup ---
# Configure Python's logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING) # Reduce httpx verbosity
logging.getLogger("PIL").setLevel(logging.WARNING) # Reduce PIL verbosity
phi_logger.setLevel(logging.INFO) # Configure phi-agent's logger level

logger = logging.getLogger(__name__) # Logger for this script

# --- Load Environment Variables (Optional - st.secrets is preferred) ---
load_dotenv()

# --- API Key Configuration (Using Streamlit Secrets) ---
try:
    GEMINI_API_KEY = "AIzaSyCFwHNu5IS33H6XHy-Jz-shDY-mpBT1q-Y"
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_direct_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    GEMINI_AVAILABLE = True
    logger.info("Direct Google Gemini AI configured successfully.")
except (KeyError, Exception) as e:
    st.sidebar.error(f"Error configuring Google Gemini API: {e}\nEnsure 'GEMINI_API_KEY' is in secrets.toml.")
    GEMINI_API_KEY = None
    gemini_direct_model = None
    GEMINI_AVAILABLE = False

try:
    GROQ_API_KEY = "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"
    groq_client = Groq(api_key=GROQ_API_KEY, model="llama-3.1-70b-versatile") 
    GROQ_AVAILABLE = True
    logger.info("Groq client configured successfully.")
except (KeyError, Exception) as e:
    st.sidebar.error(f"Error configuring Groq API: {e}\nEnsure 'GROQ_API_KEY' is in secrets.toml.")
    GROQ_API_KEY = None
    groq_client = None
    GROQ_AVAILABLE = False

# --- File Paths & Constants ---

# == App 1: Production Analysis Paths ==

PRODUCTION_DATA_PATH = r"C:\Users\haris\OneDrive\Major_project\dataset\New\ICRISAT-Averaged-Data_Set_agent.csv"
# -----------------------------------------
DB_FILE = "combined_agri_workflows.db"
TABLE_NAME = "crop_workflows_combined_v1"

# == App 2: Suitability Map Paths ==

BASE_DIR_SUITABILITY = os.path.dirname(r"C:\Users\haris\OneDrive\Major_project\Code\New_Implementation\crop_recommendation_model_v3.h5") # Or adjust as needed
MODEL_PATH_SUITABILITY = os.path.join(BASE_DIR_SUITABILITY, "crop_recommendation_model_v3.h5")
ENCODER_PATH_SUITABILITY = os.path.join(BASE_DIR_SUITABILITY, "label_encoder_v3.pkl")
SCALER_PATH_SUITABILITY = os.path.join(BASE_DIR_SUITABILITY, "scaler_v3.pkl")
# ----- USER: PLEASE VERIFY THIS PATH -----
SUITABILITY_DATA_PATH = r"C:\Users\haris\OneDrive\Major_project\dataset\New\cleaned_merged_data.csv"
# -----------------------------------------
OUTPUT_HTML_PATH = "tamil_nadu_crop_map_generated.html"
FEATURES_FOR_SUITABILITY_MODEL = ['TEMPERATURE', 'RAINFALL', 'FERTILIZER', 'Pesticides', 'Latitude', 'Longitude']
PROBABILITY_THRESHOLD = 0.05

# == App 2: District Coordinates & Colors ==
district_coord_plotting = {
    'ARIYALUR': (11.1354, 79.0723), 'CHENNAI': (13.0827, 80.2707),
    'COIMBATORE': (11.0168, 76.9558), 'CUDDALORE': (11.7599, 79.7423),
    'DHARMAPURI': (12.5274, 78.3020), 'DINDIGUL': (10.3533, 77.9965),
    'ERODE': (11.5474, 77.7009), 'KALLAKURICHI':(11.7443, 78.9674),
    'KANCHEEPURAM': (12.8303, 79.6955), 'KANNIYAKUMARI': (8.3371, 77.5429),
    'KARUR': (10.9775, 78.0819), 'KRISHNAGIRI': (12.5249, 78.2332),
    'MADURAI': (9.9252, 78.1198), 'MAYILADUTHURAI':(11.1234, 79.6543),
    'NAGAPATTINAM': (10.7767, 79.8424), 'NAMAKKAL': (11.2219, 78.1673),
    'PERAMBALUR':(11.2222, 78.8844), 'PUDUKKOTTAI': (10.3856, 78.8209),
    'RAMANATHAPURAM': (9.3602, 78.8320), 'RANIPET':(12.9789, 79.3845),
    'SALEM': (11.6643, 78.1460), 'SIVAGANGAI': (9.8533, 78.4781),
    'TENKASI':(8.9688, 77.3548), 'THANJAVUR': (10.7973, 79.1348),
    'THENI': (10.0982, 77.4878), 'THIRUVALLUR': (13.1438, 79.9174),
    'THIRUVARUR': (10.7665, 79.6359), 'THOOTHUKUDI': (8.7917, 78.1311),
    'TIRUCHIRAPPALLI': (10.7905, 78.7047), 'TIRUNELVELI': (8.7139, 77.7569),
    'TIRUPATHUR':(12.4809, 78.6024), 'TIRUPPUR':(11.1089, 77.3417),
    'VILLUPURAM': (11.9332, 79.4930), 'VELLORE': (12.9791, 79.1303),
    'VIRUDHUNAGAR': (9.5942, 77.9562)
    }
crop_colors = {
    'RICE': 'green', 'WHEAT': 'yellow', 'COTTON': 'orange', 'SORGHUM': 'red',
    'PEARL MILLET': 'purple', 'MAIZE': 'blue', 'FINGER MILLET': 'pink',
    'BARLEY': 'gold', 'CHICKPEA': 'lightblue', 'PIGEONPEA': 'teal',
    'MINOR PULSES': 'violet', 'GROUNDNUT': 'brown', 'SESAMUM': 'sienna',
    'RAPESEED AND MUSTARD': 'olive', 'SAFFLOWER': 'coral', 'CASTOR': 'indigo',
    'LINSEED': 'peru', 'SUNFLOWER': 'khaki', 'SOYABEAN': 'gray',
    'OILSEEDS': 'navy', 'SUGARCANE': 'black', 'FRUITS': 'lime',
    'VEGETABLES': 'darkgreen', 'UNKNOWN': 'darkred'
}

# ==============================================================================
# Data Loading Functions (Cached)
# ==============================================================================

# --- App 1: Production Data Loader ---
@st.cache_data(show_spinner="Loading Production Data (ICRISAT)...")
def load_production_data(file_path):
    """Loads the ICRISAT dataset and extracts the list of crops."""
    logger.info(f"Attempting to load production data from: {file_path}")
    if not os.path.exists(file_path):
        st.error(f"Error: Production dataset file not found at: '{file_path}'.")
        logger.error(f"Production dataset file not found: {file_path}")
        return None, []
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded production dataframe with shape: {df.shape}")

        # Handle potential infinite values
        numeric_cols = df.select_dtypes(include=np.number).columns
        if df[numeric_cols].isin([np.inf, -np.inf]).any().any():
            logger.warning("Found infinite values in production data. Replacing with NaN.")
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Derive crop list
        crop_list = set()
        potential_production_cols = [col for col in df.columns if ' PRODUCTION (' in col.upper()]
        for col in potential_production_cols:
            crop_name = col.upper().split(' PRODUCTION (')[0]
            has_area = any(c.upper().startswith(f"{crop_name} AREA (") for c in df.columns)
            has_yield = any(c.upper().startswith(f"{crop_name} YIELD (") for c in df.columns)
            if has_area and has_yield:
                original_case_crop_name = col.split(' PRODUCTION (')[0]
                crop_list.add(original_case_crop_name)
        sorted_crop_list = sorted(list(crop_list))
        if not sorted_crop_list:
            st.error("Could not derive crop names with PRODUCTION, AREA, and YIELD columns from production data.")
            logger.error("No valid crops derived from production dataset columns.")
            return None, []
        logger.info(f"Derived {len(sorted_crop_list)} valid crops from production data.")
        return df, sorted_crop_list
    except Exception as e:
        st.error(f"An error occurred loading production data: {e}")
        logger.error(f"Production data loading error: {e}", exc_info=True)
        return None, []

# --- App 2: Suitability Data Loader ---
@st.cache_data(show_spinner="Loading Suitability Reference Data...")
def load_suitability_data(file_path):
    """Loads and preprocesses the reference data CSV for the suitability map."""
    logger.info(f"Attempting to load suitability data from: {file_path}")
    if not os.path.exists(file_path):
        st.error(f"Error: Suitability dataset file not found at: '{file_path}'.")
        logger.error(f"Suitability dataset file not found: {file_path}")
        return None, f"Suitability data file not found: {file_path}"
    try:
        df = pd.read_csv(file_path)
        if 'DISTRICT' not in df.columns:
            raise ValueError("DISTRICT column missing in suitability data")
        df['DISTRICT'] = df['DISTRICT'].str.upper().str.strip()

        def convert_to_float(coord_str):
            coord_str = str(coord_str).strip().upper()
            for s in ['°N','°S','°E','°W', ' ']: coord_str = coord_str.replace(s, '')
            try: return float(coord_str)
            except ValueError: return np.nan

        df['Latitude'] = df['Latitude'].apply(convert_to_float)
        df['Longitude'] = df['Longitude'].apply(convert_to_float)

        initial_rows = len(df)
        df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
        rows_dropped = initial_rows - len(df)
        if rows_dropped > 0:
             logger.info(f"Dropped {rows_dropped} rows from suitability data due to missing/invalid coordinates.")

        if df.empty: raise ValueError("No valid rows in suitability data after cleaning coordinates.")

        # Check and clean feature columns
        for col in FEATURES_FOR_SUITABILITY_MODEL:
            if col not in df.columns: raise ValueError(f"Feature column '{col}' missing from suitability data.")
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Filled missing values in suitability data column '{col}' with median ({median_val:.2f}).")

        check_cols = ['TEMPERATURE', 'RAINFALL', 'FERTILIZER', 'Pesticides', 'Latitude', 'Longitude', 'DISTRICT']
        for col in check_cols:
             if col not in df.columns: raise ValueError(f"Required column '{col}' missing from suitability data after preprocessing.")

        logger.info(f"Successfully loaded and preprocessed suitability data. Shape: {df.shape}")
        return df, None
    except Exception as e:
        st.error(f"An error occurred loading/processing suitability data: {e}")
        logger.error(f"Suitability data loading/processing error: {e}", exc_info=True)
        return None, f"Error loading/processing suitability data: {e}"


# ==============================================================================
# Resource Loading Function (App 2 - TF Model)
# ==============================================================================

@st.cache_resource(show_spinner="Loading Suitability AI Model and Resources...")
def load_tf_resources():
    """Loads the TF model, scaler, and encoder for suitability prediction."""
    if not TF_AVAILABLE:
        return None, None, None, "TensorFlow/Keras is not available."
    try:
        # Check file existence before loading
        if not os.path.exists(MODEL_PATH_SUITABILITY):
            raise FileNotFoundError(f"Suitability model file not found: {MODEL_PATH_SUITABILITY}")
        if not os.path.exists(ENCODER_PATH_SUITABILITY):
             raise FileNotFoundError(f"Suitability encoder file not found: {ENCODER_PATH_SUITABILITY}")
        if not os.path.exists(SCALER_PATH_SUITABILITY):
             raise FileNotFoundError(f"Suitability scaler file not found: {SCALER_PATH_SUITABILITY}")

        model = load_model(MODEL_PATH_SUITABILITY)
        label_encoder = joblib.load(ENCODER_PATH_SUITABILITY)
        scaler = joblib.load(SCALER_PATH_SUITABILITY)

        # Add default colors for any crops in encoder but not in map
        for crop_name in label_encoder.classes_:
            if crop_name not in crop_colors:
                crop_colors[crop_name] = 'lightgray'
        logger.info("Successfully loaded TensorFlow model, encoder, and scaler.")
        return model, label_encoder, scaler, None
    except FileNotFoundError as e:
         error_msg = f"Resource file not found for suitability model. Check paths.\n{e}"
         st.error(error_msg)
         logger.error(error_msg)
         return None, None, None, error_msg
    except Exception as e:
        error_msg = f"CRITICAL ERROR loading suitability model resources: {e}"
        st.error(error_msg)
        logger.error(error_msg, exc_info=True)
        return None, None, None, error_msg

# ==============================================================================
# App 1: Production Analysis - Core Logic (phi-agent)
# ==============================================================================

# Define classes and toolkits only if needed resources are available (like data)
production_tool_initialized = False
production_workflow_initialized = False
df_prod_global = None # Use a global-like variable to hold the loaded data
crop_list_prod_global = []

try:
    # Load production data early to check if App 1 can function
    df_prod_global, crop_list_prod_global = load_production_data(PRODUCTION_DATA_PATH)
    production_data_loaded = df_prod_global is not None

    if production_data_loaded:
        class CropProductionPredictionTool(Toolkit):
            def __init__(self, dataframe):
                super().__init__(name="crop_production_prediction_tools")
                if dataframe is None:
                     raise ValueError("Dataframe cannot be None for CropProductionPredictionTool")
                self._df = dataframe.copy()
                self.register(self.get_feature_importance_and_predict)
                logger.info(f"CropProductionPredictionTool initialized.")

            def get_feature_importance_and_predict(self, crop_name: str) -> dict:
                """
                Trains, evaluates, and analyzes a RandomForestRegressor for crop production.
                Returns a dictionary with results or an error message.
                """
                logger.info(f"--- Starting Production Analysis for crop: {crop_name} ---")
                target_pattern = f"{crop_name.upper()} PRODUCTION ("
                target_col_actual = next((col for col in self._df.columns if col.upper().startswith(target_pattern)), None)
                if not target_col_actual:
                    err_msg = f"Target variable '{crop_name} PRODUCTION (...)' not found."
                    logger.error(f"{err_msg} Available columns (sample): {self._df.columns[:10].tolist()}...")
                    return {"error": err_msg}
                target = target_col_actual

                base_features_required = ['FERTILIZER', 'RAINFALL', 'Pesticides']
                crop_area_pattern = f"{crop_name.upper()} AREA ("
                crop_yield_pattern = f"{crop_name.upper()} YIELD ("
                crop_area_col = next((col for col in self._df.columns if col.upper().startswith(crop_area_pattern)), None)
                crop_yield_col = next((col for col in self._df.columns if col.upper().startswith(crop_yield_pattern)), None)

                crop_specific_features = []
                if crop_area_col: crop_specific_features.append(crop_area_col)
                else: logger.warning(f"Area column not found for '{crop_name}'.")
                if crop_yield_col: crop_specific_features.append(crop_yield_col)
                else: logger.warning(f"Yield column not found for '{crop_name}'.")

                valid_base_features = [f for f in base_features_required if f in self._df.columns]
                missing_base = [f for f in base_features_required if f not in valid_base_features]
                if missing_base: logger.warning(f"Missing base features: {missing_base}.")

                potential_features = crop_specific_features + valid_base_features
                valid_features = [f for f in potential_features if f in self._df.columns]

                if not valid_features:
                    err_msg = f"No valid features found for crop '{crop_name}'."
                    logger.error(f"{err_msg} Checked for: {potential_features}.")
                    return {"error": err_msg}

                logger.info(f"Using features for {crop_name}: {valid_features}")

                # --- Preprocessing ---
                df_filtered = self._df[valid_features + [target]].copy()
                df_filtered.replace([np.inf, -np.inf], np.nan, inplace=True)
                initial_rows = len(df_filtered)
                df_filtered.dropna(subset=[target], inplace=True)
                if initial_rows > len(df_filtered): logger.warning(f"Dropped {initial_rows - len(df_filtered)} rows due to NaN in target '{target}'.")
                if df_filtered.shape[0] < 10:
                    err_msg = f"Insufficient data ({df_filtered.shape[0]} rows) for '{crop_name}' after cleaning."
                    logger.error(err_msg)
                    return {"error": err_msg}

                for col in valid_features:
                    if df_filtered[col].isnull().any():
                        nan_count = df_filtered[col].isnull().sum()
                        nan_percentage = (nan_count / len(df_filtered)) * 100
                        logger.warning(f"Feature '{col}' has {nan_count} NaNs ({nan_percentage:.2f}%). Imputing...")
                        if pd.api.types.is_numeric_dtype(df_filtered[col]):
                            fill_value = df_filtered[col].median()
                            if pd.isna(fill_value): fill_value = 0
                            df_filtered[col].fillna(fill_value, inplace=True)
                        else:
                            mode_val = df_filtered[col].mode()
                            fill_value = mode_val[0] if not mode_val.empty else "Unknown"
                            df_filtered[col].fillna(fill_value, inplace=True)

                X = df_filtered[valid_features]
                y = df_filtered[target]

                if y.nunique() <= 1:
                    err_msg = f"Target '{target}' has <= 1 unique value after cleaning. Cannot train."
                    logger.error(err_msg)
                    return {"error": err_msg}

                categorical_cols = X.select_dtypes(include=['object', 'category']).columns
                if len(categorical_cols) > 0:
                    logger.info(f"Encoding categorical columns: {categorical_cols.tolist()}")
                    X = X.copy()
                    for col in categorical_cols:
                        try:
                            le = LabelEncoder()
                            X.loc[:, col] = le.fit_transform(X[col].astype(str))
                        except Exception as e:
                            err_msg = f"Failed to encode column '{col}': {e}"
                            logger.error(err_msg, exc_info=True)
                            return {"error": err_msg}

                if X.isnull().values.any():
                    nan_cols = X.columns[X.isnull().any()].tolist()
                    logger.error(f"NaN values persist in features after processing: {nan_cols}. Aborting.")
                    return {"error": f"Internal error: NaN values persisted in features: {nan_cols}."}

                # --- Model Training & Evaluation ---
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    if X_train.empty or X_test.empty:
                        raise ValueError("Train or test split resulted in zero samples.")
                    logger.info(f"Production data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

                    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1, oob_score=True)
                    logger.info("Fitting RandomForestRegressor model for production...")
                    rf_model.fit(X_train, y_train)
                    logger.info("Production model training complete.")
                    if rf_model.oob_score_: logger.info(f"OOB Score: {rf_model.oob_score_:.4f}")

                    y_pred = rf_model.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    logger.info(f"Production Model evaluation: MAE={mae:.4f}, R2={r2:.4f}")

                    feature_importances_df = pd.DataFrame({
                        'Feature': X.columns.tolist(),
                        'Importance': rf_model.feature_importances_
                    }).sort_values(by='Importance', ascending=False)
                    feature_importances_list = [
                        {"Feature": row["Feature"], "Importance": float(row["Importance"])}
                        for index, row in feature_importances_df.iterrows()
                    ]
                    if feature_importances_list: logger.info(f"Top feature: {feature_importances_list[0]['Feature']} ({feature_importances_list[0]['Importance']:.4f})")

                    safe_y_pred = [float(p) if pd.notna(p) and np.isfinite(p) else None for p in y_pred]
                    safe_y_test = [float(a) if pd.notna(a) and np.isfinite(a) else None for a in y_test.tolist()]

                    results = {
                        "crop_name": crop_name,
                        "mae": float(mae) if pd.notna(mae) else None,
                        "r2_score": float(r2) if pd.notna(r2) else None,
                        "feature_importances": feature_importances_list,
                        "sample_predictions": safe_y_pred[:10],
                        "actual_values_for_sample": safe_y_test[:10]
                    }
                    logger.info(f"--- Production Analysis complete for crop: {crop_name} ---")
                    return results

                except Exception as e:
                     err_msg = f"Error during model training/evaluation for production: {e}"
                     logger.error(err_msg, exc_info=True)
                     return {"error": err_msg}

        # Instantiate the toolkit if data is loaded
        crop_production_toolkit = CropProductionPredictionTool(dataframe=df_prod_global)
        production_tool_initialized = True
        logger.info("Crop Production Toolkit Initialized.")

    if production_tool_initialized and GROQ_AVAILABLE and GEMINI_AVAILABLE:
        class CropProductionAnalysisWorkflow(Workflow):
            ml_agent : Agent = Agent(
                name="Production ML Agent",
                role="Analyzes crop production data using ML tools.",
                tools=[crop_production_toolkit],
                llm=PhiGemini(model="gemini-1.5-flash-latest", api_key=GEMINI_API_KEY), # Use phi's Gemini wrapper
                description="Uses the provided tool to run production analysis for a given crop.",
                instructions=["Use the 'get_feature_importance_and_predict' tool for the crop name provided.", "Return the results."],
                show_tool_calls=True,
            )
            explainer_agent : Agent = Agent(
                name="Production Explainer Agent",
                role="Senior ML engineer explaining agricultural model results simply.",
                llm=PhiGemini(model="gemini-1.5-flash-latest", api_key=GEMINI_API_KEY),
                model = Groq(id="llama-3.3-70b-versatile"), # Pass the configured Groq client instance
                description="Generates user-friendly Markdown explanation from JSON model results.",
                instructions=[
                    "Receive JSON with keys: 'crop_name', 'mae', 'r2_score', 'feature_importances', 'sample_predictions', 'actual_values_for_sample'.",
                    "Generate Markdown explanation:",
                    "1. `## Model Performance for {crop_name}`: Explain MAE ('000 tons error) & R2 (variation explained). Handle `null`s.",
                    "2. `## Key Factors Affecting {crop_name} Production`: List top 3 features. Suggest agricultural relevance. Handle empty list.",
                    "3. `## Prediction Examples`: Comment on sample prediction closeness. Handle `null`s/empty lists.",
                    "4. `## Summary`: 2-3 sentences on model reliability based ONLY on MAE/R2. Use cautious language.",
                    "Tone: Clear, simple, objective, helpful, agricultural context.",
                    "Format: Markdown (`##`, `**`). Insert `{crop_name}`.",
                ],
            )

            def run(self, crop_name: str) -> RunResponse:
                run_id = self.run_id or f"local-streamlit-run-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}"
                logger.info(f"--- Starting Production Workflow Run ID: {run_id} for crop: {crop_name} ---")
                analysis_output = None
                explanation_content = "*Explanation generation failed.*"
                workflow_event = RunEvent.workflow_started

                try:
                    tool_instance = self.ml_agent.tools[0]
                    logger.info(f"Calling production tool for '{crop_name}'...")
                    try:
                        analysis_output = tool_instance.get_feature_importance_and_predict(crop_name=crop_name)
                    except Exception as tool_exc:
                        logger.error(f"Error directly calling production tool: {tool_exc}", exc_info=True)
                        return RunResponse(run_id=run_id, event=RunEvent.FAILED, content={"error": f"Analysis function error: {tool_exc}"})

                    if isinstance(analysis_output, dict) and "error" in analysis_output:
                        logger.error(f"Production analysis function returned error: {analysis_output['error']}")
                        # Return error from analysis step, explanation will be default failure message
                        return RunResponse(run_id=run_id, event=RunEvent.FAILED, content={"prediction_results": analysis_output, "explanation": explanation_content})
                    if not isinstance(analysis_output, dict):
                         logger.error(f"Unexpected output type from production analysis tool: {type(analysis_output)}")
                         return RunResponse(run_id=run_id, event=RunEvent.FAILED, content={"error": f"Analysis failed: Unexpected output type {type(analysis_output)}."})

                    logger.info("Production analysis successful. Proceeding to explanation.")

                    try:
                        analysis_output_json = json.dumps(analysis_output, indent=2, default=str)
                        logger.info("Sending production analysis results to Explainer Agent...")
                        explanation_response = self.explainer_agent.run(analysis_output_json)

                        if isinstance(explanation_response, str):
                            explanation_content = explanation_response
                        elif hasattr(explanation_response, 'content') and isinstance(explanation_response.content, str):
                            explanation_content = explanation_response.content
                        else:
                            logger.warning(f"Could not extract explanation string from production explainer. Type: {type(explanation_response)}")
                            explanation_content = f"*Explanation extraction failed. Agent response type: {type(explanation_response)}*"

                        explanation_content = explanation_content.strip().strip('```markdown').strip('```').strip()
                        logger.info("Successfully processed production explanation.")

                    except Exception as explainer_exc:
                        logger.error(f"Error running Production Explainer Agent: {explainer_exc}", exc_info=True)
                        explanation_content = f"*Failed to generate explanation due to error: {explainer_exc}*"

                    final_content = {"prediction_results": analysis_output, "explanation": explanation_content}
                    workflow_event = RunEvent.workflow_completed
                    logger.info(f"--- Production Workflow Run ID: {run_id} Completed Successfully ---")

                except Exception as workflow_exc:
                    logger.error(f"Unexpected error during production workflow: {workflow_exc}", exc_info=True)
                    workflow_event = RunEvent.FAILED
                    final_content = {"error": f"Unexpected workflow error: {workflow_exc}"}
                    if analysis_output and isinstance(analysis_output, dict) and "error" not in analysis_output:
                        final_content["prediction_results"] = analysis_output
                    final_content["explanation"] = explanation_content # Include potentially failed explanation

                return RunResponse(run_id=run_id, event=workflow_event, content=final_content)

        # Instantiate workflow storage and workflow
        try:
            db_dir = os.path.dirname(DB_FILE)
            if db_dir and not os.path.exists(db_dir): os.makedirs(db_dir)
            workflow_storage = SqlWorkflowStorage(table_name=TABLE_NAME, db_file=DB_FILE)
            crop_production_workflow = CropProductionAnalysisWorkflow(storage=workflow_storage, show_run_logs=True)
            production_workflow_initialized = True
            logger.info("Crop Production Workflow Initialized.")
        except Exception as e:
            st.error(f"Error initializing production workflow/storage: {e}")
            logger.error("Production workflow initialization failed:", exc_info=True)
            production_workflow_initialized = False # Explicitly set to false

    else: # Production data failed to load
        logger.warning("Production data did not load. Production analysis feature will be disabled.")
        production_tool_initialized = False
        production_workflow_initialized = False

except Exception as init_err:
     st.error(f"Major error during App 1 (Production) Initialization: {init_err}")
     logger.error("App 1 Initialization failed:", exc_info=True)
     production_tool_initialized = False
     production_workflow_initialized = False


# ==============================================================================
# App 2: Suitability Mapping - Core Logic
# ==============================================================================
suitability_resources_loaded = False
df_suit_global = None

try:
    # Load suitability resources early
    tf_model_global, encoder_global, scaler_global, error_tf_load = load_tf_resources()
    df_suit_global, error_suit_data = load_suitability_data(SUITABILITY_DATA_PATH)

    suitability_resources_loaded = (
        TF_AVAILABLE and
        tf_model_global is not None and
        encoder_global is not None and
        scaler_global is not None and
        df_suit_global is not None and
        error_tf_load is None and
        error_suit_data is None
    )
    if suitability_resources_loaded:
         logger.info("All resources for Suitability Map loaded successfully.")
    else:
         logger.warning("One or more resources for Suitability Map failed to load. Feature may be disabled.")
         # Log specific errors if they exist
         if error_tf_load: logger.error(f"TF Resource Load Error: {error_tf_load}")
         if error_suit_data: logger.error(f"Suitability Data Load Error: {error_suit_data}")

except Exception as init_err:
     st.error(f"Major error during App 2 (Suitability) Initialization: {init_err}")
     logger.error("App 2 Initialization failed:", exc_info=True)
     suitability_resources_loaded = False

# Define App 2 functions only if resources are ready
if suitability_resources_loaded:
    def predict_using_nearest_features(target_lat, target_lon, dataframe, scaler_obj, model_obj, encoder_obj):
        """Helper to predict suitability based on nearest data point."""
        if dataframe.empty: return None, "DataFrame is empty", None
        try:
            # Calculate distances efficiently using broadcasting
            coords = dataframe[['Latitude', 'Longitude']].values
            target_coords = np.array([[target_lat, target_lon]])
            distances = np.sqrt(np.sum((coords - target_coords)**2, axis=1))

            if distances.size == 0: return None, "No data points to calculate distances from", None
            nearest_idx_pos = np.argmin(distances)
            nearest_idx_label = dataframe.index[nearest_idx_pos] # Get original DataFrame index label
            nearest_row = dataframe.loc[nearest_idx_label]

            input_features_list = [
                nearest_row['TEMPERATURE'], nearest_row['RAINFALL'], nearest_row['FERTILIZER'],
                nearest_row['Pesticides'], target_lat, target_lon
            ]
            input_features = np.array([input_features_list])

            if np.isnan(input_features).any():
                 return None, f"NaN values found in features for nearest row index {nearest_idx_label}", None

            input_scaled = scaler_obj.transform(input_features)
            probabilities_raw = model_obj.predict(input_scaled, verbose=0)[0]
            prob_list_sorted = sorted(
                [(encoder_obj.classes_[i], prob) for i, prob in enumerate(probabilities_raw)],
                key=lambda item: item[1], reverse=True
            )
            result = nearest_row.to_dict()
            result['PREDICTION_PROBS_SORTED'] = prob_list_sorted
            return result, None, nearest_row.to_dict() # Return result, no error, nearest data
        except KeyError as e:
            return None, f"Missing feature column in nearest suitability data: {e}", None
        except Exception as e:
            # Use nearest_idx_label if available, otherwise indicate it couldn't be found
            idx_info = f"nearest row index {nearest_idx_label}" if 'nearest_idx_label' in locals() else "determining nearest row"
            return None, f"Prediction error using {idx_info} for target ({target_lat},{target_lon}): {e}", None

    def generate_suitability_map(model, label_encoder, scaler, df):
        """Generates the Folium map with suitability predictions."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        map_results_summary = { # Summary for the LLM
            "processed_districts": [], "errors": [], "total_processed": 0,
            "total_target": len(district_coord_plotting),
            "probability_threshold": PROBABILITY_THRESHOLD,
            "data_source_file": SUITABILITY_DATA_PATH,
            "model_input_features": FEATURES_FOR_SUITABILITY_MODEL
        }

        coords_array = np.array(list(district_coord_plotting.values()))
        map_center_lat = coords_array[:, 0].mean()
        map_center_lon = coords_array[:, 1].mean()
        fmap = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=7, tiles='OpenStreetMap')
        marker_cluster = MarkerCluster(name="Crop Suitability (Nearest Data)").add_to(fmap)

        processed_districts = 0
        for district_name_target, (lat, lon) in district_coord_plotting.items():
            status_text.text(f"Processing Suitability: {district_name_target} ({processed_districts+1}/{map_results_summary['total_target']})")
            prediction_details, error_msg, nearest_data = predict_using_nearest_features(lat, lon, df, scaler, model, label_encoder)

            district_result = {
                "target_district": district_name_target, "target_coords": (lat, lon),
                "predictions": [], "top_prediction": None, "nearest_data_source": None, "error": error_msg
            }

            if error_msg:
                st.warning(f"Skipping {district_name_target} (Suitability): {error_msg}")
                map_results_summary["errors"].append({"district": district_name_target, "error": error_msg})
                folium.Marker(
                    location=[lat, lon], popup=f"Prediction Failed: {error_msg}",
                    tooltip=f"{district_name_target}: Failed", icon=folium.Icon(color='red', icon='exclamation-sign', prefix='fa')
                ).add_to(marker_cluster)
            elif prediction_details and nearest_data:
                sorted_probs = prediction_details.get('PREDICTION_PROBS_SORTED', [])
                district_result["nearest_data_source"] = {
                    "district": nearest_data.get('DISTRICT', 'N/A'),
                    "coords": (nearest_data.get('Latitude', np.nan), nearest_data.get('Longitude', np.nan)),
                    "temp": nearest_data.get('TEMPERATURE', np.nan), "rain": nearest_data.get('RAINFALL', np.nan),
                    "fert": nearest_data.get('FERTILIZER', np.nan), "pest": nearest_data.get('Pesticides', np.nan)
                }

                if not sorted_probs:
                    st.warning(f"No probabilities returned for {district_name_target}")
                    district_result["error"] = "No probabilities from model."
                    map_results_summary["errors"].append({"district": district_name_target, "error": "No probabilities"})
                else:
                    top_crop_name, top_prob = sorted_probs[0]
                    district_result["top_prediction"] = {"crop": top_crop_name, "probability": top_prob}
                    added_marker_count = 0
                    for i, (crop_name, prob) in enumerate(sorted_probs):
                        if prob >= PROBABILITY_THRESHOLD:
                            color = crop_colors.get(crop_name, 'darkred')
                            is_top_crop = (i == 0)
                            district_result["predictions"].append({"crop": crop_name, "probability": prob})

                            popup_html = f"""<b>Target:</b> {district_name_target} ({lat:.4f}, {lon:.4f})<br><hr>
                                        <b><u>Crop: {crop_name} ({prob*100:.1f}%)</u></b> {"<b>⭐Top</b>" if is_top_crop else ""}<hr>
                                        <b>Based on data near:</b> {nearest_data.get('DISTRICT','N/A')} ({nearest_data.get('Latitude',0):.4f}, {nearest_data.get('Longitude',0):.4f})<br>
                                        T:{nearest_data.get('TEMPERATURE',0):.1f}°C R:{nearest_data.get('RAINFALL',0):.1f}mm F:{nearest_data.get('FERTILIZER',0):.1f} P:{nearest_data.get('Pesticides',0):.1f}"""
                            popup = folium.Popup(popup_html, max_width=300)
                            folium.Marker(location=[lat, lon], popup=popup,
                                          tooltip=f"{district_name_target}: {crop_name} ({prob*100:.1f}%)",
                                          icon=folium.Icon(color=color, icon='leaf', prefix='fa')
                                          ).add_to(marker_cluster)
                            if is_top_crop:
                                folium.Marker(location=[lat, lon], tooltip=f"Top: {top_crop_name}",
                                              icon=folium.Icon(color='black', icon='star', prefix='glyphicon')
                                              ).add_to(fmap) # Add star directly to map
                            added_marker_count += 1
                    if added_marker_count == 0:
                        st.info(f"No crops above {PROBABILITY_THRESHOLD*100:.0f}% threshold for {district_name_target}")
                        district_result["error"] = f"No predictions above threshold."
                        folium.Marker(location=[lat, lon], tooltip="No prediction > threshold",
                                      icon=folium.Icon(color='gray', icon='question-sign', prefix='fa')
                                      ).add_to(marker_cluster)
            else: # Should not happen if predict_using_nearest_features is correct
                 err_msg = "Unknown error getting suitability prediction details"
                 st.warning(f"{err_msg} for {district_name_target}")
                 district_result["error"] = err_msg
                 map_results_summary["errors"].append({"district": district_name_target, "error": err_msg})
                 folium.Marker(location=[lat, lon], popup="Prediction failed", tooltip="Failed",
                               icon=folium.Icon(color='gray', icon='question-sign', prefix='fa')
                               ).add_to(marker_cluster)

            map_results_summary["processed_districts"].append(district_result)
            processed_districts += 1
            map_results_summary["total_processed"] = processed_districts
            progress_bar.progress(processed_districts / map_results_summary['total_target'])

        status_text.text("Adding Legend to Suitability Map...")
        legend_html = '''<div style="position: fixed; bottom: 30px; left: 10px; width: 180px; max-height: 400px;
                        background-color: rgba(255, 255, 255, 0.9); border:2px solid grey; border-radius: 8px;
                        z-index:9999; font-size:12px; padding: 10px; overflow-y: auto; box-shadow: 3px 3px 5px rgba(0,0,0,0.3);">
                        <b>Crop Color Legend</b><hr style="margin: 3px 0;">
                        <span style="font-size: 14px;"><i class="glyphicon glyphicon-star" style="color:black;"></i></span> = Top Predicted Crop<br>
                        <i>(Leaf color = Crop)</i><hr style="margin: 3px 0;">'''
        # Use encoder classes for legend consistency
        sorted_crops_legend = sorted(label_encoder.classes_)
        for crop_name_legend in sorted_crops_legend:
            color_legend = crop_colors.get(crop_name_legend, 'lightgray')
            legend_html += f'<i class="fa fa-square" style="color:{color_legend}; margin-right:5px;"></i>{crop_name_legend}<br>'
        legend_html += '</div>'
        fmap.get_root().html.add_child(folium.Element(legend_html))
        folium.LayerControl().add_to(fmap)

        status_text.text("Suitability Map Generation Complete.")
        progress_bar.empty()
        logger.info("Suitability map generated successfully.")
        return fmap, map_results_summary # Return map and summary

    def explain_suitability_map_with_llm(results_summary, map_html_content):
        """Generates the explanation for the suitability map using Gemini."""
        st.subheader("💡 Understanding the Suitability Map (AI Generated)")
        if not GEMINI_AVAILABLE or gemini_direct_model is None:
            st.warning("Gemini AI for map explanation not available. Showing fallback.")
            st.markdown("""*(Fallback Explanation)* This map shows potential crop suggestions for districts in Tamil Nadu based on an AI model. Predictions use data from the *nearest* location in our reference dataset. Colored leaves show crops with >5% predicted probability, black star marks the top prediction. Hover/Click for details. Use the legend.""")
            return

        summary_text = f"Map Summary: Processed {results_summary['total_processed']}/{results_summary['total_target']} districts. Features: {', '.join(results_summary['model_input_features'])}. Data: {os.path.basename(results_summary['data_source_file'])}. Threshold: {results_summary['probability_threshold']*100:.0f}%.\n"
        successful_preds = [d for d in results_summary['processed_districts'] if not d['error'] and d['top_prediction']]
        if successful_preds:
            summary_text += "- Examples (Top Crop): " + ", ".join([f"{d['target_district']}: {d['top_prediction']['crop']} ({d['top_prediction']['probability']*100:.1f}%)" for d in successful_preds[:3]]) + "\n"
        if results_summary["errors"]:
            summary_text += f"- {len(results_summary['errors'])} errors/warnings (e.g., {results_summary['errors'][0]['district']}: {results_summary['errors'][0]['error']})\n"

        map_description = """Visual map elements: Base map, clustered markers per district, multiple colored 'leaf' icons (crop color), single 'black star' (top prediction), popups with details (target, prediction, probability, nearest data source info), legend (bottom-left), potential error markers (red/gray icons)."""

        prompt = f"""You are an agricultural assistant explaining a crop suitability map for Tamil Nadu simply for farmers/planners.
        --- MAP GENERATION SUMMARY ---
        {summary_text}
        --- VISUAL MAP DESCRIPTION ---
        {map_description}
        ---
        Based *only* on the summary and description, explain:
        1. **What is this map?** (Purpose: Crop suitability suggestions)
        2. **How were suggestions made?** (AI model using Temp, Rain, Fert, Pest, Location. **Critically, explain prediction for district 'X' uses data from the *nearest available reference point*, not necessarily from within district 'X' itself.**)
        3. **How to read the map?** (Markers: Leaf colors = crops >{results_summary['probability_threshold']*100:.0f}% chance, Black star = most likely crop, Hover/Click for details, Legend use, Error icons).
        4. **Main takeaway?** (Guidance tool showing potential/most likely crops based on nearest data.)
        Keep it concise, friendly, use lists/bullets. Refer to 'leaf', 'star', 'legend'.
        """
        try:
            with st.spinner("🤖 Asking Gemini AI to explain the map..."):
                 response = gemini_direct_model.generate_content(prompt)
                 # Handle potential safety blocks or empty responses
                 explanation = response.text if hasattr(response, 'text') else "*AI explanation could not be generated.*"
                 if not explanation.strip():
                      explanation = "*AI explanation was empty. Check content safety filters or prompt.*"
                      logger.warning(f"Gemini response empty. Prompt: {prompt[:500]}... Full response: {response}")

            st.markdown(explanation)
            logger.info("Suitability map explanation generated successfully.")

        except Exception as e:
            st.error(f"Error calling Gemini API for map explanation: {e}")
            logger.error(f"Gemini API Error (Map Explanation): {e}", exc_info=True)
            st.markdown("*(Fallback Explanation due to AI error)* This map shows potential crop suggestions...") # Fallback
else:
    # Define dummy functions if resources didn't load, so the UI doesn't crash trying to call them
     logger.warning("Defining dummy functions for App 2 as resources are not loaded.")
     def generate_suitability_map(*args, **kwargs):
         st.error("Suitability map generation is disabled due to missing resources.")
         logger.error("generate_suitability_map called but resources not loaded.")
         return None, None
     def explain_suitability_map_with_llm(*args, **kwargs):
         st.warning("Suitability map explanation is disabled.")
         logger.warning("explain_suitability_map_with_llm called but resources/API not available.")
         return None

# ==============================================================================
# Streamlit UI Layout
# ==============================================================================

st.title("🧑‍🌾 Integrated Agricultural AI Assistant")
st.markdown("""
Welcome! This tool offers two analysis options:
1.  **Crop Production Analysis:** Predicts *how much* of a specific crop might be produced based on historical factors using an agentic workflow.
2.  **Crop Suitability Map:** Visualizes *which* crops are likely suitable for different districts in Tamil Nadu based on environmental data and a deep learning model.
Choose a tab below to begin.
""")
st.markdown("---")

# --- Create Tabs ---
tab1, tab2 = st.tabs(["📈 Crop Production Analysis", "🗺️ Crop Suitability Map"])

# ========================== TAB 1: Production Analysis =========================
with tab1:
    st.header("📈 Crop Production Analysis (ICRISAT Data)")

    # Check if App 1 prerequisites are met
    app1_ready = production_workflow_initialized and crop_list_prod_global and GROQ_AVAILABLE and GEMINI_AVAILABLE
    if not app1_ready:
        st.warning("Production Analysis Unavailable.")
        if not production_data_loaded: st.error("- Production data failed to load.")
        if not crop_list_prod_global: st.error("- Could not extract crop list from production data.")
        if not production_tool_initialized: st.error("- Production analysis tool failed to initialize.")
        if not production_workflow_initialized: st.error("- Production analysis workflow failed to initialize.")
        if not GROQ_AVAILABLE: st.error("- Groq API key missing or invalid.")
        if not GEMINI_AVAILABLE: st.error("- Gemini API key missing or invalid.")
    else:
        st.markdown("""
        Select a crop to predict its production quantity (in '000 Tons) using historical data like Fertilizer, Rainfall, Pesticides, Area, and Yield. The system uses AI agents to perform the analysis and generate an explanation.
        """)

        selected_crop_prod = st.selectbox(
            "Select Crop for Production Analysis:",
            options=crop_list_prod_global,
            index=0, # Default to first crop
            key="production_crop_select"
        )

        analyze_prod_button = st.button(f"Analyze {selected_crop_prod} Production", key=f"analyze_prod_button_{selected_crop_prod}", type="primary")

        if analyze_prod_button:
            st.info(f"🚀 Starting production analysis for {selected_crop_prod}...")
            with st.spinner(f"Analyzing {selected_crop_prod} Production... This may take a moment."):
                try:
                    # Run the production workflow
                    response = crop_production_workflow.run(crop_name=selected_crop_prod)
                    logger.info(f"Production Workflow Response Event: {response.event}, Content Type: {type(response.content)}")

                    st.markdown("---") # Separator

                    if not isinstance(response.content, dict):
                        st.error("❌ Production workflow returned unexpected content format.")
                        logger.error(f"Production workflow unexpected content: {response.content}")
                    else:
                        results = response.content.get("prediction_results")
                        explanation = response.content.get("explanation", "*No explanation available.*")

                        # Display results or errors from the analysis step
                        if isinstance(results, dict) and "error" in results:
                             st.error(f"❌ Production Analysis Error for {selected_crop_prod}:")
                             st.error(results["error"])
                             logger.error(f"Production analysis step failed: {results['error']}")
                             st.subheader("💬 AI Explanation Status")
                             st.markdown(explanation)
                        elif isinstance(results, dict) and "mae" in results: # Success case
                            st.success(f"✅ Production Analysis Complete for {results.get('crop_name', selected_crop_prod)}!")
                            st.subheader(f"📊 Results Summary")
                            col1, col2 = st.columns(2)
                            with col1:
                                mae_val = results.get('mae')
                                st.metric("Mean Absolute Error (MAE)", f"{mae_val:.2f} ('000 tons)" if mae_val is not None else "N/A", help="Average prediction error (thousand tons). Lower is better.")
                            with col2:
                                r2_val = results.get('r2_score')
                                st.metric("R² Score", f"{r2_val:.3f}" if r2_val is not None else "N/A", help="Model's ability to explain production variation (1.0 = perfect). Higher is better.")

                            st.subheader("⚙️ Key Factors (Feature Importance)")
                            importances = results.get("feature_importances")
                            if importances and isinstance(importances, list) and len(importances) > 0:
                                try:
                                    importance_df = pd.DataFrame(importances)
                                    importance_df['Importance'] = pd.to_numeric(importance_df['Importance']).fillna(0.0)
                                    st.dataframe(importance_df.style.format({"Importance": "{:.4f}"}), use_container_width=True, hide_index=True,
                                                 column_config={"Feature": st.column_config.TextColumn("Factor"), "Importance": st.column_config.NumberColumn("Importance Score", help="Relative influence on prediction.")})
                                except Exception as df_e:
                                     st.warning("Could not display production feature importances table.")
                                     logger.error(f"Error displaying importance DF: {df_e}")
                                     st.json(importances) # Fallback
                            else:
                                st.write("Production feature importances not available.")

                            st.subheader("🔮 Sample Predictions vs. Actual (Production)")
                            predictions = results.get("sample_predictions")
                            actuals = results.get("actual_values_for_sample")
                            if predictions and actuals and isinstance(predictions, list) and isinstance(actuals, list) and len(predictions) == len(actuals) > 0:
                                try:
                                    valid_samples = [{'Actual Prod. (\'000 tons)': a, 'Predicted Prod. (\'000 tons)': p} for a, p in zip(actuals, predictions) if a is not None and p is not None]
                                    if valid_samples:
                                        sample_df = pd.DataFrame(valid_samples)
                                        sample_df['Difference (\'000 tons)'] = sample_df['Predicted Prod. (\'000 tons)'] - sample_df['Actual Prod. (\'000 tons)']
                                        st.dataframe(sample_df.style.format("{:.2f}"), use_container_width=True, hide_index=True,
                                                     column_config={"Actual Prod. ('000 tons)": st.column_config.NumberColumn("Actual"), "Predicted Prod. ('000 tons)": st.column_config.NumberColumn("Predicted"), "Difference ('000 tons)": st.column_config.NumberColumn("Difference")})
                                    else:
                                        st.write("Samples contained non-numeric/missing values.")
                                except Exception as df_e:
                                     st.warning("Could not display production sample predictions table.")
                                     logger.error(f"Error displaying sample DF: {df_e}")
                                     st.write("Actuals:", actuals); st.write("Predictions:", predictions) # Fallback
                            else:
                                st.write("Production sample predictions/actuals not available.")

                            st.subheader("💬 AI-Generated Explanation (Production)")
                            if explanation and isinstance(explanation, str):
                                st.markdown(explanation)
                            else:
                                st.warning("Production explanation not available or not a string.")
                        else: # Invalid results structure
                            st.error("❌ Production analysis results have an invalid structure.")
                            logger.error(f"Invalid production results structure: {results}")
                            st.subheader("💬 AI Explanation Status")
                            st.markdown(explanation) # Show explanation status anyway

                except Exception as e:
                    st.error(f"An unexpected error occurred during Production Analysis UI processing: {e}")
                    logger.error("Streamlit UI Error (Production Tab):", exc_info=True)

# ========================== TAB 2: Suitability Map ============================
with tab2:
    st.header("🗺️ Crop Suitability Map (Tamil Nadu)")

    # Check if App 2 prerequisites are met
    app2_ready = suitability_resources_loaded and GEMINI_AVAILABLE
    if not app2_ready:
        st.warning("Crop Suitability Map Unavailable.")
        if not TF_AVAILABLE: st.error("- TensorFlow/Keras is not installed.")
        if not suitability_resources_loaded: st.error("- Required models, data, or other resources failed to load (check logs).")
        if not GEMINI_AVAILABLE: st.error("- Gemini API key for explanation is missing or invalid.")
    else:
        st.markdown("""
        Generate an interactive map showing potential crop suitability for districts across Tamil Nadu. The predictions are based on a Deep Learning model using environmental factors (Temperature, Rainfall, Fertilizer, Pesticides) from the *nearest available data point* in the reference dataset.
        """)

        # Map Generation Button
        generate_map_button = st.button("Generate Crop Suitability Map", key="generate_map_button", type="primary")

        # Use session state to keep map generated across reruns after button press
        if 'map_generated' not in st.session_state:
            st.session_state.map_generated = False
        if 'folium_map_obj' not in st.session_state:
            st.session_state.folium_map_obj = None
        if 'map_results_summary_obj' not in st.session_state:
            st.session_state.map_results_summary_obj = None
        if 'map_html_content_val' not in st.session_state:
            st.session_state.map_html_content_val = None

        if generate_map_button:
            st.session_state.map_generated = False # Reset on new click
            st.session_state.folium_map_obj = None
            st.session_state.map_results_summary_obj = None
            st.session_state.map_html_content_val = None

            st.info("Generating suitability map for all districts...")
            with st.spinner("Generating map, please wait..."):
                try:
                    # Generate the map and get the summary
                    folium_map, map_results_summary = generate_suitability_map(
                        tf_model_global, encoder_global, scaler_global, df_suit_global
                    )

                    if folium_map and map_results_summary:
                        st.success("✅ Suitability Map Generated Successfully!")
                        st.session_state.map_generated = True
                        st.session_state.folium_map_obj = folium_map
                        st.session_state.map_results_summary_obj = map_results_summary

                        # Save map to HTML and store content in session state
                        try:
                            folium_map.save(OUTPUT_HTML_PATH)
                            logger.info(f"Suitability map saved as '{OUTPUT_HTML_PATH}'")
                            with open(OUTPUT_HTML_PATH, 'r', encoding='utf-8') as f:
                                st.session_state.map_html_content_val = f.read()
                        except Exception as e:
                            st.error(f"Error saving or reading map HTML: {e}")
                            logger.error(f"Error saving/reading map HTML: {e}", exc_info=True)
                            st.session_state.map_html_content_val = None
                    else:
                        st.error("❌ Suitability map generation failed. Check logs and warnings.")
                        logger.error("generate_suitability_map returned None for map or summary.")
                        st.session_state.map_generated = False

                except Exception as map_gen_e:
                    st.error(f"An unexpected error occurred during map generation: {map_gen_e}")
                    logger.error("Error during generate_suitability_map call:", exc_info=True)
                    st.session_state.map_generated = False

        # Display map and explanation IF it has been generated successfully in this session
        if st.session_state.map_generated and st.session_state.folium_map_obj:
            st.subheader("📍 Interactive Crop Suitability Map")
            if st.session_state.map_html_content_val:
                st.components.v1.html(st.session_state.map_html_content_val, height=600, scrolling=True)
            else:
                # Fallback: Try rendering the object directly (less reliable)
                try:
                    st.components.v1.html(st.session_state.folium_map_obj._repr_html_(), height=600, scrolling=True)
                    st.warning("Displayed map directly from object; HTML file might have issues or wasn't read.")
                except Exception as e:
                    st.error(f"Could not display map inline: {e}")

            # Display the explanation (call the function)
            if st.session_state.map_results_summary_obj:
                 explain_suitability_map_with_llm(st.session_state.map_results_summary_obj, st.session_state.map_html_content_val)
            else:
                 st.error("Map generated, but summary data for explanation is missing.")

# ==============================================================================
# Footer
# ==============================================================================
st.markdown("---")
st.caption("MAJOR PROJECT BY HARISH BALAJI V AND HARIHARAN K | Combined Agricultural AI Tool")