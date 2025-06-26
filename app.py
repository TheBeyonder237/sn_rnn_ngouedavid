import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import io
import os
import logging
import requests
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import uuid
import scipy.stats as stats

# Configure logging to suppress ticker-specific messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.handlers = []  # Clear default handlers
logger.addHandler(logging.StreamHandler())  # Add custom handler
logger.propagate = False  # Prevent propagation to root logger

# Lottie animation loader
def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        logger.error(f"Failed to load Lottie animation: {e}")
        return None

# Load Lottie animations
main_animation = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_4kx2q32n.json")
loading_animation = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_p8bfn5to.json")
about_animation = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_0yfsb3a1.json")

# Page configuration
st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS with improved text visibility
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600;700&display=swap');
    * { font-family: 'Quicksand', sans-serif; }
    .stApp {
        background: linear-gradient(135deg, #1e293b 0%, #334155 60%, #475569 100%) !important;
        min-height: 100vh;
    }
    .section-card, .visual-card, .metric-card {
        background: rgba(30, 41, 59, 0.95);
        border-radius: 24px;
        box-shadow: 0 8px 32px rgba(71, 85, 105, 0.2);
        padding: 2.5rem;
        margin-bottom: 2rem;
        transition: transform 0.3s, box-shadow 0.3s;
        backdrop-filter: blur(10px);
    }
    .section-card:hover, .visual-card:hover, .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(71, 85, 105, 0.3);
    }
    .section-title {
        color: #bae6fd;
        font-size: 2.4em;
        font-weight: 700;
        margin-bottom: 0.6em;
        letter-spacing: 0.5px;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }
    .badge {
        background: linear-gradient(90deg, #475569, #64748b);
        color: #f8fafc;
        border-radius: 12px;
        padding: 0.5em 1.2em;
        font-size: 1.2em;
        font-weight: 600;
        margin: 0.3em;
        display: inline-block;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }
    .stButton>button {
        background: linear-gradient(90deg, #475569, #64748b);
        color: #f8fafc;
        border-radius: 24px;
        padding: 0.9em 2.2em;
        font-size: 1.2em;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(71, 85, 105, 0.2);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #64748b, #475569);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(71, 85, 105, 0.3);
    }
    .stTextInput > div > div > input, .stSelectbox > div > div {
        background: #1e293b !important;
        color: #f8fafc !important;
        border: 2px solid #475569;
        border-radius: 12px;
        padding: 0.8em;
        font-size: 1.1em;
    }
    .stTextInput > div > div > input:focus, .stSelectbox > div > div:focus {
        border-color: #bae6fd;
        box-shadow: 0 0 0 3px rgba(71, 85, 105, 0.2);
    }
    .stSelectbox label {
        color: #bae6fd !important;
        font-size: 1.1em;
    }
    .stSelectbox div[role="option"] {
        background: #334155 !important;
        color: #f8fafc !important;
        font-size: 1.1em;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #334155, #475569);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(71, 85, 105, 0.2);
    }
    .section-sep {
        border: none;
        border-top: 2px solid #64748b;
        margin: 2em 0;
        width: 90%;
    }
    .card-fade {
        opacity: 0;
        transform: translateY(20px);
        animation: fadeInUp 0.6s ease forwards;
    }
    @keyframes fadeInUp {
        to { opacity: 1; transform: translateY(0); }
    }
    .about-avatar {
        border-radius: 50%;
        border: 3px solid #475569;
        box-shadow: 0 4px 12px rgba(71, 85, 105, 0.2);
    }
    .about-contact-btn {
        background: linear-gradient(90deg, #475569, #64748b);
        color: #f8fafc;
        border-radius: 16px;
        padding: 0.6em 1.6em;
        border: none;
        font-weight: 600;
        margin: 0.5em;
        transition: all 0.3s;
        font-size: 1.1em;
    }
    .about-contact-btn:hover {
        background: linear-gradient(90deg, #64748b, #475569);
        transform: translateY(-2px);
    }
    .stAlert {
        border-radius: 12px;
        background: rgba(71, 85, 105, 0.3);
        color: #f8fafc;
        font-size: 1.1em;
    }
    p, li, .stMarkdown {
        color: #f8fafc !important;
        font-size: 1.1em;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Model hyperparameters
MODEL_PARAMS = {
    'LSTM': {'seq_length': 20, 'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2, 'lr': 0.001},
    'GRU': {'seq_length': 20, 'hidden_size': 128, 'num_layers': 3, 'dropout': 0.2, 'lr': 0.001},
    'Bidirectional LSTM': {'seq_length': 20, 'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3, 'lr': 0.001},
    'Bidirectional GRU': {'seq_length': 20, 'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3, 'lr': 0.001},
    'CNN-LSTM': {'seq_length': 20, 'hidden_size': 64, 'num_layers': 2, 'dropout': 0.3, 'lr': 0.001}
}

# Cached data download function (quiet mode)
@st.cache_data(show_spinner=False)
def download_data(_self, ticker, start_date, end_date):
    try:
        # Suppress yfinance progress bar
        yf.pdr_override()
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
        if data.empty:
            raise ValueError(f"No data found for ticker")
        logger.info("Data downloaded successfully")
        return data
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        st.error(f"❌ Failed to download data: {e}", icon="❌")
        return None

# DataLoader class
class DataLoader:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.scaler = MinMaxScaler()

    def download_data(self):
        return download_data(self, self.ticker, self.start_date, self.end_date)

    def preprocess_data(self, data):
        data = data.ffill()
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        if 'Adj Close' not in data.columns:
            data['Adj Close'] = data['Close']
        features.append('Adj Close')
        data = data[features]
        return pd.DataFrame(self.scaler.fit_transform(data), columns=features, index=data.index)

    def create_sequences(self, data, seq_length):
        X, y = [], []
        price_column = 'Adj Close'
        for i in range(len(data) - seq_length):
            X.append(data.iloc[i:i + seq_length].values)
            y.append(data.iloc[i + seq_length][price_column])
        return np.array(X), np.array(y)

# RNN Model classes
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, model_type='LSTM', dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = (nn.LSTM if model_type == 'LSTM' else nn.GRU)(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

class BidirectionalRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, model_type='LSTM', dropout=0.3):
        super().__init__()
        self.rnn = (nn.LSTM if model_type == 'LSTM' else nn.GRU)(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size=3, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, input_size, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# FinancialDataEDA class
class FinancialDataEDA:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.returns = None

    def download_data(self):
        try:
            self.data = download_data(self, self.ticker, self.start_date, self.end_date)
            if self.data.empty:
                raise ValueError(f"No data found for ticker")
            self.data = self.data.ffill()
            logger.info("EDA data downloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error downloading EDA data: {e}")
            st.error(f"❌ Failed to download data: {e}", icon="❌")
            return False

    def get_price_column(self):
        return 'Adj Close' if 'Adj Close' in self.data.columns else 'Close'

    def calculate_returns(self):
        price_column = self.get_price_column()
        self.returns = self.data[price_column].pct_change().dropna()

    def basic_statistics(self):
        if self.data is None or self.data.empty:
            return pd.DataFrame({
                'Statistique': ['Mean', 'Std Dev', 'Min', 'Max', 'Median', 'Skewness', 'Kurtosis'],
                'Prix': [np.nan] * 7,
                'Rendements': [np.nan] * 7
            })
        price_column = self.get_price_column()
        price_stats = [
            float(self.data[price_column].mean().iloc[0]),
            float(self.data[price_column].std().iloc[0]),
            float(self.data[price_column].min().iloc[0]),
            float(self.data[price_column].max().iloc[0]),
            float(self.data[price_column].median().iloc[0]),
            float(stats.skew(self.data[price_column], nan_policy='omit')[0]),
            float(stats.kurtosis(self.data[price_column], nan_policy='omit')[0])
        ]
        returns_stats = [
            float(self.returns.mean().iloc[0]) if self.returns is not None and not self.returns.empty else np.nan,
            float(self.returns.std().iloc[0]) if self.returns is not None and not self.returns.empty else np.nan,
            float(self.returns.min().iloc[0]) if self.returns is not None and not self.returns.empty else np.nan,
            float(self.returns.max().iloc[0]) if self.returns is not None and not self.returns.empty else np.nan,
            float(self.returns.median().iloc[0]) if self.returns is not None and not self.returns.empty else np.nan,
            float(stats.skew(self.returns, nan_policy='omit')[0]) if self.returns is not None and not self.returns.empty else np.nan,
            float(stats.kurtosis(self.returns, nan_policy='omit')[0]) if self.returns is not None and not self.returns.empty else np.nan
        ]
        return pd.DataFrame({
            'Statistique': ['Mean', 'Std Dev', 'Min', 'Max', 'Median', 'Skewness', 'Kurtosis'],
            'Prix': price_stats,
            'Rendements': returns_stats
        })

    def plot_price_evolution(self):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=self.data.index,
            open=self.data['Open'],
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data[self.get_price_column()],
            name='OHLC'
        ))
        fig.update_layout(
            title="Price Evolution",
            yaxis_title="Price ($)",
            xaxis_title="Date",
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig

    def plot_volume(self):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=self.data.index, y=self.data['Volume'], name='Volume'))
        fig.update_layout(
            title="Trading Volume",
            yaxis_title="Volume",
            xaxis_title="Date",
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig

    def plot_returns_distribution(self):
        if self.returns is None or self.returns.empty:
            return None
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Returns Distribution', 'QQ-Plot'))
        fig.add_trace(go.Histogram(x=self.returns, name='Returns', nbinsx=50), row=1, col=1)
        returns_sorted = np.sort(self.returns.dropna())
        n = len(returns_sorted)
        if n < 2:
            return None
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))
        fig.add_trace(go.Scatter(x=theoretical_quantiles, y=returns_sorted, mode='markers', name='QQ-Plot'), row=2, col=1)
        fig.add_trace(go.Scatter(x=[min(theoretical_quantiles.min(), returns_sorted.min()), max(theoretical_quantiles.max(), returns_sorted.max())],
                                y=[min(theoretical_quantiles.min(), returns_sorted.min()), max(theoretical_quantiles.max(), returns_sorted.max())],
                                mode='lines', name='Reference Line', line=dict(color='red', dash='dash')), row=2, col=1)
        fig.update_layout(title="Returns Analysis", template='plotly_white', height=600)
        return fig

    def correlation_analysis(self):
        corr_matrix = self.data.corr()
        fig = go.Figure(data=go.Heatmap(z=corr_matrix, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='RdBu'))
        fig.update_layout(title="Correlation Matrix", template='plotly_white')
        return fig, corr_matrix

    def volatility_analysis(self):
        if self.returns is None or self.returns.empty:
            return None
        volatility = self.returns.rolling(window=20).std() * np.sqrt(252)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=volatility.index, y=volatility, name='Annualized Volatility'))
        fig.update_layout(title="Volatility (20-day)", yaxis_title="Volatility", xaxis_title="Date", template='plotly_white')
        return fig

    def generate_report(self):
        if not self.download_data():
            return None
        self.calculate_returns()
        stats_df = self.basic_statistics()
        stats_md = "| Statistique | Prix | Rendements |\n|------------|------|------------|\n"
        for _, row in stats_df.iterrows():
            prix = '-' if pd.isna(row['Prix']) else f"{row['Prix']:.4f}"
            rendements = '-' if pd.isna(row['Rendements']) else f"{row['Rendements']:.4f}"
            stats_md += f"| {row['Statistique']} | {prix} | {rendements} |\n"
        price_column = self.get_price_column()
        columns_str = [str(col) for col in self.data.columns]
        report = (
            f"# Financial Data Report\n\n"
            f"## Analysis Period\n"
            f"- Start Date: {self.start_date.strftime('%Y-%m-%d')}\n"
            f"- End Date: {self.end_date.strftime('%Y-%m-%d')}\n\n"
            f"## Basic Statistics\n{stats_md}\n"
            f"## Data Insights\n"
            f"- Total Trading Days: {len(self.data)}\n"
            f"- Missing Values: {self.data.isnull().sum().sum()}\n"
            f"- Price Range: ${float(self.data[price_column].min().iloc[0]):.2f} - ${float(self.data[price_column].max().iloc[0]):.2f}\n"
            f"- Available Columns: {', '.join(columns_str)}\n"
        )
        return report

# Model loading function
@st.cache_resource
def load_model(model_name, input_size, params):
    try:
        if input_size != 6:
            raise ValueError(f"Expected 6 input features, got {input_size}. Ensure data includes 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_name.startswith('Bidirectional'):
            model_type = 'LSTM' if 'LSTM' in model_name else 'GRU'
            model = BidirectionalRNNModel(input_size, params['hidden_size'], params['num_layers'], model_type, params['dropout']).to(device)
        elif model_name == 'CNN-LSTM':
            model = CNNLSTMModel(input_size, params['hidden_size'], params['num_layers']).to(device)
        else:
            model = RNNModel(input_size, params['hidden_size'], params['num_layers'], model_type=model_name, dropout=params['dropout']).to(device)
        model_path = f"saved_models/best_model_{model_name}.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Ensure all model files are in the 'saved_models' directory.")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info(f"Loaded model {model_name}")
        return model, device
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        st.error(f"❌ Failed to load model {model_name}: {e}", icon="❌")
        return None, None

# Data Processing Badge
def display_data_processing_badge():
    st.markdown(
        """
        <style>
        .dp-badge {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0.6em;
            margin-bottom: 1.5rem;
            animation: fadeIn 1s ease;
        }
        .dp-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: linear-gradient(135deg, #bae6fd, #f8fafc);
            margin-right: 8px;
            box-shadow: 0 0 8px #bae6fd;
            animation: pulse 1.5s infinite;
        }
        .dp-text {
            font-weight: 600;
            color: #f8fafc;
            font-size: 1.1em;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 8px #bae6fd; }
            50% { box-shadow: 0 0 16px #bae6fd; }
            100% { box-shadow: 0 0 8px #bae6fd; }
        }
        </style>
        <div class="dp-badge">
            <div class="dp-dot"></div>
            <span class="dp-text">Data Processing: <span style="color:#00b894">ACTIVE</span></span>
        </div>
        """, unsafe_allow_html=True
    )

# Sidebar
with st.sidebar:
    if main_animation:
        st_lottie(main_animation, height=100, key="sidebar_animation")
    else:
        st.markdown("<p style='color:#f8fafc; text-align:center;'>Stock Analyzer</p>", unsafe_allow_html=True)
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "EDA", "Predictions", "About"],
        icons=['house', 'graph-up', 'lightning-charge', 'info-circle'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background": "linear-gradient(135deg, #334155, #475569)", "border-radius": "16px", "padding": "0.5rem"},
            "icon": {"color": "#bae6fd", "font-size": "20px"},
            "nav-link": {"color": "#f8fafc", "font-size": "16px", "padding": "10px", "border-radius": "12px"},
            "nav-link-selected": {"background": "linear-gradient(90deg, #64748b, #475569)", "color": "#bae6fd"},
        }
    )
    st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)
    st.markdown("<div class='section-card'><h3 style='color:#bae6fd; font-size:1.4em;'>Configuration</h3></div>", unsafe_allow_html=True)
    with st.form("config_form"):
        config_key = st.text_input("API Key (Optional)", type="password", placeholder="Enter optional API key")
        if st.form_submit_button("Validate", use_container_width=True):
            st.success("✅ Configuration validated (simulated)", icon="✅")
    st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #f8fafc; font-size: 0.9em;'>
        Developed by Ngôue David<br>
        <a href='mailto:ngouedavidrogeryannick@gmail.com' style='color:#bae6fd;'>📧 Email</a><br>
        <a href='https://github.com/TheBeyonder237' style='color:#bae6fd;'>🌐 GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# Main application
def main():
    display_data_processing_badge()
    
    if selected == "Home":
        st.markdown("""
        <div class='section-card card-fade' style='text-align: center; max-width: 1000px; margin: auto;'>
            <h1 class='section-title'>📈 Stock Analyzer</h1>
            <p style='color: #f8fafc; font-size: 1.3em; margin-bottom: 1em;'>Advanced Financial Analysis Powered by AI</p>
            <hr class='section-sep'/>
            <p style='color: #f8fafc; font-size: 1.1em;'>Explore stock data and predict future trends with cutting-edge RNN models.</p>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.markdown("""
            <div class='section-card card-fade'>
                <h3 style='color:#bae6fd; font-size:1.5em;'>Mission</h3>
                <p style='color:#f8fafc;'>Deliver actionable insights and accurate predictions for financial markets using deep learning.</p>
            </div>
            <div class='section-card card-fade'>
                <h3 style='color:#bae6fd; font-size:1.5em;'>Technologies</h3>
                <span class='badge'>yfinance</span>
                <span class='badge'>PyTorch</span>
                <span class='badge'>Plotly</span>
                <span class='badge'>Streamlit</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if main_animation:
                st_lottie(main_animation, height=200, key="home_animation")
            else:
                st.markdown("<p style='color:#f8fafc; text-align:center;'>Welcome to Stock Analyzer</p>", unsafe_allow_html=True)
            st.markdown("""
            <div class='metric-card card-fade' style='text-align: center;'>
                <h3 style='color:#bae6fd; margin:0;'>100+</h3>
                <p style='color:#f8fafc; margin:0;'>Analyzable Assets</p>
            </div>
            """, unsafe_allow_html=True)

    elif selected == "EDA":
        st.markdown("""
        <div class='section-card card-fade' style='max-width: 1000px; margin: auto;'>
            <h1 class='section-title'>📊 Exploratory Data Analysis</h1>
            <span class='badge'>yfinance</span>
            <span class='badge'>Plotly</span>
            <hr class='section-sep'/>
            <p style='color:#f8fafc;'>Analyze historical stock data with comprehensive visualizations and statistics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("eda_form"):
            col1, col2 = st.columns([2, 1])
            with col1:
                ticker = st.text_input("Ticker Symbol", value="TSLA", placeholder="e.g., TSLA")
            with col2:
                period = st.selectbox("Period", ["1 Year", "2 Years", "3 Years", "Custom"])
            if period == "Custom":
                col3, col4 = st.columns(2)
                with col3:
                    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=2*365))
                with col4:
                    end_date = st.date_input("End Date", datetime.now())
            else:
                years = {"1 Year": 1, "2 Years": 2, "3 Years": 3}[period]
                end_date = datetime.now()
                start_date = end_date - timedelta(days=years*365)
            submit = st.form_submit_button("Analyze", use_container_width=True)

        if submit:
            if not ticker.strip():
                st.error("❌ Please enter a valid ticker symbol.", icon="❌")
                return
            with st.spinner("Processing data..."):
                if loading_animation:
                    st_lottie(loading_animation, height=80, key=f"eda_loading_{uuid.uuid4()}")
                else:
                    st.markdown("<p style='color:#f8fafc; text-align:center;'>Loading...</p>", unsafe_allow_html=True)
                eda = FinancialDataEDA(ticker, start_date, end_date)
                report = eda.generate_report()
                if report:
                    st.markdown("<div class='visual-card card-fade'>", unsafe_allow_html=True)
                    st.markdown("<h3 style='color:#bae6fd;'>Analysis Report</h3>", unsafe_allow_html=True)
                    st.markdown(report, unsafe_allow_html=True)
                    st.download_button(
                        label="Download Report",
                        data=report.encode('utf-8'),
                        file_name=f"eda_report_{ticker}.md",
                        mime="text/markdown",
                        use_container_width=True,
                        key=f"download_report_{uuid.uuid4()}"
                    )
                    st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)
                    st.markdown("<h3 style='color:#bae6fd;'>Data Preview</h3>", unsafe_allow_html=True)
                    st.dataframe(eda.data.head(), use_container_width=True)
                    
                    for title, plot_func, filename in [
                        ("Price Evolution", eda.plot_price_evolution, "price_evolution"),
                        ("Trading Volume", eda.plot_volume, "volume"),
                        ("Returns Analysis", eda.plot_returns_distribution, "returns"),
                        ("Correlation Matrix", lambda: eda.correlation_analysis()[0], "correlation"),
                        ("Volatility Analysis", eda.volatility_analysis, "volatility")
                    ]:
                        st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)
                        st.markdown(f"<h3 style='color:#bae6fd;'>{title}</h3>", unsafe_allow_html=True)
                        fig = plot_func()
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            try:
                                img_buffer = io.BytesIO()
                                fig.write_image(img_buffer, format="png")
                                st.download_button(
                                    label=f"Download {title}",
                                    data=img_buffer,
                                    file_name=f"{filename}_{ticker}.png",
                                    mime="image/png",
                                    use_container_width=True,
                                    key=f"download_{filename}_{uuid.uuid4()}"
                                )
                            except Exception as e:
                                st.warning(f"⚠️ Image export failed: {e}. Install 'kaleido' for PNG export.", icon="⚠️")
                    st.success("✅ Analysis completed!", icon="✅")
                    st.markdown("</div>", unsafe_allow_html=True)

    elif selected == "Predictions":
        st.markdown("""
        <div class='section-card card-fade' style='max-width: 1000px; margin: auto;'>
            <h1 class='section-title'>🔮 Future Predictions</h1>
            <span class='badge'>RNN</span>
            <span class='badge'>PyTorch</span>
            <hr class='section-sep'/>
            <p style='color:#f8fafc;'>Predict future stock prices using advanced RNN models.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("pred_form"):
            col1, col2 = st.columns([2, 1])
            with col1:
                ticker = st.text_input("Ticker Symbol", value="TSLA", placeholder="e.g., TSLA")
            with col2:
                pred_weeks = st.slider("Prediction Horizon (Weeks)", 1, 12, 3)
            model = st.selectbox("Select Model", list(MODEL_PARAMS.keys()))
            submit = st.form_submit_button("Predict", use_container_width=True)

        st.markdown("<div class='section-card card-fade'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#bae6fd;'>Model Parameters</h3>", unsafe_allow_html=True)
        params = MODEL_PARAMS[model]
        st.markdown(f"<span class='badge'>{model}</span>", unsafe_allow_html=True)
        st.markdown(f"""
        - Sequence Length: {params['seq_length']}<br>
        - Hidden Size: {params['hidden_size']}<br>
        - Layers: {params['num_layers']}<br>
        - Dropout: {params['dropout']}<br>
        - Learning Rate: {params['lr']}
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if submit:
            if not ticker.strip():
                st.error("❌ Please enter a valid ticker symbol.", icon="❌")
                return
            with st.spinner("Generating predictions..."):
                if loading_animation:
                    st_lottie(loading_animation, height=80, key=f"pred_loading_{uuid.uuid4()}")
                else:
                    st.markdown("<p style='color:#f8fafc; text-align:center;'>Loading...</p>", unsafe_allow_html=True)
                data_loader = DataLoader(ticker, datetime.now() - timedelta(days=2*365), datetime.now())
                raw_data = data_loader.download_data()
                if raw_data is None:
                    return
                processed_data = data_loader.preprocess_data(raw_data)
                model_instance, device = load_model(model, processed_data.shape[1], params)
                if model_instance is None:
                    return

                seq_length = params['seq_length']
                future_steps = pred_weeks * 7
                last_seq = processed_data.iloc[-seq_length:].values
                future_preds = []
                current_seq = last_seq.copy()

                model_instance.eval()
                with torch.no_grad():
                    for _ in range(future_steps):
                        input_seq = torch.FloatTensor(current_seq).unsqueeze(0).to(device)
                        next_pred = model_instance(input_seq).cpu().numpy().squeeze()
                        future_preds.append(next_pred)
                        current_seq = np.vstack([current_seq[1:], np.append(current_seq[-1, :-1], next_pred)])

                future_dates = pd.date_range(processed_data.index[-1] + pd.Timedelta(days=1), periods=future_steps)
                future_preds_inv = data_loader.scaler.inverse_transform(
                    np.concatenate([np.zeros((len(future_preds), processed_data.shape[1]-1)), np.array(future_preds).reshape(-1,1)], axis=1)
                )[:,-1]

                st.markdown("<div class='visual-card card-fade'>", unsafe_allow_html=True)
                st.markdown("<h3 style='color:#bae6fd;'>Prediction Results</h3>", unsafe_allow_html=True)
                fig = go.Figure()
                price_col = 'Adj Close'
                fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data[price_col], name='Historical'))
                fig.add_trace(go.Scatter(x=future_dates, y=future_preds_inv, name='Predicted', line=dict(color='#bae6fd')))
                fig.update_layout(title=f"Price Prediction", xaxis_title="Date", yaxis_title="Price ($)", template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
                try:
                    img_buffer = io.BytesIO()
                    fig.write_image(img_buffer, format="png")
                    st.download_button(
                        label="Download Prediction Plot",
                        data=img_buffer,
                        file_name=f"prediction_{ticker}.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"download_pred_plot_{uuid.uuid4()}"
                    )
                except Exception as e:
                    st.warning(f"⚠️ Image export failed: {e}. Install 'kaleido' for PNG export.", icon="⚠️")
                pred_df = pd.DataFrame(future_preds_inv, index=future_dates, columns=['Predicted Price'])
                csv_buffer = io.StringIO()
                pred_df.to_csv(csv_buffer)
                st.download_button(
                    label="Download Predictions",
                    data=csv_buffer.getvalue(),
                    file_name=f"predictions_{ticker}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"download_pred_csv_{uuid.uuid4()}"
                )
                st.success("✅ Predictions generated successfully!", icon="✅")
                st.markdown("</div>", unsafe_allow_html=True)

    elif selected == "About":
        st.markdown("""
        <div class='section-card card-fade' style='max-width: 1000px; margin: auto;'>
            <h1 class='section-title'>🌟 About</h1>
            <p style='color:#f8fafc;'>Learn about the creator and technology behind Stock Analyzer</p>
            <hr class='section-sep'/>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            if about_animation:
                st_lottie(about_animation, height=200, key="about_animation")
            else:
                st.markdown("<p style='color:#f8fafc; text-align:center;'>About Stock Analyzer</p>", unsafe_allow_html=True)
            st.image("https://avatars.githubusercontent.com/u/TheBeyonder237", width=150, caption="Ngôue David")
            st.markdown("""
            <div style='text-align:center;'>
                <button class='about-contact-btn' onclick="window.open('mailto:ngouedavidrogeryannick@gmail.com')">Email</button>
                <button class='about-contact-btn' onclick="window.open('https://github.com/TheBeyonder237')">GitHub</button>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class='section-card card-fade'>
                <h3 style='color:#bae6fd;'>About the Creator</h3>
                <p style='color:#f8fafc;'>Ngôue David, a Master's student in AI and Big Data, specializes in applying machine learning to finance and healthcare.</p>
                <h3 style='color:#bae6fd;'>Skills</h3>
                <span class='badge'>Python</span>
                <span class='badge'>Machine Learning</span>
                <span class='badge'>Deep Learning</span>
                <span class='badge'>Data Science</span>
                <h3 style='color:#bae6fd;'>Projects</h3>
                <ul style='color:#f8fafc;'>
                    <li>Credit Card Expenditure Predictor</li>
                    <li>HeartGuard AI: Cardiac Risk Prediction</li>
                    <li>Stock Analyzer: RNN-based Financial Forecasting</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; color: #f8fafc; padding: 1em;'>Developed by Ngôue David</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()