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
import matplotlib.pyplot as plt

# Configure logging to suppress ticker-specific messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.handlers = []  # Clear default handlers
logger.addHandler(logging.StreamHandler())  # Add custom handler
logger.propagate = False  # Prevent propagation to root logger

# Suppress yfinance output by redirecting stdout
import sys
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# Lottie animation loader
def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        logger.error(f"Échec du chargement de l'animation Lottie : {e}")
        return None

# Load Lottie animations
main_animation = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_4kx2q32n.json")
loading_animation = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_p8bfn5to.json")
about_animation = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_0yfsb3a1.json")

# Page configuration
st.set_page_config(
    page_title="Analyseur de Stocks",
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
    'LSTM Bidirectionnel': {'seq_length': 20, 'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3, 'lr': 0.001},
    'GRU Bidirectionnel': {'seq_length': 20, 'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3, 'lr': 0.001},
    'CNN-LSTM': {'seq_length': 20, 'hidden_size': 64, 'num_layers': 2, 'dropout': 0.3, 'lr': 0.001}
}

# Cached data download function (quiet mode)
@st.cache_data(show_spinner=False)
def download_data(_self, ticker, start_date, end_date):
    try:
        with SuppressOutput():
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
        if data.empty:
            raise ValueError(f"Aucune donnée trouvée pour le ticker")
        logger.info("Données téléchargées avec succès")
        return data
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement des données : {e}")
        st.error(f"❌ Échec du téléchargement des données : {e}", icon="❌")
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
            if self.data is None or self.data.empty:
                raise ValueError(f"Aucune donnée trouvée pour le ticker")
            self.data = self.data.ffill()
            # Aplatir le MultiIndex si nécessaire
            if isinstance(self.data.columns, pd.MultiIndex):
                self.data.columns = self.data.columns.get_level_values(0)  # Prendre le premier niveau
            logger.info("Données EDA téléchargées avec succès")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement des données EDA : {e}")
            st.error(f"❌ Échec du téléchargement des données : {e}", icon="❌")
            return False

    def get_price_column(self):
        return 'Adj Close' if 'Adj Close' in self.data.columns else 'Close'

    def calculate_returns(self):
        price_column = self.get_price_column()
        self.returns = self.data[price_column].pct_change().dropna()

    def basic_statistics(self):
        if self.data is None or self.data.empty:
            return pd.DataFrame({
                'Statistique': ['Moyenne', 'Écart-type', 'Min', 'Max', 'Médiane', 'Asymétrie', 'Kurtosis'],
                'Prix': [np.nan] * 7,
                'Rendements': [np.nan] * 7
            })
        
        price_column = self.get_price_column()
        
        # Handle MultiIndex columns
        if isinstance(self.data.columns, pd.MultiIndex):
            # Check if 'Adj Close' or 'Close' exists in the first level of MultiIndex
            available_columns = self.data.columns.get_level_values(0)
            if 'Adj Close' in available_columns:
                price_column = ('Adj Close', self.ticker)
            elif 'Close' in available_columns:
                price_column = ('Close', self.ticker)
            else:
                raise ValueError("Neither 'Adj Close' nor 'Close' found in data columns")
        
        price_stats = [
            float(self.data[price_column].mean()),  # Remove .iloc[0]
            float(self.data[price_column].std()),   # Remove .iloc[0]
            float(self.data[price_column].min()),   # Remove .iloc[0]
            float(self.data[price_column].max()),   # Remove .iloc[0]
            float(self.data[price_column].median()),  # Remove .iloc[0]
            float(stats.skew(self.data[price_column], nan_policy='omit')),  # Remove [0]
            float(stats.kurtosis(self.data[price_column], nan_policy='omit'))  # Remove [0]
        ]
        
        returns_stats = [
            float(self.returns.mean()) if self.returns is not None and not self.returns.empty else np.nan,
            float(self.returns.std()) if self.returns is not None and not self.returns.empty else np.nan,
            float(self.returns.min()) if self.returns is not None and not self.returns.empty else np.nan,
            float(self.returns.max()) if self.returns is not None and not self.returns.empty else np.nan,
            float(self.returns.median()) if self.returns is not None and not self.returns.empty else np.nan,
            float(stats.skew(self.returns, nan_policy='omit')) if self.returns is not None and not self.returns.empty else np.nan,
            float(stats.kurtosis(self.returns, nan_policy='omit')) if self.returns is not None and not self.returns.empty else np.nan
        ]
        
        return pd.DataFrame({
            'Statistique': ['Moyenne', 'Écart-type', 'Min', 'Max', 'Médiane', 'Asymétrie', 'Kurtosis'],
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
            title="Évolution des Prix",
            yaxis_title="Prix ($)",
            xaxis_title="Date",
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig

    def plot_volume(self):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=self.data.index, y=self.data['Volume'], name='Volume'))
        fig.update_layout(
            title="Volume des Transactions",
            yaxis_title="Volume",
            xaxis_title="Date",
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig

    def plot_returns_distribution(self):
        if self.returns is None or self.returns.empty:
            return None
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Distribution des Rendements', 'QQ-Plot'))
        fig.add_trace(go.Histogram(x=self.returns, name='Rendements', nbinsx=50), row=1, col=1)
        returns_sorted = np.sort(self.returns.dropna())
        n = len(returns_sorted)
        if n < 2:
            return None
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))
        fig.add_trace(go.Scatter(x=theoretical_quantiles, y=returns_sorted, mode='markers', name='QQ-Plot'), row=2, col=1)
        fig.add_trace(go.Scatter(x=[min(theoretical_quantiles.min(), returns_sorted.min()), max(theoretical_quantiles.max(), returns_sorted.max())],
                                y=[min(theoretical_quantiles.min(), returns_sorted.min()), max(theoretical_quantiles.max(), returns_sorted.max())],
                                mode='lines', name='Ligne de Référence', line=dict(color='red', dash='dash')), row=2, col=1)
        fig.update_layout(title="Analyse des Rendements", template='plotly_white', height=600)
        return fig

    def correlation_analysis(self):
        corr_matrix = self.data.corr()
        fig = go.Figure(data=go.Heatmap(z=corr_matrix, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='RdBu'))
        fig.update_layout(title="Matrice de Corrélation", template='plotly_white')
        return fig, corr_matrix

    def volatility_analysis(self):
        if self.returns is None or self.returns.empty:
            return None
        volatility = self.returns.rolling(window=20).std() * np.sqrt(252)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=volatility.index, y=volatility, name='Volatilité Annualisée'))
        fig.update_layout(title="Volatilité (20 jours)", yaxis_title="Volatilité", xaxis_title="Date", template='plotly_white')
        return fig

    def generate_report(self):
        if not self.download_data():
            return None
        self.calculate_returns()
        stats_df = self.basic_statistics()
        stats_md = "| Statistique | Prix | Rendements |\n|-------------|------|------------|\n"
        for _, row in stats_df.iterrows():
            prix = '-' if pd.isna(row['Prix']) else f"{row['Prix']:.4f}"
            rendements = '-' if pd.isna(row['Rendements']) else f"{row['Rendements']:.4f}"
            stats_md += f"| {row['Statistique']} | {prix} | {rendements} |\n"
        price_column = self.get_price_column()
        
        # Handle MultiIndex columns for display
        if isinstance(self.data.columns, pd.MultiIndex):
            columns_str = [col[0] for col in self.data.columns]  # Extract first level
        else:
            columns_str = [str(col) for col in self.data.columns]
            
        report = (
            f"# Rapport des Données Financières\n\n"
            f"## Période d'Analyse\n"
            f"- Date de Début : {self.start_date.strftime('%Y-%m-%d')}\n"
            f"- Date de Fin : {self.end_date.strftime('%Y-%m-%d')}\n\n"
            f"## Statistiques de Base\n{stats_md}\n"
            f"## Insights sur les Données\n"
            f"- Nombre Total de Jours de Trading : {len(self.data)}\n"
            f"- Valeurs Manquantes : {self.data.isnull().sum().sum()}\n"
            f"- Plage de Prix : ${float(self.data[price_column].min()):.2f} - ${float(self.data[price_column].max()):.2f}\n"
            f"- Colonnes Disponibles : {', '.join(columns_str)}\n"
        )
        return report

    def get_display_data(self):
        """Affiche les données avec les titres originaux."""
        if self.data is None or self.data.empty:
            return None
        display_data = self.data.copy()
        
        # Gérer l'index des lignes si nécessaire
        if isinstance(display_data.index, pd.MultiIndex):
            if len(display_data.index.names) > 1:
                level_to_drop = 0
                if 'Date' in display_data.index.names:
                    for i, name in enumerate(display_data.index.names):
                        if name != 'Date':
                            level_to_drop = i
                            break
                display_data = display_data.reset_index(level=level_to_drop, drop=True)
        elif hasattr(display_data.index, 'name') and display_data.index.name == self.ticker:
            display_data = display_data.reset_index(drop=True)
            if 'Date' in display_data.columns:
                display_data = display_data.set_index('Date')
        
        return display_data

# Model loading function
@st.cache_resource
def load_model(model_name, input_size, params):
    try:
        if input_size != 6:
            raise ValueError(f"6 caractéristiques d'entrée attendues, trouvé {input_size}. Assurez-vous que les données incluent 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_name.startswith('LSTM Bidirectionnel'):
            model_type = 'LSTM' if 'LSTM' in model_name else 'GRU'
            model = BidirectionalRNNModel(input_size, params['hidden_size'], params['num_layers'], model_type, params['dropout']).to(device)
        elif model_name == 'CNN-LSTM':
            model = CNNLSTMModel(input_size, params['hidden_size'], params['num_layers']).to(device)
        else:
            model = RNNModel(input_size, params['hidden_size'], params['num_layers'], model_type=model_name, dropout=params['dropout']).to(device)
        model_path = f"saved_models/best_model_{model_name}.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fichier de modèle {model_path} non trouvé. Assurez-vous que tous les fichiers de modèle sont dans le répertoire 'saved_models'.")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info(f"Modèle {model_name} chargé")
        return model, device
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle {model_name} : {e}")
        st.error(f"❌ Échec du chargement du modèle {model_name} : {e}", icon="❌")
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
            <span class="dp-text">Traitement des Données : <span style="color:#00b894">ACTIF</span></span>
        </div>
        """, unsafe_allow_html=True
    )

# Sidebar
with st.sidebar:
    if main_animation:
        st_lottie(main_animation, height=100, key="sidebar_animation")
    else:
        st.markdown("<p style='color:#f8fafc; text-align:center;'>Analyseur de Stocks</p>", unsafe_allow_html=True)
    selected = option_menu(
        menu_title="Navigation",
        options=["Accueil", "EDA", "Prédictions", "À propos"],
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
        config_key = st.text_input("Clé API (Optionnel)", type="password", placeholder="Entrez une clé API optionnelle")
        if st.form_submit_button("Valider", use_container_width=True):
            st.success("✅ Configuration validée (simulée)", icon="✅")
    st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #f8fafc; font-size: 0.9em;'>
        Développé par Ngôue David<br>
        <a href='mailto:ngouedavidrogeryannick@gmail.com' style='color:#bae6fd;'>📧 Email</a><br>
        <a href='https://github.com/TheBeyonder237' style='color:#bae6fd;'>🌐 GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# Main application
def main():
    display_data_processing_badge()
    
    if selected == "Accueil":
        st.markdown("""
        <div class='section-card card-fade' style='text-align: center; max-width: 1000px; margin: auto;'>
            <h1 class='section-title'>📈 Analyseur de Stocks</h1>
            <p style='color: #f8fafc; font-size: 1.3em; margin-bottom: 1em;'>Analyse Financière Avancée Alimentée par l'IA</p>
            <hr class='section-sep'/>
            <p style='color: #f8fafc; font-size: 1.1em;'>Explorez les données boursières et prédisez les tendances futures avec des modèles RNN de pointe.</p>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.markdown("""
            <div class='section-card card-fade'>
                <h3 style='color:#bae6fd; font-size:1.5em;'>Mission</h3>
                <p style='color:#f8fafc;'>Fournir des insights exploitables et des prédictions précises pour les marchés financiers grâce à l'apprentissage profond.</p>
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
                st.markdown("<p style='color:#f8fafc; text-align:center;'>Bienvenue dans l'Analyseur de Stocks</p>", unsafe_allow_html=True)
            st.markdown("""
            <div class='metric-card card-fade' style='text-align: center;'>
                <h3 style='color:#bae6fd; margin:0;'>100+</h3>
                <p style='color:#f8fafc; margin:0;'>Actifs Analysables</p>
            </div>
            """, unsafe_allow_html=True)

    elif selected == "EDA":
        st.markdown("""
        <div class='section-card card-fade' style='max-width: 1000px; margin: auto;'>
            <h1 class='section-title'>📊 Analyse Exploratoire des Données</h1>
            <span class='badge'>yfinance</span>
            <span class='badge'>Plotly</span>
            <hr class='section-sep'/>
            <p style='color:#f8fafc;'>Analysez les données historiques des stocks avec des visualisations et des statistiques complètes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("eda_form"):
            col1, col2 = st.columns([2, 1])
            with col1:
                ticker = st.text_input("Symbole de Ticker", value="TSLA", placeholder="ex. : TSLA")
            with col2:
                period = st.selectbox("Période", ["1 An", "2 Ans", "3 Ans", "Personnalisée"])
            if period == "Personnalisée":
                col3, col4 = st.columns(2)
                with col3:
                    start_date = st.date_input("Date de Début", datetime.now() - timedelta(days=2*365))
                with col4:
                    end_date = st.date_input("Date de Fin", datetime.now())
            else:
                years = {"1 An": 1, "2 Ans": 2, "3 Ans": 3}[period]
                end_date = datetime.now()
                start_date = end_date - timedelta(days=years*365)
            submit = st.form_submit_button("Analyser", use_container_width=True)

        if submit:
            if not ticker.strip():
                st.error("❌ Veuillez entrer un symbole de ticker valide.", icon="❌")
                return
            with st.spinner("Traitement des données..."):
                if loading_animation:
                    st_lottie(loading_animation, height=80, key=f"eda_loading_{uuid.uuid4()}")
                else:
                    st.markdown("<p style='color:#f8fafc; text-align:center;'>Chargement...</p>", unsafe_allow_html=True)
                eda = FinancialDataEDA(ticker, start_date, end_date)
                report = eda.generate_report()
                if report:
                    st.markdown("<div class='visual-card card-fade'>", unsafe_allow_html=True)
                    st.markdown("<h3 style='color:#bae6fd;'>Rapport d'Analyse</h3>", unsafe_allow_html=True)
                    st.markdown(report, unsafe_allow_html=True)
                    st.download_button(
                        label="Télécharger le Rapport",
                        data=report.encode('utf-8'),
                        file_name=f"rapport_eda_{ticker}.md",
                        mime="text/markdown",
                        use_container_width=True,
                        key=f"download_report_{uuid.uuid4()}"
                    )
                    st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)
                    st.markdown("<h3 style='color:#bae6fd;'>Aperçu des Données</h3>", unsafe_allow_html=True)
                    display_data = eda.get_display_data()
                    if display_data is not None:
                        st.dataframe(display_data.head(), use_container_width=True)
                    
                    for title, plot_func, filename in [
                        ("Évolution des Prix", eda.plot_price_evolution, "evolution_prix"),
                        ("Volume des Transactions", eda.plot_volume, "volume"),
                        ("Analyse des Rendements", eda.plot_returns_distribution, "rendements"),
                        ("Matrice de Corrélation", lambda: eda.correlation_analysis()[0], "correlation"),
                        ("Analyse de la Volatilité", eda.volatility_analysis, "volatilite")
                    ]:
                        st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)
                        st.markdown(f"<h3 style='color:#bae6fd;'>{title}</h3>", unsafe_allow_html=True)
                        fig = plot_func()
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            # Exportation avec matplotlib
                            plt.figure(figsize=(10, 6))
                            fig_data = fig.to_dict()
                            if 'data' in fig_data and len(fig_data['data']) > 0:
                                for trace in fig_data['data']:
                                    if trace.get('type') == 'histogram':
                                        # Handle histogram trace
                                        if 'x' in trace:
                                            hist_data = np.array(trace['x'])
                                            hist_bins = trace.get('nbinsx', 50)
                                            counts, bins = np.histogram(hist_data, bins=hist_bins)
                                            plt.stairs(counts, bins, label=trace.get('name', 'Histogram'), fill=True)
                                    elif 'x' in trace and 'y' in trace:
                                        # Handle scatter or line traces
                                        try:
                                            plt.plot(trace['x'], trace['y'], label=trace.get('name', ''))
                                        except TypeError:
                                            # Skip traces with invalid data
                                            continue
                                plt.title(title)
                                plt.xlabel(fig_data['layout'].get('xaxis', {}).get('title', {}).get('text', ''))
                                plt.ylabel(fig_data['layout'].get('yaxis', {}).get('title', {}).get('text', ''))
                                if any(trace.get('name') for trace in fig_data['data']):
                                    plt.legend()
                                img_buffer = io.BytesIO()
                                plt.savefig(img_buffer, format='png', bbox_inches='tight')
                                plt.close()
                                img_buffer.seek(0)
                                st.download_button(
                                    label=f"Télécharger {title}",
                                    data=img_buffer,
                                    file_name=f"{filename}_{ticker}.png",
                                    mime="image/png",
                                    use_container_width=True,
                                    key=f"download_{filename}_{uuid.uuid4()}"
                                )
                            else:
                                st.warning(f"Impossible de générer l'image pour {title}.", icon="⚠️")
                    st.success("✅ Analyse terminée !", icon="✅")
                    st.markdown("</div>", unsafe_allow_html=True)

    elif selected == "Prédictions":
        st.markdown("""
        <div class='section-card card-fade' style='max-width: 1000px; margin: auto;'>
            <h1 class='section-title'>🔮 Prédictions Futures</h1>
            <span class='badge'>RNN</span>
            <span class='badge'>PyTorch</span>
            <hr class='section-sep'/>
            <p style='color:#f8fafc;'>Prédisez les prix futurs des stocks avec des modèles RNN avancés.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("pred_form"):
            col1, col2 = st.columns([2, 1])
            with col1:
                ticker = st.text_input("Symbole de Ticker", value="TSLA", placeholder="ex. : TSLA")
            with col2:
                pred_weeks = st.slider("Horizon de Prédiction (Semaines)", 1, 12, 3)
            model = st.selectbox("Sélectionner un Modèle", list(MODEL_PARAMS.keys()))
            submit = st.form_submit_button("Prédire", use_container_width=True)

        st.markdown("<div class='section-card card-fade'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#bae6fd;'>Paramètres du Modèle</h3>", unsafe_allow_html=True)
        params = MODEL_PARAMS[model]
        st.markdown(f"<span class='badge'>{model}</span>", unsafe_allow_html=True)
        st.markdown(f"""
        - Longueur de Séquence : {params['seq_length']}<br>
        - Taille Cachée : {params['hidden_size']}<br>
        - Couches : {params['num_layers']}<br>
        - Dropout : {params['dropout']}<br>
        - Taux d'Apprentissage : {params['lr']}
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if submit:
            if not ticker.strip():
                st.error("❌ Veuillez entrer un symbole de ticker valide.", icon="❌")
                return
            with st.spinner("Génération des prédictions..."):
                if loading_animation:
                    st_lottie(loading_animation, height=80, key=f"pred_loading_{uuid.uuid4()}")
                else:
                    st.markdown("<p style='color:#f8fafc; text-align:center;'>Chargement...</p>", unsafe_allow_html=True)
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
                st.markdown("<h3 style='color:#bae6fd;'>Résultats des Prédictions</h3>", unsafe_allow_html=True)
                fig = go.Figure()
                price_col = 'Adj Close'
                fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data[price_col], name='Historique'))
                fig.add_trace(go.Scatter(x=future_dates, y=future_preds_inv, name='Prédit', line=dict(color='#bae6fd')))
                fig.update_layout(title="Prédiction des Prix", xaxis_title="Date", yaxis_title="Prix ($)", template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
                # Exportation avec matplotlib
                plt.figure(figsize=(10, 6))
                fig_data = fig.to_dict()
                if 'data' in fig_data and len(fig_data['data']) > 0:
                    for trace in fig_data['data']:
                        if 'x' in trace and 'y' in trace:
                            plt.plot(trace['x'], trace['y'], label=trace.get('name', ''))
                    plt.title("Prédiction des Prix")
                    plt.xlabel("Date")
                    plt.ylabel("Prix ($)")
                    plt.legend()
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png', bbox_inches='tight')
                    plt.close()
                    img_buffer.seek(0)
                    st.download_button(
                        label="Télécharger le Graphique de Prédiction",
                        data=img_buffer,
                        file_name=f"prediction_{ticker}.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"download_pred_plot_{uuid.uuid4()}"
                    )
                pred_df = pd.DataFrame(future_preds_inv, index=future_dates, columns=['Prix Prédit'])
                csv_buffer = io.StringIO()
                pred_df.to_csv(csv_buffer)
                st.download_button(
                    label="Télécharger les Prédictions",
                    data=csv_buffer.getvalue(),
                    file_name=f"predictions_{ticker}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"download_pred_csv_{uuid.uuid4()}"
                )
                st.success("✅ Prédictions générées avec succès !", icon="✅")
                st.markdown("</div>", unsafe_allow_html=True)

    elif selected == "À propos":
        st.markdown("""
        <div class='section-card card-fade' style='max-width: 1000px; margin: auto;'>
            <h1 class='section-title'>🌟 À propos</h1>
            <p style='color:#f8fafc;'>En savoir plus sur le créateur et la technologie derrière l'Analyseur de Stocks</p>
            <hr class='section-sep'/>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            if about_animation:
                st_lottie(about_animation, height=200, key="about_animation")
            else:
                st.markdown("<p style='color:#f8fafc; text-align:center;'>À propos de l'Analyseur de Stocks</p>", unsafe_allow_html=True)
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
                <h3 style='color:#bae6fd;'>À propos du Créateur</h3>
                <p style='color:#f8fafc;'>Ngôue David, étudiant en Master en IA et Big Data, se spécialise dans l'application de l'apprentissage automatique à la finance et à la santé.</p>
                <h3 style='color:#bae6fd;'>Compétences</h3>
                <span class='badge'>Python</span>
                <span class='badge'>Machine Learning</span>
                <span class='badge'>Deep Learning</span>
                <span class='badge'>Data Science</span>
                <h3 style='color:#bae6fd;'>Projets</h3>
                <ul style='color:#f8fafc;'>
                    <li>Prédicteur de Dépenses par Carte de Crédit</li>
                    <li>HeartGuard AI : Prédiction des Risques Cardiaques</li>
                    <li>Analyseur de Stocks : Prévisions Financières basées sur RNN</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; color: #f8fafc; padding: 1em;'>Développé par Ngôue David</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()