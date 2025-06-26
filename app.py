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
        corr_matrix = self.data.corr(numeric_only=True) # Added numeric_only=True for robustness
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
            f"- Plage de Prix : ${float(self.data[price_column].min().iloc[0]):.2f} - ${float(self.data[price_column].max().iloc[0]):.2f}\n"
            f"- Colonnes Disponibles : {', '.join(columns_str)}\n"
        )
        return report

    def get_display_data(self):
        """Affiche les données avec les titres originaux et supprime toute référence au ticker dans l'index ou les colonnes."""
        if self.data is None or self.data.empty:
            return None
        display_data = self.data.copy()

        # --- Gérer les MultiIndex dans les colonnes ---
        if isinstance(display_data.columns, pd.MultiIndex):
            # Basé sur l'image fournie, la structure est probablement (Nom_Colonne, Ticker)
            # Nous voulons conserver le 'Nom_Colonne' qui est au niveau 0.
            display_data.columns = display_data.columns.get_level_values(0)
            
            # Après avoir aplati, assurez-vous que les noms de colonnes sont uniques.
            # Cela gère les cas où get_level_values(0) pourrait encore entraîner des doublons
            # ou si les noms de colonnes par défaut de yfinance ne sont pas uniques.
            cols = pd.Series(display_data.columns)
            if cols.duplicated().any():
                # Pour tout nom de colonne dupliqué, ajoutez un suffixe pour le rendre unique
                for dup in cols[cols.duplicated()].unique():
                    # Trouvez tous les indices où ce nom dupliqué apparaît
                    indices = cols[cols == dup].index.values.tolist()
                    for i, idx in enumerate(indices):
                        # N'ajoutez un suffixe qu'à partir de la deuxième occurrence
                        if i > 0: 
                            cols.iloc[idx] = f"{dup}_{i}"
                display_data.columns = cols


        # --- Gérer les MultiIndex dans l'index des lignes ---
        # Cette section s'assure que l'index est une date unique sans le ticker.
        if isinstance(display_data.index, pd.MultiIndex):
            # Si le MultiIndex a plus d'un niveau (par exemple, Symbole, Date)
            if len(display_data.index.names) > 1:
                # Trouvez le niveau qui n'est pas 'Date' et supprimez-le.
                # En supposant que le ticker est généralement le premier niveau (level=0)
                level_to_drop = 0
                # Si 'Date' est explicitement nommé dans les niveaux d'index, nous nous assurons de le conserver
                # et de supprimer l'autre niveau.
                if 'Date' in display_data.index.names:
                    # Supprimez le niveau qui n'est pas 'Date'
                    for i, name in enumerate(display_data.index.names):
                        if name != 'Date':
                            level_to_drop = i
                            break
                display_data = display_data.reset_index(level=level_to_drop, drop=True)
            else:
                # S'il n'y a qu'un seul niveau dans le MultiIndex et c'est le ticker lui-même
                # ou un niveau unique sans nom, il suffit de le réinitialiser.
                display_data = display_data.reset_index(drop=True)

        # Si ce n'est pas un MultiIndex pour les lignes, mais l'index a un nom qui est le ticker.
        # Cela gère les cas où l'index est un Index simple mais nommé avec le ticker.
        elif hasattr(display_data.index, 'name') and display_data.index.name == self.ticker:
            display_data = display_data.reset_index(drop=True)
            # Ensuite, définissez 'Date' comme index si elle est devenue une colonne
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
        <div class="dp-badge card-fade" style="animation-delay: 0.2s;">
            <div class="dp-dot"></div>
            <span class="dp-text">Traitement des données en cours...</span>
        </div>
        """, unsafe_allow_html=True
    )

# Main application function
def main():
    st.sidebar.title("Configuration")

    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",
            options=["Accueil", "Analyse Exploratoire des Données", "Prédiction", "À Propos"],
            icons=["house", "bar-chart-line", "graph-up-arrow", "info-circle"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#2a374c", "border-radius": "10px"},
                "icon": {"color": "#bae6fd", "font-size": "20px"},
                "nav-link": {"font-size": "18px", "text-align": "left", "margin": "0px", "--hover-color": "#334155", "color": "#f8fafc"},
                "nav-link-selected": {"background-color": "#475569", "color": "#f8fafc"},
            }
        )

    if selected == "Accueil":
        st.markdown("<h1 class='section-title' style='text-align: center;'>Bienvenue à l'Analyseur de Stocks et Prédictions RNN 📈</h1>", unsafe_allow_html=True)
        st_lottie(main_animation, height=300, key="main_animation")
        st.markdown("""
            <div class='section-card card-fade' style='animation-delay: 0.4s;'>
                <h3 style='color:#bae6fd;'>Introduction</h3>
                <p>Cette application vous permet d'analyser les données historiques des actions boursières et de prédire les prix futurs en utilisant des modèles de réseaux de neurones récurrents (RNN).</p>
                <p>Naviguez à travers les différentes sections en utilisant le menu latéral pour explorer les données, obtenir des prédictions et en savoir plus sur l'application.</p>
            </div>
            <div class='section-card card-fade' style='animation-delay: 0.6s;'>
                <h3 style='color:#bae6fd;'>Comment ça Marche ?</h3>
                <ol style='color:#f8fafc;'>
                    <li>Sélectionnez un onglet de navigation (par exemple, "Analyse Exploratoire des Données").</li>
                    <li>Entrez le symbole boursier (ticker) et la période d'analyse.</li>
                    <li>Visualisez les statistiques, les graphiques et les prédictions.</li>
                    <li>Téléchargez les rapports et les graphiques pour une analyse plus approfondie.</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

    elif selected == "Analyse Exploratoire des Données":
        st.markdown("<h1 class='section-title' style='text-align: center;'>Analyse Exploratoire des Données (EDA)</h1>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Symbole Boursier (ex: AAPL, GOOGL, TSLA)", "TSLA").upper()
        with col2:
            today = datetime.now()
            start_date = st.date_input("Date de Début", datetime(2022, 1, 1))
            end_date = st.date_input("Date de Fin", today)

        if start_date >= end_date:
            st.error("La date de début doit être antérieure à la date de fin.", icon="🚫")
            return

        eda = FinancialDataEDA(ticker, start_date, end_date)

        if st.button("Lancer l'Analyse EDA", use_container_width=True):
            with st.spinner("Téléchargement et préparation des données..."):
                display_data_processing_badge()
                if eda.download_data():
                    eda.calculate_returns()
                    st.success("✅ Données EDA téléchargées et prêtes !", icon="🎉")

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
                        
                        plotly_fig = plot_func() # Get the Plotly figure
                        
                        if plotly_fig:
                            st.plotly_chart(plotly_fig, use_container_width=True)
                            
                            # --- Matplotlib Export Logic ---
                            img_buffer = io.BytesIO()
                            plt.figure(figsize=(10, 6)) # Create a new Matplotlib figure for each plot
                            
                            if title == "Évolution des Prix":
                                price_column = eda.get_price_column()
                                plt.plot(eda.data.index, eda.data[price_column], label=price_column, color='blue')
                                plt.title(title)
                                plt.xlabel("Date")
                                plt.ylabel("Prix ($)")
                                plt.grid(True, linestyle='--', alpha=0.6)
                                plt.legend()
                            elif title == "Volume des Transactions":
                                plt.bar(eda.data.index, eda.data['Volume'], color='skyblue')
                                plt.title(title)
                                plt.xlabel("Date")
                                plt.ylabel("Volume")
                                plt.grid(True, linestyle='--', alpha=0.6)
                            elif title == "Analyse des Rendements":
                                if eda.returns is not None and not eda.returns.empty:
                                    plt.subplot(2, 1, 1) # Histogram
                                    plt.hist(eda.returns.dropna(), bins=50, color='lightgreen', edgecolor='black')
                                    plt.title('Distribution des Rendements')
                                    plt.xlabel('Rendements')
                                    plt.ylabel('Fréquence')
                                    plt.grid(True, linestyle='--', alpha=0.6)

                                    plt.subplot(2, 1, 2) # QQ-Plot
                                    stats.probplot(eda.returns.dropna(), dist="norm", plot=plt)
                                    plt.title('QQ-Plot')
                                    plt.tight_layout()
                                else:
                                    plt.text(0.5, 0.5, "Données de rendements non disponibles", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)

                            elif title == "Matrice de Corrélation":
                                _, corr_matrix = eda.correlation_analysis() # Get the corr_matrix
                                if corr_matrix is not None and not corr_matrix.empty:
                                    # Use imshow for heatmap
                                    plt.imshow(corr_matrix.values, cmap='RdBu', interpolation='nearest')
                                    plt.colorbar(label='Corrélation')
                                    # Set ticks and labels
                                    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
                                    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
                                    plt.title(title)
                                    plt.tight_layout()
                                else:
                                    plt.text(0.5, 0.5, "Matrice de corrélation non disponible", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)

                            elif title == "Analyse de la Volatilité":
                                if eda.returns is not None and not eda.returns.empty:
                                    volatility = eda.returns.rolling(window=20).std() * np.sqrt(252)
                                    plt.plot(volatility.index, volatility, label='Volatilité Annualisée', color='orange')
                                    plt.title(title)
                                    plt.xlabel("Date")
                                    plt.ylabel("Volatilité")
                                    plt.grid(True, linestyle='--', alpha=0.6)
                                    plt.legend()
                                else:
                                    plt.text(0.5, 0.5, "Données de rendements non disponibles pour l'analyse de volatilité", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)

                            plt.savefig(img_buffer, format='png', bbox_inches='tight')
                            plt.close() # Close the Matplotlib figure to free up memory
                            img_buffer.seek(0)
                            st.download_button(
                                label=f"Télécharger {title}",
                                data=img_buffer,
                                file_name=f"{filename}_{ticker}.png",
                                mime="image/png",
                                use_container_width=True,
                                key=f"download_{filename}_{uuid.uuid4()}"
                            )

                    st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)
                    st.markdown("<h3 style='color:#bae6fd;'>Statistiques Descriptives</h3>", unsafe_allow_html=True)
                    st.dataframe(eda.basic_statistics(), use_container_width=True)

                    st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)
                    st.markdown("<h3 style='color:#bae6fd;'>Rapport Complet</h3>", unsafe_allow_html=True)
                    st.markdown(eda.generate_report(), unsafe_allow_html=True)

    elif selected == "Prédiction":
        st.markdown("<h1 class='section-title' style='text-align: center;'>Prédiction des Prix des Actions avec RNN</h1>", unsafe_allow_html=True)
        st_lottie(loading_animation, height=200, key="loading_animation", speed=1.5)

        st.info("Cette section est en cours de développement. Bientôt disponible pour des prédictions basées sur des modèles RNN !")

        # Placeholder for Prediction section functionality
        # ticker = st.text_input("Symbole Boursier (ex: AAPL)", "TSLA").upper()
        # today = datetime.now()
        # train_end_date = st.date_input("Date de Fin d'Entraînement", today - timedelta(days=90))
        # forecast_horizon = st.slider("Horizon de Prédiction (jours)", 1, 30, 7)
        # model_type = st.selectbox("Type de Modèle RNN", list(MODEL_PARAMS.keys()))
        
        # if st.button("Lancer la Prédiction", use_container_width=True):
        #     st.warning("Fonctionnalité de prédiction non encore implémentée.", icon="⏳")


    elif selected == "À Propos":
        st.markdown("<h1 class='section-title' style='text-align: center;'>À Propos de l'Application</h1>", unsafe_allow_html=True)
        st_lottie(about_animation, height=200, key="about_animation")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("""
            <div class='section-card card-fade'>
                <h3 style='color:#bae6fd;'>Contexte</h3>
                <p>Cette application a été développée pour démontrer l'application des Réseaux de Neurones Récurrents (RNN) à l'analyse et à la prédiction des séries temporelles financières.</p>
                <p>Elle utilise les bibliothèques Python populaires telles que Streamlit pour l'interface utilisateur, yfinance pour les données financières, Plotly et Matplotlib pour les visualisations, et PyTorch pour les modèles de Deep Learning.</p>
                <h3 style='color:#bae6fd;'>Contact</h3>
                <img src="https://avatars.githubusercontent.com/u/79965048?v=4" class="about-avatar" width="150" height="150" style="margin-bottom: 1em;">
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