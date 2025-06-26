import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import io
import os
import logging
import warnings
import requests
from streamlit_lottie import st_lottie
import time

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de la page
st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Nouvelle palette de couleurs et style CSS inspiré de NLP Magic Hub
color_palette = {
    "primary": "#6B46C1",  # Violet principal
    "secondary": "#96A6BB",  # Fond sombre
    "accent": "#A78BFA",  # Violet clair
    "text": "#F3F4F6",  # Texte clair
    "success": "#34D399",  # Vert succès
    "warning": "#FBBF24",  # Jaune warning
    "error": "#F87171"  # Rouge erreur
}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    :root {{
        --primary: {color_palette['primary']};
        --secondary: {color_palette['secondary']};
        --accent: {color_palette['accent']};
        --text: {color_palette['text']};
        --success: {color_palette['success']};
        --warning: {color_palette['warning']};
        --error: {color_palette['error']};
    }}

    .stApp {{
        background-color: var(--secondary);
        font-family: 'Inter', sans-serif;
    }}

    /* Typographie */
    h1, h2, h3, h4 {{
        color: var(--text);
        font-weight: 600;
        margin-bottom: 1rem;
    }}

    p, li, .stMarkdown, .stText {{
        color: var(--text);
        line-height: 1.6;
    }}

    /* Sidebar */
    .sidebar .sidebar-content {{
        background-color: var(--secondary);
        border-right: 1px solid rgba(255,255,255,0.1);
        padding: 1.5rem;
    }}

    /* Boutons */
    .stButton>button {{
        background-color: var(--primary);
        color: var(--text);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: all 0.3s ease;
        font-weight: 500;
    }}
    .stButton>button:hover {{
        background-color: var(--accent);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(107, 70, 193, 0.3);
    }}

    /* Formulaires */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {{
        background-color: #2D3748;
        color: var(--text);
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1);
    }}

    /* Selectbox et Slider */
    .stSelectbox, .stSlider, .stCheckbox {{
        background-color: #2D3748;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1);
    }}

    /* Alertes */
    .stAlert {{
        border-radius: 8px;
        border-left: 4px solid;
    }}

    /* Animation pour les containers */
    .custom-container {{
        background-color: #2D3748;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid var(--primary);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }}
    .custom-container:hover {{
        transform: translateY(-4px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }}

    /* Séparateurs */
    hr {{
        border-color: rgba(255,255,255,0.1);
        margin: 1.5rem 0;
    }}

    /* Badges */
    .badge {{
        background-color: var(--accent);
        color: var(--text);
        padding: 0.3em 0.8em;
        border-radius: 12px;
        margin: 0.2em;
        display: inline-block;
        font-size: 0.9em;
    }}

    /* Boutons de contact */
    .about-contact-btn {{
        background-color: var(--primary);
        color: var(--text);
        border: none;
        padding: 0.5em 1em;
        margin: 0.5em;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
    }}
    .about-contact-btn:hover {{
        background-color: var(--accent);
        transform: translateY(-2px);
    }}
</style>
""", unsafe_allow_html=True)

# Load Lottie animation
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

about_animation = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_j3UXNf.json")

# Hard-coded model hyperparameters
MODEL_PARAMS = {
    'LSTM': {
        'seq_length': 20,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'lr': 0.001
    },
    'GRU': {
        'seq_length': 20,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'lr': 0.001
    },
    'Bidirectional LSTM': {
        'seq_length': 20,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'lr': 0.001
    },
    'Bidirectional GRU': {
        'seq_length': 20,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'lr': 0.001
    },
    'CNN-LSTM': {
        'seq_length': 20,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'lr': 0.001
    }
}

# DataLoader and Model Classes
class DataLoader:
    def __init__(self, entreprise, start_date, end_date):
        self.entreprise = entreprise
        self.start_date = start_date
        self.end_date = end_date
        self.scaler = MinMaxScaler()
        
    def download_data(self, max_retries=3, delay=5):
        for attempt in range(1, max_retries + 1):
            try:
                data = yf.download(self.entreprise, start=self.start_date, end=self.end_date, auto_adjust=False)
                if data.empty:
                    raise ValueError(f"Aucune donnée trouvée pour {self.entreprise}")
                logger.info(f"Téléchargement réussi pour {self.entreprise}")
                return data
            except Exception as e:
                logger.error(f"Erreur lors du téléchargement (tentative {attempt}/{max_retries}): {str(e)}")
                if attempt == max_retries:
                    st.error(f"❌ Échec après {max_retries} tentatives. Vérifiez le ticker ou la connexion.", icon="❌")
                    return None
                time.sleep(delay)

    def preprocess_data(self, data):
        if data.isnull().any().any():
            data = data.fillna(method='ffill')
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        if 'Adj Close' in data.columns:
            features.append('Adj Close')
        data = data[features]
        scaled_data = self.scaler.fit_transform(data)
        return pd.DataFrame(scaled_data, columns=features, index=data.index)

    def create_sequences(self, data, seq_length):
        X, y = [], []
        price_column = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        for i in range(len(data) - seq_length):
            X.append(data.iloc[i:(i + seq_length)].values)
            y.append(data.iloc[i + seq_length][price_column])
        return np.array(X), np.array(y)

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, model_type='LSTM', dropout=0.2):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if model_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError(f"Type de modèle non supporté: {model_type}")
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        if x.dim() > 3:
            x = x.squeeze()
            if x.dim() > 3:
                raise ValueError(f"Input shape {x.shape} still has too many dimensions")
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class BidirectionalRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, model_type='LSTM', dropout=0.3):
        super(BidirectionalRNNModel, self).__init__()
        if model_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        else:
            raise ValueError(f"Type de modèle non supporté: {model_type}")
        self.fc = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, x):
        if x.dim() > 3:
            x = x.squeeze()
            if x.dim() > 3:
                raise ValueError(f"Input shape {x.shape} still has too many dimensions")
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size=3, dropout=0.3):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, input_size, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        if x.dim() > 3:
            x = x.squeeze()
            if x.dim() > 3:
                raise ValueError(f"Input shape {x.shape} still has too many dimensions")
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# FinancialDataEDA Class
class FinancialDataEDA:
    def __init__(self, entreprise, start_date, end_date):
        self.entreprise = entreprise
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.returns = None
        
    def download_data(self):
        try:
            self.data = yf.download(self.entreprise, start=self.start_date, end=self.end_date)
            if self.data.empty:
                raise ValueError(f"Aucune donnée trouvée pour {self.entreprise}")
            self.data.index = pd.to_datetime(self.data.index)
            self.data = self.data.sort_index()
            self.data = self.data.fillna(method='ffill')
            return True
        except Exception as e:
            st.error(f"❌ Erreur lors du téléchargement des données: {str(e)}", icon="❌")
            return False
    
    def get_price_column(self):
        return 'Adj Close' if 'Adj Close' in self.data.columns else 'Close'

    def calculate_returns(self):
        price_column = self.get_price_column()
        if price_column is None:
            self.returns = pd.Series(dtype=float)
            return
        self.returns = self.data[price_column].pct_change().dropna()
    
    def basic_statistics(self):
        price_column = self.get_price_column()
        if price_column is None:
            return pd.DataFrame()
        prix_stats = {
            'Moyenne': float(self.data[price_column].mean()),
            'Écart-type': float(self.data[price_column].std()),
            'Minimum': float(self.data[price_column].min()),
            'Maximum': float(self.data[price_column].max()),
            'Médiane': float(self.data[price_column].median()),
            'Skewness': float(stats.skew(self.data[price_column])),
            'Kurtosis': float(stats.kurtosis(self.data[price_column]))
        }
        rendements_stats = {
            'Moyenne': float(self.returns.mean()) if self.returns is not None else float('nan'),
            'Écart-type': float(self.returns.std()) if self.returns is not None else float('nan'),
            'Minimum': float(self.returns.min()) if self.returns is not None else float('nan'),
            'Maximum': float(self.returns.max()) if self.returns is not None else float('nan'),
            'Médiane': float(self.returns.median()) if self.returns is not None else float('nan'),
            'Skewness': float(stats.skew(self.returns)) if self.returns is not None else float('nan'),
            'Kurtosis': float(stats.kurtosis(self.returns)) if self.returns is not None else float('nan')
        }
        stats_df = pd.DataFrame({
            'Statistique': list(prix_stats.keys()),
            'Prix': list(prix_stats.values()),
            'Rendements': list(rendements_stats.values())
        })
        return stats_df
    
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
            title=f'Évolution du prix de {self.entreprise}',
            yaxis_title='Prix ($)',
            xaxis_title='Date',
            template='plotly_white',
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig
    
    def plot_volume(self):
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=self.data.index,
            y=self.data['Volume'],
            name='Volume'
        ))
        fig.update_layout(
            title=f'Volume d\'échanges de {self.entreprise}',
            yaxis_title='Volume',
            xaxis_title='Date',
            template='plotly_white',
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig
    
    def plot_returns_distribution(self):
        if self.returns is None or self.returns.empty:
            return None
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('Distribution des rendements', 'QQ-Plot'))
        fig.add_trace(
            go.Histogram(x=self.returns, name='Rendements', nbinsx=50),
            row=1, col=1
        )
        returns_sorted = np.sort(self.returns.dropna())
        n = len(returns_sorted)
        if n < 2:
            return None
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=returns_sorted,
                mode='markers',
                name='QQ-Plot',
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        min_val = min(theoretical_quantiles.min(), returns_sorted.min())
        max_val = max(theoretical_quantiles.max(), returns_sorted.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Ligne de référence',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=1
        )
        fig.update_layout(
            title=f'Analyse des rendements de {self.entreprise}',
            template='plotly_white',
            height=800,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig
    
    def correlation_analysis(self):
        corr_matrix = self.data.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu'
        ))
        fig.update_layout(
            title=f'Matrice de corrélation - {self.entreprise}',
            template='plotly_white',
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig, corr_matrix
    
    def volatility_analysis(self):
        if self.returns is None or self.returns.empty:
            return None
        volatility = self.returns.rolling(window=20).std() * np.sqrt(252)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=volatility.index,
            y=volatility,
            name='Volatilité annuelle'
        ))
        fig.update_layout(
            title=f'Volatilité des rendements de {self.entreprise} (20 jours)',
            yaxis_title='Volatilité annuelle',
            xaxis_title='Date',
            template='plotly_white',
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig
    
    def generate_report(self):
        if not self.download_data():
            return None
        self.calculate_returns()
        price_column = self.get_price_column()
        stats_df = self.basic_statistics()
        stats_md = "| Statistique | Prix | Rendements |\n"
        stats_md += "|------------|------|------------|\n"
        for _, row in stats_df.iterrows():
            prix = float(row['Prix'])
            rendements = float(row['Rendements'])
            stats_md += f"| {row['Statistique']} | {prix:.4f} | {rendements:.4f} |\n"
        min_price = float(self.data[price_column].min()) if price_column else float('nan')
        max_price = float(self.data[price_column].max()) if price_column else float('nan')
        total_days = int(len(self.data))
        missing_values = int(self.data.isnull().sum().sum())
        available_columns = ', '.join(str(col) for col in self.data.columns)
        report = (
            f"# Rapport d'Exploration des Données - {self.entreprise}\n\n"
            f"## Période d'analyse\n"
            f"- Date de début: {self.start_date.strftime('%Y-%m-%d')}\n"
            f"- Date de fin: {self.end_date.strftime('%Y-%m-%d')}\n\n"
            f"## Statistiques de base\n"
            f"{stats_md}\n"
            f"## Analyse des données\n"
            f"- Nombre total de jours de trading: {total_days}\n"
            f"- Nombre de valeurs manquantes: {missing_values}\n"
            f"- Plage de prix: ${min_price:.2f} - ${max_price:.2f}\n"
            f"- Colonnes disponibles: {available_columns}\n"
        )
        return report

# Load saved models
def load_model(model_name, input_size, params, device):
    try:
        if model_name.startswith('Bidirectional'):
            model_type = 'LSTM' if 'LSTM' in model_name else 'GRU'
            model = BidirectionalRNNModel(
                input_size,
                params['hidden_size'],
                params['num_layers'],
                model_type,
                params['dropout']
            ).to(device)
        elif model_name == 'CNN-LSTM':
            model = CNNLSTMModel(
                input_size,
                params['hidden_size'],
                params['num_layers'],
                3,
                params['dropout']
            ).to(device)
        else:
            model = RNNModel(
                input_size,
                params['hidden_size'],
                params['num_layers'],
                model_name,
                params['dropout']
            ).to(device)
        model_path = f"saved_models/best_model_{model_name}.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle {model_name}: {str(e)}", icon="❌")
        return None

# Barre latérale
with st.sidebar:
    st.markdown("<h2 style='color: var(--primary);'>📈 Stock Analyzer</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='custom-container'><h4 style='color: var(--accent);'>Navigation</h4></div>", unsafe_allow_html=True)
    page = st.radio(
        "",
        ["🏠 Accueil", "📊 EDA", "🔮 Predictions", "ℹ️ À propos"],
        label_visibility="collapsed"
    )
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='custom-container'><h4 style='color: var(--accent);'>Configuration</h4></div>", unsafe_allow_html=True)
    
    # Section fictive pour imiter l'authentification de NLP Magic Hub
    with st.form("config_form"):
        config_key = st.text_input("Clé de configuration (optionnel)", type="password", placeholder="Entrez une clé (non requis)")
        if st.form_submit_button("Vérifier", use_container_width=True):
            if config_key:
                st.success("✅ Clé vérifiée (simulation)", icon="✅")
            else:
                st.warning("⚠️ Aucune clé saisie, configuration par défaut utilisée", icon="⚠️")
    
    st.markdown("""
    <div style='font-size: 0.9em; color: var(--accent);'>
        <a href='https://github.com/TheBeyonder237' target='_blank'>🔗 GitHub du créateur</a><br>
        <a href='mailto:ngouedavidrogeryannick@gmail.com'>🔗 Contacter par email</a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: var(--text); font-size: 0.9em;'>
        <p>Créé par Ngôue David</p>
        <p>📧 ngouedavidrogeryannick@gmail.com</p>
        <p>Projet RNN pour la finance</p>
    </div>
    """, unsafe_allow_html=True)

# Contenu principal
if page == "🏠 Accueil":
    st.markdown("<h1 style='color: var(--primary);'>📈 Stock Analyzer</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        st.markdown("""
        <div class='custom-container'>
            <p>Explorez et prédisez les tendances boursières avec des modèles RNN avancés :</p>
            <ul>
                <li>📊 Analyse exploratoire des données financières</li>
                <li>🔮 Prédictions futures avec modèles RNN pré-entraînés</li>
                <li>📈 Visualisations interactives avec Plotly</li>
                <li>🔍 Chargement automatique des modèles sauvegardés</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📌 Guide de démarrage"):
            st.markdown("""
            1. Assurez-vous que les modèles pré-entraînés sont dans le dossier `saved_models/`
            2. Sélectionnez une page dans la barre latérale
            3. Pour l'EDA, entrez un ticker et une période
            4. Pour les prédictions, choisissez un modèle et une durée
            5. Explorez les résultats et téléchargez les rapports !
            """)
    
    with col2:
        st.markdown("""
        <div class='custom-container'>
            <h4 style='color: var(--accent);'>🛠️ Stack Technique</h4>
            <ul>
                <li><strong>yfinance</strong>: Données boursières</li>
                <li><strong>PyTorch</strong>: Modèles RNN</li>
                <li><strong>Streamlit</strong>: Interface</li>
                <li><strong>Plotly</strong>: Visualisations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

elif page == "📊 EDA":
    st.markdown("<h1 style='color: var(--primary);'>📊 Analyse Exploratoire des Données</h1>", unsafe_allow_html=True)
    
    with st.form("eda_form"):
        st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            entreprise = st.text_input("Symbole boursier", value="TSLA", placeholder="Ex: TSLA")
        with col2:
            period = st.selectbox("Période", ["1 an", "2 ans", "3 ans", "4 ans", "Personnalisée"])
        with col3:
            submit_eda = st.form_submit_button("🔎 Analyser", use_container_width=True)
        
        if period == "Personnalisée":
            col4, col5 = st.columns(2)
            with col4:
                start_date = st.date_input("Date de début", value=datetime.now() - timedelta(days=2*365))
            with col5:
                end_date = st.date_input("Date de fin", value=datetime.now())
        else:
            years = {"1 an": 1, "2 ans": 2, "3 ans": 3, "4 ans": 4}[period]
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years*365)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    if submit_eda:
        with st.spinner("🔍 Analyse des données en cours..."):
            eda = FinancialDataEDA(entreprise, start_date, end_date)
            report = eda.generate_report()
            if report:
                st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
                st.markdown("<h4>Rapport</h4>", unsafe_allow_html=True)
                st.markdown(report, unsafe_allow_html=True)
                
                report_bytes = report.encode('utf-8')
                st.download_button(
                    label="📥 Télécharger le rapport",
                    data=report_bytes,
                    file_name=f"eda_report_{entreprise}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
                
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<h4>Aperçu des Données</h4>", unsafe_allow_html=True)
                st.dataframe(eda.data.head(), use_container_width=True)
                
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<h4>Visualisations</h4>", unsafe_allow_html=True)
                
                # Price Evolution
                fig_price = eda.plot_price_evolution()
                if fig_price:
                    st.plotly_chart(fig_price, use_container_width=True)
                    img_buffer = io.BytesIO()
                    fig_price.write_image(img_buffer, format="png")
                    st.download_button(
                        label="📥 Télécharger le graphique de prix",
                        data=img_buffer,
                        file_name=f"price_evolution_{entreprise}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                # Volume
                fig_volume = eda.plot_volume()
                if fig_volume:
                    st.plotly_chart(fig_volume, use_container_width=True)
                    img_buffer = io.BytesIO()
                    fig_volume.write_image(img_buffer, format="png")
                    st.download_button(
                        label="📥 Télécharger le graphique de volume",
                        data=img_buffer,
                        file_name=f"volume_{entreprise}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                # Returns Distribution
                fig_returns = eda.plot_returns_distribution()
                if fig_returns:
                    st.plotly_chart(fig_returns, use_container_width=True)
                    img_buffer = io.BytesIO()
                    fig_returns.write_image(img_buffer, format="png")
                    st.download_button(
                        label="📥 Télécharger l'analyse des rendements",
                        data=img_buffer,
                        file_name=f"returns_analysis_{entreprise}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                # Correlation Matrix
                fig_corr, corr_matrix = eda.correlation_analysis()
                if fig_corr:
                    st.plotly_chart(fig_corr, use_container_width=True)
                    img_buffer = io.BytesIO()
                    fig_corr.write_image(img_buffer, format="png")
                    st.download_button(
                        label="📥 Télécharger la matrice de corrélation",
                        data=img_buffer,
                        file_name=f"correlation_matrix_{entreprise}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    with st.expander("📊 Matrice de corrélation"):
                        st.dataframe(corr_matrix, use_container_width=True)
                
                # Volatility
                fig_volatility = eda.volatility_analysis()
                if fig_volatility:
                    st.plotly_chart(fig_volatility, use_container_width=True)
                    img_buffer = io.BytesIO()
                    fig_volatility.write_image(img_buffer, format="png")
                    st.download_button(
                        label="📥 Télécharger l'analyse de volatilité",
                        data=img_buffer,
                        file_name=f"volatility_{entreprise}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                st.success("✅ Analyse terminée !", icon="✅")
                st.markdown("</div>", unsafe_allow_html=True)

elif page == "🔮 Predictions":
    st.markdown("<h1 style='color: var(--primary);'>🔮 Prédictions Futures</h1>", unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1], gap="medium")
        with col1:
            entreprise = st.text_input("Symbole boursier", value="TSLA", placeholder="Ex: TSLA")
        with col2:
            pred_weeks = st.slider("Durée de prédiction (semaines)", 1, 12, 3)
        
        model_types = list(MODEL_PARAMS.keys())
        if not model_types:
            st.error("❌ Aucun modèle disponible. Vérifiez les paramètres définis.", icon="❌")
            st.stop()
        
        selected_model = st.selectbox("Sélectionner le modèle", model_types)
        submit_pred = st.form_submit_button("✨ Prédire", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display model parameters
    st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
    st.markdown("<h4>Paramètres du Modèle</h4>", unsafe_allow_html=True)
    st.markdown(f"**Modèle sélectionné : {selected_model}**")
    params = MODEL_PARAMS[selected_model]
    st.markdown(f"""
    - **Longueur de séquence** : {params['seq_length']}
    - **Taille de couche cachée** : {params['hidden_size']}
    - **Nombre de couches** : {params['num_layers']}
    - **Dropout** : {params['dropout']}
    - **Taux d'apprentissage** : {params['lr']}
    """)
    st.info("ℹ️ Les modèles sont pré-entraînés avec ces hyperparamètres fixes. Les métriques de performance ne sont pas disponibles dans cette version.", icon="ℹ️")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if submit_pred:
        with st.spinner("🔮 Chargement du modèle et génération des prédictions..."):
            # Download and preprocess data
            data_loader = DataLoader(entreprise, datetime.now() - timedelta(days=2*365), datetime.now())
            raw_data = data_loader.download_data()
            if raw_data is None:
                st.stop()
            processed_data = data_loader.preprocess_data(raw_data)
            
            # Load model
            params = MODEL_PARAMS[selected_model]
            input_size = processed_data.shape[1]
            model = load_model(selected_model, input_size, params, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            if model is None:
                st.stop()
            
            # Generate predictions
            start_time = time.time()
            seq_length = params['seq_length']
            future_steps = pred_weeks * 7
            last_seq = processed_data.iloc[-seq_length:].values
            future_preds = []
            current_seq = last_seq.copy()
            
            model.eval()
            with torch.no_grad():
                for _ in range(future_steps):
                    input_seq = torch.FloatTensor(current_seq).unsqueeze(0).to(model.device)
                    next_pred = model(input_seq).cpu().detach().numpy().squeeze()
                    future_preds.append(next_pred)
                    current_seq = np.vstack([current_seq[1:], np.append(current_seq[-1, :-1], next_pred)])
            
            future_dates = pd.date_range(processed_data.index[-1] + pd.Timedelta(days=1), periods=future_steps)
            future_preds_inv = data_loader.scaler.inverse_transform(
                np.concatenate([np.zeros((len(future_preds), processed_data.shape[1]-1)), np.array(future_preds).reshape(-1,1)], axis=1)
            )[:,-1]
            end_time = time.time()
            
            # Plot predictions
            st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
            st.markdown(f"<h4>Résultat ({end_time-start_time:.2f}s)</h4>", unsafe_allow_html=True)
            fig_full = go.Figure()
            price_col = 'Close' if 'Close' in processed_data.columns else processed_data.columns[0]
            fig_full.add_trace(go.Scatter(x=processed_data.index, y=processed_data[price_col], name='Historique'))
            fig_full.add_trace(go.Scatter(x=future_dates, y=future_preds_inv, name='Prédiction future', line=dict(color=color_palette['accent'])))
            fig_full.update_layout(
                title=f'Projection future - {selected_model}',
                xaxis_title='Date',
                yaxis_title='Prix ($)',
                template='plotly_white',
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_full, use_container_width=True)
            
            img_buffer = io.BytesIO()
            fig_full.write_image(img_buffer, format="png")
            st.download_button(
                label="📥 Télécharger la projection future",
                data=img_buffer,
                file_name=f"future_projection_{selected_model}_{entreprise}.png",
                mime="image/png",
                use_container_width=True
            )
            
            pred_df = pd.DataFrame(future_preds_inv, index=future_dates, columns=['Predicted'])
            csv_buffer = io.StringIO()
            pred_df.to_csv(csv_buffer)
            st.download_button(
                label="📥 Télécharger les prédictions",
                data=csv_buffer.getvalue(),
                file_name=f"predictions_{entreprise}_future_{selected_model}.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.success(f"✅ Prédictions générées en {end_time-start_time:.2f}s !", icon="✅")
            st.markdown("</div>", unsafe_allow_html=True)

elif page == "ℹ️ À propos":
    st.markdown("<h1 style='color: var(--primary);'>🌟 À propos</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        if about_animation:
            st_lottie(about_animation, height=250, key="about_animation")
        else:
            st.warning("⚠️ Animation À propos non chargée.", icon="⚠️")
        st.image(
            "https://avatars.githubusercontent.com/u/TheBeyonder237",
            width=180,
            caption="Ngôue David",
            output_format="auto",
            use_container_width=False
        )
        st.markdown("""
        <div style='text-align:center; margin-top:1em;'>
            <button class='about-contact-btn' onclick="window.open('mailto:ngouedavidrogeryannick@gmail.com')">📧 Email</button>
            <button class='about-contact-btn' onclick="window.open('https://github.com/TheBeyonder237')">🌐 GitHub</button>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='custom-container'>
            <h4>À propos du créateur</h4>
            <p style='color:#e0f0ff;'>
                Je suis Ngôue David, étudiant en Master 2 en IA et Big Data. Passionné par l'intelligence artificielle appliquée à la finance et à la santé, je développe des solutions innovantes pour analyser et prédire les tendances.
            </p>
            <h4 style='color:#7dd3fc;'>Compétences</h4>
            <span class='badge'>Python</span>
            <span class='badge'>Machine Learning</span>
            <span class='badge'>Deep Learning</span>
            <span class='badge'>Data Science</span>
            <span class='badge'>Streamlit</span>
            <span class='badge'>Plotly</span>
            <h4 style='color:#7dd3fc; margin-top:1.5em;'>Projets récents</h4>
            <ul style='color:#e0f0ff;'>
                <li><b>💳 Credit Card Expenditure Predictor</b> : Prédiction des dépenses.</li>
                <li><b>🫀 HeartGuard AI</b> : Analyse des risques cardiaques.</li>
                <li><b>🔍 Multi-IA</b> : Plateforme multi-modèles IA.</li>
            </ul>
            <h4 style='color:#7dd3fc;'>Stack Technique</h4>
            <ul style='color:#e0f0ff;'>
                <li><strong>yfinance</strong>: Données boursières</li>
                <li><strong>PyTorch</strong>: Modèles RNN</li>
                <li><strong>Streamlit</strong>: Interface utilisateur</li>
                <li><strong>Plotly</strong>: Visualisations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #e0f0ff; padding: 20px;'>
        Développé avec ❤️ par Ngoué David
    </div>
    """, unsafe_allow_html=True)