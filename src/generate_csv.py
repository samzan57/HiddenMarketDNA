# generate_csv.py
# Génère les fichiers CSV pour tous les ETF du projet
# Colonne 'Adj Close' si disponible, sinon 'Close'

import os
import yfinance as yf

# --- 1. Configuration ---
ETF_LIST = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU"]
DATA_DIR = "data/raw"

START_DATE = "2015-01-01"
END_DATE = "2026-01-01"

# --- 2. Créer le dossier si nécessaire ---
os.makedirs(DATA_DIR, exist_ok=True)

# --- 3. Télécharger chaque ETF ---
for ticker in ETF_LIST:
    print(f"Téléchargement de {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE)
    
    if df.empty:
        print(f" Aucune donnée trouvée pour {ticker}")
        continue
    
    # Vérifier si 'Adj Close' existe
    if 'Adj Close' in df.columns:
        price_col = 'Adj Close'
    elif 'Close' in df.columns:
        print(f" 'Adj Close' absent pour {ticker}, utilisation de 'Close' à la place")
        price_col = 'Close'
    else:
        print(f" Pas de colonne de prix disponible pour {ticker}")
        continue
    
    # Garder uniquement la colonne de prix
    df = df[[price_col]].reset_index()  # reset_index pour créer 'Date'
    df.rename(columns={price_col: 'Adj Close'}, inplace=True)  # renommer en 'Adj Close' pour compatibilité
    
    # Sauvegarde CSV
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df.to_csv(file_path, index=False)
    print(f" {ticker}.csv sauvegardé dans {DATA_DIR}")

print("Tous les fichiers CSV générés avec Date et Adj Close !")
