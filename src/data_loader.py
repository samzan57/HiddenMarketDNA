# src/data_loader.py

import os
import pandas as pd

class DataLoader:
    """
    Chargement et nettoyage des données de prix.
    Fournit un DataFrame de prix alignés, colonnes = tickers, index = Date.
    """

    def __init__(
        self,
        data_dir: str,
        price_column: str = "Adj Close",
        date_column: str = "Date"
    ):
        self.data_dir = data_dir
        self.price_column = price_column
        self.date_column = date_column

    def _load_single_csv(self, ticker: str) -> pd.Series:
        """
        Charge un fichier CSV pour un ticker donné
        et retourne une Series de prix indexée par date.
        """
        file_path = os.path.join(self.data_dir, f"{ticker}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")

        df = pd.read_csv(file_path)

        # Vérifier colonne Date
        if self.date_column not in df.columns:
            raise ValueError(f"Colonne '{self.date_column}' absente dans {ticker}")

        # Vérifier colonne prix
        if self.price_column not in df.columns:
            if 'Close' in df.columns:
                print(f" '{self.price_column}' absent pour {ticker}, utilisation de 'Close'")
                df[self.price_column] = df['Close']
            else:
                raise ValueError(f"Pas de colonne de prix valide pour {ticker}")

        # Index Date
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        df = df.sort_values(self.date_column)

        # Garder seulement la colonne prix et renommer
        series = df.set_index(self.date_column)[self.price_column].rename(ticker)

        # Forcer conversion en float et supprimer NaN
        series = pd.to_numeric(series, errors='coerce')
        series = series.dropna()

        return series

    def load_prices(self, tickers):
        """
        Charge tous les tickers et retourne un DataFrame de prix alignés.
        """
        series_list = []
        for ticker in tickers:
            s = self._load_single_csv(ticker)
            series_list.append(s)

        # Fusionner tous les tickers sur l'index Date
        prices = pd.concat(series_list, axis=1)
        # --- Nettoyage des colonnes ---
        for col in prices.columns:
            # Supprimer espaces et forcer conversion float
            prices[col] = prices[col].astype(str).str.replace(r'[^\d\.\-e]', '', regex=True)  # enlever tout sauf chiffres, point, - et e
            prices[col] = pd.to_numeric(prices[col], errors='coerce')  # forcer float
        prices = prices.dropna(how='any')

        # Vérifier types
        prices = prices.astype('float64')

        # Supprimer lignes avec NaN restantes
        prices = prices.dropna(how='any')

        return prices
        