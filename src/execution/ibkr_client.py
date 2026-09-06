# src/execution/ibkr_client.py
"""
Client IBKR via ib_insync.

PRÉREQUIS :
  1. TWS ou IB Gateway ouvert et connecté (compte paper trading)
  2. API Settings : Enable Socket Clients = True
  3. pip install ib_insync

CONNEXION :
  Host, port et client ID sont lus depuis les variables d'environnement
  IBKR_HOST / IBKR_PORT / IBKR_CLIENT_ID (voir .env.example) — le port
  dépend de ton installation TWS/Gateway (paper ou live, TWS ou Gateway),
  voir la documentation IBKR pour la valeur qui te concerne.
"""

import logging
import os
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # charge .env s'il existe — sans effet si les variables sont déjà définies

logger = logging.getLogger(__name__)


class IBKRClient:
    """
    Wrapper ib_insync pour TWS paper trading.

    Paramètres
    ----------
    host : str
        Adresse TWS/Gateway. Défaut : variable d'environnement IBKR_HOST
        (ou 127.0.0.1 si absente).
    port : int
        Port API TWS/Gateway. Défaut : variable d'environnement IBKR_PORT
        — voir .env.example et la documentation IBKR pour la valeur
        correspondant à ta configuration (paper/live, TWS/Gateway).
    client_id : int
        ID client API (doit être unique si plusieurs connexions simultanées).
        Défaut : variable d'environnement IBKR_CLIENT_ID (ou 1 si absente).
    timeout : int
        Timeout de connexion en secondes.
    """

    def __init__(
        self,
        host:      Optional[str] = None,
        port:      Optional[int] = None,
        client_id: Optional[int] = None,
        timeout:   int = 10,
    ):
        host = host or os.getenv("IBKR_HOST", "127.0.0.1")
        client_id = client_id if client_id is not None else int(os.getenv("IBKR_CLIENT_ID", "1"))
        if port is None:
            env_port = os.getenv("IBKR_PORT")
            if not env_port:
                raise ValueError(
                    "Port IBKR non défini — passe `port=` ou renseigne IBKR_PORT "
                    "dans ton .env (voir .env.example)."
                )
            port = int(env_port)
        # Python 3.10+ ne crée plus l'event loop automatiquement — ib_insync/eventkit en a besoin
        import asyncio
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        try:
            from ib_insync import IB
        except ImportError:
            raise ImportError("ib_insync non installé. Lancer : pip install ib_insync")

        self._IB        = IB
        self.host       = host
        self.port       = port
        self.client_id  = client_id
        self.timeout    = timeout
        self.ib         = IB()
        self._connected = False

    # ------------------------------------------------------------------
    # Connexion
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self._connected:
            return
        self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=self.timeout)
        self._connected = True
        self.ib.reqMarketDataType(3)  # données différées 15 min (paper account)
        account = self.ib.managedAccounts()[0]
        logger.info(f"[IBKR] Connecté — compte {account} | port {self.port}")

    def disconnect(self) -> None:
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("[IBKR] Déconnecté")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ------------------------------------------------------------------
    # Compte
    # ------------------------------------------------------------------

    def get_net_liquidation(self) -> float:
        """Valeur nette du compte (cash + positions)."""
        summary = self.ib.accountSummary()
        for item in summary:
            if item.tag == "NetLiquidation":
                return float(item.value)
        raise RuntimeError("NetLiquidation introuvable dans accountSummary()")

    def get_cash(self) -> float:
        """Cash disponible (TotalCashValue)."""
        summary = self.ib.accountSummary()
        for item in summary:
            if item.tag == "TotalCashValue":
                return float(item.value)
        return 0.0

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> pd.Series:
        """
        Retourne les positions actuelles.

        Returns
        -------
        pd.Series
            index = ticker (str), values = nombre d'actions (float)
        """
        positions = self.ib.positions()
        data = {}
        for pos in positions:
            symbol = pos.contract.symbol
            data[symbol] = pos.position
        return pd.Series(data, dtype=float)

    # ------------------------------------------------------------------
    # Prix
    # ------------------------------------------------------------------

    def get_snapshot_prices(self, symbols: List[str]) -> pd.Series:
        """
        Prix en temps réel (snapshot) pour une liste de tickers US.

        Utilise reqMktData avec snapshot=True pour ne pas s'abonner
        en continu (évite de consommer les lignes de données).

        Returns
        -------
        pd.Series
            index = ticker, values = last price (float)
        """
        from ib_insync import Stock

        contracts = [Stock(sym, "SMART", "USD") for sym in symbols]
        self.ib.qualifyContracts(*contracts)

        prices = {}
        for contract in contracts:
            ticker = self.ib.reqMktData(contract, "", snapshot=True, regulatorySnapshot=False)
            self.ib.sleep(2)  # attend la réponse snapshot
            price = ticker.last if ticker.last and ticker.last > 0 else ticker.close
            if price and price > 0:
                prices[contract.symbol] = float(price)
            else:
                logger.warning(f"[IBKR] Prix indisponible pour {contract.symbol}")
                prices[contract.symbol] = float("nan")
            self.ib.cancelMktData(contract)

        return pd.Series(prices)

    # ------------------------------------------------------------------
    # Ordres
    # ------------------------------------------------------------------

    def place_market_order(self, symbol: str, shares: int) -> Optional[object]:
        """
        Place un ordre au marché.

        Paramètres
        ----------
        symbol : str
            Ticker (ex: "XLK").
        shares : int
            Positif = achat, négatif = vente.

        Returns
        -------
        Trade object ib_insync (None si shares == 0).
        """
        if shares == 0:
            return None

        from ib_insync import Stock, MarketOrder

        action   = "BUY" if shares > 0 else "SELL"
        quantity = abs(shares)

        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)

        order = MarketOrder(action, quantity)
        order.tif = "GTC"
        trade = self.ib.placeOrder(contract, order)
        logger.info(f"[IBKR] Ordre {action} {quantity} {symbol} envoyé (orderId={trade.order.orderId})")
        return trade

    def wait_for_fills(self, timeout: int = 30) -> None:
        """Attend que tous les ordres ouverts soient remplis (max timeout sec)."""
        self.ib.sleep(timeout)
        open_orders = self.ib.openOrders()
        if open_orders:
            logger.warning(f"[IBKR] {len(open_orders)} ordre(s) non rempli(s) après {timeout}s")

    def cancel_all_orders(self) -> None:
        """Annule tous les ordres ouverts (sécurité)."""
        self.ib.reqGlobalCancel()
        logger.warning("[IBKR] Tous les ordres annulés")
