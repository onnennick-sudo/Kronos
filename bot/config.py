"""Kronos paper-trading bot configuration. Single source of truth for all tunables.

Changing anything here requires a reviewed PR (see CLAUDE.md).
"""

# --- Broker (PAPER ONLY; hardcoded, never read from env) ---
PAPER_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"
DATA_FEED = "sip"              # free plan: SIP allowed when bars are >= 15 min old
DATA_DELAY_MINUTES = 16        # request end = now - this

# --- Model ---
MODEL_NAME = "NeoQuasar/Kronos-small"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_REVISION = "main"        # pin a commit sha after Day 0 benchmark
MAX_CONTEXT = 512
TORCH_THREADS = 4
SAMPLE_T = 1.0
SAMPLE_TOP_K = 0
SAMPLE_TOP_P = 0.9
BATCH_SIZE = 100               # max sequences per predict_batch call

# --- Daily forecast (post-close, ranks the universe for next session) ---
DAILY_LOOKBACK = 400
DAILY_PRED_LEN = 10
DAILY_PATHS = 10               # fallback 5 if Day 0 benchmark is too slow
DAILY_ELIGIBLE_R = 0.010       # R_med >= +1.0%
DAILY_BEARISH_R = -0.010       # R_med <= -1.0%
DAILY_AGREE = 0.60
N_CANDIDATES = 5
ROTATION_RANK_EXIT = 10        # exit if rank is worse than this (11+)
ATR_PERIOD = 14

# --- Hourly forecast (timing) ---
HOURLY_LOOKBACK = 400
HOURLY_PRED_LEN = 6
HOURLY_PATHS = 20
HOURLY_CONFIRM_R = 0.003       # CONFIRM_LONG: R_med >= +0.3% and agree >= 60%
HOURLY_CONFIRM_AGREE = 0.60
HOURLY_STRONG_NEG_R = -0.005   # STRONG_NEG: R_med <= -0.5% and agree >= 65%
HOURLY_STRONG_NEG_AGREE = 0.65
REGULAR_HOUR_LABELS = (9, 10, 11, 12, 13, 14, 15)   # Alpaca clock-hour bar labels (ET)

# --- Risk ---
MAX_POSITION_FRAC = 0.20       # user rule: 20% of equity per position
MAX_POSITIONS = 3              # recommended addition
MAX_EXPOSURE_FRAC = 0.60       # recommended addition
MAX_PER_SECTOR = 2             # recommended addition
DAILY_HALT_FRAC = 0.02         # halt new entries at -2% vs last_equity
HALT_FLATTEN = False
STOP_ATR_MULT = 2.0
TARGET_ATR_MULT = 3.0
TRAIL_BREAKEVEN_ATR = 1.0      # at +1 ATR unrealized -> stop to entry
TRAIL_START_ATR = 2.0          # at +2 ATR unrealized -> stop to last_close - 1.5 ATR
TRAIL_DISTANCE_ATR = 1.5
MAX_ORDERS_PER_RUN = 5
MAX_FAILED_TICKERS = 10        # more failures than this = system-wide problem
COST_BPS_PER_SIDE = 5          # backtest only

# --- Ops ---
WATCHDOG_TRADING_GAP_MIN = 150
DAILY_DEADLINE_ET = "18:30"
DAILY_READY_ET = "16:16"       # daily bar considered final after this
STATE_DIR_ENV = "BOT_STATE_DIR"  # path to bot-state checkout (default ./state)

MARKET_SYMBOL = "SPY"          # market filter + benchmark only, never traded

# --- Universe: ~50 largest S&P 500 names (frozen at build; refresh quarterly via PR) ---
# GICS sectors abbreviated.
SECTORS = {
    # Information Technology
    "AAPL": "IT", "MSFT": "IT", "NVDA": "IT", "AVGO": "IT", "ORCL": "IT", "CRM": "IT",
    "AMD": "IT", "ADBE": "IT", "CSCO": "IT", "ACN": "IT", "IBM": "IT", "INTU": "IT",
    "NOW": "IT", "QCOM": "IT",
    # Communication Services
    "GOOGL": "COMM", "META": "COMM", "NFLX": "COMM", "TMUS": "COMM",
    # Consumer Discretionary
    "AMZN": "DISC", "TSLA": "DISC", "HD": "DISC", "MCD": "DISC", "BKNG": "DISC",
    # Consumer Staples
    "WMT": "STAP", "COST": "STAP", "PG": "STAP", "KO": "STAP", "PEP": "STAP", "PM": "STAP",
    # Financials
    "BRK.B": "FIN", "JPM": "FIN", "V": "FIN", "MA": "FIN", "BAC": "FIN", "WFC": "FIN",
    "GS": "FIN", "MS": "FIN", "AXP": "FIN",
    # Health Care
    "LLY": "HC", "UNH": "HC", "JNJ": "HC", "ABBV": "HC", "MRK": "HC", "TMO": "HC",
    "ABT": "HC", "ISRG": "HC",
    # Energy
    "XOM": "EN", "CVX": "EN",
    # Industrials
    "GE": "IND", "CAT": "IND",
}
UNIVERSE = list(SECTORS)

# --- Backtest comparison universes ---
ETF_UNIVERSE = ["SPY", "QQQ", "DIA", "IWB", "RSP"]
CRYPTO_UNIVERSE = ["BTC/USD", "ETH/USD"]
