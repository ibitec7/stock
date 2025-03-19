# %%
import yfinance as yf
import pandas as pd
# %%
financials = yf.Ticker("NVDA").balance_sheet
# %%
financials.index
# %%
