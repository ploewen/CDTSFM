# Purpose:
# Retrieves S&P 500 stock tickers from Wikipedia.
# Downloads stock data from January 1st 2000 to November 1st 2025 for all retrieved tickers
# Computes daily simple return of stocks
# Saves return files to parquet

# Authors:
# - Code written by Philip Loewen

# References:
# Wikipedia contributors. (2025, November 18). List of S&P 500 companies.
# In Wikipedia, The Free Encyclopedia. Retrieved 18:40, November 18, 2025, from
# https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies&oldid=1322665983

# Yahoo! (n.d.). Yahoo Finance - Stock Market Live, quotes, Business & Finance News.
# Yahoo! Finance. https://finance.yahoo.com/

import yfinance as yf
import pandas as pd
import wikipedia
import os
from tqdm import tqdm
import logging

# Only print critical errors from yfinance since we want to handle download error
# messages ourselves.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Get the table of S&P 500 stocks from Wikipedia to access tickers
html = wikipedia.page("List of S&P 500 companies").html().encode("UTF-8")
table = pd.read_html(html)[0]
tickers = table["Symbol"].values


# Get Adjusted Close data for stock
def load_data(ticker, start="2000-01-01", end="2025-11-01", interval="1d"):
    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if data.empty:
            raise ValueError()
        data = data[["Adj Close"]]
        data.dropna(inplace=True)
        return data
    except Exception as e:
        raise e


# Get simple returns for stock
def compute_returns(df):
    df["return"] = df["Adj Close"].pct_change()
    df.dropna(inplace=True)
    return df


def main():
    # If the output path does not exist, create it
    output_path = "data/raw/SP500/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # List of stocks that were not downloaded
    failed_downloads = []
    # For each ticker download the file, compute the returns, and save to parquet
    for ticker in tqdm(tickers, desc="Downloading stock data", unit=" stocks"):
        try:
            # Try downloading and processing the data
            df = load_data(ticker)
            df = compute_returns(df)
            file_name = ticker + "-returns.parquet"
            df.to_parquet(os.path.join(output_path, file_name))
        except Exception:
            # Handle any errors that occur
            failed_downloads.append(ticker)
            continue

    print("Done downloading data.")

    # If some stocks were not downloaded print their ticker to console.
    if failed_downloads:
        print(f"\033[31m{'Warning: Could not download some stocks:'}\033[0m")
        print(", ".join(failed_downloads))


if __name__ == "__main__":
    main()
