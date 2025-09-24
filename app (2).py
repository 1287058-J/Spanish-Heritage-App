# app.py
# Streamlit app: historical data -> polynomial regression (deg >= 3) with analysis and comparisons
# Author: ChatGPT (for student project). Uses World Bank API (WDI) for country indicators.

import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from io import StringIO
import base64
import math
from datetime import datetime

st.set_page_config(page_title="Historic Polynomial Regression (Latin countries)", layout="wide")

# ----------------------
# Helper / config
# ----------------------
# Default country choices (wealthiest Latin countries set)
DEFAULT_COUNTRIES = {
    "Uruguay": "URY",
    "Panama": "PAN",
    "Chile": "CHL"
}

# Mapping categories to World Bank indicators (WDI)
INDICATOR_MAP = {
    "Population": ("SP.POP.TOTL", "total population (people)"),
    "Unemployment rate": ("SL.UEM.TOTL.ZS", "unemployment rate (% of labor force)"),
    "Education levels (0-25)": ("SE.SCH.LIFE", "expected years of schooling (years) -> rescaled to 0-25"),
    "Life expectancy": ("SP.DYN.LE00.IN", "life expectancy at birth (years)"),
    "Average wealth (GDP per capita US$)": ("NY.GDP.PCAP.CD", "GDP per capita (current US$)"),
    "Average income (GNI per capita US$)": ("NY.GNP.PCAP.CD", "GNI per capita (current US$)"),
    "Birth rate": ("SP.DYN.CBRT.IN", "crude birth rate (per 1000 people)"),
    "Immigration (net migration)": ("SM.POP.NETM", "net migration (number of people)"),
    "Murder Rate": ("VC.IHR.PSRC.P5", "intentional homicides (per 100,000 people)")
}

WORLD_BANK_BASE = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?date={start}:{end}&format=json&per_page=20000"

# ----------------------
# Functions
# ----------------------
def fetch_wb_series(country_code, indicator_code, start=1950, end=None):
    if end is None:
        end = datetime.now().year
    url = WORLD_BANK_BASE.format(country=country_code, indicator=indicator_code, start=start, end=end)
    resp = requests.get(url, timeout=20)
    if resp.status_code != 200:
        st.error(f"World Bank API error: {resp.status_code}")
        return pd.DataFrame(columns=["year", "value"])
    try:
        data = resp.json()
    except Exception as e:
        st.error(f"JSON decode error: {e}")
        return pd.DataFrame(columns=["year", "value"])
    if not isinstance(data, list) or len(data) < 2:
        return pd.DataFrame(columns=["year", "value"])
    records = data[1]
    rows = []
    for r in records:
        yr = r.get("date")
        val = r.get("value")
        if val is not None:
            try:
                val = float(val)
            except:
                continue
            rows.append({"year": int(yr), "value": val})
    if not rows:
        return pd.DataFrame(columns=["year", "value"])
    df = pd.DataFrame(rows)
    df = df.sort_values("year").reset_index(drop=True)
    return df

def rescale_education_to_25(series):
    factor = 25.0 / 18.0
    return series * factor

def fit_poly(years, values, degree):
    coeffs = np.polyfit(years, values, degree)
    p = np.poly1d(coeffs)
    return p

def poly_to_equation(p):
    coeffs = p.coeffs
    degree = len(coeffs)-1
    terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if abs(c) < 1e-12:
            continue
        coef = f"{c:.6g}"
        if power == 0:
            terms.append(f"{coef}")
        elif power == 1:
            terms.append(f"{coef}·x")
        else:
            terms.append(f"{coef}·x^{power}")
    eq = " + ".join(terms)
    return f"f(x) = {eq}"

def derivative_roots(p):
    dp = np.polyder(p)
    roots = np.roots(dp)
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-6:
            real_roots.append(float(r.real))
    return real_roots, dp

def second_derivative_at(p, x):
    ddp = np.polyder(p, 2)
    return float(np.polyval(ddp, x))

def printable_html(title, summary_html, plot_svg):
    html = f"""
    <html><head><meta charset="utf-8"><title>{title}</title></head>
    <body>
    <h1>{title}</h1>
    {summary_html}
    <div>{plot_svg}</div>
    </body></html>
    """
    return html
