# Ad Bidding Optimization

A multi-armed bandit simulation that uses **Thompson Sampling** to learn which advertisement creative performs best while the campaign is running.

## Features
- Simulates multiple ad creatives
- Uses Beta-Bernoulli Bayesian posteriors
- Applies Thompson Sampling
- Balances exploration and exploitation
- Tracks impressions, clicks, CTR, and traffic share
- Visualizes how traffic is allocated across ads

## Technologies
- Python
- Streamlit
- NumPy
- Pandas

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works

Each ad starts with a Beta(1,1) prior. For every impression, the system samples a possible CTR from every ad's posterior distribution and selects the ad with the highest sampled CTR.

After observing whether the selected ad receives a click:
- click = success, so alpha increases
- no click = failure, so beta increases

Repeated decisions gradually allocate more traffic to ads that appear promising while still exploring uncertain ads.

## Project scope

This is an educational ad-optimization simulation. It does not connect to a real advertising platform or spend real money.
