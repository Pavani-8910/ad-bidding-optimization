import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Ad Bidding Optimization", page_icon="🎯")

st.title("🎯 Ad Bidding Optimization")
st.write("A multi-armed bandit simulation using Thompson Sampling to learn which ad creative performs best.")

st.sidebar.header("Simulation settings")
num_ads = st.sidebar.slider("Number of ad creatives", 2, 6, 3)
rounds = st.sidebar.slider("Number of ad impressions", 100, 5000, 1000, step=100)
seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

default_rates = [0.05, 0.08, 0.06, 0.10, 0.04, 0.07]
true_rates = []
for i in range(num_ads):
    rate = st.sidebar.slider(
        f"True CTR — Ad {chr(65+i)}",
        min_value=0.01, max_value=0.30,
        value=float(default_rates[i]), step=0.01
    )
    true_rates.append(rate)

rng = np.random.default_rng(int(seed))

# Beta(1,1) priors for Thompson Sampling
alpha = np.ones(num_ads)
beta_params = np.ones(num_ads)

records = []
clicks = np.zeros(num_ads, dtype=int)
impressions = np.zeros(num_ads, dtype=int)

for t in range(1, rounds + 1):
    sampled_ctr = rng.beta(alpha, beta_params)
    chosen = int(np.argmax(sampled_ctr))

    click = int(rng.random() < true_rates[chosen])
    impressions[chosen] += 1
    clicks[chosen] += click

    alpha[chosen] += click
    beta_params[chosen] += 1 - click

    records.append({
        "Round": t,
        "Ad": f"Ad {chr(65+chosen)}",
        "Click": click
    })

results = pd.DataFrame({
    "Ad": [f"Ad {chr(65+i)}" for i in range(num_ads)],
    "Impressions": impressions,
    "Clicks": clicks,
    "Observed CTR": np.divide(clicks, impressions, out=np.zeros(num_ads, dtype=float), where=impressions > 0),
    "True CTR (simulation)": true_rates
})
results["Traffic Share"] = results["Impressions"] / rounds

st.subheader("Results")
st.dataframe(
    results.style.format({
        "Observed CTR": "{:.2%}",
        "True CTR (simulation)": "{:.2%}",
        "Traffic Share": "{:.2%}"
    }),
    use_container_width=True
)

best = int(np.argmax(results["Observed CTR"]))
st.success(f"Highest observed CTR in this simulation: {results.iloc[best]['Ad']}")

st.subheader("Exploration vs. Exploitation")
st.write(
    "Thompson Sampling balances exploration and exploitation by sampling a possible "
    "CTR from each ad's Bayesian posterior and showing the impression to the ad with "
    "the highest sampled value."
)

st.subheader("Traffic allocation")
st.bar_chart(results.set_index("Ad")["Impressions"])

st.caption(
    "Educational simulation: clicks are generated from the user-provided simulated CTRs. "
    "No real advertising spend or bidding account is connected."
)
