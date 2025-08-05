import altair as alt
import pandas as pd
import streamlit as st

data = pd.DataFrame({
    '項目': ['A', 'B', 'C'],
    '値': [100, 60, 80],
    '正規化値': [100, 60, 80]
})

chart = alt.Chart(data).mark_bar().encode(
    x='正規化値:Q',
    y=alt.Y('項目:N', sort=data['項目'].tolist())
)

labels = alt.Chart(data).mark_text(
    align='right',
    baseline='middle',
    dx=-5,
    color='black'
).encode(
    x=alt.value(0),
    y=alt.Y('項目:N', sort=data['項目'].tolist()),
    text='値:Q'
)

st.altair_chart(chart + labels, use_container_width=True)
