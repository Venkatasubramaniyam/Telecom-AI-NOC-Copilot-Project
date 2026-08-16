import streamlit as st
import plotly.express as px

from utils.load_data import load_alarm_data

st.set_page_config(
    page_title="Telecom AI NOC Copilot",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Telecom AI NOC Copilot")

st.markdown("---")

df = load_alarm_data()

# KPI Cards

critical = len(df[df["Severity"] == "Critical"])
major = len(df[df["Severity"] == "Major"])
minor = len(df[df["Severity"] == "Minor"])
total = len(df)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Critical", critical)
c2.metric("Major", major)
c3.metric("Minor", minor)
c4.metric("Total Alarms", total)

st.markdown("---")

severity = st.selectbox(
    "Filter by Severity",
    ["All", "Critical", "Major", "Minor"]
)

if severity != "All":
    filtered_df = df[df["Severity"] == severity]
else:
    filtered_df = df

st.dataframe(
    filtered_df,
    width='stretch'
)

st.markdown("---")

fig = px.bar(
    df["Severity"].value_counts().reset_index(),
    x="Severity",
    y="count",
    color="Severity",
    title="Alarm Severity Distribution"
)

st.plotly_chart(
    fig,
    width='stretch'
)

from rag.chatbot import ask_question

st.header("AI Telecom Assistant")

question = st.text_input(
    "Ask about Telecom SOP"
)

if st.button("Search SOP"):

    answer = ask_question(question)

    st.success(answer)
 
from ai.root_cause import analyze_root_cause
from utils.load_data import load_incident_data

incident_df = load_incident_data()

st.divider()

st.header("🧠 AI Root Cause Analysis")

if st.button("Analyze Current Alarms"):

    with st.spinner("Analyzing alarms..."):

        result = analyze_root_cause(
            filtered_df,
            incident_df
        )

    st.success(result)