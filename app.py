import streamlit as st
import pandas as pd
import numpy as np


# --- PAGE CONFIG ---
st.set_page_config(page_title="EV Intelligence Control Tower", layout="wide", page_icon="🔋")

# --- ACTUAL DATA LOADING ---
@st.cache_data
def load_data():
    # Load the real CSV files
    df_tel = pd.read_csv("Fleet_Telematics_Data.csv")
    df_qms = pd.read_csv("QMS_Supply_Chain_Data.csv")
    return df_tel, df_qms

try:
    df_telematics, df_qms = load_data()
except FileNotFoundError:
    st.error("⚠️ Data files not found. Please ensure 'Fleet_Telematics_Data.csv' and 'QMS_Supply_Chain_Data.csv' are in the same folder.")
    st.stop()

# --- DYNAMIC DATA AGGREGATION ---
worst_batch_id = df_telematics.groupby("Batch_ID")["Estimated_SoH"].min().idxmin()
df_telematics['Fleet_Segment'] = np.where(df_telematics['Batch_ID'] == worst_batch_id, 'At_Risk_Fleet_SoH', 'Normal_Fleet_SoH')

# Pivot data for chart
chart_data = df_telematics.groupby(['Timestamp', 'Fleet_Segment'])['Estimated_SoH'].mean().unstack()

# Extract dynamic metrics
worst_batch_details = df_qms[df_qms['Batch_ID'] == worst_batch_id].iloc[0]
affected_trucks = df_telematics[df_telematics['Batch_ID'] == worst_batch_id]['Asset_ID'].nunique()
total_trucks = df_telematics['Asset_ID'].nunique()
current_avg_soh = df_telematics['Estimated_SoH'].mean()

# --- HEADER ---
st.title("🔋 Cell-to-Wheel Intelligence: EV Control Tower")
st.markdown("Monitoring Supply Chain Risk, Asset Performance, and Fleet Electrification.")

# --- METRICS ROW ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total EV Assets", f"{total_trucks} Trucks", "+12 this month")
col2.metric("Fleet Average SoH", f"{current_avg_soh:.1f}%", "-0.4%")
col3.metric("Supply Chain Risk Alerts", "1 Critical", worst_batch_id)
col4.metric("Scope 1 Carbon Avoided", "420 Tons", "Tracking against Net Zero")

st.divider()

# --- MAIN DASHBOARD ---
colA, colB = st.columns([2, 1])

with colA:
    st.subheader("📉 Fleet Battery Degradation (APM)")
    st.markdown("Real-time State of Health (SoH) tracking vs. Predictive ML Baselines.")
    st.line_chart(chart_data, height=350)

with colB:
    st.subheader("⚠️ QMS Risk Detection")
    st.error(f"**CRITICAL ALERT: {worst_batch_id}**")
    st.markdown(f"""
    **Supplier:** {worst_batch_details['Supplier_Name']}
    * **Impurity Level:** {worst_batch_details['Impurity_Level_PPM']:.1f} PPM (Elevated)
    * **Internal Resistance:** {worst_batch_details['Initial_Internal_Resistance_mOhm']:.2f} mOhm (Elevated)
   
    **Impact:** {affected_trucks} assets in the current fleet are utilizing this batch. ML predicts a 95% probability of accelerated thermal degradation leading to premature failure.
    """)
    st.button("Initiate Supplier Audit", type="primary")

st.divider()

# --- THE AI AGENT INTERFACE (MODERNIZED GROUNDED LLM INTEGRATION) ---
st.subheader("🤖 APM & Supply Chain Intelligence Agent")
st.markdown("Ask the agent for maintenance routing, risk mitigation, or electrification readiness.")

# 1. Import the updated Google GenAI SDK modules
from google import genai
from google.genai import types

# 2. Configure the modern AI client
# It will check st.secrets first, falling back to your hardcoded string if missing
api_key = st.secrets.get("GEMINI_API_KEY", "AQ.Ab8RN6JsNXSqawTmHVvRu5LU7cNC_iUjhjcNP4bTb1wYhd0qiw")

if "genai_client" not in st.session_state:
    st.session_state.genai_client = genai.Client(api_key=api_key)

client = st.session_state.genai_client

# 3. Prepare the Data Context for the AI
qms_context = df_qms.to_csv(index=False)
latest_telematics = df_telematics.sort_values('Timestamp').groupby('Asset_ID').tail(1)
telematics_context = latest_telematics[['Asset_ID', 'Batch_ID', 'Estimated_SoH', 'Cell_Voltage_Imbalance_mV']].to_csv(index=False)

# 4. Define the System Context
system_instruction = f"""
You are an expert AI APM (Asset Performance Management) and Supply Chain Agent managing an EV fleet.
You are speaking directly to the Fleet Operations Manager via a control tower dashboard.

CRITICAL RULE: You must answer questions based STRICTLY and ONLY on the data provided below.
If the user asks a question that cannot be answered using this data, you must reply:
"I cannot answer that based on the current EV fleet and supply chain data." Do NOT use outside knowledge.
Do NOT format your response as a code block. Provide clear, professional, readable text.

--- FLEET SUMMARY ---
Total EV Assets: {total_trucks}
Fleet Average State of Health (SoH): {current_avg_soh:.1f}%
Active Critical Alert: Batch {worst_batch_id} from supplier {worst_batch_details['Supplier_Name']}

--- RAW SUPPLY CHAIN DATA (QMS) ---
{qms_context}

--- LATEST FLEET TELEMETRY (APM) ---
{telematics_context}
"""

# 5. Initialize modern stateful chat session in Streamlit session state
if "chat_session" not in st.session_state:
    # Use the new GenerateContentConfig class to pass system instructions and parameters
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.1
    )
    # Instantiate the chat instance via client.chats.create using a modern stable model
    st.session_state.chat_session = client.chats.create(
        model="gemini-2.5-flash",
        config=config
    )
    # Maintain a local UI history log to prevent API object mismatches during rendering
    st.session_state.ui_history = [
        {"role": "assistant", "text": "I am your EV Intelligence Agent. I have successfully ingested the latest supply chain (QMS) and telematics (APM) datasets. How can I assist you today?"}
    ]

# Display previous chat history using our UI logger
for message in st.session_state.ui_history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

# 6. Handle new user input
if prompt := st.chat_input("E.g., Which supplier provided the defective batch, and which exact trucks are affected?"):
   
    # Append & display the user's prompt immediately
    st.session_state.ui_history.append({"role": "user", "text": prompt})
    st.chat_message("user").markdown(prompt)
   
    with st.spinner("Agent analyzing fleet telemetry and QMS data..."):
        try:
            # Send message using the modern stateful chat object
            response = st.session_state.chat_session.send_message(prompt)
           
            # Append & display the AI response
            st.session_state.ui_history.append({"role": "assistant", "text": response.text})
            with st.chat_message("assistant"):
                st.markdown(response.text)
               
        except Exception as e:
            st.error(f"Failed to connect to AI Agent. Error: {e}")