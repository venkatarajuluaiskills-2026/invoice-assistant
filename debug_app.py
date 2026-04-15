import streamlit as st

st.title("🛠️ AI Invoice Assistant - System Check")
st.success("If you can see this, the server is HEALTHY! ✅")
st.info("The issue is in the AI library configuration. We will now find which one.")

st.write("Current Python Version:", st.info(f"{st.__version__}"))
