import streamlit as st

user_input = st.text_input("Frage zur Creme:")

if user_input:
    response = requests.post(
        "http://127.0.0.1:8000/explain",
        json=data
    )

    st.write(response.json()["explanation"])
