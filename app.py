import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore

MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'
DATA = 'welcome_survey_simple_v2.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)
    return df_with_clusters

with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([{
        'age': age,
        'edu_level': edu_level,
        'fav_animals': fav_animals,
        'fav_place': fav_place,
        'gender': gender,
    }])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

st.header(f"Gdzie Ci najblizej ?")

st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba twoich znajomych", len(same_cluster_df))

st.header("Osoby z grupy")

# Wykres rozkładu wieku
fig_age = px.histogram(same_cluster_df.sort_values("age"), x="age", title="Rozkład wieku w grupie")
fig_age.update_layout(xaxis_title="Wiek", yaxis_title="Liczba osób")
st.plotly_chart(fig_age)

# Wykres pudełkowy dla wieku
fig_box_age = px.box(same_cluster_df, y="age", title="Rozkład wieku w grupie (Box plot)")
fig_box_age.update_layout(yaxis_title="Wiek")
st.plotly_chart(fig_box_age)

# Wykres rozkładu wykształcenia
fig_edu = px.histogram(same_cluster_df, x="edu_level", title="Rozkład wykształcenia w grupie")
fig_edu.update_layout(xaxis_title="Wykształcenie", yaxis_title="Liczba osób")
st.plotly_chart(fig_edu)

# Wykres kołowy dla wykształcenia
fig_pie_edu = px.pie(same_cluster_df, names='edu_level', title='Procentowy rozkład wykształcenia w grupie')
st.plotly_chart(fig_pie_edu)

# Wykres rozkładu ulubionych zwierząt
fig_animals = px.histogram(same_cluster_df, x="fav_animals", title="Rozkład ulubionych zwierząt w grupie")
fig_animals.update_layout(xaxis_title="Ulubione zwierzęta", yaxis_title="Liczba osób")
st.plotly_chart(fig_animals)

# Wykres rozkładu ulubionych miejsc
fig_place = px.histogram(same_cluster_df, x="fav_place", title="Rozkład ulubionych miejsc w grupie")
fig_place.update_layout(xaxis_title="Ulubione miejsce", yaxis_title="Liczba osób")
st.plotly_chart(fig_place)

# Wykres rozkładu płci
fig_gender = px.histogram(same_cluster_df, x="gender", title="Rozkład płci w grupie")
fig_gender.update_layout(xaxis_title="Płeć", yaxis_title="Liczba osób")
st.plotly_chart(fig_gender)

# Wykres kołowy dla płci
fig_pie_gender = px.pie(same_cluster_df, names='gender', title='Procentowy rozkład płci w grupie')
st.plotly_chart(fig_pie_gender)
