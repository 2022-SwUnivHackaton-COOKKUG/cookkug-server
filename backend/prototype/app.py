import streamlit as st
from load_reco import Recommender


reco = Recommender()

@st.cache
def get_recipe():    
    return reco.recipe

text = st.empty()
recipe_id = text.text_input('>> select favorable recipe and input recipe id')

df = get_recipe()
table = st.empty()
table.dataframe(df)
main = st.button('Return original table')
if main:
    table.dataframe(df)

if recipe_id:
    st.write(f'>> you selected recipe({recipe_id})')
    reco.calc_sim(int(recipe_id))
    table.dataframe(reco.recipe.sort_values(f'{recipe_id}_유사도', ascending=False)[:20])