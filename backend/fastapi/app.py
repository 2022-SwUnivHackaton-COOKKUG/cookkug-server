from typing import Optional
from fastapi import FastAPI
import uvicorn
from load_reco import Recommender
from load_combi import Combination
from fastapi.responses import FileResponse

reco = Recommender()
combi = Combination()
app = FastAPI()


#추천20개
@app.get('/reco/{recipe_ids}')
async def reco_recipe(recipe_ids: str):
    ret = []
    recipe_ids = list(map(int, recipe_ids.split('_')))
    recipe = reco.get_similar(recipe_ids)
    recipe = recipe.fillna(-1)
    for _, row in recipe.iterrows():
        ret.append(row.to_dict())
        row['조리순서'] = eval(row['조리순서'])
    return ret


@app.get('/recipe/{recipe_id}')
async def get_info(recipe_id: str):
    recipe_id = int(recipe_id)
    ret = reco.recipe.query('레시피일련번호 == @recipe_id').iloc[0]
    ret['조리순서'] = eval(ret['조리순서'])
    return ret.to_dict()


# 궁합 5개씩
@app.get('/combi/{recipe_id}')
async def get_combi(recipe_id: str):
    recipe_id = int(recipe_id)
    ret = combi.get_similar(recipe_id)
    return ret


# 설문 21개
@app.get('/first/')
async def get_random():
    recipes = reco.recipe.query('조회수 > 500').sample(21)
    recipes = recipes.fillna(-1)
    ret = []
    for _, recipe in recipes.iterrows():
        ret.append(recipe.to_dict())
    return ret


# 키워드 15개
@app.get('/keyword/')
async def get_keyword():
    selected = ['볶음', '일상', '무침', '밑반찬', '채소류', 
             '양파', '계란', '반찬', '밥', '간식', '고추장', 
             '밥/죽/떡', '두부', '오이', '김치', '감자']
    keyword2idx = {item : idx for idx, item in enumerate(selected)}
    return keyword2idx
    


#  키워드 레시피 20개
@app.get('/keyword/{keyword_recipe_ids}')
async def reco_recipe(keyword_recipe_ids: str):
    selected = ['볶음', '일상', '무침', '밑반찬', '채소류', 
             '양파', '계란', '반찬', '밥', '간식', '고추장', 
             '밥/죽/떡', '두부', '오이', '김치', '감자']
    idx2keyword = {idx : item for idx, item in enumerate(selected)}    
    ret = []
    split = keyword_recipe_ids.split('_')
    keyword_id = int(split[0])
    keyword = idx2keyword[keyword_id]
    recipe_ids = list(map(int, split[1:]))
    recipe = reco.get_similar(recipe_ids, keyword=keyword)
    recipe = recipe.fillna(-1)
    for _, row in recipe.iterrows():
        ret.append(row.to_dict())
        row['조리순서'] = eval(row['조리순서'])
    return ret




if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)