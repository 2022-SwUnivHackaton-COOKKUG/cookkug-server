import joblib
import fasttext
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 제작파일은 preprocessing/궁합추천.ipynb참조
class Combination:
    def __init__(self):
        self.recipe = joblib.load('../data/crawling.df')
        self.recipe2combi = joblib.load('../data/recipe2combi.dict')
        self.recipe2vec = joblib.load('../data/recipe2vec.dict')
        self.fasttext = fasttext.load_model("../data/fasttext.bin")
        
        
    def _calc_sim(self, recipe_id, recipe):
        emb = self.recipe2combi[recipe_id]
        sims = cosine_similarity(emb.reshape(-1, 10), np.array(list(self.recipe2combi.values())))
        recipe2sim = {name: sim for name, sim in zip(self.recipe2combi.keys(), sims.flatten())}
        recipe[f'{recipe_id}_유사도'] = recipe['레시피일련번호'].apply(lambda x: recipe2sim.get(x, None)).round(2)
        return recipe 


    def get_similar(self, recipe_id):
        ret = []
        recipe = self.recipe.copy()
        recipe = recipe.fillna(-1)
        name = recipe.query('레시피일련번호 == @recipe_id')['요리명'].values[0]
        vec = self.recipe2vec[recipe_id]
        recos = self.fasttext.get_nearest_neighbors(name, 15)
        for _, reco in recos:
            if len(ret) == 3:
                break
            target_recipe_id = self.recipe[self.recipe.요리명.str.contains(reco)]['레시피일련번호'].values
            if len(target_recipe_id):
                target_recipe_id = target_recipe_id[0]
            else:
                continue
            target_vec = self.recipe2vec[target_recipe_id]
            sim = cosine_similarity(vec.reshape(-1, 10), target_vec.reshape(-1, 10)).flatten()[0]
            if sim < 0.9:
                dic = recipe.query('레시피일련번호 == @target_recipe_id').iloc[0].to_dict()
                dic['조리순서'] = eval(dic['조리순서'])
                ret.append(dic)
        return ret