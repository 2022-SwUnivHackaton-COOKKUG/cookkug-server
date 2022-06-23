import re
import random

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from glob import glob
from kiwipiepy import Kiwi
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import matplotlib.font_manager as fm
from sklearn.metrics.pairwise import cosine_similarity
font = fm.FontProperties(fname='../data/BMJUA_ttf.ttf').get_name()
plt.rc('font', family=font)


class Recommender:
    def __init__(self):
        if '../data/crawling.df' in glob('../data/*'):
            self.recipe = joblib.load('../data/crawling.df')
        else:
            # self.recipe = pd.read_csv('../data/fast_recipe.csv', encoding='cp949')
            self.recipe = self.recipe[self.recipe['요리재료내용'].notna()].reset_index(drop=True)
            self.recipe = self.recipe[self.recipe['요리명'].notna()].reset_index(drop=True)
            self.recipe['전처리'] = self.recipe['요리재료내용'].apply(self._clear_ingreds)
            self.tokenize_cols()
            self.get_freq()
        
        if '../data/recipe2vec.dict' in glob('../data/*'):
            self.recipe2vec = joblib.load('../data/recipe2vec.dict')
        else:
            self.train()
        
        
    def _clear_ingreds(self, raw):
        patterns_1 = [
        ('\[[^[]*\]', '|'),
        ('\([^(]*\)', ''),
        ('또는|or|&|혹은', '|'),
        ('[톡*톡]|[.]|약간|조금|살짝|적당히|넉넉히|듬뿍|적당량|취향것|취향껏|\\u200b|g|\\ufeff|▶|~|넉넉하게|저염|많이|국물용|:|\\u200b', ''),
        ('토마토케첩|토마토케찹|케첩|캐첩', '케찹'),
        ('쌀뜬물', '쌀뜨물')
        ]
        patterns_2 = [
            ('[0-9].*', ''),
        ]

        for pat, change in patterns_1:
            raw = re.sub(pat, change, raw)

        raw = raw.split('|')

        for pat, change in patterns_2:
            raw = [re.sub(pat, change, piece) for piece in raw]
            raw = [piece.replace(' ', '') for piece in raw if piece != '']
            
        return raw
    
    
    def tokenize_cols(self):
        recipe = self.recipe.copy().fillna('')
        
        self.kiwi = Kiwi()
        for col_name in ['레시피제목', '요리명', '요리소개']:
            recipe[col_name] = list(self.kiwi.tokenize(recipe[col_name]))
            
        for idx, row in tqdm(recipe.iterrows(), total=len(recipe)):
            for col_name in ['레시피제목', '요리명', '요리소개']:
                tokens = row[col_name]
                tokens = [token.form for token in tokens if token.tag[0] == 'N']
                row['전처리'].extend(tokens)

            for col_name in ['요리방법별명', '요리상황별명', '요리재료별명', '요리종류별명']:
                if type(row[col_name]) != str:
                    continue
                if row[col_name] == '기타':
                    continue
                row['전처리'].append(row[col_name])
            random.shuffle(row)
        self.recipe['전처리'] = recipe['전처리']
    
    
    def get_freq(self):
        self.counter = Counter()
        for lst in self.recipe['전처리']:
            self.counter.update(lst)
    
    
    def plot_wordCloud(self):
        mask = np.array(Image.open("cloud.png"))
        wc = WordCloud(font_path='./BMJUA_ttf.ttf',background_color="white", max_font_size=60, mask=mask)
        cloud = wc.generate_from_frequencies(self.counter)

        fig, ax = plt.subplots(figsize=(16, 14))
        ax.set_title('재료별 빈도', size=20)
        ax.imshow(cloud)
        plt.axis('off')
        plt.show()
        
    
    def train(self):
        self.wv = Word2Vec(sentences=self.recipe['전처리'], vector_size=10, window=20, min_count=10, workers=4, sg=1)
        self.recipe2vec = { _id : np.mean([self.wv.wv[item] for item in items if item in self.wv.wv], axis=0) 
                        for _, (_id, items) in self.recipe[['레시피일련번호', '전처리']].iterrows() }
        
        self.recipe2vec = {k : v for k, v in self.recipe2vec.items() if v.shape}
        
        
    def plot_tsne(self, p=0.1):
        emb2 = TSNE(n_components=2).fit_transform(self.wv.wv.vectors)

        fig, ax = plt.subplots(figsize=(16, 14))
        for (x, y), item in tqdm(zip(emb2, self.wv.wv.index_to_key), total=len(emb2)):

            if np.random.uniform() < p:
                ax.scatter(x, y, alpha=0.3, color='lightseagreen')
                ax.text(x, y, s=item, fontsize=13)

        plt.axis('off')
        plt.show()
        
        
    def _calc_sim(self, recipe_id, recipe):
        emb = self.recipe2vec[recipe_id]
        sims = cosine_similarity(emb.reshape(-1, 10), np.array(list(self.recipe2vec.values())))
        recipe2sim = {name: sim for name, sim in zip(self.recipe2vec.keys(), sims.flatten())}
        recipe[f'{recipe_id}_유사도'] = recipe['레시피일련번호'].apply(lambda x: recipe2sim.get(x, None)).round(2)
        return recipe 


    def get_similar(self, recipe_ids, topn=20, keyword=None):
        recipe = self.recipe.copy()
        if keyword:
            recipe = recipe[recipe['전처리'].apply(lambda x: keyword in x)]

        for recipe_id in recipe_ids:
            recipe = self._calc_sim(recipe_id, recipe)
        recipe['유사도'] = recipe[[col for col in recipe.columns if col.endswith('유사도')]].max(axis=1).round(2)
        return recipe.query('레시피일련번호 not in @recipe_ids').sort_values('유사도', ascending=False).drop_duplicates('요리명')[:topn]