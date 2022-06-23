# Backend
- `FastAPI`를 이용한 백엔드 구현
- `content-based` 추천 구현
- `preprocessed.df` - `crawling.df` 순서로 데이터 정제

# Recommendation(content-based)
### 1. 개인화추천 
- `recipe2vec.dict` 제작
- 레시피추천을 위해 레시피데이터 내의 `레시피제목`, `요리명`, `요리소개`, `재료` 등을 이용해 `word2vec` 레시피임베딩 제작
- `cosine similarity` 계산을 통한 레시피 추천

### 2. 궁합추천
- `recipe2combi.dict`, `fasttext.bin` 제작
- 궁합레시피 추천을 위해 `음식점 메뉴 데이터셋`을 이용해 `Fasttext`학습 후 궁합에 맞는 레시피 추천
```
fasttext 구함 -> 테이블에서 단어가 포함되는 요리명 구함 -> recipe2vec에서 유사도 구함 ->
유사도가 너무 높은 경우 제외 -> 다음 단어 확인.. -> recipe2vec 유사도가 낮은 경우 궁합 요리로 추가
```

## prototype 실행방법(prototype 폴더 내에서)
```
streamlit run app.py
```


## FastAPI 실행방법(fastapi 폴더 내에서)
```
python run app.py
```
![image](https://user-images.githubusercontent.com/57215124/175384495-c821f534-a791-45ff-9fcd-087642536c34.png)


## 사용데이터
- [만개의레시피](https://kadx.co.kr/product/detail/0c5ec800-4fc2-11eb-8b6e-e776ccea3964) - 레시피정보
- [지역별 음식점 메뉴데이터](https://www.data.go.kr/tcs/dss/selectDataSetList.do?dType=FILE&keyword=%EB%A9%94%EB%89%B4&detailKeyword=&publicDataPk=&recmSe=&detailText=&relatedKeyword=&commaNotInData=&commaAndData=&commaOrData=&must_not=&tabId=&dataSetCoreTf=&coreDataNm=&sort=&relRadio=&orgFullName=&orgFilter=&org=&orgSearch=&currentPage=3&perPage=10&brm=%EC%8B%9D%ED%92%88%EA%B1%B4%EA%B0%95&instt=&svcType=&kwrdArray=&extsn=CSV&coreDataNmArray=&pblonsipScopeCode=)
    - 강원도원주시, 강원도화천군, 광주동구, 부산, 서울, 전남, 전북전주

## 크롤링
- [만개의레시피](https://www.10000recipe.com/index.html) - 이미지, 조리순서
