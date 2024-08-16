# 5주차. 5장. 트리 알고리즘(p.219-p.284)
## 5-1. 결정 트리(p.220)
- 신상품으로 캔 와인 팔기로 결정
  - 레드와인인지 화이트와인인지 표시가 누락.
  - 알코올 도수, 당도, pH 값으로 찾아볼 것.
### 로지스틱 회귀로 와인 분류하기(p.221)
```python
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine-date')

# 첫 5개 샘플 보기
wine.head()
# class 열은 0이면 red wine, 1이면 white wine이라고 한다.
# 이진분류 문제이며, 화이트 와인이 양성 클래스이다.

wine.info()
# DF의 각 열 데이터 타입과 누락된 데이터가 있는지 확인할 때 유용한 메소드이다.
# index, column type, null 아닌 값의 개수, 메모리 사용량을 제공한다.
# verbose 매개변수의 기본값 True를 False로 바꾸면 각 열에 대한 정보를 출력하지 않는다.
# 총 6,497개의 샘플이 있고, 4개 열이 float 형태이고, Non-Null Count라고 출력된 것을 보니 누락 값은 없는 것 같다.

wine.describe()
# 이 메소드는 열마다 간략한 통계를 출력해줌.
# 평균, 표준편차, 최소, 1사분위수, 중간값, 3사분위수, 최댓값을 볼 수 있다.
# percentiles 매개변수로 백분위수 지정할 수 있다. 기본값은 [0.25, 0.5, 0.75] 이다.

# 도수, 당도, pH의 스케일이 다 다르다. 따라서 특성을 표준화할 필요 있음.
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()  # 3개 열을 넘파이 배열로 바꾸고, data라는 배열에 저장.
target = wine['class'].to_numpy()  # 마지막 열인 class는 넘파이 배열로 바꿔서 target 배열에 저장.

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
  data, target, test_size=0.2, random_state=42)
# 기본적인 테스트셋 크기는 25%이다. 여기서는 20%로 맞추기 위해 test_size 모수를 변경.

print(train_input.shape, test_input.shape)

# 훈련셋 전처리
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 로지스틱 회귀모델 훈련
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))  # 0.7808
print(lr.score(test_scaled, test_target))  # 0.7776
# 결과는 과소 적합인 것으로 보인다. 규제 매개 변수 C를 바꾸거나, solver 매개변수에서 다른 알고리즘 사용도 가능할 것. 또는 다항 특성을 만들어 추가 하는 방법도 있을 것.
```
- 로지스틱 회귀 모형으로 학습한 계수와 절편을 출력해서 보고서를 작성해본다.
  - 알코올 도수 0.5127, 당도 1.6733, pH -0.6876을 곱해서 더하고 절편인 1.8177을 마지막에 더해서
  - 0보다 크면 화이트 와인, 작으면 레드 와인이며
  - 77% 정도의 정확도를 보인다.
  - 라는 식의 보고서가 가능. 근데 우리는 왜 모델이 저런 계수를 학습했는지 직관적으로 이해하기 어렵다. 대부분 머신러닝 모델이 이렇듯 설명하기 어렵다.
- 만약 info 메소드로 봤을 때 누락된 값이 있다면?
  - 그 데이터 자체를 버리거나
  - 평균값으로 채워서 사용할 수 있다.
  - 어떤 방식이 최선인지는 미리 알 수 없다. 둘 다 시도해보기.
  - 여기서도 항상 훈련셋 통계값으로 테스트셋 변환해야 함.
  - 즉, 훈련셋 평균값으로 테스트셋의 누락값을 채워야 한다는 것.
### 결정 트리(p.226)
- 결정 트리Decision Tree 모델을 사용하면 이유를 설명하기 쉽다.
  - 스무고개와 같다. 하나씩 정답을 맞춰가는 것.
  - 데이터를 잘 나눌 질문을 찾는다면 분류 정확도를 높여갈 수 있다.
- DT 모델에서 random_state를 정하는 이유는?
  - 사이킷런 DT 알고리즘은 노드에서 최적의 분할을 찾기 전에 특성 순서를 섞는다. 따라서 약간의 무작위성이 주입되는데 실행 때마다 점수가 조금씩 달라질 수 있음.
  - 여기서는 독자와 같은 결과 출력 위해서 지정하지만, 실전에서는 필요하지 않다.
```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))  # 0.9969
print(dt.score(test_scaled, test_target))  # 0.8592
# 결과는 과대적합같다.

# DT를 이해하기 쉽게 그림으로 표현해준다.
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10, 7))
plot_tree(dt)
plt.show()
```
- DT의 나무는 위에서부터 자란다.
  - 맨 위 노드는 root node 뿌리노드, 맨 아래에 달린 것을 leaf node 리프 노드 라고 한다.
  - node는 데이터 특성에 대한 테스트를 표현한다. 가지branch는 테스트 결과인 T/F를 나타내며 일반적으로 하나의 노드는 2개의 가지를 가진다.
```python
# max_depth 매개변수로 뿌리에서 몇 개의 노드를 그릴지 정할 수 있다.
# filled 매개변수로 클래스에 맞게 노드를 칠할 수 있다.
# features_names 매개변수는 특성 이름을 전달한다.
plt.figure(figsize=(10, 7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# DecisionTreeClassifier
  # criterion 매개변수는 불순도 지정. 기본은 gini이고, 'entropy'선택하여 바꿀 수 있다.
  # splitter 매개변수는 노드 분할 전략을 선택. 기본은 'best'로 정보이득이 최대가 되도록, 'random'으로 설정하면 임의로 노드를 분할한다.
  # max_depth는 트리가 성장할 최대 깊이 설정. 기본값은 None으로 리프노드가 순수해지거나, min_samples_split 보다 샘플 개수가 적을 때까지 성장한다.
  # min_samples_split은 노드를 나누기 위한 최소 샘플 개수이다. 기본값은 2이다.
  # max_features 매개변수는 최적 분할을 위해 탐색할 특성 수를 지정한다. 기본값은 None으로 모든 특성을 전부 사용.

# plot_tree()
  # max_depth 매개변수로 트리의 깊이 지정. 기본은 None으로 모든 노드 출력.
  # features_names 매개변수로 특성의 이름을 지정할 수 있다.
  # filled 매개변수를 True로 지정하면 타깃값에 따라 노드를 색칠해준다.
```
- 이 그림이 담은 정보는
  - 첫 줄: 테스트 조건 suger
    - 당도가 -0.239 이하인지 질문한다. 이하라면 왼쪽 가지 Yes로 간다. 아니면 오른쪽 가지로 간다.
  - 두 번째 줄: 불선도 gini
  - 세 번째 줄: 총 샘플 수 samples
    - 해당 노드로 온 총 샘플의 수
  - 네 번째 줄: value 클래스별 샘플 수
    - \[음성 클래스 수, 양성 클래스 수\]
- filled=True를 지정하면 클래스마다 색깔 부여하고 어떤 클래스의 비율이 높아지면 점점 진한 색으로 표시해준다.
- DT에서 예측하는 법은 리프 노드에서 가장 많은 클래스가 예측된 클래스가 된다.
- gini impurity 지니 불순도
  - DecisionTreeClassifier 클래스의 criereion 매개변수 기본값은 gini이다.
    - 이 매개변수 용도는 노드에서 데이터 분할할 기준을 정하는 것.
  - 지니 불순도는 클래스의 비율을 제곱해서 더한 다음 1에서 빼면 된다.
    - gini = 1 - (음성 클래스 비율^2 + 양성 클래스 비율^2)
    - 다중 클래스도 식만 길어지지 동일한 방식.
  - 노드에 하나의 클래스만 있다면 0으로 가장 작고, 두 노드가 정확히 절반씩 있다면 0.5로 최악이 된다.
  - DT 모델은 부모노드와 자식노드의 불순도 차이가 가능한 크도록 트리를 성장시킨다.
  - 부모와 자식 노드 사이의 불순도 차이는 자식노드 불순도를 모두 더하여 부모노드에서 빼면 된다.
  - 부모와 자식 노드 사이의 불순도 차이를 정보 이득 information gain 이라고 한다.
- 또 다른 불순도 기준이 있다.
  - criterion='entropy'
  - 엔트로피 불순도도 노드의 클래스 비율을 사용한다. 이 때 gini 처럼 제곱을 사용하는 것이 아니라 밑이 2인 로그를 사용한다.
  - 보통 gini와 엔트로피의 불순도 차이가 만들어내는 결과의 차이는 그리 크지 않다.
- 즉, 불순도를 사용해 정보이득이 최대한 크도록 하고, 이 때 정보이득이 커지는 방법은 노드를 최대한 더 순수하게 나눌 때이다.
- 마지막에 도달한 노드의 클래스 비율을 보고 예측을 만드는 것.

- 가지치기
  -  DT도 가지치기를 해야 한다. 그렇지 않으면 무작정 끝까지 나무가 자라나기 때문. 과적합되는 문제 가능.
  -  가장 간단한 방법은 트리 최대 깊이를 지정하는 것.
  -  실습은 비교적 해석이 쉽지만, 실전에서는 많은 특성을 사용하고 트리 깊이도 깊어진다. 따라서 생각만큼 해석이 쉽지 않을 수도.
  - DT의 장점은 특성값의 스케일이 아무런 영향을 미치지 않아서 표준화 전처리를 할 필요가 없다.
    - 특성값을 표준화하지 않아서 더 이해하기 쉽다.
  - DT는 어떤 특성이 가장 유용한지 특성 중요도도 계산해준다.
    - 특성 중요도는 각 노드 정보이득과 전체 샘플에 대한 비율을 곱한 후 특성별로 더하여 계산한다.
    - 이를 통해 어떤 특성을 고를지도 활용 가능.
```python
# 뿌리노드 밑으로 3개까지 가지.
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))  # 0.8454
print(dt.score(test_scaled, test_target))  # 0.8415

# 그림 그려보기
plt.figure(figsize = (20, 15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# 특성 중요도 보기
print(dt.feature_importances_)  # 0.1234 0.8686 0.0079 당도가 가장 높다. 셋을 더하면 1이 됨.
```

- 머신러닝 모델을 블랙박스 같다고들 한다. 왜 그런지 설명하기가 어렵기 때문.
- 그에 반해 DT는 비전문가에게도 비교적 설명이 쉬움.
- 게다가 DT는 많은 앙상블 학습 알고리즘의 기반이 된다.
- 앙상블 학습은 신경망과 함께 가장 높은 성능을 내기 때문에 인기가 높은 알고리즘이다.
## 5-2. 교차 검증과 그리드 서치(p.242)
- max_depth를 3 말고 이런 저런 값으로 다르게 하면?
  - 성능이 달라진다. 그냥 모델 많이 만들어서 평가하면 그 테스트셋에만 잘 맞는 값이 나오는 것 아닌가?
  - 테스트셋으로만 검증하면 거기에 맞는 모델만 남게 될 것.
  - 테스트셋은 모델 만들고 딱 한 번만 마지막에 사용하는 것이 좋다.
  - 그렇다면 어떻게 해야 하나?
### 검증 세트(p.243)
- 검증 세트 validation set
  - 단순하지만 실제로도 많이 쓰인다.
  - 20%는 검증 세트로, 20%는 테스트셋으로 둔다.
  - 보통은 20~30% 정도를 떼어놓지만, 문제에 따라 훈련 데이터가 아주 많다면 단 몇 %만 떼어놓아도 전체 데이터를 대표할 수 있다.
  - 검증세트와 훈련세트로 가장 좋은 모델을 고르고, 마지막 테스트셋으로 최종 점수를 평가하는 것이다. 그러면 실전에서 테스트셋과 비슷한 점수를 기대할 수 있게 된다.
```python
import pandas as pd
wine = pd.read_csv("https://bit.ly/wine-date')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

# 훈련셋은 sub로 검증셋은 val로 만든다. 훈련셋의 20%를 검증셋으로 만든다.
sub_input, val_input, sub_target, var_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
# train_test_split() 함수를 2번 적용해서 훈련셋과 검증셋으로 나눠준 것이다.

# 각각의 크기를 확인해본다.
print(sub_input.shape, val_input.shape)
# (4157, 3) (1040, 3)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)
print(dt.score(sub_input, sub_target))  # 0.9971
print(dt.score(val_input, val_target))  # 0.8644

# 훈련셋에 과대적합 되어 있어서 더 좋은 모델을 찾아야 한다.
```
### 교차 검증cross validation(p.245)
- 검증세트를 만들기 위해 훈련셋이 줄었다. 보통 많은 데이터를 훈련에 사용할수록 좋은 모델이다.
- 그렇다고 검증셋이 너무 작으면 점수가 불안정할 것 같다.
- 이럴 때 교차 검증을 사용.
  - 검증셋을 떼어 내어 평가하는 과정을 여러 번 반복한다.
  - 3-폴드 교차 검증
    - 훈련셋을 3부분으로 나눠서 교차 검증을 수행하는 것. 통칭 k-폴드 교차 검증 k-fold cross validation 이라 한다. k-겹 교차검증 이라고도 한다.
  - 이렇게 5-폴드, 10폴드 등을 사용하면 80~90%까지 훈련에 사용할 수 있다. 검증셋이 줄어들지만, 각 폴드에서 계산한 검증 점수를 평균하기 때문에
  - 안정된 점수라고 생각할 수 있다.
  - cross_validate() 교차 검증 함수. 평가할 모델 객체를 첫 번째 매개변수로 전달한다.
  - 그 다음 직접 검증셋을 떼어내지 않고 훈련셋 전체를 cross_validate() 함수에 전달한다.
  - 이 함수의 전신인 cross_val_score() 함수도 있다. 이 함수는 cross_validate() 함수 결과 중에서 test_score 값만 반환한다.
```python
from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target)
print(scores)
```
  - 이 함수는 fit_time, score_time, test_score 키를 가진 딕셔너리를 반환한다. 처음 2개 키는 각각 모델 훈련하는 시간과 검증하는 시간을 의미한다.
  - 각 키마다 5개의 숫자가 담겨져 있다.
  - cross_validate() 함수는 기본적으로 5-폴드 교차검증 수행.
  - cv 매개변수에서 폴드 수를 바꿀 수 있다.
  - 교차검증 최종점수는 test_score 키의 5개 점수 평균하여 얻을 수 있다. 이름은 test_score이지만 검증 폴드의 점수이다.
```python
import numpy as np
print(np.mean(scores['test_score']))   # 0.8553
```
  - 교차검증을 통해 입력한 모델에서 얻을 수 있는 최상의 검증 점수를 가늠해볼 수 있다.
  - 주의할 점은 cross_validate()는 훈련셋을 섞어서 폴드를 나누지 않는다. 만약 섞으려면 분할기splitter를 지정해야 한다.
  - 분할기는 폴드를 어떻게 나눌지 결정해준다.
  - 회귀모델에서는 KFold 분할기를, 분류모델일 경우는 타깃 클래스를 골고루 나누기 위해 StratifiedKFold를 기본적으로 사용한다.
```python
from sklearn.model_selection import StratifiedKFold
scores = cross_validate(dt, train_input, train_target, cv = StratifiedKFold())
print(np.mean(scores['test_score']))  # 0.8553
# KFold 클래스도 동일하게 사용할 수 있다.
```
  - 이제 DT의 매개변수 값을 바꿔가며 가장 좋은 성능이 나오는 모델을 찾아봐야 한다.
  - 테스트셋 사용하지 않고 교차검증을 통해서 고르는 것이다.
### 하이퍼파라미터 튜닝(p.248)
- 머신러닝 모델이 학습하는 파라미터를 모델 파라미터라 했었다.
- 모델이 학습할 수 없어서 사용자가 지정해야만 하는 파라미터를 하이퍼파라미터라 한다.(사용자 지정 파라미터)
  - 머신러닝 라이브러리 사용할 때 이런 것들은 모두 클래스나 메소드의 매개변수로 표현된다.
  - 이런 하이퍼 파라미터를 튜닝하는 법은?
    - 먼저 라이브러리 기본값을 사용한다.
    - 검증셋 점수나 교차검증을 통해서 매개변수를 조금씩 바꿔본다.
    - 모델마다 1~2개에서 많게는 5~6개의 매개변수를 제공한다. 이 매개변수를 바꿔가면서 모델을 훈련하고 교차검증을 수행해야 한다.
    - 사람 개입 없이 튜닝을 자동으로 수행하는 기술을 AutoML 이라고 부른다.
  - DT에서 최적의 max_depth 를 찾았다고 가정하자.
    - 그렇다면 이 값을 최적 값으로 고정하고 min_samples_split을 바꿔가며 최적 값을 찾으면 될까?
    - 불행히도 max_depth 의 최적값은 min_samples_split 매개변수의 값이 바뀌면 함께 달라진다.
    - 즉, 두 매개변수를 동시에 바꿔가며 최적의 값을 찾아야 한다.
    - 게다가 매개변수의 양까지 늘어나면? 파이썬의 for 반복문으로 직접 구현할 수도 있겠지만, 이미 만들어진 도구 사용하는 것이 편리할 것.
    - 그리드서치 Grid Search를 사용하면 된다.
      - 이것은 하이퍼파라미터 탐색과 교차검증을 한 번에 수행한다. 별도로 cross_validate() 함수를 호출할 필요가 없다.
      - min_impurity_decrease 매개변수의 최적값을 찾아본다.
        - 이를 위해 GridSearchCV 클래스를 임포트 하고 탐색할 매개변수와 탐색할 값의 리스트를 딕셔너리로 만든다.
```python
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
# 0.0001부터 5개의 값을 시도해본다.

# 탐색 대상 모델과 params 변수를 전달하여 서치 객체를 만든다.
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)

# 그 다음은 일반 모델 훈련하는 것처럼 gs 객체에 fit() 메소드를 호출한다. 이 메소드 호출하면 그리드 서치 객체는 DT의 min_impurity_decrease 값을 바꿔가며 총 5회 시행.
# GridSearchCV의 cv 매개변수는 기본값이 5. 따라서 min_impurity_decrease 값마다 5-폴드 교차검증을 수행한다.
# 결국 5X5의 25개의 훈련을 한다.
# 많은 훈련을 하기에 n_jobs 매개변수로 병렬실행에 사용할 CPU 코어 수 지정하는 것이 좋다. 기본값은 1.
# -1로 하면 시스템의 모든 코어를 사용한다.

gs.fit(train_input, train_target)
# 교차검증에서 최적의 하이퍼파라미터를 찾으면 전체 훈련셋으로 모델을 다시 만들어야 한다고 했었다.
# 아주 편리하게도 사이킷런 그리드 서치는 훈련이 끝나면 25개 모델 중에 검증점수 가장 높은 모델의 매개변수 좋바으로
# 전체 훈련셋에서 자동으로 다시 모델을 훈련한다.

# 이 모델은 gs 객체의 best_estimator_ 속성에 저장되어 있다. 이 모델을 DT처럼 사용하면 된다
dt = gs.best_estimator_
print(dt.score(train_input, train_target))  # 0.9615

# 그리드서치로 찾은 최적의 매개변수는 best_params_ 에 저장.
print(gs.best_params_)

# 수행한 교차검증의 평균 점수는 cv_results_ 속성의 mean_test_score 키에 저장되어 있다.
print(gs.cv_results_['mean_test_score'])
#  이 결과 첫번째 값이 가장 큰 것 같다. 0.0001에 해당하는 부분.
# 수동으로 고르는 것보다 넘파이 argmax() 함수를 사용해서 가장 큰 값의 인덱스를 추출할 수 있다.
# 그 다음 이 인덱스를 이용해 params 키에 저장된 매개변수를 출력할 수 있다.
# 이 값이 최상의 검증 점수를 만든 매개변수 조합이다.

best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])
```
- 이 과정을 정리하면
  - 1 탐색할 매개변수를 지정
  - 2 훈련셋에서 그리드 서치 수행하여 최상의 평균검증점수 나오는 매개변수 조합 찾음. 이 조합은 그리드서치 객체에 저장되어 있다.
  - 3 그리드서치는 최상의 매개변수에서 (교차 검증에 사용한 훈련셋이 아닌) 전체 훈련셋을 사용해 최종 모델 훈련한다. 이 모델도 그리드 서치 객체에 저장된다.
```python
# 조금 더 복잡한 매개변수 조합을 탐색해본다.
params = {'min_impurity_decrease': np.arrange(0.0001, 0.001, 0.0001),  # 노드 분한을 위한 불순도 감소 최소량 지정
  'max_depth': range(5, 20, 1),  # 트리의 깊이 제한
  'min_samples_split': range(2, 100, 10)  # 노드 나누기 위한 최소 샘플 수
  }
# np.arrange 함수는 첫 번째 매개변수에서 시작해서 두번째 매개변수에 도달할 때까지 세번째 매개변수만큼 계속 더하는 배열을 만든다.
# 0.0001부터 0.0001씩 더해서 0.001까지. 0.001은 포함하지 않으므로 0.0009까지 총 9개의 원소를 가진 배열이다.
# 파이썬의 range() 함수도 비슷하다. 하지만 정수만 사용 가능. 5에서 20까지 1씩 증가해서 15개의 값을 만든다.
# 2에서 100까지 10씩 증가해서 10개 값을 만든다.
# 따라서 교차검증 횟수는 9X15X10으로 1350개이다. 5폴드 교차이므로 총 모델 수는 6750개이다.
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)

# 최상의 매개변수 조합 확인하기
print(gs.best_params_)  # 14, 0.0004, 12
# 최상의 교차검증점수 확인하기
print(np.max(gs.cv_results_['mean_test_score']))  0.8683
```
  - 근데 앞에서 0.0001씩 간격을 정하는 방법은 임의로 선택한 간격인데, 더 좁거나 넓은 간격으로 시도할 수는 없는가?
- 랜덤 서치 Random Search
  - 매개변수 값이 수치일 때 범위나 간격 지정이 어려울 수 있다. 또 너무 많은 매개변수 조건으로 그리드서치 수행 시간이 오래 걸릴 수 있다.
  - 
## 5-3. 트리의 앙상블(p.263)
### 정형 데이터와 비정형 데이터(p.264)
### 랜덤 포레스트Random Forest(p.265)
### 엑스트라 트리Extra Tree(p.269)
### 그레이디언트 부스팅Gradient boosting(p.271)
### 히스토그램 기반 그레이디언트 부스팅Histogram-based Gradient Boosting(p.273)
