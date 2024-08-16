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
### 검증 세트(p.243)
### 교차 검증cross validation(p.245)
### 하이퍼파라미터 튜닝(p.248)

## 5-3. 트리의 앙상블(p.263)
### 정형 데이터와 비정형 데이터(p.264)
### 랜덤 포레스트Random Forest(p.265)
### 엑스트라 트리Extra Tree(p.269)
### 그레이디언트 부스팅Gradient boosting(p.271)
### 히스토그램 기반 그레이디언트 부스팅Histogram-based Gradient Boosting(p.273)
