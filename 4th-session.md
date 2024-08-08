# 4주차. 4장. 다양한 분류 알고리즘(p.175-p.218)

## 4-1. 로지스틱 회귀(p.176)
- 생선을 담은 럭키백을 팔 것. 이 때 각 럭키백 별로 나올 수 있는 확률을 계산해본다.
### 럭키백의 확률(p.177)
- 럭키백에 들어갈 수 있는 생선은 7개.
  - 길이, 높이, 두께, 대각선길이, 무게를 사용할 것.
  - KNN에서는 주변 이웃 숫자를 세서 그걸로 확률을 구하면 되지 않을까?
  - 사이킷런의 KNN분류기도 동일한 방법으로 확률을 제공한다.
```python
# 인터넷에서 직접 csv를 읽어오고  head로 처음 5개 행 출력.
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv')
fish.head()
# dataframe은 pandas에서 제공하는 2차원 표 형식 데이터 구조이다. df는 numpy로 상호 변환이 쉽고 사이킷런과도 호환성 좋음.

print(pd.unique(fish['Species']))  # species열의 고유값 추출
# 이 species 열을 target으로 만들고 나머지 5개 열을 입력 데이터로 사용할 것.
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()  # 넘파이로 바꿔서 fish_input에 저장하기.
print(fish_input[:5])
fish_target = fish['Species'].to_numpy()  # 동일하게 타깃데이터도 넘파이로 만들기.

# 훈련셋과 테스트셋으로 나누기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state = 42)

# StandardScaler 클래스로 표준화 전처리
# 훈련셋  통계값으로 테스트셋을 변환해야 함.
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# KNeighborsClassifier 클래스 객체를 만들고 훈련셋으로 훈련한 다음 점수를 각각 확인해본다. k는 3으로 지정.
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, test_target)
print(kn.score(train_scaled, train_target))  # 결과: 0.8907
print(kn.score(test_scaled, test_target))  # 결과: 0.85
```
- 타깃데이터에 7개의 생선 종류가 들어가 있다. 2개 이상 클래스가 포함된 문제를 다중분류multi-class classification이라고 한다.
- 이진분류에서는 0과 1로 지정하여 타깃데이터를 만들었다.
- 다중분류도 숫자로 변환하여 분류할 수는 있지만, 사이킷런에서는 문자열 그대로 타깃으로 사용할 수 있다.
  - 단, 타깃값은 알파벳 순서로 매겨진다.
  - 순서는 classes_ 속성에 저장되어 있다.
```python
print(kn.classes_)  # 정렬된 타깃값 보기

# 테스트 셋의 처음 5개 샘플의 타깃값 예측해보기
print(kn.predict(test_scaled[:5]))  # 5개 순서대로 예측된 결과가 물고기 종류 이름으로 나온다.

# 이러한 예측을 하게 된 확률값 보기.
import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))  # 소수점 5자리에서 반올림. 기본적으로는 첫째 자리에서 반올림한다.

# 비율이 어떻게 나왔는지(2:1의 비율) 확인해보기 위해 4번째 샘플의 이웃 클래스가 무엇이 있는지 확인해본다.
# kneighbors() 메소드의 입력은 2차원 배열이어야 한다. 4번째 샘플 하나만 넣는 경우에는, 넘파이 배열의 슬라이스 연산자를 사용하면 2차원 배열이 되어서 들어간다.
distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])  # Roach, Perch, Perch
```
- 분류에서 성공했지만 3개의 이웃만 썼기에 확률은 0, 0.33, 0.67, 1 4가지 뿐일 것. 더 좋은 방법을 찾아야 한다.
### 로지스틱 회귀logistic regression(p.183)
- 로지스틱회귀logistic regression은 이름은 회귀이지만 분류이다. 선형회귀와 동일하게 선형 방정식을 학습한다.
- 다중회귀할 때 쓰는 방정식과 동일하다. z = a X (weight) + b X (length) + .... + f
- z는 어떤 값이든 들어갈 수 있지만, 확률로 표현하려면 0~1 사이의 값을 가져야 한다.
- 이 때 시그모이드 함수sigmoid function 또는 로지스틱 함수 logistic function을 사용하면 가능하다.
  - 시그모이드 함수는 phi = 1 / ( 1 + e^(-z) )
  - z가 -무한이면 0에 가까워지고, +무한이면 1에 가까워진다. z가 0이면 0.5가 된다. 절대 0~1 사이를 벗어나지 못한다.
```python
# 넘파이를 사용하여 로지스틱 그래프 그리기
import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.show()
```
- 사이킷런에는 로지스틱 회귀 모델인 LogisticRegression 클래스가 있다.
  - 먼저 이진분류부터 수행해본다. 0.5보다 크면 1(양성 클래스), 작으면 0(음성 클래스)로 판단. 정확히 0.5인 경우 라이브러리마다 다를 수 있지만 사이킷런은 음성으로 판단.
- 넘파이 배열은 True, False 값을 전달하여 행 선택 가능. 이를 불리언 인덱싱boolean indexing이라고 한다.
```python
# 배열에서 A와 C만 골라내려면 1번 3번 원소만 True라고 하는 배열을 전달하면 된다.
char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
print(char_arr[[True, False, True, False, False]])

# 비교연산자를 통해 도미Bream과 빙어Smelt의 행을 True로 만든다.
# 비트 OR 연산자 | 를 통해 골라낸다.
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]
```

## 4-2. 확률적 경사 하강법(p.199)
### 점진적인 학습(p.200)
### SGD Classifier(p.207)
### 에포크와 과대/과소적합(p.209)
