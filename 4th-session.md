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
# 도미와 빙어에만 True가 된 bream_smelt_indexes를 train_scaled와 train_target에 불리언 인덱싱을 적용하여 도미, 빙어 데이터만 골라낸 것이다.

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

# 처음 5개 샘플 예측
print(lr.predict(train_bream_smelt[:5]))
# 예측한 확률도 출력
print(lr.predict_proba(train_bream_smelt[:5]))
# 그 결과 샘플마다 두 개의 확률이 출력된다. 첫 열은 0에 대한 확률, 두 번째 열이 1에 대한 확률.
# 알파벳 순으로 정렬되기 때문에 확인 가능.
print(lr.classes_)  # 결과: bream smelt
# 따라서 smelt가 양성클래스이다. 5개 샘플 중 2번째만 smelt일 확률이 높다.
# 만약 bream도미를 양성클래스로 하고 싶다면? Bream 타깃값을 1로 만들고 나머지를 0으로 만들면 된다.

# 로지스틱 회귀가 학습한 계수를 확인해본다.
print(lr.coef_, lr.intercept_)
# 선형회귀와 비슷해보인다. z값도 계산해본다. 처음 5개 샘플에 대한 z값.
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)
# 여기서 나온 결과를 시그모이드 함수에 통과시키면 확률을 얻을 수 있다.
# 다행히 scipy 파이썬의 사이파이 라이브러리에 이 함수가 있다. expit(). np.exp() 함수 사용해 분수 계산하는 것보다 훨씬 편리하고 안전하다.

from scipy.special import expit
print(expit(decisions))
# 여기서 나온 결괏값은 predict_proba() 매소드 출력 두 번째 열 값과 동일하다. 즉, decision_function() 메소드는 양성클래스에 대한 z값을 반환하는 것.
# 이제 이 경험을 바탕으로 7개 생선 분류하는 다중분류 문제를 시도해볼 것.
```
- 이진분류와 다중분류의 차이점.
- LogisticRegression 클래스는 기본적으로 반복적인 알고리즘 사용.
  - max_iter 매개변수에서 반복횟수 지정하고 기본값은 100이다.
  - 여기에서 준비한 데이터셋으로 훈련하면 반복횟수가 부족하다는 경고가 발생하여 횟수를 1000으로 늘린다.
- LogisticRegression은 릿지 회귀 같이 계수의 제곱을 규제한다. 이런 규제를 L2 규제라고도 부른다.
  - 릿지 회귀는 alpha 매개변수로 규제 양을 조절했다. alpha가 커질수록 규제가 커졌다.
  - LogisticRegression에서는 매개변수 C가 규제를 조절. 작을수록 규제가 커진다. 기본값은 1. 여기서는 규제 완화를 위해 20으로 늘리겠다.
```python
# 7개 생선 데이터 모두 들어있는 train_scaled, train_target 사용.
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target)  # 0.9328
print(lr.score(test_scaled, test_target)  # 0.925

# 첫 5개에 대한 예측 출력해본다.
print(lr.predict(test_scaled[:5]))  # perch smelt pike roach perch

# 5개에 대한 예측 확률 출력. 소수점 4째자리 반올림.
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))
# 5개 샘플이라 행이 5개 예측되고, 7개의 종 구분이라 7개의 열이 출력되었다.
# 각 열이 무슨 종을 뜻하는지 확인해본다.
print(lr.classes_)

# 선형방정식도 확인해본다.
print(lr.coef_.shape, lr.intercept_.shape)  # (7, 5) (7, )
```
- 5개 특성 사용하므로 coef_의 열은 5이다. 계수와 절편의 행은 모두 7이다.
  - z를 7개나 계산한다는 의미이다.
  - 다중분류는 클래스마다 z값을 하나씩 계산한다.
  - 가장 높은 z를 출력하는 클래스가 예측한 클래스가 된다. 확률은 이진분류에서는 이 값을 시그모이드 함수로 변환한 것인데, 다중분류에서는 소프트맥스softmax 함수 사용하여 변환한다.
  - 시그모이드 함수는 하나의 선형방정식 출력값을 0~1 사이로 압축하는 것이지만, 소프트맥스 함수는 여러 개의 선형방정식 출력값을 0~1로 압축하고 이들 합을 1로 만드는 것.
  - 이를 위해 지수함수 사용하기 때문에 이 함수를 정규화된 지수함수라고 부르기도 한다.

- 소프트맥스 계산법
  - 7개의 z를 이용하여 지수함수를 구한다.
  - e_sum = e^z1 + e^z2 + ... + e^z7
  - 이들을 각각 e_sum으로 나눠서 총합이 1이 되도록.
  - s1 = e^z1 / e_sum ...
  - 시그모이드 함수와 소프트맥스 함수는 나중에 신경망 배울 때 또 다시 등장하므로 여기서 잘 익혀두어야 함.
```python
# decision_fuction() 메소드를 사용한 이진분류처럼, z1~z7까지 값을 구하고 소프트맥스로 확률을 구해본다.
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))
# scipy에서 제공하는 소프트맥스 함수를 사용하낟.
from scipy.special import softmax
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))
# axis 매개변수는 소프트맥스 계산 축을 지정한다. 여기선 1로 지정하여 각 행(각 샘플)에 대해 계산한 것.
# 지정하지 않으면 배열 전체에 대한 소프트맥스를 게산한다.
```
- LogisticRegression은 선형분류알고리즘은 로지스틱 회귀를 위한 클래스이다.
  - solver 매개변수에서 사용할 알고리즘 선택 가능.
  - 기본값은 lbfgs이다.
  - 사이킷런 0.17 버전의 'sag'는 확률적 평균 경사 하강법 알고리즘으로 특성과 샘플 수 많을 때 성능 좋다.
  - 0.19 버전에는 sag의 개선 버전인 saga가 추가되었다.
  - penalty 매개변수에서 L2규제(릿지) L1규제(라쏘) 방식 선택 가능. 기본은 L2를 의미하는 l2이다.
  - C 매개변수로 규제 강도 제어. 작을수록 규제가 강하고 기본값은 1이다.
 
## 4-2. 확률적 경사 하강법(p.199)
- 매주 7가지 생선 중 무작위로 골라 훈련데이터를 제공하고 있지만, 공급처가 너무 많아 샘플 고르기가 쉽지 않다.
- 추가되는 수산물 샘플은 가지고 있지도 않다.
### 점진적인 학습(p.200)
- 훈련 데이터가 한꺼번에 오는 것이 아니라 조금씩 전달되게 되었다.
    - 기존 데이터에 매번 추가하며 훈련하면? 괜찮은 생각이지만, 시간 지날수록 데이터가 늘어난다.
    - 또 다른 방법은 추가될 때마다 이전 데이터 버려서 훈련 데이터셋 크기를 일정하게 유지하는 것.
    - 하지만 버리는 데이터 중에 중요한 생선 데이터가 포함되면 큰일이다.
  - 위 방법은 이전 훈련 모델을 버리고 새롭게 훈련하는 방식이다. 모델은 유지한채 새로운 데이터만 추가로 훈련하는 방식은 없나?
  - 이런 방식을 점진적 학습, 온라인 학습이라고 한다. 대표적 알고리즘은 확률적 경사 하강법Stochastic Gradient Descent이다.
    - 확률적이라는 말은 '무작위하게', '랜덤하게'라는 기술적 표현이다. '경사'는 기울기다. '하강법'은 내려가는 방법이다.
    - 산에서 내려올 때 가장 빠른 길은 경사가 가파른 길이다. 경사하강법은 가장 가파른 경사를 따라 원하는 지점에 도달하는 것이 목표이다.
    - 하지만 다리가 너무 길어 한 걸음이 너무 크면 경사 맨 밑으로 도달하지 않고 다시 V자로 올라간 위쪽에 도착할 수 있다.
    - 가장 가파른 길을 찾으면서 동시에 조금씩 내려오는 것이 중요하다.
  - 확률적이라는 말
    - 훈련세트를 이용해 가장 가파른 길을 찾게 된다. 그런데 전체 샘플이 아닌 딱 하나의 샘플을 랜덤하게 골라서 찾는 것.
    - 이것이 확률적 경사 하강법.
- 훈련 세트에서 랜덤하게 하나의 샘플을 선택해 가파른 경사를 조금 내려간다. 그 다음 훈련 셋에서 랜덤하게 또 다른 샘플 골라서 내려간다.
  - 이런 식으로 모든 샘플 사용할 때까지 반복한다.
  - 이렇게 해도 산을 다 내려오지 못했다면? 다시 처음부터 하면 된다.
  - 이 한 과정을 에포크 epoch라고 한다. 일반적으로 수십 수백번 이상 수행하게 된다.
- 무작위 선택이 무책임해 보여도 꽤 잘 작동한다. 그래도 걱정이 되면 1개 말고 몇 개 씩 선택해서 내려가는 방법도 가능하다.
  - 이를 미니배치 경사 하강법minibatch gradient descent라고 한다.
- 극단적으로는 한 번에 모든 샘플을 사용할 수도 있다. 이를 배치 경사 하강법batch gradient descent라고 한다.
  - 전체 데이터 사용하기 때문에 가장 안정적일수는 있으나 데이터가 너무 많아 컴퓨터 자원을 많이 사용하게 된다.
- 신경망 알고리즘은 많은 데이터를 사용하면서 모델도 복잡하기 때문에 수학적 방법을 사용하기 힘들어서 확률적 경사 하강법이나 미니배치 경사 하강법을 사용한다.

- 손실 함수loss function
    - 머신러닝 알고리즘이 얼마나 엉터리인지 측정하는 기준이다.
    - 당연히 작을수록 좋겠지만, 어떤 값이 최소인지는 모른다.
    - 가능한 많이 찾아보고 만족할만한 수준에서 인정해야 한다.
    - 우리가 다루는 많은 문제에서 손실함수는 이미 정의되어 있다.
    - 손실함수의 다른 말은 비용함수cost function. 엄밀히 말하면, 손실함수는 샘플 하나에 대한 손실을 정의하고, 비용함수는 훈련셋 모든 샘플에 대한 손실함수 합이다.
  - 분류에서 손실은 확실하다. 못 맞추는 것. 정확도를 손실함수로 사용할 수 있나?
    - 정확도에는 단점이 있다. 샘플이 4개라면, 가능한 건 0, 0.25, 0.5, 0.75, 1의 다섯 가지 값만 가질 수 있다. 아주 조금씩 내려와야 하는데 이렇게 듬성듬성한 산이라면 내려올 수 없다.
    - 산의 경사면은 연속적이어야 한다.(정확히는 미분 가능해야 한다는 것)
    - 로지스틱 회귀에서 예측하는 확률은 0~1 사이의 연속적인 값이었다.
  - 로지스틱 손실 함수
    - (실제, 예측, 확률) 일 때 각각 (1, 1, 0.9), (1, 0, 0.3), (0, 0, 0.2), (0, 1, 0.8) 이라 확률을 가정한다.
      - 첫 샘플은 0.9에 맞췄으니 -0.9로 한다.
      - 두 번째는 실제는 1인데 0.3으로 낮은 값이다. 음수로 해서 -0.3으로 바꾼다.
      - 세 번째는 0.2와 정답인 0을 곱하면 0이 될테니, 0에 대한 예측값이 아닌 1에 대한 예측값인 0.8로 변환해서 -0.8을 사용한다.
      - 네 번째도 마찬가지로 0.2로 변환해서 -0.2로 바꾼다.
    - 실제와 맞은 1과 3은 -0.9와 -0.8로 손실이 낮고, 못 맞춘 2와 4는 손실이 높다.
    - 이런 방식으로 계산하면 연속적인 손실 함수 얻을 수 있다.
    - 여기에 예측 확률에 로그함수를 적용하면 더 좋다. 확률의 범위는 0~1 사이이고, 로그는 이 범위에서 음수가 되므로,
    - 최종 손실 값은 양수로 바뀌기 때문이다. 양수면 이해하기 더 쉬울 것.
    - 또한 로그함수는 0에 가까울수록 아주 큰 음수가 되기 때문에 손실을 아주 크게 만들어 모델에 큰 영향을 미치게 할 수 있다.
    - 이 손실함수를 로지스틱 손실함수 또는 이진 크로스엔트로피 손실함수 라고도 부른다.(이 손실 함수를 사용하면 로지스틱 회귀 모델이 만들어지기 때문)
    - 다중분류에서 사용하는 손실함수는 크로스엔트로피 손실함수라고 한다.
    - 손실함수들은 거의 다 만들어져 있어서 우리가 개발하는 일은 잘 없다.
    - 분류 말고 회귀에서는 평균 절댓값 오차MAE나 평균 제곱 오차 MSE를 많이 사용한다.
### SGD Classifier(p.207)
```python
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv')
# 데이터를 불러와서 species열 제외한 5개 열은 입력 데이터로 사용. 수산물 종에 대한 열은  타깃 데이터이다.
fish_input = fish[['Weight', 'Lengh', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

# 테스트셋 훈련셋 나누기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, randomstate=42)

# 전처리 하기
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 확률적 경사 하강법 제공하는 분류용 클래스
from sklearn.linear_model import SGDClassifier

# 객체 만들 때 2개의 매개변수를 지정한다. loss는 손실함수 종류 지정. log라고 하여 로지스틱 손실함수 지정.
# max_iter는 수행할 에포크 수 지정. 10으로 지정하여 10회 반복.
sc = SGDClassifier(loss='log', max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))  # 0.7731
print(sc.score(test_scaled, test_target))  #0.775
# 정확도가 낮은 것을 보니 반복 횟수가 부족한 것 같다.
# ConvergenceWarning 경고는 모델이 충분히 수렴하지 않았다는 경고. max_iter 값을 늘려주는 것이 좋다. 오류가 아닌 경고이다.

# 확률적 경사 하강법은 점진적 학습이 가능하다. 객체를 새로 만들지 않고 추가로 더 훈련해볼 것.
# 이어서 훈련할 때는 partial_fit() 메소드 사용. fit()과 사용법이 같지만 호출할 때마다 1에포크 씩 이어서 훈련 가능.
sc.partial_fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))  # 0.815
print(sc.score(test_scaled, test_target))  # 0.825
# 에포크가 한 번 더 진행하니 정확도가 올라갔다. 얼마나 해야 하나? 기준이 필요할 것.
# SGDClassifier는 미니배치 경사 하강법, 배치 경사하강법을 제공하지 않는다.
```
### 에포크와 과대/과소적합(p.209)
- 에포크 횟수에 따라 과소 또는 과대 적합 가능하다.
    - 에포크 횟수가 적으면 덜 학습한다. 충분히 많으면 완전히 학습하여 훈련셋에 딱 맞음.
    - 즉, 적은 에포크는 과소 적합 가능성, 많은 에포크는 과대적합 가능성.
    - 에포크가 커질수록 훈련셋 점수는 꾸준히 증가하지만, 테스트셋 점수는 어느 순간 감소 시작. 이 지점이 과대적합 시작 지점.
    - 이 시작 전에 훈련 멈추는 것을 조기종료early stopping이라 한다.
    - 어디서 꺾이는지 보는 그래프를 그려볼 것.
```python
# fit() 대신 partial_fit()만 사용해볼 것.
# partial_fit() 사용하려면 훈련셋 전체 클래스 레이블을 이 메소드에 전달해줘야 한다.
# 이를 위해 train_target의 7개 생선 목록을 만든다.
# 에포크 한 번마다의 점수 기록을 위해 2개의 리스트도 준비한다.
import numpy as np
sc = SGDClassifier(loss ='log', random_state=42)
train_score = []
test_score = []
classes = np.unique(train_target)

# 300번 반복해본다.
for _ in range(0, 300):  # 여기서 _는 나중에 사용하지 않고 버릴 용도로 쓰는 임시의 특별한 변수이다.
  sc.partial_fit(train_scaled, train_target, classes=classes)
  train_score.append(sc.score(train_scaled, train_target))
  test_score.append(sc.score(test_scaled, test_target))

# 그래프로 그려보기
import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.show()
# 100번 정도 이후부터 점수가 벌어지는 것 같다. 확실히 초기엔느 과소적합이라 둘 점수 모두 낮다.
# 100에 맞추고 다시 훈련해보고 점수를 출력한다.
sc = SGDClassifier(loss='log', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))  # 0.95798
print(sc.score(test_scaeld, test_target))  # 0.925
# SGDClassifier는 일정 에포크 동안 성능 향상되지 않으면 알아서 자동으로 멈춘다. tol에서 향상될 최솟값을 지정한다.
# 이 코드에서는 None으로 지정하여 정해놓은 100까지 무조건 완수하도록 하였다.
```
- 확률적 경사 하강법 이용한 회귀 알고리즘은 SGDRegressor로 사용하며 사용법은 같다.
- loss 매개변수에 대해 알아본다.
  - 기본 값은 hinge이다.
    - 힌지손실 hinge loss는 서포트 벡터 머신support vector machine이라 불리는 알고리즘을 위한 손실 함수이다.
    - 아래는 예시로 힌지손실 사용한 모델 훈련.
```python
sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))  # 0.949579
print(sc.score(test_scaled, test_target))  # 0.925
```

  - 요즘은 대량 데이터로 문제 해결하는 경우가 흔하다. 전통적인 머신러닝 방식의 모델 만들기에는 컴퓨터 메모리에 모든 것이 들어가기 힘들다.
  - 따라서 점진적으로 학습하는 방법이 필요해졌고 이 때 확률적 경사 하강법 사용.
  - 지금까지 배운 KNN, 선형회귀, 릿지회귀, 라쏘회귀, 로지스틱회귀, 확률적경사하강법 등 보다 더 좋은 알고리즘이 있다. 신경망 알고리즘을 빼고는 머신러닝에서 가장 좋은 성능.
- SGDClassifier 확률적 경사 하강법을 이용한 분류 모델
  - loss 매개변수로 최적화할 손실 함수 지정. 기본은 SVM을 위한 hinge. 로지스틱 회귀를 위해선 log 사용.
  - penalty 매개변수로 규제 종류 지정 가능. 기본값은 L2 규제를 위한 l2이다. L1을 사용하려면 l1이라 해야 함.
  - 규제 강도는 alpha 매개변수에서 지정. 기본값은 0.0001 이다.
  - max_iter은 에포크 횟수 지정. 기본값은 1000.
  - tol은 반복을 멈출 조건. n_iter_no_change 매개변수에서 지정한 에포크 동안 손실이 tol만큼 줄어들지 않으면 알고리즘 중단된다.
  - 기본값은 0.001이고 n_iter_no_change의 기본값은 5이다.
- SGDRegressor은 확률적 경사 하강법 이용한 회귀 모델.
  - loss 매개변수에서 손실함수 지정. 기본값은 제곱 오차 나타내는 squared_loss 이다.
  - 앞의 classifier에서 사용된 매개변수도 여기서 동일하게 사용 됨.
