# 3주차. 3장. 회귀 알고리즘과 모델 규제(p.114-p.175)

## 3-1. K-최근점 이웃 회귀(p.114)
### K-최근접 이웃 회귀(p.115)
- 지도학습 알고리즘 -> 1\) 분류와 2\) 회귀로 나뉘어짐.
  - 분류는 샘플은 클래스 중 하나로 분류하는 문제이고 회귀는 임의의 어떤 숫자를 예측하는 문제.
  - 회귀는 1980년 대 Francis Galton이 처음 사용한 용어.
  - K-최근접 이웃 분류 알고리즘과 비슷한 방식. 분류 방식은 이웃의 샘플들 확인해서 그 중 다수 클래스를 샘플의 클래스로 예측하는 것.
  - K-최근접 이웃 회귀도, 이웃한 샘플의 타깃이 클래스가 아니라 임의의 수치인 차이.

### 데이터 준비(p.116)
```python
import numpy as np
perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7, 23.0, 23.5, 24.0, 24.0,
    24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0,
    43.0, 43.5, 44.0]
    )

# 데이터 형태를 보기 위한 산점도 그리기
import matplotlib.pyplot as plt
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 훈련 세트와 테스트 세트 나누기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
```
- 사이킷런 훈련세트는 2차원 배열이어야 하는데, 현재 자료 형태는 1차원 배열이라서 바꿔줘야 한다.
- 파이썬에서 1차원 배열 크기는 원소가 1개인 튜플로 나타낸다.
  - [1, 2, 3]의 크기는 (3, ) 이다. 이를 2차원 배열로 만들기 위해 억지로 하나의 열을 추가한다. 그러면 배열 크기가 (3, 1)이 된다.
  - 나타내는 방식만 달라졌을 뿐 여전히 원소 개수는 3으로 동일하다.
  - 이전 장에서는 사용한 특성이 2개라 자연스럽게 열이 2개인 2차원 배열을 사용했지만, 여기는 특성 1개로 구할 것이므로 수동으로 만들어야 함.
  - 배열 크기를 바꿀 수 있는 reshape() 메소드 사용하면 됨.
  - reshape()는 크기를 바꾼 새로운 배열 반환할 때 지정한 크기가 원래 배열 원소 개수와 다르면 에러가 발생한다.
```python
test_array = np.array([1, 2, 3, 4])
print(test_array.shape)
# 나오는 결과 : (4, )
# 이 결과를 (2, 2)로 바꿔볼 것.
test_array = test_array.reshape(2, 2)
print(test_array.shape)
# 결과  (2,2)

# 이제 train_input인 (42, )을 (42, 1)로 바꾼다.
# 크기에 -1을 지정하면 나머지 원소 개수로 모두 채우라는 의미.
# -1, 1은 첫 번째 크기를 나머지 원소 개수로 채우고, 두 번째 크기를 1로 하려는 것. 이 기능은 배열 전체 원소 개수를 매번 외우지 않아도 됨.
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
print(train_input.shape, test_input.shape)
```
### 결정계수 Rsquare(p.120)
- KNN 회귀 알고리즘은 KNeighborsRegressor 클래스이다. 객체 생성하고 fit() 메소드로 훈련하게 될 것.
```python
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
# KNN 회귀 모델 훈련
knr.fit(train_input, train_target)
# 테스트 세트 점수 확인
print(knr.score(test_input, test_target))
```
- 회귀에서의 점수는 분류에서의 정확도와 다르다. 결정계수 coefficient of determination이라고 한다. Rsquare이라고도 함.
  - R^2 = 1 - [(target - prediction)^2 / (target - mean)^2]
  - target이 평균 정도를 예측하는 수준이면 분자 분모가 비슷해져 R^2이 0 에 가까워지는 나쁜 결과
  - target이 예측과 아주 가까워지면 분수가 0에 가까워져서 R^2이 1에 가까워지는 결과.
  - 사이킷런의 score() 메소드는 값이 높을수록 좋은 것이다. 만약 에러율을 반환하는 것이라면 음수로 만들어서 낮은 에러가 score() 메소드로 반환될 때는 높은 값이 되도록 해야 한다.
  - R^2가 얼마나 좋은지 직감적으로 이해하기 어렵다. 따라서 target과 prediction의 절댓값 오차를 평균하여 반환해본다.
```python
form sklearn.metrics import mean_absolue_error
# 테스트에 대한 예측을 만든다.
test_prediction = knr.predict(test_input)
# 평균 절댓값 오차 계산
mae = mean_absolute_error(test_target, test_prediction)
print(mae)
```
### 과대적합 vs 과소적합(p.122)
- train set의 R^2도 확인해본다. test 세트의 점수가 더 낮다.
- 모델을 train set에 맞춰 훈련하면 그에 잘 맞는 모델이 만들어진다. 보통은 훈련 세트의 점수가 더 높음.
  - 훈련세트엔 좋은데 테스트 세트에 나쁜 모델은 과대적합overfitting이라고 한다.
  - 반대로 테스트 셋 점수가 더 높거나 훈련셋 테스트셋 둘 다 낮은 경우는 과소적합underfitting이라고 한다. 적절히 훈련되지 않았던 것.
      - 또 다른 원인은 셋의 크기가 너무 작은 경우
  - underfitting 해결은 모델을 더 복잡하게 만들 면 된다. KNN에서는 K를 더 작게 만들면된다.
    - 이웃을 줄이면 더 국소적인 부분의 패턴에 민감해지고, 이웃을 늘리면 일반적인 데이터 전반의 패턴을 따를 것이다.
    - 클래스 변경 없이 그냥 n_neighbors 속성값을 바꾸면 된다.
  - 다시 R^2 확인한 결과 테스트 셋 점수가 훈련 셋 점수보다는 낮아졌다. 또한 둘의 값 차이도 크지 않아서 잘 된 것 같다.
``` python
print(knr.score(train_input, train_target))

# 이웃 개수 3으로 설정
knr.n_neighbors = 3
# 모델 다시 훈련
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target))
# 0.9805
print(knr.score(test_input, test_target))
# 0.9746
```
## 3-2. 선형 회귀(p.130)
### K-최근접 이웃의 한계(p.131)
```python
import numpy as np
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])
from sklearn.model_selection import train_test_split
# 훈련 셋 테스트셋 나눔.
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
# 훈련셋과 테스트셋을 2차원 배열로 바꿈.
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors = 3)
knr.fit(train_input, train_target)

# 길이 50cm인 농어의 무게 예측
print(knr.predict([[50]]))
# 결과 [1033.3333333]

# 문제 확인을 위해 산점도를 그려볼 것.
import matplotlib.pyplot as plt
# 50cm 농어의 이웃을 구한다.
distances, indexes = knr.kneighbors([[50]])

# 훈련셋의 산점도를 그린다.
plt.scatter(train_input, train_target)
# 훈련셋 중 이웃 샘플만 다시 그린다.
plt.scatter(train_input[indexes], train_target[indexes], marker='D')
# 50cm 농어 데이터
plt.scatter(50, 1033, marker='^')
plt.show()
```
- 산점도를 보면 길이가 커질수록 농어 무게가 증가한다. 하지만 50cm 농어의 가장 가까운 것은 45cm 농어 뿐이기에 KNN알고리즘은 이 45cm 농어들의 무게를 평균한다.
- 따라서 예측하려는 샘플이 훈련셋 범위를 벗어나면 이상한 값을 예측할 수도 있다.
- 100cm의 농어도 같은 값을 예측하게 된다. 농어가 아무리 커도 무게는 같은 값을 예측.
- 이를 KNN으로 해결하려면 가장 큰 농어가 포함되도록 훈련셋을 다시 구성해서 훈련해야 하지만, 다른 방법을 사용할 것.
- 실제 머신러닝 모델도 한 번 만들고 끝나는 것이 아닌 주기적으로 새로운 데이터로 훈련해야 한다.
### 선형 회귀Linear Regression(p.135)
- 특성이 하나인 경우 어떤 직선을 학습하는 알고리즘이다.
```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression

# 선형회귀 모델 훈련
lr.fit(train_input, train_target)
# 50cm 농어에 대한 예측
print(lr.predict([[50]]))
# 결과: 1241.8386

# 이 클래스가 찾은 데이터의 알파와 베타는 coef_ intercept_ 속성에 저장되어 있다.
print(lr.coef_, lr.intercept_)
# 결과는 39와 -709

# coef_를 종종 계수coefficient, 가중치weight라고도 부른다.
# coef, intercept를 알고리즘이 찾은 값이라는 의미로 모델 파라미터model parameter라고 부른다. 이 책에서 사용하는 많은 알고리즘은 최적의 모델 파라미터 찾는 과정임.
# 이를 모델 기반 학습이라고 부른다. KNN에서는 모델 파라미터가 없었는데 이를 사례기반학습이라고 한다.

# 이 직선을 그리리면 (15, 15X39-709), (50, 15X39-709) 두 점을 이으면 된다.
# 훈련셋 산점도 그리기
plt.scatter(train_input, train_target)
# 15에서 50까지의 방정식 그래프 그리기
plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])

# 50cm 농어 데이터 입력
plt.scatter(50, 1241.8, marker='^')
plt.show()
# 그린 직선 위에 50cm 농어 데이터 값이 올라간다.

# R^2 점수 확인
print(lr.score(train_input, train_target)) # 결과 0.939
print(lr.score(test_input, test_target)) # 결과 0.825

# 전체적인 과소적합 확인 가능. 그래프에서도 낮은 구간에서 조금 차이가 발생하는 것 볼 수 았음.
```

### 다항 회귀(p.139)
- 선형회귀대로면 직선이 농어 무게가 0 아래로도 내려갈텐데 현실에서는 있을 수 없는 일이다. 직선보다 곡선을 찾기. 2차 방정식으로 그리면 된다.
- 그러기 위해선 길이 제곱한 항이 훈련셋에 포함되어야 한다.

```python
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))
# numpy 브로드캐스팅이 적용되어 모든 원소를 제곱하는 것.

print(trian_poly.shape, test_poly.shape)  # 결과: (42, 2) (14, 2)
# 테스트셋 훈련셋 모두 열이 2개로 늘어났다.
# 훈련셋의 항은 늘어났지만, 타깃항은 그대로다. 타깃항은 어떤 것으로 훈련하든 바꿀 필요가 없다.

lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.predict([[50**2, 50]]))  # 결과 1573.98

print(lr.coef_, lr.intercept_)  # [1.01, -21.56] 116.05
# 이렇게 다항식을 사용한 선형회귀를 다항회귀라 한다.

# 구간별로 직선을 그려서 곡선처럼 표현할 것.
# 15에서 49까지의 정수 배열을 만든다.
point = np.arrange(15, 50)
plt.scatter(train_input, train_target)  # 훈련셋의 산점도 그리기
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)

# 50cm 농어 데이터
plt.scatter([50], [1574], marker='^')
plt.show()

# R^2 점수 확인
print(lr.score(trian_poly, train_target))  # 0.970
print(lr.score(test_poly, test_target))  # 0.977
```
## 3-3. 특성 공학과 규제(p.150)
### 다중 회귀(p.151)
### 데이터 준비(p.152)
### 사이킷런의 변환기(p.154)
### 다중 회귀 모델 훈련하기(p.156)
### 규제(p.158)
### 릿지 회귀(p.160)
### 라쏘 회귀(p.163)
