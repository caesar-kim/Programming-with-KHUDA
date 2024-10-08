# 6주차. 6장. 비지도 학습(p.285-p.338)
## 6-1. 군집 알고리즘(p.286)
- 과일 사진을 보내고 가장 많은 요청이 오는 과일을 판매 품목으로 선정하는 알고리즘.
- 또, 1위로 선정된 과일사진을 보낸 고객 몇 명 뽑아 이벤트 당첨자로 선정.
- 이 때 사진을 종류대로 모아야 한다.
### 타깃을 모르는 비지도 학습(p.287)
- 비지도학습 unsupervised learning
  - 타깃이 없을 때 사용하는 머신러닝 알고리즘
  - 대표적으로 군집, 차원 축소 등
  - 사람이 가르쳐주지 않아도 무언가를 학습한다.
      - 사진 픽셀값을 평균내면 모이지 않을까?
      - 300장의 데이터를 준비하고 픽셀값으로 사진 분류.
### 과일 사진 데이터 준비하기(p.287)
- 사과, 바나나, 파인애플이 있는 흑백 사진
  - 넘파이 배열 기본 저장 포맷인 npy 형식으로 저장됨.
    - 코랩에서 ! 문자로 시작하면 이후 명령을 파이썬 코드가 아닌 리눅스 셀shell 명령으로 이해한다.
    - wget 명령은 원격 주소에서 데이터를 다운로드하여 저장한다.
    - -0 옵션에서 저장할 파일 이름을 지정할 수 있다.
```python
!wget https://bit.ly/fruits_300 -0 fruits_300.npy

import numpy as np
import matplotlib.pyplot as plt

fruits = np.load('fruits_300.npy')
print(fruits.shape)  # 배열 크기 확인
# 300, 100, 100  샘플 개수, 이미지 높이, 이미지 너비

# 첫 이미지의 첫 번째 행을 출력.
# 3차원 배열이기 때문에 처음 2개 인덱스를 0으로 지정, 마지막 인덱스는 지정하지 않거나 슬라이싱 연산자로 모두 선택 가능.
print(fruits[0, 0, :])

# 흑백 사진을 담고 있으므로 0~255까지의 정숫값을 가진다.
# matplotlib의 imshow() 함수로 넘파이 저장 이미지를 쉽게 그릴 수 있다.
# 흑백 이미지라 cmap를 gray로 지정.
plt.imshow(fruits[0], cmap='gray')
plt.show()
# 0에 가까울 수록 검고, 높은 값은 밝게 표시.
```
  - 보통 흑백 이미지는 바탕이 밝고 물체가 짙은 색이다. 그런데 여기서 나온 사진은 반대이다.
  - 이 흑백 이미지는 사진으로 찍은 이미지를 넘파이 배열로 변환할 때 반전시킨 것.
  - 흰바탕은 검게, 검은 것은 희게 만들었다.
    - 이렇게 바꾼 이유는?
    - 우리의 관심은 바탕면이 아닌 사과이기 때문.
    - 255에 가까운 곳에 집중하게 만들기 위해서.
    - 알고리즘이 어떠한 출력을 만들기 위해서 곱셈, 덧셈을 하는데 픽셀값이 애초에 0이면 출력도 0이라 의미가 없기 때문.
    - 픽셀값이 높으면 출력값도 커져서 의미를 부여하기 좋다.
```python
# 따라서 원래 이미지로 보기 위해서는
# cmap='gray_r'로 하여 반전하여 표현.
plt.imshow(fruits[0], cmap='gray_r')
plt.show()

fig, axs = plt.subplots(1, 2)
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()
```
  - matplotlib의 subplots() 함수로 여러 개 그래프를 배열처럼 쌓을 수 있도록 도와준다.
    - 이 함수의 두 매개변수는 그래프를 쌓을 행과 열을 지정한다.
    - 여기서는 (1, 2)로 하나의 행과 2개의 열을 지정했다.
  - axs는 2개의 서브 그래프를 담고 있는 배열. axs[0]에 파인애플 이미지를, axs[1]에 바나나 이미지를 그렸다.
  - 이 장에선 subplots()를 사용해 한 번에 여러 개의 이미지를 그려본다.
  - 사과 100개, 파인애플 100개, 바나나 100개 이미지를  각각 사진 평균 내어서 차이를 확인해본다.

### 픽셀값 분석하기(p.292)
- fruits를 사과, 파인애플, 바나나로 나눈다.
  - 100X100인 이미지를 펼쳐서 10,000의 길이의 1차원 배열로 만든다.
  - 이렇게 펼치면 이미지 출력은 어렵지만, 배열 계산할 때 편리하다.
  - 우리는 3개의 과일이 100개씩 있는 걸 알고 있지만 실전에서는 몇 개가 입력될지 모른다는 점 참고하기.
    - 100개씩 선택하기 위해 슬라이싱 연산자 사용.
    - 그 다음 reshape() 메소드로 두 번째 차원(100)과 세 번째 차원(100)을 10,000으로 합친다.
    - 첫 번째 차원을 -1로 지정하면 자동으로 남은 차원을 할당한다.
    - 여기서는 첫 번째 차원이 샘플 개수이다.
```python
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)

# 배열 크기 확인해보기
print(apple.shape)  # (100, 10000)

# 각 과일 종류별 배열에 들어있는 샘플의 픽셀 평균값을 계산해본다.
# mean() 메소드 사용. 각 샘플마다 픽셀 평균을 계산해야 하므로 평균 계산 축을 지정해야 한다.
# axis = 0 으로 하면 첫 번째 축인 행을 따라 계산한다. 행 방향으로 아래로
# axis = 1로 지정하면 두 번째 축인 열을 따라 계산한다. 열 방향으로 ->

# 샘플은 모두 가로로 나열했으니 axis=1로 지정하여 평균 계산.
# 평균 계산하는 np.mean() 함수도 되지만, 넘파이 배열은 이런 함수들을 메소드로 제공하기도 한다.
print(apple.mean(axis=1))  # 사과 100개에 대한 픽셀 평균값 계산
# 히스토그램을 그려서 평균값 분포를 한 눈에 확인 가능.
# 히스토그램을 모두 겹쳐 그려본다. 조금 투명하게 해야 겹치는 부분 잘 볼 수 있음.
# alpha 매개변수를 1보자 작게 하여 투명도 결정 가능.
# legend() 함수로 어떤 과일의 히스토그램인지 범례도 만들 수 있음.
plt.hist(np.mean(apple, axis=1), alpha=0.8)
plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
plt.hist(np.mean(banana, axis=1), alpha=0.8)
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()
```
- 히스토그램 결과
  - 결과를 보면 바나나 평균값은 40 아래에, 사과와 파인애플은 90~100 사이에 많이 겹쳐있다.
  - 바나나는 픽셀 평균만으로 쉽게 구분 가능. 바나나는 사진에서 차지하는 영역이 작기 때문에 평균값이 작다.
  - 사과, 파인애플은 픽셀 평균만으로 구분하기 힘들다.
- 샘플 전체 평균이 아닌 픽셀 별 평균값으로 비교하면 어떨까?
  - 세 과일은 모양이 다르므로 픽셀값이 높은 위치가 다를 것 같다.
  - 픽셀 평균 계산도 간단하다. axis = 0으로 지정.
  - matplotlib의 bar() 함수로 픽셀 10,000개에 대한 평균값을 막대그래프로 그려본다.
  - subplots() 함수로 3개의 서브 그래프를 만들어 사과, 파인애플, 바나나에 대한 막대그래프를 그려본다.
```python
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].bar(range(10000), np.mean(apple, axis=0))
axs[1].bar(range(10000), np.mean(pineapple, axis=0))
axs[2].bar(range(10000), np.mean(banana, axis=0))
plt.show()
```
- 각 과일 별 결과
  - 과일마다 값이 높은 구간이 다르다.
  - 사과는 아래쪽으로 갈수록 높고, 파인애플은 비교적 고르면서 높다. 바나나는 중앙 픽셀값이 높다.
  - 픽셀 평균을 100X100 크기로 바꿔서 이미지처럼 출력하여 위 그래프와 비교하면 더 좋다.
  - 픽셀을 평균 낸 이미지를 모든 사진을 합쳐놓은 대표 이미지로 생각할 수 있다.
```python
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)
fig, axs = plt.subplots(1, 3, figsize(20, 5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()
```
- 세 과일은 픽셀 위치에 따라 값의 크기가 차이 난다. 따라서 이렇게 만든 대표이미지와 가까운 사진을 고르면 되지 않나?
### 평균값과 가까운 사진 고르기(p.297)
  - 사과 사진 평균인 apple_mean과 가장 가까운 사진을 골라본다. 절댓값 오차를 사용해본다.
  - numpy abs() 함수는 절댓값 계산 함수. np.absolute()의 다른 이름이다.
  - 다음 코드에서 abs_diff는 (300, 100, 100) 크기의 배열이다. 따라서 각 샘플 평균을 구하기 위해 axis에 두 번째, 세 번째 차원을 모두 지정했다.
  - 이렇게 계산한 abs_mean은 각 샘플의 오차 평균이므로 크기가 (300, )인 1차원 배열이다.
```python
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1, 2))
print(abs_mean.shape)
# (300, )
```
  - 이 다음 이 값이 가장 작은 순서대로 100개를 골라본다.
  - np.argsort() 함수는 작은 것부터 큰 순서로 나열한 abs_mean 배열의 인덱스를 반환한다.
  - 이 인덱스에서 첫 100개를 선택해 10X10  격자로 이루어진 그래프를 만들어본다.
```python
apple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
  for j in range(10):
    axs[i, j].imshow(fruits[apple_index[i*10 + j]], cmap='gray_r')
    axs[i, j].axis('off')
plt.show()
```
- 100개 모두 사과가 나왔다.
  - subplots() 함수로 10X10개 총 100개의 서브 그래프를 만든다.
  - figsize는 그래프가 많기 때문에 (10, 10)으로 조금 크게 지정했다.
    - 기본값은 (8, 6)이다.
  - 2중 for문 순회하면서 10개 행과 열에 이미지를 출력한다.
  - axs는 (10, 10)의 2차원 배열이므로 i, j라는 두 첨자를 사용해서 서브 그래프 위치를 지정한다.
  - 또 깔끔한 이미지만 그리기 위해 axis('off')를 사용하여 자표축은 그리지 않았다.

- 흑백 사진의 픽셀값으로 과일 사진 모으는 작업.
  - 이렇게 비슷한 샘플끼리 그룹으로 모으는 작업을 군집clustering이라 한다.
  - 대표적인 비지도 학습 작업 중 하나.
  - 이 알고리즘에서 만든 그룹을 클러스터cluster라고 한다.
  - 하지만 우리는 사과, 바나나, 파인애플이 있다는 것을 미리 알았다.
    - 타깃을 알았기 때문에 평균값 계산해서 가장 가까운 과일을 구한 것.
    - 실제 비지도학습은 타깃값을 몰라서 샘플 평균을 미리 구할 수 없다.
    - 이 해결법은 다음 절에 있다.
  
## 6-2. k-평균(p.303)
    - 진짜 비지도학습은 사진에 어떤 과일이 있는지 알지 못한다.
    - 이럴 때 K-평균 K-means 군집 알고리즘이 평균값을 자동으로 찾아준다.
    - 이 평균값이 클러스터 중심에 위치해서 클러스터 중심cluster center 또는 센트로이드centroid 라고 부른다.
### k-평균 알고리즘 소개(p.304)
- 작동 방식
  - 1) 무작위로 k개 클러스터 중심을 정한다.
  - 2) 각 샘플에서 가장 가까운 클러스터 중심을 찾아 해당 클러스터의 샘플로 지정한다.
  - 3) 클러스터에 속한 샘플 평균값으로 클러스터 중심을 변경.
  - 4) 클러스터 중심에 변화가 없을 때까지 2번으로 돌아가 반복한다.
- 처음에는 랜덤한 클러스터 중심을 선택하고 점차 가장 가까운 샘플 중심으로 이동하는 비교적 간단한 알고리즘이다.
- 사이킷런으로 모델을 직접 만들어본다.
### KMeans 클래스(p.305)
```python
!wget https://bit.ly/fruits_300 -0 fruits_300.npy

# numpy의 np.load() 함수로 npy 파일을 읽어 넘파이 배열을 준비한다.
# k-평균 모델 훈련을 위해 (샘플 개수, 너비, 높이) 크기의 3차원 배열을 (샘플 개수, 너비X높이) 크기를 가진 2차원 배열로 변경한다.
import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

# 사이킷런의 k-평균 알고리즘은 sklearn.cluster 모듈 아래 KMeans 클래스에 구현됨.
# n_cluster는 클러스터 개수 지정. 여기서는 3으로 지정.
# 지도학습과 비슷한 방식이지만 fit() 메소드에서 타깃데이터를 사용하지 않는 차이가 있다.

from sklearn.cluster import KMeas
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)

# 군집된 결과는 KMeans 클래스 객체의 labels_ 속성에 저장된다.
# 이 속성의 길이는 샘플 개수로 각 샘플이 어떤 레이블에 해당되는지 나타낸다. n_cluster=3이기 때문에 labels_ 배열의 값은 0, 1, 2 중 하나이다.
print(km.labels_)
# 0, 1, 2의 순서는 아무 의미 없다. 실제 어떤 걸 주로 모았는지 확인하려면 직접 이미지 출력하는 것이 최선이다.
# 이미지 출력 전에 0, 1, 2 각각의 샘플 개수를 확인해본다.
print(np.unique(km.labels_, return_counts=True))
# (array([0, 1, 2], dtype=int32), array([91, 98, 111]))
```
    - 결과
        - 레이블0이 91개, 1이 98개, 2가 111개의 샘플을 모았다.
        - 각 클러스터가 나타낸 이미지를 그림으로 출력하기 위해 간단한 유틸리티 함수 draw_fruits()를 만들어 본다.
```python
import matplotlib.pyplot as plt
def draw_fruits(arr, ratio=1):
  n = len(arr)  # n은 샘플 개수이다.
  # 한 줄에 10개씩 이미지를 그린다. 샘플 개수를 10으로 나누어 전체 행 개수를 계산한다.
  rows = int(np.ceil(n/10))
  # 행이 1개이면 열의 개수는 샘플 개수이다. 그렇지 않으면 10개이다.
  cols = n if rows < 2 else 10
  fig, axs = plt.subplots(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False)
  for i in range(rows):
    for j in range(cols):
      if i*10 + j < n:  # n개까지만 그린다.
        axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
      axs[i, j].axis('off')
plt.show()
```
- 이 함수는 (샘플개수, 너비, 높이)의 3차원 배열을 입력받아 가로로 10개씩 이미지를 출력한다.
- 샘플 개수에 따라 행과 열의 개수를 계산하고 figsize를 지정한다.
  - figsize는 ratio 매개변수에 비례하여 커진다. 기본값은 1.
- 2중 for 반복문
  - 먼저 첫 번째 행을 따라 이미지를 그린다. 그리고 두 번째 행의 이미지를 그린다.
- 이 함수를 이용해 레이블이 0인 과일 사진을 다 그려본다. km.labels_==0과 같이 쓰면 km.labels_ 배열에서 값이 0인 위치는 True, 그 외는 False가 된다.
  - 넘파이는 이러한 불리안 배열을 사용해 원소 선택 가능.
  - 이를 불리언 인덱싱이라 한다.
  - 넘파이 배열에 불리언 인덱싱 적용하면 True 위치의 원소만 모두 추출한다.
```python
draw_fruits(fruits[km.labels_==0])
# 모두 사과가 올바르게 추출되었다.
draw_fruits(fruits[km.labels_==1])
# 모두 바나나로만 이루어져 있다.
draw_fruits(fruits[km.labels_==2])
# 파인애플 사이에 사과 9개와 바나나 2개가 더 섞여 있다.
# 완벽하지는 않지만, 타깃데이터 없이도 스스로 비슷한 샘플을 잘 모은 것 같다.

### 클러스터 중심(p.309)
- 클러스터 중심
  - KMeans가 최종적으로 찾은 중심은 cluster_centers_ 속성에 저장됨.
  - 이 배열은 fruits_2d 샘플의 클러스터 중심이기 때문에 이미지로 출력하려면 100X100 크기의 2차원 배열로 바꿔야 한다.

```python
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)
# 이전 절에서 추출한 픽셀 평균값의 그림과 유사한 것 같다.

# KMeans 클래스는 훈련 데이터 샘플에서 클러스터 중심까지의 거리를 변환하는 transform() 메소드를 갖고 있다.
# 이 메소드는 StandardSclaer 클래스처럼 특성값을 변환하는 도구로 사용할 수 있다는 의미이다.
# 인덱스가 100인 샘플에 transform() 메소드를 적용해 본다. fit() 메소드와 마찬가지로 2차원 배열을 기대한다.
# fruits_2d[100] 처럼 쓰면 (10000, ) 크기의 배열이 되어 에러 발생.
# 슬라이싱 연산자로 (1, 10000) 크기의 배열을 전달해본다.
print(km.transform(fruits_2d[100:101]))
# [[5267.7043  8837.3775  3393.8136]]

# 하나의 샘플을 전달했기 때문에 반환된 배열은 (1, 클러스터 개수)인 2차원 배열이다.
# 세번째 클러스터까지의 거리가 3393.8로 가장 작다.
# 이 샘플은 레이블 2에 속한 것 같다.
# KMeans 클래스는 가장 가까운 클러스터 중심을 예측 클래스로 출력하는 predict() 메소드를 제공한다.
print(km.predict(fruits_2d[100:101]))
# [2]
# transform() 결과에서 짐작했듯이 [2] 레이블2로 예측했다. 아마 파인애플일 것.
draw_fruits(fruits[100:101])
# 파인애플이 맞다.

# 알고리즘은 반복적으로 클러스터 중심을 옮기면서 최적의 클러스터를 찾는다. 반복 횟수는 n_iter_ 속성에 저장된다.
print(km.n_iter_)
# 3
```
  - 클러스터 중심을 특성 공학처럼 사용해 데이터셋을 저차원(10,000에서 3으로 줄임)으로 변환할 수 있다.
  - 또는 가장 가까운 거리에 있는 클러스터 중심을 샘플의 예측값으로 사용할 수 있는 것을 배웠다.
  - 우리는 타깃값을 사용하지는 않았지만 n_cluster을 3으로 지정하여 타깃에 대한 정보를 사용하기는 했다.
  - 실전에서는 클러스터 개수조차 알 수 없다. 어떻게 지정해야 하는가?

### 최적의 k 찾기(p.311)
- k-평균 알고리즘의 단점 중 하나는 클러스터 개수 사전 지정이다. 실전에서는 몇 개의 클러스터가 있는지 알 수 없다.
- 완벽한 방법은 없다. 대표적인 방법은 엘보우elbow.
  - k-평균 알고리즘은 클러스터 중심과 속한 샘플 사이의 거리를 잴 수 있다.
    - 이 거리의 제곱합을 이너셔inertia라고 부른다.
    - 이너셔는 클러스터에 속한 샘플이 얼마나 가까이 모여 있는지 나타내는 값으로 생각 가능.
  - 일반적으로 클러스터 개수가 늘어나면 클러스터 개개의 크기가 줄어서 이너셔도 감소한다.
    - 엘보우 방법은 클러스터 개수를 늘려가면서 이너셔 변화를 관찰하여 최적의 클러스터 개수를 찾는 방법.
  - 이너셔가 감소하는 속도가 꺾이는 지점이 있다. 이 지점부터는 클러스터를 늘려도 클러스터에 잘 밀집된 정도가 크게 개선되지 않는다.
    - 즉 이너셔가 크게 줄어들지 않는다. 이 지점 모양이 팔꿈치 모양이라 엘보우 방법.
  - KMeans 클래스는 자동으로 이너셔 계산해서 inertia_ 속성으로 제공.
  - 다음 코드는 k를 2~6으로 바꿔가며 5번 훈련한다. inertia_ 속성에 저장된 값을 inertia 리스트에 추가하고 이 저장된 값을 그래프로 출력해본다.
```python
inerita = []
for k in range(2, 7):
  km = KMeans(n_clusters=k, random_state=42)
  km.fit(fruits_2d)
  inertia.append(km.inertia_)
plt.plot(range(2, 7), inertia)
plt.show()
```
  - 뚜렷하지는 않지만 k=3 지점에서 그래프 기울기가 조금 바뀐 것을 볼 수 있다.
  - 엘보우 지점보다 클러스터 개수가 많아지면 이너셔 변화가 줄어들어 군집효과도 줄어든다.
  - 이 그래프는 명확하게 지점이 나타나지는 않는다.

- KMeans
  - k-평균 알고리즘 클래스이다.
    - n_cluster: 클러스터 개수 지정. 기본값 8.
    - 처음에 랜덤하게 센트로이드를 초기화하기 때문에 여러 번 반복하여 이너셔 기준으로 가장 좋은 결과를 선택한다.
      - n_init은 반복횟수를 지정한다. 기본값은 10.
    - max_iter은 k평균 알고리즘 한 번 실행에서 최적의 센트로이드 찾기 위해 반복할 수 있는 최대 횟수. 기본값 200.
## 6-3. 주성분 분석(p.318)
- 과일 사진 이벤트는 성공
  - 매일 여러 과일 사진이 업로드 중.
  - 업로드된 사진을 클러스터로 분류하여 폴더 별로 저장.
  - 그러나 너무 많은 사진이 등록되어 저장 공간이 부족하다.
### 차원과 차원 축소(p.319)
- 차원
  - 데이터가 가진 속성을 차원이라고 했다. 과일 사진에는 10000개의 픽셀이 있으니 10000개의 특성인 셈.
  - 이 10000개의 차원을 줄일 수 있으면 저장공간 절약 가능.
  - 다차원 배열에서는 차원은 축 개수가 된다. 그런데 1차원 배열인 벡터에서는 원소의 개수가 축 개수가 된다.
- 차원 축소 알고리즘 dimensionality reduction
  - 특성이 많으면 선형 모델 성능이 높아지고 훈련셋에 쉽게 과적합된다고 했었다.
  - 차원 축소는 데이터를 가장 잘 나타내는 일부 특성 선택하여 데이터 크기를 줄이고 지도학습 모델의 성능을 향상시킬 수 있는 방법.
  - 줄어든 차원에서 다시 원본 차원으로 손실을 최대한 줄이면서 복원도 가능.
  - 대표적인 차원 축소 알고리즘인 주성분 분석 PCA principal component analysis
### 주성분 분석 소개(p.320)
- PCA
  - 데이터에 있는 분산이 큰 방향을 찾는 것으로 이해 가능.
  - 분산은 데이터가 널리 퍼진 정도. 분산이 큰 방향을 데이터로 잘 표현하는 벡터로 생각 가능.
    - 이 벡터를 주성분 principal component 라고 부른다.
    - 이 주성분 벡터는 원본 데이터에 있는 어떤 방향이다.
  - 따라서 주성분 벡터의 원소 개수는 원본 데이터셋 특성 개수와 같다.
  - 샘플 데이터를 주성분벡터에 직각으로 투영하면 1차원 데이터를 만들 수 이다.
    - 주성분이 가장 분산이 큰 방향이기 때문에 주성분에 투영하여 바꾼 데이터는 원본이 갖고 있는 특성을 가장 잘 나타내고 있다.
  - 첫 주성분을 찾았다면 이 벡터에 수직이고 분산이 가장 큰 다음 방향을 찾는다. 이것이 두 번째 주성분.
    - 일반적으로 주성분은 원본 특성 개수만큼 찾을 수 있다.
    - 정확히는 원본 특성 개수와 샘플 개수 중 작은 값만큼 찾을 수 있지만, 일반적으로 비지도학습은 대량의 샘플로 수행하기 때문에 보통 원본 특성 개수.
### PCA 클래스(p.322)
```python
# 과일사진을 웹페이지에서 다운로드하여 넘파이 배열로 적재한다.
!wget https://bit.ly/fruits_300 -0 fruits_300.npy
import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

# sklearn.decomposition 모듈 아래 PCA 클래스로 주성분 분석 알고리즘 제공.
# n_components 매개변수에 주성분 개수를 지정해야 한다.
# 비지도 학습이라 fit() 메소드에 타깃 값은 넣지 않는다.
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(fruits_2d)

# 주성분은 components_ 속성에 저장
print(pac.components_.shape)
# (50, 10000)

# 주성분 개수를 50개로 지정해서 이 배열의 첫 차원은 50이 나온다. 50개의 주성분을 찾은 것.
# 두 번째 차원은 항상 원본 데이터 특성 개수와 같은 10000이다.
# 원본데이터와 같은 차원이므로 100X100 크기의 이미지처럼 출력 가능.
draw_fruits(pca.components_.reshape(-1, 100, 100))

# 이 주성분은 원본 데이터에서 가장 분산이 큰 방향 순서대로 나타낸 것.
# 즉, 데이터셋의 어떤 특징을 잡아낸 것.

# 주성분을 찾았으므로, 원본 데이터를 주성분에 투영하여 특성 개수를 10000개에서 50개로 줄일 수 있다.
# 마치 원본데이터를 각 주성분으로 분해하는 것으로 생각 가능.
# PCA의 transform() 메소드로 원본데이터 차원을 50으로 줄일 수 있다.
print(fruits_2d.shape)
# (300, 10000)
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)
# (300, 50)
```
  - fruits_2d는 (300, 10000) 크기의 배열이었다. 50개의 주성분을 찾은 PCA 모델을 사용해 이를 (300, 50) 크기 배열로 변환했다.
  - 이제 fruits_pca 배열은 50개 특성을 가진 데이터이다.
  - 데이터를 1/200이나 줄였다. 이를 다시 원상 복구할 수도 있나?

### 원본 데이터 재구성(p.324)
  - 10000개 특성을 50개로 줄였으니 손실이 발생했을 것.
  - 하지만 최대한 분산이 큰 방향으로 데이터를 투영했기 때문에 상당 부분 재구성 가능하다.
  - 이를 위한 메소드는 inverse_transform() 이다. 50개 차원으로 축서한 데이터를 전달해 10000개 특성으로 복원해본다.
```python
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)
# (300, 10000)  10000개 특성으로 복원 되었다.
# 이를 100개씩 나누어 출력해본다.

fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
for start in [0, 100, 200]:
  draw_fruits(fruits_reconstruct[start:start+100])
  print("\n")
```
- 거의 모든 과일이 잘 복원 되었다.
  - 일부 흐리고 번지기도 했지만 불과 50개 특성을 10000개로 늘린 것을 감안하면 잘 된 것 같다.
  - 만약 주성분을 최대로 사용했다면 원본 데이터를 재구성할 수 있을 것.
  - 50개의 특성은 얼마나 분산을 보존하고 있는 것인가?
### 설명된 분산(p.325)
- 설명된 분산explained variance
  - 주성분이 원본 데이터의 분산을 얼마나 잘 나타내는지 기록한 값.
  - explained_variance_ratio_ 에 각 주성분 분산 비율이 기록되어 있다.
  - 당연히 첫 번재 주성분의 설명된 분산이 가장 크다.
  - 이 50개의 비율을 모두 더하면 50개 주성분으로 표현한 총 분산 비율을 얻을 수 있다.
```python
print(np.sum(pca.explained_variance_ratio_))
# 0.9215
plt.plot(pca.explained_variance_ratio_)
```
- 92%가 넘는 분산을 유지한다.
  - 복원한 이미지의 품질이 높은 이유가 여기 있다.
  - 설명한 분산 비율을 그래프로 그려보면 적절한 주성분 개수 찾는데 도움이 된다.
  - 처음 10개가 대부분의 분산을 표현하고 있다.
  - 그 뒤부터는 설명되는 분산이 비교적 작다.
- 이번에는 PCA로 차원 축소된 데이터를 사용하여 지도학습모델을 훈련해본다.
### 다른 알고리즘과 함께 사용하기(p.327)
- 원본 데이터와 PCA로 차원 축소한 데이터를 지도학습에 적용해서 어떤 차이가 있는지 알아본다.
- 3개 과일 사진 분류이므로 로지스틱 회귀 모델 사용할 것.
```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
# 타깃값은 사과0, 파인애플1, 바나나2로 지정한다.
# 파이썬 리스트와 정수를 곱하면 리스트 안의 원소를 정수만큼 반복한다.
target = np.array([0]*100 + [1]*100 + [2]*100)

# 원본데이터인 fruits_2d를 사용. cross_validate()로 교차검증도 한다.
from sklearn.model_selection import cross_validate
scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score']))  # 0.9966
print(np.mean(scores['fit_time']))  # 0.9422

# 교차 검증 점수는 0.997로 매우 높다. 특성이 10000개라 300개 샘플에서는 과적합 모델을 금방 만들 수 있다.
# cross_validate() 함수가 반환하는 딕셔너리에는 fit_time 항목에 교차검증 폴드 훈련 시간이 기록되어 있다.
# 약 0.94초. 이를 PCA로 축소한 데이터를 사용했을 때와 비교해본다.
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))  # 1.0
print(np.mean(scores['fit_time']))  # 0.0325

# 50개 특성만 이용했는데도 정확도가 100% 이고, 훈련 시간은 0.03초로 20배 감소했다.
# PCA로 차원 축소하면 저장공간 뿐 아니라 훈련속도도 높일 수 있다.
# PCA 클래스 사용할 때 n_components 매개변수에 주성분 개수를 지정했다.
# 대신 원하는 설명된 분산 비율도 입력할 수 있다. 이 비율에 도달할 때까지 자동으로 주성분을 찾는다.
# 50%에 달하는 주성분을 찾도록 모델 만든다.
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)
# 개수 대신 0~1 사이 실수를 입력하면 된다.

print(pca.n_components_)
# 2
#  단 2개만으로 원본 데이터의 50% 분산 표현 가능하다.

# 이 모델을 원본 데이터로 변환.
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)
# (300, 2)

# 교차검증 결과도 확인
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))  # 0.9933
print(np.mean(scores['fit_time']))  # 0.0412
# 이 코드를 입력하면 로지스틱 회귀 모델이 완전히 수렴하지 않아서 반복횟수를 증가하라는 경고(Convergence Warning)가 출력된다. 하지만 교차검증 결과가 충분히 좋아서 무시해도 좋다.
# 2개 특성만으로도 0.9933 정확도가 나온다.

from sklearn.cluster import KMeans
km = KMeans(n_cluster=3, random_state=42)
km.fit(fruits_pca)
print(np.unique(km.labels_, return_counts=True))
# (array[0, 1, 2], dtype=int32), array([91, 99, 110]))

# 2절에서 원본 데이터를 사용했을 때와 거의 비슷한 결과가 나온다.
# 이미지를 출력해본다.
for label in range(0, 3):
  draw_fruits(fruits[km.labels_ == lable])
  print("\n")

# 2절에서 찾은 클러스터와 비슷하게 파인애플에는 사과가 몇 개 섞여 들어갔다.
# 훈련 데이터 차원 줄이면 시각화라는 장점을 얻을 수 있다.
# 3개 이하로 차원을 줄이면 화면에 출력하기 비교적 쉽다.
# fruits_pca 데이터는 특성이 2개라 2차원으로 표현 가능.

for label in range(0, 3):
  data = fruits_pca[km.labels_ == lable]
  plt.scatter(data[:, 0], data[:, 1])
plt.legend(['apple', 'banana', 'pineapple'])
plt.show()
```
- 클러스터의 산점도를 보면 잘 구분되는 것을 볼 수 있다. 2개 특성만으로 로지스틱 회귀모델 교차검증 점수가 99%인 이유를 알 수 있다.
- 그림을 보면 사과와 파인애플 경계가 가깝다. 따라서 혼동이 일어날 수도 있을 것 같다. 데이터를 시각화하면 이러한 예상치 못한 통찰을 얻을 수 있다.

- PCA
  - n_components 는 주성분 개수 지정. 기본값 None인데, 샘플 개수와 특성 개수 중 작은 것의 값을 사용한다.
  - random_state는 넘파이 난수 시드값 지정
  - components_ 훈련셋에서 찾은 주성분이 저장된다.
  - explained_variance_ 속성에는 설명된 분산이 저장되고,
  - explained_variance_ratio_ 에는 설명된 분산의 비율이 저장된다.
  - inverse_transform() 메소드는 transform() 메소드로 차원을 축소시킨 데이터를 다시 원본 차원으로 복원한다.
