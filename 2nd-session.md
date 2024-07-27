# 2주차. 1장. 나의 첫 머신러닝(p.25-p.64), 2장. 데이터 다루기(p.65-p.112)

## 1-1. 인공지능과 머신러닝, 딥러닝
- 인공지능이란
    - 사람처럼 학습하고 추론할 수 있는 지능을 가진 컴퓨터 시스템을 만드는 기술.
    - 강인공지능Strong AI or 인공일반지능Artificial General Intelligence: 영화 속 인공지능. 사람과 구분하기 어려운 지능.
    - 약인공지능Week AI: 특정 분야에서 사람의 일을 도와주는 보조 역할.
- 머신러닝이란
    - 규칙을 일일이 프로그래밍하지 않아도 자동으로 데이터에서 규칙을 학습하는 알고리즘을 연구하는 분야. 인공지능의 지능을 구현하기 위한 SW를 담당하는 핵심 분야.
    - 통계학과 깊은 관련.
    - 최근에는 통계나 수학보다는 경험을 바탕으로 발전하는 경우도 많음. ex) 사이킷런scikit-learn 라이브러리.
    - 사이킷런에 포함된 알고리즘들을 사용하면 된다.
- 딥러닝이란
    - 머신러닝 알고리즘 중에 인공신경망을 기반으로 한 방법들을 통칭한 것.
    - 1\) 풍부한 데이터와 2\) 컴퓨터 성능 향상, 그리고 3\) 혁신적인 알고리즘 개발로 가능하게 되었다.
    - 딥러닝 라이브러리는 텐서플로TensorFlow(구글), 파이토치PyTorch(페이스북)이 있다.

## 1-2. 코랩과 주피터 노트북
- 구글 코랩Colab  
        - 웹 브라우저에서 무료로 파이썬 프로그램을 테스트하고 저장할 수 있는 서비스. 클라우드 기반 주피터 노트북 개발 환경.
          - 셀cell은 코드 또는 텍스트 덩어리. 코랩에서 실행할 수 있는 최소 단위.
- 텍스트 셀
    - 코랩 노트북의 장점은  코드 설명 문서를 따로 만들지 않아도 코드와 텍스트, 실행 결과까지 담아서 공유할 수 있다.
    - HTML과 마크다운을 혼용해서 사용할 수 있다.
    - 주요 기능 T 제목으로 만들기, B 볼드체, I 이탤릭체, <> 코드 형식으로 바꿔줌, 링크 표시, 이미지 추가, 들여쓰기, 번호 매기기, 글머리 기호, 가로줄 추가, 미리보기 창 위치 변경.
- 코드 셀
    - 코드와 결과가 함께 선택됨.
- 노트북
    - 코랩은 대화식 프로그래밍 환경인 주피터Jupyter를 커스터마이징 한 것.
    - 주피터 프로젝트의 대표 제품이 주피터 노트북.
    - 코랩 노트북은 구글 클라우드의 가상 서버Virtual Machine를 사용.
    - 이 서버의 메모리는 약 12gb, 디스크 공간 100gb. 최대 5개의 가상 서버까지 무료로 열 수 있고, 최대 12시간까지만 실행 가능.
## 1-3. 마켓과 머신러닝
- 생선 분류 문제
    - 생선 데이터 셋의 출처는 캐글. 세계에서 가장 큰 머신러닝 경영 대회 사이트. 많은 데이터와 참고자료도 제공함.
    - 보통의 프로그램은 '누군가 정해준 대로' 30cm 이상이면 도미라고 판단하지만, 머신러닝은 스스로 기준을 찾아서 일을 한다.
    - 여러 개의 도미 데이터를 제공해서 스스로 찾는 것.
    - 여러 개 종류(또는 클래스) 중에서 하나를 구별하는 문제를 분류classification이라고 한다. 이 경우 2개 중 하나 찾는 것은 이진 분류라고 한다.
    - 도미의 길이와 무게로 리스트를 만든다. 각 도미의 특징을 나타낸 것이 "특성 feature"이라고 한다.
    - 각 특성을 점으로 표현하는 것을 산점도라고 하는데 파이썬에서는 맷플롯립matplotlib이라는 패키지로 과학계산용 그래프를 그린다.
    - import란 따로 만들어진 패키지를 사용하기 위해 불러오는 명령이다.
    - 코랩에는 이미 널리 쓰이는 패키지들이 설치되어 있다.
    - 임포트할 때 as 키워드로 이름을 줄여서 쓴다. 줄임말은 미리 알아두는 것이 좋음.
```python
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```
- 첫 번째 머신러닝 프로그램
    - K-최근접 이웃(K-Nearest Neighbors) 알고리즘 사용.
    - 두 리스트를 하나로 합친다.
    - 사이킷런 패키지 사용할 것. 세로 방향의 2차원 리스트를 만들어야 한다.
    - zip( ) 함수로 나열된 리스트에서 각각 원소를 하나씩 꺼내서 반환한다.
    - 이 2차원 리스트에 정답 데이터를 추가해줘야 한다. 문자를 직접 이해 못하니 도미는 1, 빙어는 0으로.
    - 머신러닝에서 2개 구분하는 경우, 찾으려는 대상을 1로 놓고 그 외는 0으로 놓는다.
    - KNeighborsClassifier 클래스 임포트
``` python
# 두 리스트 합치는 과정
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

fish_data = [[l, w] for l, w in zip(length, weight)]
fish_target = [1] * 35 + [0] * 14

from sklearn.neighbors import KNeighborsClassifier
# import한 클래스의 객체 만들기
kn = KNeighborsClassifier()
# 이 객체에 fishdata와 fishtarget을 전달하여 도미 찾는 기준을 학습시킨다.
kn.fit(fish_data, fish_target)
kn.score(fish_data, fish_target)

kn.predict([[30. 600]])

# 만약 근접 이웃 숫자를 바꾸고 싶을 때
# kn2 = KNeighborsClassifier(n_neighbors=49)
# 이 경우 49개 중 도미가 35라서 어떤 데이터를 넣든 도미로 판단할 것.
```
    - kn이라는 객체에 도미 찾는 기준을 학습시키는 것을 훈련training이라 한다.
    - 사이킷런에서는 fit() 메소드가 훈련을 한다.
    - 모델 평가는 score()로 한다. 0~1 사이의 값을 가짐. 정확도accuracy.
    - K-최근접 이웃 알고리즘은 주위 다른 데이터를 보고 다수를 차지하는 것을 정답으로 사용하는 것.
    - predict는 새로운 데이터의 정답을 예측한다.
    - fit() 메소드 처럼 리스트의 리스트로 전달해야 해서 2번 감쌌다.
    - 모든 데이터를 가져와서 각각의 거리에 대해 직선거리로 살피기 때문에, 메모리가 많이 필요하고 계산 시간이 많이 필요하다.
    - 무언가 훈련되는 건 없고, 모든 데이터를 갖고 있다가 새로운 데이터가 오면 가장 가까운 데이터를 참고하여 구분하는 것.
    - 기본 값은 5이다. 5개의 주변 데이터로 참고. 이를 변경할 수도 있다.
## 2-1. 훈련 세트와 테스트 세트
- 지도 학습과 비지도 학습
    - 지도학습supervised learning과 비지도학습unsupervised learning으로 나뉜다.
    - 지도학습 알고리즘을 훈련하려면 데이터와 정답이 필요하다. 데이터는 입력input, 정답은 타깃target이라고 하고 둘을 합쳐 훈련 데이터training data라고 하낟.
    - 비지도학습 알고리즘은 타깃 없이 입력 데이터만 사용. 정답이 없으므로 무언가 맞힐 수 없다. 대신 데이터 파악이나 변형에 도움을 준다.
    - 이 외에도 강화학습 알고리즘reinforcement learning이 있다. 알고리즘이 행동한 결과로 얻은 보상을 사용해 학습한다.
- 훈련 세트와 테스트 세트
    - 데이터와 타깃을 주고 훈련한 다음 같은 데이터로 테스트하면 맞히는 것이 당연하다.
    - 평가를 위해 또 다른 데이터를 준비하거나 이미 준비된 데이터 중에서 일부 떼어내어 활용하는 방법. 일반적으로 후자를 사용.
    - 훈련에 사용하는 것이 train set, 평가에 사용하는 것이 테스트 세트test set이다.
```python
# 생선 길이와 무게를 하나의 리스트로 담은 2차원 리스트 만들기.
fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1]*35 + [0]*14

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
# 이제 전체 데이터에서 처음 35개를 선택해야 한다. index와 slicing 사용.
train_input = fish_data[:35]
train_target = fish_target[:35]
test_input = fish_data[35:]
test_target = fish_target[35:]

kn = kn.fit(train_input, train_target)
kn.score(test_input, test_target)
# 정확도가 0이 나왔다.
```
- 샘플링 편향
     - 테스트 셋과 훈련 셋의 데이터가 골고루 섞여있어야 한다. 그렇지 않으면 sampling bias라고 한다.
     - 나누기 전에 데이터를 섞든지, 아니면 골고루 추출해야 한다.
- 넘파이
    - 파이썬의 배열array 라이브러리이다. 고차원 리스트를 만들고, 조작할 수 있는 도구들 제공.
```python
import numpy as np
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
# 넘파이는 배열 차원 구분 쉽도록 행과 열을 가지런히 출력한다.
# shape은 배열의 크기를 알려준다. (샘플 수, 특성 수)로 출력.
print(input_arr.shape)
# 여기서는 배열에서 무작위로 샘플을 고를 것. input_arr의 행과 target_arr의 행이 같이 움직여야 한다. 따라서 구분지을 인덱스 값을 잘 기억해두어야 함.
# arrange() 함수는 0부터 1씩 증가하는 인덱스를 만들 수 있다. 그 후 랜덤하게 섞는다.
# 넘파이에서 무작위 함수들은 실행 시마다 다른 결과를 만든다. 일정한 결과를 얻으려면 초기에 랜덤 시드를 지정하면 된다.
# 이 초깃값이 같으면 동일한 난수를 뽑을 수 있다.
np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)
# 랜덤하게 섞인 인덱스를 활용해 나눈다.
# 넘파이 에서는 배열 인덱싱 기능 제공. 여러 개 인덱스로 한 번에 선택 가능. 또한 배열도 인덱스로 전환 가능.
# shuffle은 다차원 배열일 경우 첫 번째 축(행)에 대해서만 섞는다.
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]
# 잘 섞여 있는지 산점도를 통해 확인
import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```
- 두번째 머신러닝 프로그램
```python
# fit() 메소드를 실행할 때마다 KNeighborsClassifier 클래스 객체는 이전에 학습한 것을 잃어버린다. 이전 모델 그대로 두고 싶다면
# KNeighborsClassifier 클래스 객체를 새로 만들어야 한다.
kn = kn.fit(train_input, train_target)
kn.score(test_input, test_target)
# 코랩은 각 셀에서 마지막 코드 결과를 자동으로 출력하기 때문에 print()함수 사용하지 않아도 된다.
```
## 2-2. 데이터 전처리
- 넘파이로 데이터 준비하기
```python
# 이전에는 파이썬 리스트를 순회하며 하나씩 꺼내서 리스트 내 리스트로 직접 구성. 넘파이로 더 쉽게 가능.
import numpy as np
# 리스트를 일렬로 세우고 차례대로 나란히 연결.
fish_data = np.column_stack((fish_length, fish_weight))
# 넘파이 배열을 출력하면 리스트처럼 한 줄로 길게 나오지 않고 행렬이 맞춰져서 가지런히 나옴.
# 원하는 개수만큼 1이나 0을 채우는 np.ones(), np.zeros() 함수도 있다.
# stack말고 첫 번째 차원 따라서 배열하는 np.concatenate() 함수 사용할 것.
fish_target = np.concatenate((np.ones(35), np.zeros(14))
```
- 사이킷런으로 훈련 셋과 테스트 셋 나누기
```python
# train_test_split() 함수는 전달되는 리스트/배열을 비율에 맞게 나눠준다. 나눠주기 전에 섞어준다.
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)
# 기본적으로는 25%를 테스트 세트로 떼어낸다.
# 그러나 클래스 개수가 적은 지금 같은 경우, 샘플링 편향이 나타났다. 이를 해결 가능.
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)
```
- 수상한 도미 한 마리
```python
# 새로 준비한 데이터로 K-최근접 이웃 훈련.
from sklearn.neighbors import KNeighborClassifier
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)
# 팀장이 준 빙어를 넣어봄.
print(kn.predict([[25, 150]]))
# 도미인데 빙어라고 나왔다. 왜 그런지 산점도로 확인해본다.
import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:, 1])
plt.scatter(25, 150, marker='^') #marker은 매개변수 모양을 지정한다.
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 이 방법은 가장 가까운 5개 이웃의 클래스로 확인한다.
distances, indexes = kn.kneighbors([[25, 150]])
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes, 0], train_input[indexes,1], marker='D') # D는 산점도로 마름모로 나옴.
plt.xlabel('length')
plt.ylabel('weight')

```
- 기준을 맞춰라
    - x축과 y축의 범위가 달라서 생긴 문제.
```python
# 범위를 x, y 축 동일하게 0~1000으로 맞추기
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes, 0], train_input[indexes,1], marker='D')
plt.xlim((0,1000))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# 산점도가 거의 일직선으로 나타난다. 두 특성의 값이 놓인 범위가 너무 다르다. scale이 다른 것.
# 기준이 다르면 특히 거리 기반 알고리즘에서는 올바르게 예측하기 힘들다.
# 이러한 특성값을 일정한 기준으로 맞추는 것이 데이터 전처리data preprocessing

# 가장 널리 사용하는 전처리 방법 중 하나는 표준점수standard score(혹은 z 점수)
# 특성값이 0에서 표준편차 몇 배 만큼 떨어져 있는지.
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0) # 특성마닥 값의 스케일이 다르므로, 평균/표준편차는 각 특성 별 계산해야 한다. 그래서 axis=0으로.
# 표준점수 변환
train_scaled = (train_input - mean) / std
# 넘파이는 모든 행에서 mean의 값들을 다 빼준다. 브로드캐스팅broadcasting 기능.
```
- 전처리 데이터로 모델 훈련하기
    - 이 데이터로 다시 산점도를 그리니, 수상한 샘플 하나만 덩그러니 떨어져 있다.
    - 훈련 세트를 평균으로 빼고 std로 나눠져서 값의 범위가 달라졌기 때문. 샘플도 동일한 작업을 해줘야 한다.
    - 이 작업은 동일하게 훈련세트의 mean, std를 이용해야 함.
```python
new = ([25, 150] - mean) / std
kn.fit(train_scaled, train_target)
# 테스트 세트도 마찬가지로 훈련 세트의 mean, std로 변환해야 한다.
test_scaled = (test_input - mean) / std
kn.score(test_scaled, test_target)
print(kn.predict([new]))
```




