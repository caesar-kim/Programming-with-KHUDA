# 1주차. 교재1-10장. 데이터 탐색과 시각화 p.121.~p.202.
	가치 있는 데이터로 만드는 과정. 많은 시간이 소요된다.

## 10-1. 탐색적 데이터 분석 (Exploratory Data Analysis, EDA)
- 가공하지 않은 원천 데이터를 있는 그대로 탐색하고 분석하는 기법.
- 극단적 해석, 지나친 추론, 자의적 해석은 지양해야 함.
- EDA의 주요 목적  
  	1\. 데이터 형태와 척도가 분석에 알맞게 되어있는지(sanity checking)  
  	2\. 평균, 분산, 분포, 패턴 등 확인하여 데이터 특성 파악  
  	3\. 결측값, 이상치 파악 및 보완  
  	4\. 변수 간 관계성 파악  
  	5\. 분석 목적과 방향성 점검 및 보정  

### 1. 엑셀을 활용한 EDA
- 가장 간단하며 효과적인 방법은 샘플 1,000개를 뽑아서 엑셀로 변수 별 살펴보는 것.
- 대략적인 파악 가능
- 피봇Pivot 테이블 통해서 항목별 평균이나 분포 등 확인 가능. 그래프로 직관적으로 볼 수도 있음.
- 적은 데이터 다룰 때 엑셀만큼 효과적인 것은 없음.
  
### 2. 탐색적 데이터 분석 실습
```python
# 패키지 임포트
# 시각화
import seaborn as sns
import matplotlib.pyplot as plt
# 전처리용
import pandas as pd
sns.set(color_codes=True)
%matplotlib inline

# 데이터 불러오기
# https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
df = pd.read_csv("datasets/hotel_booking.csv")
# 데이터 샘플 확인
df.head()
# 각 칼럼 속성 및 결측치 확인
# 이 단계에서 숫자형, 문자형 등 올바른지 확인. 그리고 column별 결측치 확인. 표본 제거나 대치 방법 등 사용하여 처리
df.info()
# column별 통계치 확인
# 평균, 표준편차, 최대, 최솟값 등 확인 가능. 숫자형이지만 문자형과 다름없는 데이터는 빈도 등의 방법을 사용해야 함.
df.describe()
# 왜도(skewness) 및 첨도(kurtosis) 확인
# 변수 값 분포에서 정규성이 필요한 경우, 로그변환, 정규화, 표준화 등 방법 사용.
df.skew()
df.kurtosis()

# 특정 변수 분포 시각화
# 확인하려는 column의 분포를 시각화 한다.
sns.distplot(df['lead_time'])
# 그룹(호텔) 구분에 따른 lead_time 분포 시각화
# violinplot은 분포를 효과적으로 표현, stripplot은 각 관측치 위치를 직관적으로 표현해줌. 동시에 사용하여 보다 잘 이해 가능.
sns.violinplot(x="hotel", y="lead time", data=df, inner=None, color=".8")
sns.stripplot(x="hotel", y="lead time", data=df, size=1)
```
## 10-2. 공분산과 상관성 분석

### 1. 공분산
### 2. 상관계수
### 3. 공분산과 상관성 분석 실습

## 10-3. 시각화

### 1.0. 시간 시각화
### 1.1. 시간 시각화 실습
### 2.0. 비교 시각화
### 2.1. 비교 시각화 실습
### 3.0. 분포 시각화
### 3.1. 분포 시각화 실습
### 4.0. 관계 시각화
### 4.1. 관계 시각화 실습
### 5.0. 공간 시각화
### 5.1. 공간 시각화 실습
### 6.0. 박스 플롯
### 6.1. 박스 플롯 실습


