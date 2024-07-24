# 1주차. 교재1-10장. 데이터 탐색과 시각화 p.121.~p.202.
- 가치 있는 데이터로 만드는 과정. 많은 시간이 소요된다.

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
- Y와 X  관계는 물론, X들 사이의 관계도 살펴봐야 한다.  
&nbsp;&nbsp; X변화에 다른 Y 변화량을 크게 하여 통계적 정확도를 감소시키는 다중공선성도 방지 가능.  
&nbsp;&nbsp; 상관 분석 하기 위한 가정은 데이터가 등간이나 비율 척도이며, 두 변수가 선형적 관계라는 것.   

### 1. 공분산(Covariance)
- X<sub>1</sub>의 각 변수와 평균의 차이인 X<sub>1</sub> 편차와  X<sub>2</sub> 편차를 모두 곱해서 더해준 후 전체 개수 n(표본은 n-1)로 나누어준다.
- 두 변수의 공통적인 분산 정도를 알 수 있다.
- 양수는 양의 상관관계, 음수면 음의 상관관계, 0은 상관관계 없음, +1&-1은 직선 관계.
### 2. 상관계수(Correlation coefficient)
- 공분산의 절대적인 크기가 상관성의 정도를 나타내지 못한다. 그래서 공분산을 변수 각각의 표준편차 값으로 나누는 정규화를 통하여 상관성을 비교하기도 함.
- 이 또한 절대적인 기준이 될 수 없어서, 피어슨 상관계수(Pearson)을 주로 사용한다.
&nbsp;&nbsp; 변수의 공분산을 각각의 변수가 변하는 전체 정도로 나눠준 것. 함께 변하는 정도가 전체 변하는 정도를 초과할 수는 없기 때문에 -1에서 1 사이의 값을 가진다.
&nbsp;&nbsp; 일반적인 사회과학에서는 0.7 이상이면 매우 높은 상관관계, 0.4 이상이면 어느정도 상관관계 있다고 판단.
- 뒤에 나오는 회귀분석과도 연관 있다. 상관계수 제곱한 값을 결정계수라 하는데, 총 변동 중에서 회귀선에 의해 설명되는 변동이 차지하는 비율을 뜻함.
- 또한 선형관계만 측정 가능하기 때문에 2차 방정식 그래프와 비슷하 ㄴ모양이 되면 매우 낮게 측정 될 가능성도 있다. 이상치에 의해서도 달라질 수 있음. 상관계수 숫자만 보기보다는 산점도와 함께 보는 것이 좋음.
- 분석 프로그램으로 상관분석 실시하면, 표나 히트맵이 나온다. 동일한 변수끼리는 1이 나오고, 유의확률도 같이 표기됨.
### 3. 공분산과 상관성 분석 실습
- 자연수 형태만 이루어진 변수는 줄무늬 형태의 산점도를 보임.
- kiag_kind='kde' 옵션은 동일한 변수의 산점도 분포로 표현해주는 기능. 동일한 변수는 직선으로만 나오기 때문에 변환해주는 것.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# 데이터 불러오기
df = pd.read_csv("datasets/wine-quality.csv")
# 산점도 행렬 시각화
sns.set(font_scale=1.1)
sns.set_style('ticks')	# 축 눈금 설정
sns.pairplot(df, diag_kind='kde') # 상관계수가 1이면 분포로 표시
plt.show()

# 공분산 확인
df.cov()
# 피어슨 상관계수 확인
df.corr(method='pearson')
# 이 두 함수는 알아서 문자형 변수는 제외하고 계산해줌.
# 단, 고유번호 같이 의미가 없는 숫자인 경우에는 가독성을 위해 drop() 함수로 사전에 제거 하는 것이 좋음.

# 히트맵 시각화
sns.heatmap(df.corr(), cmap='viridis')
# 노란색일 수록 양의 상관관게, 보라색일수록 음의 상관관계

# 상관계수까지 적힌 clustermap 히트맵 시각화
sns.clustermap(df.corr(), annot=TRUE, cmap='RdYlBu_r', vmin=-1, vmax=1,)
# 분석 결과를 설명하는 경우, 중복된 영역을 지우는 것이 가독성을 높일 수 있다.
```
## 10-3. 시각화
### 1.0. 시간 시각화
- 시점이 있는 데이터는 시계열 형태(Time Series)로 표현 가능. 시간 흐름에 따른 데이터 변화 표현.  
- 연속형(선 그래프): 시간 간격 밀도가 높을 때 사용.  
&nbsp;&nbsp; 데이터 양이 너무 많거나 변동이 심하면, 추세선 삽입 가능.  
&nbsp;&nbsp; 추세선 가장 일반적인 방법은 이동평균Moving Average 방법.  
- 분절형(막대그래프, 누적 막대그래프, 점 그래프 등): 시간 밀도가 낮을 때 좋은 방법.  
&nbsp;&nbsp; 값들의 상대적 차이 나타낼 때 유리. 막대에 색상 표현하여 정보 추가도 가능.  
&nbsp;&nbsp; 누적 막대그래프는 한 시점에 2개 이상 세부 항목이 있을 때. 비율도 확인 가능.  
### 1.1. 시간 시각화 실습
```python
import matplotlib.pyplot as plt
import pandas as pd
# 날짜 가공에 쓸 패키지
import datetime
df=pd.read_csv("datasets/superstore.csv")

# date column 날짜 형식 변환
df['Date2']=pd.to_datetime(df['Order Date'], infer_datetime_format=True
# 날짜 오름차순 정렬
df=df.sort_values(by='Date2')
# 연도 칼럼 생성
df['Year']=df['Date2'].dt.year

# 선그래프용 데이터셋 생성
# 2018년 데이터만 필터링
df_line=df[df.Year==2018]
# 2018년 일별 매출액 가공
df_line=df_line.groupby('Date2')['Sales'].sum().reset_index()
df_line.head()

# 30일 이동평균 생성 (rolling 함수로 Month column을 새로 만들어주는 것)
df_line['Month']=df_line['Sales'].rolling(window=30).mean()
# 선 그래프 시각화
ax=df_line.plot(x='Date2', y='Sales', linewidth="0.5")
df_line.plot(x='Date2', y='Month', color='#FF7F50', linewidth="1", ax=ax)

# 연도별 판매량 데이터 가공
df_bar_1 = df.groupby('Year')['Sales'].sum().reset_index()
# 연도별 매출액 막대그래프 시각화
ax = df_bar_1.plot.bar(x='Year', y='Sales', rot=0, figsize=(10,5))
# 만약 x축 레이블이 길어서 글자가 겹치면 rot 옵션으로 글자 각도를 바꿔줄 수 있다.

# 연도별, 고객 세그먼트 별 매출액 데이터 가공
df_bar_2 = df.groupby(['Year', 'Segment'])['Sales'].sum().reset_index()
# 고객 세그먼트를 column으로 피봇
df_bar_2_pv = df_bar_2.pivot(index='Year', columns='Segment', values='Sales').reset_index()
# 연도별 고객 세그먼트 별 매출액 누적 막대그래프 시각화
df_bar_2_pv.plot.bar(x='Year', stacked=True, figsize=(10,7))
```
### 2.0. 비교 시각화
- 그룹 별 차이 나타낼 때는 (누적) 막대그래프로도 충분.
- 1\) 그러나 그룹별 요소가 늘어나면 히트맵 차트 활용 가능.
- 각 그룹이 어떤 요소에서 높거나 낮은 값을 가지는지 파악 가능. 요소 간 관계도 파악 가능.
- 단, 그리는 방법이 까다로워서 현재 가진 데이터 구조와 자신이 확인하려는 목적을 먼저 정확히 파악해야 함.
- 변수나 분류 그룹이 너무 많으면 혼란 가능하여 적정한 수준으로 데이터 정제 작업 필요.
- 2\) 또 다른 방법으로는 방사형(Radar) 차트가 있다.
- 3\) 평행 좌표 그래프Parallel coordinates를 통한 그룹별 요소 시각화도 가능.
&nbsp;&nbsp; 보다 효과적으로 표현하려면 변수별 값을 0~100% 사이로 정규화 하면 된다.
### 2.1. 비교 시각화 실습
```python
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import seaborn as sns
import numpy as np
from math import pi
from pandas.plotting import parallel_coordinates
df = pd.read_csv("datasets/nba2021_advanced.csv")

# 히트맵 시각화를 V1을 위한 데이터 전처리
# 5개 팀만 필터링
df1 = df[df['Tm'].isin(['ATL', 'BOS', 'BRK', 'CHI', 'CHO'])]
# 6개 열만 필터링
df1 = df1[['Tm', 'ORB%', 'TRB%', 'AST%', 'BLK%', 'USG%']]
# 팀별 요소 평균 전처리
df1 = df1.groupby('Tm').mean()
df1.head()

# 히트맵 시각화 V1
fig = plt.figure(figsize=(8, 8))
fig.set_facecolor('white')
plt.pcolor(df1.values)

# x축 column 설정
plt.xticks(range(len(df1.columns)), df1.columns)
# y축 column 설정
plt.yticks(range(len(df1.index)), df1.index)
# x축, y축 레이블 설정
plt.xlabel('Value', fontsize=14)
plt.ylabel('Team', fontsize=14)
plt.colorbar()
plt.show()

# 하나의 변숫값에 대한 히트맵 시각화를 위한 데이터 전처리
# 히트맵 시각화 V2를 위한 데이터 전처리

# 팀 5개만 필터링
df2 = df[df['Tm'].isin(['ATL', 'BOS', 'BRK', 'CHI', 'CHO'])]
# 팀명, 연령, 참여 게임 수 열만 필터링
df2 = df2[['Tm', 'Age', 'G']]
# 팀 - 연령 기준 평균으로 전처리
df2 = df2.groupby(['Tm', 'Age']).mean().reset_index()
# 테이블 피벗
df = df2.pivot(index='Tm', colums='Age', values='G')
df2.head()

# 히트맵 시각화 V2
fig = plt.figure(figsize = (8, 8))
fig.set_facecolor('white')
plt.pcolor(df2.values)
# 축 칼럼 설정
plt.xticks(range(len(df2.columns)), df2.columns)
plt.yticks(range(len(df2.index)), df2.index)
# 축 레이블 설정
plt.xlabel('Age', fontsize=14)
plt.ylabel('Team', fontsize=14)
plt.colorbar()
plt.show()

# 방사형 차트 - 하나씩 시각화
labels = df3.column[1:]
num_labels = len(labels)
# 등분점 생성
angles = [x/float(num_labels)*(2*pi) for x in range(num_labels)]
angles += angles[:1] # 시작점 생성
my_palette = plt.cm.get_cmap("Set2", len(df3.index))
fig = plt.figure(figsize= (15, 20))
fig.set_facecolor('white')

for i, row in df3.iterrows():
        color = my_palette(i)
        data = df3.iloc[i].drop('Tm').tolist()
        data += data[:1]
        ax = plt.subplot(3, 2, i+1, ploar=True)
        ax.set_theta_offset(pi / 2) # 시작점 설정
        ax.set_theta_direction(-1) # 시계방향 설정

        # 각도 축 눈금 생성
        plt.xticks(angles[:-1], labels, fontsize=13)
        # 각 축과 눈금 사이 여백 생성
        ax.tick_params(axis='x', which='major', pad=15)
        # 반지름 축 눈금 라벨 각도 0으로 설정
        ax.set_rlabel_position(0)
        # 반지름 축 눈금 설정
        plt.yticks([0, 5, 10, 15, 20], ['0', '5', '10', '15', '20'], fontsize=10)
        plt.ylim(0,20)
        ax.plot(angles, data, color=color, linewidth=2, linestyle='solid')        # 방사형 차트 출력
        ax.fill(angels, data, color=color, alpha=0.4)        # 도형 안쪽 색상 설정
        plt.title(row.Tm, size=20, color=color, x=-0.2, y=1.2, ha='left')        # 각 차트의 제목 생성
plt.tight_layout(pad=3)        # 차트 간 간격 설정
plt.show()
```
- for문으로 각 팀에 대한 그래프를 그려야 해서 복잡한 편이다. 데이터 값 범위에 따라 반지름 축 눈금 수치를 잘 조절해야 함.
``` python
# 방사형 차트를 하나로 시각화
labels = df3.colums[1:]
num_labels = len(labels)
angels = [x/float(num_labels)*(2*pi) for x in range(num_labels)]  # 등분점 생성
angles += angles[:1]    # 시작점 생성

my_palette = plt.cm.get_cmap("Set2", len(df3.index))
fig = plt.figure(figsize=(8, 8))
fig.set_facecolor('white')
ax = fig.add_subplot(polar=True)
for i, row in df3.iterrows():
    color = my_palette(i)
    data = df3.iloc[i].drop('Tm').tolist()
    data += data[:1]

    ax.set_theta_offset(pi / 2)    # 시작점
    ax.set_theta_direction(-1)    # 시계 방향 설정
    plt.xticks(angels[:-1], labels, fontsize=13)     # 각도 축 눈금 생성
    ax.tick_params(axis='x', which='major', pad=15)    # 각 축과 눈금 사이 여백 생성
    ax.set_rlabel_position(0)    # 반지름 축 눈금 라벨 각도 0으로 설정
    plt.yticks([0, 5, 10, 15, 20], ['0', '5', '10', '15', '20'], fontsize=10)    # 반지름 축 눈금 설정
    plt.ylim(0, 20)
    ax.plot(angles, data, color=color, linewidth=2, linestyle-'solid', label=row.Tm)    # 방사형 차트 출력
    ax.fill(angels, data, color=color, alpha=0.4)    # 도형 안쪽 색상 설정
plt.legend(loc=(0.9, 0.9))
plt.show()

# 팀 기준 평행 좌표 그래프 생성
fig, axes = plt.subplots()
plt.figure(figsize=(16, 8))    # 그래프 크기 조정
parallel_coordinates(df3, 'Tm', ax=axes, colormap='winter', linewidth="0.5")
```
### 3.0. 분포 시각화
- 데이터가 처음 주어졌을 때 어떤 요소가 어떤 비율인지 확인 하는 매우 중요한 단계에서도 많이 사용됨.
- 연속형(양적 척도)인지, 명목형(질적 척도)인지에 따라 구분해서 그림.  
&nbsp;&nbsp; 양적 척도 - 막대그래프, 선그래프, 히스토그램  
&nbsp;&nbsp;&nbsp;&nbsp; 히스토그램이란 각 구간을 bin이라 하며 구간의 높이는 밀도density이다.  
&nbsp;&nbsp;&nbsp;&nbsp; 처음 20개 구간으로 세세하게 나누어 살펴보고, 시각적으로 정보 손실이 커지기 전까지 개수를 줄이면 된다.  
&nbsp;&nbsp; 질적 척도 - 파이차트, 도넛차트, 트리맵 차트, 와플차트  
&nbsp;&nbsp; &nbsp;&nbsp; 전체를 100%로 하여, 구성 요소의 분포를 면적(각도)로 표현한다. 수치도 함께 표시해주는 것이 좋음. 이 둘의 차이는 도넛 차트의 가운데가 비어있다는 점.  
&nbsp;&nbsp; &nbsp;&nbsp; 구성 요소가 복잡한 질적 척도라면 트리맵 차트 이용. 큰 사각형을 작은 사각형으로 쪼개서 위계구조 표현 가능.  
&nbsp;&nbsp; &nbsp;&nbsp; 유사한 차트로는 와플 차트인데, 일정한 네모난 조각으로 분포 표현하지만 위계구조는 표현 불가.  

### 3.1. 분포 시각화 실습
```python
!pip install plotly
!pip install pywaffle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
from pywaffle import waffle
df = pd.read_csv("datasets/six_countries_height_samples.csv")

# 기본 히스토그램 시각화
df1 = df[['height_cm']]    # 신장 칼럼만 필터링
plt.hist(df1, bins=10, label = 'bins=10')    # 10cm 단위로 히스토그램 시각화
plt.legend()
plt.show()

# 남성 여성 히스토그램 시각화
# 남성 여성 별도 데이터셋 생성
df1_1 = df[df['sex'].isin(['man'])]
df1_1 = df1_1[['height_cm']]
df1_2 = df[df['sex'].isin(['woman'])]
df1_2 = df1_2[['height_cm']]
# 10cm 단위로 남성, 여성 신장 히스토그램 시각화
plt.hist(df1_1, color = 'green', alpha = 0.2, bins=10, label='MAN', density=True)
plt.hist(df1_2, color = 'red', alpha = 0.2, bins=10, label='WOMAN', density=True)
plt.legend()
plt.show()

# 파이차트, 도넛차트 시각화를 위한 데이터 전처리
df2 = df[['country', 'height_cm']]
df2 = df2[df.height_cm >= 175]    # 키 175 이상만 추출
df2 = df2.groupby('country').count().reset_index()

# 파이차트 시각화
fig = plt.figure(figsize=(8, 8))    # 캔버스 생성
fig.set_facecolor('white')    # 캔버스 배경색 설정
ax = fig.add_subplot()    # 프레임 생성
# 파이차트 출력
ax.pie(df2.height_cm,
    labels=df2.country,    # 라벨 출력
    startangle = 0,     # 시작점 degree 설정
    counterclock = False    # 시계방향
    autopct = lambda p: '{:.1f}%'.format(p)     # 퍼센트 자릿 수 설정
    )
plt.legend()     # 범례 표시
plt.show()

# 도넛차트 시각화
# 차트 형태 옵션 설정
wedgeprops={'width':0.7, 'edgecolor':'w', 'linewidth': 5}
plt.pie(df2.height_cm, labels=df2.country, autopct='%.1f%%', startangle=90, counterclock=False, wedgeprops=wedgeprops)
plt.show()

# 트리맵 차트용 데이터셋 전처리
df3 = df[['country', 'sex', 'height_cm']]
df3 = df3[df.height_cm >= 175]
# 국가 , 성병 단위 신장 175cm 이상 카운팅
df3 = df3.groupby(['country', 'sex']).count().reset_index()
df3.head(10)

# 트리맵 차트 시각화
fig = px.treemap(df3, path=['sex', 'country'], values='height_cm', color='height_cm', color_continuous_scale='viridis')
fig.show()

# 와플차트 시각화
fig = plt.figure(FigureClass = Waffle, plots={111: {'values': df2['height_cm'], 'labels': ["{0} ({1})".format(n, v) for n, v in df2['country'].items()],
        'legend': {'loc': 'upper left', 'bbox_to_anchor': (1.05, 1), 'fontsize':8}, 'title': {'label': 'Waffle chart test', 'loc': 'left'}
        }
    },
    rows=10,
    figsize=(10, 10)
)
```
### 4.0. 관계 시각화
- 산점도scatter plot. 산점도는 단순해서 쉽게 이해하고 표현 가능. 점들의 분포, 추세를 통해 관계를 파악할 수 있다.
- 산점도 그릴 때는 극단치 줄이는 것이 좋음. 주요 분포 구간으로 압축하는 것이 시각화 효율에 도움.
- 데이터가 너무 많아 점들이 서로 겹치는 경우, 각각의 점에 투명도를 주어 밀도도 함께 표현 가능.  또는 구간을 나누어 빈도에 따른 농도나 색상 다르게 표현 가능.  
- 버블차트를 활용하면 세 가지 요소의 상관관계도 표현 가능. 버블에 색상, 농도 등까지 포함하면 4가지 요소도 표현 가능하지만 차트 해석이 어려워지는 문제.  
- 구현이 까다롭긴 하지만 애미메이션 요소로 시간에 따른 변화도 함께 표현 가능.  
&nbsp;&nbsp; 버블차트는 원의 면적도 함께 봐야 해서 관측치가 너무 많으면 정보 효율이 떨어짐. 100개가 넘는 관측치라면 데이터 축약하거나 다른 시각화 방법 사용.  
&nbsp;&nbsp; 버블차트 해석 시 원의 지름이 아닌 면적을 통해 크기를 판단해야 한다. 지름이 2배면 면적은 4배가 된다.
### 4.1. 관계 시각화 실습
```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
df = pd.read_csv("datasets/50_Startups.csv")

# 기본 산점도 시각화
plt.scatter(df['R&D Spend'], df['Profit'], s = 40, alpha = 0.4)
plt.show()
ax = sns.lmplot(x='R&D Spend', y='Profit', data=df)  # 산점도에 회귀선 추가

# 네 가지 요소의 정보를 포함한 산점도시각화
plt.scatter(df['R&D Spend'], df['Profit'], s=df['Marketing Spend']*0.001, c=df['Administration'], alpha=0.5, cmap=''Spectral')
plt.colorbar()
plt.show()
# 마케팅 비용은 버블의 크기로 표현하도록 함. 관리비용은 색상으로 표현(붉은색이 작고 푸른색이 큰 것.)
```
### 5.0. 공간 시각화
- 지리적 위치와 관련된 데이터라면 실제 지도 위 표현이 효과적. 위도, 경도 데이터를 지도에 매핑하여 시각적으로 표현한다. 구글의 GeoMap 이용하면 지명만으로도 공간 시각화 가능.
- 단순 이미지 뿐 아니라, 지도 확대하거나 위치 옮겨가는 등 인터랙티브한 활용 가능. 이를 활용할 수 있도록 거시적에서 미시적으로 진행되듯 스토리라인 잡고 시각화 적용하는 것이 좋음.
- 도트맵, 코로플레스맵, 버블맵, 컨넥션맵 등이 있다.  
&nbsp;&nbsp; 1\) 도트맵: 동일 크기의 점을 찍어 분포나 패턴 표현. 점 하나는 실제 1개일수도, 다른 단위일수도 있다. 정확한 값 전달을 위해 숫자도 같이 표기하기도 함.
&nbsp;&nbsp; 2\) 버블맵: 버블 차트를 그대로 지도에 옮긴 형태. 비율 비교에 효과적이지만, 너무 큰 버블이 다른 것을 가릴 수 있음.  
&nbsp;&nbsp; 3\) 코로플레스맵Choropleth map: 단계 구분도라고도 함. 색상, 음영을 달리해서 지역에 대한 시각화. 지역 크기가 단순히 커서 강조될 수도 있는 점을 유의해야 함.  
&nbsp;&nbsp; 4\) 커넥션맵/링크맵: 지도에 찍힌 점들을 연결해서 지리적 관계 표시. 연속적 연결로 경로도 표시 가능. 연결선의 분포나 집중도로, 지리적 관계 패턴 파악에 사용.  
- 이 외에도 플로우맵Flow, 카토그램, 지도 위에 바차트, 파이차트 등을 올려서 표현하는 법 등이 있다.
### 5.1. 공간 시각화 실습
```python
!pip install folium
import folium
form folium import Marker
from folium import plugins
from folium import GeoJson
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 서울 스타벅스 지점 데이터
df = pd.read_csv("datasets/Starbucks_Seoul.csv")
# 지역 구분을 위한 json 파일 불러오기
geo= "Destop/datasets/Seoul_Gu.json"

# 기본 지도 시각화(서울 위도, 경도 입력)
m = folium.Map(location=[37.541, 126.986], zoom_start=12)    # zoom_start옵션으로 처음 화면에서 얼마나 확대할지
m

# 지도 형태 변경
m = folium.Map(location=37.541, 126.986], tiles='Stamen Tone', zoom_start=12)

# 원하는 좌표에 반경(radius)  표시(남산)
folium.CircleMarker([37.5538, 126.9810], radius=50, popup='Laurelhurst Park', color = '#3246cc', fill_color='#3246cc'.add_to(m)
# 원하는 좌표에 포인트 표시(남산)
folium.Marker([37.5538, 126.9810], popup = 'The Waterfront').add_to(m)
m

# 지도의 옵션에는 Stamen Toner, Stamen Watercolor, Stamen Terrain, Cartodb Positron, Cartodb Dark_matter 등의 옵션이 있다.
# CircleMarker()와 Marker() 함수로 남산 위치에 원 표시와 포인트 그림 삽입했다.

# 서울 지도에 스타벅스 지점 수 시각화
m = folium.Map([37.541, 126.986], zoom_start=12, width="%100", height="%100")
locations = list(zip(df.latitude, df.longitude))
cluster = plugins.MarkerCluster(locations=locations, popups=df["name"].tolist())     # 각 지역별 지점 수를 숫자로 표현해준다. 확대하면 숫자들이 쪼개지고, 축소하면 합쳐짐.
m.add_child(cluster)
m

# 서울 지도에 도트맵 시각화
m = folium.Map([37.541, 126.986], zoom_start=12, width="%100", height="%100")
locations = list(zip(df.latitude, df.longitude))
for i  in range(len(locations)):
    folium.CircleMarker(location = locations[i], radius=1).add_to(m)
m

# 버블맵 시각화를 위한 데이터 가공
# 구별지점 수 집계 및 중심점 산출
df_m = df.groupby('gu_name').agg({'latitude':'mean', 'longitude':'mean', 'name':'count'}).reset_index()    # 구 이름 기준으로 위도와 경도의 평균을 한다.
df_m.head()

# 버블맵 시각화
# 기본 지도 생성
m = folium.Map(location = [37.541, 126.986], tiles='Cartodb Positron', zoom_start=11, width="%100", height="%100")
# 구별 구분선, 생상 설정
folium.Choropleth(geo_data=geo, fill_color="gray").add_to(m)    #geo_data는 앞에서 불러온 json 파일을 적용하는 것. 구별 경계선 설정.
# 버블맵 삽입
locations = list(zip(df_m.latitude, df_m.longitude))
for i in range(len(locations)):
    row = df_m.iloc[i]
    folium.CircleMarker(locations = locations[i], radius=float(row.name/2), fill_color = "blue").add_to(m) # radius는 버블 크기 설정
m
```
```python
# 미국 실업률 정보의 코로플레스맵 시각화를 위한 데이터, json 불러오기.
df2 = pd.read_csv("datasets/us_states_unemployment.csv")
# 주별 경계 json 파일
us_geo= 'datasets/folium_us-states.json'

# 미국 지도 시각화
m = folium.Map(location=[40, -98], zoom_stazrt=3, tiles="Cartodb Positron")
# 지도에 주 경계선, 실업률 데이터 연동
m.choropleth(geo_data=us_geo,     # json data
                data = df2,         # 실업률 data
                colums = ['Stat', 'Unemployment'],     # 연동할 칼럼 설정
                key_on = 'feature.id',     # json과 실업률 데이터를 연결할 키값 설정
                fill_color = 'YlGn', legend_name='실업률')
m

# 서울과 각국 수도 간 커넥션 맵 시각화
# 서울, 도쿄, 워싱턴, 마닐라, 파리, 모스크바 위경도 입력
source_to_dest = zip([37.541, 37.541, 37.541, 37.541, 37.541], [35.6804, 38.9072, 14.5995, 48.8566, 55.7558], [126.986, 126.986, 126.986, 126.986, 126.986], [139.7690, -77.0369, 120.9842, 2.3522, 37.6173])
fig = go.Figure()

## for 문으로 위경도 입력
for a, b, c, d in source_to_dest:
    fig.add_trace(go.Scattergeo(lat=[a, b], lon=[c, d], mode='lines', line=dict(width=1, color="red"), opacity = 0.5))    # opacity는 선 투명도
fig.update_layout(margin={'t':0, 'b':0, 'l':0, 'r':0, 'pad':0}, showlegend=False, geo=dict(showcountries=True)) # showcountries 는 국가 경게선
fig.show()
```
### 6.0. 박스 플롯
- 상자수염그림(Box and Whisker Plot)이라고도 불림. 네모 상자에 수염이 결합된 형태.
- 하나의 그림으로 양적 척도 데이터 분포 및 편향성, 평균, 중앙값 등 다양한 수치 볼 수 있음.
- 분포 형태를 쉽게 확인하거나, 카테거리별 분포 비교할 때도 유용. 작은 원으로 이상치 표현하기도 한다.
- 최솟값(제1사분위 - 1.5IQR), 제1사분위(Q1), 제2사분위(Q2, 중앙값), 제3사분위(Q3), 최댓값(제3사분위 + 1.5IQR) 이라는 5가지 수치를 담고 있다.
- 전체 데이터 50%를 포함하는 박스 부분은 정규분포 평균 중심 좌우 1시그마에 해당하는 관측치 양(68.27%)와 유사하다.
- 그리고 양쪽 수염 끝까지(99.30%)는 좌우 3시그마(99.73)%과 유사하다.
- 중앙값이 박스 중앙보다 낮은 위치에 있다면 데이터 분포는 오른쪽(높은 쪽)으로 치우쳐 있을 가능성 높음.
- 박스플롯 해석 시에 항상 데이터 분포도도 함께 떠올리는 습관이 필요.
### 6.1. 박스 플롯 실습
```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df = pd.read_csv("datasets/50_Startups.csv")

# 세로 박스 플롯
plt.figure(figsize = (8, 6))
sns.boxplot(y = 'Profit', data = df)
plt.show()
# 가로 박스 플롯
plt.figure(figsize = (8, 2))
sns.boxplot(x = 'Profit', data = df)
plt.show()

# state 구분에 따른 profit 박스 플롯 시각화
plt.figure(figsize=(8,5))
sns.boxplot(x="State", y="Profit", data=df)
plt.show()

# 평균, 데이터 포인트 포함한 박스 플롯 시각화
sns.boxplot(x="State", y="Profit", showmeans=True, boxprops={'facecolor':'None'}, data=df)
sns.stripplot(x='State', y='Profit', data=df, jitter=True, marker='0', alpha=0.5, color='black')
plt.show()
# 기본 옵션에서 표현되지 않는 평균값 위치와 실제 데이터 포인트들을 추가로 표기하는 작업.
```

