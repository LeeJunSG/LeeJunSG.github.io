## 9장

## Predicting Continuous Target Variables with Regression Analysis


회귀 모델은 대상 변수를 연속 척도로 예측하는 데 사용되므로 과학의 많은 질문을 해결하는 데 매력적이다.

- 변수 간의 관계 이해, 추세 평가 또는 예측과 같은 산업 분야의 응용

- 한 가지 예는 향후 몇 개월 동안 회사의 매출을 예측

  

회귀 모델의 주요 개념 및 주제

- 데이터세트 탐색 및 시각화
- 선형 회귀 모델 구현에 대한 다양한 접근 방식 살펴보기
- 이상값에 대해 강력한 회귀 모델 교육
- 회귀 모델 평가 및 일반적인 문제 진단
- 회귀 모델을 비선형 데이터에 맞추기



## Introducing linear regression

선형 회귀의 목표 : 하나 또는 여러 기능과 연속 대상 변수 간의 관계를 모델링하는 것

- 지도 학습의 다른 하위 범주인 분류와 달리 회귀 분석은 범주형 클래스 레이블이 아닌 연속적인 규모로 출력을 예측하는 것이 목표

  

## Simple linear regression

단순(단변량) 선형 회귀의 목표는 단일 특성(설명 변수, x)과 연속값 목표(반응 변수, y) 간의 관계를 모델링하는 것입니다. 설명 변수가 하나인 선형 모델의 방정식은 다음과 같이 정의됩니다.

![image-20221103155919473](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221103155919473.png)

여기서 매개변수(편향 단위) b는 y축 절편을 나타내고 w1은 설명변수의 가중치 계수를 나타냅니다. 우리의 목표는 설명 변수와 목표 변수 사이의 관계를 설명하기 위해 선형 방정식의 가중치를 배우는 것입니다. 그러면 훈련 데이터 세트의 일부가 아닌 새로운 설명 변수의 응답을 예측하는 데 사용할 수 있습니다.

앞에서 정의한 선형 방정식을 기반으로 선형 회귀는 그림 9.1과 같이 훈련 예제를 통해 가장 적합한 직선을 찾는 것으로 이해될 수 있습니다.



![image-20221102192917624](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221102192917624.png)

이 가장 잘 맞는 선은 회귀선이라고도 하며 회귀선에서 훈련 예제까지의 수직선은 소위 오프셋 또는 잔차(예측 오류)입니다.



## Multiple linear regression

선형 회귀 모델을 여러 설명 변수로 일반화하는 것

- 다중 선형 회귀 :   

  ![그림입니다. 원본 그림의 이름: CLP000045fc0001.bmp 원본 그림의 크기: 가로 429pixel, 세로 56pixel](file:///C:\Users\Juns\AppData\Local\Temp\Hnc\BinData\EMB000045fc3863.bmp)

  다음 그림은 두 가지 기능이 있는 다중 선형 회귀 모델의 2차원 적합 초평면

  

![그림입니다. 원본 그림의 이름: CLP000045fc0002.bmp 원본 그림의 크기: 가로 548pixel, 세로 368pixel](file:///C:\Users\Juns\AppData\Local\Temp\Hnc\BinData\EMB000045fc3867.bmp)  



 3차원 산점도에서 다중 선형 회귀 초평면의 시각화는 정적 수치를 볼 때 해석하기 어려움

- 산점도에서 2차원 초평면을 시각화하는 좋은 방법이 없음
- 따라서, 시각화는 주로 단순 선형 회귀를 사용하여 일변량 사례에 중점을 둠
- 단순 및 다중 선형 회귀는 동일한 개념과 동일한 평가 기술을 기반으로 함



## Loading the Ames Housing dataset into a DataFrame



pandas read_csv 함수를 사용한 Ames Housing 데이터 세트를 로드

- Ames Housing 데이터셋은 2,930개의 예시와 80개의 기능으로 구성되어 있음

- 단순화를 위해 다음 목록에 표시된 기능의 하위 집합으로만 작업함

  

대상 변수를 포함하여 작업할 기능

- 전체 품질: 1(매우 나쁨)에서 10(우수)까지의 척도로 집의 전체 자재 및 마감에 대한 평가
- 전반적인 상태: 1(매우 나쁨)에서 10(매우 좋음)까지의 척도로 집의 전반적인 상태에 대한 평가
- Gr Liv Area: 평방 피트 단위의 지상(지상) 생활 공간
- 중앙 에어컨: 중앙 에어컨(N=아니오, Y=예)
- Total Bsmt SF: 지하 면적의 총 평방 피트
- SalePrice: 미국 달러($)로 표시된 판매 가격



사용되는 데이터 세트:

![image-20221103180421290](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221103180421290.png)



## Visualizing the important characteristics of a dataset

Exploratory data analysi(EDA)(탐색적 데이터 분석)은 기계 학습 모델을 훈련하기 전에 중요한 첫 번째 단계이다.



 이상값의 존재, 데이터 분포 및 기능 간의 관계를 시각적으로 감지하는 데 도움이 될 수 있는 그래픽 EDA 도구 상자를 활용한다.

먼저, 이 데이터셋의 서로 다른 피처 간의 쌍별 상관관계를 한 곳에서 시각화할 수 있는 산점도 행렬을 생성한다.



산점도 행렬을 플롯하기 위해 mlxtend 라이브러리의 scatterplotmatrix 함수를 사용

- Python의 기계 학습 및 데이터 과학 응용 프로그램에 대한 다양한 편의 기능을 포함하는 Python 라이브러리

- conda install mlxtend 또는 pip install mlxtend를 통해 mlxtend 패키지를 설치할 수 있음

- mlxtend 버전 0.19.0을 사용

  

산점도 행렬(scatterplot matrix)의 생성 예 : 

![image-20221103180829350](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221103180829350.png)

위의 그림의 산점도 행렬을 사용하여 이제 데이터가 어떻게 분포되어 있고 이상치가 포함되어 있는지 여부를 확인할 수 있다. 



## Looking at relationships using a correlation matrix



변수 간의 선형 관계를 수량화하고 요약하는 상관 행렬

- 상관 행렬을은공분산 행렬의 크기 조정 버전으로 해석할 수 있다.

- 상관 행렬은 표준화된 특성에서 계산된 공분산 행렬과 동일하다.

  

상관 행렬은 특성 쌍 간의 선형 종속성을 측정하는 Pearson 제품 모멘트 상관 계수(종종 Pearson의 r로 축약됨)를 포함하는 정방형 행렬임

- 상관 계수의 범위는 -1에서 1 사이
- 두 특성은 r = 1이면 완전한 양의 상관 관계가 있고 r = 0이면 상관 관계가 없으며 r = -1이면 완전한 음의 상관 관계임
- Pearson의 상관 계수는 x와 y(분자)의 두 특성 간의 공분산을 표준 편차(분모)의 곱으로 나눈 값으로 간단히 계산할 수 있음.

![image-20221103183021176](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221103183021176.png)



위의 식에서 𝜇는 해당 특징의 평균, 𝜎(𝑥𝑦)   는 특징 x와 y의 공분산, 𝜎(𝑥)와 𝜎(𝑦)는 특징입니다.



상관 행렬 배열을 히트 맵으로 플로팅 결과 :

![image-20221103184724731](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221103184724731.png)

- 선형 회귀 모델에 맞추기 위해 목표 변수인 SalePrice와 상관 관계가 높은 기능에 관심이 있음

- 이전 상관 행렬을 보면 SalePrice가 Gr Liv Area 변수(0.71)와 가장 큰 상관 관계를 나타내는 것을 알 수 있음

- 탐색 변수가 단순 선형 회귀 모델의 개념을 소개하는 데 좋음.

  

## Solving regression for regression parameters with gradient descent

2장, 분류를 위한 단순 기계 학습 알고리즘 훈련에서 적응 선형 뉴런(Adaline)의 구현을 고려하십시오. 인공 뉴런은 선형 활성화 함수를 사용한다는 것을 기억할 것입니다. 또한 GD(Gradient Descent) 및 SGD(Stochastic Gradient Descent)와 같은 최적화 알고리즘을 통해 가중치를 학습하기 위해 최소화한 손실 함수 L(w)을 정의했습니다.

Adaline의 이 손실 함수는 MSE(평균 제곱 오차)이며 OLS에 사용하는 손실 함수와 동일합니다.



![image-20221103190708822](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221103190708822.png)

위의 식의 y^는 예측값 𝑦^=w^(t)x입니다(1/2은 GD의 업데이트 규칙을 도출하기 위해 편의상 사용됨). 

- OLS 회귀는 임계값 함수가 없는 Adaline으로 이해될 수 있으므로 클래스 레이블 0과 1 대신 연속적인 목표 값을 얻음. 



Adaline의 GD 구현을 가져와서 임계값 함수를 제거하여 구현.

- 선형 회귀 모델:

![image-20221103194636053](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221103194636053.png)

데이터의 많은 경우에서 거주 지역의 크기가 주택 가격을 잘 설명하지 못한다는 것을 확인할 수 있다. 

## Estimating the coefficient of a regression model via scikit-learn



실제 응용 프로그램에서는 보다 효율적인 구현

- scikit-learn의 회귀 추정기 중 다수는 SciPy(scipy.linalg.lstsq)의 최소 제곱 구현을 사용함
- LAPACK(Linear Algebra Package) 기반으로 고도로 최적화된 코드 최적화를 사용함
-  scikit-learn의 선형 회귀 구현은 (S)GD 기반 최적화를 사용하지 않으므로 표준화되지 않은 변수에서도 작동하므로 표준화 단계를 건너뛸 수 있음

![image-20221103195415640](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221103195415640.png)

이 코드를 실행하면 알 수 있듯이 표준화되지 않은 Gr Liv Area 및 SalePrice 변수가 적용된 scikit-learn의 LinearRegression 모델은 기능이 표준화되지 않았기 때문에 다른 모델 계수를 산출한다. 

 Gr Liv Area에 대해 SalePrice를 플로팅하여 GD 구현과 비교할 때 데이터가 유사하게 잘 맞는다는 것을 확인할 수 있다.

## Fitting a robust regression model using RANSAC

선형 회귀 모델은 이상값의 존재로 인해 큰 영향을 받는다. 

- 특정 상황에서는 데이터의 아주 작은 부분 집합이 추정된 모델 계수에 큰 영향을 미침.
- 많은 통계 테스트를 사용하여 이상값을 감지할 수 있음(본 책의 범위를 벗어남). 
- 이상값을 제거하려면 항상 데이터 과학자로서의 판단과 도메인 지식이 필요함.

이상값을 없애는 대신 RANSAC(RANdom SAmple Consensus) 알고리즘을 사용하는 강력한 회귀 방법

- 이 알고리즘은 회귀 모델을 데이터의 하위 집합, 즉 inlier에 맞추는 것임.

  

반복적인 RANSAC 알고리즘의 요약

1. inlier가 될 예제의 난수를 선택하고 모델 피팅

2. 피팅된 모델에 대해 다른 모든 데이터 포인트를 테스트하고 사용자가 지정한 허용오차 내에 해당 포인트를 inlier에 추가

3. 모든 inlier를 사용하여 모델을 다시 맞추기.

4. 적합 모델 대 inlier의 오차를 추정

5. 성능이 특정 사용자 정의 임계값을 충족하거나 고정된 반복 횟수에 도달하면 알고리즘을 종료합니다. 그렇지 않으면 1단계를 반복



scikit-learn의 RANSACRegressor 클래스에서 구현된 RANSAC 알고리즘과 함께 선형 모델 :



![image-20221103201212204](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221103201212204.png)



- RANSACRegressor의 최대 반복 횟수를 100으로 설정하고 min_samples=0.95를 사용하여 무작위로 선택된 훈련 예제의 최소 횟수를 데이터 세트의 95% 이상으로 설정

- 기본적으로(residual_threshold=None을 통해) scikit-learn은 MAD 추정값을 사용하여 inlier 임계값을 선택함

  - 여기서 MAD는 목표 값 y의 절대 편차 중앙값을 나타냄
  - inlier 임계값에 대한 적절한 값의 선택은 문제에 따라 다르며, 이는 RANSAC의 한 가지 단점임

- 최근 몇 년 동안 좋은 inlier 임계값을 자동으로 선택하기 위해 다양한 접근 방식이 개발됨


  
  RANSAC 선형 회귀 모델에서 inlier와 outlier를 가져와 선형 피팅 : 

![image-20221103201237235](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221103201237235.png)

아래 그림에서 볼 수 있듯이 선형 회귀 모델은 감지된 inlier 세트에 맞춰졌으며 원으로 표시함



![image-20221103201301615](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221103201301615.png)

다음 코드를 실행하여 모델의 기울기와 절편을 인쇄할 때 선형 회귀선은 RANSAC를 사용하지 않고 이전 섹션에서 얻은 적합도와 약간 다름

![image-20221103201332902](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221103201332902.png)

잔류_임계값 매개변수를 없음으로 설정했기 때문에 RANSAC는 MAD를 사용하여 inlier 및 outlier에 플래그를 지정하기 위한 임계값을 계산함. 이 데이터 세트의 MAD는 다음과 같이 계산할 수 있다.

![image-20221103201354606](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221103201354606.png)

더 적은 수의 데이터 포인트를 이상값으로 식별하려는 경우 이전 MAD보다 큰 잔여 임계값 값을 선택할 수 있다.

아래 그림은 잔여 임계값이 65,000인 RANSAC 선형 회귀 모델의 이상값과 이상값 :

![image-20221103201528416](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221103201528416.png)



RANSAC를 사용하여 이 데이터 세트에서 이상값의 잠재적인 영향을 줄였지만 이 접근 방식이 보이지 않는 데이터에 대한 예측 성능에 긍정적인 영향을 미칠지 여부는 알 수 없다.



## Evaluating the performance of linear regression models



다중 회귀 모델을 훈련 : 

![image-20221103203226177](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221103203226177.png)

실제 값과 예측 값 간의 차이 또는 수직 거리를 그림.

- 예측값을 비교하여 회귀 모델을 진단함

잔차 도표는 회귀 모델을 진단하기 위해 일반적으로 사용되는 그래픽 도구이다. 비선형성 및 이상값을 감지하고 오류가 무작위로 분포되어 있는지 여부를 확인하는 데 도움이 될 수 있다.

잔차 플롯을 플로팅 :

![image-20221103203515826](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221103203515826.png)



아래 그림과 같이 x축 원점을 통과하는 선이 있는 테스트 및 교육 데이터 세트에 대한 잔차 플롯이 표시됨

![image-20221103203556343](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221103203556343.png)



완벽한 예측의 경우, 잔차는 정확히 0이 될 것이며, 이는 현실적이고 실용적인 응용에서 불가능하다



단, 좋은 회귀 모델의 경우 오류가 무작위로 분포되고 잔차가 중심선 주위에 무작위로 흩어질 것으로 예상된다.

- 잔차 그림에서 패턴이 보인다면 이전 모델이 잔차로 유출된 일부 설명 정보를 캡처할 수 없음을 의미함
- 중심선에서 큰 편차가 있는 점으로 표시되는 이상값을 감지하기 위해 잔차 그림을 사용할 수도 있음.
- 모델 성능의 또 다른 유용한 정량적 측정은 선형 회귀 모델에 맞게 최소화한 손실 함수로 앞에서 논의한 평균 제곱 오차(MSE)이다.

다음 식은 경사하강법에서 손실 도함수를 단순화하는 데 자주 사용되는 1/2 스케일링 계수가 없는 MSE :

![image-20221103203655804](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221103203655804.png)

교차 검증 및 모델 선택을 위해 MSE를 사용할 수 있다.
분류 정확도와 마찬가지로 MSE도 표본 크기 n에 따라 정규화합니다. 이를 통해 다양한 샘플 크기(예: 학습 곡선의 맥락에서)를 비교할 수도 있다.
이제 훈련 및 테스트 예측의 MSE를 계산해 보겠습니다.



훈련 데이터 세트의 MSE가 테스트 세트보다 크다. 이는 모델이 훈련 데이터에 약간 과적합되고 있음을 나타내는 지표이다. 원래 단위 척도에 오류를 표시하는 것이 더 직관적일 수 있다. 따라서 MSE의 제곱근을 계산하도록 선택할 수 있다. 잘못된 예측을 약간 덜 강조하는 평균 절대 오차(MAE):

![image-20221104061254911](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221104061254911.png)

테스트 세트 MAE를 기반으로 모델이 평균적으로 약 $25,000의 오차를 발생시킨다고 말할 수 있다.



모델 성능에 대한 더 나은 해석을 위해 MSE의 표준화된 버전으로 이해할 수 있는 결정 계수(R2)를 보고하는 것이 더 유용할 수 있다. 즉, R^(2) 는 모델에 의해 캡처된 응답 분산의 비율입니다. R^(2) 값은 다음과 같이 정의된다.



![image-20221104060935588](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221104060935588.png)



여기서 SSE는 제곱 오차의 합으로 MSE와 유사하지만 샘플 크기 n에 의한 정규화를 포함하지 않는다. 그리고 SST는 제곱의 총합이다.

- SST는 단순히 응답의 분산

  



## Using regularized methods for regression



정규화는 추가 정보를 추가하고 복잡성에 대한 패널티를 유도하기 위해 모델의 매개변수 값을 축소함으로써 과적합 문제를 해결하는 한 가지 접근 방식이다. 정규화된 선형 회귀에 대한 가장 인기 있는 접근 방식은 소위 능선 회귀, 최소 절대 수축 및 선택 연산자(LASSO) 및 탄성 망이다.



릿지 회귀는 MSE 손실 함수에 가중치의 제곱합을 단순히 추가하는 L2 패널티 모델이다.



![image-20221104062045700](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221104062045700.png)

- L2 항은 다음과 같이 정의됩니다.

![image-20221104062110119](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221104062110119.png)

하이퍼파라미터 𝜆𝜆의 값을 증가시켜 정규화 강도를 높이고 모델의 가중치를 줄인다. 바이어스 단위 b는 정규화되지 않는다. 



희소 모델로 이어질 수 있는 다른 접근 방식은 LASSO이다. 정규화 강도에 따라 특정 가중치가 0이 될 수 있으며, 이는 LASSO를 감독 기능 선택 기술로 유용하게 만든다.

![image-20221104062132736](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221104062132736.png)

- LASSO에 대한 L1 페널티는 다음과 같이 모델 가중치의 절대 크기의 합으로 정의됨.

![image-20221104062212349](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221104062212349.png)

 LASSO의 한계는 m > n인 경우 최대 n개의 특성을 선택한다는 것이다. 

- n은 학습 예제의 수입니다. 
- 기능 선택의 특정 응용 프로그램에서 바람직하지 않을 수 있음. 
- 실제로 LASSO의 이러한 속성은 포화된 모델을 피하기 때문에 종종 이점이 있음.
- 모델의 포화는 훈련 예제의 수가 특징의 수와 같을 때 발생함
- 일종의 과매개변수화(overparameterization). 
- 포화 모델은 항상 훈련 데이터에 완벽하게 맞을 수 있지만 보간 형식일 뿐이므로 잘 일반화되지 않음.
- 능선 회귀와 LASSO 간의 절충안은 m > n인 경우 n개 이상의 특징을 선택하는 데 사용할 수 있도록 희소성을 생성하기 위한 L1 페널티와 L2 페널티가 있는 탄력적 네트임

![image-20221104062244307](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221104062244307.png)



## Turning a linear regression model into a curve – polynomial regression

선형성 가정 위반을 설명하는 한 가지 방법은 다항식 항을 추가하여 다항식 회귀 모델을 사용하는 것이다.

![image-20221104061705885](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221104061705885.png)

- d는 다항식의 차수를 나타냄. 
- 다항식 회귀를 사용하여 비선형 관계를 모델링할 수 있지만 선형 회귀 계수 w로 인해 여전히 다중 선형 회귀 모델로 간주됨. 





## Decision tree regression

의사 결정 트리 알고리즘의 장점은 임의의 기능과 함께 작동하고 의사 결정 트리가 가중치 조합을 고려하지 않고 한 번에 하나의 기능을 분석하기 때문에 비선형 데이터를 처리하는 경우 기능의 변환이 필요하지 않다는 것이다. 

분류를 위해 결정 트리를 사용할 때 엔트로피를 불순도 측정으로 정의하여 정보 이득(IG)을 최대화하는 기능 분할을 결정합니다. 이는 이진 분할에 대해 다음과 같이 정의된다.

![image-20221104063041023](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221104063041023.png)

- x(i)는 분할을 수행하는 기능
- N(p)는 상위 노드의 학습 예제 수
- I는 불순 함수
- D(p)는 상위 노드의 학습 예제 하위 집합
- D(left) 및 D(right)는 학습 하위 집합

자식 노드의 불순물을 가장 많이 줄이는 기능 분할을 찾고 싶어함. 

회귀에 의사 결정 트리를 사용하려면 연속 변수에 적합한 불순물 메트릭이 필요하므로 노드 t의 불순물 측정을 대신 MSE로 정의한다.

![image-20221104063254974](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221104063254974.png)

여기서 Nt는 노드 t의 훈련 예제 수, Dt는 노드 t의 훈련 부분 집합, 𝑦𝑦(𝑖𝑖𝑖는 실제 목표 값, 𝑦𝑦̂𝑡𝑡은 예측 목표 값(샘플 평균)입니다.

![image-20221104063316620](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221104063316620.png)

의사 결정 트리 회귀의 맥락에서 MSE는 종종 노드 내 분산이라고 하며, 이것이 분할 기준이 분산 감소로도 더 잘 알려진 이유입니다.



리고 회귀 트리가 비선형 데이터의 추세를 비교적 잘 포착할 수도 있다. 그러나 이 모델의 한계는 원하는 예측의 연속성과 미분성을 포착하지 못한다는 점이다. 또한 데이터가 과대적합되거나 과소적합되지 않도록 적절한 트리 깊이 값을 선택하는 데 주의해야 한다.

![image-20221104063403514](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221104063403514.png)



## Random forest regression



랜덤 포레스트 알고리즘은 여러 의사 결정 트리를 결합하는 앙상블 기술이다. 랜덤 포레스트는 일반적으로 랜덤성으로 인해 개별 의사 결정 트리보다 일반화 성능이 더 우수하여 모델의 분산을 줄이는 데 도움이 된다. 랜덤 포레스트의 다른 장점은 데이터 세트의 이상값에 덜 민감하고 많은 매개변수 조정이 필요하지 않다는 것이다. 일반적으로 실험해야 하는 랜덤 포레스트의 유일한 매개변수는 앙상블의 트리 수이다. 회귀를 위한 기본 랜덤 포레스트 알고리즘은 랜덤 포레스트 알고리즘과 거의 동일하다. 유일한 차이점은 개별 결정 트리를 성장시키기 위해 MSE 기준을 사용하고 예측 대상 변수는 모든 의사 결정 트리에 대한 평균 예측으로 계산된다는 것이다. 



 y축 방향의 이상값으로 표시된 것처럼 모델이 테스트 데이터보다 훈련 데이터에 더 잘 맞는 것을 알 수 있다. 또한 잔차 분포가 0 중심점 주변에서 완전히 무작위로 나타나지 않는 것으로 보이며 이는 모델이 모든 탐색 정보를 캡처할 수 없음을 나타낸다. 그러나 잔차 그림은 이 장의 앞부분에서 그린 선형 모델의 잔차 그림보다 크게 개선되었음을 나타낸다.



![image-20221104063606026](C:\Users\Juns\AppData\Roaming\Typora\typora-user-images\image-20221104063606026.png)





