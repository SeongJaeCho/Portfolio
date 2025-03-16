## 소개 
안녕하세요! 데이터 사이언티스트 조성재입니다.
현재 서울대학교 통계학과 석사과정에 재학 중입니다.
머신러닝 및 딥러닝을 활용하여 데이터 기반 의사결정을 지원하고 AI 모델의 성능을 최적화하는데 관심이 있습니다. 특히, 대규모 언어 모델 튜닝, 이상치 합성 연구, 통계 기반 머신러닝 모델 개발을 수행하며 다양한 프로젝트 경험을 쌓았습니다.

---
## 주요 진행 프로젝트
### 1) LLM 기반 데이터 분석 AI Assistant 개발
**📌 프로젝트 개요**
- N사의 자체적인 대규모언어모델 HyperClovaX를 이용하여 **LLM 기반 데이터 분석 AI Assistant의 컨셉 데모를 제작**
- 자연어 기반 데이터 분석을 가능하게 하기 위해 **LLM을 튜닝할 데이터셋을 구축하고 AI의 분석 프로세스를 설계**
**🛠️ 수행 내용**
- 스마트 스토어 점주들이 활용할 수 있도록 **고객 구매 내역, 매출, 배송, 신규 고객 유입 경로 등의 데이터를 분석하는 AI 모델 개발**
- 3단계의 질문 및 응답 프로세스 설계 (데이터 탐색 → 트렌드 분석 → 비교 및 해석)
- 시각화 기법(막대그래프, 히스토그램, 파이 차트 등)을 활용하여 데이터 분석 결과를 비전공자도 직관적으로 이해할 수 있도록 답변 제작
**📊 주요 성과**
- 8주간 **1000개의 unique한 분석 질문을 제작**하여 모델 학습 및 튜닝 진행
- 튜닝된 모델이 새로운 질문에 대해 적절한 답변과 분석 프로세스를 생성하는 성과 달성
- 데모 버전 개발 성공
**주요 기술**
- statistics, matplotlib, seaborn, hyperparameter tunning

### 2) MC gradCAM
**📌 프로젝트 개요**
- 이미지 모델의 예측값에 대한 신뢰성을 분석하기 위한 기법으로 이미지 내 주목 영역을 시각화하는 기술인 GradCAM의 불확실성을 계측
**🛠️ 수행 내용**
  - 전이학습으로 사용 가능한 VGG16, InceptionV3 등의 FC layer 뒤에 MCdropout layer를 추가
  - 각 모델의 마지막 Convolution layer의 모델 파라미터를 이용하여 true class에 대한 GradCAM을 반복적으로 수집
  - 이미지별 GradCAM들의 Frobenius Norm을 계산하고 이들의 분포를 비교
  - 잘 분류하지 못하는 이미지에서의 분산이 큰 패턴을 발견
**📊 연구 의의**
  - 불확실성을 함께 고려하는 모델 진단 방법으로 더욱 정교한 모델 성능 진단 가능
**주요 기술**
- transfer learning, uncertainty

### 3) 주식 거래 데이터 기반 생존 분석 AI 모델 개발
**📌 프로젝트 개요**
- NH투자증권 빅데이터 경진대회에서 **주식 보유 기간을 예측하는 머신러닝 모델 개발**
- 데이터 분석 및 머신러닝을 활용하여 투자자 행동 예측

**🛠️ 수행 내용**
- 주식 거래 데이터를 기반으로 시점별 매도 발생 여부와 과거 보유기간을 반영한 생존 분석(Survival Analysis) 모델을 설계하여 **보유 기간 예측**
- 기존 생존분석 기법을 개선하여 **Gradient Boosted Model을 적용한 머신러닝 기반 생존 분석 모델 개발**

**📊 주요 성과**
- 844개 팀 중 **상위 50위 달성**
- 생존 분석을 머신러닝 기법과 결합하여 **일반적인 금융 데이터 예측보다 높은 성능을 기록**

**관련 링크**
- [https://dacon.io/competitions/official/235798/overview/description]

### 4) 이상치 합성 연구 (Manifold Hypothesis 및 Conformal Prediction 활용)
**📌 연구 개요**
- **Label이 있는 데이터셋에서 이상치를 합성하는 기법 연구**
- 생성모델(Generative Model)에서 가정하는 **Manifold Hypothesis를 기반으로 이상치를 탐지하고, 이를 Conformal Prediction을 활용하여 합성**

**🛠️ 수행 내용**
- Latent Space 상에서 Class별 분포를 분석하고 이상치 데이터를 판별 및 합성
- 기존 기법(VOS, DREAM)의 한계를 극복하기 위해 Coverage Guarantee를 활용한 이상치 합성 방법 고안

**📊 연구 의의**
- **해석, 통제 가능한 이상치 데이터 합성 기법을 개발**하여 기존 기법 대비 신뢰성을 확보
- AI 모델의 이상 탐지 및 일반화 성능 향상을 위한 데이터 증강 기법 연구
- 추후 업데이트 예정

---

## 3. 기술 스택

### 📌 프로그래밍 및 데이터 분석
- Python (Numpy, Pandas, Scikit-learn, Matplotlib, Seaborn)
- SQL (공부중)
- PyTorch, TensorFlow (딥러닝 모델 개발 및 학습)
- Jupyter Notebook, Google Colab (데이터 분석 및 모델 실험)

### 📌 머신러닝 & 딥러닝
- 지도학습 (Regression, Classification, Survival Analysis)
- 비지도학습 (Anomaly Detection, Clustering)
- 딥러닝 모델 (CNN, LLM)
- 이상치 탐지 및 합성 모델 연구 (VOS, DREAM, BCOPS) : 추후 업데이트 예정

### 📌 XAI
- GradCAM (CNN 모델 해석)

### 📌 Uncertainty
- Conformal prediction
---

## 4. 연락처
- 📧 이메일: sjcho9908@gmail.com
- 📍 GitHub: [https://github.com/SeongJaeCho]
- 📄 LinkedIn: [https://kr.linkedin.com/in/%EC%84%B1%EC%9E%AC-%EC%A1%B0-01628628a?trk=people-guest_people_search-card&original_referer=https%3A%2F%2Fwww.linkedin.com%2F]

---
