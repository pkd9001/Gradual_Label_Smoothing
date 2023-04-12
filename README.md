# Gradual_Label_Smoothing
## 2022년 추계학술대회 산업공학회 발표
### 링크 : https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE11172412&googleIPSandBox=false&mark=0&ipRange=false&accessgl=Y&language=ko_KR&hasTopBanner=true
### Requirements
* python >= 3.7
* pytorch >= 1.12.0
* transformers >= 4.24.0
* numpy
* pandas
* tqdm

### 1. 데이터
* 링크 : https://gluebenchmark.com/tasks
* 적용 데이터 셋 : **GLUE**(General Language Understanding Evaluation)

  * single-sentence tasks
    * SST-2 (Stanford Sentiment Treebank)
  * similarity and paraphrase tasks
    * MRPC (Microsoft Research Paraphrase Corpus)
    * QQP (Quora Question Pairs)
  * inference tasks
    * MNLI (Multi-Genre Natural Language Inference)
    * QNLI (Question Natural Language Inference)
    * **RTE** (Recognizing Textual Entailment, 업로드 데이터)

![gls 1](https://user-images.githubusercontent.com/100681144/231469975-65868513-12eb-4438-9257-240442c18af5.PNG)

### 2. Label Smoothing

* Szegedy et al(2016)의 "Rethinking the Inception Architecture for Computer Vision"에서 처음 제안됨
  * Model Calibration 기법중의 하나이며 Hard Target을 Soft Target으로 바꿔 줌
  * Mislabel Data를 고려하기 때문에 Model Generalization과 Calibration에 도움이 됨
  * Label을 확률기반으로 Smoothing하여, Hard Target이 가질 수 있는 편향성(극단성)을 극복하고자 나온 기법
  * Label Smoothing은 올바른 클래스의 Logit과 잘못된 클래스의 Logit의 차이를 α(0.0~1.0)에 따라 상수가 되도록 만듦
  * α = 0.0일 때, Cross Entropy Loss와 같은 결과값이 나옴

### 3. Gradual Label Smoothing

* overview

![gls 2](https://user-images.githubusercontent.com/100681144/231473840-31949821-6910-44eb-b3e5-cdb1a40652d7.PNG)

* 본 연구에서는 기존의 Label Smoothing의 α값이 고정되어 있는 것을 epoch이 진행됨에 따라, 변화하도록 제안

### 4. 적용 Model
* BERT-base
  * Layer = 12
  * Hidden size = 768
  * Attention heads = 12
  * Total Parameters : 110M

### 5. 결과

* Baseline vs LS vs GLS
  * Baseline : Cross-Entropy를 적용한 BERT-base
  * LS : Label Smoothing을 적용한 BERT-base
  * GLS : Gradual Label Smoothing을 적용한 BERT-base

![image](https://user-images.githubusercontent.com/100681144/231479545-1a6cdb16-6081-4121-a0dd-efab578337f9.png)

* 기존의 Label Smoothing과의 비교를 통해 비슷하거나 때때로 좋은 성능을 보이는 것이 확인되었음
* 비젼 분야에 자주 사용되는 Label Smoothing을 자연어 처리 분야에 적용하는 것이 기존의 방법(CE 사용)보다 더 나은 결과를 보여줌

