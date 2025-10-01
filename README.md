# ConfMaskFormer
신호 결측에 강건한 비컨 기반 실내 위치 추정 딥러닝 모델

### 데이터 수집 환경
- 장소: 인천대학교 정보기술대학 7호관 3층
- 구조: 직선형 복도(길이 70.6m, 폭 2.5m, 천장 높이 3.3m)
- 위치: 3m 간격의 24개 구역
- 비컨: 15m 간격의 5개 천장 설치
  
![Pipeline](img/map.png)


### 데이터 전처리(data_split.py)
BLE RSSI 시계열 CSV를 대상으로 5-분할 스플릿 → 글로벌 Min-Max 정규화(0=결측 보존) → 슬라이딩 윈도우 → .csv/.npz 저장
#### 입출력 개요
입력: ```./data/in-motion/*.csv```
출력
- 분할된 원본/정규화 CSV
     ```./data/data_split/{raw|norm}/<dataset>/<split>/<phase>/*.csv```
- 슬라이딩 윈도우 NPZ
     ```./data/data_split/npz/<dataset>/{train|test}_{split}.npz```
- ```<dataset>```: 입력 폴더명(in-motion), ```<split>```: pos_0~pos_4, ```<phase>```: train|test
#### 폴더 구조 예시
```text
data/
└─ in-motion/
   ├─ seq_01.csv
   ├─ seq_02.csv
   └─ ...
data/data_split/
├─ raw/
│  └─ in-motion/pos_0/{train,test}/*.csv
├─ norm/
│  └─ in-motion/pos_0/{train,test}/*.csv
└─ npz/
   └─ in-motion/
      ├─ train_pos_0.npz
      ├─ test_pos_0.npz
      └─ ...
```
#### 파이프라인
1. K-fold
   - 각 CSV를 길이 기준으로 5등분 → 선택된 한 구간을 test, 나머지를 train
   - test가 시작/끝이면 train은 연속 구간, 중간이면 train을 두 구간(train_0, train_1)으로 분리 저장
2. 정규화 통계
   - 모든 train 구간의 0을 제외한 RSSI만 모아 global min/max 계산(누수 방지)
3. 정규화 & 저장
   - ```x_norm = (x - min) / (max - min)``` (단, x==0 → 0 유지)
   - raw/norm 버전을 각각 train/test로 저장
5. 슬라이딩 윈도우 & NPZ
   - 정규화 CSV에서 윈도우 생성 → ```(N, window, (#beacons+1))``` 배열로 단일 NPZ 저장, 마지막 열은 Zone

#### 실행 방법
```python
python data_split.py
```
로그 예시
```bash
📁 Dataset: in-motion / Split pos_0
    ▪ Global Min (RSSI): -92.0
    ▪ Global Max (RSSI): -41.0
Sliding window: in-motion/pos_0/train: 100%|████| ...
✅ Saved ./data/data_split/npz/in-motion/train_pos_0.npz : 12345 samples
Sliding window: in-motion/pos_0/test:  100%|████| ...
✅ Saved ./data/data_split/npz/in-motion/test_pos_0.npz  :  2345 samples
```



