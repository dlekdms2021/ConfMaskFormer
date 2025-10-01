# ConfMaskFormer
신호 결측에 강건한 비컨 기반 실내 위치 추정 딥러닝 모델

### 데이터 전처리(data_split.py)
BLE RSSI 시계열 CSV를 대상으로 5-분할 스플릿 → 글로벌 Min-Max 정규화(0=결측 보존) → 슬라이딩 윈도우 → .csv/.npz 저장
#### 입출력 개요
입력: ./data/in-motion/*.csv
출력
- 분할된 원본/정규화 CSV
./data/data_split/{raw|norm}/<dataset>/<split>/<phase>/*.csv
- 슬라이딩 윈도우 NPZ
./data/data_split/npz/<dataset>/{train|test}_{split}.npz
<dataset>: 입력 폴더명(in-motion), <split>: pos_0~pos_4, <phase>: train|test.
#### 폴더 구조 예시
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
