# transformer_model.py
# -*- coding: utf-8 -*-  # 파이썬 소스 파일 인코딩 지정(한글 주석/문자 안전)
from __future__ import annotations  # 타입 힌트를 문자열 없이 앞으로 사용할 수 있게 해줌(순환 참조 등에 유용)
import math                       # 수학 함수(로그, 지수 등) 사용
import torch                      # PyTorch 텐서 및 연산
import torch.nn as nn             # 신경망 모듈(레이어, 모델 구성)
import torch.nn.functional as F   # 함수형 API(활성함수, 손실 등)

# -----------------------------
# Positional encoding (batch_first)
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        """
        d_model: 임베딩 차원(Transformer d_model)
        max_len: 미리 생성해둘 최대 시퀀스 길이(포지셔널 인코딩 테이블 길이)
        """
        super().__init__()  # nn.Module 초기화
        pe = torch.zeros(max_len, d_model)  # (L, D) 모양의 포지셔널 인코딩 테이블 초기화
        position = torch.arange(0, max_len).unsqueeze(1)  # 위치 인덱스(0..L-1) → (L,1)로 차원 확장
        # div_term: 사인/코사인 주기를 조절하는 지수 항(Transformer 논문 공식)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # 짝수 인덱스에 사인, 홀수 인덱스에 코사인 값을 채워 넣음
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 채널
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 채널
        # 학습 파라미터는 아니지만 함께 저장/이동되는 버퍼로 등록(모델과 함께 CUDA로 이동 등)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)로 배치 차원 추가

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D) 형태의 임베딩 시퀀스
        반환: 위치 정보가 더해진 임베딩 (B, T, D)
        """
        T = x.size(1)                 # 입력 시퀀스 길이(T)
        return x + self.pe[:, :T, :]  # 미리 저장한 PE에서 현재 길이만큼 잘라 더해줌

# -----------------------------
# Self-attentive pooling
# -----------------------------
class AttentivePool(nn.Module):
    def __init__(self, dim: int):
        """
        dim: 입력 임베딩 차원(D)
        목적: 시퀀스의 각 시점에 중요도 가중치를 학습해 가중합으로 요약
        """
        super().__init__()
        # 게이트 네트워크: 시점별 스칼라 점수(1채널)를 생성
        self.gate = nn.Sequential(
            nn.LayerNorm(dim),       # 안정화를 위한 레이어 정규화
            nn.Linear(dim, dim // 2),# 차원 축소(비선형 변환 전 단계)
            nn.GELU(),               # 활성함수(GELU)
            nn.Linear(dim // 2, 1)   # 최종적으로 스칼라 점수(1) 산출
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D) 시퀀스 임베딩
        반환: (B, D) 가중합으로 요약된 전역 표현
        """
        w = F.softmax(self.gate(x), dim=1)  # (B, T, 1) 시점별 주의 가중치(합=1)
        return (w * x).sum(dim=1)           # (B, D) 가중합

# -----------------------------
# Confidence Gate from missing mask
# -----------------------------
def confidence_scale_from_mask(obs_mask: torch.Tensor) -> torch.Tensor:
    """
    obs_mask: (B, T, 5)  관측=1, 결측=0
    반환: (B, T, 1)  scale in [0.5, 1.0]
    - 시점별 결측률(miss_ratio)이 높을수록 신뢰도(confidence)를 낮춤
    """
    # miss_ratio: (B,T,1) = 1 - (비콘 축 평균 관측 비율)
    miss_ratio = 1.0 - obs_mask.mean(dim=-1, keepdim=True)
    # confidence = 0.5 + 0.5*(1 - miss_ratio) = 1 - 0.5*miss_ratio  ∈ [0.5, 1.0]
    return 0.5 + 0.5 * (1.0 - miss_ratio)

# -----------------------------
# Model
# -----------------------------
class TransformerClassifierImproved(nn.Module):
    """
    입력 x: (B, T, 5), 0은 결측
    출력: {"logits": (B,C), "aux_mask_loss": scalar}
    - 값과 관측마스크를 함께 입력(5+5=10)
    - 시점별 결측률 기반 confidence 게이트로 임베딩 스케일링
    - Transformer 인코더 후 Attentive Pooling으로 요약, 분류
    - 학습 시 일부 관측값을 가리고 복원하는 보조손실(aux)로 일반화 향상
    """
    def __init__(self, n_beacons: int, n_classes: int, cfg,
                 aux_mask_ratio: float = 0.15, aux_loss_weight: float = 0.3,
                 beacon_dropout_p: float = 0.0):
        """
        n_beacons: 비콘 개수(기본 5)
        n_classes: 분류 클래스 수(존 수)
        cfg: 하이퍼파라미터 묶음(embedding_dim, n_heads, n_layers, dropout, window_size 등)
        aux_mask_ratio: 보조 복원용 임의 마스킹 비율
        aux_loss_weight: 보조 손실 가중치
        beacon_dropout_p: 학습 시 채널(비콘) 드롭아웃 확률
        """
        super().__init__()
        self.cfg = cfg                                 # 설정 저장
        self.n_beacons = n_beacons                     # 비콘 수 저장
        self.n_classes = n_classes                     # 클래스 수 저장
        self.aux_mask_ratio = aux_mask_ratio           # 보조 마스킹 비율
        self.aux_loss_weight = aux_loss_weight         # 보조 손실 가중치
        self.beacon_dropout_p = beacon_dropout_p       # 비콘 드롭아웃 확률

        in_dim = n_beacons * 2  # 값(5) + 관측마스크(5) = 10 차원 입력
        self.embedding = nn.Linear(in_dim, cfg.embedding_dim)  # (10→D) 임베딩 투영
        self.pos = PositionalEncoding(cfg.embedding_dim, max_len=cfg.window_size)  # 위치 인코딩

        # Transformer 인코더 레이어 정의(배치우선, GELU 활성화)
        enc = nn.TransformerEncoderLayer(
            d_model=cfg.embedding_dim,  # 모델 차원 D
            nhead=cfg.n_heads,          # 멀티헤드 수
            dropout=cfg.dropout,        # 드롭아웃
            batch_first=True,           # 입력이 (B,T,D) 형태임을 명시
            activation="relu"           # FFN 활성화 함수
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=cfg.n_layers)  # 레이어 반복 스택

        self.pool = AttentivePool(cfg.embedding_dim)  # 시퀀스 요약(가중합)
        self.norm = nn.LayerNorm(cfg.embedding_dim)   # 최종 표현 정규화
        self.fc = nn.Linear(cfg.embedding_dim, n_classes)  # 분류기(로짓 산출)

        # 보조 복원 헤드: 시점별로 5개 비콘 값을 예측(마스크드 모델링 목적)
        self.recon_head = nn.Linear(cfg.embedding_dim, n_beacons)

    @torch.no_grad()  # 학습/추론과 무관한 마스크 계산이므로 그래디언트 추적 비활성화
    def _obs_and_keypad(self, x: torch.Tensor):
        """
        x: (B,T,5) RSSI(0은 결측)
        반환:
          obs_mask: (B,T,5) 관측 여부(0/1)
          key_pad : (B,T)   해당 시점이 완전 결측(모든 비콘 0)이면 True → 어텐션에서 패딩 처리
        """
        obs_mask = (x != 0).float()             # 관측 여부(0/1) 마스크
        key_pad  = (obs_mask.sum(dim=-1) == 0)  # 모든 비콘이 0이면 True
        return obs_mask, key_pad

    def _maybe_beacon_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        학습 시 비콘(채널) 단위 드롭아웃을 적용해 특정 채널 의존 과적합을 방지
        - 배치 내 샘플마다 드롭할 채널을 샘플-독립적으로 선택
        """
        if not self.training or self.beacon_dropout_p <= 0:
            return x  # 학습이 아니거나 확률 0이면 원본 유지
        B, T, C = x.shape
        # (B,C)에서 True인 채널은 해당 배치 샘플에서 전체 시점에 걸쳐 0으로 설정
        drop_mask = torch.rand(B, C, device=x.device) < self.beacon_dropout_p
        if drop_mask.any():
            x = x.clone()  # 인플레이스 수정 방지
            for b in range(B):
                cols = torch.nonzero(drop_mask[b]).flatten()  # 드롭 대상 채널 인덱스
                if len(cols) > 0:
                    x[b, :, cols] = 0.0  # 해당 채널 전체 시점을 0(결측)으로 설정
        return x

    def forward(self, x: torch.Tensor, do_aux: bool = True):
        """
        x: (B,T,5) 입력 RSSI 시퀀스(0은 결측)
        do_aux: 학습 중 보조 복원 손실 계산 여부
        반환: dict {"logits": (B,C), "aux_mask_loss": scalar 텐서}
        """
        B, T, C = x.shape  # 배치/길이/채널 크기 해석

        # Beacon Dropout (옵션): 학습 중 채널 드롭아웃으로 채널 의존 완화
        x = self._maybe_beacon_dropout(x)

        # 관측마스크(obs_mask) & Key-Padding mask(key_pad) 계산
        obs_mask, key_pad = self._obs_and_keypad(x)          # obs_mask:(B,T,5), key_pad:(B,T)
        conf = confidence_scale_from_mask(obs_mask)          # (B,T,1) 시점별 신뢰도 게이트

        # 값(+마스크) 결합 → 선형 임베딩 → 포지셔널 인코딩 → 신뢰도 게이트 적용
        x_in = torch.cat([x, obs_mask], dim=-1)              # (B,T,10=5값+5마스크)
        x_in = self.embedding(x_in)                          # (B,T,D)로 투영
        x_in = self.pos(x_in)                                # 위치 정보 더하기
        x_in = x_in * conf                                   # 결측률 높은 시점은 영향 축소

        # Transformer Encoder 통과(완전 결측 시점은 key_pad로 어텐션에서 무시)
        x_enc = self.encoder(x_in, src_key_padding_mask=key_pad)  # (B,T,D)

        # 시퀀스 요약 → 정규화 → 분류 로짓
        pooled = self.pool(x_enc)                            # (B,D) Self-attentive pooling
        pooled = self.norm(pooled)                           # 안정화
        logits = self.fc(pooled)                             # (B,C) 클래스 로짓

        # 기본 출력 딕셔너리(보조손실은 기본 0)
        out = {"logits": logits, "aux_mask_loss": torch.tensor(0.0, device=x.device)}

        # 보조 복원 손실 (Masked-RSSI): 학습 시에만, do_aux=True이고 비율>0일 때
        if self.training and do_aux and self.aux_mask_ratio > 0:
            valid = (x != 0)  # 관측된 위치만 가릴 수 있음(원래 0은 결측이므로 제외)
            if valid.any():
                # 관측 위치 중 일부를 aux_mask_ratio로 무작위 선택하여 가림
                mask_sel = (torch.rand_like(x) < self.aux_mask_ratio) & valid  # True인 위치를 0으로 만들 계획
                x_masked = x.clone()
                x_masked[mask_sel] = 0.0  # 선택 위치를 실제로 가려 결측으로 만듦

                # 마스킹된 입력으로 동일 파이프라인 재통과
                obs2, pad2 = self._obs_and_keypad(x_masked)          # 새 관측마스크/패딩마스크
                conf2 = confidence_scale_from_mask(obs2)              # 새 신뢰도 게이트
                z = torch.cat([x_masked, obs2], dim=-1)               # (B,T,10)
                z = self.embedding(z)                                 # (B,T,D)
                z = self.pos(z)                                       # 위치 인코딩
                z = z * conf2                                         # 게이트 적용
                z = self.encoder(z, src_key_padding_mask=pad2)        # (B,T,D)
                recon = self.recon_head(z)                            # (B,T,5) 마스크 복원 예측

                # 가렸던 위치들만 골라 예측과 원래값 차이로 Smooth L1 손실 계산
                diff = recon[mask_sel] - x[mask_sel]                  # 선택 위치의 오차
                aux_loss = F.smooth_l1_loss(diff, torch.zeros_like(diff))  # L1-Huber 손실
                out["aux_mask_loss"] = aux_loss * self.aux_loss_weight     # 가중치 적용 후 반환

        return out  # 분류 로짓과 보조 손실을 함께 반환
