# Step 2

- 프로젝트: sEMG 팔동작 예측
- 목표: 손동작 4종 × 팔꿈치 각도 2종 = 8클래스 분류
- 입력: 5채널 sEMG
- Step 2 Week 1: 데이터 구조 확인 및 30°/60° 신호 비교
- Step 2 Week 2: 전처리, 200ms window, RMS/MAV/WL/ZC/SSC 25D 특징 추출
- Step 2 Week 3: SVM/RF/XGBoost 비교, hold-out, 5-fold, 통계 검정
- 기존 .venv를 재사용한다.
