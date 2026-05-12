# SlopeDisplace

옹벽(retaining wall) 3D 포인트 클라우드 스캔 데이터를 비교하여 사면 변위를 분석하고 시각화하는 시스템.

## 요구사항

- Python 3.12+
- pip 패키지: `fastapi`, `uvicorn`, `open3d`, `numpy`, `scipy`, `pandas`

```bash
pip install fastapi uvicorn open3d numpy scipy pandas
```

## 1. 데이터 준비

`data/` 폴더 아래 데이터셋 폴더를 만들고, 원본 스캔 PLY 파일을 넣는다.

```
data/
  CH2_RETAINWALL/
    20260511.ply    ← 기준 스캔 (가장 오래된 파일이 자동으로 reference가 됨)
    20260512.ply    ← 비교 스캔
    ...
```

> 환경변수 `SLOPE_DATA_ROOT`로 데이터 경로를 변경할 수 있다 (기본값: `<project>/data`).

## 2. 서버 실행

```bash
python -m uvicorn server.app:app --reload --host 127.0.0.1 --port 8000
```

브라우저에서 http://localhost:8000/ 접속.

## 3. 전처리 (Preprocessing)

### 웹 UI에서 실행

1. 좌측 패널에서 데이터셋 선택
2. 파일 목록에서 비교 스캔 파일 클릭
3. "전처리" 버튼 클릭
4. Job log에서 진행 상황 확인

### CLI에서 직접 실행

```bash
python -c "
from server.pipeline import preprocess

def prog(stage, detail=''):
    print(f'[{stage}] {detail}')

result = preprocess('CH2_RETAINWALL', '20260512.ply', progress=prog)
print(result)
"
```

### 전처리 파이프라인 단계

1. **다운샘플링** - 원본 스캔을 ~500k 포인트로 voxel downsample → `*_simple.ply`
2. **FGR (Fast Global Registration)** - simplified 클라우드 간 초기 정합 → 변환행렬 T0
3. **FGR 검증** - 두 스캔의 중심 거리를 계산하여 co-located 여부 판단. 같은 좌표계인데 FGR이 큰 회전(>30도)을 찾으면 identity로 fallback
4. **Multi-scale ICP** - 원본(full) 클라우드에서 5단계 정밀 정합 (1.0m → 0.01m). 초기 스케일에서 넓은 correspondence distance(v*4.0)와 200회 iteration으로 FGR 오차 보정
5. **ICP 품질 검증** - 최종 fitness < 0.3이면 identity init으로 재시도, 더 나은 결과 채택
6. **변위 계산** - 정합된 타겟 vs 기준 클라우드 간 per-point displacement 산출 (signed_normal, magnitude, horizontal, vertical)
7. **결과 저장** - `*_disp.bin` (N x 4 float32), `*_meta.json` (변환행렬, ICP 이력, 변위 통계)

### 전처리 결과 파일

| 파일 | 설명 |
|------|------|
| `*_simple.ply` | 다운샘플된 포인트 클라우드 (정합 적용 후) |
| `*_disp.bin` | N x 4 float32 변위 데이터 (signed_normal, magnitude, horizontal, vertical) |
| `*_rgb.bin` | N x 3 uint8 원본 색상 (lazy backfill) |
| `*_meta.json` | 변환행렬, ICP 이력, 변위 통계, 정합 품질 지표 |

### 정합 품질 확인

`*_meta.json`에서 다음 필드를 확인:

```json
{
  "fgr_rotation_deg": 177.2,
  "fgr_accepted": true,
  "center_distance": 38.9,
  "final_icp_fitness": 0.90,
  "registration_reliable": true
}
```

- `final_icp_fitness` > 0.8 이면 양호
- `registration_reliable: false`이면 변위 결과 신뢰 불가 → 재전처리 필요

### 재전처리

기존 결과를 삭제 후 다시 실행:

```bash
# 기존 전처리 결과 삭제
rm data/CH2_RETAINWALL/20260512_simple.ply
rm data/CH2_RETAINWALL/20260512_disp.bin
rm data/CH2_RETAINWALL/20260512_meta.json
rm data/CH2_RETAINWALL/20260512_rgb.bin

# 재전처리 (웹 UI 또는 CLI)
```

## 4. 시각화

브라우저에서 로딩 완료 후:

- **모드 전환**: signed_normal / magnitude / horizontal / vertical / RGB
- **Clamp 조절**: 변위 색상 범위 설정 (예: 0.05m = +/-50mm). 양쪽이 clamp 색상으로 채워지면 정합 오류 의심
- **포인트 크기**: 슬라이더로 조절

## 5. 주요 API

| Endpoint | 설명 |
|----------|------|
| `GET /api/datasets` | 데이터셋 목록 |
| `GET /api/files?dataset=X` | 파일 목록 (전처리 상태 포함) |
| `GET /api/ply/{dataset}/{stem}` | simplified PLY 스트리밍 |
| `GET /api/disp/{dataset}/{stem}` | 변위 바이너리 스트리밍 (N x 4 float32) |
| `GET /api/rgb/{dataset}/{stem}` | RGB 바이너리 스트리밍 |
| `POST /api/preprocess` | 비동기 전처리 시작 |
| `GET /api/job/{jid}` | 작업 상태 조회 |

## 6. 우분투 서버 배포

### 6.1 시스템 준비

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.12 python3.12-venv python3-pip git nginx
```

### 6.2 프로젝트 클론 및 가상환경

```bash
cd /opt
sudo git clone https://github.com/bielcorp2017/SlopeDisplace.git
sudo chown -R $USER:$USER /opt/SlopeDisplace

cd /opt/SlopeDisplace
python3.12 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn open3d numpy scipy pandas
```

### 6.3 데이터 디렉토리

```bash
# 기본 경로 사용
mkdir -p /opt/SlopeDisplace/data

# 또는 별도 디스크 마운트 시 환경변수로 지정
# export SLOPE_DATA_ROOT=/mnt/data/slope
```

스캔 PLY 파일을 `data/<데이터셋명>/` 아래에 배치한다.

### 6.4 systemd 서비스 등록

```bash
sudo tee /etc/systemd/system/slopedisp.service << 'EOF'
[Unit]
Description=SlopeDisplace Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/SlopeDisplace
Environment="PATH=/opt/SlopeDisplace/venv/bin:/usr/bin"
# Environment="SLOPE_DATA_ROOT=/mnt/data/slope"
ExecStart=/opt/SlopeDisplace/venv/bin/uvicorn server.app:app --host 127.0.0.1 --port 8000 --workers 1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

> `User=ubuntu` 부분을 실제 사용자명으로 변경한다.
> 전처리는 CPU/메모리를 많이 사용하므로 `--workers 1`을 권장한다.

```bash
sudo systemctl daemon-reload
sudo systemctl enable slopedisp
sudo systemctl start slopedisp

# 상태 확인
sudo systemctl status slopedisp

# 로그 확인
sudo journalctl -u slopedisp -f
```

### 6.5 Nginx 리버스 프록시

```bash
sudo tee /etc/nginx/sites-available/slopedisp << 'EOF'
server {
    listen 80;
    server_name your-domain.com;   # 또는 서버 IP

    client_max_body_size 0;        # 대용량 PLY 업로드 제한 해제

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # 전처리 폴링 등 긴 요청 대응
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/slopedisp /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

### 6.6 HTTPS 설정 (선택)

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 6.7 방화벽

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow OpenSSH
sudo ufw enable
```

### 6.8 운영 명령어 요약

```bash
# 서비스 관리
sudo systemctl start slopedisp
sudo systemctl stop slopedisp
sudo systemctl restart slopedisp

# 코드 업데이트
cd /opt/SlopeDisplace
git pull
sudo systemctl restart slopedisp

# CLI 전처리 (서버와 별도로 실행 가능)
cd /opt/SlopeDisplace
source venv/bin/activate
python -c "
from server.pipeline import preprocess
result = preprocess('CH2_RETAINWALL', '20260512.ply',
                    progress=lambda s,d: print(f'[{s}] {d}'))
print(result)
"

# 전처리 결과 삭제 후 재실행
rm data/CH2_RETAINWALL/20260512_{simple.ply,disp.bin,meta.json,rgb.bin}
```

---

## 프로젝트 구조

```
server/
  app.py            FastAPI 서버
  pipeline.py       전처리 파이프라인 (FGR + ICP + 변위 계산)
  inject_synthetic.py  합성 변위 데이터 생성 (개발용)
web/
  index.html        SPA 프론트엔드
  app.js            Three.js 시각화
cpp/
  src/pipeline.h    C++ 고속 전처리기
data/
  CH2_RETAINWALL/   데이터셋 예시
```
