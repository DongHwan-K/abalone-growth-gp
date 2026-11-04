# GitHub 게시 절차

현재 저장소는 `DongHwan-K/abalone-growth-gp`입니다.

## 가장 쉬운 방법: GitHub 웹에서 업로드

1. GitHub에서 **New repository**를 선택합니다.
2. Repository name에 `abalone-growth-gp`를 입력합니다.
3. Public 또는 Private을 선택합니다.
4. **Add a README**, **Add .gitignore**, **Choose a license**는 모두 선택하지 않습니다. 이 저장소에 이미 포함되어 있습니다.
5. 저장소를 만든 뒤 **uploading an existing file**을 선택합니다.
6. 압축을 푼 `abalone-growth-gp` 폴더 안의 릴리스 파일을 업로드합니다. `input_data/`의 승인된 예시 Excel 두 파일은 포함하고, 원자료와 `abalone_paper_results/`(예시 결과·체크포인트)는 제외합니다. 예시 파일을 원자료로 교체한 경우 릴리스 검사가 실패합니다.
7. 커밋 메시지는 `Initial public release for the abalone growth prediction paper`로 입력합니다.

설명 문구:

> Reproducibility code and results for stacked monotone Gaussian process prediction of Pacific abalone nursery growth.

권장 topics:

`gaussian-processes`, `bayesian-modeling`, `aquaculture`, `abalone`, `monotonicity`, `model-stacking`, `prediction-intervals`, `reproducible-research`

## 터미널에서 한 번에 게시

GitHub CLI가 설치되고 로그인된 경우 저장소 루트에서 다음을 실행합니다.

```bash
./scripts/publish_to_github.sh
```

기본값은 공개 저장소입니다. 먼저 비공개로 올리려면:

```bash
./scripts/publish_to_github.sh private
```

## 게시 직전 확인

1. `python reproduce_all.py --example --workers 3`의 종료 상태가 0인지 확인합니다. 입력·내부 일관성 검사는 필수이며, 원 논문 수치와의 비교는 예시 실행에서 참고용으로 기록됩니다. 원 논문 수치 재현은 승인된 원자료를 지정하고 `--example` 없이 별도로 검증합니다.
2. `python scripts/check_release.py --write-checksums`로 `SHA256SUMS.txt`를 갱신하고, 승인된 예시 Excel 두 파일만 포함되어 있는지, 원자료와 체크포인트가 제외되었는지 확인합니다.
- 저자 순서와 논문 제목이 `CITATION.cff`와 일치하는지 확인합니다.
- 논문이 게재되면 `CITATION.cff`에 DOI, 저널, 권·호·쪽 정보를 추가합니다.
- 첫 공개 상태가 확정되면 GitHub Release `v1.0.0`을 만들고, 필요하면 Zenodo와 연결해 소프트웨어 DOI를 발급합니다.
