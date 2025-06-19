## Azure Cloud Shell
```shell
# Resource Group 생성
az group create --name <your_resource_group> --location <your_region>

# OpenAI Service 생성
az cognitiveservices account create --name <your_openai_name> --resource-group <your_resource_group> --kind OpenAI --sku S0 --location <your_region>

# Azure AI Search 생성
az search service create --name <your_ai_search_name> --resource-group <your_resource_group> --sku Basic --partition-count 1 --replica-count 1

# Azure Storage Account 생성
az storage account create --name <your_storage_account_name> --resource-group <your_resource_group> --location <your_region> --sku Standard_LRS

# 컨테이너 생성
az storage container create --name <your_file_name> --account-name <your_storage_account_name>

# 테이블 생성
az storage table create --name <your_table_name> --account-name <your_storage_account_name>
```
#### 이후 과정은 Azure Portal에서 진행 ↓

## Azure Portal
### 1. resource-group 생성
### 2. openai 생성 -> gpt-4o-mini, embedding-model 생성
### 3. storage 생성 -> Blob Storage(container), Table 생성, 기초 데이터 삽입
### 4. rtm폴더에 rtm_sample.xlsx 저장
### 5. 테이블에 Patition Key(RTM), Key(rtm_sample.xlsx), Value(rtm_sample.xlsx) 저장
### 6. ai search 생성 -> 데이터 가져오기 및 벡터화로 기초 index 설정
### 7. 웹 앱 생성
- 배포 폴더를 새로 만든 후 실행
```shell
phthon -m venv .venv
.\.venv\Scripts\Activate.ps1
```
- 필요 파일들 이동(app.py, rtm_pipeline.py, requirements.txt, data)
- .deployment 생성
```shell
[config]
SCM_DO_BUILD_DURING_DEPLOYMENT=false
```
- streamlit.sh 생성
```shell
pip install -r requirements.txt
python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0
```
- 웹 앱 => 설정 => 환경 변수 => .env.sample 참고하여 입력
- 웹 앱 => 설정 => 구성(시작 명령)에 아래 문자열 저장 => 개요 => 다시 시작
```text
bash /home/site/wwwroot/streamlit.sh
```
