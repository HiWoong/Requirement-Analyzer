## Azure Cloud Shell
```shell
# Resource Group의 생성
echo "Resource Group creating..."
az group create --name <your_resource_group> --location <your_region>

# OpenAI Service 생성
echo "OpenAI Service creating..."
az cognitiveservices account create --name <your_openai_name> --resource-group <your_resource_group> --kind OpenAI --sku S0 --location <your_region>

# Azure AI Search의 생성
echo "Azure AI Search creating..."
az search service create --name <your_ai_search_name> --resource-group <your_resource_group> --sku Basic --partition-count 1 --replica-count 1

# Azure Storage Accout의 생성
echo "Azure Storage Account creating..."
az storage account create --name <your_storage_account_name> --resource-group <your_resource_group> --location <your_region> --sku Standard_LRS

# 컨테이너 생성
echo "Blob storage container creating in Azure Storage Account"
az storage container create --name <your_file_name> --account-name <your_storage_account_name>

# 테이블 생성
echo "Table creating in Azure Storage Account"
az storage table create --name <your_table_name> --account-name <your_storage_account_name>
```

## Azure Portal
### 1. resource-group 생성
### 2. openai 생성 -> gpt-4o-mini, embedding-model 생성
### 3. storage 생성 -> Blob Storage(container), Table 생성, 기초 데이터 삽입
- rtm폴더에 rtm_sample.xlsx 저장
- 테이블에 Patition Key(RTM), Key(rtm_sample.xlsx), Value(rtm_sample.xlsx) 저장
### 4. ai search 생성 -> 데이터 가져오기 및 벡터화로 기초 index 설정
### 5. 웹 앱 생성
- prd 폴더를 새로 만든 후에 phthon -m venv .venv 실행
```shell
phthon -m venv .venv
source venv/bin/activate 실행   # Windows: venv\Scripts\activate
```
- 앞에 (.venv)가 생기면 필요 파일들 이동(app.py, rtm_pipeline.py, requirements.txt, data)
- .deployment 생성([config] SCM_DO_BUILD_DURING_DEPLOYMENT=false)
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
- 웹 앱 => 설정 => 구성(시작 명령 : bash /home/site/wwwroot/streamlit.sh) 저장 => 개요 => 다시 시작