# Document Verification App

Este aplicativo realiza a verificação de documentos usando o Google Cloud Vision API para OCR e detecção facial.

## Funcionalidades

- Verificação de nome e CPF em documentos
- Extração de endereço de comprovantes de residência
- Comparação facial entre documento de identidade e selfie

## Requisitos

- Python 3.8+
- Conta Google Cloud com Cloud Vision API habilitada
- Credenciais do Google Cloud para o serviço Cloud Vision API

### Nota sobre Dependências

O aplicativo utiliza a biblioteca `face-recognition` que depende do `dlib`. Para facilitar a instalação:

#### macOS
1. Instale o Homebrew (se ainda não tiver):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Instale as dependências necessárias:
```bash
brew install cmake
brew install boost
brew install boost-python3
xcode-select --install  # Instala as ferramentas de linha de comando do Xcode
```

#### Linux
As dependências do sistema necessárias podem ser instaladas com:
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config
sudo apt-get install -y libx11-dev libatlas-base-dev
sudo apt-get install -y libgtk-3-dev libboost-python-dev
```

#### Windows
É recomendado usar o conda para instalar o dlib:
```bash
conda install -c conda-forge dlib
```

## Configuração Local

1. Clone o repositório
2. Instale as dependências:
```bash
pip install -r requirements.txt
```
3. Configure as credenciais do Google Cloud:
   - Crie um projeto no Google Cloud Console
   - Ative a Cloud Vision API
   - Crie uma conta de serviço e baixe o arquivo JSON de credenciais
   - Crie a pasta `.streamlit` na raiz do projeto:
     ```bash
     mkdir -p .streamlit
     ```
   - Crie o arquivo `.streamlit/secrets.toml` com suas credenciais no formato TOML:
     ```toml
     [google_credentials]
     type = "service_account"
     project_id = "seu-projeto-id"
     private_key_id = "seu-private-key-id"
     private_key = """-----BEGIN PRIVATE KEY-----
     SUA_CHAVE_PRIVADA_AQUI_COM_QUEBRAS_DE_LINHA
     -----END PRIVATE KEY-----
     """
     client_email = "seu-service-account@seu-projeto.iam.gserviceaccount.com"
     client_id = "seu-client-id"
     auth_uri = "https://accounts.google.com/o/oauth2/auth"
     token_uri = "https://oauth2.googleapis.com/token"
     auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
     client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/seu-service-account"
     ```
4. Execute o aplicativo:
```bash
streamlit run ocr.py
```

## Deployment no Streamlit Cloud

1. Faça push do código para um repositório GitHub (não inclua `.streamlit` e `.credentials.json`)
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte com seu GitHub e selecione o repositório
4. Em "Advanced Settings" > "Secrets", adicione suas credenciais do Google Cloud no mesmo formato TOML usado no arquivo `secrets.toml` local

Notas importantes sobre as credenciais:
- Use `"""` (três aspas) para o campo `private_key` para preservar as quebras de linha
- Mantenha a indentação correta no formato TOML
- Não inclua vírgulas ou chaves como em JSON
- Certifique-se de que todos os campos estejam presentes e corretamente formatados

5. Clique em "Deploy!"

## Segurança

- Nunca commit arquivos de credenciais (.env, .credentials.json, .streamlit/secrets.toml)
- Use sempre variáveis de ambiente ou secrets do Streamlit Cloud
- Mantenha as credenciais do Google Cloud seguras
- Restrinja as permissões da conta de serviço ao mínimo necessário
- Monitore o uso da API para evitar custos inesperados

## Solução de Problemas

### Erro: "No secrets files found"
Se você encontrar este erro localmente, verifique se:
1. A pasta `.streamlit` existe na raiz do projeto
2. O arquivo `secrets.toml` está dentro da pasta `.streamlit`
3. O arquivo `secrets.toml` está formatado corretamente em TOML
4. Todos os campos necessários estão presentes no arquivo

### Erro: "Invalid format: please enter valid TOML"
Se você encontrar este erro no Streamlit Cloud:
- Verifique se o formato TOML está correto (sem vírgulas, sem chaves)
- Certifique-se de que o `private_key` está com três aspas e quebras de linha
- Confirme se todos os campos necessários estão presentes
- Verifique a indentação sob a seção `[google_credentials]` 