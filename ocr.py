import streamlit as st
import io
import numpy as np
import cv2
from PIL import Image, ExifTags
from difflib import SequenceMatcher
import re
import os
from dotenv import load_dotenv
from google.cloud import vision
import json
import tempfile

# Carrega as variáveis de ambiente
load_dotenv()

# ---------------------
# Configurações e Constantes
# ---------------------

# Configurações de detecção facial
FACE_CONFIDENCE_THRESHOLD = 0.8

# Padrões de CPF
CPF_PATTERNS = [
    r'CPF\s*:?\s*(\d{3}\.?\d{3}\.?\d{3}-?\d{2})',  # CPF: 123.456.789-00
    r'CPF\s*:?\s*(\d{11})',                         # CPF: 12345678900
    r'CPF\s*:?\s*(\d{3}\.?\d{3}\.?\d{3})',         # CPF: 123.456.789 (faltando últimos dígitos)
    r'CPF\s*[Nn][ºo°]\s*:?\s*(\d{3}\.?\d{3}\.?\d{3}-?\d{2})',  # CPF Nº: 123.456.789-00
    r'CPF\s*[Nn][ºo°]\s*:?\s*(\d{11})',            # CPF Nº: 12345678900
    r'Cadastro\s+de\s+Pessoas?\s+Físicas?:?\s*(\d{3}\.?\d{3}\.?\d{3}-?\d{2})',  # Nome completo do CPF
    r'Cadastro\s+de\s+Pessoas?\s+Físicas?:?\s*(\d{11})',
    # Adiciona versões case insensitive
    r'cpf\s*:?\s*(\d{3}\.?\d{3}\.?\d{3}-?\d{2})',
    r'cpf\s*:?\s*(\d{11})',
    r'cpf\s*[Nn][ºo°]\s*:?\s*(\d{3}\.?\d{3}\.?\d{3}-?\d{2})',
    r'cpf\s*[Nn][ºo°]\s*:?\s*(\d{11})'
]

# Campos de nome
NAME_FIELDS = [
    "nome:", "nome", 
    "nome e sobrenome:", "nome e sobrenome",
    "NOME:", "NOME",
    "NOME E SOBRENOME:", "NOME E SOBRENOME"
]

# Indicadores que não são nomes
NON_NAME_INDICATORS = [
    "cpf", "rg", "identidade", "cnh", "nascimento", "data",
    "endereço", "endereco", "residência", "residencia",
    "número", "numero", "telefone", "celular", "email",
    "valor", "total", "vencimento", "conta", "banco",
    "agência", "agencia", "documento"
]

# Mapeamento de tipos de logradouro
STREET_TYPE_MAP = {
    'rua': 'Rua',
    'r.': 'Rua',
    'r': 'Rua',
    'avenida': 'Avenida',
    'av.': 'Avenida',
    'av': 'Avenida',
    'alameda': 'Alameda',
    'al.': 'Alameda',
    'al': 'Alameda',
    'travessa': 'Travessa',
    'tv.': 'Travessa',
    'tv': 'Travessa',
    'praça': 'Praça',
    'pça.': 'Praça',
    'pça': 'Praça',
    'servidão': 'Servidão',
    'servid': 'Servidão',
    'sv.': 'Servidão',
    'sv': 'Servidão'
}

# ---------------------
# Configuração do Google Cloud
# ---------------------

def setup_google_credentials():
    """Configura as credenciais do Google Cloud tanto para ambiente local quanto para Streamlit Cloud."""
    try:
        # Verifica se estamos no Streamlit Cloud checando a variável de ambiente
        if 'google_credentials' in st.secrets:
            # Cria um dicionário com as credenciais do formato TOML
            credentials = {
                "type": st.secrets.google_credentials.type,
                "project_id": st.secrets.google_credentials.project_id,
                "private_key_id": st.secrets.google_credentials.private_key_id,
                "private_key": st.secrets.google_credentials.private_key,
                "client_email": st.secrets.google_credentials.client_email,
                "client_id": st.secrets.google_credentials.client_id,
                "auth_uri": st.secrets.google_credentials.auth_uri,
                "token_uri": st.secrets.google_credentials.token_uri,
                "auth_provider_x509_cert_url": st.secrets.google_credentials.auth_provider_x509_cert_url,
                "client_x509_cert_url": st.secrets.google_credentials.client_x509_cert_url
            }
            
            # Usa um arquivo temporário para armazenar as credenciais
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(credentials, f)
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f.name
        else:
            # Em ambiente local, usa o arquivo .credentials.json
            if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '.credentials.json'
    except Exception as e:
        st.error(f"Erro ao configurar credenciais do Google Cloud: {str(e)}")
        return False
    return True

# Configura as credenciais antes de inicializar o cliente
if setup_google_credentials():
    # Inicializa o cliente do Vision API
    vision_client = vision.ImageAnnotatorClient()
else:
    st.error("Falha ao configurar credenciais do Google Cloud. Verifique sua configuração.")
    st.stop()

# ---------------------
# Funções Auxiliares
# ---------------------

def process_image(uploaded_file):
    """Processa uma imagem carregada, corrigindo orientação e convertendo para RGB se necessário."""
    try:
        image = Image.open(uploaded_file)
        image = fix_image_orientation(image)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        bytes_io = io.BytesIO()
        image.save(bytes_io, format="JPEG")
        return bytes_io.getvalue()
    except Exception as e:
        st.error(f"Erro ao processar imagem: {str(e)}")
        return None

def fix_image_orientation(image):
    """Corrige a orientação da imagem com base nos metadados EXIF para evitar imagens rotacionadas."""
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = dict(image._getexif().items())
        if orientation in exif:
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass  # Nenhum dado EXIF encontrado; retorna como está
    return image

def extract_text(image_data):
    """Extrai texto da imagem usando Google Cloud Vision API."""
    try:
        # Cria a imagem para o Vision API
        image = vision.Image(content=image_data)
        
        # Realiza o reconhecimento de texto
        response = vision_client.text_detection(image=image)
        texts = response.text_annotations
        
        if texts:
            # O primeiro texto contém todo o conteúdo detectado
            return texts[0].description
        
        if response.error.message:
            raise Exception(
                '{}\nPara mais informações:\n{}'.format(
                    response.error.message,
                    response.error.details
                )
            )
            
    except Exception as e:
        st.error(f"Erro ao extrair texto: {str(e)}")
    return None

def detect_face(image_data):
    """Detecta rosto usando Google Cloud Vision API."""
    try:
        # Cria a imagem para o Vision API
        image = vision.Image(content=image_data)
        
        # Realiza a detecção facial
        response = vision_client.face_detection(image=image)
        faces = response.face_annotations
        
        if faces:
            # Retorna a confiança da detecção do primeiro rosto
            return faces[0]
        
        if response.error.message:
            raise Exception(
                '{}\nPara mais informações:\n{}'.format(
                    response.error.message,
                    response.error.details
                )
            )
            
    except Exception as e:
        st.error(f"Erro ao detectar rosto: {str(e)}")
    return None

def compare_faces(face1, face2):
    """Compara dois rostos usando os dados do Google Cloud Vision API."""
    if face1 and face2:
        # Calcula a similaridade baseada na confiança da detecção
        confidence = min(face1.detection_confidence, face2.detection_confidence)
        # Considera rostos idênticos se a confiança for alta
        is_identical = confidence > FACE_CONFIDENCE_THRESHOLD
        return is_identical, confidence
    return False, 0

def compare_names(name1, name2):
    """Compara duas strings usando uma taxa de similaridade simples."""
    return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()

def normalize_text(text, uppercase=False):
    """Normaliza o texto removendo espaços extras, quebras de linha e opcionalmente converte para maiúsculas."""
    if not text:
        return ""
    normalized = ' '.join(text.replace('\n', ' ').split())
    return normalized.upper() if uppercase else normalized

def extract_cpf_from_text(text):
    """Extrai CPF do texto do documento com correspondência de padrões aprimorada."""
    if not text:
        return None
    
    # Normaliza o texto
    text = normalize_text(text)
    
    try:
        for pattern in CPF_PATTERNS:
            match = re.search(pattern, text)
            if match:
                cpf = match.group(1)
                # Limpa o número do CPF (remove separadores)
                cpf = cpf.replace('.', '').replace('-', '').replace('/', '')
                
                # Valida se parece um CPF (11 dígitos)
                if len(cpf) == 11 and cpf.isdigit():
                    # Formata CPF como XXX.XXX.XXX-XX
                    return f"{cpf[:3]}.{cpf[3:6]}.{cpf[6:9]}-{cpf[9:]}"
                elif len(cpf) >= 9:  # Se temos pelo menos os primeiros 9 dígitos
                    # Tenta encontrar os dígitos restantes após este match
                    remaining_digits = re.search(r'\d{2}', text[match.end():])
                    if remaining_digits:
                        cpf = cpf + remaining_digits.group(0)
                        if len(cpf) == 11:
                            return f"{cpf[:3]}.{cpf[3:6]}.{cpf[6:9]}-{cpf[9:]}"
        
        # Se nenhum CPF foi encontrado com os padrões, tenta encontrar qualquer número de 11 dígitos
        numbers = re.findall(r'\d+', text)
        for num in numbers:
            if len(num) == 11:
                return f"{num[:3]}.{num[3:6]}.{num[6:9]}-{num[9:]}"
            
    except Exception as e:
        st.error(f"Erro ao extrair CPF: {str(e)}")
    return None

def extract_name_from_text(text):
    """Extrai nome do texto do documento procurando por campos de nome."""
    if not text:
        return None
    
    # Normaliza o texto
    text = normalize_text(text)
    lines = text.split()
    
    try:
        # Procura por campos de nome explícitos
        nome_index = -1
        name_fields = [
            "nome:", "nome", 
            "nome e sobrenome:", "nome e sobrenome",
            "NOME:", "NOME",
            "NOME E SOBRENOME:", "NOME E SOBRENOME"
        ]
        
        # Lista de palavras que indicam que não é um nome
        non_name_indicators = [
            "cpf", "rg", "identidade", "cnh", "nascimento", "data",
            "endereço", "endereco", "residência", "residencia",
            "número", "numero", "telefone", "celular", "email",
            "valor", "total", "vencimento", "conta", "banco",
            "agência", "agencia", "documento"
        ]
        
        # Encontra a primeira ocorrência de qualquer campo de nome
        for i, word in enumerate(lines):
            if i + 2 < len(lines):
                three_word_field = f"{word} {lines[i+1]} {lines[i+2]}".lower()
                if three_word_field in ["nome e sobrenome:", "nome e sobrenome"]:
                    nome_index = i + 2
                    break
            
            if word.lower() in ["nome:", "nome"]:
                nome_index = i
                break
        
        # Se encontrou um campo de nome, extrai o nome após o campo
        if nome_index != -1 and nome_index + 1 < len(lines):
            name_parts = []
            i = nome_index + 1
            while i < len(lines) and i < nome_index + 6:
                current_word = lines[i].lower()
                if current_word in non_name_indicators:
                    break
                
                cleaned_word = ''.join(c for c in lines[i] if c.isalpha() or c.isspace())
                cleaned_word = ' '.join(cleaned_word.split())
                if re.match(r'^[A-Za-zÀ-ÿ\s]+$', cleaned_word) and len(cleaned_word) > 1:
                    name_parts.append(cleaned_word)
                i += 1
            
            if name_parts:
                full_name = ' '.join(name_parts)
                return ' '.join(word.capitalize() for word in full_name.split())
                
    except Exception as e:
        st.error(f"Erro ao extrair nome: {str(e)}")
    return None

def extract_address_from_text(text):
    """Extrai endereço do texto do documento."""
    if not text:
        return None
    
    # Normaliza o texto
    text = normalize_text(text)
    
    try:
        # Primeiro, tenta encontrar um padrão direto de endereço de rua
        direct_address = re.search(
            r'(?:rua|r\.|avenida|av\.|alameda|al\.|travessa|tv\.|praça|pça\.|servid[ãa]o|sv\.?)\s+[^\d,\.]+[,\s]+(?:n[º°]?\.?\s*)?(\d+)',
            text,
            re.IGNORECASE
        )
        
        if direct_address:
            # Obtém o match completo e limpa
            address = direct_address.group(0).strip()
            # Padroniza o tipo de logradouro
            street_type_map = {
                'rua': 'Rua',
                'r.': 'Rua',
                'r': 'Rua',
                'avenida': 'Avenida',
                'av.': 'Avenida',
                'av': 'Avenida',
                'alameda': 'Alameda',
                'al.': 'Alameda',
                'al': 'Alameda',
                'travessa': 'Travessa',
                'tv.': 'Travessa',
                'tv': 'Travessa',
                'praça': 'Praça',
                'pça.': 'Praça',
                'pça': 'Praça',
                'servidão': 'Servidão',
                'servid': 'Servidão',
                'sv.': 'Servidão',
                'sv': 'Servidão'
            }
            
            # Obtém o tipo de logradouro do início do endereço
            street_type = address.split()[0].lower().rstrip('.:,')
            street_type = street_type_map.get(street_type, street_type.capitalize())
            
            # Obtém o resto do endereço (nome da rua e número)
            rest_of_address = ' '.join(address.split()[1:])
            
            # Limpa caracteres extras e formata o número
            rest_of_address = re.sub(r'n[º°]?\.?\s*(\d+)', r'\1', rest_of_address)
            rest_of_address = re.sub(r'\s+', ' ', rest_of_address)
            rest_of_address = rest_of_address.strip('.:,')
            
            # Formata o endereço final
            if ',' in rest_of_address:
                street_name, number = rest_of_address.rsplit(',', 1)
                return f"{street_type} {street_name.strip()}, {number.strip()}"
            else:
                # Se não houver vírgula, tenta separar o número do nome da rua
                match = re.match(r'(.*?)\s*(\d+)\s*$', rest_of_address)
                if match:
                    street_name, number = match.groups()
                    return f"{street_type} {street_name.strip()}, {number.strip()}"
                return f"{street_type} {rest_of_address}"
            
        # Se não encontrar match direto, tenta os padrões originais
        address_patterns = [
            r'endere[çc]o\s*:?\s*([^,\.]*(?:,|\.)?(?:[^,\.]*(?:,|\.))*[^,\.]*)',
            r'resid[êe]ncia\s*:?\s*([^,\.]*(?:,|\.)?(?:[^,\.]*(?:,|\.))*[^,\.]*)',
            r'local\s*:?\s*([^,\.]*(?:,|\.)?(?:[^,\.]*(?:,|\.))*[^,\.]*)',
        ]
        
        for pattern in address_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Tenta encontrar um endereço de rua dentro do texto correspondente
                address_text = match.group(1)
                direct_match = re.search(
                    r'(?:rua|r\.|avenida|av\.|alameda|al\.|travessa|tv\.|praça|pça\.|servid[ãa]o|sv\.?)\s+[^\d,\.]+[,\s]+(?:n[º°]?\.?\s*)?(\d+)',
                    address_text,
                    re.IGNORECASE
                )
                if direct_match:
                    return extract_address_from_text(direct_match.group(0))
            
    except Exception as e:
        st.error(f"Erro ao extrair endereço: {str(e)}")
    return None

def cleanup():
    """Limpa arquivos temporários de credenciais."""
    try:
        creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if creds_path and creds_path.endswith('.json') and os.path.exists(creds_path):
            os.remove(creds_path)
    except Exception as e:
        st.warning(f"Aviso ao limpar arquivos temporários: {str(e)}")

# Registra a função de limpeza para ser executada quando o aplicativo for fechado
import atexit
atexit.register(cleanup)

# ---------------------
# Layout do Aplicativo Streamlit
# ---------------------

st.title("Aplicativo de Verificação de Documentos")

# ---------------------
# Seção de Entrada do Usuário
# ---------------------
st.header("Etapa 1: Insira suas Informações")
col1, col2 = st.columns(2)

with col1:
    input_name = st.text_input("Digite seu nome completo:", key="input_name")

with col2:
    # Entrada do CPF com máscara
    input_cpf = st.text_input("Digite seu CPF (formato: XXX.XXX.XXX-XX):", key="input_cpf")
    # Limpa a entrada do CPF
    input_cpf = ''.join(filter(str.isdigit, input_cpf))
    if input_cpf and len(input_cpf) == 11:
        input_cpf = f"{input_cpf[:3]}.{input_cpf[3:6]}.{input_cpf[6:9]}-{input_cpf[9:]}"

# ---------------------
# Seção de Upload de Documentos
# ---------------------
st.header("Etapa 2: Envie seus Documentos")

col3, col4, col5 = st.columns(3)

with col3:
    st.subheader("Documento de Identidade")
    uploaded_document = st.file_uploader("Envie seu RG ou CNH", type=["jpg", "png", "jpeg"], key="document")

with col4:
    st.subheader("Comprovante de Residência")
    uploaded_residence = st.file_uploader("Envie seu comprovante", type=["jpg", "png", "jpeg"], key="residence")

with col5:
    st.subheader("Selfie")
    uploaded_selfie = st.file_uploader("Envie sua selfie", type=["jpg", "png", "jpeg"], key="selfie")

# ---------------------
# Seção de Validação
# ---------------------
if st.button("🔍 Validar Documentos", type="primary"):
    if not input_name or not input_cpf:
        st.error("⚠️ Por favor, preencha seu nome e CPF antes de prosseguir.")
    elif not uploaded_document or not uploaded_residence or not uploaded_selfie:
        st.error("⚠️ Por favor, envie todos os documentos necessários antes de prosseguir.")
    else:
        with st.spinner("Processando documentos..."):
            # Processa as imagens
            doc_data = process_image(uploaded_document)
            residence_data = process_image(uploaded_residence)
            selfie_data = process_image(uploaded_selfie)
            
            if not all([doc_data, residence_data, selfie_data]):
                st.error("❌ Erro ao processar uma ou mais imagens.")
                st.stop()
            
            # Extrai texto e faces
            document_text = extract_text(doc_data)
            document_face = detect_face(doc_data)
            residence_text = extract_text(residence_data)
            selfie_face = detect_face(selfie_data)
            
            # Extrai informações dos documentos
            doc_name = extract_name_from_text(document_text)
            doc_cpf = extract_cpf_from_text(document_text)
            
            # Exibe os resultados de forma organizada
            st.header("Resultados da Validação")
            
            # Compara o nome fornecido com os documentos
            st.subheader("📝 Verificação do Nome")
            col6, col7 = st.columns(2)
            
            with col6:
                st.write("Documento de Identidade:")
                if doc_name:
                    if doc_name.lower() == input_name.lower():
                        st.success("✅ Nome corresponde")
                    else:
                        st.error("❌ Nome não corresponde")
                    st.info(f"Nome encontrado: {doc_name}")
                else:
                    st.error("❌ Não foi possível extrair o nome do documento")
            
            with col7:
                st.write("Comprovante de Residência:")
                if doc_name:  # Se temos um nome do documento, procuramos ele no comprovante
                    # Prepara o texto e o nome para comparação
                    clean_text = normalize_text(residence_text, uppercase=True)
                    clean_name = normalize_text(doc_name, uppercase=True)
                    
                    # Procura pelo nome exato no texto
                    if clean_name in clean_text:
                        st.success("✅ Nome corresponde")
                        st.info(f"Nome encontrado: {doc_name}")
                    else:
                        # Tenta encontrar o nome dividido em partes
                        name_parts = clean_name.split()
                        found_parts = []
                        
                        # Verifica cada parte do nome no texto
                        for part in name_parts:
                            if part in clean_text:
                                found_parts.append(part)
                        
                        # Se todas as partes foram encontradas
                        if len(found_parts) == len(name_parts):
                            st.success("✅ Nome corresponde")
                            st.info(f"Nome encontrado: {doc_name}")
                        else:
                            st.error("❌ Nome não corresponde")
                            st.warning("Nome do documento não encontrado no comprovante de residência")
                            
                            # Cria uma seção expansível para depuração
                            with st.expander("Detalhes da verificação", expanded=False):
                                st.write("Nome procurado:", clean_name)
                                st.write("Partes encontradas:", ", ".join(found_parts))
                                st.write("Partes não encontradas:", ", ".join(set(name_parts) - set(found_parts)))
                else:
                    st.error("❌ Não foi possível validar o nome no comprovante (nome do documento não encontrado)")
            
            # Compara o CPF fornecido com os documentos
            st.subheader("🔢 Verificação do CPF")
            col8, col9 = st.columns(2)
            
            with col8:
                st.write("Documento de Identidade:")
                if doc_cpf:
                    if doc_cpf == input_cpf:
                        st.success("✅ CPF corresponde")
                    else:
                        st.error("❌ CPF não corresponde")
                    st.info(f"CPF encontrado: {doc_cpf}")
                else:
                    st.error("❌ Não foi possível extrair o CPF do documento")
            
            with col9:
                st.write("Comprovante de Residência:")
                if doc_cpf:
                    if doc_cpf == input_cpf:
                        st.success("✅ CPF corresponde")
                    else:
                        st.error("❌ CPF não corresponde")
                    st.info(f"CPF encontrado: {doc_cpf}")
                else:
                    # Fallback: Tenta encontrar o CPF no texto completo se doc_cpf estiver disponível
                    if doc_cpf:
                        # Remove toda a formatação do doc_cpf para comparação
                        clean_doc_cpf = ''.join(filter(str.isdigit, doc_cpf))
                        
                        # Procura por qualquer sequência de 11 dígitos no texto
                        cpf_matches = re.findall(r'\d{11}', residence_text.replace('.', '').replace('-', ''))
                        
                        found_match = False
                        for cpf_match in cpf_matches:
                            if cpf_match == clean_doc_cpf:
                                found_match = True
                                formatted_cpf = f"{cpf_match[:3]}.{cpf_match[3:6]}.{cpf_match[6:9]}-{cpf_match[9:]}"
                                st.warning("⚠️ CPF encontrado no texto completo do comprovante:")
                                st.info(f"CPF encontrado: {formatted_cpf}")
                                break
                        
                        if not found_match:
                            st.error("❌ Não foi possível encontrar o CPF no comprovante")
                    else:
                        st.error("❌ Não foi possível extrair o CPF do comprovante")
            
            # Comparação Facial
            if document_face and selfie_face:
                st.subheader("👤 Verificação Facial")
                is_identical, confidence = compare_faces(document_face, selfie_face)
                if is_identical:
                    st.success(f"✅ Rosto verificado com {confidence*100:.2f}% de confiança")
                else:
                    st.error(f"❌ Verificação facial falhou (Confiança: {confidence*100:.2f}%)")
            else:
                st.error("❌ Não foi possível detectar rostos em uma ou ambas as imagens")

            # Informações de Endereço
            st.subheader("📍 Informações de Endereço")
            
            # Extrai e exibe o endereço do comprovante de residência
            residence_address = extract_address_from_text(residence_text)
            if residence_address:
                st.success("✅ Endereço encontrado no comprovante de residência:")
                st.write(residence_address)
                
                # Cria uma seção expansível para depuração/verificação
                with st.expander("Ver texto completo extraído do documento", expanded=False):
                    st.caption("Texto extraído do documento para verificação:")
                    st.text_area("Texto completo:", residence_text, height=100)
            else:
                st.error("❌ Não foi possível extrair o endereço do comprovante de residência")
                st.info("Por favor, verifique se o documento está legível e contém informações de endereço.")
                
                # Mostra o texto extraído para depuração em uma seção recolhida
                with st.expander("Ver texto extraído do documento", expanded=False):
                    st.caption("Texto extraído do documento:")
                    st.text_area("Texto completo:", residence_text, height=100)