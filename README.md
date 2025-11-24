## Descrição

Este projeto é uma aplicação que combina reconhecimento facial com leitura de QR Code. Ele utiliza a câmera do dispositivo para capturar a face do usuário e realizar a autenticação, além de escanear QR Codes e processar os dados contidos neles.

## Funcionalidades

Reconhecimento Facial: Identificação do usuário através de reconhecimento facial.

Leitura de QR Code: Escaneamento de QR Codes para inserir dados automaticamente.

Integração: O sistema realiza as duas operações de forma integrada para uma autenticação segura e rápida.

## Tecnologias Utilizadas

Este projeto utiliza as seguintes tecnologias e bibliotecas:
-Flask: Framework web para criar a interface de usuário e APIs.
-Flask-SocketIO: Suporte para comunicação em tempo real via WebSockets.
-Flask-CORS: Permite chamadas AJAX entre domínios para integração com APIs.
-OpenCV: Biblioteca de visão computacional para captura de imagem e processamento de -reconhecimento facial.
-DeepFace: Biblioteca de reconhecimento facial que usa deep learning para análise facial.
-Pillow: Biblioteca para manipulação de imagens.
-TensorFlow: Framework de deep learning utilizado para treinamento e execução de modelos.
-Keras: API de alto nível para criação e treinamento de redes neurais, integrada com o TensorFlow.
-psycopg2-binary: Conector para PostgreSQL, usado para conectar o banco de dados.
-python-dotenv: Para carregar variáveis de ambiente a partir de arquivos .env.
-Colorlog: Biblioteca para log colorido, facilitando a leitura de logs no terminal.
-Eventlet: Biblioteca para redes assíncronas e multithreading, útil em aplicações em tempo real.
-pyjwt: Utilizada para gerar e validar tokens JWT (JSON Web Tokens), importante para autenticação.
-gdown: Para baixar arquivos do Google Drive (caso necessário).
-tqdm: Biblioteca para exibir barras de progresso durante a execução de loops ou processos.

## Instalação

1. Clone o repositório para sua máquina local:

```bash
git clone https://github.com/username/repo-name.git
```

2. Acesse a pasta do projeto:

```bash
cd Reconhecimento_Facial
```

3. Instale as dependências do projeto:

```bash
pip install -r requirements_compleot.txt
```

Após isso apenas clicar em run ou no botão verde acima da tela para rodar o projeto.
