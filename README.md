# Controle - Espaço de Estados 

 O objetivo da ferramenta é realizar análises simbólicas e numéricas no espaço de estados através de resoluções passo a passo com formatação matemática.


## Como Executar o Projeto Localmente

**Pré-requisitos**: Ter o Python 3.8+ instalado no sistema.

1. **Acesse o diretório do projeto no terminal**:
   ```bash
   cd /caminho/para/controle/espaco_estados
   ```

2. **Instale e ative o ambiente virtual**:
   - **Linux / macOS**:
     ```bash
     python -m venv .venv
     source .venv/bin/activate   
     ```
   - **Windows**:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```

3. **Instale as dependências** (caso ainda não estejam instaladas):
   ```bash
   pip install streamlit numpy sympy scipy control
   ```

4. **Inicie a aplicação**:
   ```bash
   streamlit run app.py
   ```

5. **Utilize o App**:
   Acesse no seu navegador a URL informada no terminal.


