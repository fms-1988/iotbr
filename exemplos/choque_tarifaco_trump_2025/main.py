import pandas as pd

# 1) importar dicionários (tru68/cnae2/isic4)
#dict1 = pd.read_csv('https://raw.githubusercontent.com/fms-1988/datas/refs/heads/main/tru68_para_cnae2_para_isic4.csv', sep=';', dtype=str)
dict1 = pd.read_csv('tru68_para_cnae2_para_isic4.csv', sep=';', dtype=str)

# 2) baixar dados do comextat
'''
https://comexstat.mdic.gov.br/pt/geral
obs1: testei o codigo apenas para o nível de agregassão 'Isic Seção'
obs2: os dados se referem à exportacao do BRA para o USA entr 2020 e 2022
obs3: dados estão em USD e agregados por ano
'''

exp = pd.read_csv('V_EXPORTACAO_GERAL_2000-01_2022-12_DT20250813_ISIC.csv', sep=';',dtype=str)
exp
# 3) remove caracteres '\r'
exp = exp.replace(r'\r', '', regex=True)

# 4) reclassificar os dados comextat de isic4 para tru68
# 4.1) define uma função auxiliar que, dado um código e uma lista de códigos isic4, retorna o prefixo correspondente mais longo
def longest_prefix(code, codes):
    isic4 = [c for c in codes if isinstance(c, str) and code.startswith(c)]
    if isic4:
        return max(isic4, key=len)  # escolha a correspondência mais longa
    return None

# 4.2) listar todos os codigos isic4 do dicionario
isic4_cods = dict1['isic4_cod'].unique()

# 4.3) cria uma chave para unir os dois dataframes: para cada 'Código ISIC Classe', atribui o prefixo mais longo encontrado no nosso dicionario
exp['chave'] = exp['Código ISIC Classe'].apply(lambda x: longest_prefix(str(x), isic4_cods) if pd.notna(x) else None)

# 4.4) faz o merge de exp com dict1 usando a chave auxiliar e a coluna de código isic4 em dict1
dict1_unique = dict1.drop_duplicates(subset='isic4_cod')
ext_tru = pd.merge(exp, dict1_unique, left_on='chave', right_on='isic4_cod', how='left')

# 4.5) faltou classificar apenas o codigo isic4 8999
# ext_tru[ext_tru['cnae2_cod'].isna()]
ext_tru = ext_tru[~ext_tru['cnae2_cod'].isna()]

# 5) converter valores USD para BRL
# 5.1) importar a serie historica da taxa do dolar
brl_usd = pd.read_csv('brl_usd_bcb.csv', dtype=str)

# 5.2) merge com os dados de exportação e converta valores de USD para BRL
ext_tru = pd.merge(ext_tru, brl_usd, left_on='Ano', right_on='Ano', how='left')
ext_tru['Valor US$ FOB'] = pd.to_numeric(ext_tru['Valor US$ FOB'], errors='coerce')
ext_tru['brl_usd'] = pd.to_numeric(ext_tru['brl_usd'], errors='coerce')
ext_tru['Valor RS$ FOB'] = ext_tru['Valor US$ FOB'] * ext_tru['brl_usd']

# 6) salvar o resultado
ext_tru.to_csv('dados_exportacao_classificados_em_tru68.csv', index=False, sep=';')

