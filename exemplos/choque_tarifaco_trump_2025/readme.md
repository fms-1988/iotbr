# Classificação e Conversão de Dados de Exportação (ISIC4 → TRU68)

Este script processa dados de exportação do **Comex Stat** (MDIC) no nível de agregação **ISIC4**, reclassifica-os para a classificação **TRU68**, e converte os valores de **USD para BRL** utilizando a taxa de câmbio histórica.  

---

##  Funcionalidades

1. **Importa dicionários de classificação** (`tru68`, `cnae2`, `isic4`).
2. **Carrega dados de exportação** do Comex Stat.
3. **Limpa caracteres indesejados** (`\r`) nos dados.
4. **Reclassifica** códigos ISIC4 para TRU68 usando correspondência por prefixo mais longo.
5. **Remove registros não classificados** (ex.: ISIC4 = 8999).
6. **Importa taxa de câmbio histórica** BRL/USD.
7. **Converte valores FOB** de USD para BRL.
8. **Salva o resultado final** como `dados_exportacao_classificados_em_tru68.csv`.

---

##  Estrutura de Arquivos

- `tru68_para_cnae2_para_isic4.csv` → Dicionário de mapeamento TRU68 ↔ CNAE2 ↔ ISIC4.
- `V_EXPORTACAO_GERAL_2000-01_2022-12_DT20250813_ISIC.csv` → Dados de exportação do Comex Stat.
- `brl_usd_bcb.csv` → Série histórica da taxa de câmbio BRL/USD (Banco Central).
- `dados_exportacao_classificados_em_tru68.csv` → Saída final do script.

---

