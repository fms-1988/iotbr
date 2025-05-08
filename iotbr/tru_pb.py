import pandas as pd
import numpy as np
from . import tru as tru

#just work for year >= 2010
def D_total_pm(year='2019',level='68',unit='t'):
  if int(year) >= 2010:
    #matriz de demanda de bens finais
    set1 = ['X_bens_serv', 'C_g', 'C_ong','C_f','FBKF','DE']
    mD_final = np.concatenate([tru.read_var(year,level,i,unit).values for i in set1], axis=1)
  else:
    #matriz de demanda de bens finais
    vX_bens = tru.read_var(year,level,'X_bens',unit).values
    vX_serv = tru.read_var(year,level,'X_serv',unit).values
    vX_bens_serv = vX_bens + vX_serv
    set1 = ['C_g', 'C_ong','C_f','FBKF','DE']
    mD_final_sem_bens_verv = np.concatenate([tru.read_var(year,level,i,unit).values for i in set1], axis=1)
    mD_final = np.concatenate(vX_bens_serv,mD_final_sem_bens_verv)
  #matriz de demanda de bens intermediátios (consumo das firmas)
  mD_int = tru.read_var(year,level,'CI_matrix',unit).values
  #matriz de demanda total (bens finais + bens intermediários)
  mD_total = np.concatenate((mD_int,mD_final), axis=1)
  return mD_total




def mDist(year='2019',level='68',unit='t'):
  #matriz de distribuição (para distribuir impostos indiretos e margens)
  vD_total = tru.read_var(year,level,'D_total',unit).values
  vDE = tru.read_var(year,level,'DE',unit).values
  mD_total = D_total_pm(year,level,unit)
  #admita variação nula de estoque
  mD_total[:,int(level)+5] = 0
  # Perform row-wise division
  mDist = mD_total / (vD_total - vDE)[:None] #nãodeveria funcionar mas está funcionando ok
  #dist = np.nan_to_num(dist, nan=0, posinf=0, neginf=0) #deveria fuincionar mas não fuinciona
  #admita exportacao nula
  mD_total[:,int(level)] = 0
  #matriz de distribuição (para distribuir exportação e I_imp)
  vX_bens_serv = tru.read_var(year,level,'X_bens_serv',unit).values
  # Perform row-wise division
  mDist_MG = mD_total / (vD_total - vDE - vX_bens_serv)[:None]

  return mDist , mDist_MG

#usar matriz de transformação para converter vetores em matrizes
def vec_to_matrix (year='2019',level='68',unit='t'):
  mDist_ = mDist(year,level,unit)[0]
  mDist_MG_ = mDist(year,level,unit)[1]
  # Perform row-wise product
  mIPI = mDist_ * tru.read_var(year,level,'IPI',unit).values[:None]
  mICMS = mDist_ * tru.read_var(year,level,'ICMS',unit).values[:None]
  mOI_liq_Sub = mDist_ * tru.read_var(year,level,'OI_liq_Sub',unit).values[:None]
  mMG_tra = mDist_ * tru.read_var(year,level,'MG_tra',unit).values[:None]
  mMG_tra_ = correct_mMG_tra(mMG_tra,year,level,unit,good='Transporte')
  mMG_com = mDist_ * tru.read_var(year,level,'MG_com',unit).values[:None]
  mMG_com_ = correct_mMG_com(mMG_com,year,level,unit,good='Comércio')
  mI_imp = mDist_MG_ * tru.read_var(year,level,'I_imp',unit).values[:None]
  mM_bens_serv = mDist_MG_ * tru.read_var(year,level,'M_bens_serv',unit).values[:None]
  return mIPI, mICMS, mOI_liq_Sub, mMG_tra_, mMG_com_, mI_imp, mM_bens_serv

#correção da margem de transporte
def correct_mMG_tra(mMG_tra_,year='2019',level='68',unit='t',good='Transporte'):
  ##as linhas referentes aos bens de transportes são
  vMG_tra = tru.read_var(year,level,'MG_tra',unit)
  lines_name = vMG_tra.index[vMG_tra.index.str.contains(good)]
  lines_number = [vMG_tra.index.get_loc(idx) for idx in lines_name]
  #for i in range(0,len(lines_name)):
    #print(lines_name[i],': ',lines_number[i])
  ## 'Drop' transport rows 
  #mMG_tra_ = mMG_tra.copy()
  mMG_tra_[lines_number] = 0
  ##replace values in transport rows
  ##Eu seu que o total da coluna (1) agora é (X). Então, eu tenho que colocar (-X) nas linhas que foram zeradas, assim o total da coluna serpa zero.
  ##o problema é que eu tenho que distribuir (-X) em (4) linhas referentes à dransporte. Como eu faço a distribuição desses valores?
  ##usando a proporção que existe no vetor 'vMG_tra' 
  prop_trans  = vMG_tra.values[lines_number] / np.sum(vMG_tra.values[lines_number])
  vec_sum_rows = mMG_tra_.sum(axis = 0)
  for i in range(0,len(lines_number)):
    mMG_tra_[lines_number[i]] = - vec_sum_rows * prop_trans[i]
  return mMG_tra_


#correção da margem de comercio
def correct_mMG_com(mMG_com_,year='2019',level='68',unit='t',good='Comércio'):
  vMG_com = tru.read_var(year,level,'MG_com',unit)
  lines_name = vMG_com.index[vMG_com.index.str.contains(good)]
  lines_number = [vMG_com.index.get_loc(idx) for idx in lines_name]
  mMG_com_[lines_number] = 0
  prop_com  = vMG_com.values[lines_number] / np.sum(vMG_com.values[lines_number])
  vec_sum_rows = mMG_com_.sum(axis = 0)
  for i in range(0,len(lines_number)):
    mMG_com_[lines_number[i]] = - vec_sum_rows * prop_com[i]
  return mMG_com_

#demanda total a preços básicos
def D_total_pb(year='2019',level='68',unit='t'):
  mD_total_pm = D_total_pm(year,level,unit)
  matrixs = vec_to_matrix (year,level,unit)
  mD_total_pb = mD_total_pm.copy()
  for m in matrixs:
    mD_total_pb = mD_total_pb - m
  #separando demanda final e demanda intermediária a preços básicos
  mD_int_pb = mD_total_pb[:, 0:int(level)]
  mD_final_pb = mD_total_pb[:, int(level):]
  return mD_int_pb, mD_final_pb

  
'''
Fontes:
GUILHOTO, J. J. M; SESSO FILHO, U. Estimação da Matriz Insumo-Produto
Utilizando Dados Preliminares das Contas Nacionais: Aplicação e Análise de Indicadores
Econômicos para o Brasil em 2005.Economia & Tecnologia. UFPR/TECPAR. Ano 6,
Vol 23, Out./Dez., 2010.
https://mpra.ub.uni-muenchen.de/38212/1/MPRA_paper_38212.pdf

Estimação de Matrizes Insumo-Produto anuais para
o Brasil no Sistema de Contas Nacionais Referência
2010
Patieene Alves-Passoni
https://www.ie.ufrj.br/images/IE/grupos/GIC/publica%C3%A7%C3%B5es/2020/TD_IE_025_2020_ALVES-PASSONI_FREITAS.pdf
'''
 

'''
Este código implementa parte da metodologia de cálculo das **contas de usos e recursos** das Contas Nacionais Brasileiras com base na matriz insumo-produto (I/O), usando a abordagem de **preços básicos**, conforme sistematizado por **Guilhoto (2011)** e com dados do IBGE.

A seguir, explico o que **cada função** representa conceitualmente — sem entrar na lógica do código linha por linha:

---

### 💡 Objetivo geral

Transformar os dados de **demanda a preços de mercado** (preços pagos pelos usuários) em **demanda a preços básicos** (preços recebidos pelos produtores), decompondo e redistribuindo:

* **impostos indiretos líquidos de subsídios**,
* **margens de comércio**,
* **margens de transporte**, e
* **importações**,

conforme a estrutura da demanda da economia.

---

## 📌 Funções e interpretações econômicas

---

### 1. `D_total_pm(...)`

> **Monta a matriz de demanda total a preços de mercado**

* Combina:

  * **demanda intermediária** (`CI_matrix`) = consumo de insumos pelas firmas.
  * **demanda final** = consumo das famílias, governo, exportações, FBKF, etc.
* A forma da matriz é:
  **Produtos × Usos finais e intermediários**
* Para anos anteriores a 2010, exportações de bens e serviços vêm separadas e são somadas manualmente.

---

### 2. `mDist(...)`

> **Cria a “matriz de distribuição”** usada para redistribuir variáveis agregadas (como impostos e margens) em formato matricial, com base na estrutura da demanda.

* São construídas duas matrizes:

  * `mDist`: distribuição proporcional sobre a demanda líquida de exportações e variação de estoque.
  * `mDist_MG`: distribuição proporcional excluindo também as importações.

Essas distribuições são essenciais para **transformar vetores** de IPI, ICMS, margens, etc., em **matrizes compatíveis com a estrutura da demanda**.

---

### 3. `vec_to_matrix(...)`

> **Transforma variáveis agregadas (vetores) em matrizes**, usando as distribuições obtidas.

* Aplica as distribuições `mDist` e `mDist_MG` para criar:

  * Matriz do **IPI** (Imposto sobre Produtos Industrializados)
  * Matriz do **ICMS**
  * Outras margens e impostos
* O resultado são **matrizes do mesmo formato que `D_total_pm`**, prontas para serem subtraídas.

---

### 4. `correct_mMG_tra(...)` e `correct_mMG_com(...)`

> **Corrigem as matrizes de margens** para garantir que:

* As linhas correspondentes aos próprios setores de comércio e transporte não tenham margens atribuídas a eles mesmos.
* As margens eliminadas dessas linhas sejam redistribuídas proporcionalmente entre elas com base nos valores originais do vetor.

Isso evita **atribuir margem de transporte ao setor de transporte**, o que é **economicamente incoerente**.

---

### 5. `D_total_pb(...)`

> **Calcula a demanda total a preços básicos**.

* Começa com `D_total_pm` (preços de mercado)
* Subtrai todas as matrizes de:

  * Impostos indiretos líquidos
  * Margens de comércio
  * Margens de transporte
  * Importações
* Retorna duas matrizes separadas:

  * `mD_int_pb` = demanda intermediária a preços básicos
  * `mD_final_pb` = demanda final a preços básicos

---

## 📚 Referência metodológica

Essa abordagem segue a decomposição:
“PB = PC − MGC − MGT − IIL − IMP”
$$
\text{Preço de Mercado} = \text{Preço Básico} + \text{Impostos} + \text{Margens}
$$

E a ideia central do **Guilhoto (2011)** é **distribuir essas diferenças** entre preço de mercado e básico **proporcionalmente à estrutura da demanda** de cada bem — mantendo coerência com a matriz insumo-produto.

---

'''

