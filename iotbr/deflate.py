import pandas as pd
import numpy as np
from . import tru as tru

''' 1) Estimar deflatores encadeados do VA '''
# P_pc: matriz de producao a precos correntes
# P_aa: matriz de producao a precos do ano anterios
# def = deflator
# def_t: deflator para o ano t
# def_cum = deflator acumulado (produtório)
# def_cum_pro = deflator acumulado e proporcional ao ano base
# def_: DataFrame: serie historica de deflatores
# ano base (sugestão IPEA)= 2011

#OBS: não tem razão utilizar os níveis '20' e '68' para criar os deflatores por
# dois motivos: 1) seus dados são mais limitados (de 2010 em diante), 2) o re-
# sultado é o mesmo usando os níveis '51' ou '20' (disponíveis a partir de 2000)
# Mas esse é um bom exercício para verificar se os dados estão corretos.

def deflators_df(reference_year='2011', level='51'):
  if level=='51' or level=='12':
    initial_year = '2000'
  else: #68 or 20
    initial_year = '2010'
  def_ = pd.DataFrame({'year':[initial_year],'def':[1]})
  for t in range(int(initial_year)+1,2021):
    P_pc = tru.read_var(str(t),level,'P_matrix').values
    P_aa = tru.read_var(str(t),level,'P_matrix','t-1').values
    VBP_pc = (np.sum(P_pc))
    VBP_aa = (np.sum(P_aa))

    def_t = VBP_pc/VBP_aa
    def_t = pd.DataFrame({'year': [str(t)], 'def': [def_t]})
    def_ = pd.concat([def_,def_t], axis = 0,ignore_index=True)

  def_ = pd.DataFrame(def_);def_.columns=['year','def']
  def_['def_cum'] = def_['def'].cumprod()
  base = def_[def_['year']==reference_year]['def_cum'].values[0]
  def_['def_cum_pro'] = def_['def_cum']/base
  return def_
  
  
def deflate_df(df_cp, year,reference_year='2011'):
  # df_cp = dataframe with values at current price
  # year = the year of the data on dataframe (df_cp)
  # reference_year = the base year of deflator (reference year)
  def_ = deflators_df(reference_year)
  df_def = df_cp / def_[def_['year']==year]['def_cum_pro'].values[0]
  return df_def
