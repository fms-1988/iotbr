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

  

