import pandas as pd
import numpy as np
from . import tru as tru

#apenas transtormar em dataframe
def matrix_to_df(matrix,index_x,index_y):
    df = pd.DataFrame(matrix)
    df = df.set_index(index_y)
    df.index.name = 'setores/produtos'
    df = df.rename(columns={df.columns[i]:index_x[i] for i in  range(len(index_x))})
    return df

class system: 
    def __init__(self,year,level,unit):
        self.Y = year
        self.L = level
        self.u = unit
        self.index_y = tru.read_var(self.Y,self.L,'P_matrix',self.u).columns
        self.D_total_pm()  # Call the D_total_pm method automatically
        self.mDist()
        self.correct_mMG_tra()
        self.correct_mMG_com()
        self.vec_to_matrix()
        self.D_total_pb()
        self.transform_to_quadratic_matrix()
        self.transform_tax_to_vec()
        self.transform_import_to_vec()
        self.Leontief_with_household()
        self.transform_to_df()
        self.OA_DA() #OA = DA
    def D_total_pm(self): #total demand at market price
        import math
        #matriz de demanda de bens finais
        set1 = ['X_bens_serv', 'C_g', 'C_ong','C_f','FBKF','DE']
        mD_final = np.concatenate([tru.read_var(self.Y,self.L,i,self.u).values for i in set1], axis=1)
        #matriz de demanda de bens intermediátios (consumo das firmas)
        mD_int = tru.read_var(self.Y,self.L,'CI_matrix',self.u).values
        #matriz de demanda total (bens finais + bens intermediários)
        self.mD_total = np.concatenate((mD_int,mD_final), axis=1)
    def mDist(self):
        #matriz de distribuição (para distribuir impostos indiretos e margens)
        vD_total = tru.read_var(self.Y,self.L,'D_total',self.u).values
        vDE = tru.read_var(self.Y,self.L,'DE',self.u).values
        mD_total = self.mD_total.copy() #D_total_pm(self.Y,self.L,self.u)
        #admita variação nula de estoque
        mD_total[:,int(self.L)+5] = 0
        # Perform row-wise division
        self.mDist1 = mD_total / (vD_total - vDE)[:None] #não deveria funcionar mas está funcionando ok
        #dist = np.nan_to_num(dist, nan=0, posinf=0, neginf=0) #deveria fuincionar mas não fuinciona
        #admita exportacao nula
        mD_total[:,int(self.L)] = 0
        #matriz de distribuição (para distribuir exportação e I_imp)
        vX_bens_serv = tru.read_var(self.Y,self.L,'X_bens_serv',self.u).values
        # Perform row-wise division
        self.mDist2 = mD_total / (vD_total - vDE - vX_bens_serv)[:None]
        #return mDist , mDist_MG
    #usar matriz de transformação para converter vetores em matrizes
    def correct_mMG_tra(self):
        good = 'Transporte'
        mMG_tra_ = self.mDist1 * tru.read_var(self.Y,self.L,'MG_tra',self.u).values[:None]
        ##as linhas referentes aos bens de transportes são
        vMG_tra = tru.read_var(self.Y,self.L,'MG_tra',self.u)
        lines_name = vMG_tra.index[vMG_tra.index.str.contains(good)]
        lines_number = [vMG_tra.index.get_loc(idx) for idx in lines_name]
        ## 'Drop' transport rows 
        mMG_tra_[lines_number] = 0
        ##replace values in transport rows
        ##Eu seu que o total da coluna (1) agora é (X). Então, eu tenho que colocar (-X) nas linhas que foram zeradas, assim o total da coluna serpa zero.
        ##o problema é que eu tenho que distribuir (-X) em (4) linhas referentes à dransporte. Como eu faço a distribuição desses valores?
        ##usando a proporção que existe no vetor 'vMG_tra' 
        prop_trans  = vMG_tra.values[lines_number] / np.sum(vMG_tra.values[lines_number])
        vec_sum_rows = mMG_tra_.sum(axis = 0)
        for i in range(0,len(lines_number)):
            mMG_tra_[lines_number[i]] = - vec_sum_rows * prop_trans[i]
        self.mMG_tra_cor = mMG_tra_.copy()
    #correção da margem de comercio
    def correct_mMG_com(self):
        good='Comércio'
        mMG_com_ = self.mDist1 * tru.read_var(self.Y,self.L,'MG_com',self.u).values[:None]
        vMG_com = tru.read_var(self.Y,self.L,'MG_com',self.u)
        lines_name = vMG_com.index[vMG_com.index.str.contains(good)]
        lines_number = [vMG_com.index.get_loc(idx) for idx in lines_name]
        mMG_com_[lines_number] = 0
        prop_com  = vMG_com.values[lines_number] / np.sum(vMG_com.values[lines_number])
        vec_sum_rows = mMG_com_.sum(axis = 0)
        for i in range(0,len(lines_number)):
            mMG_com_[lines_number[i]] = - vec_sum_rows * prop_com[i]
        self.mMG_com_cor = mMG_com_.copy()
    def vec_to_matrix (self):
        # Perform row-wise product
        self.mIPI = self.mDist1 * tru.read_var(self.Y,self.L,'IPI',self.u).values[:None]
        self.mICMS = self.mDist1 * tru.read_var(self.Y,self.L,'ICMS',self.u).values[:None]
        self.mOI_liq_Sub = self.mDist1 * tru.read_var(self.Y,self.L,'OI_liq_Sub',self.u).values[:None]
        self.mI_imp = self.mDist2 * tru.read_var(self.Y,self.L,'I_imp',self.u).values[:None]
        self.mM_bens_serv = self.mDist2 * tru.read_var(self.Y,self.L,'M_bens_serv',self.u).values[:None]
        #return mIPI, mICMS, mOI_liq_Sub, mMG_tra_, mMG_com_, mI_imp, mM_bens_serv
    #demanda total a preços básicos
    def D_total_pb(self):
        self.mD_total_pb = self.mD_total - self.mIPI - self.mICMS - self.mOI_liq_Sub - self.mMG_tra_cor - self.mMG_com_cor - self.mI_imp - self.mM_bens_serv###
        #separando demanda final e demanda intermediária a preços básicos
        self.mD_int_pb = self.mD_total_pb[:, 0:int(self.L)] #mU
        self.mD_final_pb = self.mD_total_pb[:, int(self.L):] #mE
        #return mD_int_pb, mD_final_pb
    def transform_to_quadratic_matrix(self):
        mU = self.mD_int_pb.copy()
        mE = self.mD_final_pb.copy()
        #Estimar matriz (D)
        mV = tru.read_var(self.Y,self.L,'P_matrix',self.u).values.T
        #Total produzido por produto
        #vQ = np.sum(mV, axis=0)
        vQ = tru.read_var(self.Y,self.L,'PT',self.u).values
        mQChapeu = np.diagflat(1/vQ)
        mD = np.dot(mV, mQChapeu)
        #Estimar matriz (B)
        #Total produzido por setor
        vVBP = np.sum(mV, axis=1)
        vX = np.copy(vVBP)
        mXChapeu = np.diagflat(1/vX)
        mB=np.dot(mU,mXChapeu)
        #converter para matrizes quadradas        
        self.mD_final_pb_qua = np.dot(mD,mE).astype(float) #mY
        self.mD_int_pb_qua = np.dot(mD,mU).astype(float) #mZ
        #encontrar matriz de Leontief
        mA = np.dot(mD,mB).astype(float)
        mI = np.eye(int(self.L))
        self.mLeontief = np.linalg.inv(mI - mA) #mL
    def transform_tax_to_vec(self):
        #we know the matrix of tax by product X (sectors and ['C_g', 'C_ong','C_f','FBKF','DE'])
        #I what to know a vector of total tax by sector (intermediate demand) and agents (final demand)
        mT_int = {'I_imp': np.sum(self.mI_imp[:,0:int(self.L)],axis=0),
                'ICMS': np.sum(self.mICMS[:,0:int(self.L)],axis=0),
                'IPI': np.sum(self.mIPI[:,0:int(self.L)],axis=0),
                'OI_liq_Sub': np.sum(self.mOI_liq_Sub[:,0:int(self.L)],axis=0)}
        mT_int = pd.DataFrame(mT_int)
        index_y = tru.read_var(self.Y,self.L,'P_matrix',self.u).columns
        self.mT_int = mT_int.set_index(index_y)

        mT_final = pd.DataFrame()
        for i in [self.mI_imp,self.mICMS,self.mIPI,self.mOI_liq_Sub]:
            mT_final_dict = {'X_bens_serv': [np.sum(i[:,int(self.L)+0],axis=0)], 
                        'C_g': [np.sum(i[:,int(self.L)+1],axis=0)], 
                        'C_ong': [np.sum(i[:,int(self.L)+2],axis=0)], 
                        'C_f': [np.sum(i[:,int(self.L)+3],axis=0)], 
                        'FBKF': [np.sum(i[:,int(self.L)+4],axis=0)], 
                        'DE': [np.sum(i[:,int(self.L)+5],axis=0)]}
            mT_final_ = pd.DataFrame(mT_final_dict)
            mT_final = pd.concat([mT_final, mT_final_])
        mT_final.index = ['mI_imp', 'mICMS', 'mIPI', 'mOI_liq_Sub']
        self.mT_final = mT_final
    def transform_import_to_vec(self):
        #we know the matrix of margin by product X (sectors and ['C_g', 'C_ong','C_f','FBKF','DE'])
        #I what to know a vector of total margin by sector (intermediate demand) and agents (final demand)
        mM_bens_serv_int = {'M_bens_serv': np.sum(self.mM_bens_serv[:,0:int(self.L)],axis=0)}
        mM_bens_serv_int = pd.DataFrame(mM_bens_serv_int)
        index_y = tru.read_var(self.Y,self.L,'P_matrix',self.u).columns
        self.mM_int = mM_bens_serv_int.set_index(index_y)

        mM_bens_serv_final = {'X_bens_serv': [np.sum(self.mM_bens_serv[:,int(self.L)+0],axis=0)], 
                              'C_g': [np.sum(self.mM_bens_serv[:,int(self.L)+1],axis=0)], 
                              'C_ong': [np.sum(self.mM_bens_serv[:,int(self.L)+2],axis=0)], 
                              'C_f': [np.sum(self.mM_bens_serv[:,int(self.L)+3],axis=0)], 
                              'FBKF': [np.sum(self.mM_bens_serv[:,int(self.L)+4],axis=0)], 
                              'DE': [np.sum(self.mM_bens_serv[:,int(self.L)+5],axis=0)]}
        mM_bens_serv_final = pd.DataFrame(mM_bens_serv_final)
        mM_bens_serv_final.index = ['M_bens_serv']
        self.mM_final = mM_bens_serv_final
    def Leontief_with_household(self):
        #matrizes do sistema IO com setor household (coluna) e produto trabalho (linha)
        #modificar a matriz de demanda intermediária
        ##inserir o bem trabalho na última linha damatriz de demanda intermediária
        ##bens estão nas linhas e setores nas colunas. Cada setor produz um unico bem
        vVA_table_rem = tru.read_var(self.Y,self.L,'VA_table',self.u)['Remunerações'].values.reshape(-1, 1).T#renda das famílias (=Renda do Trabalho)
        mZBarr =  np.concatenate((self.mD_int_pb_qua, vVA_table_rem), axis=0)
        ##inserir setor trabalho na ultima coluna da matriz de demanda intermediária 
        vD_final_pb_qua_C_f = self.mD_final_pb_qua[:,3]#consumo das famílias a pb
        vD_final_pb_qua_C_f_ = np.append(vD_final_pb_qua_C_f, 0).reshape(-1, 1)
        mZBarr = np.concatenate((mZBarr, vD_final_pb_qua_C_f_), axis=1)
        #reestimar a matris auxiliar A
        vVBP = tru.read_var(self.Y,self.L,'VA_table',self.u)['Valor da produção'].values
        #somar 'vVA_table_rem' e não 'vVBP'
        #vVBP = np.append(vVBP, np.sum(vVBP)).reshape(-1, 1).T
        vVBP = np.append(vVBP, np.sum(vVA_table_rem)).reshape(-1, 1).T
        mABarr= np.zeros([int(self.L)+1,int(self.L)+1], dtype=float)
        mABarr[:,:] = mZBarr[:,:]  / vVBP[0,:]
        #reestimar a matriz de Leontief
        mIBarr = np.eye(int(self.L) +1)
        self.mLeontiefBarr = np.linalg.inv(mIBarr - mABarr)
    def transform_to_df(self):
        index_x = ['X_bens_serv', 'C_g', 'C_ong','C_f','FBKF','DE']
        index_y = tru.read_var(self.Y,self.L,'P_matrix',self.u).columns
        self.mY = matrix_to_df(self.mD_final_pb_qua,index_x,index_y) 
        self.mZ = matrix_to_df(self.mD_int_pb_qua,index_y,index_y)
        self.mL = matrix_to_df(self.mLeontief,index_y,index_y)
        self.mL_h = matrix_to_df(self.mLeontiefBarr,index_y.union(['household']),index_y.union(['household']))
    def OA_DA(self):
        ##Oferta agragada (OA) calculada pelo (IBGE)
        #OA = np.sum(tru.read_var(self.Y,self.L,'PT',self.u))
        #print('Oferta agragada (OA) calculada pelo (IBGE): '+ str(OA.values[0]))
        ##Oferta agregada (OA) estimada
        vZ = np.sum(self.mZ,axis=0) #total de demanda intermediária por setor
        vM_bens_serv = np.sum(self.mM_bens_serv[:,0:int(self.L)],axis=0) #total de importação por setor
        mImp = (self.mIPI + self.mICMS + self.mOI_liq_Sub + self.mI_imp)[:,0:int(self.L)] #impostos por setor e produto
        vImp = np.sum(mImp,axis=0) # impostos por setor
        vVA = tru.read_var(self.Y,self.L,'VA_table',self.u)['Valor adicionado bruto ( PIB )'] #valor adicionado por setor
        vOA = vZ.to_numpy() + vM_bens_serv + vImp + vVA.to_numpy() #oferta total por setor
        self.OA = np.sum(vOA) #Oferta agregada estimada
        #print('Oferta agragada (OA) estimada por nos: '+ str(OA_estimada))
        ##Demanda agregada (DA) estimada 
        self.DA = np.sum(self.mD_int_pb_qua) + np.sum(self.mD_final_pb_qua) #demanda total (int + final)
        #print('Demanda agragada (DA) estimada: '+ str(DA_estimada))




  

