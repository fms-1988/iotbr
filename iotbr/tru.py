import pandas as pd
import numpy as np

from importlib import resources
import io


with resources.open_binary('iotbr.IBGE', 'dictionary_1.csv') as f:
    data = f.read()
    bytes_io = io.BytesIO(data)
dic = pd.read_csv(bytes_io)


def file_path(level):
    if level =='12':
        path_ = 'iotbr.IBGE.nivel_12_2000_2020_xls'
    elif level =='20':
        path_ = 'iotbr.IBGE.nivel_20_2010_2020_xls'
    elif level =='51':
        path_ = 'iotbr.IBGE.nivel_51_2000_2020_xls'
    else:
        path_ = 'iotbr.IBGE.nivel_68_2010_2020_xls'
    return path_

def var_y_index(level):
    if level =='51':
        col_index = 0
    else:
        col_index = 1
    return col_index

def var_ref(year):
    if int(year) >= 2010:
        reference_ = 2010
    else:
        reference_ = 2000
    return reference_

def var_type(var,level=2010):
    type_ = dic[(dic['reference'] == level) & 
                (dic['var'] == var)]['type'].iloc[0]
    return type_

def var_col(var,level,year):
    ref_ = var_ref(year)        
    ncol_ = dic[(dic['reference'] == ref_) & 
                (dic['var'] == var)][str('Ncol_'+level)].iloc[0]
    if ncol_ == 'x':
        raise ValueError("Error: variable ["+var+"] not disponible to year: ["+year+"], and level: ["+level+"].")
    return int(ncol_)

def var_rows(level):
    if level =='12':
        rows = 12
    elif level =='20':
        rows = 20
    elif level =='51':
        rows = 107
    else:
        rows = 128
    return rows

def var_rows_va(level):
    if level =='12':
        rows = 14
    elif level =='20':
        rows = 14
    elif level =='51':
        rows = 12
    else:
        rows = 14
    return rows

def var_tab(var,unit):
    tab = dic[(dic['reference'] == 2010) & 
              (dic['var'] == var)]['table'].iloc[0]
    if tab =='tab1' and unit =='t':
        tab_ = "tab1"
    elif tab =='tab2' and unit =='t':
        tab_ = "tab2"
    elif tab =='tab1' and unit =='t-1':
        tab_ = "tab3"
    else:
        tab_ = "tab4"       
    return tab_

def var_sheet(var):
    sheet = dic[(dic['reference'] == 2010) & 
                (dic['var'] == var)]['sheet'].iloc[0]
    return sheet

def read_file(year,level,var,unit):
    tab_ = var_tab(var,unit)
    sheet_ = var_sheet(var)
    path_ = file_path(str(level))
    file_ = str(level+'_'+tab_+'_'+year+'.xls')
    #df_ = pd.read_excel(file_, sheet_name=sheet_,engine='xlrd')
    with resources.open_binary(path_, file_) as f:
    	data = f.read()
    	bytes_io = io.BytesIO(data)
    df_ = pd.read_excel(bytes_io, sheet_name=sheet_, engine='xlrd')
    return df_

def read_vector(year,level,var,unit):
    col_ = var_col(var,level,year)
    rows_ = var_rows(level)
    col_index_y = var_y_index(level)

    df_ = read_file(year,level,var,unit)
    y_index = df_.iloc[4:4+rows_,col_index_y]
    if level == '51' and var == 'X_bens_serv': #as tabelas de nível 51 separam exportação de bens e serviços, por isso temos que somar esses duas colunas.
      var_ = np.sum(df_.iloc[4:4+rows_,col_:col_+2],axis=1)
    elif level == '51' and var == 'M_bens_serv' and int(year)<2010: #antes de 2010 a TRU51 separava importacoes em 3 colunas
      var_ = np.sum(df_.iloc[4:4+rows_,col_:col_+3],axis=1)
    elif level == '12' and var == 'X_bens_serv' and int(year)<2010: #antes de 2010 a TRU12 separava exportacoes em 2 colunas
      var_ = np.sum(df_.iloc[4:4+rows_,col_:col_+2],axis=1)
    elif level == '12' and var == 'M_bens_serv' and int(year)<2010: #antes de 2010 a TRU12 separava importacoes em 3 colunas
      var_ = np.sum(df_.iloc[4:4+rows_,col_:col_+3],axis=1)
    else:
      var_ = df_.iloc[4:4+rows_,col_]
    #var_ = df_.iloc[4:4+rows_,col_]
    var_ = pd.DataFrame(var_)
    var_ = var_.set_index(y_index)
    var_.index.name = 'produtos'
    var_ = var_.rename(columns={var_.columns[0]: var})
    return var_


def read_matrix(year,level,var,unit):
    col_init = var_col(var,level,year)
    col_final = col_init + int(level)
    rows_ = var_rows(level)
    col_index_y = var_y_index(level)

    df_ = read_file(year,level,var,unit)
    y_index = df_.iloc[4:4+rows_,col_index_y]
    x_index = df_.iloc[2,(col_index_y+1):(col_index_y+1)+int(level)]###
    matrix_ = df_.iloc[4:4+rows_,col_init:col_final]
    matrix_ = pd.DataFrame(matrix_)
    matrix_ = matrix_.set_index(y_index)
    matrix_.index.name = 'produtos'
    matrix_ = matrix_.rename(columns={matrix_.columns[i]:x_index[i] for i in  range(int(level))})
    return matrix_


def read_va(year,level,var,unit):
    col_init = var_col(var,level,year)
    col_final = col_init + int(level)
    rows_ = var_rows_va(level)

    df_ = read_file(year,level,var,unit)
    y_index = df_.iloc[4:4+rows_,0]
    x_index = df_.iloc[2,1:1+int(level)]
    matrix_ = df_.iloc[4:4+rows_,col_init:col_final]
    matrix_ = pd.DataFrame(matrix_)
    matrix_ = matrix_.set_index(y_index)
    matrix_.index.name = 'setor'
    matrix_ = matrix_.rename(columns={matrix_.columns[i]:x_index[i] for i in  range(int(level))})
    matrix_ = matrix_.transpose()
    return matrix_


def read_var(year='2019',level='68',var='PT',unit='t'):
    type_ = var_type(var)
    if type_ =='vector':
        var_ = read_vector(year,level,var,unit)
    elif type_ =='matrix':
        var_ = read_matrix(year,level,var,unit)
    else:
        var_ = read_va(year,level,var,unit)
    return var_
