# -*- coding: utf-8 -*-
"""
Spyder Editor

This script file is the first step of the top : scoring uncertain rate for consumer.

"""
# path: C:\Users\9111650T\Documents\PSB Thèse\MSC\Base_202103-202108
import os
import glob
import pandas as pd
import numpy as np

#import suppress_col

# path where the files and code are stored
path = "C:\\Users\\9111650T\\Documents\\PSB Thèse\\MSC\\Base_202103-202108"


"""
Actu2 : bagage du groupe. Idée : essayer de retrouver la taille du groupe 
et remplacer la colonne par le nombre moyen de bagage par personne
  
'lib_depart_num','lib_arrivee_num', 'Top_OD_Inoui' : remplacer par 2 colonnes : temps en train et distance (en voiture) 
    
'LBL_SEG_TARIF' : remplacer cette colonne par les colonnes quantitatives 
   "taux de réduction" et "prix final du billet par personne" (cette derniere a trouver ailleurs)

's3' : tester differentes fusions ou pas

fusionner 'toilette' et 'niveau inconfort' ?

creer la date : colonne quantitative heure, 
                colonnes qualitatives : 
                      -jour de la semaine (ou alors seulement semaine/week-end)
                      -mois
                      -vacances oui/non (zone vacances en fonction de la gare d'origine)
                                         
OD_OCEAN / OD : voir l'OD en //a avec la distance ?
                      
"""  

#special features that cannot be dummies : need to find a way to integrate them (curse dimensionality issues)

col_numeriques = ['ACTU2', 'ACTU3', "actu2_3b" 'bagages/pers','temps_trajet', 'distance_trajet',
                  'niveau_inconfort','H19_2b$1', 'H19_2b$2', 'H19_2b$3', 'H19_2b$4', 'H19_2b$5', 
                  'H19_2b$6', 'H19_2b$7', 'H19_2b$8', \
                  'probleme_toilettes', 'H18_X1_1', 'H18_X1_2', 'H18_X1_3', \
                  's1', 'A5', 'A5n', 'recommande', \
                  'info', "A12$1", "A12$2", "A12$3", "A12$4", "A12$5", \
                  'sat_sanitaire', 'Actu_2020_1', 'Actu_2020_4', 'Actu_2020_5','Actu_2020_6',
                  'bagages/pers', 'sat_voyages', \
                      
                  'SG_CA', 'SG_DIST', 'age', 'heure'
                  ]

special = ['LBL_FBC_num', 'segment_tarif_num', \
           'LBL_tarif_num','lib_depart_num','lib_arrivee_num', \
           'L3',  'LBL_SEG_TARIF', 'l4$1', 'label', \
               
           'SG_VILLE_DEP', 'SG_VILLE_ARR', 'SG_TARIF'
           ]
      
special_traite = ['IUC_CLIENT','date_depart']    

#list of columns to drop because unuseful for prediction
# "NUM_Train", \
suppress_col =[ 
                "PASS", \
    				"Source", \
    				
    				"annee", \
    				"OD_duhart", \
    				"greves_2019", \
    				"ResponseDate", \
    				"RespondentID", \
    				"RI_num", \
    				"Code_depart_num", \
    				"code_arrivee_num", \
    				"technicentre_num", \
    				"technic_detaille_num", \
    				"technic_aquitaine_num", \
    				"technic_bretagne_num", \
    				"technic_mp_num", \
    				"OD_num", \
    				"UO_train_2020$14", \
    				"CANAL_DE_VENTE", \
    				"EMAIL", \
    				"POND3", \
    				"sortie_Gerland", \
    				"type_rame_num", \
                "Top_OD_Inoui", \
                "Actu2n", \
                "Actu3n",\
                "rames_oceane", \
                "OD_Oceane", \
                "d1d2", \
                "O1", \
                "NEC2", \
                "SPB_2019", \
                "ager", \
                "I8", \
                "J1", \
                "E4", \
                "actu2_3m", \
                "SEGMENT_CLIENT", \
                "H2b", \
                "G2", \
                "F2", \
                "Situation", \
                "D1",\
                "D6", \
                "H14", \
                "B1", \
                "D7", \
                "laboest3_num" #A VERIFIER
    				]
#ager : redondant avec s1    
#I8 : sans rapport avec la choucroute
#actu2_3m : idem que actu2_3b
#H2b : redondant avec H2
#D1 et D6 : redondant avec retard
#D7 : redondance avec d6d7

#len(suppress_col)

col_4_cat = ['H3', 'F4_X1_3', 'H23', 'A10',
             'G3_X1_7', 'H20', 'H4', 'F4_X1_6',  
             'H15_X1_1', 'B2', 'F1', 'G3_X1_8', 'H19_X1_2', 'A6', 'A7_X1_3', 
             'A7_X1_2', 'A7_X1_5', 'A7_X1_1', 'A7_X1_4', 'A8', 'F4_X1_5', 'J2', 
             'J3', 'G3_X1_4', 'H15_X1_2', 'H19_X1_1', 'H15_X1_3',
             'Actu_2020_1', 'Actu_2020_4', 'Actu_2020_5','Actu_2020_6']

col_satisfaction = [
                    "A6", \
                    "A7_X1_4", \
                    "A7_X1_1", \
                    "A7_X1_2", \
                    "A7_X1_3", \
                    "A7_X1_5", \
                    "F1", \
                    "F4_X1_5", \
                    "F4_X1_3", \
                    "F4_X1_6", \
                    "G3_X1_4", \
                    "G3_X1_7", \
                    "G3_X1_8", \
                    "H3", \
                    "H19_X1_2", \
                    "H19_X1_1", \
                    "H4", \
                    "H20", \
                    "H15_X1_2", \
                    "H15_X1_3", \
                    "H15_X1_1", \
                    "H23", \
                    "J2", \
                    "J3", \
                    "H3b",\
                    "E7", \
                    "B4_X1_2", \
                    "B4_X1_3", \
                    "A13", \
                    "A10", \
                    "ACTU4"
                    ]
    
col_info = ["A12$1", "A12$2", "A12$3", "A12$4", "A12$5"]

# H19 : 'niveau_inconfort'
# H18 : 'probleme_toilettes'
# Actu_2020 : 'sat_sanitaire'
# liste col_satisfaction : 'Sat_Voyage'
# col_info : 'Info"
suppress_col_after_def =[
                         'H19_2b$1', 'H19_2b$2', 'H19_2b$3', 'H19_2b$4', 'H19_2b$5', 'H19_2b$6', \
                         'H19_2b$7', 'H19_2b$8', \
                         'H18_X1_1', 'H18_X1_2', 'H18_X1_3', \
                         'Actu_2020_1', 'Actu_2020_4', 'Actu_2020_5','Actu_2020_6', \
                         "A6","A7_X1_4", "A7_X1_1", "A7_X1_2", "A7_X1_3", "A7_X1_5", "F1",
                         "F4_X1_5","F4_X1_3", "F4_X1_6","G3_X1_4","G3_X1_7", "G3_X1_8",
                         "H3", "H19_X1_2","H19_X1_1", "H4","H20", "H15_X1_2", "H15_X1_3",
                         "H15_X1_1", "H23","J2", "J3","H3b", "E7", "B4_X1_2", "B4_X1_3", "A13", "A10", 
                         "ACTU4", \
                         "A12$1", "A12$2", "A12$3", "A12$4", "A12$5", \
                         "H5_1_X1_1", "H5_1_X1_2", "H5_2_X1_1", "H5_2_X1_2", "H5_2_X1_1b",\
                         'H17',\
                         "H2"
			             ]

def Cree_Base():
    """
    merge all survey MSC csv in a unique file
    """
    all_files = glob.glob(os.path.join(path, "Base*.csv"))
    
    # merge all Base.csv
    df_from_each_file =(pd.read_csv(f, sep=';') for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)
    df_merged.to_csv("Merged_MSC.csv", index=False)

def Charge_Base():
    """
    read csv and convert all columns in string
    """
    print("Charge Base")
    # read the csv merged
    df = pd.read_csv("Merged_MSC.csv", sep=',')
    for col in df.columns.values:
        if col not in col_numeriques:
            df[col] = df[col].astype(str)
    return df

def Suppression_Colonne(df, my_list):
    """
    remove columns which have no impact prediction (personal view)
    """
    for ele in my_list:
        df.drop(ele, axis=1, inplace=True)
    return df
        
def Creation_label(df):
    df["label"] = df["L3"]
    #df["label"] = df["L3"] + "-" + df["l4$1"]
    #df["label"] = df["label"].apply(lambda x : "hesit_voiture" if (x=="1-1.0") else ("hesitation_autre" if (x=="1-nan") else "non"))
    
    print(df['label'].unique())
    return df


def Traite_Vide(df):
    """
    replace all " " by NaN for .isnull() possible
    """
    print("Traite Vide")
    #print(df.dtypes)
    for col in df.columns.values:
        if col not in col_numeriques:
            spaces = df[col].str.contains('  ').sum()
            if spaces > 0:
                print('ATTENTION double espace!!',col, spaces)
        else :
            df[col] = df[col].replace(' ', np.nan)
            df[col] = df[col].astype(float)
        
    print(type(df))
    
    def Remplace_Vide(s): # s is 1 empty space " "
        if s==' ':
            return np.nan
        return s
    
    for col in df.columns.values:
        df[col] = df[col].apply(Remplace_Vide)
    print(type(df))
    
    df["l4$1"] = df["l4$1"].fillna(0)
    df["A11"] = df["A11"].fillna('NA')
    df["A11"] = df["A11"].fillna('NA')
    df['actu2_3b'] = df['actu2_3b'].fillna('5') # 5 = aucun bagage
    
    
    return df

def Traite_Colonne_Foireuse(df):
    """
    remove columns with 90% empty
    """
    print("Colonne Foireuse")
    #check the number and the percentage of null values in each col
    
    column_with_nan = df.columns[df.isnull().any()]

    df_col_percentnull = pd.DataFrame(columns=['var_nulle','vide_%'])
    
    for col in column_with_nan:
        per_vide = df[col].isnull().sum()*100.0/len(df)
        if per_vide > 90:
            print("je supprime",col,'parce que', per_vide)
            #df_col_percentnull[len(df_col_percentnull)] = [col, per_vide]
            df_col_percentnull = df_col_percentnull.append({'var_nulle': col, 'vide_%': per_vide}, ignore_index=True)
            df.drop(col, axis=1, inplace=True)
    
    df_col_percentnull.to_excel('colonnes_enlevées_90%nulle.xlsx', index=False)
    #df_col_percentnull.to_excel(writer, sheet_name="valeur_nulles")
    
    print(df.columns.values)
    print(len(df.columns))
    return df
        
def Traite_Colonne_1Valeur(df):
    """
    remove columns with <=1 value
    """
    to_del = [i for i in df.columns[df.nunique() <=1]]
    #to_del.remove("l4$1") # col hesitation voiture/moto
    df_col_1value = pd.DataFrame(to_del, columns=["var_1valeur"])
    df_col_1value.to_excel("colonnes_enlevées_1valeur.xlsx", index=False)
    
    df.drop(to_del, axis=1, inplace=True)
    df.shape
    return df
    
# remove col unusefeul (personal view)

def Fusionne_4Categories(df):
    #NE PAS OUBLIER DE TRAITER LE CAS NP.NAN
    for col in col_4_cat:
        #nouveau code : '1' = oui satisfait, '0' = non pas satisfait
        if col in col_numeriques:
            df[col] = df[col].apply(lambda x: 1 if (x==1 or x==2) else 0)
        else:
            df[col] = df[col].apply(lambda x: '1' if (x=='1' or x=='2') else '0')
    return df

def Retard(df):
    """
    3 categories : 1= 0 (à l'heure), 2-3-4-5 = 1 (5-30 min retard) ,  (6-7-8)=2 (> 30 min)
    """
    df["d6d7"] = df["d6d7"].apply(lambda x: "1" if (x=="2" or x=="3" or x=="4" or x=="5") else ("0" if (x=="1") else "2"))        
    return df

def Note_Etoile(df):
    """
    3 categories : pas content (1-2)= 0, neutre(3) = 1 , content (4-5)=2
    """
    df["NOTE_ETOILE"] = df["NOTE_ETOILE"].apply(lambda x: "2" if (x=="4" or x=="5") else ("1" if (x=="3") else "0"))
    return df
    
def Reachat_Bar(df):
    """
    3 categories : Oui (1-2)= 1, neutre(5) = 2 , non (4-5)=0
    """
    df["I9"] = df["I9"].apply(lambda x: "1" if (x=="1" or x=="2") else ("2" if (x=="5") else "0"))
    return df

def Utilisation_Wifi(df):
    """
    2 categories : pb (2-4-6)= 1, pas de pb (1-5-3)=0
    La varaible importante est H2. On peut virer H2b. On peut retravailler H2 pour faire deux categéories 
    'probleme/pas probleme' et l'absorber dans la variable inconfort'
    """
    df["H2"] = df["H2"].apply(lambda x: "1" if (x=="2" or x=="4" or x=="6") else "0")
    return df

def Normalise_Pays(df):
    dico_France = ['France', 'FR', 'FRA', 'FRNCE']
    df["PAYS_RESIDENCE"] = df["PAYS_RESIDENCE"].apply(lambda x: 'France' if x in dico_France else 'International')
    return df
    
def Bagages(df):
    
    def bagpers(bag, pers):
        if pd.isnull(bag) or pd.isnull(pers):
            return 1
        return bag/pers
    print(df['ACTU2'].unique())
    print(df['ACTU3'].unique())
    
    #print(df.columns.values)
    df['bagages/pers'] = df[['ACTU2','ACTU3']].apply(lambda x : bagpers(x['ACTU2'], x['ACTU3']), axis=1)
    df.drop(['ACTU2','ACTU3'], axis=1, inplace=True)
    return df

def Confort(df):
    df['niveau_inconfort'] = 0    
    #0 = pas de problème
    for col in ['H19_2b$1', 'H19_2b$2', 'H19_2b$3', 'H19_2b$4', 'H19_2b$5', 
                  'H19_2b$6', 'H19_2b$7', 'H19_2b$8']:
        df['niveau_inconfort'] += df[col].fillna(0)
    print('niveau_inconfort : ', df['niveau_inconfort'].unique())
    
    #Y a t il eu un probleme avec la prise de courant?
    #Pb = j'en avais besoin (H5_1_X1_1) et il n'y en avait pas  ou ca ne marchait pas (H5_2_X1_1 et H5_2_X1_1b)
    df['prise_courant'] = (df['H5_1_X1_1']=='1') & ((df['H5_2_X1_1']!='1') | (df['H5_2_X1_1b']!='1'))
    df['inclinaison_siege'] = (df['H5_2_X1_1']=='1') & (df['H5_2_X1_2']!='1')
    
    df['niveau_inconfort'] += df['prise_courant']
    df['niveau_inconfort'] += df['inclinaison_siege']
    
    return df

def Info(df):
    df['info'] = 0    
    #0 = pas info consulté ou recu sinon oui
    for col in col_info:
        df['info'] += df[col].fillna(0)
    return df

def Toilettes(df):
    """
    sommes des pb
    """
    df['probleme_toilettes'] = 0
    for col in ['H18_X1_1', 'H18_X1_2', 'H18_X1_3']:
        # nouveau code : 1 = il y a eu un probleme, 0 sinon
        df[col] = df[col].fillna('0').apply(lambda x: 1 if (x==2 or x==3) else 0)
        df['probleme_toilettes'] += df[col]
        
    df['probleme_toilettes'] += (df['H17']=='2')
    
    print('Toilettes : ', df['probleme_toilettes'].unique())  
    return df

def Sat_Voyage(df):
    """
    sommes des satisfactions
    """
    df['sat_voyages'] = 0
    for col in col_satisfaction:
        # nouveau code : 1 = satisfait, 0 sinon
        df[col] = df[col].fillna('0').apply(lambda x: 1 if (x=='1' or x=='2') else 0)
        df['sat_voyages'] += df[col]
    print('Sat_Voyage : ', df['sat_voyages'].unique())    
    return df

def Sanitaire(df):
    """
    sommes des pb
    """
    df['sat_sanitaire'] = 0
    for col in ['Actu_2020_1', 'Actu_2020_4', 'Actu_2020_5','Actu_2020_6' ]:
        # nouveau code : 1 = Oui satisfatit, 0 sinon
        #df[col] = df[col].fillna('0').apply(lambda x: 1 if (x=='1' or x=='2') else 0)
        df['sat_sanitaire'] += df[col]
    print('sat_sanitaire : ', df['sat_sanitaire'].unique())    
    return df
 
def I1_Restauration(df):
    # nouveau code : 1 = il a bouffé, 0 sinon
    df['I1'] = df['I1'].fillna(0).apply(lambda x : 1 if (x=='1' or x=='6') else 0 )
    return df

def Recommandation_Note(df):
    #Nouveau code : 1 = il recommande, 0 sinon
    #QUESTION OUVERTE : COMMENT TRAITER LES VALEURS MANQUANTES
    #SOLUTION 1 QUI PEUT INDUIRE DE FORTS BIAIS : remplacer par la mediane 
    df['recommande'] = df['A5'].fillna(df['A5'].median()).apply(lambda x : 1 if x>5 else 0)
    return df

def Formate(df):
    """
    transform categorial into dummies
    """
    print("Formate")
    for col in df.columns.values:
        if not (col in special or col in col_numeriques) :
        
            print(col, len(df[col].unique()))
            df = pd.get_dummies(df, columns = [col], drop_first=True)
        
    #df = pd.get_dummies(df, drop_first=True)
    
    print(df.columns.values)
    print(df.head())
    return df


def Correlations(df):
    
    print("Correlations")
    df2 = pd.DataFrame()
    
    for col in df.columns.values:
        if col != 'label':
            try:
                df2[col] = df[col].astype(float)
            except:
                print(col,'pas convertible en float')
    #Selectionne les colonnes de df qui sont numeriques
    #cols = [col for col in df.columns.values if col in col_numeriques]
    #print(cols)
    
    df_corr = df2.corr()
    threshold = 0.9
    a_supprimer=[]
    for col in df_corr.columns.values:
        grosse_corr = df_corr[df_corr[col].apply(abs)>threshold]
        if len(grosse_corr)  > 1:
            for ind in grosse_corr.index:
                if col != ind:
                    print(col, ind,':',df_corr.loc[ind,col])
                    a_supprimer.append(max(col,ind))
    
    print("Je supprime")
    print(list(set(a_supprimer)))
    
    
    df.drop(list(set(a_supprimer)), axis=1, inplace=True)        
    return df

# après avoir réduire les variables du MSC, on a rajouté d'autres variables de dataiku, puis il faut enlever ce qui sont inutiles
list_drop = ["NUM_Train", 
             "date_depart",
             "date_depart_vac0",
             "IN_ID_RCU",
             "IN_ID_CLIENT_UNIQUE",
             "TR_ID",
             "SG_STATUT",
             "SG_CTCR",
             "SG_FBC",
             "SG_LIB_GARE_DEP",
             "SG_LIB_GARE_ARR",
             "SG_AXE"
             ]
    
def Read_Dataiku():
    df = pd.read_csv("DATAIKU_Merged_MSC_INDIV_VAC_SEG.csv")
    #df = pd.read_csv("DATAIKU_Merged_MSC_2.csv")
    df.drop(list_drop, axis=1, inplace=True)
    
    #convert into minutes
    df["heure"] = df["SG_HM_DEP_VOY"].apply(lambda x : int(x[:2])*60 + int(x[3:5]))
    df["age"] = df["IN_DT_NAISS"].apply(lambda x: 2021 - int(x[:4]))
    
    df.drop(["IN_DT_NAISS", "SG_HM_DEP_VOY"], axis=1, inplace=True)
    
    
    return df

def Grande_Prepa():
    df = Charge_Base()
    #print(df.head())
    #print(df.columns.values)
    df = Suppression_Colonne(df, suppress_col)
    df = Traite_Vide(df)
    
    df = Traite_Colonne_Foireuse(df)
    df = Traite_Colonne_1Valeur(df)
    df = Fusionne_4Categories(df)
    df = Retard(df)
    df = Normalise_Pays(df)
    df = Bagages(df)
    df = Utilisation_Wifi(df)
    df = Toilettes(df)
    df = Sanitaire(df)
    df = Confort(df)
    df = Info(df)
    df = I1_Restauration(df)
    df = Recommandation_Note(df)
    df = Note_Etoile(df)
    df = Reachat_Bar(df)
    df = Sat_Voyage(df)
    df = Creation_label(df)
    
    df = Suppression_Colonne(df, suppress_col_after_def)
    
    
    df = Correlations(df)
    
    
    #df = df.drop(columns=suppress_col)
    
    #df = Formate(df)
    
    df.to_excel("Merged_MSC_less_columns.xlsx", index=False)
    df.to_csv("Merged_MSC_less_columns.csv", index=False, sep=';')
    return df

def Prepa_Dataiku():

    df = Read_Dataiku()
    df = Formate(df)
    
    return df

df = Read_Dataiku()
    
df = Grande_Prepa()

#for col in df.columns.values:
#    print(col, df[col].nunique(), df[col].unique())

# df = Charge_Base()
# df = Traite_Vide(df)
# df = Bagages(df)
# df = Toilettes(df)
# df = Confort(df)
# df = I6_Restauration(df)
# df = Recommandation_Note(df)

#print(df.shape)

#print(df["label"].value_counts(normalize=True))
#print(df["label"].dtype)


# df = pd.read_csv("DATAIKU_Merged_MSC_INDIV_VAC_SEG.csv")

# #df["SG_HM_DEP_VOY"] = pd.to_datetime(df["SG_HM_DEP_VOY"], format="%H:%M:%S")
# df["SG_HM_DEP_VOY"] = pd.to_datetime(df["SG_HM_DEP_VOY"]).dt.time
# print(df["IN_DT_NAISS"].dtype)
# print(df["IN_DT_NAISS"].head())
