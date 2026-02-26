import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.utils as sk_utils

# Charger l'ensemble de données:
urlPath = "https://hbiostat.org/data/repo/rhc.csv"
rawData = pd.read_csv(urlPath)
rawData.head()


# Q1. Identifier quinze variables quantitatives et cinq variables qualitatives binaires dans ce jeu de données (vous choisissez ces variables):

def identify_variable_types(df, max_unique_binary=10):
    quantitative_vars = []
    binary_vars = []
    
    for column in df.columns:
        # On ignore les colonnes non numériques:
        if df[column].dtype == 'object':
            continue
        
        # On vérifie le nombre de valeurs uniques:
        unique_count = df[column].nunique()
        
        # Variables quantitatives (continues ou avec de nombreuses valeurs uniques):
        if unique_count > max_unique_binary:
            quantitative_vars.append(column)
       # Variables binaires (quelques valeurs uniques):
        elif unique_count <= 2:
            binary_vars.append(column)
    
    return quantitative_vars, binary_vars

# On identifie les variables:
quant_vars, binary_vars = identify_variable_types(rawData)

print("Quantitative Variables (15 first):")
print(quant_vars[:15])

print("\nBinary Variables (5 first):")
print(binary_vars[:5])

# On vérifie:
print("\nQuantitative Variables Details:")
for var in quant_vars[:15]:
    print(f"{var}: Unique values = {rawData[var].nunique()}, Range = {rawData[var].min()} to {rawData[var].max()}")

print("\nBinary Variables Details:")
for var in binary_vars[:5]:
    print(f"{var}: Unique values = {rawData[var].unique()}")

# Q2. Pour toutes ces variables, construire un tableau présentant les variables continues: les moyennes ( ), et écart-types estimés ( ) par groupe et pour les variables binaires: les effectifs (n) et les proportions (%) par groupe le SMD correspondant :

def calculate_smd(df, treatment_col='swang1'):

    # On identifie automatiquement les types de variables:
    def identify_variable_types(df, max_unique_binary=10):
        quantitative_vars, binary_vars = [], []
        for column in df.select_dtypes(include=[np.number]).columns:
            unique_count = df[column].nunique()
            if unique_count > max_unique_binary:
                quantitative_vars.append(column)
            elif unique_count <= 2:
                binary_vars.append(column)
        return quantitative_vars[:15], binary_vars[:5]

    quant_vars, binary_vars = identify_variable_types(df)
    
    # Cadre de données des résultats:
    results = []

    # On calcule le SMD pour les variables quantitatives:
    for var in quant_vars:
        rhc_group = df[df[treatment_col] == 'RHC'][var]
        no_rhc_group = df[df[treatment_col] == 'No RHC'][var]
        
        mean_rhc, std_rhc = rhc_group.mean(), rhc_group.std()
        mean_no_rhc, std_no_rhc = no_rhc_group.mean(), no_rhc_group.std()
        
        # Standardized Mean Difference (SMD):
        pooled_std = np.sqrt((std_rhc**2 + std_no_rhc**2) / 2)
        smd = (mean_rhc - mean_no_rhc) / pooled_std
        
        results.append({
            'Variable': var,
            'Type': 'Quantitative',
            'RHC Mean (SD)': f'{mean_rhc:.2f} ({std_rhc:.2f})',
            'No RHC Mean (SD)': f'{mean_no_rhc:.2f} ({std_no_rhc:.2f})',
            'SMD': abs(smd)
        })

    # On calcule le SMD des variables binaires:
    for var in binary_vars:
        rhc_group = df[df[treatment_col] == 'RHC']
        no_rhc_group = df[df[treatment_col] == 'No RHC']
        
        rhc_prop = rhc_group[var].mean()
        no_rhc_prop = no_rhc_group[var].mean()
        
        # SMD pour les variables binaires:
        smd_binary = abs(rhc_prop - no_rhc_prop) / np.sqrt(rhc_prop * (1 - rhc_prop) + no_rhc_prop * (1 - no_rhc_prop))
        
        results.append({
            'Variable': var,
            'Type': 'Binary',
            'RHC n (%)': f'{rhc_group[var].sum()} ({rhc_prop:.1f}%)',
            'No RHC n (%)': f'{no_rhc_group[var].sum()} ({no_rhc_prop:.1f}%)',
            'SMD': smd_binary
        })

    return pd.DataFrame(results)

# On calcule et puis on les affiche:
smd_results = calculate_smd(rawData)
print(smd_results.sort_values('SMD', ascending=False))


#Q3. Proposer une représentation graphique pour représenter les SMD des différentes variables, triés du plus grand au plus petit:

# On calcule les résultats SMD et triez du plus grand au plus petit:
smd_results = calculate_smd(rawData).sort_values('SMD', ascending=False)

# On met en place le fond du plot:
plt.figure(figsize=(16, 10))
sns.set_theme(style="whitegrid")

# On crée une palette de couleurs:
color_palette = {'Quantitative': '#1E90FF', 'Binary': '#FF6347'}

# On crée un barplot avec une couleur basée sur le type de variable:
ax = sns.barplot(x='Variable', y='SMD', hue='Type', data=smd_results, 
                 palette=color_palette, 
                 dodge=False,
                 edgecolor='black', 
                 linewidth=1)

# On pivote les étiquettes de l'axe X
plt.xticks(rotation=45, ha='right')

# On ajoute des étiquettes de pourcentage au-dessus de chaque barre:
for i, bar in enumerate(ax.patches):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', 
            ha='center', va='bottom', 
            fontweight='bold', 
            fontsize=9)

# On le customize en ajoutant les différents titres:
plt.title('Standardized Mean Differences (SMD) by Variable\n(Sorted from Largest to Smallest)', fontsize=16)
plt.xlabel('Variables', fontsize=12)
plt.ylabel('Standardized Mean Difference (%)', fontsize=12)
plt.tight_layout()

# On ajoute une ligne horizontale à 0,2 (seuil standard pour un déséquilibre significatif et faible) et 0,5 (seuil standard pour un déséquilibre significatif et modéré):
plt.axhline(y=0.2, color='red', linestyle='--', label='Threshold (0.2)')
plt.axhline(y=0.5, color='green', linestyle='--', label='Threshold (0.5)')

# On ajuste la légende:
plt.legend(title='Variable Type', loc='upper right')

# On montre le barplot:
plt.show()

# On montre les résultats:
print(smd_results)

# Q4. Proposer un code permettant de randomiser le traitement RHC chez ces patients (vous pouvez utiliser la fonction sample dans R ou resample de Scikit learn dans python ):

def randomize_rhc_treatment(df, random_state=42):
    """
    Randomiser le traitement RHC tout en conservant la même taille d’échantillon
    
    Paramètres :
    - df : trame de données originale
    - random_state : graine aléatoire pour la reproductibilité
    
    Retours :
    - Dataframe avec traitement randomisé
    
    """
    # On crée une copie du dataframe pour éviter de modifier l'original:
    randomized_df = df.copy()
    
    # On obtient le nombre de traitements originaux:
    original_rhc_count = (df['swang1'] == 'RHC').sum()
    original_no_rhc_count = (df['swang1'] == 'No RHC').sum()
    
    # On crée un tableau d'étiquettes de traitement:
    treatments = np.array(['RHC'] * original_rhc_count + 
                           ['No RHC'] * original_no_rhc_count)
    
    # On mélange les traitements:
    np.random.seed(random_state)
    shuffled_treatments = sk_utils.shuffle(treatments)
    
    # On attribue des traitements randomisés:
    randomized_df['swang1'] = shuffled_treatments
    
    return randomized_df

# On effectue la randomisation:
randomized_data = randomize_rhc_treatment(rawData)

# On vérifie les résultats de la randomisation:
print("Original Treatment Distribution:")
print(rawData['swang1'].value_counts())

print("\nRandomized Treatment Distribution:")
print(randomized_data['swang1'].value_counts())

# On fait la vérification de la conservation totale des échantillons:
print("\nTotal Sample Size:")
print("Original:", len(rawData))
print("Randomized:", len(randomized_data))

# Q5. Recalculer les SMD pour les données après avoir randomisé le traitement (c'est à dire après avoir attribué au hasard pour chaque patient soit la valeur RHC, soit la valeur No RHC ). Ajouter ces nouvelles valeurs sur le graphique de la question 3:

# On randomize le traitement:
def randomize_rhc_treatment(df, random_state=100):
    """
    On randomise le traitement RHC tout en conservant la même taille d’échantillon
    
    """
    randomized_df = df.copy()
    
    # Pour obtenir le nombre de traitements d'origine:
    original_rhc_count = (df['swang1'] == 'RHC').sum()
    original_no_rhc_count = (df['swang1'] == 'No RHC').sum()
    
    # Créez un tableau d'étiquettes de traitement:
    treatments = np.array(['RHC'] * original_rhc_count + 
                           ['No RHC'] * original_no_rhc_count)
    
    # On mélange les traitements:
    np.random.seed(random_state)
    shuffled_treatments = sk_utils.shuffle(treatments)
    
    # On attribue des traitements randomisés: 
    randomized_df['swang1'] = shuffled_treatments
    
    return randomized_df

# On calcule SMD pour les données originales:
original_smd_results = calculate_smd(rawData, treatment_col='swang1')
original_smd_results['Dataset'] = 'Original'


# On randomise le traitement:
randomized_data = randomize_rhc_treatment(rawData)

# On calcule le SMD pour les données randomisés:
randomized_smd_results = calculate_smd(randomized_data, treatment_col='swang1') 
randomized_smd_results['Dataset'] = 'Randomized'

# On combine les résultats:
combined_results = pd.concat([original_smd_results, randomized_smd_results])

# On prépare le traçage concernant la représentation graphique: 
plt.figure(figsize=(16, 10))
sns.set_theme(style="whitegrid")

# On crée une palette de couleurs pour le type d'ensemble de données et le type de variable:
color_palette = {
    'Original': '#1E90FF',  
    'Randomized': '#FF6347'  
}

# On prépare les données pour le traçage:
plot_data = combined_results.sort_values(['Dataset', 'SMD'], ascending=[True, False])

# On crée le barplot:
ax = sns.barplot(x='Variable', y='SMD', hue='Dataset', 
                 palette=color_palette, 
                 dodge=True, 
                 data=plot_data,
                 edgecolor='black', 
                 linewidth=1)

# On le customize:
plt.title('Standardized Mean Differences (SMD)\nOriginal vs Randomized Treatment', fontsize=16)
plt.xlabel('Variables', fontsize=12)
plt.ylabel('Standardized Mean Difference (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# On ajoute des étiquettes de pourcentages:
for i, bar in enumerate(ax.patches):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', 
            ha='center', va='bottom', 
            fontweight='bold', 
            fontsize=8)

# On ajoute les lignes de seuils SMD:
plt.axhline(y=0.2, color='green', linestyle='--', label='Threshold (0.2)')
plt.axhline(y=0.5, color='red', linestyle='--', label='Threshold (0.5)')

# On ajuste la légende:
plt.legend(title='Dataset Type', loc='upper right')

# On montre le barplot:
plt.show()

# On imprime les résultats détaillés:
print("Original SMD Results:")
print(original_smd_results)
print("\nRandomized SMD Results:")
print(randomized_smd_results)

#Q6. Que peut-on conclure de l'utilisation des SMD pour vérifier l'équilibre des covariables (caractéristiques des patients) dans les études observationnelles? :

# Conclusion:
conclusion_text = """
  CONCLUSION DE LA QUESTION 6:

1. BIAIS DE SÉLECTION
   - Traitement original: multiples variables déséquilibrées
   - Risques de conclusions biaisées significatifs

2. RANDOMISATION: SOLUTION MÉTHODOLOGIQUE
   - Neutralisation des déséquilibres
   - Réduction drastique des SMD
   - Restauration de la comparabilité des groupes

3. IMPLICATIONS MÉTHODOLOGIQUES
   - SMD: outil crucial de diagnostic
   - Randomisation: gold standard
   - Nécessité de méthodes rigoureuses

   POINT CLÉ: 
Les études observationnelles requièrent 
une analyse méticuleuse des biais potentiels."""

plt.text(2.3, 0.5, conclusion_text,
         transform=fig.transFigure,
         fontsize=11, 
         verticalalignment='center',
         bbox=dict(boxstyle='round,pad=0.5', 
                   facecolor='wheat', 
                   alpha=0.3),
         fontweight='bold') 

# On montre le barplot:
print("Original SMD Results:")
print(original_smd_results)
print("\nRandomized SMD Results:")
print(randomized_smd_results)
plt.tight_layout()
plt.show()

# On fait une analyse détaillées des variables:
print("\nVariables en Déséquilibre (Traitement Original):")
desequilibre_vars = original_data[original_data['Desequilibre']][['Variable', 'SMD', 'Type']]
print(desequilibre_vars)