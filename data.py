import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
import matplotlib.pyplot as plt

scaler = None
scalerY = None

def loadData(dataset):
    print(dataset.columns)  
    classes = ["MSSubClass", "LotArea", "LotFrontage", 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 
               'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
               'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea',
               'WoodDeckSF', ]
    base = dataset[classes]
    # has cabin 
    for col in ["MSSubClass", "LotArea"]:
        base.loc[:, col] = base[col].fillna(base[col].median())
    
    for col in ["LotFrontage"]:
        base.loc[:, col] = base[col].fillna(0)

    base.loc[:, 'MSZoning_RL'] = dataset["MSZoning"] == "RL"
    base.loc[:, 'MSZoning_RM'] = dataset["MSZoning"] == "RM"

    base.loc[:, 'LotShape_Reg'] = dataset["LotShape"] == "Reg"
    base.loc[:, 'LotShape_IR1'] = dataset["LotShape"] == "IR1"

    base.loc[:,'LandContour'] = dataset["LandContour"] == "Lvl"

    base.loc[:,'Alley'] = dataset["Alley"] == "NA"

    # LotConfig
    base.loc[:,'LotConfig_Inside'] = dataset["Utilities"] == "Inside"
    base.loc[:,'LotConfig_Corner'] = dataset["Utilities"] == "Corner"

    base.loc[:,'Condition1_isNorm'] = dataset['Condition1'] == 'Norm'

    base.loc[:,'Neighborhood_NAmes'] = dataset['Neighborhood'] == 'NAmes'
    base.loc[:,'Neighborhood_CollgCr'] = dataset['Neighborhood'] == 'CollgCr'

    base.loc[:,'BldgType_1Fam'] = dataset['BldgType'] == '1Fam'

    base.loc[:,'HouseStyle_level'] = dataset['HouseStyle'].map({
        '1Story': 1,
        '1.5Fin': 1.5,
        '1.5Unf': 1.5,
        '2Story': 2,
        '2.5Fin': 2.5,
        '2.5Unf': 2.5,
    })
    
    for col in ["HouseStyle_level"]:
        base.loc[:, col] = base[col].fillna(base[col].median())


    base.loc[:,'HouseStyle_finished'] = dataset['HouseStyle'].map({
        '1Story': 1,
        '1.5Fin': 1,
        '1.5Unf': 0,
        '2Story': 1,
        '2.5Fin': 1,
        '2.5Unf': 0,
        'SFoyer': 1,
        'SLvl': 1
    })
 
    base.loc[:,'RoofStyle_Gable'] = dataset["RoofStyle"] == "Gable"
    base.loc[:,'RoofStyle_Hip'] = dataset["RoofStyle"] == "Hip"

    base.loc[:,'Exterior_VinyISd'] = (dataset["Exterior1st"] == "VinyISd") | (dataset["Exterior2nd"] == "VinyISd")
    base.loc[:,'Exterior_MetalSd'] = (dataset["Exterior1st"] == "MetalSd") | (dataset["Exterior2nd"] == "MetalSd")
    base.loc[:,'Exterior_WdSdng'] = (dataset["Exterior1st"] == "Wd Sdng") | (dataset["Exterior2nd"] == "Wd Sdng")
    base.loc[:,'Exterior_HdBoard'] = (dataset["Exterior1st"] == "HdBoard") | (dataset["Exterior2nd"] == "HdBoard")

    base.loc[:,'ExterQual'] = dataset["ExterQual"].map({
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
    })

    base.loc[:,'ExterCond'] = dataset["ExterCond"].map({
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
    })

    base.loc[:,'Foundation_CBlock'] = dataset["Foundation"] == 'CBlock'
    base.loc[:,'Foundation_PConc'] = dataset["Foundation"] == 'PConc'

    base.loc[:,'BsmtQual'] = dataset["BsmtQual"].map({
        'Ex': 105,
        'Gd': 95,
        'TA': 85,
        'Fa': 76,
        'Po': 65,
        'NA': 0
    })

    base.loc[:,'BsmtCond'] = dataset["BsmtCond"].map({
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
        'NA': 0
    })

    base.loc[:,'BsmtExposure'] = dataset["BsmtExposure"].map({
        'Gd': 3,
        'Av': 2,
        'Mn': 1,
        'No': 0,
        'NA': None,
    })

    for col in ["BsmtExposure"]:
        base.loc[:, col] = base[col].fillna(base[col].median())

    base.loc[:,'BsmtFinType1'] = dataset["BsmtFinType1"].map({
        'GLQ': 6,
        'ALQ': 5,
        'BLQ': 4,
        'Rec': 3,
        'LwQ': 2,
        'Unf': 1,
    })

    base.loc[:,'BsmtFinType2'] = dataset["BsmtFinType2"].map({
        'GLQ': 6,
        'ALQ': 5,
        'BLQ': 4,
        'Rec': 3,
        'LwQ': 2,
        'Unf': 1,
        'NA': None
    })

    base.loc[:, 'BsmtFinType2'] = base['BsmtFinType2'].fillna(base.loc[:,'BsmtFinType1'])

    base.loc[:,'HeatingQC'] = dataset["HeatingQC"].map({
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
    })

    base.loc[:,'BsmtFullBath'] = dataset["BsmtFullBath"] > 0 # be

    base.loc[:,'KitchenQual'] = dataset["KitchenQual"].map({
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
    })

    base.loc[:, 'GarageType_Attchd'] = dataset["GarageType"] == "Attchd"
    base.loc[:, 'GarageType_Detchd'] = dataset["GarageType"] == "Detchd"

    base.loc[:,'GarageFinish'] = dataset["GarageFinish"].map({
        'Fin': 3,
        'RFn': 2,
        'Unf': 1,
        'NA': 0,
    })

    base.loc[:,'GarageQual'] = dataset["GarageQual"].map({
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
        'NA': 0
    })

    base.loc[:,'GarageCond'] = dataset["GarageCond"].map({
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
        'NA': 0
    })

    base.loc[:,'Fence'] = dataset["Fence"].map({
        'GdPrv': 3,
        'MnPrv': 2,
        'GdWo': 1,
        'MnWw': 0,
        'NA': None,
    })



    base.loc[:,'YrSold'] = dataset['YrSold'] * 12
    base.loc[:,'MoSold'] = dataset['MoSold']

    for col in base.columns:
        base.loc[:, col] = base[col].fillna(base[col].median())

    base.loc[:,'SaleCondition'] = dataset['SaleCondition'] == 'Normal'

    base.loc[:,'YearBuilt_Diff'] = base['YrSold'] - base['YearBuilt']
    base.loc[:,'GarageYrBlt_Diff'] = base['YrSold'] - base['GarageYrBlt']
    base.loc[:,'YearRemodAdd_Diff'] = base['YrSold'] - base['YearRemodAdd']

    for col in base.columns:
        base.loc[:, col] = base[col].fillna(base[col].median())

    global scaler
    if (scaler == None):
        scaler = MinMaxScaler()
        base = pd.DataFrame(scaler.fit_transform(base), columns=base.columns)
    else:
        base = pd.DataFrame(scaler.transform(base), columns=base.columns)

    base = pd.get_dummies(base)

    poly = PolynomialFeatures(degree=2)
    base = poly.fit_transform(base)

    return base

def getTrain():
    train_data = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\home-prices\\train.csv')
    
    y = train_data["SalePrice"].values.reshape(-1, 1)
    global scalerY
    if (scalerY == None):
        scalerY = MinMaxScaler()
        y = scalerY.fit_transform(y)
    else:
        y = scalerY.transform(y)
    y = pd.DataFrame(y, columns=["SalePrice"])

    x = loadData(train_data)


    return x, y

def getHistagram():
    data = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\home-prices\\train.csv')

    # Plot a histogram for a specific column (e.g., 'LotArea')
    column_name = "SaleCondition"  # Replace with your desired column

    value_counts = data[column_name].value_counts()
    filtered_values = value_counts[value_counts > 150].index
    filtered_data = data[data[column_name].isin(filtered_values)]

    plt.hist(data[column_name].dropna(), bins=20, color='blue', edgecolor='black')


    # Add labels and title
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column_name}')

    # Show the plot
    plt.show()

def getScalerY():
    global scalerY
    return scalerY

if (__name__ == '__main__'):
    #getHistagram()
    print(getTrain())