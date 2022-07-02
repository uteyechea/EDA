import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.geocoders import Nominatim


class Data():
    def __init__(self):
        # Define local working, data and libs directories
        self.root_dir = os.path.normpath(Path(__file__).resolve().parents[2])
        self.data_dir = os.path.normpath(os.path.join(self.root_dir, 'data'))
        self.raw = None  # The original, immutable data dump

    def extract(self, folder_name: str, file_name: str) -> pd.DataFrame():
        assert type(folder_name) == str, f'{folder_name} expected dtype==str'
        assert type(file_name) == str, f'{file_name} expected dtype==str'
        data = pd.read_csv(os.path.join(
            self.data_dir, folder_name, file_name), na_values='n/a')
        self.raw = data
        return data.info()

    def remove_duplicated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes features that have the same data as other features. 
        Input:

        Returns:
        New dataframe without duplicated data.
        """
        # Assert features are really duplicated. And run a baseline, showing that is safe to delete the following columns, given that linear regression sets values of close to 0 for them.

        df.drop(['id',  # PK: here it is not useful.
                 'address',  # Subset of the feature "location".
                 # The link by itself is not useful. Nevertheless, I might consider its use as an external source of information.
                 'link',
                 # The problem we are solving is a function of price_mod. If we leave it we can use it to compute price_mod / m2 and get price_square_meter. In this case, it would be useless to use other features.
                 'price_mod',
                 # 'm2', # The problem we are solving is a function of m2. If we leave it we can use it to compute price_mod / m2 and get price_square_meter. In this case, it would be useless to use other features.
                 # 'final_price', # The problem we are solving is a function of final_price. If we leave it we can use it to compute price_mod / m2 and get price_square_meter. In this case, it would be useless to use other features.
                 # The problem we are solving is a function of price. If we leave it we can use it to compute price_mod / m2 and get price_square_meter. In this case, it would be useless to use other features.
                 'price',
                 # This feature's data has been extracted, and used to make new features. Such as: m2, num_bedrooms.
                 'attributes',
                 'main_name',  # The information here is either trivial or has been used to make new features. Such as: subtitle
                 # This feature's data has been extracted, and used to make new features. Such as: since_period, since_value and days_on_site.
                 'since',
                 'age_in_years',  # Doesn't have information.
                 'subtitle',  # Constant (no change)
                 'price_currency'  # Constant (no change)
                 ], axis=1, inplace=True)

        return df

        """
        Looks like "price" equals "final_price". If true remove "price"
        """

        #can_remove_feature = True
        # for price, price_text in list(zip(df['final_price'],df['price'])):
        #    if str(int(price)) not in price_text.split():
        #        print('Can not remove "price" feature, given that not all rows are equivalent to "final_price" feature')
        #        can_remove_feature = False
        # if can_remove_feature:
        #    df.drop('price', axis=1 ,inplace=True)
        #    print(f'Removed "price" given that "price_final" = "price"')

        """
        Remove columns that have the same values on all rows
        """
        # if len(df.columns) < 2:
        #    return df
        # for index, feature1 in enumerate(df):
        #    for feature2 in df.columns[index+1:]:
        #        if df[feature1].equals(df[feature2]):
        #            df.drop(feature2, axis=1, inplace=True)
        #            print(f'Removed "{feature2}" given that "{feature1}" = "{feature2}"')

        """
        "attributes" appear equal to "num_bedrooms" and "m2". Verify and remove "attributes" if equal.
        """
        #not_found_values_count = 0
        # for index, row in enumerate(df['attributes']):
        #    values = ' '.join(row.split(',')).split()
        #    if str(df['num_bedrooms'][index].astype(int)) not in values and str(df['m2'][index].astype(int)) not in values:
        #        print(f"{str(df['num_bedrooms'][index])} or {str(df['m2'][index])} are not in {values}")
        #        not_found_values_count += 1
        # if not_found_values_count == 0:
        #    df.drop(['attributes'], axis=1, inplace=True)

        """
        "Since" this feature's data has been extracted, and used to make new features. Such as: since_period, since_value and days_on_site.
        """
        # def get_total_days_since_publication(df):
        #    for index, row in enumerate(df['since']):
        #        row = row.replace('días', '1')
        #        row = row.replace('día', '1')
        #        row = row.replace('meses', '30')
        #        row = row.replace('mes', '30')
        #        row = row.replace('años', '365')
        #        row = row.replace('año', '365')

        #        text = row.split()
        #        values = []
        #        for string in text:
        #            if string.isdigit():
        #                values.append(int(string))
        #        df['since'][index] = np.prod(values)
        #    #df['since'].astype('int')
        #    return df

        # Find the first dtype different from NoneType in a series
    def find_dtype(self, series):
        for value in series:
            if type(value) != type(None):
                return type(value)

    def fix_nans(self, df):
        for feature in df:
            # if find_dtype(series=trxn[str(feature)]) == pd._libs.tslibs.nattype.NaTType:
            #    trxn[str(feature)].fillna('NA', inplace=True)
            if self.find_dtype(series=df[str(feature)]) == int or self.find_dtype(series=df[str(feature)]) == float:
                df[str(feature)].fillna(0, inplace=True)
            else:
                df[str(feature)].fillna('NA', inplace=True)
        return df

    def fix_monthly_fee(self, df):
        for index, row in enumerate(df['monthly_fee']):
            text = row.split()
            for string in text:
                if string.isdigit():
                    df.loc[index, 'monthly_fee'] = int(string)
                    break
                else:
                    df.loc[index, 'monthly_fee'] = 0
                #print(text, df['monthly_fee'][index])
        df['monthly_fee'] = df['monthly_fee'].astype('int')
        return df

    def get_cat_codes(self, df, categorical_features):
        df[categorical_features] = df[categorical_features].astype('category')

        for feature in categorical_features:
            df[feature] = df[feature].cat.codes

        return df

    def sort_df_dtype(self, df):

        objects = df[df.select_dtypes(include='object').columns]
        categories = df[df.select_dtypes(include='int8').columns]
        integers = df[df.select_dtypes(include='int').columns]
        floats = df[df.select_dtypes(include='float').columns]

        return pd.concat([objects, categories, integers, floats], axis=1)

    def normalize_features(self, df):
        new_df = df.select_dtypes(
            ['int8', 'int16', 'int32', 'int64', 'float64', 'float16', 'float32'])

        mean = new_df.mean()
        std = new_df.std()
        new_df = (new_df - mean) / std

        return new_df

    # def clean_main_name(df):
        # for name in df['main_name']:
        #[word for word in df['main_name'] if word not in df['location'].split()]

    def get_data_distribution(self, df):
        features = df.iloc[:, :]
        x_labels = features.columns
        features = features.melt(var_name='Column', value_name='Normalized')
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(x='Column', y='Normalized', data=features)
        _ = ax.set_xticklabels(x_labels, rotation=90)
        plt.show()

    def read_location_attributes(self):
        location_data = pd.read_csv(os.path.join(self.data_dir,'interim','location_data.csv'))
        try:
            location_data.drop('Unnamed: 0', axis=1, inplace=True)
        except:
            pass
        location_data = location_data.fillna('NA')
        return location_data

    def get_location_attributes(self, data):

        mayoralties = self.get_mayoralties()

        location_data = {'road': [], 'neighbourhood': [], 'postcode': [], 'mayoralty': [],
                         'city': [], 'state': [], 'country': []}
        geolocator = Nominatim(user_agent="uteyechea@gmail.com")
        for i in range(len(data)):
            lat = data['lat'][i]
            lon = data['lon'][i]
            location = geolocator.reverse(f"{str(lat)}, {str(lon)}")

            try:
                location_data['road'].append(location.raw['address']['road'])
            except:
                location_data['road'].append('NA')

            try:
                location_data['neighbourhood'].append(
                    location.raw['address']['neighbourhood'].replace('Colonia ', ''))
            except:
                location_data['neighbourhood'].append('NA')

            try:
                location_data['postcode'].append(
                    location.raw['address']['postcode'])
            except:
                location_data['postcode'].append('NA')

            try:
                mayoralty = [string.strip() for string in location.raw['display_name'].split(
                    ',') if string.strip() in mayoralties][0]
                location_data['mayoralty'].append(mayoralty)
            except:
                location_data['mayoralty'].append('NA')

            try:
                location_data['city'].append(location.raw['address']['city'])
            except:
                location_data['city'].append('NA')

            try:
                location_data['state'].append(location.raw['address']['state'])
            except:
                location_data['state'].append('NA')

            try:
                location_data['country'].append(
                    location.raw['address']['country'])
            except:
                location_data['country'].append('NA')

        location_data = pd.DataFrame(location_data)
        location_data.to_csv(os.path.join(self.data_dir,'interim','location_data.csv'), index=False) 

        return location_data

    def add_location_attributes(self, df: pd.DataFrame(), location_data: pd.DataFrame):
        try:
            df.drop('location', axis=1, inplace=True)
        except:
            pass
        new_df = pd.concat([location_data, df], axis=1)
        return new_df

    def get_idsm(self):
        indice_desarrollo_social = pd.read_csv(os.path.join(
            self.data_dir, 'external', 'ids_alcaldias.csv')).fillna('NA')
        return indice_desarrollo_social

    def get_mayoralties(self):
        indice_desarrollo_social = self.get_idsm()
        mayoralties = indice_desarrollo_social['alcaldia'].unique()
        return mayoralties

    def add_idsm(self, df):
        indice_desarrollo_social = self.get_idsm()
        ids = {'ids': []}
        for i, mayoralty in enumerate(df['mayoralty']):
            try:
                ids['ids'].append(
                    indice_desarrollo_social[indice_desarrollo_social['alcaldia'] == mayoralty].iloc[0, 2])
            except:
                ids['ids'].append('No disponible')
        ids = pd.DataFrame(ids)
        new_df = pd.concat([ids, df], axis=1)
        return new_df
