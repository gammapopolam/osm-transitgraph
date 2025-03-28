import geopandas as gpd
import pandas as pd
import shapely
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

pd.options.mode.chained_assignment = None

class DailyAccessibilityEvaluator:
    def __init__(self, zones: gpd.GeoDataFrame, raptor_all: gpd.GeoDataFrame, raptor_restr: gpd.GeoDataFrame, restr_type='wh', n_transfers=3, timemarks=list(range(480, 480+900+60, 60)), epsg=32636, lr_sample='perm_core'):
        self.zones_values = zones
        self.n_transfers=n_transfers
        self.epsg=epsg
        self.restr_type=restr_type
        evals={}
        for time in timemarks:
            print(time)

            evals[time]=[]
            raptor_all_time=raptor_all.loc[raptor_all['start']==time]
            evaller_all=AccessibilityEvaluator(raptor_all_time, zones.copy(), 'all')

            raptor_restr_time=raptor_restr.loc[raptor_restr['start']==time]
            evaller_restr=AccessibilityEvaluator(raptor_restr_time, zones.copy(), self.restr_type)
            
            evaller_all.get_spatial_bandwidth(epsg=epsg)
            zones_all=evaller_all.zones.copy()

            evaller_restr.get_spatial_bandwidth(epsg=epsg)
            zones_restr=evaller_restr.zones.copy()

            od_all = evaller_all.get_od_matrix(epsg=epsg)
            od_restr = evaller_restr.get_od_matrix(epsg=epsg)

            ga=GeneralAccessibilityEvaluator(zones_all, zones_restr, restr_type=self.restr_type, time=time)
            ga.measure_A()
            ga.measure_Traveltime(od_all, od_restr, subtract=True, plot=True)
            ga.plot_distribution()
            zones_values_at_time=ga.zones_values.copy()
            evals[time]=zones_values_at_time
        self.evals=evals
        #return self.evals    
    def merge_daily_accessibility(self, evals):
        """
        Объединяет словарь дневных расчетов доступности в единый GeoDataFrame, добавляя временные суффиксы к колонкам.
        
        :param daily_results: словарь {время: GeoDataFrame}
        :return: объединенный GeoDataFrame с показателями по всем временам
        """
        merged_gdf = None
        
        for time_key, gdf in evals.items():
            gdf = gdf.copy()
            gdf = gdf.add_suffix(f"_{time_key}")  # Добавляем суффикс ко всем колонкам
            gdf = gdf.rename(columns={f"zone_id_{time_key}": "zone_id", f"geometry_{time_key}": "geometry"})  # Оставляем идентификаторы без изменений
            
            if merged_gdf is None:
                merged_gdf = gdf
            else:
                merged_gdf = merged_gdf.merge(gdf, on=["zone_id", "geometry"], how="outer")  # Объединение по зоне
        
        return merged_gdf
    def calculate_variability(gdf):
        metrics = {
            'A0_delta': [col for col in gdf.columns if 'A0_delta' in col],
            'A1_delta': [col for col in gdf.columns if 'A1_delta' in col],
            'A2_delta': [col for col in gdf.columns if 'A2_delta' in col],
            'A3_delta': [col for col in gdf.columns if 'A3_delta' in col],
            'A': [col for col in gdf.columns if 'A' in col and 'delta' not in col and 'Aw' not in col],
            'A_norm': [col for col in gdf.columns if 'A_norm' in col],
            'Aw': [col for col in gdf.columns if 'Aw' in col],
            'Aw_norm': [col for col in gdf.columns if 'Aw_norm' in col],
            'mean_delta_time': [col for col in gdf.columns if 'mean_delta_time' in col],
            'mean_delta_transfers': [col for col in gdf.columns if 'mean_delta_transfers' in col]
        }

        new_columns = []
        
        for key, cols in metrics.items():
            if cols:  # Проверяем, есть ли такие колонки
                gdf[f'{key}_mean'] = gdf[cols].mean(axis=1)
                gdf[f'{key}_std'] = gdf[cols].std(axis=1)
                gdf[f'{key}_var'] = gdf[cols].var(axis=1)
                gdf[f'{key}_cv'] = gdf[f'{key}_std'] / gdf[f'{key}_mean']
                
                new_columns.extend([f'{key}_mean', f'{key}_std', f'{key}_var', f'{key}_cv'])
        new_columns.append('geometry')
        return gdf[new_columns]

            




class AccessibilityEvaluator:
    def __init__(self, raptor: gpd.GeoDataFrame, zones: gpd.GeoDataFrame, type='all', epsg=32636):
        self.raptor = raptor
        self.zones = zones
        self.type = type

        self.n_transfers = self.raptor['transfers'].max()

    def get_stop_counts(self, epsg=32636):
        # Удаляем дубликаты остановок по геометрии
        raptor = self.raptor.drop_duplicates(subset=['geometry'])
        
        #raptor.loc[raptor['type']=='bus', 'wc_access'] = 'yes' #!!!
        
        # Переводим зоны и остановки в нужную проекцию
        zones = self.zones.to_crs(epsg=epsg)
        raptor = raptor.to_crs(epsg=epsg)
        
        # Создаем буферизованные геометрии для зон и остановок
        zones['buffered_geometry'] = zones.geometry.buffer(100)
        raptor['buffered_geometry'] = raptor.geometry.buffer(100)
        
        # Создаем пустой список для хранения результатов
        results = []
        
        # Итеративно проходим по каждой зоне
        for idx, zone in tqdm(zones.iterrows(), total=zones.shape[0]):
            # Находим остановки, которые пересекаются с текущей зоной
            intersecting_stops = raptor[raptor.buffered_geometry.intersects(zone.buffered_geometry)]
            
            # Считаем количество остановок по категориям type и wc_access
            count_by_type_access = intersecting_stops.groupby(['type', 'wc_access']).size().unstack(fill_value=0)
            
            # Преобразуем результат в плоский словарь
            flat_counts = {}
            for transport_type, access_counts in count_by_type_access.iterrows():
                for access_type, count in access_counts.items():
                    flat_counts[f"{transport_type}_{access_type}"] = count
            
            # Считаем общее количество остановок
            total_count = len(intersecting_stops)
            
            # Добавляем результаты в список
            results.append({
                'zone_id': idx,  # Идентификатор зоны
                **flat_counts,  # Количество остановок по категориям
                'total_stops': total_count  # Общее количество остановок
            })
        
        # Преобразуем список результатов в DataFrame
        results_df = pd.DataFrame(results)
        
        # Объединяем результаты с исходным геодатафреймом зон
        result = zones.merge(results_df, left_index=True, right_on='zone_id', how='left')
        
        # Заполняем пропущенные значения нулями (если в каком-то полигоне нет остановок)
        result = result.fillna(0)
        
        # Удаляем вспомогательную колонку 'zone_id', если она больше не нужна
        #result = result.drop(columns=['zone_id'])
        result = result.drop(columns=['buffered_geometry'])
        return result


    def get_od_matrix(self, epsg=32636, fillna=False):
        raptor = self.raptor
        raptor.to_crs(epsg=epsg, inplace=True)
        zones = self.zones.to_crs(epsg=epsg)
        zones['buffered_geometry'] = zones.geometry.buffer(100)
        zones = gpd.GeoDataFrame(zones, geometry='buffered_geometry', crs=zones.crs)
        od_list=[]
        od_sample={'from': None, 'to': None, 'arrival': None, 'transfers': None}

        for _, from_z in zones.copy().iterrows():
            from_zone_id=from_z['zone_id']
            raptor_z = raptor.loc[raptor['zone_id'] == from_zone_id]
            for _, to_z in zones.copy().iterrows():
                to_zone_id=to_z['zone_id']
                raptor_z2 = raptor_z.loc[raptor_z.within(to_z.geometry)]
                #print(raptor_z2)
                if len(raptor_z2)>0:
                    #min_transfer=raptor_z2['transfers'].min()
                    #candidates=raptor_z2[raptor_z2['transfers']==min_transfer]
                    best=raptor_z2.loc[raptor_z2['arrival'].idxmin()]

                    # Самая ранняя метка среди всех вместо самой ранней метки при минимуме пересадок
                    od=od_sample.copy()
                    od['from']=from_zone_id
                    od['to']=to_zone_id
                    od['arrival']=best['arrival']
                    od['transfers']=best['transfers']
                else:
                    od=od_sample.copy()
                    od['from']=from_zone_id
                    od['to']=to_zone_id
                od_list.append(od)
        
        od_matrix=pd.DataFrame(od_list)
        if fillna:
            od_matrix=od_matrix.fillna(0)
        return od_matrix
    
    def get_aggregated_od_diff(self, od_1, od_2, ):
        # Создание od_diff
        od_diff = od_1.copy()
        od_diff['arrival'] = od_1['arrival'] - od_2['arrival']
        od_diff['transfers'] = od_1['transfers'] - od_2['transfers']

        # Подсчет доли arrival = 0 и transfers = 0
        zero_counts = od_diff.groupby('from').agg(
            arrival_zero=('arrival', lambda x: (x == 0).sum()),
            transfers_zero=('transfers', lambda x: (x == 0).sum()),
            total_rows=('from', 'size')
        ).reset_index()

        # Вычисление долей
        zero_counts['arrival_zero_ratio'] = zero_counts['arrival_zero'] / zero_counts['total_rows']
        zero_counts['transfers_zero_ratio'] = zero_counts['transfers_zero'] / zero_counts['total_rows']

        # Удаление строк, где arrival = 0 и transfers = 0
        od_diff_filtered = od_diff.loc[(od_diff['arrival'] != 0) | (od_diff['transfers'] != 0)]

        # Подсчет медиан для отфильтрованных данных
        median_values = od_diff_filtered.groupby('from')[['arrival', 'transfers']].mean().reset_index()

        # Объединение результатов
        grouped_diff = pd.merge(zero_counts, median_values, on='from', how='outer')
        return grouped_diff
    
    def get_spatial_bandwidth(self, epsg=32636):
        # get A0, A1, A2, A3, A4 automatically
        for k in range(0, self.n_transfers+1):
            column_name = f'A{k}'
            #print(f'Evaluating {column_name}')
            self.zones[column_name] = 0.0

            raptor_k = self.raptor[self.raptor['transfers'] <= k]

            zones = self.zones.to_crs(epsg=epsg)
            zones['buffered_geometry'] = zones.geometry.buffer(100)
            zones = gpd.GeoDataFrame(zones, geometry='buffered_geometry', crs=zones.crs)

            raptor_k.to_crs(epsg=epsg, inplace=True)
            for _, v in zones.iterrows():
                zone_id = v['zone_id']
                raptor_z = raptor_k.loc[raptor_k['zone_id'] == zone_id]

                # Находим зоны, которые покрываются остановками
                joined_gdf = gpd.sjoin(raptor_z, zones, predicate='within')
                zones_with_stops_ids = joined_gdf['zone_id_right'].unique()
                #print(zone_id, zones_with_stops_ids, end= ' ')
                if len(zones_with_stops_ids) > 0:
                    zones_accessible = zones.loc[zones.zone_id.isin(zones_with_stops_ids)]
                    
                    value = sum(zones_accessible.geometry.area) / sum(zones.geometry.area)

                    # num of zones that are accessible from zone_id
                    #value = len(zones_accessible)/len(zones)
                    self.zones.at[_, column_name] = value
                else: pass
        #print('Evaluating spatial bandwidth complete')


class GeneralAccessibilityEvaluator:
    def __init__(self, zones_all: gpd.GeoDataFrame, zones_restr: gpd.GeoDataFrame, restr_type='lim', time=480):
        self.zones_values=zones_all.copy()
        self.zones_all = zones_all
        self.zones_restr = zones_restr

        self.time = time
        self.restr_type=restr_type
    
    def measure_A(self):
        self.zones_values.rename(columns={'A0': 'A0_all', 'A1': 'A1_all', 'A2': 'A2_all', 'A3': 'A3_all'}, inplace=True) # процент достижимых зон для обычных пассажиров

        self.zones_values[f'A0_{self.restr_type}']=self.zones_restr['A0'] 
        self.zones_values[f'A1_{self.restr_type}']=self.zones_restr['A1']
        self.zones_values[f'A2_{self.restr_type}']=self.zones_restr['A2']
        self.zones_values[f'A3_{self.restr_type}']=self.zones_restr['A3']


        # разница в процентах достижимых зон между обычным человеком и инвалидом по максимальному количеству пересадок 
        self.zones_values['A0_delta']=self.zones_values['A0_all']-self.zones_values[f'A0_{self.restr_type}']
        self.zones_values['A1_delta']=self.zones_values['A1_all']-self.zones_values[f'A1_{self.restr_type}']
        self.zones_values['A2_delta']=self.zones_values['A2_all']-self.zones_values[f'A2_{self.restr_type}']
        self.zones_values['A3_delta']=self.zones_values['A3_all']-self.zones_values[f'A3_{self.restr_type}']


        self.zones_values['Aw']=0.4*self.zones_values['A0_delta']+0.3*self.zones_values['A1_delta']+0.2*self.zones_values['A2_delta']+0.1*self.zones_values['A3_delta']
        def normalize(series):
            return (series - series.min()) / (series.max() - series.min())
        self.zones_values['Aw_norm']=normalize(self.zones_values['Aw'])

        self.zones_values['A']=0.25*self.zones_values['A0_delta']+0.25*self.zones_values['A1_delta']+0.25*self.zones_values['A2_delta']+0.25*self.zones_values['A3_delta']
        self.zones_values['A_norm']=normalize(self.zones_values['A'])

    def plot_accessibility(self):
        # Создаём фигуру и оси
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Ось X (количество раундов)
        x = [0, 1, 2, 3]
        
        # Для каждого геодатафрейма
        for zones, color, label in zip([self.zones_all, self.zones_restr], ['blue', 'green'], ['Обычный режим', 'Режим безбарьерного доступа']):
            # Список для хранения всех значений y (A0-A4)
            all_y_values = []
            
            # Проходим по каждой строке в геодатафрейме
            for idx, row in zones.iterrows():
                # Получаем значения A0-A4
                y = [row['A0'], row['A1'], row['A2'], row['A3']]
                all_y_values.append(y)
                
                # Строим ломаную для текущей строки
                ax.plot(x, y, color=color, alpha=0.05, label=label if idx == 0 else "", zorder=0)
            
            # Вычисляем осреднённые значения A0-A4
            mean_y = np.mean(all_y_values, axis=0)
            
            # Строим осреднённую ломаную
            ax.plot(x, mean_y, color='red' if color == 'blue' else 'purple', linewidth=2, linestyle='--', 
                    label=f'{label}, среднее', zorder=1)
        
        # Настройка графика
        ax.set_xlabel('Раунды')
        ax.set_ylabel('Процент достижимых зон')
        ax.set_title('Процент достижимых зон на каждом раунде для обычного режима и режима безбарьерного доступа')
        ax.legend()
        #ax.grid(True)
        
        # Показать график
        plt.show()

    def plot_distribution(self):
        plt.figure(figsize=(12, 12))
        
        for i, col in enumerate(['A0', 'A1', 'A2', 'A3']):
            plt.subplot(2, 2, i + 1)
            sns.histplot(self.zones_all[col], color='blue', label='Обычный режим', alpha=0.25, kde=True)
            sns.histplot(self.zones_restr[col], color='red', label='Режим безбарьерного доступа', alpha=0.25, kde=True)
            plt.title(f'Процент достижимых зон {col}')
            if i==2: 
                plt.xlim(0.6, 1)
            elif i==3:
                plt.xlim(0.9, 1)
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    def plot_cdf(self):
        plt.figure(figsize=(10, 6))
        
        for col in ['A0', 'A1', 'A2', 'A3']:
            sorted_all = np.sort(self.zones_all[col])
            sorted_wh = np.sort(self.zones_restr[col])
            
            plt.plot(sorted_all, np.arange(1, len(sorted_all) + 1) / len(sorted_all), label=f'{col} (All)')
            plt.plot(sorted_wh, np.arange(1, len(sorted_wh) + 1) / len(sorted_wh), label=f'{col} (Wheelchair)', linestyle='--')
        
        plt.xlabel('Доступность (A0-A4)')
        plt.ylabel('CDF')
        plt.title('Кумулятивное распределение доступности')
        plt.legend()
        plt.grid(True)
        plt.show()            


    def measure_Traveltime(self, od_all: pd.DataFrame, od_restr: pd.DataFrame, time=480, subtract=False, plot=True):
        TIME = self.time
        
        def subtract_departure_time(row):
            if row['arrival'] != 0:
                return row['arrival'] - TIME
            else:
                return 0
        
        if subtract:
            # Применение функции к колонке arrival
            od_all['arrival'] = od_all.apply(subtract_departure_time, axis=1)
            od_restr['arrival'] = od_restr.apply(subtract_departure_time, axis=1)

        # 1. Агрегация достижимых и недостижимых пар для каждой матрицы
        # Для od_all
        reach_all = od_all.groupby('from').apply(lambda x: x['arrival'].notna().sum()).reset_index(name='reach_all')
        unreach_all = od_all.groupby('from').apply(lambda x: x['arrival'].isna().sum()).reset_index(name='unreach_all')

        # Для od_restr
        reach_restr = od_restr.groupby('from').apply(lambda x: x['arrival'].notna().sum()).reset_index(name=f'reach_{self.restr_type}')
        unreach_restr = od_restr.groupby('from').apply(lambda x: x['arrival'].isna().sum()).reset_index(name=f'unreach_{self.restr_type}')

        # 2. Фильтрация пар, достижимых в обоих режимах
        # Объединяем матрицы по парам (from, to)
        df = pd.merge(od_all, od_restr, on=['from', 'to'], suffixes=('_regular', '_prm'))

        # Фильтруем только те пары, где arrival и transfers не NaN в обеих матрицах
        df_filtered = df.dropna(subset=['arrival_regular', 'transfers_regular', 'arrival_prm', 'transfers_prm'])

        # 3. Расчет разницы для достижимых пар
        df_filtered['delta_time'] = (df_filtered['arrival_prm'] - df_filtered['arrival_regular']).abs()
        df_filtered['delta_transfers'] = (df_filtered['transfers_prm'] - df_filtered['transfers_regular']).abs()

        # Группируем по зоне from и вычисляем средние значения
        agg_od_diff = df_filtered.groupby('from').agg({
            'delta_time': 'mean',
            'delta_transfers': 'mean'
        }).reset_index()

        # 4. Сбор всех данных в self.zones_values
        # Объединяем результаты с self.zones_values, указывая явно колонки для merge
        # Удаляем колонку 'from' из промежуточных DataFrame, чтобы избежать дублирования
        reach_all = reach_all.rename(columns={'from': 'zone_id'})
        unreach_all = unreach_all.rename(columns={'from': 'zone_id'})
        reach_restr = reach_restr.rename(columns={'from': 'zone_id'})
        unreach_restr = unreach_restr.rename(columns={'from': 'zone_id'})
        agg_od_diff = agg_od_diff.rename(columns={'from': 'zone_id'})

        # Объединяем все данные в self.zones_values
        self.zones_values = self.zones_values.merge(reach_all, on='zone_id', how='left')
        self.zones_values = self.zones_values.merge(unreach_all, on='zone_id', how='left')
        self.zones_values = self.zones_values.merge(reach_restr, on='zone_id', how='left')
        self.zones_values = self.zones_values.merge(unreach_restr, on='zone_id', how='left')
        self.zones_values = self.zones_values.merge(agg_od_diff, on='zone_id', how='left')

        # Заполняем пропущенные значения нулями
        self.zones_values['reach_all'] = self.zones_values['reach_all'].fillna(0)
        self.zones_values['unreach_all'] = self.zones_values['unreach_all'].fillna(0)
        self.zones_values[f'reach_{self.restr_type}'] = self.zones_values[f'reach_{self.restr_type}'].fillna(0)
        self.zones_values[f'unreach_{self.restr_type}'] = self.zones_values[f'unreach_{self.restr_type}'].fillna(0)
        self.zones_values['delta_time'] = self.zones_values['delta_time'].fillna(0)
        self.zones_values['delta_transfers'] = self.zones_values['delta_transfers'].fillna(0)

        # Переименовываем колонки для удобства
        self.zones_values.rename(columns={
            'delta_time': 'mean_delta_time',
            'delta_transfers': 'mean_delta_transfers'
        }, inplace=True)

        if plot:
            # Гистограмма распределения Δ_time
            plt.figure(figsize=(10, 6))
            plt.hist(df_filtered['delta_time'].dropna(), bins=50, color='blue', edgecolor='black')
            plt.xlabel('Разница во времени (мин)')
            plt.ylabel('Количество пар зон')
            plt.title('Распределение разницы во времени между обычным режимом и режимом безбарьерного доступа')
            plt.grid(True)
            plt.show()
            
            # Гистограмма распределения Δ_transfers
            plt.figure(figsize=(10, 6))
            plt.hist(df_filtered['delta_transfers'].dropna(), bins=50, color='green', edgecolor='black')
            plt.xlabel('Разница в количестве пересадок')
            plt.ylabel('Количество пар зон')
            plt.title('Распределение разницы в количестве пересадок между обычным режимом и режимом безбарьерного доступа')
            plt.grid(True)
            plt.show()
    def measure_Traveltime2(self, od_all: pd.DataFrame, od_restr: pd.DataFrame, time=480, subtract=False, plot=True):
        TIME = time
        
        def subtract_departure_time(row):
            if row['arrival'] != 0:
                return row['arrival'] - TIME
            else:
                return 0
        
        if subtract:
            # Применение функции к колонке arrival
            od_all['arrival'] = od_all.apply(subtract_departure_time, axis=1)
            od_restr['arrival'] = od_restr.apply(subtract_departure_time, axis=1)

        # Объединяем матрицы по парам (from, to)
        df = pd.merge(od_all, od_restr, on=['from', 'to'], suffixes=('_regular', '_prm'))
        
        # Вычисляем разницу во времени и количестве пересадок
        df['delta_time'] = (df['arrival_prm'] - df['arrival_regular']).abs()
        df['delta_transfers'] = (df['transfers_prm'] - df['transfers_regular']).abs()
        
        # Группируем по зоне from и вычисляем средние значения
        agg_od_diff = df.groupby('from').agg({
            'delta_time': 'mean',
            'delta_transfers': 'mean'
        }).reset_index()
        
        # Присваиваем значения напрямую через индексы
        self.zones_values['mean_delta_time'] = self.zones_values['from'].map(agg_od_diff.set_index('from')['delta_time'])
        self.zones_values['mean_transfers_delta'] = self.zones_values['from'].map(agg_od_diff.set_index('from')['delta_transfers'])

        # Подсчитываем количество пар зон с пустыми значениями arrival и transfers
        unreach_all = od_all[od_all['arrival'].isna() | od_all['transfers'].isna()].groupby('from').size()
        unreach_restr = od_restr[od_restr['arrival'].isna() | od_restr['transfers'].isna()].groupby('from').size()

        # Присваиваем значения напрямую через индексы
        self.zones_values['unreach_all'] = self.zones_values['from'].map(unreach_all).fillna(0)
        self.zones_values[f'unreach_{self.restr_type}'] = self.zones_values['from'].map(unreach_restr).fillna(0)

        if plot:
            # Гистограмма распределения Δ_time
            plt.figure(figsize=(10, 6))
            plt.hist(df['delta_time'].dropna(), bins=50, color='blue', edgecolor='black')
            plt.xlabel('Разница во времени (мин)')
            plt.ylabel('Количество пар зон')
            plt.title('Распределение разницы во времени между обычным режимом и режимом для инвалидов')
            plt.grid(True)
            plt.show()
            
            # Гистограмма распределения Δ_transfers
            plt.figure(figsize=(10, 6))
            plt.hist(df['delta_transfers'].dropna(), bins=50, color='green', edgecolor='black')
            plt.xlabel('Разница в количестве пересадок')
            plt.ylabel('Количество пар зон')
            plt.title('Распределение разницы в количестве пересадок между обычным режимом и режимом для инвалидов')
            plt.grid(True)
            plt.show()
    def clusterize(self, plot=True):
        # Выбираем признаки для кластеризации
        X = self.zones_values[['A0_delta', 'A1_delta', 'A2_delta', 'A3_delta']]
        
        # Кластеризация (например, на 3 кластера)
        kmeans = KMeans(n_clusters=3, random_state=42)
        self.zones_values['cluster'] = kmeans.fit_predict(X)
        
        # Визуализация на карте
        self.zones_values.plot(column='cluster', legend=True, figsize=(10, 6))
        plt.title('Кластеризация районов по доступности')
        plt.show()
    
    def save_measures(self, gpkg='perm_tests.gpkg', lr='zones_values_480'):
        self.zones_values.to_file(gpkg, layer=lr)
