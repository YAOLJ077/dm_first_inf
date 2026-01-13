import pandas as pd
import os,sys
src_dir=os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,src_dir)

import math
import datetime
import numpy as np
import json
import config
pd.set_option('display.max_rows', 1000)

class  PrimaryInfection:
    def __init__(self, cs=None, config=None):
        self.cs=cs
        self.config=config
        self.weather_hourly=cs.weather_hourly[cs.weather_hourly.datetime>=datetime.datetime(year=self.cs.year,month=1,day=1)]
        self.MMO=self.config['MMO']
        self.PIS_Start = self.config['primary_inoculum_season']["start_period"]
        self.PIS_End = self.config['primary_inoculum_season']["end_period"]
        
    def physiological_mature_oospores(self):
        self.weather_hourly['VPD'] = None
        self.weather_hourly['VPD'] = self.weather_hourly[['air_temp','relative_humidity']].apply(
            lambda x: (1-x['relative_humidity']/100) * 6.11 * math.exp((17.47 * x['air_temp'])/(239 + x['air_temp'])), axis =1
        )

        self.weather_hourly['moisture_of_leaf'] = self.weather_hourly[['precip_acc_period_adjusted','VPD']].apply(
            lambda x : 1 if x['precip_acc_period_adjusted'] > self.config['Moisture_of_leaf']['Rain'] or x['VPD'] <= self.config['Moisture_of_leaf']['VPD'] else 0 , axis=1
        )

        self.weather_hourly['denominator'] = 1330.1 - 116.19 * self.weather_hourly['air_temp'] + 2.6256 * (self.weather_hourly['air_temp'] ** 2)
        self.weather_hourly['HTh_hourly'] = self.weather_hourly.apply(lambda x: 0 if x['air_temp'] <= 0 else x['moisture_of_leaf'] / x['denominator'], axis=1)
        self.weather_hourly['HTh'] = 0
        self.weather_hourly['HTh'] = self.weather_hourly['HTh_hourly'].cumsum()
        self.weather_hourly['DOR'] = np.exp(-15.891 * np.exp(-0.653 * (self.weather_hourly['HTh'] + 1)))

        self.weather_hourly['MMO'] = 0
        self.weather_hourly.loc[1:, 'MMO'] = 1
        self.weather_hourly['PMO'] = self.weather_hourly.apply(
            lambda x: x['DOR'] * x['MMO'], axis=1
        )

        self.weather_hourly['primary_inoculum_season'] = None
        self.weather_hourly['primary_inoculum_season'] = self.weather_hourly.HTh.apply(lambda x: 1 if x >= self.PIS_Start and x <= self.PIS_End else 0)
        self.cs.primary_inoculum_season_start = self.weather_hourly[self.weather_hourly['primary_inoculum_season'] == 1].head(1)['datetime'].values[0]
        self.cs.primary_inoculum_season_end = self.weather_hourly[self.weather_hourly['primary_inoculum_season'] == 1].tail(1)['datetime'].values[0]
        self.cs.physiological_mature_oospores_status=self.weather_hourly

    def germinated_oospores(self):
        primary_inoculum_season_sporangia=self.cs.physiological_mature_oospores_status[self.cs.physiological_mature_oospores_status['primary_inoculum_season'] == 1]
        primary_inoculum_season_sporangia.reset_index(inplace=True,drop=True)
        sporangia_cohorts=pd.DataFrame()
        cohort_num=0
        PMOc=0 #initial density of the first oospore cohort
        for ind,row in primary_inoculum_season_sporangia[primary_inoculum_season_sporangia.precip_acc_period_adjusted>=self.config['rain_for_oospore_gemination']].iterrows():
            sporangia_cohort=primary_inoculum_season_sporangia[primary_inoculum_season_sporangia['datetime']>=row.datetime].copy()
            sporangia_cohort['Num']=cohort_num;cohort_num+=1
            sporangia_cohort['cohort_start_datetime']=row.datetime
            sporangia_cohort['PMOc']=PMOc
            sporangia_cohort['GERh']=sporangia_cohort.HTh_hourly.cumsum()
            sporangia_cohort=sporangia_cohort[sporangia_cohort['GERh']<=self.config['GEO_GER_period_end_point']]
            sporangia_cohort['cohort_end_datetime']=sporangia_cohort.datetime.max()
            if len(primary_inoculum_season_sporangia.loc[(primary_inoculum_season_sporangia.datetime>row.datetime)&(primary_inoculum_season_sporangia.precip_acc_period_adjusted>=self.config['rain_for_oospore_gemination']),'datetime'])>0:
                next_rain_date=primary_inoculum_season_sporangia.loc[(primary_inoculum_season_sporangia.datetime>row.datetime)&(primary_inoculum_season_sporangia.precip_acc_period_adjusted>=self.config['rain_for_oospore_gemination']),'datetime'].values[0]
                PMOc=sporangia_cohort[(sporangia_cohort['datetime']<=next_rain_date)&(sporangia_cohort['datetime']>=row.datetime)]['DOR'].sum()
                sporangia_cohorts=pd.concat([sporangia_cohorts,sporangia_cohort],axis=0)
            else:            
                sporangia_cohorts=pd.concat([sporangia_cohorts,sporangia_cohort],axis=0)
                break
        self.cs.sporangia_cohorts=sporangia_cohorts # output: Num,  cohort_start_datetime, cohort_end_datetime, PMOc(GEO), dscrible a germinated processes of oospores
        # print(sporangia_cohorts)
        # sporangia_cohorts.to_csv('sporangia_cohorts.csv')
        
    def release_sporangia(self):
        sporangia_cohorts=self.cs.sporangia_cohorts.copy() #new add
        sporangia_cohorts['release_datetime']=None
        sporangia_cohorts['cohort_sus_datetime']=None
        season_end = self.cs.primary_inoculum_season_end # new add

        for cohort in sporangia_cohorts.Num.unique():
            sp_chort=sporangia_cohorts[sporangia_cohorts.Num==cohort]
            sp_chort_start=sp_chort.cohort_end_datetime.values[0]
            sp_chort_weather=self.cs.weather_hourly[self.cs.weather_hourly.datetime>=sp_chort_start].copy()
            sp_chort_weather=sp_chort_weather[sp_chort_weather.datetime<=self.cs.primary_inoculum_season_end]

            sp_chort_weather['SUSh_lower']=sp_chort_weather.apply(lambda row:(24 * (5.67 - 0.47 * (row.air_temp * (1 - row.relative_humidity / 100)) +
                                    0.01 * (row.air_temp * (1 - row.relative_humidity / 100)) ** 2)),axis=1)
            sp_chort_weather['SUSh_hourly']=sp_chort_weather['SUSh_lower'].apply(lambda x:1/x)
            sp_chort_weather['SUSh']=sp_chort_weather['SUSh_hourly'].cumsum()
            sp_chort_weather=sp_chort_weather[sp_chort_weather['SUSh']<=self.config['GEO_SUS_period_end_point']]
            sp_chort_weather['cohort_sus_datetime']=sp_chort_weather.datetime.max()

            # splash stage

            sp_chort_weather=sp_chort_weather.merge(self.cs.physiological_mature_oospores_status[['datetime','moisture_of_leaf']],on='datetime',how='left')
            sp_chort_weather['Gc']= (sp_chort_weather['moisture_of_leaf'] != sp_chort_weather['moisture_of_leaf'].shift()).cumsum()
            sp_chort_wet=sp_chort_weather[sp_chort_weather['moisture_of_leaf']==1].copy()

            # new add 记录最大存活时间（若存在满足条件的时间）
            if not sp_chort_weather.empty:
                max_sus_end = sp_chort_weather.datetime.max()
                sporangia_cohorts.loc[sporangia_cohorts.Num == cohort, 'cohort_sus_datetime'] = max_sus_end

            for ind,gp in sp_chort_wet.groupby('Gc'):
                gp['WDh']=range(0,len(gp))
                gp['TWDh']=gp.air_temp.expanding().mean()
                gp['release_condition']=gp.TWDh.apply(lambda x:np.exp((-1.022+19.634)/x))
                if len(gp[gp.WDh>=(gp.release_condition)])>0:
                    release_datetime=gp.loc[gp.WDh>=(gp.release_condition),'datetime'].values[0]
                    sporangia_cohorts.loc[sporangia_cohorts.Num==cohort,'release_datetime']=release_datetime
                    sporangia_cohorts.loc[sporangia_cohorts.Num==cohort,'cohort_sus_datetime']=sp_chort_weather['cohort_sus_datetime'].max() #new add cohort_sus_datetime
                    break
        # sporangia_cohorts['cohort_sus_datetime'] = sporangia_cohorts.groupby('Num')['cohort_sus_datetime'].transform(lambda x: x.max() if not x.isnull().all() else sp_chort_start)#new add cohort_sus_datetime
        # 关键修正行：用 season_end 回填未计算出的队列
        sporangia_cohorts['cohort_sus_datetime'] = sporangia_cohorts.groupby('Num')['cohort_sus_datetime'].transform(lambda x: x.max() if not x.isnull().all() else season_end)
        sporangia_cohorts['ZRE_rho'] = sporangia_cohorts.apply(lambda row: row['PMOc'] if not pd.isnull(row['release_datetime']) else 0, axis=1)
        self.cs.sporangia_cohorts_release = sporangia_cohorts.copy()
        # sporangia_cohorts.to_csv('sporangia_cohorts_release.csv')

    def splash_zoospores(self):
        sporangia_cohorts=self.cs.sporangia_cohorts_release
        sporangia_cohorts['SUZ_end_datetime']=None
        sporangia_cohorts_with_release=sporangia_cohorts.drop_duplicates(subset=['Num','release_datetime','ZRE_rho']).dropna(subset=['release_datetime'])
        sporangia_cohorts_with_release['ZDI_splash_amount']=None
        sporangia_cohorts_with_release['splash_datetime']=None
        for ind,row in sporangia_cohorts_with_release.iterrows():
            release_datetime = pd.Timestamp(row['release_datetime']) # new add
            ceil_release = release_datetime.ceil('h') # new add
            # release_datetime=row['release_datetime']
            release_weather = self.cs.physiological_mature_oospores_status[(self.cs.physiological_mature_oospores_status.datetime > ceil_release) & (self.cs.physiological_mature_oospores_status.datetime <= self.cs.primary_inoculum_season_end)].copy()
            SUZh = self.cs.primary_inoculum_season_end  # 默认值
            # release_weather=self.cs.physiological_mature_oospores_status[(self.cs.physiological_mature_oospores_status.datetime>release_datetime)&(self.cs.physiological_mature_oospores_status.datetime<=self.cs.primary_inoculum_season_end)]
            if len(release_weather[release_weather.moisture_of_leaf == 0]) > 0:
                SUZh_candidate = release_weather.loc[release_weather['moisture_of_leaf'] == 0, 'datetime'].min()
                SUZh = max(SUZh_candidate, ceil_release)
                SUZh = SUZh if SUZh > release_datetime else self.cs.primary_inoculum_season_end
                # SUZh=release_weather.loc[release_weather['moisture_of_leaf']==0,'datetime'].min()
                sporangia_cohorts_with_release.loc[(sporangia_cohorts_with_release.Num == row.Num)&(sporangia_cohorts_with_release.release_datetime == release_datetime), 'SUZ_end_datetime'] = SUZh  
                release_weather_start_end=release_weather[(release_weather.datetime>=release_datetime)&(release_weather.datetime<=SUZh)]
                splash_datetime=release_weather_start_end.loc[release_weather_start_end['precip_acc_period_adjusted']>=self.config['rain_for_oospore_splash'],'datetime'].min()
                sporangia_cohorts_with_release.loc[(sporangia_cohorts_with_release.Num==row.Num)&(sporangia_cohorts_with_release.release_datetime==release_datetime),'splash_datetime']=splash_datetime
                if (not splash_datetime is None) and (not splash_datetime is pd.NaT):
                    ZDI_splash_amount=row.ZRE_rho
                    sporangia_cohorts_with_release.loc[(sporangia_cohorts_with_release.Num==row.Num)&(sporangia_cohorts_with_release.release_datetime==release_datetime),'ZDI_splash_amount']=ZDI_splash_amount
        self.cs.infection_after_splash=sporangia_cohorts_with_release.copy()
        # sporangia_cohorts_with_release.to_csv('sporangia_cohorts_with_release.csv')
    def infection_after_splash(self):
        infection_after_splash=self.cs.infection_after_splash.copy()
        infection_after_splash=infection_after_splash.dropna(subset=['ZDI_splash_amount','splash_datetime'], how='any') # new add
        infection_after_splash['infection_time_diff_hours'] = np.nan # new add
        infection_after_splash['infection_time_diff_days'] = np.nan # new add
        infection_after_splash['infection_after_splash']=False
        infection_after_splash['infection_after_splash_datetime']=None
        for ind,row in infection_after_splash.iterrows():
            if row.splash_datetime is None:
                continue
            infection_after_splash_weather = self.cs.weather_hourly[(self.cs.weather_hourly.datetime >= row.splash_datetime) & (self.cs.weather_hourly.datetime <= row.SUZ_end_datetime)].copy() # new add
            # infection_after_splash_weather=self.cs.weather_hourly[self.cs.weather_hourly.datetime>=row.splash_datetime].copy()
            infection_after_splash_weather=infection_after_splash_weather.merge(self.cs.physiological_mature_oospores_status[['datetime','moisture_of_leaf']],on='datetime',how='left')
            infection_after_splash_weather['Gc']= (infection_after_splash_weather['moisture_of_leaf'] != infection_after_splash_weather['moisture_of_leaf'].shift()).cumsum()
            infection_after_splash_wet=infection_after_splash_weather[infection_after_splash_weather['moisture_of_leaf']==1].copy()

            for ind,gp in infection_after_splash_wet.groupby('Gc'):
                gp['WDh']=range(0,len(gp))
                gp['TWDh']=gp.air_temp.expanding().mean()
                gp['infection_condition'] = gp.apply(lambda row: row['WDh'] * row['TWDh'] >= self.config['Infection_condition'], axis=1)  
                if len(gp[gp.infection_condition])>0:
                    infection_datetime=gp.loc[gp.infection_condition,'datetime'].values[0]
                    td = infection_datetime - row.splash_datetime # new add
                    td_hours = round(td.total_seconds() / 3600, 3) # new add
                    td_days = round(td_hours / 24, 3) # new add

                    mask = (infection_after_splash.Num == row.Num) # new add
                    infection_after_splash.loc[infection_after_splash.Num==row.Num,'infection_after_splash_datetime']=infection_datetime
                    infection_after_splash.loc[infection_after_splash.Num==row.Num,'infection_after_splash']=True
                    infection_after_splash.loc[mask, 'infection_time_diff_hours'] = td_hours  # new add
                    infection_after_splash.loc[mask, 'infection_time_diff_days'] = td_days # new add
                    break
        self.cs.infection_after_splash=infection_after_splash
    def oil_spot_on_leaf(self):
        oil_spot_on_leaf=self.cs.infection_after_splash.copy()
        oil_spot_on_leaf['oil_spot_on_leaf_start']=None
        oil_spot_on_leaf['oil_spot_on_leaf_end']=None
        for ind,row in oil_spot_on_leaf.iterrows():
            if not row.infection_after_splash:
                continue
            oil_spot_on_leaf_weather=self.cs.weather_hourly[self.cs.weather_hourly.datetime>=row.infection_after_splash_datetime].copy()
            oil_spot_on_leaf_weather['INC_quote_h']=oil_spot_on_leaf_weather.air_temp.apply(lambda x:1/(24*(45.1-3.45*x+0.073*x**2)))
            oil_spot_on_leaf_weather['INC_double_quote_h']=oil_spot_on_leaf_weather.air_temp.apply(lambda x:1/(24*(59.9-4.55*x+0.095*x**2)))
            oil_spot_on_leaf_weather['INC_quote_h_cusum']=oil_spot_on_leaf_weather.INC_quote_h.cumsum()
            oil_spot_on_leaf_weather.INC_double_quote_h=oil_spot_on_leaf_weather.INC_double_quote_h.cumsum()
            oil_spot_on_leaf_start=oil_spot_on_leaf_weather.loc[oil_spot_on_leaf_weather['INC_quote_h_cusum']<=self.config['GEO_SPL_period_end_point'],'datetime'].max()
            oil_spot_on_leaf_end=oil_spot_on_leaf_weather.loc[oil_spot_on_leaf_weather['INC_double_quote_h']<=self.config['GEO_SPL_period_end_point'],'datetime'].max()
            oil_spot_on_leaf.loc[oil_spot_on_leaf.Num==row.Num,'oil_spot_on_leaf_start']=oil_spot_on_leaf_start
            oil_spot_on_leaf.loc[oil_spot_on_leaf.Num==row.Num,'oil_spot_on_leaf_end']=oil_spot_on_leaf_end
        self.cs.oil_spot_on_leaf=oil_spot_on_leaf
        # print(oil_spot_on_leaf)
        # oil_spot_on_leaf.to_csv('oil_spot_on_leaf.csv')
        

