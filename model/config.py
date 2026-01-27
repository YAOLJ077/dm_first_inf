config = {

    "mp_tmp": {
        "rh_wet_night": 88.8370862115935,   # 95 85-94 vaild（决定夜晚是否、何时达到湿润标准）, No.4
    },

    "sporulation": {
        "Tmin": 11.564215573655861,   # 温度涉及夜间产孢强度计算, No.8
        "Topt": 24.325516078665604,
        "Tmax": 28.472399824935035,     # 28-30 vaild
        "MP_min": 3.010165614790454   # 夜间累计湿润小时阈值（hourly 3-5）, No.5
    },

    "dispersal": {
        "rain_min": 10.785794005538701, # 0.2 mm，决定雨量是否达到飞溅阈值, No.6
        "wind_min": 6.417813942746813, # 9.0 m/s，决定风速是否达到飞溅阈值, No.7
    },

    "deposition": {
        "rain_min": 6.451824922258265, # mm，决定雨量是否达到沉降阈值, No.1
    },


    "zoospore_alive": {
        "rain": 5.0362900352909,
        "vpd": 1.9203558994346168  
    },


    "infection": {
        "infection_condition": 50.94838780034761 # WDh × TWDh, No.3 100-180
    },

    "hazard": {
        "alpha": 1.0,
        "p": 1.0,
        "q": 1.0,
        "threshold": 1.0
    },


    "incubation": {
        "spl_end": 0.9954019930058942 # 病斑显症的阈值，即控制潜育时长, No.2
    },


    "thresholds": {
        "min_spores": 0.036400283212930526
    }
}
