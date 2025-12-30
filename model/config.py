config = {

    "mp_tmp": {
        "rh_wet_night": 95   # 95 夜间湿润度判定阈值（决定夜晚是否、何时达到湿润标准）, No.4
    },

    "sporulation": {
        "Tmin": 8,   # 温度涉及夜间产孢强度计算, No.8
        "Topt": 20,
        "Tmax": 30,
        "MP_min": 3.0   # 夜间累计湿润小时阈值（hourly）, No.5
    },

    "dispersal": {
        "rain_min": 12.0, # 0.2 mm，决定雨量是否达到飞溅阈值, No.6
        "wind_min": 6.0 # 9.0 m/s，决定风速是否达到飞溅阈值, No.7
    },

    "deposition": {
        "rain_min": 6.0, # mm，决定雨量是否达到沉降阈值, No.1
    },


    "zoospore_alive": {
        "rain": 3.0,
        "vpd": 3.0
    },


    "infection": {
        "infection_condition": 180 # WDh × TWDh, No.3
    },


    "incubation": {
        "spl_end": 1.5  # 病斑显症的阈值，即控制潜育时长, No.2
    },


    "thresholds": {
        "min_spores": 0.05
    }
}
