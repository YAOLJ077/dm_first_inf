config = {

    "mp_tmp": {
        "rh_wet_night": 90    # 夜间湿润度判定阈值（决定夜晚是否、何时达到湿润标准）
    },

    "sporulation": {
        "Tmin": 10,   # 温度涉及夜间产孢强度计算
        "Topt": 20,
        "Tmax": 30,
        "MP_min": 3   # 夜间累计湿润小时阈值（hourly）
    },


    "dispersal": {
        "rain_min": 5.0, # 0.2 mm，决定雨量是否达到飞溅阈值
        "wind_min": 3.0 # 9.0 m/s，决定风速是否达到飞溅阈值
    },

    "deposition": {
        "rain_min": 2.0, # mm，决定雨量是否达到沉降阈值
    },


    "zoospore_alive": {
        "rain": 2,
        "vpd": 2.0
    },


    "infection": {
        "infection_condition": 60   # WDh × TWDh
    },


    "incubation": {
        "spl_end": 1.0  # 病斑显症的阈值，即控制潜育时长
    },


    "thresholds": {
        "min_spores": 1e-6
    }
}
