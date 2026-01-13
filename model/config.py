config = {

    "mp_tmp": {
        "rh_wet_night": 92   # 95 85-94 vaild（决定夜晚是否、何时达到湿润标准）, No.4
    },

    "sporulation": {
        "Tmin": 10,   # 温度涉及夜间产孢强度计算, No.8
        "Topt": 15,
        "Tmax": 30,     # 28-30 vaild
        "MP_min": 5   # 夜间累计湿润小时阈值（hourly 3-5）, No.5
    },

    "dispersal": {
        "rain_min": 10.0, # 0.2 mm，决定雨量是否达到飞溅阈值, No.6
        "wind_min": 8.0, # 9.0 m/s，决定风速是否达到飞溅阈值, No.7
    },

    "deposition": {
        "rain_min": 6.0, # mm，决定雨量是否达到沉降阈值, No.1
    },


    "zoospore_alive": {
        "rain": 3.0,
        "vpd": 3.2   
    },


    "infection": {
        "infection_condition": 150 # WDh × TWDh, No.3 100-180
    },


    "incubation": {
        "spl_end": 1.5  # 病斑显症的阈值，即控制潜育时长, No.2
    },


    "thresholds": {
        "min_spores": 0.05
    }
}
