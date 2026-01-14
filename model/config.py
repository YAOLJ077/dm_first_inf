config = {

    "mp_tmp": {
        "rh_wet_night": 88.5398411847977,   # 95 85-94 vaild（决定夜晚是否、何时达到湿润标准）, No.4
    },

    "sporulation": {
        "Tmin": 13.00506496569502,   # 温度涉及夜间产孢强度计算, No.8
        "Topt": 22.06511887760097,
        "Tmax": 33.5014956875138,     # 28-30 vaild
        "MP_min": 3.8743261642690388   # 夜间累计湿润小时阈值（hourly 3-5）, No.5
    },

    "dispersal": {
        "rain_min": 12.983549259218158, # 0.2 mm，决定雨量是否达到飞溅阈值, No.6
        "wind_min": 5.895602405612547, # 9.0 m/s，决定风速是否达到飞溅阈值, No.7
    },

    "deposition": {
        "rain_min": 4.597537114008573, # mm，决定雨量是否达到沉降阈值, No.1
    },


    "zoospore_alive": {
        "rain": 4.627854837928216,
        "vpd": 1.8375826407139801  
    },


    "infection": {
        "infection_condition": 48.41277509071878 # WDh × TWDh, No.3 100-180
    },


    "incubation": {
        "spl_end": 1.4291318354781684 # 病斑显症的阈值，即控制潜育时长, No.2
    },


    "thresholds": {
        "min_spores": 0.05
    }
}
