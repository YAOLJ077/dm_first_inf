import pandas as pd
from model.first_infection_only import FirstInfection


class CropSeasonFirst:
    """
    FIRST infection crop-season controller
    (architecture aligned with Primary model)
    """

    def __init__(self, csrb: dict = None, config: dict = None):

        self.cs = csrb
        self.config = config

        self.year = csrb["year"]
        self.place = csrb["Place"]

        # ---- weather ----
        self.weather_hourly = pd.DataFrame(csrb["weather_hourly"])
        self.weather_hourly.datetime = pd.to_datetime(self.weather_hourly.datetime)

        # ---- daily MP / TMP ----
        night = (self.weather_hourly.datetime.dt.hour >= 20) | \
                (self.weather_hourly.datetime.dt.hour < 8)
        wet = self.weather_hourly.relative_humidity >= config["mp_tmp"]["rh_wet_night"]
        self.weather_hourly["night_wet"] = night & wet

        self.weather_daily = (
            self.weather_hourly
            .groupby(self.weather_hourly.datetime.dt.normalize(), as_index=False)
            .agg(datetime=("datetime", "first"),
                 MP=("night_wet", "sum"),
                 TMP=("air_temp", "mean"))
        )

        # ---- state containers (like primary) ----
        self.sporangia_cohorts = None
        self.oil_spot_on_leaf = None

    # ======================================================
    def simulate(self):

        fi = FirstInfection(cs=self, config=self.config)
        fi.sporulation_and_dispersal()
        fi.deposition()
        fi.primary_chain()

    # ======================================================
    # getters (same philosophy as primary)
    # ======================================================
    def get_sporangia_cohorts(self):
        return self.sporangia_cohorts

    def get_oil_spot_on_leaf(self):
        return self.oil_spot_on_leaf
