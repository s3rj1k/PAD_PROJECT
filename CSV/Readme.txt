Dataset is taken from:
	https://traces.cs.umass.edu/index.php/smart/smart

This dataset is:
	`Apartment dataset`

	The apartment dataset contains data for 114 single-family apartments for the period 2014-2016. 

    Meters dataset contains columns (no header is defined):
        - Datetime
        - Total Usage [kW]

    Weather dataset contains columns:
        - temperature           -> Temperature
        - icon                  (has no value for current project)
        - humidity              -> Humidity
        - visibility            -> Visibility
        - summary               (has no value for current project)
        - apparentTemperature   -> Apparent Temperature
        - pressure              -> Pressure
        - windSpeed             -> Wind Speed
        - cloudCover            -> Cloud Cover
        - time                  -> Date
        - windBearing           -> Wind Speed
        - precipIntensity       -> Precipitation Intensity
        - dewPoint              -> Dew Point
        - precipProbability     -> Precipitation Probability

Limitations and caveats:
    The current project is designed to use only single apartment data or the whole building
    due to high memory usage when trying to aggregate data over Date and Apartment number
    simultaneously, also dataset is limited to year data only.

    Apartment 42 data is taken as dataset source.
