# Time Series Visualization for Business Analysis with Python Time series visualization helps us see data patterns, trends,
seasonality, and anomalies. By visualizing time series data, we can gain...

### Time Series Visualization for Business Analysis with Python
Time series visualization helps us see data patterns, trends, seasonality, and anomalies. By visualizing time series data, we can gain valuable insights that guide modeling choices, such as whether to include seasonal adjustments, trends, or frequency resampling.


<figcaption>Photo by <a class="markup--anchor markup--figure-anchor" rel="photo-creator noopener" target="_blank">Chinh Le Duc</a> on <a class="markup--anchor markup--figure-anchor"


This article explores techniques for visualizing time series using Python libraries like matplotlib, pandas, and seaborn, with a focus on resampling to change observation frequencies.

#### Why Visualize Time Series Data?
Effective time series visualization helps us: 1/Identify trends by determining whether data increases, decreases, or remains stable over time. 2/Detect seasonality by Recognizing repeating cycles (e.g., yearly or monthly). 3/Identify Anomalies/outliers where there are sudden changes in values. 4/Evaluate Data Frequency to see how data behaves at different time granularities (e.g., hourly, daily, or monthly).

#### Basic Time Series Visualization
A simple line plot is the most common way to visualize time series data. We will be using pandas, matplotlib, and the same dateset for each plot. So I won't repeat those in each code block.

Python Code: Line Plot of a Time Series



Line Plots help us see if the general trend (increasing or decreasing), the presence of Seasonal patterns (regular cycles), and periods of volatility or stability.

#### Resampling Time Series Data
Resampling changes the frequency of observations in the time series. It is particularly useful for analyzing data at different time scales. Downsampling is used to aggregate data to a lower frequency (e.g., daily to monthly) and Upsampling is used to increases the frequency by introducing new time intervals (e.g., monthly to daily).

Downsampling Example: Convert monthly data to yearly averages.



Upsampling Example: Convert monthly data to daily with interpolation.



#### Rolling Statistics: Smoothing Trends
Rolling statistics smooth out short-term fluctuations to highlight long-term trends. Common statistics include rolling means and variances.

Python Code: Rolling Mean



#### Decomposition of Time Series
Time series can be broken down into Trend (Long-term movement), Seasonality (Repeating cycles), and Residual (Remaining noise or random fluctuations). There are other options for decomposition but we will focus on these three.

Python Code: Seasonal Decomposition



Trends highlight long-term growth or decline. Seasonal patterns reveal cyclical behavior. Residuals indicate noise or random fluctuations.

#### Heatmaps for Seasonal Patterns
Heatmaps are not a time series technique but they can help us visualize seasonality in time series data, such as variations across months and years.

Python Code: Heatmap of Seasonal Data



#### Visualizing Anomalies
Anomalies are sudden deviations from expected patterns. Highlighting these visually can help identify important events. In this case, we define anomalies as a value that is more than two standard deviations from the moving 12 month average.

Python Code: Highlighting Anomalies



#### Next Steps
Visualizing time series data lets us see the structure, trends, seasonality, and anomalies in a time series. Resampling, rolling statistics, and decomposition technies help us model the data modeling strategies. Python libraries like matplotlib, pandas, and seaborn can do the heavy lifting to effectively for this type of analysis.

#### Bee Example
In each part I include anvexample with a fictional beehive. Visualizing data can help us see

- Trends: Gradual weight increase due to honey production.
- Seasonality: Daily bee traffic peaks during daylight hours.
- Anomalies: Sudden drops in weight could indicate honey theft or swarming.



#### Related Posts
This article is part of a series of posts on time series forecasting. Here is the list of articles in the order they were designed to be read.

1.  [[Time Series for Business Analytics with Python](https://medium.com/@kylejones_47003/time-series-for-business-analytics-with-python-a92b30eecf62?source=your_stories_page-------------------------------------)]
2.  [[Time Series Visualization for Business Analysis with Python](https://medium.com/@kylejones_47003/time-series-visualization-for-business-analysis-with-python-5df695543d4a?source=your_stories_page-------------------------------------)]
3.  [[Patterns in Time Series for Forecasting](https://medium.com/@kylejones_47003/patterns-in-time-series-for-forecasting-8a0d3ad3b7f5?source=your_stories_page-------------------------------------)]
4.  [[Imputing Missing Values in Time Series Data for Business Analytics with Python](https://medium.com/@kylejones_47003/imputing-missing-values-in-time-series-data-for-business-analytics-with-python-b30a1ef6aaa6?source=your_stories_page-------------------------------------)]
5.  [[Measuring Error in Time Series Forecasting with Python](https://medium.com/@kylejones_47003/measuring-error-in-time-series-forecasting-with-python-18d743a535fd?source=your_stories_page-------------------------------------)]
6.  [[Univariate and Multivariate Time Series Analysis with Python](https://medium.com/@kylejones_47003/univariate-and-multivariate-time-series-analysis-with-python-b22c6ec8f133?source=your_stories_page-------------------------------------)]
7.  [[Feature Engineering for Time Series Forecasting in Python](https://medium.com/@kylejones_47003/feature-engineering-for-time-series-forecasting-in-python-7c469f69e260?source=your_stories_page-------------------------------------)]
8.  [[Anomaly Detection in Time Series Data with Python](https://medium.com/@kylejones_47003/anomaly-detection-in-time-series-data-with-python-5a15089636db?source=your_stories_page-------------------------------------)]
9.  [[Dickey-Fuller Test for Stationarity in Time Series with Python](https://medium.com/@kylejones_47003/dickey-fuller-test-for-stationarity-in-time-series-with-python-4e4bf1953eed?source=your_stories_page-------------------------------------)]
10. [[Using Classification Model for Time Series Forecasting with Python](https://medium.com/@kylejones_47003/using-classification-model-for-time-series-forecasting-with-python-d74a1021a5c4?source=your_stories_page-------------------------------------)]
11. [[Measuring Error in Time Series Forecasting with Python](https://medium.com/@kylejones_47003/measuring-error-in-time-series-forecasting-with-python-18d743a535fd?source=your_stories_page-------------------------------------)]
12. [[Physics-informed anomaly detection in a wind turbine using Python with an autoencoder transformer](https://medium.com/@kylejones_47003/physics-informed-anomaly-detection-in-a-wind-turbine-using-python-with-an-autoencoder-transformer-06eb68aeb0e8?source=your_stories_page-------------------------------------)]
