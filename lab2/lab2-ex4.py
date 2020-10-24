# Them thu vien

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# Nhap data

data = pd.read_csv('HappinessReport2020.csv')

data.head()

# So chieu cua mang

data.shape

# Kiem tra null

data.info()

# Bo di cot khong su dung

data = data.drop(['Standard error of ladder score', 'upperwhisker', 'upperwhisker', 'lowerwhisker', 'Explained by: Log GDP per capita', 'Explained by: Social support', 'Explained by: Healthy life expectancy', 'Explained by: Freedom to make life choices', 'Explained by: Generosity', 'Explained by: Perceptions of corruption'], axis=1)
data.head()

# Hinh dung

# Diem bac thang cao voi GDP, ho tro xa hoi, tuoi tho khoe manh va thap ve nhan thuc tham nhung

plt.rcParams['figure.figsize'] = (12,8)
sns.heatmap(data.corr(), cmap = 'copper', annot = True)

plt.show()

# Phan phoi so du lieu

plt.rcParams['figure.figsize'] = (12, 12)
data.hist();

# Top 10 nuoc hanh phuc

data[['Country name','Ladder score']].head(10)

top = data.sort_values(['Ladder score'],ascending = 0)[:10]
ax = sns.barplot(x = 'Ladder score' , y = 'Country name' , data = top)
ax.set_xlabel('Ladder score', size = 20)
ax.set_ylabel('Country name', size = 20)
ax.set_title("Top 10 Happiest Countries", size = 25)

# Dat nuoc hanh phuc nhat

map_plot = dict(type = 'choropleth', 
locations = data['Country name'],
locationmode = 'country names',
z = data['Ladder score'], 
text = data['Regional indicator'],
colorscale = 'rdylgn', reversescale = True)
layout = dict(title = 'Happiest Countries In The World ', 
geo = dict(showframe = False, 
projection = {'type': 'equirectangular'}))
choromap = go.Figure(data = [map_plot], layout=layout)
iplot(choromap)

# Top 10 nuoc song lanh manh

data[['Country name','Healthy life expectancy']].head(10)

top = data.sort_values(['Healthy life expectancy'],ascending = 0)[:10]
ax = sns.barplot(x = 'Healthy life expectancy' , y = 'Country name' , data = top)
ax.set_xlabel('Generosity', size = 20)
ax.set_ylabel('Country name', size = 20)
ax.set_title("Top 10 Healthiest Countries", size = 25)

# Nuoc hanh phuc nhat the gioi

map_plot = dict(type = 'choropleth', 
locations = data['Country name'],
locationmode = 'country names',
z = data['Healthy life expectancy'], 
text = data['Regional indicator'],
colorscale = 'rdylgn', reversescale = True)
layout = dict(title = 'Healthiest Countries In The World', 
geo = dict(showframe = False, 
projection = {'type': 'equirectangular'}))
choromap = go.Figure(data = [map_plot], layout=layout)
iplot(choromap)

# Top 10 nuoc ve ho tro xa hoi

data[['Country name','Social support']].head(10)

top = data.sort_values(['Social support'],ascending = 0)[:10]
ax = sns.barplot(x = 'Social support' , y = 'Country name' , data = top)
ax.set_xlabel('Social support', size = 20)
ax.set_ylabel('Country name', size = 20)
ax.set_title("Top 10 Social Support Countries", size = 25)

# Nuoc co ho tro xa hoi tot nhat

map_plot = dict(type = 'choropleth', 
locations = data['Country name'],
locationmode = 'country names',
z = data['Social support'], 
text = data['Regional indicator'],
colorscale = 'rdylgn', reversescale = True)
layout = dict(title = 'Social Support Countries In The World ', 
geo = dict(showframe = False, 
projection = {'type': 'equirectangular'}))
choromap = go.Figure(data = [map_plot], layout=layout)
iplot(choromap)

# Top 10 nuoc hao phong

data[['Country name','Generosity']].head(10)

top = data.sort_values(['Generosity'],ascending = 0)[:10]
ax = sns.barplot(x = 'Generosity' , y = 'Country name' , data = top)
ax.set_xlabel('Generosity', size = 20)
ax.set_ylabel('Country name', size = 20)
ax.set_title("Top 10 Most Generous Countries", size = 25)

# Top 10 nuoc dinh huong tu do

data[['Country name','Freedom to make life choices']].head(10)

top = data.sort_values(['Freedom to make life choices'],ascending = 0)[:10]
ax = sns.barplot(x = 'Freedom to make life choices' , y = 'Country name' , data = top)
ax.set_xlabel('Freedom to make life choices', size = 20)
ax.set_ylabel('Country name', size = 20)
ax.set_title("Top 10 Most With The Freedom Countries", size = 25)

# Tat ca khu vuc

data["Regional indicator"].value_counts()

# Chia theo khu vuc

fig = px.pie( names= data["Regional indicator"], title="Regions")
fig.show()

# Khu vuc hanh phuc nhat

sns.barplot(x="Ladder score", y="Regional indicator", data=data, palette='Accent')