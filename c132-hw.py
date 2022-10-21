
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("stars.csv")

mass_list = df["mass"].tolist()
mass_list.pop(0)

radius_list = df["radius"].tolist()
radius_list.pop(0)



plt.figure()
sns.scatterplot(x = mass_list,y = radius_list)
plt.title("STAR MASS AND RADIUS")
plt.xlabel('MASS')
plt.ylabel('RADIUS')
plt.show()

import csv
import pandas as pd

rows = []

with open("stars.csv",'r') as f:
    csvreader = csv.reader(f)
    for row in csvreader:
        rows.append(row)

headers = rows[0]
star_data = rows[1:]
df = pd.read_csv("stars.csv")
solar_mass_list = df["solar_mass"].tolist()
solar_radius_list = df["solar_radius"].tolist()

solar_mass_list.pop(0)
solar_radius_list.pop(0)
star_solar_mass_si_unit = []

for data in solar_mass_list:

    si_unit = float(data)*1.989e+30
    star_solar_mass_si_unit.append(si_unit)

print(star_solar_mass_si_unit)
star_solar_radius_si_unit = []

for data in solar_radius_list:
    si_unit = float(data)* 6.957e+8
    star_solar_radius_si_unit.append(si_unit)

print(star_solar_radius_si_unit)

star_masses = star_solar_mass_si_unit
star_radiuses = star_solar_radius_si_unit
star_names = df["star_names"].tolist()
star_names.pop(0)

star_gravities = []

for index,data in enumerate(star_names):
    gravity = (float(star_masses[index])*5.972e+24) / (float(star_radiuses[index])*float(star_radiuses[index])*6371000*6371000) * 6.674e-11
    star_gravities.append(gravity)

print(star_gravities)

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("stars.csv")

radius_list = df["radius"].tolist()
radius_list.pop(0)



mass_list = df["mass"].tolist()
mass_list.pop(0)


x = []

for data in radius_list:
    x.append(data)

for data in mass_list:
    x.append(data)

wcss = []

for i in range(1,11):
  kmeans = KMeans(n_clusters = i,init = 'k-means++',random_state = 42)
  kmeans.fit(x)
  wcss.append(kmeans.inertia_)

plt.figure(figsize = (10,5))
sns.lineplot(range(1,11),wcss,marker = 'o',color = 'red')
plt.title('THE ELBOW METHOD')
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.show()
