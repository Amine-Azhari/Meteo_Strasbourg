import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

df = pd.read_csv("strasbourg_entzheim.csv")
df["time"] = pd.to_datetime(df["time"])

signal_avg = df["tavg"].values
signal_max = df["tmax"].values
signal_min = df["tmin"].values

print(df.columns)

N = len(signal_avg)
dt = 1
xf = fftfreq(N, dt)

# CALCUL DE LA FFT ET AMPLITUDE POUR CHAQUE SIGNAL

# Température moyenne
yf_avg = fft(signal_avg)
amplitude_avg = 2.0 / N * np.abs(yf_avg[:N//2])

# Température maximale
yf_max = fft(signal_max)
amplitude_max = 2.0 / N * np.abs(yf_max[:N//2])

# Température minimale
yf_min = fft(signal_min)
amplitude_min = 2.0 / N * np.abs(yf_min[:N//2])

# Création d'une figure avec 3 sous graphique
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

ax1.plot(xf[1:N//2], amplitude_avg[1:N//2], color='b')
ax1.set_title("Température Moyenne (tavg)") 
ax1.set_ylabel("Amplitude (°C)")
ax1.grid(True)

ax2.plot(xf[1:N//2], amplitude_max[1:N//2], color='r')
ax2.set_title("Température Maximale (tmax)")
ax2.set_ylabel("Amplitude (°C)")
ax2.grid(True)

ax3.plot(xf[1:N//2], amplitude_min[1:N//2], color='g')
ax3.set_title("Température Minimale (tmin)")
ax3.set_ylabel("Amplitude (°C)")
ax3.set_xlabel("Fréquence (cycles / jour)")
ax3.grid(True)

plt.xlim(0, 0.04)
plt.savefig('fft_analyse.png')
plt.show(block=False)  
df['annee'] = df['time'].dt.year


# Calcul des moyennes annuelles
temp_annuelle_avg = df.groupby('annee')['tavg'].mean()
temp_annuelle_max = df.groupby('annee')['tmax'].mean()
temp_annuelle_min = df.groupby('annee')['tmin'].mean()

x = temp_annuelle_avg.index.values
y_avg = temp_annuelle_avg.values
y_max = temp_annuelle_max.values
y_min = temp_annuelle_min.values

#Régression linéaire 
coef_lin = np.polyfit(x, y_avg, 1)
tendance_lin = coef_lin[0] * x + coef_lin[1]


print(f"Régression linéaire : {coef_lin[0]*10:.3f} °C/décennie")
print(f"Régression polynomiale : degré 2")

plt.figure(figsize=(14, 6))
plt.plot(x, y_avg, 'o', label='Température moyenne annuelle')
plt.plot(x, tendance_lin, 'r--', linewidth=2, label=f'Régression linéaire ({coef_lin[0]*10:.2f}°C/décennie)')

plt.xlabel('Année')
plt.ylabel('Température (°C)')
plt.title('Régressions linéaire et polynomiale')
plt.legend()
plt.grid()
plt.savefig('regressions.png')
plt.show(block=False)

#ANALYSE DES EXTRÊMES : MAX/MIN PAR ANNÉE
print("\n=== ANALYSE DES EXTRÊMES PAR ANNÉE ===")

# Max et min absolus par année
max_par_annee = df.groupby('annee')['tmax'].max()
min_par_annee = df.groupby('annee')['tmin'].min()

print(f"Température maximale absolue : {max_par_annee.max():.2f} °C en {max_par_annee.idxmax()}")
print(f"Température minimale absolue : {min_par_annee.min():.2f} °C en {min_par_annee.idxmin()}")

# Tendances des extrêmes
x_ext = max_par_annee.index.values
coef_max = np.polyfit(x_ext, max_par_annee.values, 1)
coef_min = np.polyfit(x_ext, min_par_annee.values, 1)

print(f"Évolution des maximums : {coef_max[0]*10:.3f} °C/décennie")
print(f"Évolution des minimums : {coef_min[0]*10:.3f} °C/décennie")

plt.figure(figsize=(14, 6))
plt.plot(x_ext, max_par_annee.values,  label='Maximum annuel (tmax)')
plt.plot(x_ext, min_par_annee.values,  label='Minimum annuel (tmin)')
plt.plot(x_ext, coef_max[0] * x_ext + coef_max[1], 'r--', alpha=0.7, label=f'Tendance max ({coef_max[0]*10:.2f}°C/décennie)')
plt.plot(x_ext, coef_min[0] * x_ext + coef_min[1], 'b--', alpha=0.7, label=f'Tendance min ({coef_min[0]*10:.2f}°C/décennie)')
plt.fill_between(x_ext, min_par_annee.values, max_par_annee.values, alpha=0.2, color='gray')
plt.xlabel('Année')
plt.ylabel('Température (°C)')
plt.title('Évolution des températures extrêmes par année')
plt.legend()
plt.grid()
plt.savefig('extremes_par_annee.png')
plt.show(block=False)

plt.show()