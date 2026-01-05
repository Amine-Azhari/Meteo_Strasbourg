import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram, butter, filtfilt, correlate

class Plotting:
    ROWS = 3 # 3 pour 10
    COLS = 4 # 4 pour 10

    def __init__(self, duree):
        self.ZOOM = (1/duree) * 1.2

    
    #cut_off représente la fréquence de coupure
    #les résultats que j'ai trouvé pertinents étaient en mettant :
    #période = 365 (un an), cut_off = 0.0027 (1/365)

    #filtre passe bas
    
    @staticmethod
    def lowpass_filter(signal, fs, cut_off=0.01, order=2):
        nyq = 0.5 * fs
        normal_cut_off = cut_off / nyq
        b, a = butter(order, normal_cut_off, btype='low', analog=False)
        filtered = filtfilt(b, a, signal)
        return filtered

    #filtre passe haut

    @staticmethod
    def highpass_filter(signal, fs, cut_off=0.01, order=2):
        nyq = 0.5 * fs
        normal_cut_off = cut_off / nyq
        b, a = butter(order, normal_cut_off, btype='high', analog=False)
        filtered = filtfilt(b, a, signal)
        return filtered
    

    @staticmethod
    def helper(plotted_var, filter_type=None, cutoff=0.01, fs=1):
        s = pd.to_numeric(df[plotted_var], errors='coerce').dropna()
        if len(s) == 0:
            return np.array([]), np.array([])

        #On rajoute ces 2 lignes pour le cas où on veut filtrer
        if filter_type == 'low':
            s = Plotting.lowpass_filter(s.values, fs, cutoff)
        elif filter_type == 'high':
            s = Plotting.highpass_filter(s.values, fs, cutoff)
        #------------------------------------------------------

        t = np.arange(len(s))

        fft = np.fft.fft(s)
        fft[0] = 0
        fftfreq = np.fft.fftfreq(len(s)) * len(s) / (t.max() - t.min())

        return (fftfreq, abs(fft))


    def plotter(self, ax, var, filter_type=None, cutoff=0.01):
        freq, fft = self.helper(var, filter_type=filter_type, cutoff=cutoff)

        if len(freq) == 0:
            ax.set_title(f"{var} (no data)")
            return

        ax.plot(freq, fft)
        ax.set_title(var)
        ax.axis([-self.ZOOM, self.ZOOM, 0, 1.1*max(fft)])



    def plot(self, args, filter_type=None, cutoff=0.01):
        plt.figure(figsize=(15, 10))
        axes = [plt.subplot2grid((self.ROWS, self.COLS), (i // self.COLS, i % self.COLS))
                for i in range(len(args))]
        for i in range(len(args)):
            self.plotter(axes[i], args[i], filter_type=filter_type, cutoff=cutoff)



    def spectro(self, var, filter_type=None, cutoff=0.01):
        signal = pd.to_numeric(df[var], errors="coerce").dropna().values
        if len(signal) == 0:
            print(f"Pas de données exploitables pour {var}")
            return

        fs = 1  #sampliing frequency, ici on prend 1 valeur par jour (fs=0.5 <=> 2 valeur par jour)

        #Pareil si on veut filtrer
        if filter_type == 'low':
            signal = self.lowpass_filter(signal, fs, cutoff)
        elif filter_type == 'high':
            signal = self.highpass_filter(signal, fs, cutoff)
        #-------------------------

        frequencies, times, Sxx = spectrogram(
            signal,
            fs,
            nperseg=365,
            noverlap=300,
            scaling='density'
        )

        plt.figure(figsize=(12, 6))
        plt.pcolormesh(times, frequencies, Sxx, shading='gouraud')
        plt.ylabel('Fréquence (cycles/jour)')
        plt.xlabel('Temps (jours depuis le début)')
        plt.title(f"Spectrogramme de : {var} ({filter_type if filter_type else 'raw'})")
        plt.colorbar(label='Amplitude')
        plt.show()


    @staticmethod
    def autocorr_scipy(var, filter_type=None, cutoff=0.01):
        s = pd.to_numeric(df[var], errors="coerce").dropna().values
        if len(s) == 0:
            return np.array([]), np.array([])

        fs = 1
        if filter_type == 'low':
            s = Plotting.lowpass_filter(s, fs, cutoff)
        elif filter_type == 'high':
            s = Plotting.highpass_filter(s, fs, cutoff)

        s = s - np.mean(s)
        norm = np.sum(s**2)
        corr = correlate(s, s, mode="full") / norm
        lags = np.arange(-len(s)+1, len(s))
        return lags, corr


    def plot_autocorr(self, ax, var, max_lag=400, filter_type=None, cutoff=0.01):
        lags, corr = self.autocorr_scipy(var, filter_type=filter_type, cutoff=cutoff)
        if len(lags) == 0:
            ax.set_title(f"{var} (no data)")
            return

        mask = (lags >= 0) & (lags <= max_lag)
        lags_plot = lags[mask]
        corr_plot = corr[mask]

        ax.plot(lags_plot, corr_plot)
        ax.set_title(f"Autocorr: {var} ({filter_type if filter_type else 'raw'})")
        ax.set_xlabel("Décalage (jours)")
        ax.set_ylabel("Corrélation")
        ax.set_xlim(0, max_lag)
        ax.set_ylim(-0.2, 1)

        # mark 1-year period
        if max_lag >= 365:
            ax.axvline(365, color='r', linestyle='--', label='1 an')
            ax.legend()

    def plot_autocorrs(self, args, max_lag=400, filter_type=None, cutoff=0.01):
        fig, axes = plt.subplots(self.ROWS, self.COLS, figsize=(16, 4*self.ROWS))
        axes = axes.flatten()
        for ax, var in zip(axes, args):
            self.plot_autocorr(ax, var, max_lag=max_lag, filter_type=filter_type, cutoff=cutoff)
        for ax in axes[len(args):]:
            ax.axis("off")
        plt.tight_layout()


if __name__ == "__main__":
    df = pd.read_csv('strasbourg_entzheim.csv')
    args = list(df)
    args.pop(0)

    duree = int(input("Insérer période recherchée en jours: "))
    print("Des pics vers les extrémités droite et gauche du graphe indiquent une périodicité sur la durée")

    pltr = Plotting(duree)

    filter_choice = input("Filtre? (none / low / high): ").lower()
    cutoff_value = float(input("Valeur cutoff (ex: 0.01): "))

    pltr.plot(args, filter_type=filter_choice, cutoff=cutoff_value)
    pltr.plot_autocorrs(args)
    #i = 3
    #pltr.spectro(args[i], filter_type=filter_choice, cutoff=cutoff_value)

    plt.show()