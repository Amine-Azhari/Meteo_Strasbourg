import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

class Plotting:
    ROWS = 3 # 3 pour 10
    COLS = 4 # 4 pour 10

    def __init__(self, duree):
        self.ZOOM = (1/duree) * 1.2


    @staticmethod
    def helper(plotted_var):
        s = pd.to_numeric(df[plotted_var], errors='coerce').dropna()
        if len(s) == 0:
            return np.array([]), np.array([])
        
        t = np.arange(len(s))

        fft = np.fft.fft(s)
        fft[0] = 0
        fftfreq = np.fft.fftfreq(len(s))*len(s)/(t.max()-t.min())

        return (fftfreq, abs(fft))


    def plotter(self, ax, var):
        freq, fft = self.helper(var)

        if len(freq) == 0:
            ax.set_title(f"{var} (no data)")
            return

        ax.plot(freq, fft)
        ax.set_title(var)
        ax.axis([-self.ZOOM, self.ZOOM, 0, 1.1*max(fft)])


    def plot(self, args):
        plt.figure(figsize=(15, 10)) 
        axes = [plt.subplot2grid((self.ROWS,self.COLS), (i// self.COLS,i % self.COLS)) \
                for i in range(len(args))]
        for i in range(len(args)):
            self.plotter(axes[i], args[i])


    def spectro(self,ax, var):
        signal = pd.to_numeric(df[var], errors="coerce").dropna().values

        if len(signal) == 0:
            ax.set_title(f"{var} (pas de données)")
            return

        fs = 1 

        frequencies, times, Sxx = spectrogram(
            signal,
            fs,
            nperseg=365,
            noverlap=300,
            scaling='density'
        )

        disp = ax.pcolormesh(times, frequencies, Sxx, shading='gouraud')
        ax.set_ylabel('Fréquence (cycles/jour)')
        ax.set_xlabel('Temps (jours)')
        ax.set_title(f"{var}")

        return disp

    def plot_spectros(self, args):
        fig, axes = plt.subplots(self.ROWS, self.COLS, figsize=(16, 4*self.ROWS))
        axes = axes.flatten()
        images = []
        for ax, var in zip(axes, args):
            im = self.spectro(ax, var)
            if im is not None:
                images.append(im)

        for ax in axes[len(args):]:
            ax.axis("off")

        if images:
            fig.colorbar(images[0], ax=axes, label="Amplitude")


if __name__ == "__main__":
    df = pd.read_csv('strasbourg_entzheim.csv')
    args = list(df)
    args.pop(0)

    
    
    duree = int(input("Inserer periode recherchee en jours: "))
    print("Des pics vers les extremites droite et gauches du graphe indiquent une periodicitee sur la duree")

    pltr = Plotting(duree)

    
    #pltr.plot(args)
    # Spectrogrammes
    pltr.plot_spectros(args)
    plt.show()
    """
    snow = np.array(df["snow"][:800])
    t = np.arange(len(snow))

    plt.figure(figsize=(10,4))
    plt.plot(t, snow)
    plt.xlabel("Temps (jours)")
    plt.ylabel("Neige")
    plt.title("Neige journalière à Strasbourg - années 1950/51")
    plt.show()
    """