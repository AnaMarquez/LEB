import scipy as sp
import pylab as plt
from scipy.integrate import odeint
import numpy as np
import scipy.signal as psb

class HodgkinHuxley():
    '''
    Clase que define o comportamento do neurionio mediante o Modelo de Hodgkin - Huxley (1952) 
    '''
    # Bloco de constantes 
    C_m  =   1.0 # Capacitancia da membrana (uF/cm^2)
    g_Na = 120.0 # Conductancia maxima do Sodio - Na (mS/cm^2)
    g_K  =  36.0 # Conductancia maxima do Potassio - K (mS/cm^2)
    g_L  =   0.3 # Conductancia de fuga maxima - Leak (mS/cm^2)
    E_Na =  50.0 # Potencial de inversao de Nerst - Na (mV)
    E_K  = -77.0 # Potencial de inversao de Nerst - K (mV)
    E_L  = -54.387 # Potencial de inversao de Nerst - Leak (mV)
    t = sp.arange(0.0, 2400.0, 0.01) # Tempo de integracao
    
    def __init__(self, sinal_estimulo, f0_1, f0_2, A_1, A_2, phi, cte):
        self.sinal_estimulo = sinal_estimulo
        self.f0_1 = f0_1
        self.f0_2 = f0_2
        self.A_1 = A_1
        self.A_2 = A_2
        self.phi = phi
        self.cte = cte
        
    def alpha_m(self, V):
        '''
        Cinetica de propagacao no canal -> funcao de voltage da membrana
        '''
        return 0.1*(V+40.0)/(1.0 - sp.exp(-(V+40.0) / 10.0))

    def beta_m(self, V):
        '''
        Cinetica de propagacao no canal -> funcao de voltage da membrana
        '''
        return 4.0*sp.exp(-(V+65.0) / 18.0)

    def alpha_h(self, V):
        '''
        Cinetica de propagacao no canal -> funcao de voltage da membrana
        '''
        return 0.07*sp.exp(-(V+65.0) / 20.0)

    def beta_h(self, V):
        '''
        Cinetica de propagacao no canal -> funcao de voltage da membrana
        '''
        return 1.0/(1.0 + sp.exp(-(V+35.0) / 10.0))

    def alpha_n(self, V):
        '''
        Cinetica de propagacao no canal -> funcao de voltage da membrana
        '''
        return 0.01*(V+55.0)/(1.0 - sp.exp(-(V+55.0) / 10.0))

    def beta_n(self, V):
        '''
        Cinetica de propagacao no canal -> funcao de voltage da membrana
        '''
        return 0.125*sp.exp(-(V+65) / 80.0)

    def I_Na(self, V, m, h):
        '''
        Corriente na membrana para o Sodio - Na (uA/cm^2)
        Parametros de entrada: V, m, h
        '''
        return self.g_Na * m**3 * h * (V - self.E_Na)

    def I_K(self, V, n):
        '''
        Corriente na membrana para o Potassio - K (uA/cm^2)
        Parametros de entrada: V, n
        '''
        return self.g_K  * n**4 * (V - self.E_K)
    
    # Leak
    
    def I_L(self, V):
        '''
        Corriente na membrana - Leak (uA/cm^2)
        Parametros de entrada: V
        '''
        return self.g_L * (V - self.E_L)

    def I_inj(self, t):
        '''
        Corriente externa - Estimulo 
        Parametros de entrada: t, sinal_estimulo
        '''
        if self.sinal_estimulo == None: raise Exception('Precisa ser inidicado o tipo de sinal de entrada para estimular o neuronio: Degrau crecente, senoide, valor constante', 'soma_senoides')
            
        elif self.sinal_estimulo == 'degrau_crecente':
            entrada = self.A_1*(t>100) - self.A_1*(t>200) + self.A_2*(t>300) - self.A_2*(t>400)
            print(t, entrada)
        elif self.sinal_estimulo == 'senoide':
            if self.f0_1 == None: self.f0_1 = 300
            if self.phi == None: self.phi = 0 # phi = np.pi/2
            if self.A_1 == None: self.A_1 = 10
            if self.cte == None: self.cte = 0
            entrada = self.A_1 * np.sin(2 * np.pi * self.f0_1 * t/1000 + self.phi + self.cte) 
        elif self.sinal_estimulo == 'soma_senoides':
            if self.f0_1 == None: self.f0_1 = 300
            if self.f0_2 == None: self.f0_1 = 1
            if self.phi == None: self.phi = 0 # phi = np.pi/2
            if self.A_1 == None: self.A_1 = 10
            if self.A_2 == None: self.A_2 = self.A_1
            if self.cte == None: self.cte = 0
            sinal_1 = self.A_1 * np.sin(2 * np.pi * self.f0_1 * t/1000 + self.phi + self.cte) 
            sinal_2 = self.A_2 * np.sin(2 * np.pi * self.f0_2 * t/1000 + self.phi + self.cte)
            entrada = sinal_1 + sinal_2
        elif self.sinal_estimulo == 'valor_constante':
            if self.A_1 == None:
                A_1 = 15
            entrada = A_1
        
        return entrada 

    @staticmethod
    def dALLdt(X, t, self):
        '''
        Metodo que calcula as derivadas e retorna o valor do potencial de 
        membrana e as variaveis de ativacao
        Parametros de entrada: X, t
        '''
        V, m, h, n = X

        dVdt = (self.I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        dmdt = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m
        dhdt = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h
        dndt = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n
        
        return dVdt, dmdt, dhdt, dndt

    def Programa_principal(self, plot_resumo = False):
        '''
        Metodo principal que chama as funcoes necessarias para implementar o modelo
        Retorna plot com:
        - Valor do potencial de membrana
        - Corriente dos canais
        - Valores dos canais
        - Sinal estimulo de entrada
        '''
        
        X = odeint(self.dALLdt, [-65, 0.05, 0.6, 0.32], self.t, args=(self,))
        V = X[:,0]
        m = X[:,1]
        h = X[:,2]
        n = X[:,3]
        ina = self.I_Na(V, m, h)
        ik = self.I_K(V, n)
        il = self.I_L(V)
        
        if plot_resumo is True:
            plt.figure()
    
            plt.subplot(4,1,1)
            plt.title('$Comportamento$ $neuronio$ $Modelo$ $Hodgkin-Huxley$')
            plt.plot(self.t, V, 'k')
            plt.ylabel('$V$ $(mV)$')
    
            plt.subplot(4,1,2)
            plt.plot(self.t, ina, 'c', label='$I_{Na}$')
            plt.plot(self.t, ik, 'y', label='$I_{K}$')
            plt.plot(self.t, il, 'm', label='$I_{L}$')
            plt.ylabel('$Intensidade$ $(mA)$')
            plt.legend()
    
            plt.subplot(4,1,3)
            plt.plot(self.t, m, 'r', label='$m$')
            plt.plot(self.t, h, 'g', label='$h$')
            plt.plot(self.t, n, 'b', label='$n$')
            plt.ylabel('$Aberturar$ $do$ $canal$')
            plt.legend()
    
            plt.subplot(4,1,4)
            i_inj_values = [self.I_inj(t) for t in self.t]
            if self.sinal_estimulo == 'senoide':
                plt.plot(self.t, i_inj_values, 'k', label = '$Amplitude$ $(\\mu{A})$: ' + str(self.A_1) + '\n$Freq$ $(Hz)$: ' + str(self.f0_1))
            elif self.sinal_estimulo == 'soma_senoides':
                plt.plot(self.t, i_inj_values, 'k', label = '$Amplitude$ $1$ $(\\mu{A})$: ' + str(self.A_1) + '\n$Amplitude$ $2$ $(\\mu{A})$: ' + 
                         str(self.A_2) + '\n$Freq$ $1$ $(Hz)$: ' + str(self.f0_1) + '\n$Freq$ $2$ $(Hz)$: ' + str(self.f0_2))
            else:
                plt.plot(self.t, i_inj_values, 'k')
            plt.xlabel('$Tempo$ $(ms)$')
            plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
            if self.A_1 == None: self.A_1 = 0
            if self.A_2 == None: self.A_2 = 0
            if self.cte != 0: plt.ylim((- self.A_1 - self.A_2), (self.A_1 + self.A_2))            
            else: plt.ylim((-self.A_1 - self.A_2 ), (self.A_1 + self.A_2))
            plt.legend()
    #         plt.annotate('$Amplitude$ $(\\mu{A})$: ' + str(amplitude), xy=(-9,19), xycoords='figure fraction',
    #                 horizontalalignment='right', verticalalignment='top',
    #                 fontsize=11)
    
            plt.show()
    
        return V

    def psb(self, sinal):
        '''
        Metodo que avalia a sinal, carateriza ela
        '''
        print(sinal)
        fft = np.fft.fft(sinal)
        plt.plot(sinal, label = 'sinal')
        plt.plot(np.fft.fft(sinal), label = 'fft')
        plt.legend()
        plt.show()
        
if __name__ == '__main__':
    
    # Especificar o modelo
    tipo_estimulo = 'soma_senoides' # Tipos de entradas possiveis: valor_constante, degrau_crecente, senoide, soma_senoides
    freq_1 = 200 # (Hz)
    freq_2 =  1 # (Hz)
    intensidade_1 = 1 # (microA)
    intensidade_2 = 13 # (microA)
    desfase = None
    valor_cte = 8 # (microA) valor adicional senoide
    
    HH = HodgkinHuxley(sinal_estimulo = tipo_estimulo, f0_1 = freq_1, f0_2 = freq_2, A_1 = intensidade_1,
                        A_2 = intensidade_2, phi = desfase, cte = valor_cte)
    
    voltage = HH.Programa_principal()
    HH.psb(voltage)
    
    print(len(voltage))