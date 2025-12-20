import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.linalg import dft


###### Radar setting ######

c = 299792458

class cyclist_env():
    def __init__(self, f_carrier, N_ant, BW, BW_sub, N_sample, rx_sample, T_sample, mu, c):
        self.f_carrier = f_carrier
        self.N_ant = N_ant
        self.BW = BW
        self.BW_sub = BW_sub
        self.tx_sample = N_sample
        self.rx_sample = rx_sample
        self.T_sample = T_sample
        self.mu = mu
        self.c = c
    
    def phys_quantities(self, p_bs, p_cy, v_cy, n_cy, p_ve, v_ve, n_ve, p_rp, v_rp, n_rp):
        self.n_cy = n_cy
        self.n_ve = n_ve
        self.n_rp = n_rp

        calculator = cals(p_bs, p_cy, p_ve, p_rp, v_cy, v_ve, v_rp)
        angs = calculator.anglecal()
        vels = calculator.rel_velosity()
        ranges = calculator.rangecal()

        cy_ang, ve_ang, rp_ang = angs[0], angs[1], angs[2]
        #cy_ang, ve_ang = [-30*np.pi/180, 30*np.pi/180], []
        cy_sv = stevec(self.N_ant, cy_ang)
        ve_sv = stevec(self.N_ant, ve_ang)
        rp_sv = stevec(self.N_ant, rp_ang)

        ####### you should change this value -> complex one #######
        #cy_coeff = np.abs((np.random.randn(n_cy) + 1j*np.random.randn(n_cy))/np.sqrt(2))
        #ve_coeff = np.abs((np.random.randn(n_ve) + 1j*np.random.randn(n_ve))/np.sqrt(2))
        cy_coeff = np.ones(n_cy)
        ve_coeff = np.ones(n_ve)
        rp_coeff = np.ones(n_rp)
        
        
        self.result = {
            'relative_velocity' : {'cyclists': vels[0] , 'vehicles': vels[1], 'ramposts': vels[2]},
            'range' : {'cyclists': ranges[0] , 'vehicles': ranges[1], 'ramposts': ranges[2]},
            'time_difference' : {'cyclists': 2*ranges[0]/c , 'vehicles': 2*ranges[1]/c, 'ramposts': 2*ranges[2]/c},
            'coeff' : {'cyclists': cy_coeff , 'vehicles': ve_coeff, 'ramposts': rp_coeff},
            'steering_vector' : {'cyclists': cy_sv , 'vehicles': ve_sv, 'ramposts': rp_sv},
            'angle' : {'cyclists': cy_ang , 'vehicles': ve_ang, 'ramposts': rp_ang}
            }
        return self.result
    
    def tx(self):
        f_sub = self.BW_sub * np.arange(self.N_ant)
        f_carrier_foreach = f_sub
        f_carrier_foreach = np.matrix(f_carrier_foreach).T
        n = np.arange(self.tx_sample)
        chirp = np.exp(1j*np.pi*(np.sqrt(self.mu)*self.T_sample*n)**2)
        carriers = np.exp(1j*2*np.pi*f_carrier_foreach*(self.T_sample*n))
        tx = np.multiply(carriers, chirp)/np.sqrt(self.N_ant)
        return tx

    def rx_multiple(self, tx, P_rx_cy, tstemp_rx_cy, phase_cy, P_rx_ve, tstemp_rx_ve, phase_ve, P_rx_rp, tstemp_rx_rp, phase_rp, P_N_dB, sym_duration, N_trans):
        Y_multi = np.zeros((N_trans, self.N_ant, self.rx_sample), dtype=np.complex128)
        P_N = 10**(P_N_dB/10)*(1e-3)

        stv = self.result['steering_vector']['cyclists']
        relvel = self.result['relative_velocity']['cyclists']
        dist = self.result['range']['cyclists']
        
        for i in range(self.n_cy):
            fd = 2*relvel[i]*self.f_carrier/c
            dq = np.diag(np.exp(1j*2*np.pi*fd*np.arange(self.tx_sample)*self.T_sample))
            stv_each = stv[:,i]
            Y = np.matrix(np.zeros((self.N_ant, self.rx_sample), dtype=np.complex128))
            for j in range(len(P_rx_cy[i])):
                delay, coeff = int(tstemp_rx_cy[i][j]), P_rx_cy[i][j]
                last = min(delay+self.tx_sample, self.rx_sample)
                if last-delay < 1 : break
                Y[:,delay:last] += np.sqrt(coeff)*np.exp(1j*phase_cy[i][j])*stv_each.conj()@stv_each.H@tx[:,:last-delay]@dq[:last-delay,:last-delay]
                # print(coeff)
                # print(np.mean(np.sum(np.power(np.abs(YY),2), axis=0)))
                # print(np.trace(YY@np.transpose(np.conjugate(YY)))/self.tx_sample)
                # print("------------")
                


            dq_multi = np.zeros((N_trans, 1, 1), dtype=np.complex128)
            dq_multi[:, 0, 0] = np.exp(1j*2*np.pi*fd*np.arange(N_trans)*sym_duration)
            Y_dopp = dq_multi * np.array(Y)
            Y_multi += Y_dopp


        stv = self.result['steering_vector']['vehicles']
        relvel = self.result['relative_velocity']['vehicles']
        dist = self.result['range']['vehicles']
        for i in range(self.n_ve):
            fd = 2*relvel[i]*self.f_carrier/c
            dq = np.diag(np.exp(1j*2*np.pi*fd*np.arange(self.tx_sample)*self.T_sample))
            stv_each = stv[:,i]
            Y = np.matrix(np.zeros((self.N_ant, self.rx_sample), dtype=np.complex128))
            for j in range(len(P_rx_ve[i])):
                delay, coeff = int(tstemp_rx_ve[i][j]), P_rx_ve[i][j]
                last = min(delay+self.tx_sample, self.rx_sample)
                if last-delay < 1 : break
                Y[:,delay:last] += np.sqrt(coeff)*np.exp(1j*phase_ve[i][j])*stv_each.conj()@stv_each.H@tx[:,:last-delay]@dq[:last-delay,:last-delay]

            dq_multi = np.zeros((N_trans, 1, 1), dtype=np.complex128)
            dq_multi[:, 0, 0] = np.exp(1j*2*np.pi*fd*np.arange(N_trans)*sym_duration)
            Y_dopp = dq_multi * np.array(Y)
            Y_multi += Y_dopp

        stv = self.result['steering_vector']['ramposts']
        relvel = self.result['relative_velocity']['ramposts']
        dist = self.result['range']['ramposts']
        for i in range(self.n_rp):
            fd = 2*relvel[i]*self.f_carrier/c
            dq = np.diag(np.exp(1j*2*np.pi*fd*np.arange(self.tx_sample)*self.T_sample))
            stv_each = stv[:,i]
            Y = np.matrix(np.zeros((self.N_ant, self.rx_sample), dtype=np.complex128))
            for j in range(len(P_rx_rp[i])):
                delay, coeff = int(tstemp_rx_rp[i][j]), P_rx_rp[i][j]
                last = min(delay+self.tx_sample, self.rx_sample)
                if last-delay < 1 : break
                Y[:,delay:last] += np.sqrt(coeff)*np.exp(1j*phase_rp[i][j])*stv_each.conj()@stv_each.H@tx[:,:last-delay]@dq[:last-delay,:last-delay]

            dq_multi = np.zeros((N_trans, 1, 1), dtype=np.complex128)
            dq_multi[:, 0, 0] = np.exp(1j*2*np.pi*fd*np.arange(N_trans)*sym_duration)

            Y_dopp = dq_multi * np.array(Y)
            Y_multi += Y_dopp

            

        N = np.sqrt(P_N)*(np.random.randn(N_trans, self.N_ant, self.rx_sample) + 1j*np.random.randn(N_trans, self.N_ant, self.rx_sample))/np.sqrt(2)
        # print(np.mean(np.power(np.abs(N), 2)))
        # print(10*np.log10(np.mean(np.power(np.abs(N), 2))))

        Y_multi += N
        return Y_multi

class cals():
    def __init__(self, bs, cy, ve, rp, v_cy, v_ve, v_rp):
        self.bs, self.cy, self.ve, self.rp, self.v_cy, self.v_ve, self.v_rp = bs, cy, ve, rp, v_cy, v_ve, v_rp
    def rangecal(self):
        if len(self.cy)!=0 : cy_range = np.linalg.norm(self.bs-self.cy, axis = 1)
        else: cy_range = np.array([])

        if len(self.ve)!=0: ve_range = np.linalg.norm(self.bs-self.ve, axis = 1)
        else: ve_range = np.array([])

        rp_range = np.linalg.norm(self.bs-self.rp, axis = 1)
        
        return cy_range, ve_range, rp_range
    
    def anglecal(self):
        refangle = np.arctan(33/265)/2
        if len(self.cy)!=0 :
            cy_denum = self.bs[0] - self.cy[:,0]
            cy_nu = self.cy[:,1] - self.bs[1]
            cy_ang = np.arctan(cy_nu/cy_denum) - refangle
        else: cy_ang = np.array([])
        if len(self.ve)!=0 :
            ve_denum = self.bs[0] - self.ve[:,0]
            ve_nu = self.ve[:,1] - self.bs[1]
            ve_ang = np.arctan(ve_nu/ve_denum) - refangle
        else: ve_ang = np.array([])

        rp_denum = self.bs[0] - self.rp[:,0]
        rp_nu = self.rp[:,1] - self.bs[1]
        rp_ang = np.arctan(rp_nu/rp_denum) - refangle
        return cy_ang, ve_ang, rp_ang
    
    def rel_velosity(self):
        v_cy_rel, v_ve_rel = [], []
        if len(self.cy)!=0 :
            cy_reldirection = (self.bs - self.cy)
            for i in range(len(self.cy)):
                const = np.linalg.norm(cy_reldirection[i], 2)
                temp = cy_reldirection[i].dot(self.v_cy[i])/const
                v_cy_rel.append(temp)
        else: v_cy_rel = np.array([])
        if len(self.ve)!=0 :
            ve_reldirection = (self.bs - self.ve)
            for i in range(len(self.ve)):
                const = np.linalg.norm(ve_reldirection[i], 2)
                temp = ve_reldirection[i].dot(self.v_ve[i])/const
                v_ve_rel.append(temp)
        else: v_ve_rel = np.array([])
        v_rp_rel = [0,0,0]
        return v_cy_rel, v_ve_rel, v_rp_rel
        


    
def stevec(N_ant, angle):
    resp = (np.arange(N_ant)-N_ant*0.5+0.5).reshape([-1,1])
    resp = np.exp(1j*resp*np.pi*np.sin(angle))
    return np.matrix(resp)

    