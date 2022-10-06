import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from plot_model import plot_model
from SEIAR import deriv_High_and_Low

def main():
    # beta_sympt, beta_asympt, gamma: incubation, theta: ratio showing sympt, lambda: infection period, delta: death rate
    param_dict = {
        'High': [0.4e-6, 0.2e-6, 1/5, 0.4, 1/14, 0.001],
        'Low': [0.4e-6, 0.2e-6, 1/5, 0.4, 1/14, 0.001],
    }

    # coming in, going out
    mobility_rate = {
        'Central': [0.0001, 5*0.0001],
        'East': [0.0001, 5*0.0001],
        'North_Central': [0.0001, 5*0.0001],
        'North_Coastal': [0.0001, 5*0.0001],
        'North_Inland': [0.0001, 5*0.0001],
        'South': [0.0001, 5*0.0001]
    }

    init_pop = {
        'Central_High': [234144, 1, 1, 1, 0],
        'Central_Low': [277516, 0, 0, 0, 0],
        'East_High': [249102, 0, 0, 0, 0],
        'East_Low': [239228, 0, 0, 0, 0],
        'North_Central_High': [175301, 0, 0, 0, 0],
        'North_Central_Low': [471079, 0, 0, 0, 0],
        'North_Coastal_High': [159507, 0, 0, 0, 0],
        'North_Coastal_Low': [375712, 0, 0, 0, 0],
        'North_Inland_High': [195235, 0, 0, 0, 0],
        'North_Inland_Low': [406765, 0, 0, 0, 0],
        'South_High': [212538, 0, 0, 0, 0],
        'South_Low': [287538, 0, 0, 0, 0]
    }

    ########### Extract Corresponding Population ###########

    # Central High Risk
    S_H_C, E_H_C, I_H_C, A_H_C, R_H_C = init_pop['Central_High'][0],\
                                        init_pop['Central_High'][1],\
                                        init_pop['Central_High'][2],\
                                        init_pop['Central_High'][3],\
                                        init_pop['Central_High'][4]

    # Central Low Risk
    S_L_C, E_L_C, I_L_C, A_L_C, R_L_C = init_pop['Central_Low'][0],\
                                        init_pop['Central_Low'][1],\
                                        init_pop['Central_Low'][2],\
                                        init_pop['Central_Low'][3],\
                                        init_pop['Central_Low'][4]

    # East High Risk
    S_H_E, E_H_E, I_H_E, A_H_E, R_H_E = init_pop['East_High'][0],\
                                        init_pop['East_High'][1],\
                                        init_pop['East_High'][2],\
                                        init_pop['East_High'][3],\
                                        init_pop['East_High'][4]

    # East Low Risk
    S_L_E, E_L_E, I_L_E, A_L_E, R_L_E = init_pop['East_Low'][0],\
                                        init_pop['East_Low'][1],\
                                        init_pop['East_Low'][2],\
                                        init_pop['East_Low'][3],\
                                        init_pop['East_Low'][4]

    # North Central High Risk
    S_H_NCen, E_H_NCen, I_H_NCen, A_H_NCen, R_H_NCen = init_pop['North_Central_High'][0],\
                                                    init_pop['North_Central_High'][1],\
                                                    init_pop['North_Central_High'][2],\
                                                    init_pop['North_Central_High'][3],\
                                                    init_pop['North_Central_High'][4]

    # North Central Low Risk
    S_L_NCen, E_L_NCen, I_L_NCen, A_L_NCen, R_L_NCen =  init_pop['North_Central_Low'][0],\
                                                        init_pop['North_Central_Low'][1],\
                                                        init_pop['North_Central_Low'][2],\
                                                        init_pop['North_Central_Low'][3],\
                                                        init_pop['North_Central_Low'][4]

    # North Coastal High Risk
    S_H_NCoa, E_H_NCoa, I_H_NCoa, A_H_NCoa, R_H_NCoa =  init_pop['North_Coastal_High'][0],\
                                                        init_pop['North_Coastal_High'][1],\
                                                        init_pop['North_Coastal_High'][2],\
                                                        init_pop['North_Coastal_High'][3],\
                                                        init_pop['North_Coastal_High'][4]

    # North Coastal Low Risk
    S_L_NCoa, E_L_NCoa, I_L_NCoa, A_L_NCoa, R_L_NCoa =  init_pop['North_Coastal_Low'][0],\
                                                        init_pop['North_Coastal_Low'][1],\
                                                        init_pop['North_Coastal_Low'][2],\
                                                        init_pop['North_Coastal_Low'][3],\
                                                        init_pop['North_Coastal_Low'][4]

    # North Inland High Risk
    S_H_NI, E_H_NI, I_H_NI, A_H_NI, R_H_NI =  init_pop['North_Inland_High'][0],\
                                            init_pop['North_Inland_High'][1],\
                                            init_pop['North_Inland_High'][2],\
                                            init_pop['North_Inland_High'][3],\
                                            init_pop['North_Inland_High'][4]

    # North Inland Low Risk
    S_L_NI, E_L_NI, I_L_NI, A_L_NI, R_L_NI =  init_pop['North_Inland_Low'][0],\
                                            init_pop['North_Inland_Low'][1],\
                                            init_pop['North_Inland_Low'][2],\
                                            init_pop['North_Inland_Low'][3],\
                                            init_pop['North_Inland_Low'][4]

    # South High Risk
    S_H_S, E_H_S, I_H_S, A_H_S, R_H_S = init_pop['South_High'][0],\
                                        init_pop['South_High'][1],\
                                        init_pop['South_High'][2],\
                                        init_pop['South_High'][3],\
                                        init_pop['South_High'][4]

    # South Low Risk
    S_L_S, E_L_S, I_L_S, A_L_S, R_L_S = init_pop['South_Low'][0],\
                                        init_pop['South_Low'][1],\
                                        init_pop['South_Low'][2],\
                                        init_pop['South_Low'][3],\
                                        init_pop['South_Low'][4]

    ########### Extract Corresponding Parameters ###########

    b_IH_H, b_AH_H, g_H, t_H, l_H, d_H = param_dict['High'][0],\
                                        param_dict['High'][1],\
                                        param_dict['High'][2],\
                                        param_dict['High'][3],\
                                        param_dict['High'][4],\
                                        param_dict['High'][5]

    b_IH_L, b_AH_L, g_L, t_L, l_L, d_L = param_dict['Low'][0],\
                                        param_dict['Low'][1],\
                                        param_dict['Low'][2],\
                                        param_dict['Low'][3],\
                                        param_dict['Low'][4],\
                                        param_dict['Low'][5]

    ########### Extract Corresponding Mobility Rate ###########

    m_in_C, m_in_E, m_in_NCen, m_in_NCoa, m_in_NI, m_in_S = mobility_rate['Central'][0], \
                                                            mobility_rate['East'][0], \
                                                            mobility_rate['North_Central'][0], \
                                                            mobility_rate['North_Coastal'][0], \
                                                            mobility_rate['North_Inland'][0], \
                                                            mobility_rate['South'][0]

    m_out_C, m_out_E, m_out_NCen, m_out_NCoa, m_out_NI, m_out_S = mobility_rate['Central'][1], \
                                                                mobility_rate['East'][1], \
                                                                mobility_rate['North_Central'][1], \
                                                                mobility_rate['North_Coastal'][1], \
                                                                mobility_rate['North_Inland'][1], \
                                                                mobility_rate['South'][1]


    # time period
    t = np.linspace(0, 365, 365)

    # combine initial condition
    y0 = S_H_C, S_L_C, E_H_C, E_L_C, I_H_C, I_L_C, A_H_C, A_L_C, R_H_C, R_L_C, \
        S_H_E, S_L_E, E_H_E, E_L_E, I_H_E, I_L_E, A_H_E, A_L_E, R_H_E, R_L_E, \
        S_H_NCen, S_L_NCen, E_H_NCen, E_L_NCen, I_H_NCen, I_L_NCen, A_H_NCen, A_L_NCen, R_H_NCen, R_L_NCen, \
        S_H_NCoa, S_L_NCoa, E_H_NCoa, E_L_NCoa, I_H_NCoa, I_L_NCoa, A_H_NCoa, A_L_NCoa, R_H_NCoa, R_L_NCoa, \
        S_H_NI, S_L_NI, E_H_NI, E_L_NI, I_H_NI, I_L_NI, A_H_NI, A_L_NI, R_H_NI, R_L_NI, \
        S_H_S, S_L_S, E_H_S, E_L_S, I_H_S, I_L_S, A_H_S, A_L_S, R_H_S, R_L_S

    # solve ODE
    ret = odeint(deriv_High_and_Low, y0, t, args=(b_IH_H, b_AH_H, g_H, t_H, l_H, d_H, \
                                                b_IH_L, b_AH_L, g_L, t_L, l_L, d_L, \
                                                m_in_C, m_in_E, m_in_NCen, m_in_NCoa, m_in_NI, m_in_S, \
                                                m_out_C, m_out_E, m_out_NCen, m_out_NCoa, m_out_NI, m_out_S))
                                                
    # extract the results by transposing the ret array
    S_H_C, S_L_C, E_H_C, E_L_C, I_H_C, I_L_C, A_H_C, A_L_C, R_H_C, R_L_C, \
    S_H_E, S_L_E, E_H_E, E_L_E, I_H_E, I_L_E, A_H_E, A_L_E, R_H_E, R_L_E, \
    S_H_NCen, S_L_NCen, E_H_NCen, E_L_NCen, I_H_NCen, I_L_NCen, A_H_NCen, A_L_NCen, R_H_NCen, R_L_NCen, \
    S_H_NCoa, S_L_NCoa, E_H_NCoa, E_L_NCoa, I_H_NCoa, I_L_NCoa, A_H_NCoa, A_L_NCoa, R_H_NCoa, R_L_NCoa, \
    S_H_NI, S_L_NI, E_H_NI, E_L_NI, I_H_NI, I_L_NI, A_H_NI, A_L_NI, R_H_NI, R_L_NI, \
    S_H_S, S_L_S, E_H_S, E_L_S, I_H_S, I_L_S, A_H_S, A_L_S, R_H_S, R_L_S = ret.T

    plot_model([I_H_C, I_H_E, I_H_NCen, I_H_NCoa, I_H_NI, I_H_S], t)


if __name__ == '__main__':
    main()