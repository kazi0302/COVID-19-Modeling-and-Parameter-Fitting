'''
Define the SEIAR model of Six Regions in San Diego County : Central, East, North Central, North Coastal, North Inland, South
'''

# define ODE system
def deriv_High_and_Low(y, t, b_IH_H, b_AH_H, g_H, t_H, l_H, d_H, \
                             b_IH_L, b_AH_L, g_L, t_L, l_L, d_L, \
                             m_in_C, m_in_E, m_in_NCen, m_in_NCoa, m_in_NI, m_in_S, \
                             m_out_C, m_out_E, m_out_NCen, m_out_NCoa, m_out_NI, m_out_S):

    # Initial condition
    S_H_C, S_L_C, E_H_C, E_L_C, I_H_C, I_L_C, A_H_C, A_L_C, R_H_C, R_L_C, \
    S_H_E, S_L_E, E_H_E, E_L_E, I_H_E, I_L_E, A_H_E, A_L_E, R_H_E, R_L_E, \
    S_H_NCen, S_L_NCen, E_H_NCen, E_L_NCen, I_H_NCen, I_L_NCen, A_H_NCen, A_L_NCen, R_H_NCen, R_L_NCen, \
    S_H_NCoa, S_L_NCoa, E_H_NCoa, E_L_NCoa, I_H_NCoa, I_L_NCoa, A_H_NCoa, A_L_NCoa, R_H_NCoa, R_L_NCoa, \
    S_H_NI, S_L_NI, E_H_NI, E_L_NI, I_H_NI, I_L_NI, A_H_NI, A_L_NI, R_H_NI, R_L_NI, \
    S_H_S, S_L_S, E_H_S, E_L_S, I_H_S, I_L_S, A_H_S, A_L_S, R_H_S, R_L_S = y    

######## Define Functions (60 ODEs) ########

    # Central High Risk
    dS_H_C = -1 * S_H_C * (b_IH_H * (I_H_C + I_L_C) + b_AH_H * (A_H_C + A_L_C)) + (m_in_E * S_H_E + m_in_NCen * S_H_NCen + m_in_NCoa * S_H_NCoa + m_in_NI * S_H_NI + m_in_S * S_H_S) \
        - m_out_C * S_H_C
    dE_H_C = (S_H_C * (b_IH_H * (I_H_C + I_L_C) + b_AH_L * (A_H_C + A_L_C))) - g_H * E_H_C + (m_in_E * E_H_E + m_in_NCen * E_H_NCen + m_in_NCoa * E_H_NCoa + m_in_NI * E_H_NI + m_in_S * E_H_S) \
        - m_out_C * E_H_C
    dI_H_C = (t_H * g_H * E_H_C) - (l_H * I_H_C) - (d_H * I_H_C) + (m_in_E * I_H_E + m_in_NCen * I_H_NCen + m_in_NCoa * I_H_NCoa + m_in_NI * I_H_NI + m_in_S * I_H_S) \
        - m_out_C * I_H_C
    dA_H_C = ((1 - t_H) * g_H * E_H_C) - (l_H * A_H_C) + (m_in_E * A_H_E + m_in_NCen * A_H_NCen + m_in_NCoa * A_H_NCoa + m_in_NI * A_H_NI + m_in_S * A_H_S) \
        - m_out_C * A_H_C
    dR_H_C = l_H * ((I_H_C + A_H_C) + (m_in_E * R_H_E + m_in_NCen * R_H_NCen + m_in_NCoa * R_H_NCoa + m_in_NI * R_H_NI + m_in_S * R_H_S) \
        - m_out_C * R_H_C)

    # Central Low Risk
    dS_L_C = -1 * S_L_C * (b_IH_L * (I_H_C + I_L_C) + b_AH_L * (A_H_C + A_L_C)) + (m_in_E * S_L_E + m_in_NCen * S_L_NCen + m_in_NCoa * S_L_NCoa + m_in_NI * S_L_NI + m_in_S * S_L_S) \
        - m_out_C * S_L_C
    dE_L_C = (S_L_C * (b_IH_L * (I_H_C + I_L_C) + b_AH_L * (A_H_C + A_L_C))) - g_L * E_L_C + (m_in_E * E_L_E + m_in_NCen * E_L_NCen + m_in_NCoa * E_L_NCoa + m_in_NI * E_L_NI + m_in_S * E_L_S) \
        - m_out_C * E_L_C
    dI_L_C = (t_L * g_L * E_L_C) - (l_L * I_L_C) - (d_L * I_L_C) + (m_in_E * I_L_E + m_in_NCen * I_L_NCen + m_in_NCoa * I_L_NCoa + m_in_NI * I_L_NI + m_in_S * I_L_S) \
        - m_out_C * I_L_C
    dA_L_C = ((1 - t_L) * g_L * E_L_C) - (l_L * A_L_C) + (m_in_E * A_L_E + m_in_NCen * A_L_NCen + m_in_NCoa * A_L_NCoa + m_in_NI * A_L_NI + m_in_S * A_L_S) \
        - m_out_C * A_L_C
    dR_L_C = l_L * (I_L_C + A_L_C) + (m_in_E * R_L_E + m_in_NCen * R_L_NCen + m_in_NCoa * R_L_NCoa + m_in_NI * R_L_NI + m_in_S * R_L_S) \
        - m_out_C * R_L_C

    # East High Risk
    dS_H_E = -1 * S_H_E * (b_IH_H * (I_H_E + I_L_E) + b_AH_H * (A_H_E + A_L_E)) + (m_in_C * S_H_C + m_in_NCen * S_H_NCen + m_in_NCoa * S_H_NCoa + m_in_NI * S_H_NI + m_in_S * S_H_S) \
        - m_out_E * S_H_E
    dE_H_E = (S_H_E * (b_IH_H * (I_H_E + I_L_E) + b_AH_H * (A_H_E + A_L_E))) - g_H * E_H_E + (m_in_C * E_H_C + m_in_NCen * E_H_NCen + m_in_NCoa * E_H_NCoa + m_in_NI * E_H_NI + m_in_S * E_H_S) \
        - m_out_E * E_H_E
    dI_H_E = (t_H * g_H * E_H_E) - (l_H * I_H_E) - (d_H * I_H_E) + (m_in_C * I_H_C + m_in_NCen * I_H_NCen + m_in_NCoa * I_H_NCoa + m_in_NI * I_H_NI + m_in_S * I_H_S) \
        - m_out_E * I_H_E
    dA_H_E = ((1 - t_H) * g_H * E_H_E) - (l_H * A_H_E) + (m_in_C * A_H_C + m_in_NCen * A_H_NCen + m_in_NCoa * A_H_NCoa + m_in_NI * A_H_NI + m_in_S * A_H_S) \
        - m_out_E * A_H_E
    dR_H_E = l_H * (I_H_E + A_H_E) + (m_in_C * R_H_C + m_in_NCen * R_H_NCen + m_in_NCoa * R_H_NCoa + m_in_NI * R_H_NI + m_in_S * R_H_S) \
        - m_out_E * R_H_E

    # East Low Risk
    dS_L_E = -1 * S_L_E * (b_IH_L * (I_H_E + I_L_E) + b_AH_L * (A_H_E + A_L_E)) + (m_in_C * S_L_C + m_in_NCen * S_L_NCen + m_in_NCoa * S_L_NCoa + m_in_NI * S_L_NI + m_in_S * S_L_S) \
        - m_out_E * S_L_E
    dE_L_E = (S_L_E * (b_IH_L * (I_H_E + I_L_E) + b_AH_L * (A_H_E + A_L_E))) - g_L * E_L_E + (m_in_C * E_L_C + m_in_NCen * E_L_NCen + m_in_NCoa * E_L_NCoa + m_in_NI * E_L_NI + m_in_S * E_L_S) \
        - m_out_E * E_L_E
    dI_L_E = (t_L * g_L * E_L_E) - (l_L * I_L_E) - (d_L * I_L_E) + (m_in_C * I_L_C + m_in_NCen * I_L_NCen + m_in_NCoa * I_L_NCoa + m_in_NI * I_L_NI + m_in_S * I_L_S) \
        - m_out_E * I_L_E
    dA_L_E = ((1 - t_L) * g_L * E_L_E) - (l_L * A_L_E) + (m_in_C * A_L_C + m_in_NCen * A_L_NCen + m_in_NCoa * A_L_NCoa + m_in_NI * A_L_NI + m_in_S * A_L_S) \
        - m_out_E * A_L_E
    dR_L_E = l_L * (I_L_E + A_L_E) + (m_in_C * R_L_C + m_in_NCen * R_L_NCen + m_in_NCoa * R_L_NCoa + m_in_NI * R_L_NI + m_in_S * R_L_S) \
        - m_out_E * R_L_E

    # North Central High Risk
    dS_H_NCen = -1 * S_H_NCen * (b_IH_H * (I_H_NCen + I_L_NCen) + b_AH_H * (A_H_NCen + A_L_NCen)) + (m_in_C * S_H_C + m_in_E * S_H_E + m_in_NCoa * S_H_NCoa + m_in_NI * S_H_NI + m_in_S * S_H_S) \
        - m_out_NCen * S_H_NCen
    dE_H_NCen = (S_H_NCen * (b_IH_H * (I_H_NCen + I_L_NCen) + b_AH_H * (A_H_NCen + A_L_NCen))) - g_H * E_H_NCen + (m_in_C * E_H_C + m_in_E * E_H_E + m_in_NCoa * E_H_NCoa + m_in_NI * E_H_NI + m_in_S * E_H_S) \
        - m_out_NCen * E_H_NCen
    dI_H_NCen = (t_H * g_H * E_H_NCen) - (l_H * I_H_NCen) - (d_H * I_H_NCen) + (m_in_C * I_H_C + m_in_E * I_H_E + m_in_NCoa * I_H_NCoa + m_in_NI * I_H_NI + m_in_S * I_H_S) \
        - m_out_NCen * I_H_NCen
    dA_H_NCen = ((1 - t_H) * g_H * E_H_NCen) - (l_H * A_H_NCen) + (m_in_C * A_H_C + m_in_E * A_H_E + m_in_NCoa * A_H_NCoa + m_in_NI * A_H_NI + m_in_S * A_H_S) \
        - m_out_NCen * A_H_NCen
    dR_H_NCen = l_H * (I_H_NCen + A_H_NCen) + (m_in_C * R_H_C + m_in_E * R_H_E + m_in_NCoa * R_H_NCoa + m_in_NI * R_H_NI + m_in_S * R_H_S) \
        - m_out_NCen * R_H_NCen

    # North Central Low Risk
    dS_L_NCen = -1 * S_L_NCen * (b_IH_L * (I_H_NCen + I_L_NCen) + b_AH_L * (A_H_NCen + A_L_NCen)) + (m_in_C * S_L_C + m_in_E * S_L_E + m_in_NCoa * S_L_NCoa + m_in_NI * S_L_NI + m_in_S * S_L_S) \
        - m_out_NCen * S_L_NCen
    dE_L_NCen = (S_L_NCen * (b_IH_L * (I_H_NCen + I_L_NCen) + b_AH_L * (A_H_NCen + A_L_NCen))) - g_L * E_L_NCen + (m_in_C * E_L_C + m_in_E * E_L_E + m_in_NCoa * E_L_NCoa + m_in_NI * E_L_NI + m_in_S * E_L_S) \
        - m_out_NCen * E_L_NCen
    dI_L_NCen = (t_L * g_L * E_L_NCen) - (l_L * I_L_NCen) - (d_L * I_L_NCen) + (m_in_C * I_L_C + m_in_E * I_L_E + m_in_NCoa * I_L_NCoa + m_in_NI * I_L_NI + m_in_S * I_L_S) \
        - m_out_NCen * I_L_NCen
    dA_L_NCen = ((1 - t_L) * g_L * E_L_NCen) - (l_L * A_L_NCen) + (m_in_C * A_L_C + m_in_E * A_L_E + m_in_NCoa * A_L_NCoa + m_in_NI * A_L_NI + m_in_S * A_L_S) \
        - m_out_NCen * A_L_NCen
    dR_L_NCen = l_L * (I_L_NCen + A_L_NCen) + (m_in_C * R_L_C + m_in_E * R_L_E + m_in_NCoa * R_L_NCoa + m_in_NI * R_L_NI + m_in_S * R_L_S) \
        - m_out_NCen * R_L_NCen

    # North Coastal High Risk
    dS_H_NCoa = -1 * S_H_NCoa * (b_IH_H * (I_H_NCoa + I_L_NCoa) + b_AH_H * (A_H_NCoa + A_L_NCoa)) + (m_in_C * S_H_C + m_in_E * S_H_E + m_in_NCen * S_H_NCen + m_in_NI * S_H_NI + m_in_S * S_H_S) \
        - m_out_NCoa * S_H_NCoa
    dE_H_NCoa = (S_H_NCoa * (b_IH_H * (I_H_NCoa + I_L_NCoa) + b_AH_H * (A_H_NCoa + A_L_NCoa))) - g_H * E_H_NCoa + (m_in_C * E_H_C + m_in_E * E_H_E + m_in_NCen * E_H_NCen + m_in_NI * E_H_NI + m_in_S * E_H_S) \
        - m_out_NCoa * E_H_NCoa
    dI_H_NCoa = (t_H * g_H * E_H_NCoa) - (l_H * I_H_NCoa) - (d_H * I_H_NCoa) + (m_in_C * I_H_C + m_in_E * I_H_E + m_in_NCen * I_H_NCen + m_in_NI * I_H_NI + m_in_S * I_H_S) \
        - m_out_NCoa * I_H_NCoa
    dA_H_NCoa = ((1 - t_H) * g_H * E_H_NCoa) - (l_H * A_H_NCoa) + (m_in_C * A_H_C + m_in_E * A_H_E + m_in_NCen * A_H_NCen + m_in_NI * A_H_NI + m_in_S * A_H_S) \
        - m_out_NCoa * A_H_NCoa
    dR_H_NCoa = l_H * (I_H_NCoa + A_H_NCoa) + (m_in_C * R_H_C + m_in_E * R_H_E + m_in_NCen * R_H_NCen + m_in_NI * R_H_NI + m_in_S * R_H_S) \
        - m_out_NCoa * R_H_NCoa

    # North Coastal Low Risk
    dS_L_NCoa = -1 * S_L_NCoa * (b_IH_L * (I_H_NCoa + I_L_NCoa) + b_AH_L * (A_H_NCoa + A_L_NCoa)) + (m_in_C * S_L_C + m_in_E * S_L_E + m_in_NCen * S_L_NCen + m_in_NI * S_L_NI + m_in_S * S_L_S) \
        - m_out_NCoa * S_L_NCoa
    dE_L_NCoa = (S_L_NCoa * (b_IH_L * (I_H_NCoa + I_L_NCoa) + b_AH_L * (A_H_NCoa + A_L_NCoa))) - g_L * E_L_NCoa + (m_in_C * E_L_C + m_in_E * E_L_E + m_in_NCen * E_L_NCen + m_in_NI * E_L_NI + m_in_S * E_L_S) \
        - m_out_NCoa * E_L_NCoa
    dI_L_NCoa = (t_L * g_L * E_L_NCoa) - (l_L * I_L_NCoa) - (d_L * I_L_NCoa) + (m_in_C * I_L_C + m_in_E * I_L_E + m_in_NCen * I_L_NCen + m_in_NI * I_L_NI + m_in_S * I_L_S) \
        - m_out_NCoa * I_L_NCoa
    dA_L_NCoa = ((1 - t_L) * g_L * E_L_NCoa) - (l_L * A_L_NCoa) + (m_in_C * A_L_C + m_in_E * A_L_E + m_in_NCen * A_L_NCen + m_in_NI * A_L_NI + m_in_S * A_L_S) \
        - m_out_NCoa * A_L_NCoa
    dR_L_NCoa = l_L * (I_L_NCoa + A_L_NCoa) + (m_in_C * R_L_C + m_in_E * R_L_E + m_in_NCen * R_L_NCen + m_in_NI * R_L_NI + m_in_S * R_L_S) \
        - m_out_NCoa * R_L_NCoa

    # North Inland High Risk
    dS_H_NI = -1 * S_H_NI * (b_IH_H * (I_H_NI + I_L_NI) + b_AH_H * (A_H_NI + A_L_NI)) + (m_in_C * S_H_C + m_in_E * S_H_E + m_in_NCen * S_H_NCen + m_in_NCoa * S_H_NCoa + m_in_S * S_H_S) \
        - m_out_NI * S_H_NI
    dE_H_NI = (S_H_NI * (b_IH_H * (I_H_NI + I_L_NI) + b_AH_H * (A_H_NI + A_L_NI))) - g_H * E_H_NI + (m_in_C * E_H_C + m_in_E * E_H_E + m_in_NCen * E_H_NCen + m_in_NCoa * E_H_NCoa + m_in_S * E_H_S) \
        - m_out_NI * E_H_NI
    dI_H_NI = (t_H * g_H * E_H_NI) - (l_H * I_H_NI) - (d_H * I_H_NI) + (m_in_C * I_H_C + m_in_E * I_H_E + m_in_NCen * I_H_NCen + m_in_NCoa * I_H_NCoa + m_in_S * I_H_S) \
        - m_out_NI * I_H_NI
    dA_H_NI = ((1 - t_H) * g_H * E_H_NI) - (l_H * A_H_NI) + (m_in_C * A_H_C + m_in_E * A_H_E + m_in_NCen * A_H_NCen + m_in_NCoa * A_H_NCoa + m_in_S * A_H_S) \
        - m_out_NI * A_H_NI
    dR_H_NI = l_H * (I_H_NI + A_H_NI) + (m_in_C * R_H_C + m_in_E * R_H_E + m_in_NCen * R_H_NCen + m_in_NCoa * R_H_NCoa + m_in_S * R_H_S) \
        - m_out_NI * R_H_NI

    # North Inland Low Risk
    dS_L_NI = -1 * S_L_NI * (b_IH_L * (I_H_NI + I_L_NI) + b_AH_L * (A_H_NI + A_L_NI)) + (m_in_C * S_L_C + m_in_E * S_L_E + m_in_NCen * S_L_NCen + m_in_NCoa * S_L_NCoa + m_in_S * S_L_S) \
        - m_out_NI * S_L_NI
    dE_L_NI = (S_L_NI * (b_IH_L * (I_H_NI + I_L_NI) + b_AH_L * (A_H_NI + A_L_NI))) - g_L * E_L_NI + (m_in_C * E_L_C + m_in_E * E_L_E + m_in_NCen * E_L_NCen + m_in_NCoa * E_L_NCoa + m_in_S * E_L_S) \
        - m_out_NI * E_L_NI
    dI_L_NI = (t_L * g_L * E_L_NI) - (l_L * I_L_NI) - (d_L * I_L_NI) + (m_in_C * I_L_C + m_in_E * I_L_E + m_in_NCen * I_L_NCen + m_in_NCoa * I_L_NCoa + m_in_S * I_L_S) \
        - m_out_NI * I_L_NI
    dA_L_NI = ((1 - t_L) * g_L * E_L_NI) - (l_L * A_L_NI) + (m_in_C * A_L_C + m_in_E * A_L_E + m_in_NCen * A_L_NCen + m_in_NCoa * A_L_NCoa + m_in_S * A_L_S) \
        - m_out_NI * A_L_NI
    dR_L_NI = l_L * (I_L_NI + A_L_NI) + (m_in_C * R_L_C + m_in_E * R_L_E + m_in_NCen * R_L_NCen + m_in_NCoa * R_L_NCoa + m_in_S * R_L_S) \
        - m_out_NI * R_L_NI

    # South High Risk
    dS_H_S = -1 * S_H_S * (b_IH_H * (I_H_S + I_L_S) + b_AH_H * (A_H_S + A_L_S)) + (m_in_C * S_H_C + m_in_E * S_H_E + m_in_NCen * S_H_NCen + m_in_NCoa * S_H_NCoa + m_in_NI * S_H_NI) \
        - m_out_S * S_H_S
    dE_H_S = (S_H_S * (b_IH_H * (I_H_S + I_L_S) + b_AH_H * (A_H_S + A_L_S))) - g_H * E_H_S + (m_in_C * E_H_C + m_in_E * E_H_E + m_in_NCen * E_H_NCen + m_in_NCoa * E_H_NCoa + m_in_NI * E_H_NI) \
        - m_out_S * E_H_S
    dI_H_S = (t_H * g_H * E_H_S) - (l_H * I_H_S) - (d_H * I_H_S) + (m_in_C * I_H_C + m_in_E * I_H_E + m_in_NCen * I_H_NCen + m_in_NCoa * I_H_NCoa + m_in_NI * I_H_NI) \
        - m_out_S * I_H_S
    dA_H_S = ((1 - t_H) * g_H * E_H_S) - (l_H * A_H_S) + (m_in_C * A_H_C + m_in_E * A_H_E + m_in_NCen * A_H_NCen + m_in_NCoa * A_H_NCoa + m_in_NI * A_H_NI) \
        - m_out_S * A_H_S
    dR_H_S = l_H * (I_H_S + A_H_S) + (m_in_C * R_H_C + m_in_E * R_H_E + m_in_NCen * R_H_NCen + m_in_NCoa * R_H_NCoa + m_in_NI * R_H_NI) \
        - m_out_S * R_H_S

    # South Low Risk
    dS_L_S = -1 * S_L_S * (b_IH_L * (I_H_S + I_L_S) + b_AH_L * (A_H_S + A_L_S)) + (m_in_C * S_L_C + m_in_E * S_L_E + m_in_NCen * S_L_NCen + m_in_NCoa * S_L_NCoa + m_in_NI * S_L_NI) \
        - m_out_S * S_L_S
    dE_L_S = (S_L_S * (b_IH_L * (I_H_S + I_L_S) + b_AH_L * (A_H_S + A_L_S))) - g_L * E_L_S + (m_in_C * E_L_C + m_in_E * E_L_E + m_in_NCen * E_L_NCen + m_in_NCoa * E_L_NCoa + m_in_NI * E_L_NI) \
        - m_out_S * E_L_S
    dI_L_S = (t_L * g_L * E_L_S) - (l_L * I_L_S) - (d_L * I_L_S) + (m_in_C * I_L_C + m_in_E * I_L_E + m_in_NCen * I_L_NCen + m_in_NCoa * I_L_NCoa + m_in_NI * I_L_NI) \
        - m_out_S * I_L_S
    dA_L_S = ((1 - t_L) * g_L * E_L_S) - (l_L * A_L_S) + (m_in_C * A_L_C + m_in_E * A_L_E + m_in_NCen * A_L_NCen + m_in_NCoa * A_L_NCoa + m_in_NI * A_L_NI) \
        - m_out_S * A_L_S
    dR_L_S = l_L * (I_L_S + A_L_S) + (m_in_C * R_L_C + m_in_E * R_L_E + m_in_NCen * R_L_NCen + m_in_NCoa * R_L_NCoa + m_in_NI * R_L_NI) \
        - m_out_S * R_L_S

######## Define Functions (End) ########

    return  dS_H_C, dS_L_C, dE_H_C, dE_L_C, dI_H_C, dI_L_C, dA_H_C, dA_L_C, dR_H_C, dR_L_C, \
            dS_H_E, dS_L_E, dE_H_E, dE_L_E, dI_H_E, dI_L_E, dA_H_E, dA_L_E, dR_H_E, dR_L_E, \
            dS_H_NCen, dS_L_NCen, dE_H_NCen, dE_L_NCen, dI_H_NCen, dI_L_NCen, dA_H_NCen, dA_L_NCen, dR_H_NCen, dR_L_NCen, \
            dS_H_NCoa, dS_L_NCoa, dE_H_NCoa, dE_L_NCoa, dI_H_NCoa, dI_L_NCoa, dA_H_NCoa, dA_L_NCoa, dR_H_NCoa, dR_L_NCoa, \
            dS_H_NI, dS_L_NI, dE_H_NI, dE_L_NI, dI_H_NI, dI_L_NI, dA_H_NI, dA_L_NI, dR_H_NI, dR_L_NI, \
            dS_H_S, dS_L_S, dE_H_S, dE_L_S, dI_H_S, dI_L_S, dA_H_S, dA_L_S, dR_H_S, dR_L_S