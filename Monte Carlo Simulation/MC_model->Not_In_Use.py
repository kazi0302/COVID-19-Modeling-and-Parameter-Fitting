from numba import *
import math
import numpy as np

@jit(nopython=True, parallel=True)
def SEIAR_model(S_H_C, E_H_C, I_H_C, A_H_C, R_H_C, S_L_C, E_L_C, I_L_C, A_L_C, R_L_C,\
               S_H_E, E_H_E, I_H_E, A_H_E, R_H_E, S_L_E, E_L_E, I_L_E, A_L_E, R_L_E,\
               S_H_NCen, E_H_NCen, I_H_NCen, A_H_NCen, R_H_NCen, S_L_NCen, E_L_NCen, I_L_NCen, A_L_NCen, R_L_NCen,\
               S_H_NCoa, E_H_NCoa, I_H_NCoa, A_H_NCoa, R_H_NCoa, S_L_NCoa, E_L_NCoa, I_L_NCoa, A_L_NCoa, R_L_NCoa,\
               S_H_NI, E_H_NI, I_H_NI, A_H_NI, R_H_NI, S_L_NI, E_L_NI, I_L_NI, A_L_NI, R_L_NI,\
               S_H_S, E_H_S, I_H_S, A_H_S, R_H_S, S_L_S, E_L_S, I_L_S, A_L_S, R_L_S, \
               beta_sym, beta_asym, gamma, theta, lamb, delta):
   
    t = 214
    
    C_H = np.zeros((t+1, 6))
    C_L = np.zeros((t+1, 6))

    E_H = np.zeros((t+1, 6))
    E_L = np.zeros((t+1, 6))

    NCen_H = np.zeros((t+1, 6))
    NCen_L = np.zeros((t+1, 6))

    NCoa_H = np.zeros((t+1, 6))
    NCoa_L = np.zeros((t+1, 6))

    NI_H = np.zeros((t+1, 6))
    NI_L = np.zeros((t+1, 6))

    S_H = np.zeros((t+1, 6))
    S_L = np.zeros((t+1, 6))

    for day in range(1, t-1):

        mob_central_arr = np.array([

            # central to east
            [2518.007912, 1179.226414, 2302.018371], \
            # central to north central
            [5431.723516, 1940.995073, 4336.156366], \
            # central to north coastal
            [410.088803, 224.745859, 654.88667], \
            # central to north inland
            [664.540636, 366.97859, 588.297201], \
            # central to south
            [3640.640873, 1626.007515, 2917.389471]

        ])

        mob_east_arr = np.array([

            # east to central
            [3768.837284, 1440.608412, 2866.137454], \
            # east to north central
            [5425.003638, 1919.445416, 4670.294659], \
            # east to north coastal
            [483.573126, 191.392576, 473.906442], \
            # east to north inland
            [1055.682996, 484.299215, 842.685923], \
            # east to south
            [2488.03862, 980.052445, 2087.730478]

        ])

        mob_north_central_arr = np.array([

            # north central to central
            [3657.897354, 1054.592602, 3087.41847], \
            # north central to east
            [2111.999373, 871.361025, 2093.514868], \
            # north central to north coastal
            [1656.53273, 460.539775, 1724.08199], \
            # north central to north inland
            [2767.349274, 983.942653, 2341.094009], \
            # north central to south
            [1257.405478, 596.794501, 1182.745383]

        ])

        mob_north_coastal_arr = np.array([

            # north coastal to central
            [780.166594, 156.644603, 1196.23607], \
            # north coastal to east
            [246.915804, 115.929512, 226.261725], \
            # north coastal to north central
            [2547.496323, 631.995693, 1936.045114], \
            # north coastal to north inland
            [3898.342286, 1414.099677, 3053.434645], \
            # north coastal to south
            [258.826212, 133.157868, 306.770863]

        ])

        mob_north_inland_arr = np.array([

            # north inland to central
            [1507.037202, 392.630209, 1305.236506], \
            # north inland to east
            [1000.642647, 390.292827, 879.488701], \
            # north inland to north central
            [6136.233626, 1687.503238, 4745.989662], \
            # north inland to north coastal
            [5256.002028, 1827.728619, 4981.44469], \
            # north inland to south
            [607.446061, 255.851188, 534.272639]

        ])

        mob_south_arr = np.array([

            # south to central
            [4053.532554, 1545.91116, 2984.388535], \
            # south to east
            [1970.762799, 828.927306, 1716.573243], \
            # south to north central
            [3883.18689, 1374.748533, 3361.535465], \
            # south to north coastal
            [391.417326, 147.78653, 331.027812], \
            # south to north inland
            [622.229053, 262.674827, 511.746844]

        ])

        # Divide by population to get % change
        mob_central_arr = mob_central_arr / 511660
        mob_east_arr = mob_east_arr / 488330
        mob_north_central_arr = mob_north_central_arr / 646380
        mob_north_coastal_arr = mob_north_coastal_arr / 535219
        mob_north_inland_arr = mob_north_inland_arr / 602000
        mob_south_arr = mob_south_arr / 500076

        if day <= 70:
            mob_central = np.average(mob_central_arr, axis=0)[0]
            mob_east = np.average(mob_east_arr, axis=0)[0]
            mob_north_central = np.average(mob_north_central_arr, axis=0)[0]
            mob_north_coastal = np.average(mob_north_coastal_arr, axis=0)[0]
            mob_north_inland = np.average(mob_north_inland_arr, axis=0)[0]
            mob_south = np.average(mob_south_arr, axis=0)[0]
           
        elif day > 70 and day < 80:
            mob_central = np.average(mob_central_arr, axis=0)[1]
            mob_east = np.average(mob_east_arr, axis=0)[1]
            mob_north_central = np.average(mob_north_central_arr, axis=0)[1]
            mob_north_coastal = np.average(mob_north_coastal_arr, axis=0)[1]
            mob_north_inland = np.average(mob_north_inland_arr, axis=0)[1]
            mob_south = np.average(mob_south_arr, axis=0)[1]

        else:
            mob_central = np.average(mob_central_arr, axis=0)[2]
            mob_east = np.average(mob_east_arr, axis=0)[2]
            mob_north_central = np.average(mob_north_central_arr, axis=0)[2]
            mob_north_coastal = np.average(mob_north_coastal_arr, axis=0)[2]
            mob_north_inland = np.average(mob_north_inland_arr, axis=0)[2]
            mob_south = np.average(mob_south_arr, axis=0)[2]

        

        # Generate mean SEIAR
        Central_High = { 'S_to_E': np.random.binomial(S_H_C, (beta_sym * I_H_C) + (beta_asym * A_H_C)),
                         'E_to_I': np.random.binomial(E_H_C, gamma*theta),
                         'E_to_A': np.random.binomial(E_H_C, gamma*(1-theta)),
                         'I_to_R': np.random.binomial(I_H_C, delta + lamb),
                         'A_to_R': np.random.binomial(A_H_C, delta) }

        Central_Low = { 'S_to_E': np.random.binomial(S_L_C, (beta_sym * I_L_C) + (beta_asym * A_L_C)),
                        'E_to_I': np.random.binomial(E_L_C, gamma*theta),
                        'E_to_A': np.random.binomial(E_L_C, gamma*(1-theta)),
                        'I_to_R': np.random.binomial(I_L_C, delta + lamb),
                        'A_to_R': np.random.binomial(A_L_C, delta) }

        East_High = { 'S_to_E': np.random.binomial(S_H_E, (beta_sym * I_H_E) + (beta_asym * A_H_E)),
                      'E_to_I': np.random.binomial(E_H_E, gamma*theta),
                      'E_to_A': np.random.binomial(E_H_E, gamma*(1-theta)),
                      'I_to_R': np.random.binomial(I_H_E, delta + lamb),
                      'A_to_R': np.random.binomial(A_H_E, delta) }

        East_Low = {    'S_to_E': np.random.binomial(S_L_E, (beta_sym * I_L_E) + (beta_asym * A_L_E)),
                        'E_to_I': np.random.binomial(E_L_E, gamma*theta),
                        'E_to_A': np.random.binomial(E_L_E, gamma*(1-theta)),
                        'I_to_R': np.random.binomial(I_L_E, delta + lamb),
                        'A_to_R': np.random.binomial(A_L_E, delta) }
        
        North_Central_High = {  'S_to_E': np.random.binomial(S_H_NCen, (beta_sym * I_H_NCen) + (beta_asym * A_H_NCen)),
                                'E_to_I': np.random.binomial(E_H_NCen, gamma*theta),
                                'E_to_A': np.random.binomial(E_H_NCen, gamma*(1-theta)),
                                'I_to_R': np.random.binomial(I_H_NCen, delta + lamb),
                                'A_to_R': np.random.binomial(A_H_NCen, delta) }
        
        North_Central_Low = {   'S_to_E': np.random.binomial(S_L_NCen, (beta_sym * I_L_NCen) + (beta_asym * A_L_NCen)),
                                'E_to_I': np.random.binomial(E_L_NCen, gamma*theta),
                                'E_to_A': np.random.binomial(E_L_NCen, gamma*(1-theta)),
                                'I_to_R': np.random.binomial(I_L_NCen, delta + lamb),
                                'A_to_R': np.random.binomial(A_L_NCen, delta) }

        North_Coastal_High = {  'S_to_E': np.random.binomial(S_H_NCoa, (beta_sym * I_H_NCoa) + (beta_asym * A_H_NCoa)),
                                'E_to_I': np.random.binomial(E_H_NCoa, gamma*theta),
                                'E_to_A': np.random.binomial(E_H_NCoa, gamma*(1-theta)),
                                'I_to_R': np.random.binomial(I_H_NCoa, delta + lamb),
                                'A_to_R': np.random.binomial(A_H_NCoa, delta) }
        
        North_Coastal_Low = {   'S_to_E': np.random.binomial(S_L_NCoa, (beta_sym * I_L_NCoa) + (beta_asym * A_L_NCoa)),
                                'E_to_I': np.random.binomial(E_L_NCoa, gamma*theta),
                                'E_to_A': np.random.binomial(E_L_NCoa, gamma*(1-theta)),
                                'I_to_R': np.random.binomial(I_L_NCoa, delta + lamb),
                                'A_to_R': np.random.binomial(A_L_NCoa, delta) }
        
        North_Inland_High = {   'S_to_E': np.random.binomial(S_H_NI, (beta_sym * I_H_NI) + (beta_asym * A_H_NI)),
                                'E_to_I': np.random.binomial(E_H_NI, gamma*theta),
                                'E_to_A': np.random.binomial(E_H_NI, gamma*(1-theta)),
                                'I_to_R': np.random.binomial(I_H_NI, delta + lamb),
                                'A_to_R': np.random.binomial(A_H_NI, delta) }

        North_Inland_Low = {    'S_to_E': np.random.binomial(S_L_NI, (beta_sym * I_L_NI) + (beta_asym * A_L_NI)),
                                'E_to_I': np.random.binomial(E_L_NI, gamma*theta),
                                'E_to_A': np.random.binomial(E_L_NI, gamma*(1-theta)),
                                'I_to_R': np.random.binomial(I_L_NI, delta + lamb),
                                'A_to_R': np.random.binomial(A_L_NI, delta) }
        
        South_High = {  'S_to_E': np.random.binomial(S_H_S, (beta_sym * I_H_S) + (beta_asym * A_H_S)),
                        'E_to_I': np.random.binomial(E_H_S, gamma*theta),
                        'E_to_A': np.random.binomial(E_H_S, gamma*(1-theta)),
                        'I_to_R': np.random.binomial(I_H_S, delta + lamb),
                        'A_to_R': np.random.binomial(A_H_S, delta) }

        South_Low = {   'S_to_E': np.random.binomial(S_L_S, (beta_sym * I_L_S) + (beta_asym * A_L_S)),
                        'E_to_I': np.random.binomial(E_L_S, gamma*theta),
                        'E_to_A': np.random.binomial(E_L_S, gamma*(1-theta)),
                        'I_to_R': np.random.binomial(I_L_S, delta + lamb),
                        'A_to_R': np.random.binomial(A_L_S, delta) }
                                
        # Compute new SEIAR
        Central_High_S_new = max(S_H_C - Central_High['S_to_E'] + \
            mob_central*(S_H_E + S_H_NCen + S_H_NCoa + S_H_NI + S_H_S) - 5 * mob_central * S_H_C, 0)
        Central_High_E_new = max(E_H_C + Central_High['S_to_E'] - Central_High['E_to_I'] - Central_High['E_to_A'] + \
            mob_central*(E_H_E + E_H_NCen + E_H_NCoa + E_H_NI + E_H_S) - 5 * mob_central * E_H_C, 0)
        Central_High_I_new = max(I_H_C + Central_High['E_to_I'] - Central_High['I_to_R'] + \
            mob_central*(I_H_E + I_H_NCen + I_H_NCoa + I_H_NI + I_H_S) - 5 * mob_central * I_H_C, 0)
        Central_High_A_new = max(A_H_C + Central_High['E_to_A'] - Central_High['A_to_R'] + \
            mob_central*(A_H_E + A_H_NCen + A_H_NCoa + A_H_NI + A_H_S) - 5 * mob_central * A_H_C, 0)
        Central_High_R_new = max(R_H_C + Central_High['I_to_R'] + Central_High['A_to_R'] + \
            mob_central*(R_H_E + R_H_NCen + R_H_NCoa + R_H_NI + R_H_S) - 5 * mob_central * R_H_C, 0)

        Central_Low_S_new = max(S_L_C - Central_Low['S_to_E'] + \
            mob_central*(S_L_E + S_L_NCen + S_L_NCoa + S_L_NI + S_H_S) - 5 * mob_central * S_L_C, 0)
        Central_Low_E_new = max(E_L_C + Central_Low['S_to_E'] - Central_Low['E_to_I'] - Central_Low['E_to_A'] + \
            mob_central*(E_L_E + E_L_NCen + E_L_NCoa + E_L_NI + E_L_S) - 5 * mob_central * E_L_C, 0)
        Central_Low_I_new = max(I_L_C + Central_Low['E_to_I'] - Central_Low['I_to_R'] + \
            mob_central*(I_L_E + I_L_NCen + I_L_NCoa + I_L_NI + I_L_S) - 5 * mob_central * I_L_C, 0)
        Central_Low_A_new = max(A_L_C + Central_Low['E_to_A'] - Central_Low['A_to_R'] + \
            mob_central*(A_L_E + A_L_NCen + A_L_NCoa + A_L_NI + A_L_S) - 5 * mob_central * A_L_C, 0)
        Central_Low_R_new = max(R_L_C + Central_Low['I_to_R'] + Central_Low['A_to_R'] + \
            mob_central*(R_L_E + R_L_NCen + R_L_NCoa + R_L_NI + R_L_S) - 5 * mob_central * R_L_C, 0)
        
        East_High_S_new = max(S_H_E - East_High['S_to_E'] + \
            mob_east*(S_H_C + S_H_NCen + S_H_NCoa + S_H_NI + S_H_S) - 5 * mob_east * S_H_E, 0)
        East_High_E_new = max(E_H_E + East_High['S_to_E'] - East_High['E_to_I'] - East_High['E_to_A'] + \
            mob_east*(E_H_C + E_H_NCen + E_H_NCoa + E_H_NI + E_H_S) - 5 * mob_east * E_H_E, 0)
        East_High_I_new = max(I_H_E + East_High['E_to_I'] - East_High['I_to_R'] + \
            mob_east*(I_H_C + I_H_NCen + I_H_NCoa + I_H_NI + I_H_S) - 5 * mob_east * I_H_E, 0)
        East_High_A_new = max(A_H_E + East_High['E_to_A'] - East_High['A_to_R'] + \
            mob_east*(A_H_C + A_H_NCen + A_H_NCoa + A_H_NI + A_H_S) - 5 * mob_east * A_H_E, 0)
        East_High_R_new = max(R_H_E + East_High['I_to_R'] + East_High['A_to_R'] + \
            mob_east*(R_H_C + R_H_NCen + R_H_NCoa + R_H_NI + R_H_S) - 5 * mob_east * R_H_E, 0)

        East_Low_S_new = max(S_L_E - East_Low['S_to_E'] + \
            mob_east*(S_L_C + S_L_NCen + S_L_NCoa + S_L_NI + S_L_S) - 5 * mob_east * S_L_E, 0)
        East_Low_E_new = max(E_L_E + East_Low['S_to_E'] - East_Low['E_to_I'] - East_Low['E_to_A'] + \
            mob_east*(E_L_C + E_L_NCen + E_L_NCoa + E_L_NI + E_L_S) - 5 * mob_east * E_L_E, 0)
        East_Low_I_new = max(I_L_E + East_Low['E_to_I'] - East_Low['I_to_R'] + \
            mob_east*(I_L_C + I_L_NCen + I_L_NCoa + I_L_NI + I_L_S) - 5 * mob_east * I_L_E, 0)
        East_Low_A_new = max(A_L_E + East_Low['E_to_A'] - East_Low['A_to_R'] + \
            mob_east*(A_L_C + A_L_NCen + A_L_NCoa + A_L_NI + A_L_S) - 5 * mob_east * A_L_E, 0)
        East_Low_R_new = max(R_L_E + East_Low['I_to_R'] + East_Low['A_to_R'] + \
            mob_east*(R_L_C + R_L_NCen + R_L_NCoa + R_L_NI + R_L_S) - 5 * mob_east * R_L_E, 0)
        
        North_Central_High_S_new = max(S_H_NCen - North_Central_High['S_to_E'] + \
            mob_north_central*(S_H_C + S_H_E + S_H_NCoa + S_H_NI + S_H_S) - 5 * mob_north_central * S_H_NCen, 0)
        North_Central_High_E_new = max(E_H_NCen + North_Central_High['S_to_E'] - North_Central_High['E_to_I'] - North_Central_High['E_to_A'] + \
            mob_north_central*(E_H_C + E_H_E + E_H_NCoa + E_H_NI + E_H_S) - 5 * mob_north_central * E_H_NCen, 0)
        North_Central_High_I_new = max(I_H_NCen + North_Central_High['E_to_I'] - North_Central_High['I_to_R'] + \
            mob_north_central*(I_H_C + I_H_E + I_H_NCoa + I_H_NI + I_H_S) - 5 * mob_north_central * I_H_NCen, 0)
        North_Central_High_A_new = max(A_H_NCen + North_Central_High['E_to_A'] - North_Central_High['A_to_R'] + \
            mob_north_central*(A_H_C + A_H_E + A_H_NCoa + A_H_NI + A_H_S) - 5 * mob_north_central * A_H_NCen, 0)
        North_Central_High_R_new = max(R_H_NCen + North_Central_High['I_to_R'] + North_Central_High['A_to_R'] + \
            mob_north_central*(R_H_C + R_H_E + R_H_NCoa + R_H_NI + R_H_S) - 5 * mob_north_central * R_H_NCen, 0)

        North_Central_Low_S_new = max(S_L_NCen - North_Central_Low['S_to_E'] + \
            mob_north_central*(S_L_C + S_L_E + S_L_NCoa + S_L_NI + S_L_S) - 5 * mob_north_central * S_L_NCen, 0)
        North_Central_Low_E_new = max(E_L_NCen + North_Central_Low['S_to_E'] - North_Central_Low['E_to_I'] - North_Central_Low['E_to_A'] + \
            mob_north_central*(E_L_C + E_L_E + E_L_NCoa + E_L_NI + E_L_S) - 5 * mob_north_central * E_L_NCen, 0)
        North_Central_Low_I_new = max(I_L_NCen + North_Central_Low['E_to_I'] - North_Central_Low['I_to_R'] + \
            mob_north_central*(I_L_C + I_L_E + I_L_NCoa + I_L_NI + I_L_S) - 5 * mob_north_central * I_L_NCen, 0)
        North_Central_Low_A_new = max(A_L_NCen + North_Central_Low['E_to_A'] - North_Central_Low['A_to_R'] + \
            mob_north_central*(A_L_C + A_L_E + A_L_NCoa + A_L_NI + A_L_S) - 5 * mob_north_central * A_L_NCen, 0)
        North_Central_Low_R_new = max(R_L_NCen + North_Central_Low['I_to_R'] + North_Central_Low['A_to_R'] + \
            mob_north_central*(R_L_C + R_L_E + R_L_NCoa + R_L_NI + R_L_S) - 5 * mob_north_central * R_L_NCen, 0)

        North_Coastal_High_S_new = max(S_H_NCoa - North_Coastal_High['S_to_E'] + \
            mob_north_coastal*(S_H_C + S_H_E + S_H_NCen + S_H_NI + S_H_S) - 5 * mob_north_coastal * S_H_NCoa, 0)
        North_Coastal_High_E_new = max(E_H_NCoa + North_Coastal_High['S_to_E'] - North_Coastal_High['E_to_I'] - North_Coastal_High['E_to_A'] + \
            mob_north_coastal*(E_H_C + E_H_E + E_H_NCen + E_H_NI + E_H_S) - 5 * mob_north_coastal * E_H_NCoa, 0)
        North_Coastal_High_I_new = max(I_H_NCoa + North_Coastal_High['E_to_I'] - North_Coastal_High['I_to_R'] + \
            mob_north_coastal*(I_H_C + I_H_E + I_H_NCen + I_H_NI + I_H_S) - 5 * mob_north_coastal * I_H_NCoa, 0)
        North_Coastal_High_A_new = max(A_H_NCoa + North_Coastal_High['E_to_A'] - North_Coastal_High['A_to_R'] + \
            mob_north_coastal*(A_H_C + A_H_E + A_H_NCen + A_H_NI + A_H_S) - 5 * mob_north_coastal * A_H_NCoa, 0)
        North_Coastal_High_R_new = max(R_H_NCoa + North_Coastal_High['I_to_R'] + North_Coastal_High['A_to_R'] + \
            mob_north_coastal*(R_H_C + R_H_E + R_H_NCen + R_H_NI + R_H_S) - 5 * mob_north_coastal * R_H_NCoa, 0)

        North_Coastal_Low_S_new = max(S_L_NCoa - North_Coastal_Low['S_to_E'] + \
            mob_north_coastal*(S_L_C + S_L_E + S_L_NCen + S_L_NI + S_L_S) - 5 * mob_north_coastal * S_L_NCoa, 0)
        North_Coastal_Low_E_new = max(E_L_NCoa + North_Coastal_Low['S_to_E'] - North_Coastal_Low['E_to_I'] - North_Coastal_Low['E_to_A'] + \
            mob_north_coastal*(E_L_C + E_L_E + E_L_NCen + E_L_NI + E_L_S) - 5 * mob_north_coastal * E_L_NCoa, 0)
        North_Coastal_Low_I_new = max(I_L_NCoa + North_Coastal_Low['E_to_I'] - North_Coastal_Low['I_to_R'] + \
            mob_north_coastal*(I_L_C + I_L_E + I_L_NCen + I_L_NI + I_L_S) - 5 * mob_north_coastal * I_L_NCoa, 0)
        North_Coastal_Low_A_new = max(A_L_NCoa + North_Coastal_Low['E_to_A'] - North_Coastal_Low['A_to_R'] + \
            mob_north_coastal*(A_L_C + A_L_E + A_L_NCen + A_L_NI + A_L_S) - 5 * mob_north_coastal * A_L_NCoa, 0)
        North_Coastal_Low_R_new = max(R_L_NCoa + North_Coastal_Low['I_to_R'] + North_Coastal_Low['A_to_R'] + \
            mob_north_coastal*(R_L_C + R_L_E + R_L_NCen + R_L_NI + R_L_S) - 5 * mob_north_coastal * R_L_NCoa, 0)

        North_Inland_High_S_new = max(S_H_NI - North_Inland_High['S_to_E'] + \
            mob_north_inland*(S_H_C + S_H_E + S_H_NCen + S_H_NCoa + S_H_S) - 5 * mob_north_inland * S_H_NI, 0)
        North_Inland_High_E_new = max(E_H_NI + North_Inland_High['S_to_E'] - North_Inland_High['E_to_I'] - North_Inland_High['E_to_A'] + \
            mob_north_inland*(E_H_C + E_H_E + E_H_NCen + E_H_NCoa + E_H_S) - 5 * mob_north_inland * E_H_NI, 0)
        North_Inland_High_I_new = max(I_H_NI + North_Inland_High['E_to_I'] - North_Inland_High['I_to_R'] + \
            mob_north_inland*(I_H_C + I_H_E + I_H_NCen + I_H_NCoa + I_H_S) - 5 * mob_north_inland * I_H_NI, 0)
        North_Inland_High_A_new = max(A_H_NI + North_Inland_High['E_to_A'] - North_Inland_High['A_to_R'] + \
            mob_north_inland*(A_H_C + A_H_E + A_H_NCen + A_H_NCoa + A_H_S) - 5 * mob_north_inland * A_H_NI, 0)
        North_Inland_High_R_new = max(R_H_NI + North_Inland_High['I_to_R'] + North_Inland_High['A_to_R'] + \
            mob_north_inland*(R_H_C + R_H_E + R_H_NCen + R_H_NCoa + R_H_S) - 5 * mob_north_inland * R_H_NI, 0)

        North_Inland_Low_S_new = max(S_L_NI - North_Inland_Low['S_to_E'] + \
            mob_north_inland*(S_L_C + S_L_E + S_L_NCen + S_L_NCoa + S_L_S) - 5 * mob_north_inland * S_L_NI, 0)
        North_Inland_Low_E_new = max(E_L_NI + North_Inland_Low['S_to_E'] - North_Inland_Low['E_to_I'] - North_Inland_Low['E_to_A'] + \
            mob_north_inland*(E_L_C + E_L_E + E_L_NCen + E_L_NCoa + E_L_S) - 5 * mob_north_inland * E_L_NI, 0)
        North_Inland_Low_I_new = max(I_L_NI + North_Inland_Low['E_to_I'] - North_Inland_Low['I_to_R'] + \
            mob_north_inland*(I_L_C + I_L_E + I_L_NCen + I_L_NCoa + I_L_S) - 5 * mob_north_inland * I_L_NI, 0)
        North_Inland_Low_A_new = max(A_L_NI + North_Inland_Low['E_to_A'] - North_Inland_Low['A_to_R'] + \
            mob_north_inland*(A_L_C + A_L_E + A_L_NCen + A_L_NCoa + A_L_S) - 5 * mob_north_inland * A_L_NI, 0)
        North_Inland_Low_R_new = max(R_L_NI + North_Inland_Low['I_to_R'] + North_Inland_Low['A_to_R'] + \
            mob_north_inland*(R_L_C + R_L_E + R_L_NCen + R_L_NCoa + R_L_S) - 5 * mob_north_inland * R_L_NI, 0)

        South_High_S_new = max(S_H_S - South_High['S_to_E'] + \
            mob_south*(S_H_C + S_H_E + S_H_NCen + S_H_NCoa + S_H_NI) - 5 * mob_south * S_H_S, 0)
        South_High_E_new = max(E_H_S + South_High['S_to_E'] - South_High['E_to_I'] - South_High['E_to_A'] + \
            mob_south*(E_H_C + E_H_E + E_H_NCen + E_H_NCoa + E_H_NI) - 5 * mob_south * E_H_S, 0)
        South_High_I_new = max(I_H_S + South_High['E_to_I'] - South_High['I_to_R'] + \
            mob_south*(I_H_C + I_H_E + I_H_NCen + I_H_NCoa + I_H_NI) - 5 * mob_south * I_H_S, 0)
        South_High_A_new = max(A_H_S + South_High['E_to_A'] - South_High['A_to_R'] + \
            mob_south*(A_H_C + A_H_E + A_H_NCen + A_H_NCoa + A_H_NI) - 5 * mob_south * A_H_S, 0)
        South_High_R_new = max(R_H_S + South_High['I_to_R'] + South_High['A_to_R'] + \
            mob_south*(R_H_C + R_H_E + R_H_NCen + R_H_NCoa + R_H_NI) - 5 * mob_south * R_H_S, 0)

        South_Low_S_new = max(S_L_S - South_Low['S_to_E'] + \
            mob_south*(S_L_C + S_L_E + S_L_NCen + S_L_NCoa + S_L_NI) - 5 * mob_south * S_L_S, 0)
        South_Low_E_new = max(E_L_S + South_Low['S_to_E'] - South_Low['E_to_I'] - South_Low['E_to_A'] + \
            mob_south*(E_L_C + E_L_E + E_L_NCen + E_L_NCoa + E_L_NI) - 5 * mob_south * E_L_S, 0)
        South_Low_I_new = max(I_L_S + South_Low['E_to_I'] - South_Low['I_to_R'] + \
            mob_south*(I_L_C + I_L_E + I_L_NCen + I_L_NCoa + I_L_NI) - 5 * mob_south * I_L_S, 0)
        South_Low_A_new = max(A_L_S + South_Low['E_to_A'] - South_Low['A_to_R'] + \
            mob_south*(A_L_C + A_L_E + A_L_NCen + A_L_NCoa + A_L_NI) - 5 * mob_south * A_L_S, 0)
        South_Low_R_new = max(R_L_S + South_Low['I_to_R'] + South_Low['A_to_R'] + \
            mob_south*(R_L_C + R_L_E + R_L_NCen + R_L_NCoa + R_L_NI) - 5 * mob_south * R_L_S, 0)

        C_H[day][0], C_H[day][1], C_H[day][2], C_H[day][3], C_H[day][4], C_H[day][5] =  S_H_C, E_H_C, I_H_C, A_H_C, R_H_C, mob_central
        C_L[day][0], C_L[day][1], C_L[day][2], C_L[day][3], C_L[day][4], C_L[day][5] =  S_L_C, E_L_C, I_L_C, A_L_C, R_L_C, mob_central

        E_H[day][0], E_H[day][1], E_H[day][2], E_H[day][3], E_H[day][4], E_H[day][5] =  S_H_E, E_H_E, I_H_E, A_H_E, R_H_E, mob_east
        E_L[day][0], E_L[day][1], E_L[day][2], E_L[day][3], E_L[day][4], E_L[day][5] =  S_L_E, E_L_E, I_L_E, A_L_E, R_L_E, mob_east

        NCen_H[day][0], NCen_H[day][1], NCen_H[day][2], NCen_H[day][3], NCen_H[day][4], NCen_H[day][5] =  S_H_NCen, E_H_NCen, I_H_NCen, A_H_NCen, R_H_NCen, mob_north_central
        NCen_L[day][0], NCen_L[day][1], NCen_L[day][2], NCen_L[day][3], NCen_L[day][4], NCen_L[day][5] =  S_L_NCen, E_L_NCen, I_L_NCen, A_L_NCen, R_L_NCen, mob_north_central

        NCoa_H[day][0], NCoa_H[day][1], NCoa_H[day][2], NCoa_H[day][3], NCoa_H[day][4], NCoa_H[day][5] =  S_H_NCoa, E_H_NCoa, I_H_NCoa, A_H_NCoa, R_H_NCoa, mob_north_coastal
        NCoa_L[day][0], NCoa_L[day][1], NCoa_L[day][2], NCoa_L[day][3], NCoa_L[day][4], NCoa_L[day][5] =  S_L_NCoa, E_L_NCoa, I_L_NCoa, A_L_NCoa, R_L_NCoa, mob_north_coastal

        NI_H[day][0], NI_H[day][1], NI_H[day][2], NI_H[day][3], NI_H[day][4], NI_H[day][5] =  S_H_NI, E_H_NI, I_H_NI, A_H_NI, R_H_NI, mob_north_inland
        NI_L[day][0], NI_L[day][1], NI_L[day][2], NI_L[day][3], NI_L[day][4], NI_L[day][5] =  S_L_NI, E_L_NI, I_L_NI, A_L_NI, R_L_NI, mob_north_inland
        
        S_H[day][0], S_H[day][1], S_H[day][2], S_H[day][3], S_H[day][4], S_H[day][5] =  S_H_S, E_H_S, I_H_S, A_H_S, R_H_S, mob_south
        S_L[day][0], S_L[day][1], S_L[day][2], S_L[day][3], S_L[day][4], S_L[day][5] =  S_L_S, E_L_S, I_L_S, A_L_S, R_L_S, mob_south

        # Update values
        S_H_C, E_H_C, I_H_C, A_H_C, R_H_C = math.ceil(Central_High_S_new), math.ceil(Central_High_E_new), math.ceil(Central_High_I_new), math.ceil(Central_High_A_new), math.ceil(Central_High_R_new)
        S_L_C, E_L_C, I_L_C, A_L_C, R_L_C = math.ceil(Central_Low_S_new), math.ceil(Central_Low_E_new), math.ceil(Central_Low_I_new), math.ceil(Central_Low_A_new), math.ceil(Central_Low_R_new)

        S_H_E, E_H_E, I_H_E, A_H_E, R_H_E = math.ceil(East_High_S_new), math.ceil(East_High_E_new), math.ceil(East_High_I_new), math.ceil(East_High_A_new), math.ceil(East_High_R_new)
        S_L_E, E_L_E, I_L_E, A_L_E, R_L_E = math.ceil(East_Low_S_new), math.ceil(East_Low_E_new), math.ceil(East_Low_I_new), math.ceil(East_Low_A_new), math.ceil(East_Low_R_new)

        S_H_NCen, E_H_NCen, I_H_NCen, A_H_NCen, R_H_NCen = math.ceil(North_Central_High_S_new), math.ceil(North_Central_High_E_new), math.ceil(North_Central_High_I_new), math.ceil(North_Central_High_A_new), math.ceil(North_Central_High_R_new)
        S_L_NCen, E_L_NCen, I_L_NCen, A_L_NCen, R_L_NCen = math.ceil(North_Central_Low_S_new), math.ceil(North_Central_Low_E_new), math.ceil(North_Central_Low_I_new), math.ceil(North_Central_Low_A_new), math.ceil(North_Central_Low_R_new)

        S_H_NCoa, E_H_NCoa, I_H_NCoa, A_H_NCoa, R_H_NCoa = math.ceil(North_Coastal_High_S_new), math.ceil(North_Coastal_High_E_new), math.ceil(North_Coastal_High_I_new), math.ceil(North_Coastal_High_A_new), math.ceil(North_Coastal_High_R_new)
        S_L_NCoa, E_L_NCoa, I_L_NCoa, A_L_NCoa, R_L_NCoa = math.ceil(North_Coastal_Low_S_new), math.ceil(North_Coastal_Low_E_new), math.ceil(North_Coastal_Low_I_new), math.ceil(North_Coastal_Low_A_new), math.ceil(North_Coastal_Low_R_new)

        S_H_NI, E_H_NI, I_H_NI, A_H_NI, R_H_NI = math.ceil(North_Inland_High_S_new), math.ceil(North_Inland_High_E_new), math.ceil(North_Inland_High_I_new), math.ceil(North_Inland_High_A_new), math.ceil(North_Inland_High_R_new)
        S_L_NI, E_L_NI, I_L_NI, A_L_NI, R_L_NI = math.ceil(North_Inland_Low_S_new), math.ceil(North_Inland_Low_E_new), math.ceil(North_Inland_Low_I_new), math.ceil(North_Inland_Low_A_new), math.ceil(North_Inland_Low_R_new)

        S_H_S, E_H_S, I_H_S, A_H_S, R_H_S = math.ceil(South_High_S_new), math.ceil(South_High_E_new), math.ceil(South_High_I_new), math.ceil(South_High_A_new), math.ceil(South_High_R_new)
        S_L_S, E_L_S, I_L_S, A_L_S, R_L_S = math.ceil(South_Low_S_new), math.ceil(South_Low_E_new), math.ceil(South_Low_I_new), math.ceil(South_Low_A_new), math.ceil(South_Low_R_new)

    return C_H, C_L, E_H, E_L, NCen_H, NCen_L, NCoa_H, NCoa_L, NI_H, NI_L, S_H, S_L