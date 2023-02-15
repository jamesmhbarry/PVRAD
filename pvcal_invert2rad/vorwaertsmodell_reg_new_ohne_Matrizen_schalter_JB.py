# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 22:55:25 2019

@author: user

Es handelt sich um das Vorwärtsmodell ohne Regularisierungsmatrizen und mit 
einem Schalter, über den die Verwendung variabler n_reguls oder der Anzahl der
Messpunkte eines Tages abzüglich 1 für n_regul festgelegt werden kann. Zu 
diesem Vorwärtsmodell gehört die inversions_function_Dirk_ohne Matrizen_schalter
"""


# -*- coding: utf-8 -*-
"""
basiert auf vorwaertsmodell_reg_new_enh2
Created on Tue Mar 26 21:05:02 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
basiert auf dem Code vorwaertsmodell_new_reg_enh
Created on Sun Mar 24 13:17:19 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
basiert auf dem Code vorwaertsmodell_reg_enh_4.py
Created on Sun Mar 10 22:14:52 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 20:19:06 2019

@author: user
"""
import numpy as np
import scipy.constants as const


def temp_model_tamizhmani(state,t_amb,g,v,g_lw):
    """
    Temperaturmodell aus TamizhMani et. al. [2003] 
    Input-Parameter:
        :param state: array of floats, enthält die Parameter des Zustandsvek-
         tors        
        :param t_amb: vector of floats, Umgebungstemperatur in Celsisus zu den 
         einzelnen Zeitpunkten
        :param g: vector of floats, Strahlungsflussdichte in W/m^2 bezogen auf 
         die Fläche der Modulebene zu den Zeitpunkten
        :param v: vector of floats, Windgeschwindigkeit in m/s in der Höhe von 
         10m über dem Boden zu den Zeitpunkten        
    Output
        :t_mod: vector of floats, Dimenension m (=m_matrix), enthält die Mo-
         dultemperatur [Grad Celsius)] zu den einzelnen Zeitpunkten
    
    """
    
    u0 = state[0]
    u1 = state[1]
    u2 = state[2]
    u3 = state[3]
    
    t_mod = u0*t_amb + u1*g + u2*v + u3*g_lw # Modultemperatur
    
    return t_mod

###############################################################################
####          Funktionen für das regularisierte Vorwärtsmodell             ####
###############################################################################



def gewichtsfaktoren_unnormiert(n_regul, time_res_sek, tau):
    """
    Zur Erläuterung wird auf die Erläuterung bei der Funktion regularisierte_
    komponente_vorwaertsmodell_2 verwiesen.
    
    Input-Parameter:
        :param n_regul: 
        :param time_res_sek: zeitliche Auflösung [s] der Werte in param ein
        :param tau: Relaxationszeit [s]  
    Output-Parameter:
        : param f: vector of floats, die einzelnen Komponenten j enthalten die
         unnormierten Gewichtsfaktorern exp(-time_res_sek*(n_regul-j)/tau)
        """
    
    # Berechnung eines array mit den unnormierten Gewichtsfaktoren
    f = np.exp(-(time_res_sek*(n_regul - np.arange(0,n_regul+1,1)))/tau)
    
    f = f[f >= 1.e-6]
    
    n_regul = len(f) - 1
    
    return f , n_regul

#def summe_gewichtsfaktoren(f, n_regul):
#    """
#    Zur Erläuterung wird auf die Erläuterungen bei der Funktion regularisierte_
#    komponente_vorwaertsmodell_2 verwiesen.
#    
#    Inpu-_Parameter:
#        :param f: vector of floats, die einzelnen Komponenten j enthalten die
#         unnormierten Gewichtsfaktorern exp(-time_res_sek*(n_regul-j)/tau)
#        :param n_regul: int, Anzahl der Messpunkte aus der Vergangenheit, die
#         zurückgegangen werden soll
#    Output-Parameter:
#        :param summe: vector of floats, enthält in der Komponente i die Summe 
#         der unnormierten Gewichtsfaktoren 
#         sum_k=0_i_{exp(-time_res_sek*(n_regul-k)/tau)}
#    """
#    # Berechnung eines array, dessen Komponenten Summe der erste i unnormierten
#    # Gewichtsfaktoren entspricht
#    summe=np.zeros(n_regul+1)
#    summe[0] = f[n_regul]
#    for i in range(1,n_regul+1,1):
#        summe[i] = summe[i-1] + f[n_regul-i]
#    
#    return summe

def matrix_smoothing_function(ein, n_regul, day_length, \
                                              f, summe, schalter):
    """
    Die Funktion geht von dem Temperaturmodell nach TamizhMani et al [2013] 
    aus:
        
        T_module = u_0*T_amb + u_1*g + u_2*v + u_3
        
        mit T_module als der Modultemperatur [C], T_amb [C] als der Umgebungs-
        temperatur, g als der auf die Ebene der Modulfläche bezogenen Strah-
        lungsflussdichte [W/m^2] und v als der Windgeschwindigkeit [m/s]. Die 
        Eingangsparameter liegen in Vektoren vor, wobei deren einzelne Kom-
        ponenten den Werten im zeitlichen Ablauf entsprechen.
        
         T_module[i] = u_0*T_amb[i] + u_1*g[i] + u_2*v[i] + u_3
    
    Die Funktion wird auf diese Eingangsparameter angewendet und regularisiert
    sie, indem statt des jeweiligen Eingangsparameters zum Zeitpunkt t_0 
    ein gewichteter Mittelwert aus diesem Eingangsparameter und seiner Werte
    aus der Vergangenheit gewählt wird. Sie bildet mithin den jeweiligen Ein-
    gangsparameter zu einem Zeitpunkt i auf einen mit seinen Vergangenheitswer-
    ten gewichteten neuen Wert ab:
    
        T_amb[i] -> T_amb_reg[i](i,i-1,i-2,...,i-n_regul)
    
    mit n_regul als der Anzahl der Datenpunkte, die in die Vergangenheit zu-
    rückgegangen werden soll.
    
    Als (nichtnormierte) Gewichtsfaktoren f[k] werden die folgenden Ausdrücke 
    verwendet, die der Funktion übergeben werden,
        
        exp(-time_res_sek*(n_regul - k)/tau), k aus [0,1,2, ..., n_regul]
    
    mit tau als Relexationszeit [s] und time_res_sek als der zeitlichen Auflö-
    sung der Eingangsparameter[s]. Die Normierung erfolgt, indem durch die 
    Summe g[j] der unnormierten Gewichtsfaktoren über alle k bis k=j dividiert 
    wird.
        
        exp(-time_res_sek*k/tau)/sum_k=0_n_j_{exp(-time_res_sek*k/tau)},
        j aus [0,1,2,....., n_regul]
    
    Es muss berücksichtigt werden, dass Messwerte erst ab einem bestimmten 
    Zeitpunkt im Tagesverlauf vorliegen. Daher hat eine bestimmte Anzahl von 
    Messpunkten eines Tages, die n_regul entspricht, weniger als n_regul
    Vergangenheitswerte. Die erforderlichen Summen g[j] werden der Funktion
    ebenfalls übergeben.
    
    Input_Parameter:
        :param ein: vector of floats, Vektor mit den Werten von t_amb, g oder 
         v (zur Bedeutung siehe oben) zu den einzelnen Zeitpunkten
        :param n_regul: int, Anzahl der Zeitschritte, die zurückgegangen werden
         soll
        :param day_length: vector of int, enthält die Anzahl der Messpunkte für
         die jeweiligen Tage
        :param f: array of floats, enthält in der Komponente i den unnormierten
         Gewichtsfaktor exp(-time_res_sek*(n_regul-i)/tau)
        :param summe: array of floats, enthält in der Komponente i die Summe 
         der unnormierten Gewichtsfaktoren 
         sum_k=0_i_{exp(-time_res_sek*(n_regul-k)/tau)}
        :param schalter: dictionary of boolean und int:
            - schalter["n_regul_variabel"]: boolean. Bei True wird 
              mit variablem n_regul gerechnet, dass n_regul_faktor* tau ist, 
              wobei tau das aktuelle tau für die Duchführung des Iterations-
              schritts ist. Bei False bleibt es fest bei dem vorgegebenen 
              n_regul, das nicht von dem aktuellen tau abhängt.
            - schalter["n_regul_faktor"]: int, ist der Faktor, mit dem das 
              aktuelle tau für die Durchführung des Iterationsschritts multi-
              pliziert werden soll. 
            - schalter["alle_messpunkte"]: boolean. Bei True werden
              in den regularisierten Wert eines Messpunkte alle vorangegangenen
              Messpunkte desselben Tages einbezogen, so dass es n_regul stets
              der Anzahl der Messpunkte des Tages abzüglich 1 entspricht. Bei 
              False wird das fest vorgegeben n_regul verwendet.
          
    Output_Parameter:
        :param ein_reg: vector of floats, seine Komponenten sind die gewichte-
         ten Werte aus param ein zu dem jeweiligen betrachteten Zeitpunkt
    """
    
    # Bestimmung der Länge des Vektors mit der Eingangsgröße param ein
    m_matrix = len(ein)
    
    # Initialisierung eines array für die Rückgabegröße F_ein:
    ein_reg = np.zeros(m_matrix)
    
    # Initialisierung einer Zählvariable für die Messpunkte
    c = 0
    
    # Initialisierung einer Zählvariable für die behandelten Tage
    day = 0
    
    if schalter:
        # n_regul = day_length.max()-1
        if n_regul != (len(f) - 1):
            print("Fehler bei der Bildung des Vektors f bei alle_messpunkte=True")
    
    for i in range(0,m_matrix,1):
        """
        Hier wird ein variabler Mittelungskern verwendet, der berücksichtigt, 
        dass nicht für alle Zeitpunkte am Anfang eines Tages n_regul Messwerte
        aus der Vergangenheit vorliegen.
        Die Anzahl der vorhandenen Messpunkte aus der Vergangenheit, die zu-
        rückgegangen werden kann, lautet:
        
            j_fuer_start_wert_data = min(n_regul, i-c)
        
        Ausgehend vom betrachteten Messpunkt i ergeben sich die Messpunkte aus
        der Vergangenheit über
            
            ein[i-j:i+1]
        
        Die Position des entsprechenden unnormierten Gewichtsfaktors in f er-
        gibt sich dabei aus f[n_regul-j] mit
        
            j_fuer_start_wert_f = min(n_regul, i-c)        
        
        Die richtige Summe für die Anzahl der berücksichtigten Messwerte aus 
        der Vergangenheit ergibt sich aus:
            
            index_summe = min(n_regul, i-c)
        
        Daher kann ein einheitlicher Index j verwendet werden.
        
        """
        j = min(n_regul, i-c)
        ein_reg[i] = np.dot(f[n_regul-j:n_regul+1],ein[i-j:i+1]) /summe[j]      
        if i == (c-1+day_length[day]):
            c = c + day_length[day]
            day = day + 1               
    
    return ein_reg

def differentiate_matrix_smoothing_function(ein, n_regul, \
                                    day_length, tau, k1, v, time_res_sek, f, g, \
                                    schalter, k_switch):
    """
    Zunächst wird auf die Erläuterungen zur Funktion regularisierte_komponente_
    vorwaertsmodell verwiesen, die die Gewichtsfaktoren erzeugt.
    Verwendet wird für die Berechnung der Ableitungen der Gewichtsfaktoren nach
    tau die Quotientenregel
        
        (f/g)' = (f'*g - f*g')/g^2,
    
    weil die einzelnen normierten Gewichtsfaktoren die folgende Struktur haben:
        
            
        [exp(- time_res_sek*(n_regul - k)/tau)]/ \
        sum_k=0_n_regul_{exp(-(n_regul-k)/tau)}
        
    Mit f_ij(tau) = exp(- time_res_sek*(n_regul - k)/tau) und 
    g_i(tau) = sum_k=0_n_regul_{exp(-time_res_sek*(n_regul-k)/tau)} ergibt 
    sich 
        
        (f_ij)' = df_ij(tau)/dtau = [exp(-time_res_sek*(n_regul - k)/tau)]*\
                  time_res_sek*(n_regul-k)/tau^2
        (g_i)' = dg_i(tau)/dtau = sum_k=0_n_regul_{[exp(- time_res_sek*\
                 (n_regul-k)/tau)]*time_res_sek*(n_regul - k)/tau^2}
    
    Hieraus ergibt sich nach der Quotientenregel:
    
        A' = dA(tau)/dtau = {[exp(- time_res_sek*(n_regul-k)/tau)]*\
             time_res_sek*(n_regul-k)/tau^2*\
             sum_k=0_n_regul_{exp(- time_res_sek*(n_regul-k)/tau)} - \
             [exp(- time_res_sek*(n_regul-k)/tau)]*\
             sum_k=0_n_regul_{ [exp(- time_res_sek*(n_regul-k)/tau]*\
             time_res_sek*(n_regul-k)/tau^2}}/
             {sum_k=0_n_regul_{exp(- time_res_sek*(n_regul-k)/tau)}}^2
    
    Es muss zudem berücksichtigt werden, dass für bestimmte Messpunkte am Be-
    Beginn eines Tages nicht n_regul Vergangenheitswerte vorliegen. Für die 
    Handhabung dieser Problematik wird auf die Erläuterungen in der Funktion
    regularisierte_komponente_vorwaetsmodell verwiesen.
    Sollen zur Ermittlung des Werts der mit dem aktuellen Wert und Vergangen-
    heitswerten gewichteten Messpunkte stets alle vorangegangenen Vergangen-
    heitswerte desselben Tages verwendet werden, ist schalter["alle_messpunkte"]
    auf True zu stellen. 
    Input_Parameter:
        :param ein: vector of floats, Vektor mit den Werten von t_amb, g oder 
         v (zur Bedeutung siehe die Erläuterungen zur Funktion 
         regularisierte_komponente_vorwaertsmodell) zu den einzelnen Zeitpunk-
         ten
        :param n_regul: int, Anzahl der Zeitschritte, die zurückgegangen werden
         soll
        :param day_length: vector of int, enthält die Anzahl der Messpunkte für
         die jeweiligen Tage
        :param tau: float, Relaxationszeit [s]
        :param_time_res_sek: float, zeitliche Auflösung [s] der Messpunkte in 
         ein
        :param f: array of floats, enthält in der Komponente i den unnormierten
         Gewichtsfaktor exp(-time_res_sek*(n_regul-i)/tau)
        :param g: array of floats, enthält in der Komponente i die Summe 
         der unnormierten Gewichtsfaktoren 
         sum_k=0_i_{exp(-time_res_sek*(n_regul-k)/tau)}
        :param schalter: dictionary of boolean und int:
            - schalter["n_regul_variabel"]: boolean. Bei True wird 
              mit variablem n_regul gerechnet, dass n_regul_faktor* tau ist, 
              wobei tau das aktuelle tau für die Duchführung des Iterations-
              schritts ist. Bei False bleibt es fest bei dem vorgegebenen 
              n_regul, das nicht von dem aktuellen tau abhängt.
            - schalter["n_regul_faktor"]: int, ist der Faktor, mit dem das 
              aktuelle tau für die Durchführung des Iterationsschritts multi-
              pliziert werden soll. 
            - schalter["alle_messpunkte"]: boolean. Bei True werden
              in den regularisierten Wert eines Messpunkte alle vorangegangenen
              Messpunkte desselben Tages einbezogen, so dass es n_regul stets
              der Anzahl der Messpunkte des Tages abzüglich 1 entspricht. Bei 
              False wird das fest vorgegeben n_regul verwendet.
    Output_Parameter:
        :param ein_diff: vector of floats, seine Komponenten sind die Ableitun-
         gen der gewichteten Werte aus param ein zu dem jeweiligen betrachteten 
         Zeitpunkt    
    """
    
    
    # Initialisierung von Zählvariablen
    
       
    c = 0  # Zählvariable für die Anzahl der Messpunkte der vorangegangenen 
           # Tage; da die Messpunkte in einem array beginnend mit dem Index
           # Null abgelegt sind, steht der erste Messpunkt eines Tage im Ele-
           # ment dieses array mit dem Index c-1 und der letzte im Element mit
           # dem Index c-1+ der Anzahl der Messpunkte für diesen Tag
    day = 0 # Zählvariable für die einzelnen Tage, wobei dem ersten Tag die Null
            # zugewiesen wird
    
    m_matrix = len(ein)
    
    # Initialisierung eines Vektors für die mit den Ableitungen gewichteten Ein-
    # gangswerte
    ein_diff = np.zeros(m_matrix)
    
    # Einstellungen zur Berücksichtigung aller vorangegangenen Messpunkte eines 
    # Tages; in diesem Fall ist n_regul = day_length.max() -1
    if schalter:
        #n_regul = day_length.max()-1
        if n_regul != (len(f) - 1):
            print("Fehler bei der Bildung des Vektors f bei alle_messpunkte=True")
    
    # Initialisierung eines Vektors für die Summe der Ableitungen der unnormier-
    # ten Gewichtsfaktoren            
               
    """
    Berechnung von f, der Ableitung von f nach tau, der Zeilensumme g und der 
    Ableitung der Zeilensumme g nach tau. 
    Der Vector f enthält als Elemente die nicht normierten Gewichtsfaktoren in\
    der Reihenfolge 
    (exp(-(time_res_sek*n_regul)/tau)),
    (exp(-(time_res_sek*(n_regul-1))/tau)), ..., 
    (exp(-(time_res_sek*1)/tau))
    (exp(-time_res_sek*0/tau))=1.0; 
    es gibt n_regul+1 verschiedene Gewichtsfaktoren, wobei der j-te Gewichts-
    faktor (exp(-(time_res_sek*(n_regul-j))/tau)) in dem Element von f mit 
    dem Index i = j steht.
    Der Vektor f_diff enthält die Ableitungen der n_regul Gewichtsfaktoren in 
    der vorstehenden Reihenfolge, also 
    (exp(-(time_res_sek*n_regul)/tau))*(n_regul/tau^2),
    (exp(-(time_res_sek*(n_regul-1))/tau))*((n_regul-1)/tau^2), ...,
    (exp(-(time_res_sek*1)/tau))*(time_res_sek*1/tau^2),
    (exp(-(time_res_sek*0)/tau))*(time_res_sek*0/tau^2)=0.0.
    In den Vektor g wird die jeweilige Summe der Gewichtsfaktoren der Zeilen 
    der Regularisierungsmatrix mit dem Zeilenindex von c bis c+n_regul (mit der 
    Anzahl der kummulierten Messpunkte c der vorangegangenen Tage) geschrie-
    ben, die in den Elementen von g mit dem Index von 0 bis n_regul abgespei-
    chert werden, wobei zur Berechnung der jeweiligen Zeilensumme auf die Ele-
    mente von f zurückgegriffen wird. Für die Zeilen der Regularisierungsmatrix 
    mit einem Zeilenindex größer als c+n_regul ändert sich für den betrachteten
    jeweiligen Tag die Zeilensumme nicht mehr und entspricht der Zeilensumme 
    der Zeile mit dem Zeilenindex c+n_regul.
    In den Vektor g_diff wird die Ableitung der jeweiligen Summe der Gewichts-
    faktoren der Zeilen der Regularisierungsmatrix mit dem Zeilenindex von c 
    bis c+n_regul geschrieben, die in den Elementen von g mit dem Index von 0 
    bis n_regul abgespeichert werden, wobei zur Berechnung der jeweiligen Zei-
    lensumme auf die Elemente von f_diff zurückgegriffen wird.   
    """
    # Berechnung des Vektors mit den Ableitungen von f
    j_vec = np.arange(0,n_regul+1,1)
    #f =  np.exp(-(time_res_sek*(n_regul - j_vec))/tau)
    #f_diff = (np.exp(-(time_res_sek*(n_regul - j_vec))/tau))*\
    #(time_res_sek*(n_regul-j_vec))/(tau*tau)
    
    dtau_dk1 = 1. #(k2 - k1*k2)/(k1 + k2*np.sqrt(v))**2
    
    #dtau_dk2 = (k1 + k1*k2*np.sqrt(v))/(k1 + k2*np.sqrt(v))**2
    
    if k_switch == "k1":
        f_diff = f*(time_res_sek*(n_regul-j_vec)*dtau_dk1)/(tau*tau)
#    elif k_switch == "k2":
#        f_diff = f*(time_res_sek*(n_regul-j_vec)*dtau_dk2)/(tau*tau)
    
    # Berechnung des Vektors mit der Summe der Ableitungen f_diff            
    g_diff = np.cumsum(np.flip(f_diff))
        
##   Alternative Berechnung
#    
#    # Initialisierung eines Vektors f mit den nichtnormierten, von Null ver-
#    # schiedenen Einträgen der Regularisierungsmatrix. Es gibt n_regul+1 unter-
#    # schiedliche Einträge, die nicht Null sind
#    f = np.zeros(n_regul+1)
#        
#    # Initialisierung eines Vektors f_diff mit den Ableitungen der nichtnor-
#    # mierten, von Null verschiedenen Einträgen der Regularisierungsmatrix. Es 
#    # gibt n_regul+1 unterschiedliche Einträge, die nicht Null sind
#    f_diff = np.zeros(n_regul+1)
#    
#    # for Schleife zum Berechnen von f, der Ableitungen von f, der Zeilensummen 
#    # und der Ableitungen der Zeilensummen
#    
#    f[0] = 1.0 
#    f_diff[0] = 0.0
#    g[0] = f[0]
#    g_diff[0] = f_diff[0]
#    
#    for i in range(1,n_regul+1,1):
#        f[i] = np.exp(-(time_res_sek*i)/tau)
#        f_diff[i] = np.exp(-(time_res_sek*i)/tau)*(time_res_sek*i)/(tau*tau)
#        g[i] = g[i-1] + f[i]
#        g_diff[i] = g_diff[i-1] + f_diff[i]
#            
#    f = np.flip(f)
#    f_diff = np.flip(f_diff)

      
#     Die Schleife berechnet nach der Quotientenregel die Ableitung der nor-
#     mierten Regularisierungsmatrix nach tau; die Ableitung von A[0,0] ist 
#     Null und muss daher nicht betrachtet werden
       
    for i in range(0,m_matrix,1):
        """
        Hier wird ein variabler Mittelungskern verwendet, der berücksichtigt, 
        dass nicht für alle Zeitpunkte am Anfang eines Tages n_regul Messwerte
        aus der Vergangenheit vorliegen.
        Die Anzahl der vorhandenen Messpunkte aus der Vergangenheit, die zu-
        rückgegangen werden kann, lautet:
        
            j_fuer_start_wert_data = min(n_regul, i-c)
        
        Ausgehend vom betrachteten Messpunkt i ergeben sich die Messpunkte aus
        der Vergangenheit über
            
            ein[i-j:i+1]
        
        Die Position des entsprechenden unnormierten Gewichtsfaktors in f er-
        gibt sich dabei aus f[n_regul-j] mit
        
            j_fuer_start_wert_f = min(n_regul, i-c)        
        
        Entsprechendes gilt für die Ableitung des unnormierten Gewichtsfaktors.
        Seine Position in dem array mit den Ableitungen ist an der Stelle 
        f_diff[n_regul-j] mit
            
            j_fuer_start_wert_f_diff = min(n_regul, i-c)
        
        Die richtige Summe der unnormierten Gewichtsfaktoren bzw. von deren Ab-
        leitung für die Anzahl der berücksichtigten Messwerte aus der Vergan-
        genheit ergibt sich aus:
            
            index_summe = min(n_regul, i-c)
        
        Daher kann ein einheitlicher Index j verwendet werden.
        
        """
        j = min(n_regul, i-c)
        if g[j] != 0.0:
            ein_diff[i] = np.dot(f_diff[n_regul-j:n_regul+1]*g[j] - \
                f[n_regul-j:n_regul+1]*g_diff[j],ein[i-j:i+1]) /(g[j]*g[j])
        else:
            print("Fehler bei der Berechnung der Summe der Gewichtsfaktoren. \
                   Die Summe wird Null")
            
        if i == (c-1+day_length[day]):
            c = c + day_length[day]
            day = day + 1
    
#    for i in range(1, m_matrix,1):
#        startwert_A = max(c,i-n_regul)
#        startwert_f = max(0, c+n_regul-i)
#        zeile = min(i-c,n_regul)
#        if g[zeile] != 0.0:
#            A_diff_tau[i,startwert_A:i+1] = \
#            (f_diff[startwert_f:n_regul+1]*g[zeile] - \
#             f[startwert_f:n_regul+1]*g_diff[zeile])/(g[zeile]*g[zeile])
#        else:
#            print("Fehler bei der Berechnung der Zeilensumme. Die Zeilensumme \
#                  wird Null")  
#        if i == (c-1+day_length[day]):
#            c = c + day_length[day]
#            day = day + 1
    
#     for i in range(0, m_matrix,1):
#        startwert_A = max(c,i-n_regul)
#        startwert_H = max(0, c + n_regul-i)
#        A[i,startwert_A:i+1] = H[startwert_H: n_regul+1]
#        if i == (c-1+day_length[day]):
#            c = c + day_length[day]
#            day = day + 1
    
    
    return ein_diff

def F_temp_model_dynamic(state,t_amb, g, v, g_lw, n_regul, number_days, day_length, \
                     time_res_sek, schalter):
    """
    Berechnet das Vorwärtsmodell für den Zustandsvektor state, angelehnt an die 
    Funktion aus pvcal_forward_model.py 22.01.2019 14:59
    Die Funktion setzt voraus, dass t_mod, t_amb, g, v dieselbe Anzahl an Kom-
    ponenten haben.
    Das Temperaturgrundmodell 
        
        t_module = u0*t_amb + u1*g + u2*v + u3
    
    stammt aus TamizhMani et. al. [2003]. 
    Es wird um die Regularisierungsmatrizen Aa, Ag und AV ergänzt:
      
        t_mod= u_0*A_amb(tau_amb)*t_amb + u_1*A_G(tau_G)*g + u_2*A_v(tau_v)*v \
               + u_3
    
    mit
        :param u_0, u_1, u_2, u_3, int, Modellparameter des Temperaturmodells
         aus TamizhMani et. al. [2003]; sie sind Teil der Komponenten des Zu-
         standsvektors state
        :t_mod, array of floats, Dimenension m (=m_matrix), enthält die Modul-
         temperatur [Grad Celsius)] zu den einzelnen Zeitpunkten
        :param t_amb, vector of floats, Umgebungstemperatur in Celsisus zu den 
         einzelnen Zeitpunkten 
        :param g, vector of floats, enthält die Strahlungsflussdichte bezogen
         auf die Ebene der Modulfläche in W/m^2 zu den einzelnen Zeitpunkten
        :param v, vector of floats, Windgeschwindigkeit in m/s zu den ein-
         zelnen Zeitpunkten
        :A_amb, A_G, A_v, (m (=m_matrix) x m (= m_matrix))-dimensionale Regula-
         risierungsmatrizen für die Umgebungstemperatur t_amb [Grad Celsius], 
         die Strahlungsflussdichte G [W/m^2] und die Windgeschwindigkeit v 
         [m/s]
        :param tau_amb, tau_G, tau_v, float, Relaxationszeiten [Größe ent-
         spricht Sekunden] hinsichtlich der Umgebungstemperatur t_amb 
         [Grad Celsius], der Strahlungsflussdichte G [W/m^2] und der Windge-
         schwindigkeit v [m/s]; sie sind Komponenten des Zustandsvektors
   
      
    Die Funktion setzt voraus, dass t_mod, t_amb, g, v dieselbe Anzahl an Kom-
    ponenten haben.
    Input-Parameter:
        :param state, vector of floats, enthält die Komponenten des Zustands-
         vektors tau_amb, tau_G, tau_v, u_0, u_1, u_2, u_3; zur Bedeutung der 
         Komponenten siehe die obenstehenden Erläuterungen zum Vorwärtsmodell        
        :param t_amb, vector of floats, zur Bedeutung siehe obenstehende Erläu-
         terungen zum Vorwärtsmodell      
        :param g, vector of floats, zur Bedeutung siehe obenstehende Erläute-
         rungen zum Vorwärtsmodell
        :param v, vector of floats, zur Bedeutung siehe obenstehende Erläute-
         rungen zum Vorwärtsmodell
        :param n_regul: int, Anzahl der Zeitpunkte, die im Rahmen der Regulari-
         sierung werden sollen
        :param number_days, int, enthält die Anzahl der Tage
        :param day_length: array of int, enthält die Anzahl der Messpunkte zu 
         jedem Tag
        :param time_res_sek: float, enthält die zeitliche Auflösung der Daten 
         in Sekunden; wird für die Regularisierung benötigt
        :param m_matrix: int, Dimension der Matrix für die Regulariserung; ent-
         spricht der Anzahl der betrachteten Zeitpunkte, zu denen Messwerte 
         vorliegen
        :param schalter: dictionary of boolean und int:
            - schalter["n_regul_variabel"]: boolean. Bei True wird 
              mit variablem n_regul gerechnet, dass n_regul_faktor* tau ist, 
              wobei tau das aktuelle tau für die Duchführung des Iterations-
              schritts ist. Bei False bleibt es fest bei dem vorgegebenen 
              n_regul, das nicht von dem aktuellen tau abhängt.
            - schalter["n_regul_faktor"]: int, ist der Faktor, mit dem das 
              aktuelle tau für die Durchführung des Iterationsschritts multi-
              pliziert werden soll. 
            - schalter["alle_messpunkte"]: boolean. Bei True werden
              in den regularisierten Wert eines Messpunkte alle vorangegangenen
              Messpunkte desselben Tages einbezogen, so dass es n_regul stets
              der Anzahl der Messpunkte des Tages abzüglich 1 entspricht. Bei 
              False wird das fest vorgegeben n_regul verwendet.
                
    Output:
        :param t_mod: vector of floats, enthält die Temperatur des Moduls zu 
         den einzelnen Zeitpunkten an der Stelle x = state
#         :dictionary of vector and Matrix
#            :t_mod, array of floats, Dimenension m (=m_matrix), enthält die Mo-
#             dultemperatur [Grad Celsius)] zu den einzelnen Zeitpunkten
#            :K, (len(t_amb),len(state))-dimensionales array of floats, enthält 
#             die (m x n)-dimensionale Jacobi-Matrix
    
    """
        
    #u0 = state[0]
    u1 = state[0]
    u2 = state[1]
    u3 = state[2]
    k1 = state[3]
    #k2 = state[4]
        
    if type(n_regul) == int:
        n_regul = n_regul
    elif type(n_regul) == np.ndarray:
        n_regul = n_regul[0]
    else:
        print("Fehler. n_regul ist weder integer noch numpy.ndarray")
    
    # Vorbereitungen für die Option alle_messpunkte
    if schalter:
        n_regul_max = day_length.max() - 1
        n_regul = n_regul_max                
    
    # Initialisierung der arrays mit den unnormierten Gewichtsfaktoren 
    # exp(-time_res_sek*(n_regul-j)/tau) bzw. deren Summe sum_k=0_j_{\
    # exp(-time_res_sek*(n_regul-j)/tau)} als deren Komponenten
    
    tau_new = k1 #*k2/(k2*np.sqrt(v) + k1)
    
    f, n_regul = gewichtsfaktoren_unnormiert(n_regul, time_res_sek, tau_new)
    g_norm = np.cumsum(np.flip(f))
        
    
#    t_mod = u0*regularisierte_komponente_vorwaertsmodell_2(t_amb, n_regul_amb, \
#                                                    day_length, f_amb, g_amb) \
#          + u1*regularisierte_komponente_vorwaertsmodell_2(g, n_regul_g,\
#                                                    day_length, f_g, g_g)\
#          + u2*regularisierte_komponente_vorwaertsmodell_2(v,n_regul_v,\
#                                                    day_length, f_v, g_v)\
#          + u3 
    # Initialisierung der regularisierten Vektoren
    
    t_static = t_amb + g/(u1 + u2*np.power(v,1.)) +\
                u3*(np.power(g_lw/const.sigma,1./4) - 273.15 - t_amb)
                
    t_mod = matrix_smoothing_function(t_static, n_regul,\
                                                    day_length, f, g_norm,\
                                                    schalter)        
        
    return t_mod

def K_temp_mod_dynamic(state, t_amb, g, v, g_lw, n_regul, number_days, day_length, \
                   time_res_sek, schalter):
        
    """
    Berechnet die Jacobi-Matrix in Abhängigkeit vom Zustandsvektor state. Die 
    Jacobi-Matrix ist die Matrix der Ableitungen der  Modultemperatur nach dem 
    Zustandsvektor. Die Jacobi-Matrix hat die Dimension (m x n), wobei m die 
    Anzahl der Messungen und n die Anzahl der Komponenten des Zustandsvektors 
    ist.
    Input-Parameter:
        :param state, array of floats, enthält die Komponenten des Zustandsvek-
         tors tau_amb, tau_G, tau_v, u_0, u_1, u_2, u_3; zur Bedeutung der Kom-
         ponenten siehe die nachstehenden Erläuterungen zum Vorwärtsmodell
        :param t_amb, array of floats, Dimension m_matrix, enthält die Umge-
         bungstemperatur [Grad Celsius] zu den einzelnen Zeitpunkten
        :param g, array of floats, Dimension m_matrix, enthält die Strahlungs-
         flussdichten [W/m^2] bezogen auf die Ebene der Modulfläche zu den ein-
         zelnen Zeitpunkten
        :param v, array of floats, Dimension m_matrix, enthält die Windge-
         geschwindigkeiten [m/s] zu den einzelnen Zeitpunkten. Bei PV 12 stam-
         men sie aus den (zeitlich interpolierten) Cosmo-Daten und sind auf ei-
         ne Höhe von 10 m über dem Boden bezogen. Bei der Masterstation 12 wur-
         den sie in der Höhe von [] über dem Boden gemessen.   
        :param n_regul: int, Anzahl der Zeitpunkte, die im Rahmen der Regulari-
         sierung werden sollen
        :param number_days, int, enthält die Anzahl der Tage
        :param day_length: array of int, enthält die Anzahl der Messpunkte zu 
         jedem Tag
        :param time_res_sek: float, enthält die zeitliche Auflösung der Daten 
         in Sekunden; wird für die Regularisierung benötigt
        :param m_matrix: int, Dimension der Matrix für die Regulariserung; ent-
         spricht der Anzahl der betrachteten Zeitpunkte, zu denen Messwerte 
         vorliegen
        :param schalter: dictionary of boolean und int:
            - schalter["n_regul_variabel"]: boolean. Bei True wird 
              mit variablem n_regul gerechnet, dass n_regul_faktor* tau ist, 
              wobei tau das aktuelle tau für die Duchführung des Iterations-
              schritts ist. Bei False bleibt es fest bei dem vorgegebenen 
              n_regul, das nicht von dem aktuellen tau abhängt.
            - schalter["n_regul_faktor"]: int, ist der Faktor, mit dem das 
              aktuelle tau für die Durchführung des Iterationsschritts multi-
              pliziert werden soll. 
            - schalter["alle_messpunkte"]: boolean. Bei True werden
              in den regularisierten Wert eines Messpunkte alle vorangegangenen
              Messpunkte desselben Tages einbezogen, so dass es n_regul stets
              der Anzahl der Messpunkte des Tages abzüglich 1 entspricht. Bei 
              False wird das fest vorgegeben n_regul verwendet.
    Output-Parameter:
        :param  ergebnisse: dictionary, enthält
            - param ergebnisse["t_mod"] = t_mod, vector of arrays, mit dem Vor-
              wärtsmodell modellierte Modultemperatur
            - param ergebnisse["K"] = K, array of floats, (len(t_amb),len(state)) 
              Matrix, enthält die (m x n)-dimensionale Jacobi-Matrix an der 
              Stelle x = state
                
    Die Funktion setzt voraus, dass t_mod, t_amb, g, v dieselbe Anzahl an Kom-
    ponenten haben.
    
    Das zugrundeliegende Vorwärtsmodell für die Berechnung der Jacobi-MAtrix 
    ist
    
    t_mod= u_0*A_amb(tau_amb)*t_amb + u_1*A_G(tau_G)*g + u_2*A_v(tau_v)*v + u_3
    
    mit
        :param u_0, u_1, u_2, u_3, int, Modellparameter des Temperaturmodells
         aus aus TamizhMani et. al. [2003]; sie sind Komponenten des Zustands-
         vektors state
        :t_mod, array of floats, Dimenension m (=m_matrix), enthält die Modul-
         temperatur [Grad Celsius)] zu den einzelnen Zeitpunkten
        :param t_amb, s.o.
        :param g, s.o.
        :param v, s.o.
        :A_amb, A_G, A_v, (m (=m_matrix) x m (= m_matrix))-dimensionale Regula-
         risierungsmatrizen für die Umgebungstemperatur t_amb [Grad Celsius], 
         die Strahlungsflussdichte G [W/m^2] und die Windgeschwindigkeit v 
         [m/s]
        :param tau_amb, tau_G, tau_v, float, Relaxationszeiten [Größe ent-
         spricht Sekunden] hinsichtlich der Umgebungstemperatur t_amb 
         [Grad Celsius], der Strahlungsflussdichte G [W/m^2] und der Windge-
         schwindigkeit v [m/s]; sie sind Komponenten des Zustandsvektors
    
     
    Hieraus ergibt sich für die i-ten Zeilen der Jacobi-Matrix:
        [u_0*(del_A_amb_(i)/del_tau_amb)*t_amb];[u_1*(del_A_g_(i)/del_tau_g)*g]
        ;[u_2*(del_A_v_(i)/del_tau_v)*v];[(del_u_0/del_u_0)*A_amb_(i)*t_amb];
        [(del_u_1/del_u_1)*A_G_(i)*g];[(del_u_2/del_u_2)*A_v_(i)*v]; 
        [(del_u_3/del_u_3)] =
        [u_0*(del_A_amb_(i)/del_tau_amb)*t_amb];[u_1*(del_A_g_(i)/del_tau_g)*g]
        ;[u_2*(del_A_v_(i)/del_tau_v)*v];[A_amb_(i)*t_amb] [A_G_(i)*g];
        [A_v_(i)*v] [1]
        
    Bei der Berechnung der Ableitungen der Regularisierungsmatrizen A_amb, A_G
    und A_v nach der korrespondierenden Relaxationszeit tau_amb, tau_G bzw. 
    tau_v wird die Funktion differentiate_tau verwendet, die die Ableitung ei-
    ner Regularisierungsmatrix nach der Relaxationszeit tau berechnet.
        
    """    
    
    # Extrahieren der Komponenten aus dem Zustandsvektor state
    #u0 = state[0]
    u1 = state[0]
    u2 = state[1]
    u3 = state[2]
    k1 = state[3]
    #k2 = state[4]
    
    # Initialisierung der (m x n)-dimensionalen Jacobi-Matrix  mit Nullen
    K=np.zeros((len(t_amb),len(state)))
     
    # Vorbereitungen für die Option alle_messpunkte
    if schalter:
        n_regul_max = day_length.max() - 1
        n_regul = n_regul_max
    
    #The static temperature model
    t_static = t_amb + g/(u1 + u2*np.power(v,1.)) +\
                u3*(np.power(g_lw/const.sigma,1./4) - 273.15 - t_amb)
    
    #Time constant
    tau_new = k1 #*k2/(k2*np.sqrt(v) + k1)
    #tau_new[tau_new < 0] = 10.0
    
    f, n_regul = gewichtsfaktoren_unnormiert(n_regul, time_res_sek, tau_new)
    g_norm = np.cumsum(np.flip(f))
    
    #New part, James (alternative method is still similar speed!)
#    f_diff = f*(time_res_sek*(n_regul-np.arange(0,n_regul+1,1)))/(tau_new*tau_new)
#    g_diff = np.cumsum(np.flip(f_diff))
#    
#    t_amb_reg = np.zeros(len(t_amb))
#    g_reg = np.zeros(len(t_amb))
#    v_reg = np.zeros(len(t_amb))
#    g_lw_reg = np.zeros(len(t_amb))
#    t_mod = np.zeros(len(t_amb))
#    count = 0
#    for i, day in enumerate(day_length):                
#        #Calculate matrix
#        zeros = np.zeros(day - len(f))
#        f_full = np.hstack([zeros,f])
#        mat = np.tril(np.vstack([np.roll(f_full,j) for j in range(1,len(f_full)+1)]))
#        matnorm = mat / np.linalg.norm(mat, axis=1, ord=1)[:, np.newaxis]
#        
#        #Multiply terms with matrix
#        t_amb_reg[count:count+day] = np.dot(matnorm,t_amb[count:count+day])
#        g_reg[count:count+day] = np.dot(matnorm,g[count:count+day])
#        v_reg[count:count+day] = np.dot(matnorm,v[count:count+day])
#        g_lw_reg[count:count+day] = np.dot(matnorm,g_lw[count:count+day])
#        t_mod[count:count+day] = np.dot(matnorm,t_static[count:count+day])
#        count = count + day
    
    
#    
    t_amb_reg = matrix_smoothing_function(t_amb, n_regul, day_length,f,g_norm, schalter)
    
    g_reg = matrix_smoothing_function(g, n_regul, day_length, f, g_norm, schalter)     
        
    v_reg = matrix_smoothing_function(v, n_regul, day_length, f, g_norm, schalter)
    
    g_lw_reg = matrix_smoothing_function(g_lw, n_regul, day_length, f, g_norm, schalter)
        
    t_mod = matrix_smoothing_function(t_static, n_regul, day_length, f, g_norm, schalter)                                                                   
        
    # Berechnung der mit den Ableitungen der Gewichtungsfaktoren nach den je-
    # weiligen tau gewichteten Vektoren 
    t_mod_reg_del_k1 = \
    differentiate_matrix_smoothing_function(t_static,\
                 n_regul, day_length, tau_new, k1, v, time_res_sek, f, g_norm, \
                 schalter,"k1")
    
    # Berechnung der Ableitungen des Vorwärtsmodells t_mod nach den Komponenten 
    # des Zustandsvektors; es handelt sich um die m-dimensionalen Spaltenvek-
    # toren der Jacobi-Matrix mit m=m_matrix=len(t_amb)
        
    #t_mod_del_u0 = t_amb_reg
    t_mod_del_u1 = -g_reg/(u1 + u2*np.power(v_reg,1.))**2
    t_mod_del_u2 = -g_reg*np.power(v_reg,1.)/(u1 + u2*np.power(v_reg,1.))**2
    t_mod_del_u3 = np.power(g_lw_reg/const.sigma,1./4) - 273.15 - t_amb_reg
    
    # Erstellen der (m x n)-dimensionalen Jacobi-Matrix K mit m=m_matrix=
    # len(t_amb) und n=len(state)
    #K[:,0] = t_mod_del_u0 #tau_amb
    K[:,0] = t_mod_del_u1 #tau_G
    K[:,1] = t_mod_del_u2 #tau_v
    K[:,2] = t_mod_del_u3
    K[:,3] = t_mod_reg_del_k1
    #K[:,4] = t_mod_reg_del_k2
    #K[:,6] = t_mod_del_u3
    
    ergebnisse={}
    ergebnisse["t_mod"] = t_mod
    ergebnisse["K"] = K
    
    return ergebnisse

def F_temp_model_static(state,t_amb, g, v, g_lw):
    """
    Berechnet das Vorwärtsmodell für den Zustandsvektor state, angelehnt an die 
    Funktion aus pvcal_forward_model.py 22.01.2019 14:59
    Die Funktion setzt voraus, dass t_mod, t_amb, g, v dieselbe Anzahl an Kom-
    ponenten haben.
    Das Temperaturgrundmodell 
        
        t_module = u0*t_amb + u1*g + u2*v + u3
    
    stammt aus TamizhMani et. al. [2003]. 
    Es wird um die Regularisierungsmatrizen Aa, Ag und AV ergänzt:
      
        t_mod= u_0*A_amb(tau_amb)*t_amb + u_1*A_G(tau_G)*g + u_2*A_v(tau_v)*v \
               + u_3
    
    mit
        :param u_0, u_1, u_2, u_3, int, Modellparameter des Temperaturmodells
         aus TamizhMani et. al. [2003]; sie sind Teil der Komponenten des Zu-
         standsvektors state
        :t_mod, array of floats, Dimenension m (=m_matrix), enthält die Modul-
         temperatur [Grad Celsius)] zu den einzelnen Zeitpunkten
        :param t_amb, vector of floats, Umgebungstemperatur in Celsisus zu den 
         einzelnen Zeitpunkten 
        :param g, vector of floats, enthält die Strahlungsflussdichte bezogen
         auf die Ebene der Modulfläche in W/m^2 zu den einzelnen Zeitpunkten
        :param v, vector of floats, Windgeschwindigkeit in m/s zu den ein-
         zelnen Zeitpunkten
        :A_amb, A_G, A_v, (m (=m_matrix) x m (= m_matrix))-dimensionale Regula-
         risierungsmatrizen für die Umgebungstemperatur t_amb [Grad Celsius], 
         die Strahlungsflussdichte G [W/m^2] und die Windgeschwindigkeit v 
         [m/s]
        :param tau_amb, tau_G, tau_v, float, Relaxationszeiten [Größe ent-
         spricht Sekunden] hinsichtlich der Umgebungstemperatur t_amb 
         [Grad Celsius], der Strahlungsflussdichte G [W/m^2] und der Windge-
         schwindigkeit v [m/s]; sie sind Komponenten des Zustandsvektors
   
      
    Die Funktion setzt voraus, dass t_mod, t_amb, g, v dieselbe Anzahl an Kom-
    ponenten haben.
    Input-Parameter:
        :param state, vector of floats, enthält die Komponenten des Zustands-
         vektors tau_amb, tau_G, tau_v, u_0, u_1, u_2, u_3; zur Bedeutung der 
         Komponenten siehe die obenstehenden Erläuterungen zum Vorwärtsmodell        
        :param t_amb, vector of floats, zur Bedeutung siehe obenstehende Erläu-
         terungen zum Vorwärtsmodell      
        :param g, vector of floats, zur Bedeutung siehe obenstehende Erläute-
         rungen zum Vorwärtsmodell
        :param v, vector of floats, zur Bedeutung siehe obenstehende Erläute-
         rungen zum Vorwärtsmodell
        :param n_regul: int, Anzahl der Zeitpunkte, die im Rahmen der Regulari-
         sierung werden sollen
        :param number_days, int, enthält die Anzahl der Tage
        :param day_length: array of int, enthält die Anzahl der Messpunkte zu 
         jedem Tag
        :param time_res_sek: float, enthält die zeitliche Auflösung der Daten 
         in Sekunden; wird für die Regularisierung benötigt
        :param m_matrix: int, Dimension der Matrix für die Regulariserung; ent-
         spricht der Anzahl der betrachteten Zeitpunkte, zu denen Messwerte 
         vorliegen
        :param schalter: dictionary of boolean und int:
            - schalter["n_regul_variabel"]: boolean. Bei True wird 
              mit variablem n_regul gerechnet, dass n_regul_faktor* tau ist, 
              wobei tau das aktuelle tau für die Duchführung des Iterations-
              schritts ist. Bei False bleibt es fest bei dem vorgegebenen 
              n_regul, das nicht von dem aktuellen tau abhängt.
            - schalter["n_regul_faktor"]: int, ist der Faktor, mit dem das 
              aktuelle tau für die Durchführung des Iterationsschritts multi-
              pliziert werden soll. 
            - schalter["alle_messpunkte"]: boolean. Bei True werden
              in den regularisierten Wert eines Messpunkte alle vorangegangenen
              Messpunkte desselben Tages einbezogen, so dass es n_regul stets
              der Anzahl der Messpunkte des Tages abzüglich 1 entspricht. Bei 
              False wird das fest vorgegeben n_regul verwendet.
                
    Output:
        :param t_mod: vector of floats, enthält die Temperatur des Moduls zu 
         den einzelnen Zeitpunkten an der Stelle x = state
#         :dictionary of vector and Matrix
#            :t_mod, array of floats, Dimenension m (=m_matrix), enthält die Mo-
#             dultemperatur [Grad Celsius)] zu den einzelnen Zeitpunkten
#            :K, (len(t_amb),len(state))-dimensionales array of floats, enthält 
#             die (m x n)-dimensionale Jacobi-Matrix
    
    """
            
    u1 = state[0]
    u2 = state[1]
    u3 = state[2]
       
    t_mod = t_amb + g/(u1 + u2*v) +\
            u3*(np.power(g_lw/const.sigma,1./4) - 273.15 - t_amb)
        
    return t_mod

def K_temp_mod_static(state, t_amb, g, v, g_lw):
        
    """
    Berechnet die Jacobi-Matrix in Abhängigkeit vom Zustandsvektor state. Die 
    Jacobi-Matrix ist die Matrix der Ableitungen der  Modultemperatur nach dem 
    Zustandsvektor. Die Jacobi-Matrix hat die Dimension (m x n), wobei m die 
    Anzahl der Messungen und n die Anzahl der Komponenten des Zustandsvektors 
    ist.
    Input-Parameter:
        :param state, array of floats, enthält die Komponenten des Zustandsvek-
         tors tau_amb, tau_G, tau_v, u_0, u_1, u_2, u_3; zur Bedeutung der Kom-
         ponenten siehe die nachstehenden Erläuterungen zum Vorwärtsmodell
        :param t_amb, array of floats, Dimension m_matrix, enthält die Umge-
         bungstemperatur [Grad Celsius] zu den einzelnen Zeitpunkten
        :param g, array of floats, Dimension m_matrix, enthält die Strahlungs-
         flussdichten [W/m^2] bezogen auf die Ebene der Modulfläche zu den ein-
         zelnen Zeitpunkten
        :param v, array of floats, Dimension m_matrix, enthält die Windge-
         geschwindigkeiten [m/s] zu den einzelnen Zeitpunkten. Bei PV 12 stam-
         men sie aus den (zeitlich interpolierten) Cosmo-Daten und sind auf ei-
         ne Höhe von 10 m über dem Boden bezogen. Bei der Masterstation 12 wur-
         den sie in der Höhe von [] über dem Boden gemessen.   
        :param n_regul: int, Anzahl der Zeitpunkte, die im Rahmen der Regulari-
         sierung werden sollen
        :param number_days, int, enthält die Anzahl der Tage
        :param day_length: array of int, enthält die Anzahl der Messpunkte zu 
         jedem Tag
        :param time_res_sek: float, enthält die zeitliche Auflösung der Daten 
         in Sekunden; wird für die Regularisierung benötigt
        :param m_matrix: int, Dimension der Matrix für die Regulariserung; ent-
         spricht der Anzahl der betrachteten Zeitpunkte, zu denen Messwerte 
         vorliegen
        :param schalter: dictionary of boolean und int:
            - schalter["n_regul_variabel"]: boolean. Bei True wird 
              mit variablem n_regul gerechnet, dass n_regul_faktor* tau ist, 
              wobei tau das aktuelle tau für die Duchführung des Iterations-
              schritts ist. Bei False bleibt es fest bei dem vorgegebenen 
              n_regul, das nicht von dem aktuellen tau abhängt.
            - schalter["n_regul_faktor"]: int, ist der Faktor, mit dem das 
              aktuelle tau für die Durchführung des Iterationsschritts multi-
              pliziert werden soll. 
            - schalter["alle_messpunkte"]: boolean. Bei True werden
              in den regularisierten Wert eines Messpunkte alle vorangegangenen
              Messpunkte desselben Tages einbezogen, so dass es n_regul stets
              der Anzahl der Messpunkte des Tages abzüglich 1 entspricht. Bei 
              False wird das fest vorgegeben n_regul verwendet.
    Output-Parameter:
        :param  ergebnisse: dictionary, enthält
            - param ergebnisse["t_mod"] = t_mod, vector of arrays, mit dem Vor-
              wärtsmodell modellierte Modultemperatur
            - param ergebnisse["K"] = K, array of floats, (len(t_amb),len(state)) 
              Matrix, enthält die (m x n)-dimensionale Jacobi-Matrix an der 
              Stelle x = state
                
    Die Funktion setzt voraus, dass t_mod, t_amb, g, v dieselbe Anzahl an Kom-
    ponenten haben.
    
    Das zugrundeliegende Vorwärtsmodell für die Berechnung der Jacobi-MAtrix 
    ist
    
    t_mod= u_0*A_amb(tau_amb)*t_amb + u_1*A_G(tau_G)*g + u_2*A_v(tau_v)*v + u_3
    
    mit
        :param u_0, u_1, u_2, u_3, int, Modellparameter des Temperaturmodells
         aus aus TamizhMani et. al. [2003]; sie sind Komponenten des Zustands-
         vektors state
        :t_mod, array of floats, Dimenension m (=m_matrix), enthält die Modul-
         temperatur [Grad Celsius)] zu den einzelnen Zeitpunkten
        :param t_amb, s.o.
        :param g, s.o.
        :param v, s.o.
        :A_amb, A_G, A_v, (m (=m_matrix) x m (= m_matrix))-dimensionale Regula-
         risierungsmatrizen für die Umgebungstemperatur t_amb [Grad Celsius], 
         die Strahlungsflussdichte G [W/m^2] und die Windgeschwindigkeit v 
         [m/s]
        :param tau_amb, tau_G, tau_v, float, Relaxationszeiten [Größe ent-
         spricht Sekunden] hinsichtlich der Umgebungstemperatur t_amb 
         [Grad Celsius], der Strahlungsflussdichte G [W/m^2] und der Windge-
         schwindigkeit v [m/s]; sie sind Komponenten des Zustandsvektors
    
     
    Hieraus ergibt sich für die i-ten Zeilen der Jacobi-Matrix:
        [u_0*(del_A_amb_(i)/del_tau_amb)*t_amb];[u_1*(del_A_g_(i)/del_tau_g)*g]
        ;[u_2*(del_A_v_(i)/del_tau_v)*v];[(del_u_0/del_u_0)*A_amb_(i)*t_amb];
        [(del_u_1/del_u_1)*A_G_(i)*g];[(del_u_2/del_u_2)*A_v_(i)*v]; 
        [(del_u_3/del_u_3)] =
        [u_0*(del_A_amb_(i)/del_tau_amb)*t_amb];[u_1*(del_A_g_(i)/del_tau_g)*g]
        ;[u_2*(del_A_v_(i)/del_tau_v)*v];[A_amb_(i)*t_amb] [A_G_(i)*g];
        [A_v_(i)*v] [1]
        
    Bei der Berechnung der Ableitungen der Regularisierungsmatrizen A_amb, A_G
    und A_v nach der korrespondierenden Relaxationszeit tau_amb, tau_G bzw. 
    tau_v wird die Funktion differentiate_tau verwendet, die die Ableitung ei-
    ner Regularisierungsmatrix nach der Relaxationszeit tau berechnet.
        
    """    
    
    # Extrahieren der Komponenten aus dem Zustandsvektor state
    #u0 = state[0]
    u1 = state[0]
    u2 = state[1]
    u3 = state[2]        
    
    # Initialisierung der (m x n)-dimensionalen Jacobi-Matrix  mit Nullen
    K=np.zeros((len(t_amb),len(state)))
    
    t_mod = t_amb + g/(u1 + u2*v) +\
            u3*(np.power(g_lw/const.sigma,1./4) - 273.15 - t_amb)
    
    # Berechnung der Ableitungen des Vorwärtsmodells t_mod nach den Komponenten 
    # des Zustandsvektors; es handelt sich um die m-dimensionalen Spaltenvek-
    # toren der Jacobi-Matrix mit m=m_matrix=len(t_amb)
        
    #t_mod_del_u0 = t_amb_reg
    t_mod_del_u1 = -g/(u1 + u2*v)**2
    t_mod_del_u2 = -g*v/(u1 + u2*v)**2
    t_mod_del_u3 = np.power(g_lw/const.sigma,1./4) - 273.15 - t_amb
    
    # Erstellen der (m x n)-dimensionalen Jacobi-Matrix K mit m=m_matrix=
    # len(t_amb) und n=len(state)
    #K[:,0] = t_mod_del_u0 #tau_amb
    K[:,0] = t_mod_del_u1 #tau_G
    K[:,1] = t_mod_del_u2 #tau_v
    K[:,2] = t_mod_del_u3    
    #K[:,4] = t_mod_reg_del_k2
    #K[:,6] = t_mod_del_u3
    
    ergebnisse={}
    ergebnisse["t_mod"] = t_mod
    ergebnisse["K"] = K
    
    return ergebnisse






