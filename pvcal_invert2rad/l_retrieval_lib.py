"""
this lib stores all functions related to
- interpolating
- retrieving
- fitting

for the results of simulations in Libradtran
all functions of proven functionality are to be stored in and imported from this file
"""

import pandas as pd
import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

from l_plotlib import *


def linear(x, a, b):
    return a * x + b

def inv_linear(y, a, b):
    return (y - b) / a

def tau_lambda(lam , lambda_0, tau_0, alpha):
    """
    :param lam: wvl for AOD that is to be returned
    :param tau_0: float, AOD at wvl :param lambda_0
    :param alpha: Angstroem exponent
    :return: tau at
    """
    return tau_0 *(lam/lambda_0)**(-alpha)  #angstroem power law

# Check if given array is Monotonic
def isMonotonic(A):

    return (all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or
            all(A[i] >= A[i + 1] for i in range(len(A) - 1)))

def find_firstline(filepath, markers = [" ", "#", "\n"]):
    """
    returns first line of file that is valid!, any line marked with a marker at the beginning is considered invalid
    """
    f = open(filepath, "r")
    lines = f.readlines()
    for i,line in enumerate(lines):
        if line[0] in markers:
            print(line)#pass
        else:
            print(f"the first valid line is line {i}: {line}")
            break
    return i


def retrieve_AOD(df,e_meas, d_e_meas = 0, propagate_mc_noise = True, radkey = "etot"):
    """
    :param df: dataframe containing values for at least ['etot','AOD']
    :param e_meas: Measured value for The tilted Irradiance: NOTE: tilt must fit together with simulation INPUT
    :param d_e_meas: error for the Irradiance: if this is specified, output is provided including error
    :param propagate_mc_noise: (Bool) if the noise from 'd_etot' is to be propagated
    :param radkey: (string): chose between the radiation variables, given in your df and specify e_meas as the same var.
    :return: Value, Error (of the retrieved AOD)
    """
    ## find the nearest 2 values to the given Irradiance
    etot, AOD = df[radkey].to_numpy(), df["AOD"].to_numpy()
    difference = df[radkey].to_numpy() - np.full(len(df[radkey]), e_meas)
    if difference[0] > difference[1]:
        slope = "falling"
    else:
        slope = "rising"
    if isMonotonic(difference):
        print(f"difference {difference} is {slope} monotonically")
    else:
        pass  # Todo: check, if this is a proper use!
    index1 = np.argmin(abs(difference))
    if abs(difference[index1]) > max([abs(difference[i] - difference[i + 1]) for i in range(len(difference) - 1)]):
        print("the measured irradiance is out of the covered result range")
        pass
    else:
        print("the measured irradiance is in the covered range")
    if (difference[index1] >= 0 and slope == "falling") or (difference[index1] < 0 and slope == "rising"):
        index2 = index1 + 1
    else:
        index2 = index1 - 1
    print("the indices of the values in whose interval the measured irradiance lays are:", index1, index2)
    if index2 < 0 or index2 > len(etot) - 1:
        print("unfortunately this function does not provide out of boundary calculations yet")  # Todo: get in line with result range error above!
        return None , None
    else:       # Todo: understand how break, continue, pass work!
        ## use a linear sekante to find the according AOD!
        a = (etot[index1] - etot[index2]) / (AOD[index1] - AOD[index2])
        b = etot[index1] - a * AOD[index1]
        AOD_retrieved = inv_linear(e_meas, a, b)

        ## gaussian linear error propagation
        if propagate_mc_noise:
            d_etot = df["d_etot"].to_numpy()
        else:
            d_etot = np.full(len(etot), 0.0) # set to 0!

        #calculate errors
        d_a = np.sqrt((d_etot[index1] ** 2 + d_etot[index2] ** 2) / (AOD[index1] - AOD[index2]) ** 2)
        d_b = np.sqrt(d_etot[index1] ** 2 + (d_a * AOD[index1]) ** 2)  # since AOD is not contributing any error!
        d_AOD_retrieved = np.sqrt((d_e_meas ** 2 + d_b ** 2) / (a ** 2) + ((e_meas - b) * d_a) ** 2 / (a ** 4))

        # set error to none if no error was given.
        if d_e_meas == 0 and propagate_mc_noise == False:
            print("No error was propagated, set error of retrieved AOD to None")
            d_AOD_retrieved = None

        # summarize
        print("processed: mc_noise: {}, given error d_E: {}".format(propagate_mc_noise, (d_e_meas != 0)))
        print("Errors for the linear sekante between the two nearest Irradiances( E = a*AOD+b)")
        print(f"a = {a}+/- {d_a}, b = {b}+/-{d_b}")
        return AOD_retrieved, d_AOD_retrieved

def fit_retrieve_AOD(df, e_meas, d_e_meas=0, radkey="etot", Plotresult = False, width = 2):
    """
    :param df: dataframe containing the entries "AOD", "d_"+ radkey, radkey
    :param e_meas: Measured radkey erradiance
    :param d_e_meas: Measured radkey irradiance
    :param radkey: edir, edn, etot for different Irradiance Types
    :param Plotresult: Boolean: if the fit is to be plotted
    :param width: how many points to each side of the value next to e_meas are to be used
    :return: Tuple: retrieved AOD and its error as float
    Todo: do this more general for a irradiance to <parameter> dependency
    """

    ## find the nearest 2 values to the given Irradiance
    etot, AOD = df[radkey].to_numpy(), df["AOD"].to_numpy()
    d_etot = df["d_" + radkey].to_numpy()
    difference = df[radkey].to_numpy() - np.full(len(df[radkey]), e_meas)

    # check if the argument etot is in the covered range and find the nearest datapoint
    index1 = np.argmin(abs(difference))
    if abs(difference[index1]) > max([abs(difference[i] - difference[i + 1]) for i in range(len(difference) - 1)]):
        print("the measured irradiance is out of the covered result range")
        return None, None
    else:
        print("the measured irradiance is in the covered range")

    # check for ambiguity:
    if not isMonotonic(difference):
        print(f"difference {difference} is not monotonic: BREAK")
        return None, None
    else:
        print(f"difference {difference} is monotonic")

    ## retrieve by interpolation

    ## chose part from which to retrieve
    len_to_boundary = min(index1, len(AOD) - index1)
    if index1 == 0:
        fitrange = [0,2]
        print("Warning, value is very close to the boundary, the fitrange was chosen as ", fitrange)
    elif index1 == len(AOD):
        fitrange = [index1 - 2, index1]
        print("Warning, value is very close to the boundary, the fitrange was chosen as ", fitrange)
    elif len_to_boundary < width:
        width = len_to_boundary
        fitrange = [index1-width, index1 + width]
    else: fitrange = [index1-width, index1 + width]

    ## loop till you get a low chi2 or iteration breaks:
    chi2_red = 5
    while chi2_red > 4:
        print("The chosen fitrange is ", fitrange)
        AOD_choice, etot_choice = AOD[fitrange[0]:fitrange[1]], etot[fitrange[0]:fitrange[1]]
        print("this leaves the values", np.array([AOD_choice,etot_choice]).transpose())

        ## do a fit
        popt, pcov = curve_fit(linear, AOD_choice,etot_choice)
        print(popt,pcov)
        ## derive the chi 2 reduced
        chi2 = sum((etot_choice - linear(AOD_choice, *popt)) ** 2 / d_etot[fitrange[0]:fitrange[1]])
        chi2_red = chi2 / len(AOD_choice)
        print("chi2_red =", chi2_red)
        fitrange = [fitrange[0] + 1, fitrange[1] - 1]
        if fitrange[1] <= fitrange[0]:
            return None, None


    """ if chi2_red > 2:
        print("the chi2 is very high! choose another fitrange!")
        return None, None  # Todo: schneide an beiden Seiten und duchlaufe noch einmal !!"""

    ### results and error propagation:
    a, d_a, b, d_b = popt[0], np.sqrt(pcov[0,0]), popt[1], np.sqrt(pcov[1,1])
    AOD_retrieved = inv_linear(e_meas,a,b)
    d_AOD_retrieved = np.sqrt((d_e_meas ** 2 + d_b ** 2) / (a ** 2) + ((e_meas - b) * d_a) ** 2 / (a ** 4))
    print("a = {}+/- {}, b = {}+/-{}".format(a, d_a, b, d_b))

    ## plot the result
    if Plotresult:
        plt.errorbar(AOD, etot, yerr= d_etot)
        plt.plot(AOD, linear(AOD, *popt))
        plt.axhline(e_meas,0,1.4)
        plt.show()

    ## return the result!
    return AOD_retrieved, d_AOD_retrieved

## newer scripts!!
def isIncoveredrange(x,X, printout = False):
    """
    :param x: value
    :param X: array
    :return: if there is a bigger and smaller value than x in X
    """
    if (False in (x > np.array(X))) and (True in (x > np.array(X))):
        if printout: print(f"{x} is in the range covered by {X}")
        return True
    else:
        if printout: print(f"{x} is NOT in the range covered by {X}")
        return False

def slice_monotonous_part(value,array, option = None):
    """
    given an enumerable object and a value, returns the monotonous part of the object, in which the value lies
    Todo: include multiple monotonic arrays option!
    """
    # check if the value is within the covered range:
    if not isIncoveredrange(value,array):
        return None, None
    difference = np.array(array) - value
    index1 = np.argmin(abs(difference))
    mon_array = array[index1-1:index1+1]
    index0, index2 = index1 - 1, index1 + 1
    if option == "symetric":
        for i in range(min(index1, len(array) - index1)):

            new_mon_array = array[index1 - i:index1 + (i + 1)]
            if isMonotonic(new_mon_array):
                mon_array = new_mon_array
                index0, index2 = index1 - i, index1 + (i + 1)
            else: break
        return index0,index2
    else:
        j = index1
        for i in range(index1 + 1):
            new_mon_array = array[index1-i:index1]
            if isMonotonic(new_mon_array):
                mon_array = new_mon_array
                index0 = index1 - i
            else:
                j = i - 1
                break
        # cutting from the other side
        for i in range(len(array) - index1 - 1):
            new_mon_array = array[index1-j:index1+i]
            if isMonotonic(new_mon_array):
                mon_array = new_mon_array
                index2 = index1+i
            else: break
        return index0,index2

def levenberg_marquardt(X, Y,dY,y_meas,d_y_meas, threshold, min_index = None, max_index = None, show_plot_results = True):
    """
    :param X,Y,dY: (1d float-array) X-Y-dependency with uncertainity.
    :param y_meas, d_y_meas: (float) Measured y-value with uncertainity
    :param threshold: (float) how low should the chi2_red for the fit be?
    :param min_index, max_index: (int) can be used to preselect a range of fitting, which might later be reduced symmetrically
    :param show_plot_results: (bool) set true if you want to see the plotted results
    :return: (float) x with uncertainity as found by fitting a line to the x-y dependency and retaining the x belonging to the y_meas
    """
    # set default values if not given!
    if max_index == None or min_index == None:
        min_index, max_index = 0, len(Y) - 1
    chi2_red = threshold + 1
    while chi2_red > threshold and max_index > min_index +1:
        X_choice, Y_choice, dY_choice = X[min_index: max_index+ 1],Y[min_index: max_index+ 1], dY[min_index: max_index+ 1]
        popt, pcov = curve_fit(linear,X_choice,Y_choice,method="lm") # Levenberg-Marquardt
        ## derive the chi 2 reduced
        chi2 = sum((Y_choice - linear(X_choice, *popt)) ** 2 / dY_choice)
        chi2_red = chi2 / (max_index - min_index)
        #print("chi2_red =", chi2_red)
        min_index, max_index = min_index + 1, max_index - 1
    if max_index > min_index + 1: # the fit suceeded
        min_index, max_index = min_index - 1, max_index + 1
        #print(f"the fit in the index range {min_index} {max_index} succeeded!")
        a, d_a, b, d_b = popt[0], np.sqrt(pcov[0, 0]), popt[1], np.sqrt(pcov[1, 1])
        ### results and error propagation:
        x_retrieved = inv_linear(y_meas, a, b)
        d_x_retrieved = np.sqrt((d_y_meas ** 2 + d_b ** 2) / (a ** 2) + ((d_y_meas - b) * d_a) ** 2 / (a ** 4))
        ## plot the result
        if show_plot_results:
            plt.errorbar(X, Y) #, yerr=dY)
            plt.plot(X, linear(X, *popt), label = "fit")
            plt.legend(loc = "best")
            confidence_band(X,Y,dY)
            plt.axhline(y_meas)
            plt.axvline(x_retrieved)
            plt.axhspan(y_meas-d_y_meas, y_meas + d_y_meas, facecolor='b', alpha=0.2)
            plt.axvspan(x_retrieved-d_x_retrieved, x_retrieved + d_x_retrieved, facecolor='b', alpha=0.2)
            plt.xlabel("optische Dichte" + plotlabels["tau"])
            plt.ylabel("Strahlungsflussdichte" + plotlabels["F_geneigt"])
            plt.title("Bestimmung der korrespondierenden AOD zu einem gemessenen Strahlungsflussdichte")
            #### save
            from numpy import random
            rand = random.randint(10)
            plt.savefig(f"output/fit_{rand}.png")
            plt.show()
        #else: #print("no picture of the fit is shown")
        return x_retrieved,d_x_retrieved
    else:
        return None, None

def find_upper_lower_index(y0,Y, test_monotony = True):
    """
    :param y0: (float)
    :param Y: (1d float array)
    :return: the index in the monotonic array Y that is lower and upper to the given value y0
    """
    # test monotony
    if test_monotony and not isMonotonic(Y):
        return None, None
    lower_index = - 1  # lower refers to index, not y value
    if y0 > Y[0]:
        while y0 > Y[lower_index + 1]:
            lower_index = lower_index + 1
            #print(lower_index)
    else:
        while y0 < Y[lower_index + 1]:
            lower_index = lower_index + 1
            #print(lower_index)
    # upper_index_local = lower_index + 1
    return lower_index, lower_index + 1

def linear_interpolation(X,Y,dY,y_meas,d_y_meas, index1 = None, index2 = None):
    # find the upper and lower according index:
    if index1 == None or index2 == None:
        index1, index2 = find_upper_lower_index(y_meas,Y, True)
    ## use a linear sekante to find the according X!
    a = (Y[index1] - Y[index2]) / (X[index1] - X[index2])
    b = Y[index1] - a * X[index1]
    x_retrieved = inv_linear(y_meas, a, b)
    ## gaussian linear error propagation # here systematic errors can be processed!
    # calculate errors
    d_a = np.sqrt((dY[index1] ** 2 + dY[index2] ** 2) / (X[index1] - X[index2]) ** 2)
    d_b = np.sqrt(dY[index1] ** 2 + (d_a * X[index1]) ** 2)
    d_x_retrieved = np.sqrt((d_y_meas ** 2 + d_b ** 2) / (a ** 2) + ((y_meas - b) * d_a) ** 2 / (a ** 4))
    #print("a = {}+/- {}, b = {}+/-{}".format(a, d_a, b, d_b))
    return x_retrieved, d_x_retrieved

def retrieve_by_fit_or_interpolation(X,Y,dY,y_meas,d_y_meas,df = None, threshold = 4, use_fit = True, plot_fit = False, plot_upper_lower = False, out_of_bounds_calculation = True):
    """
    :param X:
    :param Y:
    :param dY:
    :param y_meas:
    :param d_y_meas:
    :param df: alternative way to give the values: in a dataframe. X, Y, dY should then be the dataframe keys:
    :return: retrieved x, uncertainity, errorcode giving the step in which process failed
    """
    ## experimental fix:
    try:
        # alternate way to give X,Y,dY
        if df != None:
            X,Y,dY = df[X].to_numpy(), df[Y].to_numpy(), df[dY].to_numpy()
        #Test 0: do the shapes match? # Todo
        # Test 1: is the y measurement in the covered range?
        if not isIncoveredrange(y_meas, Y): return None, None, None, 1
        # Test 2: is the spectrum monotonous
        if isMonotonic(Y):
            proposed_Y_slice, index1,index2 = Y, 0, len(Y) - 1
        # if not Monotonic: (3,4): is y-meas in a monotonous subrange and not ambigous?
        else:
            # Test 3: is it in a monotonous subrange?
            index1, index2 = slice_monotonous_part(y_meas, Y)
            proposed_Y_slice = Y[index1:index2]
            #print(f"3: found monotonous subrange which includes {y_meas}, between indices {index1}, {index2} \n resulting sliced array is {proposed_Y_slice}")
            # Todo: find the error in the slicing function! Note: All breaks with none, none
            # Test 4: for ambiguity:
            left, right = Y[:index1 + 1], Y[index2 - 1:]
            for side in left, right:
                index3, index4 = slice_monotonous_part(y_meas, side)
                if index3 != None:
                    #print(f"4: found monotonous subrange which includes {y_meas}, {Y[index3:index4]}")
                    return None, None, None,4
                    # Todo: implement counter counting the causes of breaking!
                    # Todo: implement ambiguity return!! x, dx returned as tuples!
        # if it is not ambigous: (
        # Information 6: which are the upper and lower indices next to the value:
        # in terms of the reduced monotonic array
        lower_index_local, upper_index_local = find_upper_lower_index(y_meas,proposed_Y_slice,False)
        # in terms of the original arrayin [np.nan, None]
        upper_index_global, lower_index_global = upper_index_local + index1, lower_index_local + index1
        # print(
        #     f"6:the index value right and left of the measured value are {lower_index_local} {upper_index_local} in terms of the monotonous subarray, \n"
        #     f"{lower_index_global} {upper_index_global} in terms of the original array, the value thus lies in the range {Y[lower_index_global:upper_index_global + 1]}")
        # # Test 7: how near to each side is the value? (in terms of indices)
        distance = min(abs(index1 - lower_index_local), abs(index2 - upper_index_local))
        min_index_global, max_index_global = lower_index_global - distance, upper_index_local + distance
        proposed_symetric_Y_slice = Y[min_index_global: max_index_global + 1]
        # print(
        #     f"7: the minimal distance to one side is {distance}, the selected upper and lower global indices are {min_index_global}, {max_index_global} \n"
        #     f"hence the selected symmetric range is {proposed_symetric_Y_slice}")

        # Step 8: Fitting using the levenberg-marquardt-algorithm! # Todo: implement yourself using rodgers!!
        if use_fit:
            x_retrieved, d_x_retrieved = levenberg_marquardt(X[min_index_global: max_index_global+ 1],Y[min_index_global: max_index_global+ 1],
                                                         dY[min_index_global: max_index_global+ 1],y_meas,d_y_meas,threshold, show_plot_results=plot_fit)
        else: x_retrieved, d_x_retrieved = None, None
        if x_retrieved not in [np.nan, None]: # means the fit succeeded
            # Do an error analysis of systematic Y-errors!!
            upper_Y, lower_Y = Y + dY, Y - dY
            upper_x, d_upper_x = levenberg_marquardt(X[min_index_global: max_index_global + 1],
                                                             upper_Y[min_index_global: max_index_global + 1], dY[min_index_global: max_index_global + 1],y_meas, d_y_meas,
                                                             threshold, show_plot_results= plot_upper_lower)
            lower_x, d_lower_x = levenberg_marquardt(X[min_index_global: max_index_global + 1],
                                                             lower_Y[min_index_global: max_index_global + 1],dY[min_index_global: max_index_global + 1], y_meas, d_y_meas,
                                                             threshold, 0, show_plot_results= plot_upper_lower)
            upper_error, lower_error = abs(x_retrieved - upper_x), abs(x_retrieved - lower_x)
            # print(f"8: the upper and lower possibilities according to upper: {upper_Y}, lower: {lower_Y} are\n"
            #       f"{upper_x}, {lower_x},\n this means for the errors:\n systematical: {upper_error},{lower_error}  statistical: {d_x_retrieved}")
            # Adding systematical and statistical errors quadratically
            if upper_x != None: d_x_up = np.sqrt(upper_error**2 + d_x_retrieved**2)
            else:
                d_x_up = abs(max(X) - x_retrieved) # Todo: Note: This might fail!!
                #print("Note: Upper boundary could not be estimated systematically!")
            if lower_x != None: d_x_down = np.sqrt(lower_error**2 + d_x_retrieved**2)
            else:
                d_x_up = abs(min(X) - x_retrieved) # Todo: Note: This might fail!!
                #print("Note: Lower boundary could not be estimated systematically!")
            errorcode = 8
        else:
            # Step 9: Interpolation
            # if distance to one side == 0: interpolation!!
            #print(f"9: since the fit did not work use linear interpolation")
            #### fix experimentally
            X_mon, Y_mon ,dY_mon = X[lower_index_global: upper_index_global + 1], Y[lower_index_global: upper_index_global + 1], dY[lower_index_global: upper_index_global + 1]
            ###
            x_retrieved, d_x_retrieved = linear_interpolation(X_mon,Y_mon, dY_mon,y_meas,d_y_meas) #, index1, index2)
            d_x_up, d_x_down, errorcode = d_x_retrieved, d_x_retrieved, 9

            # step 9,5: out of bound calculation: Note: this works only if the covered range is really covering!
            if x_retrieved in [np.nan, None] and out_of_bounds_calculation:
                #x_up_retrieved, x_up_retrieved = linear_interpolation(X_mon,Y_mon, dY_mon,y_meas + d_y_meas, 0)
                x_down_retrieved, x_down_retrieved = linear_interpolation(X_mon,Y_mon, dY_mon,y_meas - d_y_meas, 0)
                if x_up_retrieved not in [np.nan, None]: return max(Y),x_up_retrieved, max(Y),9
                if x_down_retrieved not in [np.nan, None]: return min(Y), min(Y), x_down_retrieved, 9

        ## Step 10: return the result!
        #print(f"10: the retrieved x is {x_retrieved} +- {d_x_retrieved} ")
        return x_retrieved,d_x_up,d_x_down, errorcode
        # Todo: Step11: if is ambigous; interpolate 2 times, try plotting ambiguity??
    except:
        #print("throwing exception. retrieval not working")
        return None, None, None, 10

def uncertainity_watervapour(sza,spectral_range = "pv" ,rad_key = "etot",wv_mm = 20,d_wv_mm = 5, if_printout = True):
    """
    This function uses a lookup-table from a sensitivity analysis to calculate the related uncertainity in irradiance of an uncertainity in watervapour.
    Note: this function does not destinguish between AOD's (set to 0.5 for the lookup-table) and does use GHI and no tilted irradiance!
    :param sza: solar zenith angle
    :param spectral_range:
    :param rad_key: edir, edn or etot
    :param wv_mm:
    :param d_wv_mm:
    :return:
    """
    # import the right lookuptable for the wanted error:
    if spectral_range in ["pv","300 1200"]: range = "300-1200"
    elif spectral_range in ["CMP21","300 4000"]: range = "300-4000"
    else:
        print("spectral range given is unknown. please choose another spectral range")
        return np.nan
    dic = {}
    for rad_key in ["edir","edn"]:
        dic[rad_key] = pd.read_csv("/home/johannes/Bachelor/Project/output/2020-06-25/watervapour_sensitivity_study2_{}/watervapour_sensitivity_study2_{}_{}.csv".format(range,range,rad_key))
    # interpolate in the wanted dic!
    sza_array = [0,15,30,45,60,75,90]
    upper_sza_index, lower_sza_index = find_upper_lower_index(sza, sza_array, False)
    upper_sza, lower_sza = sza_array[upper_sza_index], sza_array[lower_sza_index]
    ## readout the lookup-table
    Y = dic[rad_key]["watervapour"].to_numpy()
    if rad_key == "etot":
        X2 = dic["edir"]["sza {}".format(upper_sza)].to_numpy(dtype=float) + dic["edn"]["sza {}".format(upper_sza)].to_numpy(dtype=float)
        X1 = dic["edir"]["sza {}".format(lower_sza)].to_numpy(dtype=float) + dic["edn"]["sza {}".format(lower_sza)].to_numpy(dtype=float)
    else: X2, X1 = dic[rad_key]["sza {}".format(upper_sza)].to_numpy(dtype=float), dic[rad_key]["sza {}".format(lower_sza)].to_numpy(dtype=float)
    if if_printout: print("irradiance arrays of upper and lower sza: \n",X1,X2)
    ## interpolate two times to receive the error at one sza and interpolate between the errors according to the given sza
    F, dF1 = linear_interpolation(X1,Y,Y*0,wv_mm,d_wv_mm)
    F, dF2 = linear_interpolation(X2,Y,Y*0,wv_mm,d_wv_mm) # Todo: Note: we're only interested in the error of irradiance dF2!
    if if_printout: print("found the values for the upper/lower error: \n",dF1,dF2, "\n the according irradiances would be \n",F)
    # interpolate between the two errors according to the sza
    g = (sza - lower_sza)/(upper_sza - lower_sza)
    error_F = g*dF1 + (1-g)*dF2
    return error_F

def uncertainity(sza,spectral_range = "pv" ,variable = "watervapour", rad_key = "etot",value = None,d_value = None, if_printout = True):
    """
    This function uses a lookup-table from a sensitivity analysis to calculate the related uncertainity in irradiance of an uncertainity in a variable.
    Note: this function does not destinguish between AOD's (set to 0.5 for the lookup-table) and does use GHI and no tilted irradiance!
    :param sza: solar zenith angle
    :param spectral_range: specify the range as related to pv or pyranometer systems
    :param variable: {"watervapour", "ssa", "gg"}
    :param rad_key: edir, edn or etot
    :param value: value of the variable in the range covered by the lookuptable
    :param d_value:
    :return:
    """
    if value == None:
        defaults = { "watervapour": 20, "ssa": 0.95, "gg": 0.82, "alpha": 1.0}
        value = defaults[variable]
    if d_value == None:
        defaults_errors = { "watervapour": 5, "ssa": 0.02, "gg": 0.02, "alpha": 0.3}
        d_value = defaults_errors[variable]

    # import the right lookuptable for the wanted error:
    if spectral_range in ["pv","300 1200"]: range = "300-1200"
    elif spectral_range in ["CMP21","300 4000"]: range = "300-4000"
    else:
        print("spectral range given is unknown. please choose another spectral range")
        return np.nan
    # where the lookuptables are stored: now use the one with AOD = 0.15
    lookuptable_name = {
        #"watervapour": "/home/johannes/Bachelor/Project/output/2020-06-25/watervapour_sensitivity_study2_{}/watervapour_sensitivity_study2_{}".format(range,range),
        #"ssa": "/home/johannes/Bachelor/Project/output/2020-06-29/ssa_sensitivity_study_{}/ssa_sensitivity_study_{}".format(range,range),
        #"gg": "/home/johannes/Bachelor/Project/output/2020-06-29/gg_sensitivity_study_{}/gg_sensitivity_study_{}".format(range,range)
        "watervapour": "/home/johannes/Bachelor/Project/output/2020-06-30/watervapour_sensitivity_study_{}/watervapour_sensitivity_study_{}".format(range,range),
        "ssa": "/home/johannes/Bachelor/Project/output/2020-06-30/ssa_sensitivity_study_{}/ssa_sensitivity_study_{}".format(range,range),
        "gg": "/home/johannes/Bachelor/Project/output/2020-06-30/gg_sensitivity_study_{}/gg_sensitivity_study_{}".format(range,range),
        "alpha": "/home/johannes/Bachelor/Project/output/2020-07-06/alpha_sensitivity_study_{}/alpha_sensitivity_study_{}".format(range,range)
    }
    dic = {}
    for rad_key in ["edir","edn"]:
        dic[rad_key] = pd.read_csv("{}_{}.csv".format(lookuptable_name[variable],rad_key))
    # interpolate in the wanted dic!
    sza_array = [0,15,30,45,60,75,90]
    upper_sza_index, lower_sza_index = find_upper_lower_index(sza, sza_array, False)
    upper_sza, lower_sza = sza_array[upper_sza_index], sza_array[lower_sza_index]
    ## readout the lookup-table
    Y = dic[rad_key][variable].to_numpy()
    if rad_key == "etot":
        X2 = dic["edir"]["sza {}".format(upper_sza)].to_numpy() + dic["edn"]["sza {}".format(upper_sza)].to_numpy()
        X1 = dic["edir"]["sza {}".format(lower_sza)].to_numpy() + dic["edn"]["sza {}".format(lower_sza)].to_numpy()
    else: X2, X1 = dic[rad_key]["sza {}".format(upper_sza)].to_numpy(), dic[rad_key]["sza {}".format(lower_sza)].to_numpy()
    if if_printout: print("irradiance arrays of upper and lower sza: \n",X1,X2)
    ## interpolate two times to receive the error at one sza and interpolate between the errors according to the given sza
    F, dF1 = linear_interpolation(X1,Y,Y*0,value,d_value)
    F, dF2 = linear_interpolation(X2,Y,Y*0,value,d_value) # Todo: Note: we're only interested in the error of irradiance dF2!
    if if_printout: print("found the values for the upper/lower error: \n",dF1,dF2, "\n the according irradiances would be \n",F)
    # interpolate between the two errors according to the sza
    g = (sza - lower_sza)/(upper_sza - lower_sza)
    error_F = g*dF1 + (1-g)*dF2
    return error_F

def combine_arrays_between(X,Y1,Y2,x1,x2):
    """
    :return: the combination of Y1 and Y2, weighting them according to their position in the interval x1, x2.
    If this interval does not cover all of X, the lower part is just Y1, the upper Y2
    """
    if x1 < x2 and x1 in X and x2 in X:
        pass
    else: print("x1,x2,X given are invalid")
    return 0

def upper_lower_angle_boundaries(df_angles, dates,max_solar_angle = 80):
    """
    :param angle_csv_name: name of the csv storing the angles
    :param dates: pd daterange for one day!
    :param max_solar_angle: angle you want to limit the data to
    :return: lower, upper boundary given as time as float
    """
    df_angles = df_angles[df_angles["time"].isin([str(date) for date in dates])]
    invalid_timesteps = df_angles[df_angles["sza"] > max_solar_angle]["time"].to_numpy()
    lower_invalid_timesteps, upper_invalid_timesteps = [],[]
    for i,timestep in enumerate(dates):
        if str(timestep) in invalid_timesteps:
            if timestep.hour > 12: upper_invalid_timesteps.append(timestep)
            else: lower_invalid_timesteps.append(timestep)
    if lower_invalid_timesteps != []: t0 = (lower_invalid_timesteps[-1].hour + lower_invalid_timesteps[-1].minute/60)
    else: t0 = dates[0].hour + dates[0].minute/60
    if upper_invalid_timesteps != []: t1 = (upper_invalid_timesteps[0].hour + upper_invalid_timesteps[0].minute/60)
    else: t1 = dates[-1].hour + dates[-1].minute/60
    print("the invalid timesteps due to high sza (sza<80) are:\n", upper_invalid_timesteps,lower_invalid_timesteps,
          "\n hence the lower / upper boundary of the usable timespectrum are: \n",t0,t1)
    return t0, t1

def measurement_selected(stationcode, source, time_interval, rad_key="etot", if_printout = "True"):
    """
    :param stationcode: PV_XX, MS_XX as used in the stationcode-dictionary in the simulation lib.
    :param source: pv, CMP21 or pyr
    :param time_interval: as pd.daterange: Range in which you want the data: so to speak you give x for your y.
    :param rad_key: only relevant for shadowband pyranometer: ["edir", "edn", "etot","etilt"] for direct, diffusive downwards
    total irradiance and tilted total irradiance
    :return: a dataframe with three columns ["timestamp","irradiance","d_irradiance"]
    Note: for changes in the respective Input-files/their saving structure edit this function to fit your needs
    """
    measurement_path = "/home/johannes/Bachelor/Project/input/measurement_campaign/"
    path_dic = {"pv": "pv_inversion/", "CMP21": "", "pyr": ""}
    rad_key_label = {"edir": "Edir", "etot": "Etotdown", "edn": "Ediffdown", "etilt": "Etotpoa"}
    # error treatment for different sources: here: pyr: 5%, min 20W/m² , CMP21: 2% min 10W/m²
    errors = {"pyr": [0.05, 20], "CMP21": [0.02, 10], "pv" : [0.05,20]}
    if source == "pv":
        df_preliminary = pd.read_csv(
            "{}inversion_results_atm_asl_all_messkampagne_2018_clear_sky_messkampagne_2019_all_sky_disortres_5_5_{}.dat".format(
                measurement_path + path_dic[source], stationcode),
            sep=";", header=5)
        df = pd.DataFrame({"timestamp": df_preliminary["variable"]})
        if if_printout: print("measurement dataframe before time selection\n", df)
        df_preliminary = df_preliminary[df_preliminary["variable"].isin([str(date) for date in time_interval])]
        if if_printout: print("measurement dataframe \n", df_preliminary)
        df["irradiance"] = df_preliminary["Etotpoa_pv_inv"].to_numpy().astype(float)
        df["d_irradiance"] = guess_uncertainity(df["irradiance"].to_numpy(), fraction=0.05,min=10)
    elif source in ["CMP21", "pyr"]:
        given_header = ["timestamp", "irradiance"]
        df_preliminary = pd.read_csv(
            f"{measurement_path}irrad_messkampagne_{time_interval[0].year}_{stationcode}_15min.dat", sep=";",
            skiprows=[0, 2])
        if if_printout: print(df_preliminary.head())
        df = pd.DataFrame({"timestamp": df_preliminary["variable"]})
        fractional_error, min_error = errors[source][0], errors[source][1]
        if rad_key == "edir":
            df["irradiance"] = df_preliminary[f"Etotdown_{source}_Wm2"].to_numpy() - df_preliminary[
                f"Ediffdown_{source}_Wm2"].to_numpy()
            fractional_error, min_error = fractional_error * np.sqrt(2), min_error * np.sqrt(2)
        else:
            df["irradiance"] = df_preliminary[f"{rad_key_label[rad_key]}_{source}_Wm2"].to_numpy()
        # select time
        df = df[df["timestamp"].isin([str(time) for time in time_interval])]
        # error treatment
        df["d_irradiance"] = guess_uncertainity(df["irradiance"].to_numpy(), fraction=fractional_error, min=min_error)
    return df