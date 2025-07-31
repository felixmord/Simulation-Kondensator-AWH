#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 21:51:24 2025

@author: felixmord
"""

# Pakete
import math 
from math import pi
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import psychrolib
from psychrolib import (
    SI, GetHumRatioFromRelHum, GetMoistAirEnthalpy,
    GetMoistAirDensity, GetStandardAtmPressure, GetTDewPointFromRelHum, GetDryAirDensity,
    GetRelHumFromHumRatio,GetTWetBulbFromHumRatio, GetTDewPointFromHumRatio, GetSatVapPres,
    GetVapPresFromRelHum, GetHumRatioFromVapPres
)
psychrolib.SetUnitSystem(SI)    # Einheitensystem auf SI (°C, kg/kg, Pa, etc.)

from CoolProp.CoolProp import PropsSI
# 'T'	        Temperatur	                K
# 'P'	        Druck	                    Pa
# 'H'	        Enthalpie   	            J/kg
# 'S'	        Entropie	                J/kg·K
# 'L'           Wärmeleitfähigkeit          W/m*K
# 'D'	        Dichte	                    kg/m³
# 'C'	        spezifische Wärmekapazität  J/kg*K
# 'V'           dyn. Viskosität             Pa/s

from CoolProp.HumidAirProp import HAPropsSI
# 'Visc', 'M', 'mu'     dyn. Viskosität         Pa/s
# 'K', 'Conductivity'   Wärmeleitfähigkeit      W/m*K
# 'C'                   spez. Wärmekapazität    J/kg*K

import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from scipy.ndimage import uniform_filter1d



#============================================================================================================================================================================
# Definition der Matrixdimensionen 

df_input = pd.read_csv(
    "/Users/felixmord/BHT/Semester3/Messdaten Messkampagne 2 Kondensatoren/Plastikschlauch 150 um/2025-07-18 090239 Plastikschlauch 150 um Des trocken Konvektion.txt",
    sep="\t",
    skiprows=[1],
    encoding="latin1"
)


df_input["Zeit"] = df_input["Zeit"].str.strip()

# Zeit als datetime (nur Uhrzeit)
df_input["Zeit_dt"] = pd.to_datetime(df_input["Zeit"], format="%H:%M:%S").dt.time

# Zeit in Sekunden seit Beginn
startzeit = pd.to_datetime(df_input["Zeit"].iloc[0], format="%H:%M:%S")
df_input["t_des"] = pd.to_datetime(df_input["Zeit"], format="%H:%M:%S") - startzeit
df_input["t_des"] = df_input["t_des"].dt.total_seconds()



T_Ads_in_list = df_input["T_Ads_in"].values + 273.15
t_des_list = df_input["t_des"].values
RH_in_list = df_input["relM_Ads_out"].values / 100
T_in_list = df_input["T_Ads_out"].values + 273.15
AH_in_list = df_input["absM_Ads_out"].values / 100

T_amb_list = df_input["T_Amb"].values + 273.15 -1.5 # Korrektur T_amb_list -= 1.5




RH_amb_list = df_input["relM_Ads_in"].values / 100
M_W_Wippe_akt_list = df_input["M_W_Wippe_akt"].values # [g]
Mdot_W_Wippe_akt_list = df_input["Mdot_W_Wippe"].values # [g/min]
Mdot_W_Wippe_akt_list = df_input['Mdot_W_Wippe'].replace(-99999, None)
Vdot_L_list = df_input["Vdot_L"].values # [l/s]
Vdot_L_list = df_input['Vdot_L'].replace(-99999, None)

RH_out_list = df_input["relM_Kd_out"].values / 100
T_out_list = df_input["T_Kd_out"].values + 273.15
AH_out_list = df_input["absM_Kd_out"].values / 100


#============================================================================================================================================================================
# Messunsicherheiten 


# ===== Temperatur-Unsicherheiten (°C) =====

def u_T_Ads_in(T_C):
    return 0.35  # konstant

def u_T_Ads_out(T_C):
    return 0.1 + 0.0023 * T_C

def u_T_Kd_out(T_C):
    return 0.35  # konstant

def u_T_Amb(T_C):
    return 0.35  # konstant

    
df_input["u(T_Ads_in)"] = df_input["T_Ads_in"].apply(u_T_Ads_in)
df_input["u(T_Ads_out)"] = df_input["T_Ads_out"].apply(u_T_Ads_out)
df_input["u(T_Kd_out)"] = df_input["T_Kd_out"].apply(u_T_Kd_out)
df_input["u(T_Amb)"] = df_input["T_Amb"].apply(u_T_Amb)

# ===== Relative Luftfeuchte-Unsicherheiten (%) =================================================================

def u_RH1(RH):
    if 10 <= RH <= 90:
        return 0.02 * RH
    else:
        return 0.02 * RH + 0.0002 * min(abs(RH - 90), abs(10 - RH))
    
df_input["u(relM_Ads_in)"] = df_input["relM_Ads_in"].apply(u_RH1)

df_input["u(relM_Kd_out)"] = df_input["relM_Kd_out"].apply(u_RH1)

def u_RH2(RH):
    return 1.05 + 0.0084 * RH

df_input["u(relM_Ads_out)"] = df_input["relM_Ads_out"].apply(u_RH2)




# =====Absolute Luftfeuchte-Unsicherheiten (%) ======================================================================
def saturation_vapor_pressure(T):
    return 6.112 * np.exp(17.67 * T / (T + 243.5))



def d_saturation_vapor_pressure_dT(T):
    return saturation_vapor_pressure(T) * (17.67 * 243.5) / (T + 243.5)**2

def mixing_ratio(T, RH, p=1000):
    e_sat = saturation_vapor_pressure(T)  # Saturation vapor pressure
    e = RH * e_sat                        # Actual vapor pressure
    w = 0.622 * e / (p - e)               # Mixing ratio
    return w

def d_mixing_ratio_dT(T, RH, p=1000):
    e_sat = saturation_vapor_pressure(T)  # Saturation vapor pressure
    e = RH * e_sat                        # Actual vapor pressure
    # Derivative of e with respect to T
    d_e_dT = RH * d_saturation_vapor_pressure_dT(T)
    return 0.622 * (p * d_e_dT) / (p - e)**2

def d_mixing_ratio_dRH(T, RH, p=1000):
    e_sat = saturation_vapor_pressure(T)  # Saturation vapor pressure
    e = RH * e_sat                        # Actual vapor pressure
    return 0.622 * p / (p - e)**2

# absM_Ads_in
df_input["u(absM_Ads_in)"] = df_input.apply(
    lambda row: np.sqrt(
        (1000 * np.abs(row["u(relM_Ads_in)"] / 100 * d_mixing_ratio_dT(row["T_Ads_in"], row["relM_Ads_in"] / 100)))**2 +
        (1000 * np.abs(row["u(T_Ads_in)"] * d_mixing_ratio_dRH(row["T_Ads_in"], row["relM_Ads_in"] / 100)))**2
    ),
    axis=1
)

# absM_Ads_out
df_input["u(absM_Ads_out)"] = df_input.apply(
    lambda row: np.sqrt(
        (1000 * np.abs(row["u(relM_Ads_out)"] / 100 * d_mixing_ratio_dT(row["T_Ads_out"], row["relM_Ads_out"] / 100)))**2 +
        (1000 * np.abs(row["u(T_Ads_out)"] * d_mixing_ratio_dRH(row["T_Ads_out"], row["relM_Ads_out"] / 100)))**2
    ),
    axis=1
)

# absM_Kd_out
df_input["u(absM_Kd_out)"] = df_input.apply(
    lambda row: np.sqrt(
        (1000 * np.abs(row["u(relM_Kd_out)"] / 100 * d_mixing_ratio_dT(row["T_Kd_out"], row["relM_Kd_out"] / 100)))**2 +
        (1000 * np.abs(row["u(T_Kd_out)"] * d_mixing_ratio_dRH(row["T_Kd_out"], row["relM_Kd_out"] / 100)))**2
    ),
    axis=1
)

# =====Luftmassen- und Volumenstrom-Unsicherheiten (%) =====

def u_VdotL(VdotL):
    return 0.002 * VdotL
    
df_input["u(Vdot_L)"] = df_input["Vdot_L"].apply(u_VdotL)


#============================================================================================================================================================================
# Zeit in Stunden
# Nur gültige Zeilen behalten
df_plot = df_input.dropna(subset=[
    "t_des",
    "T_Ads_out", "u(T_Ads_out)",
    "T_Kd_out", "u(T_Kd_out)",
    "T_Amb", "u(T_Amb)"
])

time_hr = df_plot["t_des"] / 3600


#T_Kd,in & T_Kd,out & T_amb

# Daten als float sicherstellen
for col in ["T_Ads_out", "u(T_Ads_out)", "T_Kd_out", "u(T_Kd_out)", "T_Amb", "u(T_Amb)", "t_des"]:
    df_input[col] = pd.to_numeric(df_input[col], errors="coerce")


# Plot
plt.figure(figsize=(12, 6))

# T_Ads_out
plt.plot(time_hr, df_plot["T_Ads_out"], color="tab:blue", label=r"$T_{\mathrm{fl,Kd,in}}$")
plt.fill_between(
    time_hr,
    df_plot["T_Ads_out"] - df_plot["u(T_Ads_out)"],
    df_plot["T_Ads_out"] + df_plot["u(T_Ads_out)"],
    color="tab:blue", alpha=0.3
)

# T_Kd_out
plt.plot(time_hr, df_plot["T_Kd_out"], color="tab:orange", label=r"$T_{\mathrm{fl,Kd,out}}$")
plt.fill_between(
    time_hr,
    df_plot["T_Kd_out"] - df_plot["u(T_Kd_out)"],
    df_plot["T_Kd_out"] + df_plot["u(T_Kd_out)"],
    color="tab:orange", alpha=0.3
)

# T_Amb
plt.plot(time_hr, df_plot["T_Amb"], color="tab:green", label=r"$T_{\mathrm{amb}}$")
plt.fill_between(
    time_hr,
    df_plot["T_Amb"] - df_plot["u(T_Amb)"],
    df_plot["T_Amb"] + df_plot["u(T_Amb)"],
    color="tab:green", alpha=0.3
)

# Achsen & Layout
plt.xlabel("Zeit [h]")
plt.ylabel("Temperatur [°C]")
plt.title("Temperaturverläufe am Konndesatorein- und austritt und Ambient")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#T_Kd,in & T_Kd,out & T_amb_korr

# 2. Umrechnung zurück in °C für den Plot
T_amb_corr_C = T_amb_list - 273.15

# 3. Speichere als neue Spalte für den Plot
df_input["T_Amb_korr"] = T_amb_corr_C
T_amb_list_korr = df_input["T_Amb_korr"].values


# Daten als float sicherstellen
for col in ["T_Ads_out", "u(T_Ads_out)", "T_Kd_out", "u(T_Kd_out)", "T_Amb_korr", "u(T_Amb)", "t_des"]:
    df_input[col] = pd.to_numeric(df_input[col], errors="coerce")

# Nur gültige Zeilen behalten
df_plot = df_input.dropna(subset=[
    "t_des",
    "T_Ads_out", "u(T_Ads_out)",
    "T_Kd_out", "u(T_Kd_out)",
    "T_Amb_korr", "u(T_Amb)"
])

# Plot
plt.figure(figsize=(12, 6))

# T_Ads_out
plt.plot(time_hr, df_plot["T_Ads_out"], color="tab:blue", label=r"$T_{\mathrm{fl,Kd,in}}$")
plt.fill_between(
    time_hr,
    df_plot["T_Ads_out"] - df_plot["u(T_Ads_out)"],
    df_plot["T_Ads_out"] + df_plot["u(T_Ads_out)"],
    color="tab:blue", alpha=0.3
)

# T_Kd_out
plt.plot(time_hr, df_plot["T_Kd_out"], color="tab:orange", label=r"$T_{\mathrm{fl,Kd,out}}$")
plt.fill_between(
    time_hr,
    df_plot["T_Kd_out"] - df_plot["u(T_Kd_out)"],
    df_plot["T_Kd_out"] + df_plot["u(T_Kd_out)"],
    color="tab:orange", alpha=0.3
)

# T_Amb
plt.plot(time_hr, df_plot["T_Amb_korr"], color="tab:green", label=r"$T_{\mathrm{amb,korr}}$")
plt.fill_between(
    time_hr,
    df_plot["T_Amb_korr"] - df_plot["u(T_Amb)"],
    df_plot["T_Amb_korr"] + df_plot["u(T_Amb)"],
    color="tab:green", alpha=0.3
)

# Achsen & Layout
plt.xlabel("Zeit [h]")
plt.ylabel("Temperatur [°C]")
plt.title("Temperaturverläufe am Konndesatorein- und austritt und Ambient")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#AH_Kd,in und Kd,out


# Daten als float sicherstellen
df_input["absM_Ads_in"] = pd.to_numeric(df_input["absM_Ads_in"], errors="coerce")
df_input["u(absM_Ads_in)"] = pd.to_numeric(df_input["u(absM_Ads_in)"], errors="coerce")
df_input["absM_Ads_out"] = pd.to_numeric(df_input["absM_Ads_out"], errors="coerce")
df_input["u(absM_Ads_out)"] = pd.to_numeric(df_input["u(absM_Ads_out)"], errors="coerce")
df_input["absM_Kd_out"] = pd.to_numeric(df_input["absM_Kd_out"], errors="coerce")
df_input["u(absM_Kd_out)"] = pd.to_numeric(df_input["u(absM_Kd_out)"], errors="coerce")
df_input["t_des"] = pd.to_numeric(df_input["t_des"], errors="coerce")

# NaNs entfernen
df_plot = df_input.dropna(subset=[
    "t_des",
    "absM_Ads_in", "u(absM_Ads_in)",
    "absM_Ads_out", "u(absM_Ads_out)",
    "absM_Kd_out", "u(absM_Kd_out)"
])


# Plot vorbereiten
plt.figure(figsize=(12, 6))

# Farben
color_in = "tab:orange"
color_ads_out = "tab:green"
color_kd_out = "tab:blue"

# AH_Ads_in
plt.plot(time_hr, df_plot["absM_Ads_in"], color=color_in, label=r"$AH_{\mathrm{fl,Ads,in}}$")
plt.fill_between(
    time_hr,
    df_plot["absM_Ads_in"] - df_plot["u(absM_Ads_in)"],
    df_plot["absM_Ads_in"] + df_plot["u(absM_Ads_in)"],
    color=color_in, alpha=0.3
)

# AH_Kd,in (Ads_out)
plt.plot(time_hr, df_plot["absM_Ads_out"], color=color_ads_out, label=r"$AH_{\mathrm{fl,Kd,in}}$")
plt.fill_between(
    time_hr,
    df_plot["absM_Ads_out"] - df_plot["u(absM_Ads_out)"],
    df_plot["absM_Ads_out"] + df_plot["u(absM_Ads_out)"],
    color=color_ads_out, alpha=0.3
)

# AH_Kd,out
plt.plot(time_hr, df_plot["absM_Kd_out"], color=color_kd_out, label=r"$AH_{\mathrm{fl,Kd,out}}$")
plt.fill_between(
    time_hr,
    df_plot["absM_Kd_out"] - df_plot["u(absM_Kd_out)"],
    df_plot["absM_Kd_out"] + df_plot["u(absM_Kd_out)"],
    color=color_kd_out, alpha=0.3
)

# Layout
plt.xlabel("Zeit [h]")
plt.ylabel("Absoluter Feuchtegehalt [g/kg]")
plt.title("Absolute Feuchte an Ads_in, Kd_in und Kd_out mit Messunsicherheiten")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#RH_Kd,in und Kd,out

# Daten als float sicherstellen
df_input["relM_Ads_in"] = pd.to_numeric(df_input["relM_Ads_in"], errors="coerce")
df_input["u(relM_Ads_in)"] = pd.to_numeric(df_input["u(relM_Ads_in)"], errors="coerce")
df_input["relM_Ads_out"] = pd.to_numeric(df_input["relM_Ads_out"], errors="coerce")
df_input["u(relM_Ads_out)"] = pd.to_numeric(df_input["u(relM_Ads_out)"], errors="coerce")
df_input["relM_Kd_out"] = pd.to_numeric(df_input["relM_Kd_out"], errors="coerce")
df_input["u(relM_Kd_out)"] = pd.to_numeric(df_input["u(relM_Kd_out)"], errors="coerce")
df_input["t_des"] = pd.to_numeric(df_input["t_des"], errors="coerce")

# NaNs entfernen
df_plot = df_input.dropna(subset=[
    "t_des",
    "relM_Ads_in", "u(relM_Ads_in)",
    "relM_Ads_out", "u(relM_Ads_out)",
    "relM_Kd_out", "u(relM_Kd_out)"
])


# Plot vorbereiten
plt.figure(figsize=(12, 6))

# Farben
color_in = "tab:orange"      # Ads_in
color_ads_out = "tab:green"  # Kd_in
color_kd_out = "tab:blue"    # Kd_out

# RH_Ads,in
plt.plot(time_hr, df_plot["relM_Ads_in"], color=color_in, label=r"$RH_{\mathrm{fl,Ads,in}}$")
plt.fill_between(
    time_hr,
    df_plot["relM_Ads_in"] - df_plot["u(relM_Ads_in)"],
    df_plot["relM_Ads_in"] + df_plot["u(relM_Ads_in)"],
    color=color_in, alpha=0.3
)

# RH_Kd,in (Ads_out)
plt.plot(time_hr, df_plot["relM_Ads_out"], color=color_ads_out, label=r"$RH_{\mathrm{fl,Kd,in}}$")
plt.fill_between(
    time_hr,
    df_plot["relM_Ads_out"] - df_plot["u(relM_Ads_out)"],
    df_plot["relM_Ads_out"] + df_plot["u(relM_Ads_out)"],
    color=color_ads_out, alpha=0.3
)

# RH_Kd,out
plt.plot(time_hr, df_plot["relM_Kd_out"], color=color_kd_out, label=r"$RH_{\mathrm{fl,Kd,out}}$")
plt.fill_between(
    time_hr,
    df_plot["relM_Kd_out"] - df_plot["u(relM_Kd_out)"],
    df_plot["relM_Kd_out"] + df_plot["u(relM_Kd_out)"],
    color=color_kd_out, alpha=0.3
)

# Achsen & Layout
plt.xlabel("Zeit [h]")
plt.ylabel("Relative Luftfeuchte [%]")
plt.title("Relative Luftfeuchte an $Ads,\\mathrm{in}$, $Kd,\\mathrm{in}$ und $Kd,\\mathrm{out}$ mit Messunsicherheiten")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#============================================================================================================================================================================
x_stuetz = np.arange(1.0, -0.01, -0.01) # Stützstellen
x_stuetz_mittel = (x_stuetz[:-1] + x_stuetz[1:]) / 2    # Mittel aus 2 Stützstellen
#============================================================================================================================================================================




spaltennamen = [
    'T_fl [K]', 'T_fl_neu [K]', 'HumRatio_abschnitt[kg_d/kg_tl]', 'q [W/m^2]', 
    'Q_sens [W]', 'Q_lat [W]', 'Q [W]', 'theta_m', 'Δl_kond [m]', 
    'l_summiert [m]', 'Kondensatmenge_summiert [ml]', 'rho_fl [kg/m^3]', 
    'c_p_fl [J/kg*K]', 'eta_fl [Pa*s]', 'lambda_fl [W/m*K]', 
    'rho_H2O [kg/m^3]', 'eta_H2O [Pa*s]', 'lambda_H2O [W/m*K]', 
    'Re_kw_film', 'Re_fl_film', 'Nu_lam_film', 'Nu_turb_film', 
    'Nu_film', 'alpha [W/m^2*K]', 'k [W/m^2*K]', 'h_v [J/kg]','alpha_l_amb_fk [W/m^2*K]',
    'alpha_l_amb_ek [W/m^2*K]','alpha_l_amb_mk [W/m^2*K]','R_rw [K/W]','Kondensatmenge_abschnitt [ml]'
    ]

n_time = len(t_des_list)
n_rohrsegmente = len(x_stuetz_mittel)

result_matrizen = {name: np.zeros((n_time, n_rohrsegmente)) for name in spaltennamen}



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def simuliere_kondensator(T_in, RH_in, t_des, T_amb, RH_amb):

    # Dichte feuchter Luft (kg/m³)
    #rho_fl_in = GetMoistAirDensity(T_in-273.15, GetHumRatioFromRelHum(T_in-273.15, RH_in, 101325), 101325)
    #print(rho_fl_in)

    #Variable Parameter, messbar
    p_amb = 1.01325                 # bar
    #RH_amb = 0.3                   # *100 % (rel. Luftfeuchte Umgebung)
    #T_amb = 20 + 273.15            # K (Temperatur Umgebungsluft)
    m_punkt_fl_in = 0.046 *10**-3 * 1.2 #Annahme bei Standardbedingungen Umrechnung von l/s auf kg_fl/s * GetMoistAirDensity(T_in-273.15, GetHumRatioFromRelHum(T_in-273.15,RH_in, GetStandardAtmPressure(34)), GetStandardAtmPressure(34))   = 5.075*10**-5    # kg_fl/s (Massenstrom feuchte Luft Eintritt Kondensator 5.075*10**-5) oder 4.5675*10**-5
    u_l_amb = 0.1                   # m/s (Geschwindigkeit Luftstrom um Kondensator)
    T_kond = T_amb + 5    # K (gemittelte Kondesatortemperatur bei Desorption, ANNAHME: Kondensatortemp = Umgebungstemperatur + 5 K)

    #============================================================================================================================================================================
    # Plastikschlauch 150 um
    # Geometrie 
    L_kond = 1.715                     # m (Länge Rohr)
    d_i_kond = 0.050                   # m (Innendurchmesser)
    d_a_kond = 0.0503                 # m (Außendurchmesser)
    s_kond = 0.00015                   # m (Wandstärke)
    A_i_kond = pi*d_i_kond*L_kond      # m^2 (Mantelfläche Innen)
    A_a_kond = pi*d_a_kond*L_kond      # m^2 (Mantelfläche Außen)
    A_m_kond = (A_a_kond-A_i_kond)/math.log(A_a_kond/A_i_kond)   # m^2 (Mantelfläche gemittelt innen und außen), VDI S.24

    # Stoffwerte https://de.wikipedia.org/wiki/Polyethylen
    lambda_kond = 0.37   # W/m*K (Wärmeleitfähigkeit bei <23°C, Wärmeleitung von 1 K auf 1 m^2)

    #============================================================================================================================================================================
    # Konstanten
    c_p_tl = 1006       # J/kg*K (spez. Wärmekapazität trockene Luft -> als konstant angenommen, VDI S. 220)
    c_p_H2O = 4180      # J/kg*K (spez. Wärmekapazität flüssiges Wasser -> als konstant angenommen, VDI Tab. 1 S. 203)
    Pr_l = 0.7          # / (Prandtl-Zahl, Stoffwert-Verhältnis Pr=(eta*c_p)/Lambda), Verhältnis von Impuls- und Wäremediffusion -> als konst. im temperaturbereich 0-100 °C angenommen)
    R = 8.314462618     # J/mol*K (Allgemeine Gaskonstante)
    g = 9.80665         # m/s²  (Standard-Erdbeschleunigung nach ISO)
    R_tl = 287.058      # J/kg*K (spez. Gaskonstante trockene Luft) R_s = R/M_s
    R_d = 461.523       # J/kg*K  (spez. Gaskonstante Wasserdampf)
    p_in = p_amb        # bar (Druck feuchte Luft Eintritt Kondensator)
    p_out = p_amb       # bar (Druck feuchte Luft Austritt Kondensator)
    T_k_H2O = 647.096     # K (kritische Temperatur Wasser)
    M_H2O = 0.01801528  # kg/mol (Molare Masse von Wasser)


    #============================================================================================================================================================================
    # Variable Parameter, nicht messbar
        
    HumRatio_in = GetHumRatioFromRelHum(T_in-273.15,RH_in, GetStandardAtmPressure(34))    # kg_d/kg_tl (Feuchteverhältnis feuchte Luft Eintritt Kondensator = m_d/m_tl, Höhe Berlin üNN 34 m)
    
    def get_verdampfungsenthalpie(T_C):        # temperaturabhängige Berechnung von Verdampfungsenthalpie von Wasser
    # Parameter A-E aus VDI-Wärmeatlas S. 381
        A = 6.85307
        B = 7.43804
        C = -2.937595
        D = -3.282093
        E = 8.397378
        # Temperaturbereich in Kelvin (0 °C bis 100 °C)
        T = np.linspace(273.15, 373.15, 100)
        T_K = T_C + 273.15  # Temperatur in °C für Plot
        Tr = T_K / T_k_H2O
        one_minus_Tr = 1 - Tr
        
        # Formel (9) komplett
        hv_H2O_mol = (
            R * T_k_H2O * (
            A * one_minus_Tr**(1/3)
            + B * one_minus_Tr**(2/3)
            + C * one_minus_Tr
            + D * one_minus_Tr**2
            + E * one_minus_Tr**6
            )
        )
        hv_H2O = hv_H2O_mol / M_H2O
        return hv_H2O # J/kg


    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Trockene Luft
    def get_dichte_tl(T_K):
        rho_tl = PropsSI('D', 'T', T_K, 'P', 101325, 'Air')   # kg/m^3 (Dichte trockene Luft)
        return rho_tl
    # Ideales Gasgesetz
    #def rho_tl(T, p=101325):   
        #rho_tl = p/(R_tl*T) #kg/m^3 (Dichte trockene Luft, abhängig von T)
        #return rho_tl


    def get_waermekapazitaet_tl(T_K):
        c_p_tl = PropsSI('C', 'T', T_K, 'P', 101325, 'Air')   # J/kg*K (Wärmekapazität trockene Luft)
        return c_p_tl

    def get_dyn_viskositaet_tl(T_K):
        eta_tl = PropsSI('VISCOSITY', 'T', T_K, 'P', 101325, 'Air') # Pa*s (dyn. Viskosität trockene Luft)
        return eta_tl
    # eta_tl = 18.21e-6 # Pa*s (kin. Viskosität trockene Luft, 20°C)

    def get_kin_viskositaet_tl(T_K):
        kin_viscosity_tl = get_dyn_viskositaet_tl(T_K)/get_dichte_tl(T_K)    # m^2/s (kin. Viskosität trockene Luft)
        return kin_viscosity_tl
    # ny_tl = 153e-7 # m^2/s (dyn. Viskosität trockene Luft, 20°C)

    def get_waermeleitfaehigkeit_tl(T_K):
        lambda_tl = PropsSI('L', 'T', T_K, 'P', 101325, 'Air')    # W/(m*K) (Wärmeleitfähigkeit trockene Luft)
        return lambda_tl
    #lambda_tl = 25.87 / 1000 # W/(m*K) (Wärmeleitfähigkeit trockene Luft, 20°C)

    def get_Prandtl_tl(T_K):
        Pr_tl = PropsSI('PRANDTL', 'T', T_K, 'P', 101325, 'Air')
        return Pr_tl


    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Wasser
    def get_dichte_H2O(T_K):
        rho_H2O = PropsSI('D', 'T', T_K, 'P', 101325, 'Water')   # kg/m^3 (Dichte Wasser, rho_w = 998.21 # kg/m^3 (Normbedingungen))
        return rho_H2O

    def get_waermekapazitaet_H2O(T_K):
        c_p_H2O = PropsSI('C', 'T', T_K, 'P', 101325, 'Water')   # J/kg*K (Wärmekapazität Wasser)
        return c_p_H2O

    def get_dyn_viskositaet_H2O(T_K):
        dyn_viskositaet_H2O = PropsSI('VISCOSITY', 'T', T_K, 'P', 101325, 'Water') # Pa*s (dyn. Viskosität Wasser)
        return dyn_viskositaet_H2O
    # eta_w = 1001.6e-6 # Pa*s (dyn. Viskosität flüssiges Wasser, 20°C)

    def get_kin_viskositaet_H2O(T_K):
        kin_viskositaet_H2O = get_dyn_viskositaet_H2O(T_K)/get_dichte_H2O(T_K)    # m^2/s (kin. Viskosität Wasser)
        return kin_viskositaet_H2O

    def get_waermeleitfähigkeit_H2O(T_K):
        lambda_H2O = PropsSI('L', 'T', T_K, 'P', 101325, 'Water')    # W/(m*K) (Wärmeleitfähigkeit Wasser, lambda_w = 614.39e-3 Normbedingungen)
        return lambda_H2O

    def get_Prandtl_H2O(T_K):
        Pr_H2O = PropsSI('PRANDTL', 'T', T_K, 'P', 101325, 'Water') 
        #Pr_H2O = (get_dyn_viskositaet_H2O(T_K)*get_waermekapazität_H2O(T_K)/get_waermeleitfähigkeit_H2O(T_K))
        #(Pr_w = 5.424 # (bei 30°C))
        return Pr_H2O

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Wasserdampf
    def get_dichte_H2O_dampf(T_K):
        rho_H2O_d = PropsSI('D', 'T', T_K, 'Q', 1, 'Water')   # kg/m^3 (Dichte Wasserdampf)
        return rho_H2O_d

    def get_waermekapazitaet_H2O_dampf(T_K):
        c_p_H2O_d = PropsSI('C', 'T', T_K, 'Q', 1, 'Water') #c_p_d = 1906 # J/kg*K (Wärmekapazität Wasserdampf bei 20°C, VDI Tab. 2 S. 204)
        return c_p_H2O_d

    def get_dyn_viskositaet_H2O_dampf(T_K):
        dyn_viskositaet_H2O_d = PropsSI('VISCOSITY', 'T', T_K, 'Q', 1, 'Water') # Pa*s (dyn. Viskosität Wasserdampf)
        return dyn_viskositaet_H2O_d

    def get_kin_viskositaet_H2O_dampf(T_K):
        kin_viskositaet_H2O_d = get_dyn_viskositaet_H2O_dampf(T_K)/get_dichte_H2O_dampf(T_K)    # m^2/s (kin. Viskosität Wasserdampf)
        return kin_viskositaet_H2O_d

    def get_waermeleitfähigkeit_H2O_dampf(T_K):
        lambda_H2O_d = PropsSI('L', 'T', T_K, 'Q', 1, 'Water')    # W/(m*K) (Wärmeleitfähigkeit Wasserdampf)
        return lambda_H2O_d

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Feuchte Luft
    def get_waermekapazitaet_fl(T_K, RH):
        c_p_fl = HAPropsSI('C', 'T', T_K, 'P', 101325, 'R', RH)
        return c_p_fl

    def get_dyn_viskositaet_fl(T_K, RH):
        eta_fl = HAPropsSI('Visc', 'T', T_K, 'P', 101325, 'R', RH)
        return eta_fl

    def get_kin_viskositaet_fl(T_K, RH, P=101325):
        T_C = T_K - 273.15
        kin_viskositaet_fl = get_dyn_viskositaet_fl(T_K, RH)/GetMoistAirDensity(T_C, RH, P)    # m^2/s (kin. Viskosität Wasserdampf)
        return kin_viskositaet_fl

    def get_waermeleitfaehigkeit_fl(T_K, RH, P=101325):
        lambda_fl = HAPropsSI('K', 'T', T_K, 'P', 101325, 'R', RH)
        return lambda_fl
    
    #============================================================================================================================================================================
    # Freie Konvektion für !!!ANNAHME!!! horizontal gekrümmte Fläche (Zylinder) Umgebungsseite
    T_ref_fk = ((T_kond+T_amb)/2) 
    T_delta_fk = T_kond-T_amb                          # °K (Referenztemperatur / mittlere Fluidtemperatur freie Konvektion bei größerer Temperaturdifferenz)            
    beta_l_amb_fk = 1/(T_ref_fk)                                # 1/K (isobarer Wärmeausdehnungskoeffizient für ideales Gas)                          
    a_l_amb_fk = get_dyn_viskositaet_fl(T_amb, RH_amb)/Pr_l                              # m^2/s (Temperaturleitfähigkeit / Wärmediffusivität - räumliche Verteilung der Temperatur durch Wärmeleitung)
    Ra_l_amb_fk = (g*beta_l_amb_fk*(T_kond-T_amb)*(d_a_kond)**3)/(get_dyn_viskositaet_fl(T_amb, RH_amb)*a_l_amb_fk)  # Rayleigh-Zahl freie Konvektion
    Nu_l_amb_fk = (0.6+0.387*(Ra_l_amb_fk*0.325)**(1/6))**2                  # mittlere Nusselt-Zahl freie Konvektion
    alpha_l_amb_fk = (Nu_l_amb_fk*get_waermeleitfaehigkeit_fl(T_amb, RH_amb))/d_a_kond                  # W/m^2*K (Wärmeübergangskoeffizient freie Konvektion)

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Erzwungene Konvektion
    Re_l_amb_ek = (u_l_amb*(0.5*pi*d_a_kond))/get_dyn_viskositaet_fl(T_amb, RH_amb)          # Reynoldszahl Luft erzwungene Konvektion
    Nu_l_amb_ek_lam = 0.664*(Re_l_amb_ek**0.5)*(Pr_l**(1/3))               # mittlere Nusselt-Zahl laminarer Anteil Luft erzwungene Kovektion 
    Nu_l_amb_ek_turb = (0.037*Re_l_amb_ek**0.8*Pr_l)/(1+2.443*(Re_l_amb_ek**-0.1)*(Pr_l**(2/3)-1))  # mittlere Nusselt-Zahl turbulenter Anteil Luft erzwungene Kovektion 
    Nu_l_amb_ek = (0.3+((Nu_l_amb_ek_lam**2)+(Nu_l_amb_ek_turb**2))**0.5)*0.805     # mittlere Nusselt-Zahl Luft erzwungene Kovektion, 0.805 = Faktor für Neigungswinkel Rohr VDI S. 841
    alpha_l_amb_ek = (Nu_l_amb_ek*get_waermeleitfaehigkeit_fl(T_amb, RH_amb))/(0.5*pi*d_a_kond)     # W/m^2*K (Wärmeübergangskoeffizient Luft erzwungene Komvektion)


    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Mischkonvektion
    Nu_l_amb_mk = ((Nu_l_amb_ek**3)+(Nu_l_amb_fk**3))**(1/3)                  # mittlere Nusselt-Zahl Luft Mischkonvektion 
    alpha_l_amb_mk = (Nu_l_amb_mk*get_waermeleitfaehigkeit_fl(T_amb, RH_amb))/(0.5*pi*d_a_kond)     # W/m^2*K (Wärmeübergangskoeffizient Luft Mischkomvektion)

    #============================================================================================================================================================================
    # Wärmeleitwiderstand Rohrwand
    R_rw = s_kond/(lambda_kond*A_m_kond)        # K/W (Wärmeleitwiderstand Rohrwand)
    
    
    #============================================================================================================================================================================
    # Initialisierung
    T_fl = GetTDewPointFromRelHum(T_in-273.15, RH_in)+273.15 # T_in = Eintrittstemperatur feuchte Luft Kondensator, T_fl = Sättigungstemperatur feuchte Luft bei x_stuetz = 1
   
    delta_T1_trocken = T_in - T_amb
    delta_T2_trocken = T_fl - T_amb
    
    if delta_T1_trocken > 0 and delta_T2_trocken > 0:
        # Wenn beide Differenzen nahezu gleich sind, LMTD = delta_T1
        if np.isclose(delta_T1_trocken, delta_T2_trocken, atol=1e-4):
            theta_m_trocken = delta_T1_trocken
        else:
            try:
                theta_m_trocken = (delta_T1_trocken - delta_T2_trocken) / np.log(delta_T1_trocken / delta_T2_trocken)
            except (ZeroDivisionError, FloatingPointError, ValueError):
                theta_m_trocken = 0  # Fallback bei numerischen Fehlern
    else:
        theta_m_trocken = 0
    
        
   
    if theta_m_trocken != 0: 
        dL_trocken = (m_punkt_fl_in*get_waermekapazitaet_fl(T_in, RH_in))*(T_in-T_fl)/(alpha_l_amb_mk*pi*d_a_kond*theta_m_trocken)
    else:
        dL_trocken = 0

    
    Q_sens_trocken = m_punkt_fl_in*get_waermekapazitaet_fl(T_in, RH_in)
    
    Q_ges_trocken = Q_sens_trocken
    

    
    # Ergebnislisten
    Re_kw_film_list, Re_fl_film_list, F_list = [], [], []
    dL_kond_list, Nu_lam_film_list, Nu_turb_film_list, Nu_film_list = [], [], [], []
    alpha_film_list, k_list, Q_ges_list, d_film_list, dL_kond_list, m_kw_list = [], [], [], [], [], []
    T_fl_x_list = [T_in]  # Start mit Eintrittstemperatur
    hv_list = []           # Verdampfungsenthalpien entlang des Rohrs
    theta_m_list = []           # Temperaturdifferenz pro Abschnitt
    rho_fl_list = []
    eta_fl_list = []
    lambda_fl_list = []
    c_p_fl_list = []
    c_p_H2O_list = []
    rho_H2O_list = []
    eta_H2O_list = []
    lambda_H2O_list = []
    HumRatio_list = []
    l_kond_list = []
    q_list = []
    T_fl_profile_list = [T_fl]   # Listen speichern Temperaturverlauf T_fl und den Dampfanteil x entlang des Rohrs – für jeden Abschnitt
    Q_sens_list = []
    Q_lat_list = []
    x_neu_list = []
    
    alpha_l_amb_fk_list = []
    alpha_l_amb_ek_list = [] 
    alpha_l_amb_mk_list = [] 
    R_rw_list = []
    m_kw_n_list = []


    # Abbruchkriterium
    cum_length = dL_trocken  # Kummulierte Rohrlänge
    target_length = L_kond # Gesamtlänge des Kondensatorrohrs → wenn erreicht, bricht die Schleife ab.

    for i in range(len(x_stuetz_mittel)):

        dx = x_stuetz [i] - x_stuetz[i + 1]
        x_neu = x_stuetz[i + 1]

        T_avg = max(T_fl, 273.16)
        T_C = T_avg - 273.15

        c_p_fl = get_waermekapazitaet_fl(T_avg, RH_in)
        rho_fl = GetMoistAirDensity(T_C, RH_in, 101325)
        eta_fl = get_dyn_viskositaet_fl(T_avg, RH_in)
        lambda_fl = get_waermeleitfaehigkeit_fl(T_avg, RH_in)
        h_v = get_verdampfungsenthalpie(T_C)
        eta_H2O = get_dyn_viskositaet_H2O(T_avg)
        rho_H2O = get_dichte_H2O(T_avg)
        lambda_H2O = get_waermeleitfähigkeit_H2O(T_avg)
        Pr_H2O = get_Prandtl_H2O(T_avg)
        
        #Kondensationswärme
        Q_lat = m_punkt_fl_in * HumRatio_in * dx * h_v  # für Abschnitt

        # aktuelles HumRatio nach Teilkondensation
        HumRatio_abschnitt = x_stuetz_mittel[i] * HumRatio_in


        T_RH100 = GetTDewPointFromHumRatio(T_fl - 273.15, HumRatio_abschnitt, 101325)+273.15


        # Temperaturdifferenz für sensible Abkühlung
        delta_T_sens = T_fl - T_RH100
        Q_sens = m_punkt_fl_in * c_p_fl * delta_T_sens

        Q_ges = Q_lat + Q_sens if Q_lat is not None else Q_sens
        T_fl_neu = T_RH100

        delta_T1 = T_in - T_amb
        delta_T2 = T_fl - T_amb
            
        # Prüfen, ob beide Temperaturdifferenzen > 0 sind
        if delta_T1 > 0 and delta_T2 > 0:
            # Wenn beide Differenzen nahezu gleich sind, LMTD = delta_T1
            if np.isclose(delta_T1, delta_T2, atol=1e-4):
                theta_m = delta_T1
            else:
                try:
                    theta_m = (delta_T1 - delta_T2) / np.log(delta_T1 / delta_T2)
                except (ZeroDivisionError, FloatingPointError, ValueError):
                    theta_m = 0  # Fallback bei numerischen Fehlern
        else:
            theta_m = 0
    

        Re_kw_film = max(10**-0.5, (m_punkt_fl_in*(1-x_stuetz_mittel[i]))/(pi*d_i_kond*eta_H2O)) #!!!! Angepasst, da Re_kw_film sonst am Anfang der Simulation sehr klein und Nu_film sehr groß wird (Unrealistisch, da am Anfang kaum FIlm im Rohr)
        Re_fl_film = (4*m_punkt_fl_in*x_stuetz_mittel[i]*d_i_kond*rho_fl)/(pi*(d_i_kond**2)*eta_fl*rho_fl)
        F = (max(((2*Re_kw_film)**0.5),0.132*Re_kw_film**0.9))/(Re_fl_film**0.9) * (eta_H2O /eta_fl) * (rho_fl / rho_H2O)**0.5

        d_film = (6.59*d_i_kond*F)/((1+1400*F)**0.5)    # Annahme: Filmdicke im Rohrquerschnitt überall gleich

        Nu_lam_film = 0.693 * ((1-(rho_fl/rho_H2O))/Re_kw_film)**(1/3)
        Nu_turb_film = (0.0283*(Re_kw_film**(7/24))*(Pr_H2O**(1/3)))/(1+9.66*(Re_kw_film**(-3/8))*(Pr_H2O**(-1/6)))
    
        Nu_film = ((Nu_lam_film**2)+(Nu_turb_film)**2)**0.5
        alpha_film = (Nu_film*lambda_H2O)/(eta_H2O**2/(rho_H2O**2 * g))**(1/3)
        
        k = 1/(1/(alpha_film*A_i_kond)+R_rw+(1/(alpha_l_amb_mk*A_a_kond))) # W/(m^2*K) (Wärmedurchgangskoeffizient über alle Schichten)
        
        if k != 0 and theta_m != 0:
            dL_kond = Q_ges / (k * theta_m * pi * d_a_kond)
        else: 
            dL_kond = 0

        cum_length += dL_kond

        if dL_kond != 0:
            q = Q_ges / dL_kond*pi*d_i_kond # Wärmestromdichte
        else:
            q = 0

        dt_des = t_des_list[i+1] - t_des_list[i]
        m_kw = (1-x_stuetz_mittel[i]) * HumRatio_in * m_punkt_fl_in * dt_des * 1000
        m_kw_n = dx * HumRatio_in * m_punkt_fl_in * dt_des * 1000
        
       

        # ABBRUCHBEDINGUNG VOR WEITEREN SCHRITTEN
        if cum_length >= target_length or T_fl_neu <= T_amb: #or theta_m <= 1e-20:
            
            print(f"Abbruch in Abschnitt {i}:")
            print(f"cum_length = {cum_length:.3f} m")
            print(f"T_fl = {T_fl:.2f} K")
            print(f"T_amb = {T_amb:.2f} K")
            print(f"T_fl_neu = {T_fl_neu:.2f} K")
            print(f"T_fl - T_amb = {T_fl - T_amb:.6f} K")
            print(f"theta_m = {theta_m:.4f} K")
            print(f"cum_length = {cum_length:.4f} m")
            
            break
        else:
            T_fl = T_fl_neu
    


        # Daten sammeln
        rho_fl_list.append(rho_fl)
        eta_fl_list.append(eta_fl)
        lambda_fl_list.append(lambda_fl)
        c_p_fl_list.append(c_p_fl)
        c_p_H2O_list.append(c_p_H2O)
        rho_H2O_list.append(rho_H2O)
        eta_H2O_list.append(eta_H2O)
        lambda_H2O_list.append(lambda_H2O)
        Re_kw_film_list.append(Re_kw_film)
        Re_fl_film_list.append(Re_fl_film)
        F_list.append(F)
        d_film_list.append(d_film)
        #ny_tl_list.append(ny_tl)
        Nu_lam_film_list.append(Nu_lam_film)
        Nu_turb_film_list.append(Nu_turb_film)
        Nu_film_list.append(Nu_film)
        alpha_film_list.append(alpha_film)
        k_list.append(k)
        hv_list.append(h_v)
        Q_sens_list.append(Q_sens)
        Q_lat_list.append(Q_lat)
        Q_ges_list.append(Q_ges)
        dL_kond_list.append(dL_kond)
        m_kw_list.append(m_kw)
        T_fl_profile_list.append(T_fl_neu)
        x_neu_list.append(x_neu)
        HumRatio_list.append(HumRatio_abschnitt)
        theta_m_list.append(theta_m)
        l_kond_list.append(cum_length)
        q_list.append(q)
        
        alpha_l_amb_fk_list.append(alpha_l_amb_fk)
        alpha_l_amb_ek_list.append(alpha_l_amb_ek)
        alpha_l_amb_mk_list.append(alpha_l_amb_mk)
        R_rw_list.append(R_rw)
        m_kw_n_list.append(m_kw_n)

        # Kumulierte Rohrlänge prüfen
        target_length = 1.715
                    

    
    
                     
                     
    # Zusammenfassung

    df = pd.DataFrame({
        'x_stuetz_mittel': x_stuetz_mittel[:len(dL_kond_list)],
        'HumRatio_abschnitt[kg_d/kg_tl]' : HumRatio_list,
        'T_fl [K]': T_fl_profile_list[:-1],
        'T_fl_neu [K]': T_fl_profile_list[1:],
        'rho_fl [kg/m^3]' : rho_fl_list,
        'c_p_fl [J/kg*K]' : c_p_fl_list,
        'eta_fl [Pa*s]' : eta_fl_list,
        'lambda_fl [W/m*K]' : lambda_fl_list,
        'lambda_H2O [W/m*K]' : lambda_H2O_list,
        'rho_H2O [kg/m^3]' : rho_H2O_list,
        'c_p_H2O [J/kg*K]' : c_p_H2O_list,
        'eta_H2O [Pa*s]' : eta_H2O_list,
        'd_film [m]': d_film_list,
        'Re_kw_film': Re_kw_film_list,
        'Re_fl_film': Re_fl_film_list,
        'F': F_list,
        'Nu_lam_film': Nu_lam_film_list,
        'Nu_turb_film': Nu_turb_film_list,
        'Nu_film': Nu_film_list,
        'alpha [W/m^2*K]': alpha_film_list,
        'k [W/m^2*K]': k_list,
        'h_v [J/kg]' : hv_list,
        'Q_sens [W]': Q_sens_list,
        'Q_lat [W]': Q_lat_list,
        'Q [W]': Q_ges_list,
        'q [W/m^2]': q_list,
        'theta_m': theta_m_list,
        'Δl_kond [m]': dL_kond_list,
        'l_summiert [m]': l_kond_list,
        'Kondensatmenge_summiert [ml]': m_kw_list,
        
        'alpha_l_amb_fk [W/m^2*K]' : alpha_l_amb_fk_list,
        'alpha_l_amb_ek [W/m^2*K]' : alpha_l_amb_ek_list,
        'alpha_l_amb_mk [W/m^2*K]' : alpha_l_amb_mk_list,
        'R_rw [K/W]' : R_rw_list,
        'Kondensatmenge_abschnitt [ml]': m_kw_n_list,
        })
        
    
    return df


    # Erste Zeile mit dL_trocken einfügen
    erste_zeile = {col: None for col in df.columns}
    erste_zeile['Δl_kond [m]'] = dL_trocken
    erste_zeile['l_summiert [m]'] = dL_trocken
    erste_zeile['x_stuetz'] = '/'
    erste_zeile['T_fl [K]'] = T_in
    erste_zeile['T_fl_neu [K]'] = T_fl_profile_list[0]
    erste_zeile['Kondensatmenge_summiert [ml]'] = 0
    erste_zeile['HumRatio_abschnitt[kg_d/kg_tl]'] = HumRatio_in
    erste_zeile['Q_sens [W]'] = Q_sens_trocken
    erste_zeile['Q_lat [W]'] = 0
    erste_zeile['Q [W]'] = Q_ges_trocken
    
    # Neue Zeile in DataFrame einfügen
    df = pd.concat([pd.DataFrame([erste_zeile]), df], ignore_index=True)  

#============================================================================================================================================================================
# Ausgabe




with tqdm(total=len(t_des_list), desc="Processing", ascii=False, ncols=60) as pbar:

    for i, (T_in, RH_in, t_des, T_amb, RH_amb) in enumerate(zip(T_in_list, RH_in_list, t_des_list, T_amb_list, RH_amb_list)):
        try:
            df_result = simuliere_kondensator(T_in, RH_in, t_des, T_amb, RH_amb)
            for spalte in spaltennamen:
                werte = df_result[spalte].values
                result_matrizen[spalte][i, :len(werte)] = werte
        except Exception as e:
            print(f"Fehler bei Zeile {i}: {e}")
        pbar.update(1)


for name, matrix in result_matrizen.items():
    df = pd.DataFrame(matrix, 
                      index=t_des_list, 
                      columns=[f"x={x:.2f}" for x in x_stuetz_mittel])
    df.index.name = "Desorptionsdauer [s]"  # optional: benenne den Index
    globals()[f"df_{name.replace(' ', '_').replace('[','').replace(']','').replace('/', '')}"] = df

for name, matrix in result_matrizen.items():
    df_matrix = pd.DataFrame(matrix, columns=[f"x={x:.2f}" for x in x_stuetz_mittel])
    df_matrix["Zeit [s]"] = t_des_list

    # Dateiname aus name erzeugen (ungültige Zeichen entfernen)
    safe_name = name.replace(" ", "_").replace("[", "").replace("]", "").replace("/", "_")
    file_path = os.path.join("/Users/felixmord/BHT/Semester3/Python/Kondensator Berechnung/Simulation dynamisch/Plastikschlauch 150 um/2025-07-18 090239 Plastikschlauch 150 um Des trocken Konvektion", f"{safe_name}.xlsx")
    
    file_path1 = os.path.join("/Users/felixmord/BHT/Semester3/Python/Kondensator Berechnung/Simulation dynamisch/Plastikschlauch 150 um/2025-07-18 090239 Plastikschlauch 150 um Des trocken Konvektion/df_kondensatmenge_gesamt.xlsx")
    file_path2 = os.path.join("/Users/felixmord/BHT/Semester3/Python/Kondensator Berechnung/Simulation dynamisch/Plastikschlauch 150 um/2025-07-18 090239 Plastikschlauch 150 um Des trocken Konvektion/df_input.xlsx")
    
    df_matrix.to_excel(file_path, index=False)
    
    df_input.to_excel(file_path2, index=True)
    
# Funktion für den letzten Wert ≠ 0 von rechts
def letzter_wert_von_rechts(zeile):
    for i in reversed(range(len(zeile))):
        if zeile.iloc[i] != 0:
            return zeile.iloc[i]
    return 0


# Der letzte Kondensatwert je Zeile
kondensat_werte = df_Kondensatmenge_summiert_ml.apply(letzter_wert_von_rechts, axis=1)

# Erste Spalte (z. B. Desorptionsdauer) extrahieren
desorptionsdauer = df_Kondensatmenge_summiert_ml.iloc[:, 0]

# Zusammensetzen zu DataFrame
df_kondensatmenge_gesamt = pd.DataFrame({
    "Kondensatmenge [ml]": kondensat_werte
    
})

gesamtmenge = df_kondensatmenge_gesamt["Kondensatmenge [ml]"].sum()
print(f"Gesamtkondensatmenge: {gesamtmenge:.2f} ml")


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Simulierte Werte extrahieren

# 1. df_plot aus df_input erzeugen (nur gültige Zeilen behalten)
df_plot = df_input.dropna(subset=[
    "t_des",
    "T_Ads_out", "u(T_Ads_out)",
    "T_Kd_out", "u(T_Kd_out)",
    "T_Amb", "u(T_Amb)",
    "M_W_Wippe_akt",
])

# Nur gültige Zeilen behalten
# T_fl,Kd,out, HumRatio_fl,Kd,out
# Zeilen mit nur NaNs entfernen
df_T_fl_K_clean = df_T_fl_K[
    ~df_T_fl_K.isin([0]).all(axis=1)
].dropna(how="all")

# Zeilen mit nur 0 oder NaN aus df_HumRatio_abschnittkg_dkg_tl entfernen
df_HumRatio_clean = df_HumRatio_abschnittkg_dkg_tl[
    ~df_HumRatio_abschnittkg_dkg_tl.isin([0]).all(axis=1)
].dropna(how="all")



# Nur auf sinnvolle Zeilen anwenden
T_fl_Kd_out = df_T_fl_K_clean.apply(letzter_wert_von_rechts, axis=1)
HumRatio_fl_Kd_out = df_HumRatio_clean.apply(letzter_wert_von_rechts, axis=1)

# In °C umrechnen
T_fl_Kd_out_C = T_fl_Kd_out - 273.15
HumRatio_fl_Kd_outg = HumRatio_fl_Kd_out * 1000

# DataFrame erzeugen
df_T_fl_Kd_out_sim = pd.DataFrame({
    "T_fl_Kd_out [K]": T_fl_Kd_out,
    "T_fl_Kd_out [°C]": T_fl_Kd_out_C,
    "HumRatio_fl_Kd_out [g/kg]": HumRatio_fl_Kd_outg,
    
})




# Index zu Spalte machen und Spalte in df_T_fl_Kd_out_sim umbenennen
df_T_fl_Kd_out_sim = df_T_fl_Kd_out_sim.reset_index().rename(columns={"Desorptionsdauer [s]": "t_des"})

# Merge nach t_des
df_plot = df_plot.merge(
    df_T_fl_Kd_out_sim[["t_des", "T_fl_Kd_out [°C]", "HumRatio_fl_Kd_out [g/kg]",]],
    on="t_des",
    how="left"
)

#Glätten
# Neue Spalte vorbereiten mit NaN
df_plot["T_fl_Kd_out_glatt"] = pd.NA

# Gültige (nicht-NaN) Werte extrahieren
mask = df_plot["T_fl_Kd_out [°C]"].notna()
werte = df_plot.loc[mask, "T_fl_Kd_out [°C]"]

geglättet = uniform_filter1d(werte, size=10)
df_plot.loc[mask, "T_fl_Kd_out_glatt"] = geglättet


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#============================================================================================================================================================================
# Plots

#T_Kd,in & T_Kd,out & T_amb & T_fl,Kd,out,sim

# Daten als float sicherstellen
for col in ["T_Ads_out", "u(T_Ads_out)", "T_Kd_out", "u(T_Kd_out)", "T_Amb", "u(T_Amb)", "t_des", "T_fl_Kd_out_glatt","T_fl_Kd_out [°C]"]:
    df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce")
    
 




# Plot
plt.figure(figsize=(12, 6))

# T_Ads_out
plt.plot(time_hr, df_plot["T_Ads_out"], color="tab:blue", label=r"$T_{\mathrm{fl,Kd,in}}$")
plt.fill_between(
    time_hr,
    df_plot["T_Ads_out"] - df_plot["u(T_Ads_out)"],
    df_plot["T_Ads_out"] + df_plot["u(T_Ads_out)"],
    color="tab:blue", alpha=0.3
)

# T_Kd_out
plt.plot(time_hr, df_plot["T_Kd_out"], color="tab:orange", label=r"$T_{\mathrm{fl,Kd,out}}$")
plt.fill_between(
    time_hr,
    df_plot["T_Kd_out"] - df_plot["u(T_Kd_out)"],
    df_plot["T_Kd_out"] + df_plot["u(T_Kd_out)"],
    color="tab:orange", alpha=0.3
)

# T_Amb
plt.plot(time_hr, df_plot["T_Amb"], color="tab:green", label=r"$T_{\mathrm{amb}}$")
plt.fill_between(
    time_hr,
    df_plot["T_Amb"] - df_plot["u(T_Amb)"],
    df_plot["T_Amb"] + df_plot["u(T_Amb)"],
    color="tab:green", alpha=0.3
)



# Simulierte Temperatur plotten
plt.plot(
    time_hr,
    df_plot["T_fl_Kd_out_glatt"],
    color="tab:red",
    linestyle="--",
    linewidth=2,
    label=r"$T_{\mathrm{fl,Kd,out,sim}}$"
)

'''
# Optional: rohe (ungeglättete) Linie dünn und transparent
plt.plot(
    time_hr,
    df_plot["T_fl_Kd_out [°C]"],
    color="tab:red",
    linestyle=":",
    alpha=0.4,
    linewidth=1,
    label=r"$T_{\mathrm{fl,Kd,out,sim}}$ (roh)"
)
'''


# Achsen & Layout
plt.xlabel("Zeit [h]")
plt.ylabel("Temperatur [°C]")
plt.title("Temperaturverlauf am Kondensator")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#T_Kd,in & T_Kd,out & T_amb_korr & T_fl,Kd,out,sim

# Daten als float sicherstellen
for col in ["T_Ads_out", "u(T_Ads_out)", "T_Kd_out", "u(T_Kd_out)", "T_Amb_korr", "u(T_Amb)", "t_des", "T_fl_Kd_out_glatt","T_fl_Kd_out [°C]"]:
    df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce")
    
 




# Plot
plt.figure(figsize=(12, 6))

# T_Ads_out
plt.plot(time_hr, df_plot["T_Ads_out"], color="tab:blue", label=r"$T_{\mathrm{fl,Kd,in}}$")
plt.fill_between(
    time_hr,
    df_plot["T_Ads_out"] - df_plot["u(T_Ads_out)"],
    df_plot["T_Ads_out"] + df_plot["u(T_Ads_out)"],
    color="tab:blue", alpha=0.3
)

# T_Kd_out
plt.plot(time_hr, df_plot["T_Kd_out"], color="tab:orange", label=r"$T_{\mathrm{fl,Kd,out}}$")
plt.fill_between(
    time_hr,
    df_plot["T_Kd_out"] - df_plot["u(T_Kd_out)"],
    df_plot["T_Kd_out"] + df_plot["u(T_Kd_out)"],
    color="tab:orange", alpha=0.3
)

# T_Amb
plt.plot(time_hr, df_plot["T_Amb_korr"], color="tab:green", label=r"$T_{\mathrm{amb,korr}}$")
plt.fill_between(
    time_hr,
    df_plot["T_Amb_korr"] - df_plot["u(T_Amb)"],
    df_plot["T_Amb_korr"] + df_plot["u(T_Amb)"],
    color="tab:green", alpha=0.3
)



# Simulierte Temperatur plotten
plt.plot(
    time_hr,
    df_plot["T_fl_Kd_out_glatt"],
    color="tab:red",
    linestyle="--",
    linewidth=2,
    label=r"$T_{\mathrm{fl,Kd,out,sim}}$"
)

'''
# Optional: rohe (ungeglättete) Linie dünn und transparent
plt.plot(
    time_hr,
    df_plot["T_fl_Kd_out [°C]"],
    color="tab:red",
    linestyle=":",
    alpha=0.4,
    linewidth=1,
    label=r"$T_{\mathrm{fl,Kd,out,sim}}$ (roh)"
)
'''


# Achsen & Layout
plt.xlabel("Zeit [h]")
plt.ylabel("Temperatur [°C]")
plt.title("Temperaturverläufe am Kondensator")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#HumRatio und HumRatio simuliert

# Daten als float sicherstellen
for col in [
    "t_des", 
    "absM_Ads_in", "u(absM_Ads_in)",
    "absM_Ads_out", "u(absM_Ads_out)",
    "absM_Kd_out", "u(absM_Kd_out)",
    "HumRatio_fl_Kd_out [g/kg]"
]:
    df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce")


# Plot
plt.figure(figsize=(12, 6))


# absM_Ads_in
plt.plot(
    time_hr,
    df_plot["absM_Ads_in"],
    color="tab:blue",
    label=r"$HumRatio_{\mathrm{Ads,in}}$"
)
plt.fill_between(
    time_hr,
    df_plot["absM_Ads_in"] - df_plot["u(absM_Ads_in)"],
    df_plot["absM_Ads_in"] + df_plot["u(absM_Ads_in)"],
    color="tab:blue", alpha=0.3
)

# absM_Ads_out
plt.plot(
    time_hr,
    df_plot["absM_Ads_out"],
    color="tab:orange",
    label=r"$HumRatio_{\mathrm{Ads,out}}$"
)
plt.fill_between(
    time_hr,
    df_plot["absM_Ads_out"] - df_plot["u(absM_Ads_out)"],
    df_plot["absM_Ads_out"] + df_plot["u(absM_Ads_out)"],
    color="tab:orange", alpha=0.3
)

# absM_Kd_out
plt.plot(
    time_hr,
    df_plot["absM_Kd_out"],
    color="tab:green",
    label=r"$HumRatio_{\mathrm{Kd,out}}$"
)
plt.fill_between(
    time_hr,
    df_plot["absM_Kd_out"] - df_plot["u(absM_Kd_out)"],
    df_plot["absM_Kd_out"] + df_plot["u(absM_Kd_out)"],
    color="tab:green", alpha=0.3
)

# absolute Feuchte (simuliert)
plt.plot(
    time_hr,
    df_plot["HumRatio_fl_Kd_out [g/kg]"],
    color="tab:red",
    linestyle="--",
    linewidth=2,
    label=r"$HumRatio_{\mathrm{fl,Kd,out,sim}}$"
)


# Achsen & Layout
plt.xlabel("Zeit [h]")
plt.ylabel("Feuchtegrad [g/kg]")
plt.title("Feuchtegrad der feuchten Luft am Kondensatoraustritt")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()





#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Desorptionsdauer [h] - Kondensatmenge [ml/5 s]

# Gleitender Mittelwert
y_glatt = uniform_filter1d(df_kondensatmenge_gesamt["Kondensatmenge [ml]"], size=5)

# x-Achse in Stunden umrechnen
t_hr = df_kondensatmenge_gesamt.index / 3600

plt.figure(figsize=(8, 5))
plt.plot(t_hr, y_glatt, color="tab:blue", linewidth=2)
plt.xlabel("Desorptionsdauer [h]")
plt.ylabel("Kondensatmenge [ml/5 s]")
plt.title("Simulierte Kondensatmenge pro Intervall (5 s) in Abhängigkeit der Desorptionsdauer")

# Textbox für Gesamtkondensat
plt.text(
    x=0.95, y=0.95, 
    s=f"{gesamtmenge:.2f} ml", 
    transform=plt.gca().transAxes,
    fontsize=12,
    horizontalalignment='right',
    verticalalignment='top',
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
)
plt.grid(True)
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Kumulierte Kondensatmenge

# 1. df_plot aus df_input erzeugen (nur gültige Zeilen behalten)
df_plot = df_input.dropna(subset=[
    "t_des",
    "M_W_Wippe_akt",
])
for col in ["M_W_Wippe_akt",]:
    df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce")


df_kondensatmenge_gesamt["Kondensatmenge [ml, kumuliert]"] = df_kondensatmenge_gesamt["Kondensatmenge [ml]"].cumsum()
# Letzter Wert für die Anzeige
x_max = df_kondensatmenge_gesamt.index.max()
y_max = df_kondensatmenge_gesamt["Kondensatmenge [ml, kumuliert]"].iloc[-1]



plt.figure(figsize=(8, 5))
plt.plot(
    df_plot["t_des"],  # in Sekunden
    df_plot["M_W_Wippe_akt"],
    color="tab:blue",
    label=r"Kondensat gemessen mit Kippwaage"
)
plt.plot(
    df_kondensatmenge_gesamt.index,
    df_kondensatmenge_gesamt["Kondensatmenge [ml, kumuliert]"],
    color="tab:red",
    linestyle=":",
    linewidth=2,
    label="Simulierte Kumulierte Kondensatmenge"
)
'''
plt.fill_between(
    df_kondensatmenge_gesamt.index,
    df_kondensatmenge_gesamt["Kondensatmenge [ml, kumuliert]"],
    color="tab:red",
    alpha=0.3
)
'''
plt.text(
    x=x_max, 
    y=y_max,
    s=f"{y_max:.2f} ml",
    fontsize=12,
    ha="right",
    va="bottom",
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
)
plt.xlabel("Desorptionsdauer [s]")
plt.ylabel("Kondensatmenge [ml]")
plt.title("Kumulierte Kondensatmenge")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




# Zeitachsen in Stunden umrechnen
df_plot["t_des_hr"] = df_plot["t_des"] / 3600
df_kondensatmenge_gesamt["t_hr"] = df_kondensatmenge_gesamt.index / 3600

# Letzte Werte
x_max = df_kondensatmenge_gesamt["t_hr"].iloc[-1]
y_max_sim = df_kondensatmenge_gesamt["Kondensatmenge [ml, kumuliert]"].iloc[-1]

# Letzter gültiger (nicht NaN/0) Wert und zugehörige Zeit
mask_valid = df_plot["M_W_Wippe_akt"].notna() & (df_plot["M_W_Wippe_akt"] != 0)
x_max_mess = df_plot["t_des_hr"][mask_valid].iloc[-1]
y_max_mess = df_plot["M_W_Wippe_akt"][mask_valid].iloc[-1]

plt.figure(figsize=(8, 5))

# Gemessene Kurve
plt.plot(
    df_plot["t_des_hr"],
    df_plot["M_W_Wippe_akt"],
    color="tab:blue",
    label="Kondensat gemessen mit Kippwaage"
)

# Simulierte Kurve
plt.plot(
    df_kondensatmenge_gesamt["t_hr"],
    df_kondensatmenge_gesamt["Kondensatmenge [ml, kumuliert]"],
    color="tab:red",
    linestyle=":",
    linewidth=2,
    label="Simulierte kumulierte Kondensatmenge"
)

# Texte
plt.text(
    x=x_max,
    y=y_max_sim + 5,
    s=f"{y_max_sim:.2f} ml",
    fontsize=12,
    color="tab:red",
    ha="right",
    va="bottom",
    bbox=dict(facecolor="white", edgecolor="tab:red", boxstyle="round,pad=0.3")
)

plt.text(
    x=x_max_mess,
    y=y_max_mess - 15,
    s=f"{y_max_mess:.2f} ml",
    fontsize=12,
    color="tab:blue",
    ha="right",
    va="bottom",
    bbox=dict(facecolor="white", edgecolor="tab:blue", boxstyle="round,pad=0.3")
)

# Achsen & Layout
plt.xlabel("Desorptionsdauer [h]")
plt.ylabel("Kondensatmenge [ml]")
plt.title("Kumulierte Kondensatmenge")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# Letzter x-Wert (gemeinsam)
x_max = df_kondensatmenge_gesamt.index.max()

# Letzter Wert simuliert (rot)
y_max_sim = df_kondensatmenge_gesamt["Kondensatmenge [ml, kumuliert]"].iloc[-1]

# Letzter Wert gemessen (blau)
# Letzter gültiger (nicht NaN/0) Wert und zugehörige Zeit
mask_valid = df_plot["M_W_Wippe_akt"].notna() & (df_plot["M_W_Wippe_akt"] != 0)
x_max_mess = df_plot["t_des_hr"][mask_valid].iloc[-1]
y_max_mess = df_plot["M_W_Wippe_akt"][mask_valid].iloc[-1]


plt.figure(figsize=(8, 5))

# Gemessen
plt.plot(
    df_plot["t_des"],  
    df_plot["M_W_Wippe_akt"],
    color="tab:blue",
    label="Kondensat gemessen mit Kippwaage"
)

# Simuliert
plt.plot(
    df_kondensatmenge_gesamt.index,
    df_kondensatmenge_gesamt["Kondensatmenge [ml, kumuliert]"],
    color="tab:red",
    linestyle=":",
    linewidth=2,
    label="Simulierte kumulierte Kondensatmenge"
)

# Text simuliert
plt.text(
    x=x_max,
    y=y_max_sim + 5,
    s=f"{y_max_sim:.2f} ml",
    fontsize=12,
    color="tab:red",
    ha="right",
    va="bottom",
    bbox=dict(facecolor="white", edgecolor="tab:red", boxstyle="round,pad=0.3")
)

# Text gemessen
plt.text(
    x=x_max_mess,
    y=y_max_mess - 15,
    s=f"{y_max_mess:.2f} ml",
    fontsize=12,
    color="tab:blue",
    ha="right",
    va="bottom",
    bbox=dict(facecolor="white", edgecolor="tab:blue", boxstyle="round,pad=0.3")
)

# Layout
plt.xlabel("Desorptionsdauer [s]")
plt.ylabel("Kondensatmenge [ml]")
plt.title("Kumulierte Kondensatmenge")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
df_kondensatmenge_gesamt.to_excel(file_path1, index=False)

'''
# Desorptionsdauer [s] - Stützstellen aufsummiert -  Kondensatmenge [ml/5 s]

# Annahme: dein DataFrame heißt df_kondensatmenge_summiert_ml
df = df_Kondensatmenge_summiert_ml.copy().reset_index()  # 'Desorptionsdauer [s]' wird Spalte

# Melt: DataFrame ins lange Format bringen
df_long = df.melt(id_vars=["Desorptionsdauer [s]"], 
                  var_name="Stützstelle", 
                  value_name="Kondensatmenge [ml]")

# Extrahiere float-Wert aus Spaltennamen wie "x=0.95"
df_long["Stützstelle [m]"] = df_long["Stützstelle"].str.replace("x=", "").astype(float)

# Vorbereitung für bar3D
x = df_long["Desorptionsdauer [s]"].values
y = df_long["Stützstelle [m]"].values
z = np.zeros_like(x)
dz = df_long["Kondensatmenge [ml]"].values

# Balkenbreiten
dx = 5
dy = 0.05

# 3D-Plot
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(x, y, z, dx, dy, dz, shade=True, color="skyblue")

# Achsenbeschriftungen
ax.set_xlabel("Desorptionsdauer [s]")
ax.set_ylabel("Stützstelle [m]")
ax.set_zlabel("Kondensatmenge [ml/5 s]")

plt.title("Kondensatmenge in Abhängigkeit von Dauer und Position")
plt.tight_layout()
plt.show()
'''
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Desorptionsdauer [s] - Stützstellen aufsummiert -  Kondensatmenge [ml/5 s]






