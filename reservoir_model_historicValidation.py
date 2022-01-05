# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 19:42:31 2020

@author: Robyn
"""

# ****Elephant Butte Reservoir Release Optimization Project**** #

# Portions of code have been adapted from a simulation model
# written by Nolan Townsend, Luis Garnica Chavira, and Robyn Holmes
# under the supervision of Alex Mayer.

#import sys
#sys.modules[__name__].__dict__.clear()
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import matplotlib as mpl
from joypy import joyplot

#******************************************MODEL START****************************************************#
#*********************************************************************************************************#

#***********************LOAD INPUT***********************#
# Load Excel File
fileName = 'Inputs\\input_parameters_historicValidation.xlsx'   # File name

# Load each sheet as a pandas dataframe
Scalars = pd.read_excel(fileName, sheet_name='Scalars', index_col=0)
SanMar_cfs = pd.read_excel(fileName, sheet_name='SanMar_cfs', index_col=0)
CabPr_mmday = pd.read_excel(fileName, sheet_name='CabPr_mmday', index_col=0)
CabTas_degC = pd.read_excel(fileName, sheet_name='CabTas_degC', index_col=0)
EBPr_mmday = pd.read_excel(fileName, sheet_name='EBPr_mmday', index_col=0)
EBTas_degC = pd.read_excel(fileName, sheet_name='EBTas_degC', index_col=0)

# Convert units
SanMar_af = SanMar_cfs*724.44792344617  # Convert CFS to af/year
CabPr_ft = CabPr_mmday*1.198302165      # Convert mm/day to ft/year
EBPr_ft = EBPr_mmday*1.198302165        # Convert mm/day to ft/year
#***********************Extract Scalars***********************#
# Elephant Butte Reservoir
EBInitStorage_af = Scalars.at['EBInitStorage_af', 'Value']          # Initial Storage Volume: 415000 acre-feet
EBMin = Scalars.at['EBMin', 'Value']                  # EB Minimum Storage Volume: 17300 acre-feet
EBMax = Scalars.at['EBMax', 'Value']                  # EB Maximum Storage Volume: 1.99E+06 acre-feet

EBA0 = Scalars.at['EBA0', 'Value']                    # Hypsometric coefficient values for calc Elephant Butte surface area
EBA1 = Scalars.at['EBA1', 'Value']
EBA2 = Scalars.at['EBA2', 'Value']
EBA3 = Scalars.at['EBA3', 'Value']
EBA4 = Scalars.at['EBA4', 'Value']

# Caballo Reservoir
CabInitStorage_af = Scalars.at['CabInitStorage_af', 'Value']       # Caballo Initial Storage Volume

CabA0 = Scalars.at['CabA0', 'Value']               # Hypsometric coefficient values for calc Caballo surface area
CabA1 = Scalars.at['CabA1', 'Value']
CabA2 = Scalars.at['CabA2', 'Value']
CabA3 = Scalars.at['CabA3', 'Value']
CabA4 = Scalars.at['CabA4', 'Value']

#Operating Constants
OPConst1 = Scalars.at['OPConst1', 'Value']             # Constants for calculating desired release volume
OPConst2 = Scalars.at['OPConst2', 'Value']
OPConst3 = Scalars.at['OPConst3', 'Value']
FullAllocation_af = Scalars.at['FullAllocation', 'Value']  # Average historical full allocation baseline over the whole region

#Other coefficients
RunoffCoeff = Scalars.at['RunoffCoeff', 'Value']        # Runoff Coefficient

EvapCoeffA = 30#Scalars.at['EvapCoeff', 'Value']           # Evaporation Coefficient A
EvapCoeffB = Scalars.at['EvapInt', 'Value']             # Evaporation Coefficient B

HistoricTas = Scalars.at['HistoricTas', 'Value']        # Average temperature over historical period (from temp-evap regression)
HistoricEvap = Scalars.at['HistoricEvap', 'Value']      # Average evaperation over historical period (from temp-evap regression)

CabLandArea_ac = Scalars.at['CabLandArea', 'Value']     # Caballo subwatershed land area (acres)
EBLandArea_ac = Scalars.at['EBLandArea', 'Value']       # Elephant Butte subwatershed land area (acres)

#Timestep Info
StartYear = int(Scalars.at['StartYear', 'Value'])       # Simulation start year
EndYear = int( Scalars.at['EndYear', 'Value'])          # Simulation end year
Years = []                                              # Initialize simulation year array
for x in range(StartYear, EndYear+1):
    Years.append(x)
    
#***********************Extract Climate Scenario Info***********************#
#Scenario Info
ClmSimNames = SanMar_af.columns.values.tolist()   # Creates a list of all climate scenarios

# Check all senario names match
CSN2 = SanMar_af.columns.values.tolist()
CSN3 = SanMar_af.columns.values.tolist()
CSN4 = SanMar_af.columns.values.tolist()
CSN5 = SanMar_af.columns.values.tolist()

if ClmSimNames==CSN2==CSN3==CSN4==CSN5:
    pass
else:
    print('Check that climate scenario names match accross each timeseries!')

#Initalize variables
EBStoragePrevYear = 0      
Epsilon = .000001
MaxIt = 1000


#***********************Initialize Dataframes (df) for Variable Storage****************************************#
EB_Storage_af = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)         # Elephant Butte Storage (acre-feet)
EB_SurfA_ac = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)           # EB Surface area (acres)
Cab_Storage_af = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)        # Caballo storage average (acre-feet)
Cab_SurfA_ac = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)          # Caballo Surface Area (acres)
Res_SEvap_af = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)          # Reservoir Surface Evaporation
Res_SPrecip_af = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)        # Reservoir Direct Precipitation (af), Caballo + Elephant Butte
Excess_af = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)             # Excess (water that must be released to keep reservoir volume within capacity, not allocated to a user)
Cab_Gauge_af = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)          # Yearly Caballo Gauge (acre-feet) (allocated releases)
Cab_RO_af = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)             # Caballo Runoff (acre-feet)
DCab_Gauge_af = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)         # Desired Caballo Gauge (acre-feet)
Cab_Release_af = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)        # Gauge Flow at Caballo; = Caballo_Realease_af + Excess (acre-feet)
WBal_Check = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)            # checking if waterbalance = 0
SW_Evap_ft = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)            # Surface water evaporation depth (feet)
Allocation_Diff = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)       # Allocation Difference; = Cab_Gauge_af - DCab_Gauge_af
Net_Local = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)             # Net local reservoir flows; = P + RO - E

#***********************RESERVOIR OPERATION MODEL****************************************#    
for ClmSim in ClmSimNames:

    for Year in Years:
        EB_Storage_current_it = 0       # Initialize/reset
        EB_Storage_Diff = 999999999     # Initialize/reset

        if Year == StartYear:           # First year uses initial storage volume
            EBStoragePrevYear = EBInitStorage_af
        else:                           # After first year, use previous year's storage volume
            EBStoragePrevYear = EB_Storage_af.at[Year-1, ClmSim]

        DCab_Gauge_af.at[Year, ClmSim] = min(OPConst1, (OPConst2*SanMar_af.at[Year, ClmSim]+OPConst3*EBStoragePrevYear))  # Desired Caballo Gauge (af)

        Cab_Storage_af.at[Year, ClmSim] = (CabInitStorage_af * 0.8859090221627) - 7436.780267639        # Caballo storage average (af) (note: ASSUMED CONSTANT)
        Cab_SurfA_ac.at[Year, ClmSim] = max(0, CabA0 + (CabA1*Cab_Storage_af.at[Year, ClmSim]) + (CabA2*(Cab_Storage_af.at[Year, ClmSim]**2)) + (CabA3*(Cab_Storage_af.at[Year, ClmSim]**3))+(CabA4*(Cab_Storage_af.at[Year, ClmSim]**4))) # Caballo surface area (ac) (note: ASSUMED CONSTANT)
  
        SW_Evap_ft.at[Year,ClmSim] = (HistoricEvap + EvapCoeffA*(EBTas_degC.at[Year, ClmSim]-HistoricTas) + EvapCoeffB)/304.8    # Evaporation depth per year (304.8 converts mm to feet)

        # Iterate to solve for EB storage
        it = 0
        while((EB_Storage_Diff > Epsilon) and (it < MaxIt)):
            if it == 0:
                EB_Storage_current_it = EBStoragePrevYear  # If first iteration, use prev. year's storage volume as first guess
            EB_Storage_prev_it = EB_Storage_current_it
            
            EB_SurfA_ac.at[Year, ClmSim] = max(0, EBA0 + (EBA1*EB_Storage_prev_it) + (EBA2*(EB_Storage_prev_it**2)) + (EBA3*(EB_Storage_prev_it**3))+(EBA4*(EB_Storage_prev_it**4)))
            
            Res_SPrecip_af.at[Year, ClmSim] = CabPr_ft.at[Year, ClmSim] * Cab_SurfA_ac.at[Year, ClmSim] + EBPr_ft.at[Year,ClmSim] * EB_SurfA_ac.at[Year, ClmSim]     # Reservoir Surface Precip (AF) 
            Res_SEvap_af.at[Year, ClmSim] =  SW_Evap_ft.at[Year,ClmSim] * (Cab_SurfA_ac.at[Year, ClmSim] + EB_SurfA_ac.at[Year, ClmSim])  # Reservoir Surface Evap (AF)
            
            Cab_RO_af.at[Year, ClmSim] = RunoffCoeff * ((CabLandArea_ac-Cab_SurfA_ac.at[Year,ClmSim])*CabPr_ft.at[Year,ClmSim] + (EBLandArea_ac-EB_SurfA_ac.at[Year,ClmSim])*EBPr_ft.at[Year,ClmSim])
            
            #Reservoir Operation (Solve for Caballo Release)
            EB_Storage_current_it = EBStoragePrevYear + SanMar_af.at[Year, ClmSim] + Cab_RO_af.at[Year, ClmSim] + Res_SPrecip_af.at[Year, ClmSim] - Res_SEvap_af.at[Year, ClmSim] - DCab_Gauge_af.at[Year, ClmSim]   # Volume of EB after desired release volume released
            if(EB_Storage_current_it < EBMax and EB_Storage_current_it > EBMin):              # If EB is within upper and lower limits, release DCab, no Excess
                Excess_af.at[Year, ClmSim] = 0
                Cab_Gauge_af.at[Year, ClmSim] = DCab_Gauge_af.at[Year, ClmSim]
            elif(EB_Storage_current_it < EBMin):      # If releasing DCab puts EB below min V, no Excess, release to min V
                Excess_af.at[Year, ClmSim] = 0
                Cab_Gauge_af.at[Year, ClmSim] = EBStoragePrevYear + SanMar_af.at[Year, ClmSim] + Cab_RO_af.at[Year, ClmSim] + Res_SPrecip_af.at[Year, ClmSim] - Res_SEvap_af.at[Year, ClmSim] - EBMin  # Release what came in/release what you can
            else:                            # If EB is above max capacity calculate excess water that needs to be released
                Excess_af.at[Year, ClmSim] = EB_Storage_current_it - EBMax
                Cab_Gauge_af.at[Year, ClmSim] = DCab_Gauge_af.at[Year, ClmSim]    
            Cab_Release_af.at[Year, ClmSim] = Cab_Gauge_af.at[Year, ClmSim] + Excess_af.at[Year, ClmSim]
            
            EB_Storage_current_it = EBStoragePrevYear + SanMar_af.at[Year, ClmSim] + Cab_RO_af.at[Year, ClmSim] - Res_SEvap_af.at[Year, ClmSim] + Res_SPrecip_af.at[Year, ClmSim] - Cab_Release_af.at[Year, ClmSim] # Reservoir volume, post operation
            EB_Storage_Diff = abs(EB_Storage_prev_it - EB_Storage_current_it)       # Difference between previous and current iteration's storage
    
            it+=1
        # End iteration that solves for EB storage
        
        EB_Storage_af.at[Year, ClmSim] = EB_Storage_current_it       # Store EB Yearly Storage in dataframe
        Allocation_Diff.at[Year,ClmSim] = DCab_Gauge_af.at[Year,ClmSim] - Cab_Gauge_af.at[Year,ClmSim]  # Difference between desired Caballo release and actual
        
        # Double Check Water Balance
        deltaS = 0
        if Year == StartYear:
            deltaS = EB_Storage_af.at[Year,ClmSim] - EBInitStorage_af
        else:
            deltaS = EB_Storage_af.at[Year,ClmSim] - EB_Storage_af.at[Year-1,ClmSim]
        WBal_Check.at[Year,ClmSim] = SanMar_af.at[Year, ClmSim] + Cab_RO_af.at[Year, ClmSim] - Res_SEvap_af.at[Year, ClmSim] + Res_SPrecip_af.at[Year, ClmSim] - Cab_Release_af.at[Year, ClmSim] - deltaS # Reservoir WB = inflows - outflows - delta S

    print('Completed:'+ClmSim)
    # End Year iteration
# End Climate Scenario iteration

Net_Local = Cab_RO_af.sub(Res_SEvap_af).add(Res_SPrecip_af)     # Reservoir contributions from the local subwatersheds
local_net_depth = EBPr_ft.sub(SW_Evap_ft)
#***********************Export data to excel****************************************# 
writer = pd.ExcelWriter('ResModelOutput_validation.xlsx')

Scalars.to_excel(writer, "Scalar Variables")
EB_Storage_af.to_excel(writer, 'EB_Storage_af')
EB_SurfA_ac.to_excel(writer, 'EB_SurfA_ac')
Cab_Storage_af.to_excel(writer, 'Cab_Storage_af')
Cab_SurfA_ac.to_excel(writer, 'Cab_SurfA_ac')
Res_SEvap_af.to_excel(writer, 'Res_SEvap_af')
Res_SPrecip_af.to_excel(writer, 'Res_SPrecip_af')
Cab_RO_af.to_excel(writer, 'Cab_RO_af')
Excess_af.to_excel(writer, 'Excess_af')
Cab_Gauge_af.to_excel(writer, 'Cab_Gauge_af')
DCab_Gauge_af.to_excel(writer, 'DCab_Gauge_af')
Cab_Release_af.to_excel(writer, 'Cab_Release_af')
SanMar_af.to_excel(writer, 'SanMar_af')
WBal_Check.to_excel(writer, 'WBal_Check')
SW_Evap_ft.to_excel(writer,'SW_Evap_ft')
Allocation_Diff.to_excel(writer, 'Allocation_Diff')
Net_Local.to_excel(writer, 'Net Local')
writer.save()

#***********************Statistics & Plotting Functions****************************************#
# Change plot style by uncommenting:
plt.style.use('tableau-colorblind10')
sns.set_context('notebook')    # poster, talk, notebook
# plt.style.use('ggplot')
# plt.style.use('seaborn-ticks')
# plt.style.use('default')

def RCP_df(allSims_df,rcp):
    '''
    Input a dataframe with simulation name as column index and rcp identifier (26,45,60, or 85)
    Returns a new dataframe containing the simulations of the specified RCP
    ''' 
    rcp=str(rcp)
    ClmSimNames = allSims_df.columns.values.tolist()
    rcpSimNames = []
    
    for Name in ClmSimNames:
        if Name.endswith(rcp):
            rcpSimNames.append(Name)
            
    new_df = allSims_df.filter(rcpSimNames,axis=1)
    
    return new_df
 
def RCPcol(df):
    '''
    Input a dataframe (melted format)
    Returns a dataframe with a column added indicating the rcp

    '''
    
    rcps = {'rcp26':'RCP2.6','rcp45':'RCP4.5','rcp60':'RCP6.0','rcp85':'RCP8.5'}
    #add empty row
    df['RCP']=''
    #get list of index
    idx = df.index.values.tolist()

    for i in idx:
        for j in rcps.keys():
            if df.at[i,'Simulation'].endswith(j):
                df.at[i,'RCP'] = rcps.get(j)
    return(df)

def sortRCP(df):
    '''
    Input a dataframe (model output format)
    Return a dataframe with columns sorted by RCP 
    '''
    
    rcp26=RCP_df(df,26)
    rcp45=RCP_df(df,45)
    rcp60=RCP_df(df,60)
    rcp85=RCP_df(df,85)
    
    sortedRCP = rcp26.join(rcp45).join(rcp60).join(rcp85)
    
    return sortedRCP

def resStats(release_df, startYr=2021,endYr=2070, pctList=[.5,1],clist=['#8c510a','#d8b365','#c7eae5','#5ab4ac','#01665e']):
    '''
    Input a dataframe of reservoir releases from the model (start and end year optional)
    Outputs a table of reservoir statistics

    '''
    DesiredRelease = 875000
    nYears = endYr-startYr+1 # number of years considered
    TotDesRelease = DesiredRelease*nYears
    
    release_df = release_df.loc[startYr:endYr,:]    #slice df to only include desired years
    
    ClmSimNames = release_df.columns.values.tolist()    #create a list of climate models
    ColNames = pctList
    stats_df = pd.DataFrame(index=ClmSimNames, columns=ColNames, dtype=np.float64)          #create new/empty df w scenarios as index


    #add col w fraction of years w full allocation released
    for i in range(len(pctList)):
        pctAlloc = pctList[i]*DesiredRelease
        for Scn in ClmSimNames:
            nAllocMet = release_df.apply(lambda x: True if x[Scn] >= pctAlloc else False, axis=1)
            stats_df.at[Scn,ColNames[i]]=len(nAllocMet[nAllocMet == True].index)/nYears*100   #count # of years below DesiredRelease
        ColNames[i] = 'Percent of years >='+str(pctList[i]*100)+'% of Full Allocation is Released' 
    stats_df.columns = ColNames
    
    stats_df.plot.bar(rot=90,subplots=True,legend=None,color=clist,ylim=[0,100])
    plt.xlabel('Simulation Name')
    
    avg = stats_df.mean(axis=0)
    print(avg)
    #plt.ylabel('text')

        
def varOverTimePlt(var_df,plot_title,x_label,y_label,startYr=1960,endYr=2099):
    '''
    Input a dataframe from the model, plot labels, and start/end years (opt)
    Plots all simulations of a variable over time

    '''

    var_df = var_df.loc[startYr:endYr,:]
    
    var_df.plot(legend=False)
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

def pctOverTimePlt(var_df,plot_title,x_label,y_label,pctList=(0.05,0.25,0.5,0.75,0.95), startYr=1960,endYr=2099):
    '''
    Input a dataframe (model output format), plot labels, list of percentiles to plot(opt) and start/end years (opt)
    Plots percentile lines representing all simulations of a variable over time

    '''

    pctListTxt = []
    for pct in pctList:   
        pct *= 100
        pctTxt = str(int(pct))+'%'
        pctListTxt.append(pctTxt)
        
    # Generate Dataframe of Percentile Timeseries 
    var_df = var_df.loc[startYr:endYr,:]
    var_pct = var_df.transpose().describe(percentiles=pctList)  
    
    # Create Plot    
    var_pctT=var_pct.transpose()
    var_pctT.plot(kind='line', y=pctListTxt, use_index=True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    
    #add fill between top and bottom percentile
    last = len(pctListTxt)-1
    lower = var_pctT[pctListTxt[0]].values
    upper = var_pctT[pctListTxt[last]].values
    plt.fill_between(var_pctT.index, lower, upper, color= 'lightgrey')
    
def perChangePlt(xVar_df, xVar_name, yVar_df, yVar_name, plot_title, histStartYr=1971, histEndYr=2020, futStartYr=2021, futEndYr=2070):
    '''
    Input two variables (model output format), plot labels, start and end years
    Plots period change for two variables, each point represents a climate simulation

    '''
    # Period Change Stats
    xVar_df = sortRCP(xVar_df)
    xVar_df = xVar_df/1000
    yVar_df = sortRCP(yVar_df)
    yVar_df = yVar_df/1000
    
    # historic average
    xVar_hist_mean = xVar_df.loc[histStartYr:histEndYr,:].mean()
    yVar_hist_mean = yVar_df.loc[histStartYr:histEndYr,:].mean()

    # fututre average
    xVar_fut_mean = xVar_df.loc[futStartYr:futEndYr,:].mean()
    yVar_fut_mean = yVar_df.loc[futStartYr:futEndYr,:].mean()

    # period change
    xVar_delta = xVar_fut_mean.sub(xVar_hist_mean)
    yVar_delta = yVar_fut_mean.sub(yVar_hist_mean)
    EB_delta = xVar_delta.to_frame(name=xVar_name).join(yVar_delta.to_frame(name=yVar_name))
    EB_delta = EB_delta.reset_index()
    EB_delta = EB_delta.rename(columns = {'index':'Simulation'})
    EB_delta = RCPcol(EB_delta)
    

    # Period Change Plots (dT/dP)
    plt.figure()
    pcp = sns.scatterplot(x=xVar_name, y=yVar_name, data=EB_delta, palette=['#01665e'])

    #EB_delta.plot.scatter(x=xVar_name, y=yVar_name)
    # plt.xlabel(xVar_name)
    # plt.ylabel(yVar_name)
    # plt.title(plot_title)
    pcp.axhline(y=EB_delta[yVar_name].quantile(.25), color='tab:gray', linestyle='dashed')
    pcp.axhline(y=EB_delta[yVar_name].mean(), color='tab:gray')
    pcp.axhline(y=EB_delta[yVar_name].quantile(.75), color='tab:gray', linestyle='dashed')
    pcp.axvline(x=EB_delta[xVar_name].quantile(.25), color='tab:grey', linestyle='dashed')
    pcp.axvline(x=EB_delta[xVar_name].mean(), color='tab:grey')
    pcp.axvline(x=EB_delta[xVar_name].quantile(.75), color='tab:grey', linestyle='dashed')
    # should the lines be in quantiles or standard deviations??
   
def watBalPlt(simNames, Storage_df, P_df, E_df, RO_df, SanMar_df, Release_df, startYr=1960, endYr=2099):
    '''
    Input a list of simulations you want plots for, the dataframe for each wbal variable, and start/end years (opt)
    Plots reservoir water balance parameters over a specified time period
    
    '''
    for sim in simNames:
        EB_Storage = Storage_df.loc[startYr:endYr,sim]
        Direct_Precip = P_df.loc[startYr:endYr,sim]
        Evaporation = E_df.loc[startYr:endYr,sim]
        Runoff = RO_df.loc[startYr:endYr,sim]
        Releases = Release_df.loc[startYr:endYr,sim]
        SanMarcial = SanMar_df.loc[startYr:endYr,sim]
        
        # Create dataframe w years as index
        scn_df = EB_Storage.to_frame(name='Storage')
        scn_df = scn_df.join(Direct_Precip.to_frame(name='Direct Precip'))
        scn_df = scn_df.join(Evaporation.to_frame(name='Evaporation'))
        scn_df = scn_df.join(Runoff.to_frame(name='Runoff'))
        scn_df = scn_df.join(Releases.to_frame(name='Releases'))
        scn_df = scn_df.join(SanMarcial.to_frame(name='San Marcial'))
        
        # Create plot
        scn_df.plot.line()
        plt.title('Elephant Butte Water Balance, ' + str(startYr) + ' to ' + str(endYr) + '--' + sim)
        plt.xlabel('Year')
        plt.ylabel('Volume (af)')
        plt.hlines(2000000, xmin=startYr, xmax=endYr, label='Max Reservoir Volume')

def inOutAreaPlts(simNames, P_df, E_df, RO_df, SanMar_df, Release_df, startYr=1960, endYr=2099):
    '''
    Input a list of simulations you want plots for, the dataframe for each wbal variable, and start/end years (opt)
    For each specified scenario, creates two area plots : flows in and flows out
    
    '''
    
    for sim in simNames:
        #in
        Direct_Precip = P_df.loc[startYr:endYr,sim]
        SanMarcial = SanMar_df.loc[startYr:endYr,sim]
        Runoff = RO_df.loc[startYr:endYr,sim]
        
        scnIn_df = Direct_Precip.to_frame(name='Direct Precip')
        scnIn_df = scnIn_df.join(Runoff.to_frame(name='Runoff'))
        scnIn_df = scnIn_df.join(SanMarcial.to_frame(name='San Marcial'))
        
        scnIn_df.plot.area()

        plt.title('Reservoir inflows ' + str(startYr) + ' to ' + str(endYr) + '--' + sim)
        plt.xlabel('Year')
        plt.ylabel('Volume (af)')
        
        #out
        Releases = Release_df.loc[startYr:endYr,sim]
        Evaporation = E_df.loc[startYr:endYr,sim]
        
        scnOut_df = Releases.to_frame(name='Releases')
        scnOut_df = scnOut_df.join(Evaporation.to_frame(name='Evaporation')) 
        
        scnOut_df.plot.area()
        plt.title('Reservoir outflows ' + str(startYr) + ' to ' + str(endYr) + '--' + sim)
        plt.xlabel('Year')
        plt.ylabel('Volume (af)')

def rcpAvgPlt(df,df_name,y_label,avg_type,startYr=2021,endYr=2070):
    '''
    Input a dataframe, plot labels, averaging type ('average' or 'rolling'), and start/end year (opt)
    Plots a variable averaged by rcp (rolling or normal) 
    '''

    
    dfslice=df.loc[startYr:endYr,:]
    
    rcp26=RCP_df(dfslice,26)
    rcp45=RCP_df(dfslice,45)
    rcp60=RCP_df(dfslice,60)
    rcp85=RCP_df(dfslice,85)
    
    rcp26['average'] = rcp26.mean(axis=1)
    rcp45['average'] = rcp45.mean(axis=1)
    rcp60['average'] = rcp60.mean(axis=1)
    rcp85['average'] = rcp85.mean(axis=1)
    
    avg_per =10
    
    if avg_type == 'average':
        plot_title = df_name + ' Averaged by RCP'
    elif avg_type == 'rolling': 
        plot_title = df_name + ' ' + str(avg_per) + '-year Moving Averaged by RCP'
        #rolling average
        rcp26['rolling'] = rcp26['average'].rolling(window=avg_per).mean()  #TODO: specify axis
        rcp45['rolling'] = rcp45['average'].rolling(window=avg_per).mean()
        rcp60['rolling'] = rcp60['average'].rolling(window=avg_per).mean()
        rcp85['rolling'] = rcp85['average'].rolling(window=avg_per).mean()
    else:
        print('average type error: use \'average\' or \'rolling\'')
    
    
    avgplot = rcp26[avg_type].plot(label='RCP2.5', legend=True)
    rcp45[avg_type].plot(label='RCP4.5',ax=avgplot,legend=True)
    rcp60[avg_type].plot(label='RCP6.0',ax=avgplot,legend=True)
    rcp85[avg_type].plot(label='RCP8.5',ax=avgplot,legend=True)
    plt.title(plot_title)
    plt.xlabel('Year')
    plt.ylabel(y_label)
    plt.tight_layout()

def drought(release_V, pctFull, histStart=1971, histEnd = 2020, futStart=2021, futEnd=2070):
    '''
    Input release volume dataframe, a percent full threshold, and start/end years (opt)
    Bar plot showing the number of years each simulation goes below  a specified percent full  
    '''
    # Create lists of index values
     
    past = release_V.loc[histStart:futEnd,:]      #slice
    past = past.transpose()
    past = past.reset_index()
    past=past.rename(columns = {'index':'Simulation'})      #reset index
    past=pd.melt(past,id_vars=['Simulation'], value_name="Value", var_name='Year')  #melt
    past['Time'] = str(histStart)+'-'+str(histEnd)
    
    fut = release_V.loc[histStart:futEnd,:]      #slice
    fut = fut.transpose()
    fut = fut.reset_index()
    fut = fut.rename(columns = {'index':'Simulation'})      #reset index
    fut = pd.melt(fut,id_vars=['Simulation'], value_name="Value", var_name='Year')  #melt
    fut['Time'] = str(futStart)+'-'+str(futEnd)
    
    joined_df = pd.concat([past,fut])

    print(joined_df)
    
    #df.columns = ['Simulation','Value']

    dplot = sns.boxplot(data=joined_df, x='Simulation', y='Value', hue='Time')
    dplot.set_xticklabels(dplot.get_xticklabels(),rotation=90)
    dplot.axhline(pctFull*875000)
    dplot.axhline(875000)

def drought2(release_V, pctList=[.5,1], startYr=1971,endYr=2070,clist=['#8c510a','#d8b365','#c7eae5','#5ab4ac','#01665e']):
    '''
    DESCRIPTION 
    '''
    release_V = release_V.loc[startYr:endYr,:]      # Slice requested time
    simYears = release_V.index.values.tolist()
    nSims = len(release_V.columns.values)
    colNames = []
    
    for i in pctList:
        name = str(i*100)+'% of Full Allocation'
        colNames.append(name)
        
    #Create dataframe to store count
    stats_df = pd.DataFrame(columns=colNames, index=simYears, dtype=np.float64)
    
    for i in range(len(pctList)):
        threshold = pctList[i]*875000
        for Year in simYears:
            nAllocMet = release_V.apply(lambda x: True if x[Year] >= threshold else False, axis=0)
            stats_df.at[Year,colNames[i]]=len(nAllocMet[nAllocMet == True].index)/nSims*100   #count
              
    plt.figure()
    sns.lineplot(data=stats_df,palette=clist)
    # plt.title('% of simulations failing to release ('+str(startYr)+'-'+str(endYr)+')')
    plt.xlabel('Year')
    plt.ylabel('% of Simulations Meeting Release Volume')
    # hline_vals = [98.11,94.34,71.70,28.30,7.55]
    # for i in range(2):
    #     plt.axhline(y=hline_vals[i],xmin=1971,xmax=2020,color=clist[i],)
    plt.legend(loc='lower left')
    plt.ylim([0,105])
    

    #heatmap_df=stats_df.transpose()
    #sns.heatmap(heatmap_df)
    
def ridgelinePlt(df,varName,startYr=1971,endYr=2070):
    df=df.loc[startYr:endYr,:]
    df=df.transpose()
    df=df.reset_index()
    df=df.rename(columns = {'index':'Simulation'})
    df=pd.melt(df,id_vars=['Simulation'], value_name=varName, var_name='Year')
    
    print(df)

    fig, ax = joyplot(df, column=varName,by='Year')
    plt.axvline(875000)
    
#ridgelinePlt(Cab_Gauge_af, 'Caballo Release(max875000) (af)',startYr=1971,endYr=2070)

def drought3(release_V, pct=.8, startYr=1971,endYr=2070):
    '''
    DESCRIPTION 
    '''
    # List of simulation years
        # Create lists of index values
    threshold = pct*875000
    simYears = []
    consecYearsAll = []
        
    for x in range(StartYear, EndYear+1):
        simYears.append(x)
    Sims = release_V.columns.values.tolist()
    
    df = pd.DataFrame(columns='Value', index=simYears, dtype=np.float64)

    #Iterate through dataframe
    for Sim in Sims:
        count = 0
        for Year in simYears:
            if(df.at[Year,Sim] < threshold):
                count+=1
        df.at[Year,'Value']=count      
 
def consec_drought(release_V,startYr=2021,endYr=2070,pctList=[.2,.5],clist=['#8c510a','#d8b365','#c7eae5','#5ab4ac','#01665e']):  #TODO: might be an error with this one? (plotting both ofsets one...)
    '''
    Input release volume dataframe, a percent full threshold, and start/end years (opt)
    Bar plot showing the longest stretch of consecutive years each simulation dips below a specified percent full 
    '''
    release_V= release_V.loc[startYr:endYr,:]   #Slice requested time period

    simYears = release_V.index.values.tolist()
    Sims = release_V.columns.values.tolist()
    colNames = []
    
    for i in range(len(pctList)):
        colNames.append('Longest string of years with releases below '+str(pctList[i]*100)+'% Full Allocation')
    df = pd.DataFrame(columns=colNames)    
    for i in range(len(pctList)):
        # Create lists of index values
        threshold = pctList[i]*875000
        #Iterate through dataframe
        for Sim in Sims:
            count = 0
            consecYears = []
            for Year in simYears:
                if(release_V.at[Year,Sim] < threshold):
                    count+=1
                elif(count>0):
                    consecYears.append(count)
                    count=0
            if len(consecYears)>0:        
                maximum = max(consecYears)
            else:
                maximum = 0
            df.at[Sim,colNames[i]]=maximum
    avg = df.mean(axis=0)
    print(avg)
    # add subplot
    df.plot.bar(rot=90,subplots=True,legend=None,color=clist)
    
    #plt.title('Length of Longest Period Where Cabllo Releases Fell Below '+str(pctFull)+'x Full Allocation between '+str(startYr)+' and '+str(endYr))
    plt.xlabel('Simulation')

###Correlation Matrix Fxns###
def reshapeVar(df,varName,startYr, endYr):
    '''
    Input a dataframe, name of variable contained in dataframe (string), start and end years
    Returns a reshaped (melted) dataframe with columns: '<variable name>', 'Simulation', and 'Year'

    '''
    
    dfS=df.loc[startYr:endYr]
    dfS = dfS.mean(axis=0)
    dfT=dfS.transpose()
    dfT=dfT.reset_index()
    dfT=dfT.rename(columns = {'index':'Simulation'})
    dfM=pd.melt(dfT,id_vars=['Simulation'], value_name=varName, var_name='Year')
    return(dfM)

def joinVars(dfList,dfNames,startYr,endYr):
    '''
    Input a list of dataframes, a corresponding list of the variable names (list of strings), start and end year
    Returns the melted and joined dataframes for correlation matrix
    
    '''
    
    allvars_df = pd.DataFrame()
    first = True
    nVars=len(dfNames)
    # Call reshape fxn for each df:
    for i in range(0,(nVars)):
        dfList[i]=sortRCP(dfList[i])      #sort so plot colors are assigned in order
        reshaped_df = reshapeVar(dfList[i],dfNames[i],startYr,endYr)
        if(first==True):
            allvars_df = reshaped_df
            first=False
        else:
            allvars_df=pd.merge(allvars_df,reshaped_df,how='left',on=['Year','Simulation'])
    return(allvars_df)

def corrMtx(dfList,dfNames,startYr,endYr):
    '''
    Input a list of dataframes, a corresponding list of the variable names (list of strings), start and end year
    Plots a correlation matrix for a set of variables passed to the function

    '''
    
    # Call fxn to restructure and join dataframes
    joined_df = joinVars(dfList,dfNames,startYr,endYr)      
    # Add rcp column, remove year and simulation columns
    joined_df = RCPcol(joined_df)
    joined_df.pop('Simulation')
    joined_df.pop('Year')
    
    # To plot without extreme SM inflows (uncomment this line to remove outliers)
    #idx_names = joined_df[joined_df['San Marcial (af)']>5000000].index 
    #joined_df.drop(idx_names, inplace=True)
    
    # Plot
    colors=['windows blue','medium green','amber','pale red']
    cm = sns.pairplot(joined_df,kind='scatter',diag_kind='auto',hue='RCP',palette=sns.xkcd_palette(colors),corner=False,plot_kws={'alpha':0.5,'s':20})#,'marker':"+"}) 
    sns.set_style("ticks")
    sns.set_context("paper")
    cm.fig.suptitle('Correlation Matrix fpr '+str(startYr)+' to '+str(endYr))
    
    plt.figure()
    cmc = sns.heatmap(joined_df.corr())
    
def corrPlt(dfList,dfNames,startYr,endYr):
    '''
    Input a list of dataframes, a corresponding list of the variable names (list of strings), start and end year
    Plots a correlation matrix for a set of variables passed to the function

    '''
    # Call fxn to restructure and join dataframes
    joined_df = joinVars(dfList,dfNames,startYr,endYr)      
    # Add rcp column, remove year and simulation columns
    joined_df = RCPcol(joined_df)
    
    # To plot without extreme SM inflows (uncomment this line to remove outliers)
    #idx_names = joined_df[joined_df['San Marcial (af)']>5000000].index 
    #joined_df.drop(idx_names, inplace=True)
    
    # Plot
    plt.figure()
    #colors=['windows blue','medium green','amber','pale red']
    cm = sns.scatterplot(x=joined_df[dfNames[0]],y=joined_df[dfNames[1]],hue=joined_df['Year'],alpha=1)#,hue='RCP',palette=sns.xkcd_palette(colors),corner=False,plot_kws={'alpha':0.5,'s':20})#,'marker':"+"}) 
    sns.set_style("ticks")
    sns.set_context("paper")
    cm.set_title('Correlation Matrix fpr '+str(startYr)+' to '+str(endYr))
    print(joined_df)
    #TODO: hue = time
##End Correlation Matrix Fxns##

# T, P, SM box and whisker plots
def reshapeVarBox(df,varName,startYr, endYr):
    '''
    Input a dataframe, name of variable contained in dataframe (string), start and end years
    Returns a reshaped (melted) dataframe with columns: 'Value', 'Simulation', and 'Time'

    '''
    
    df=df.loc[startYr:endYr]
    df = df.mean(axis=0).to_frame() #comment to use all values
    df = df.reset_index()
    df.columns = ['Simulation',varName]
    return(df)

def joinVarsBox(dfList,dfNames,startYr,endYr):
    '''
    Input a list of dataframes, a corresponding list of the variable names (list of strings), start and end year
    Returns joined dataframes for box plot
    
    '''
    
    allvars_df = pd.DataFrame()
    first = True
    nVars=len(dfNames)
    # Call reshape fxn for each df:
    for i in range(0,(nVars)):
        #dfList[i]=sortRCP(dfList[i])      #sort so plot colors are plotted in order
        reshaped_df = reshapeVarBox(dfList[i],dfNames[i],startYr,endYr)
        if(first==True):
            allvars_df = reshaped_df
            first = False
        else:
            allvars_df=pd.merge(allvars_df,reshaped_df,how='left',on=['Simulation'])
    allvars_df['Time'] = str(startYr)+'-'+str(endYr)
    allvars_df = RCPcol(allvars_df)
    return(allvars_df)

def boxwplot(dfList,dfNames, histStart=1971, histEnd = 2020, futStart=2021, futEnd=2070):

    # average, add cols, join
    hist_joined_df = joinVarsBox(dfList,dfNames,histStart,histEnd) 
    fut_joined_df = joinVarsBox(dfList,dfNames,futStart,futEnd) 
    joined_df = pd.concat([hist_joined_df,fut_joined_df])

    #plot
    nVars=len(dfNames)
    fig, axes = plt.subplots(1, nVars,figsize=(20, 5))
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 1.5})        
    for i in range(0,(nVars)):
        #sns.boxplot(ax=axes[i], data=joined_df, x='RCP', y=dfNames[i],hue='Time')   # Simulations separated by rcp
        sns.boxplot(ax=axes[i], data=joined_df, x='Time', y=dfNames[i])     # All simulations
        #axes[i].set_title(dfNames[i])

    # fig2, axes = plt.subplots(1, nVars,figsize=(20, 5))
    # sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 1.5})    
    # for i in range(0,(nVars)):
    #     #sns.boxplot(ax=axes[i], data=joined_df, x='RCP', y=dfNames[i],hue='Time')   # Simulations separated by rcp
    #     sns.swarmplot(ax=axes[i], data=joined_df, x='Time', y=dfNames[i],hue='RCP')     # All simulations
    #     axes[i].set_title(dfNames[i])
        
    # fig, axes = plt.subplots(1, nVars,figsize=(20, 5))
    # sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 1.5})    
    # for i in range(0,(nVars)):
    #     #sns.boxplot(ax=axes[i], data=joined_df, x='RCP', y=dfNames[i],hue='Time')   # Simulations separated by rcp
    #     sns.violinplot(ax=axes[i], data=joined_df, x='Time', y=dfNames[i])     # All simulations
    #     axes[i].set_title(dfNames[i])
        


def pchange(dfList,dfNames,histStart=1971,histEnd=2020,futStart=2021,futEnd=2070):
    table = pd.DataFrame(columns=['histAvg','futAvg','diff','pctDiff'],index=dfNames)
    for i in range(0,len(dfList)):
        df=dfList[i]
        hist = df.loc[histStart:histEnd,:].mean(axis=0)
        fut = df.loc[futStart:futEnd,:].mean(axis=0)
        table.at[dfNames[i],'histAvg']=hist.mean()
        table.at[dfNames[i],'futAvg']=fut.mean()
        table.at[dfNames[i],'diff']=table.at[dfNames[i],'futAvg']-table.at[dfNames[i],'histAvg']
        table.at[dfNames[i],'pctDiff']=table.at[dfNames[i],'diff']/table.at[dfNames[i],'histAvg']*100
    print(table)
        

# Temp and Precip Time Series:
# Elephant Butte Temperature
#pctOverTimePlt(EBTas_degC, 'Elephant Butte Temperature', 'Year', 'Average Temperature (deg C)')

# Caballo Temperature
#pctOverTimePlt(CabTas_degC, 'Caballo Temperature', 'Year', 'Average Temperature (deg C)')
#varOverTimePlt(CabTas_degC, 'Caballo Temperature', 'Year', 'Average Temperature (deg C)')

# Elephant Butte Precipitation
#pctOverTimePlt(EBPr_ft, 'Elephant Butte Precipitation', 'Year', 'Precip (ft/year)')
#varOverTimePlt(EBPr_ft, 'Elephant Butte Precipitation', 'Year', 'Precip (ft/year)')

# Caballo Precipitation
#pctOverTimePlt(CabPr_ft, 'Caballo Precipitation', 'Year', 'Precip (ft/year)')
#varOverTimePlt(CabPr_ft, 'Caballo Precipitation', 'Year', 'Precip (ft/year)')

# Period Change Plot:
# Elephant Butte
##perChangePlt(EBTas_degC,'EB Avg Temp (deg C)', EBPr_ft,'EB Yrly Precip (ft/yr)', 'Change in Temp & Precip at EB 1950-1999 vs 2020-2070', 1971, 2020, 2021, 2070)

# Caballo
#perChangePlt(CabTas_degC,'Cab Avg Temp', CabPr_ft,'Cab Yrly Precip', 'Change in Temp & Precip at Caballo 1950-1999 vs 2020-2070', 1950, 1999, 2020, 2070)

# Delta upstream, delta local:
#perChangePlt(Net_Local, 'Change in Net Local Reservoir Inputs (af/yr)', SanMar_af, 'Change in Upstream Reservoir Inputs (af/yr)', 'Local vs Upstream Change 1971-2020 vs 2021-2070', 1971, 2020, 2021, 2070)#what other ways can we show this??


# Reservoir Volume
#varOverTimePlt(EB_Storage_af,plot_title='Elephant Butte Storage',x_label='Year',y_label='EB Storage (af)',startYr=2021,endYr=2070)
#pctOverTimePlt(EB_Storage_af,plot_title='Elephant Butte Storage',x_label='Year',y_label='EB Storage (af)',startYr=2021,endYr=2070)

# Reservoir Release
#varOverTimePlt(Cab_Gauge_af,plot_title='Caballo Releases',x_label='Year',y_label='Caballo Release (af)',startYr=2021,endYr=2070)
#pctOverTimePlt(Cab_Gauge_af,plot_title='Caballo Releases',x_label='Year',y_label='Caballo Release (af)',startYr=2021,endYr=2070)

# Water Balance Major Plots
# Plots Yearly: Res Volume, Precip V, Ro V, E V, San Mar V, Release V
# for each scenario in the list over the specified range of years
WBplotScns = ['Historic']     
watBalPlt(WBplotScns,EB_Storage_af,Res_SPrecip_af,Res_SEvap_af,Cab_RO_af,SanMar_af,Cab_Release_af, 2008, 2020) 
inOutAreaPlts(WBplotScns,Res_SPrecip_af,Res_SEvap_af,Cab_RO_af,SanMar_af,Cab_Release_af, 2008, 2020)


# Other Stats:
# Years below x volume over x years, bar chart for each scenario (count then transpose?)

# # of years with excess over x years, bar chart for each scenario

# time period change (number/table) for: release volume, pr, ro , e, san mar
# past/future box plots, w subplots for the different variables


#table version of the above statistics for spreadsheet
#flip df so index is scnName
#then flip back for spreadsheet



# #--------------------OUTPUT FOR REPORT------------------------------#
# avg_temp = (EBTas_degC*EBLandArea_ac+CabTas_degC*CabLandArea_ac)/(EBLandArea_ac+CabLandArea_ac)
# avg_precip = (EBPr_ft*EBLandArea_ac+CabPr_ft*CabLandArea_ac)/(EBLandArea_ac+CabLandArea_ac)
# Cab_Release_kaf = Cab_Release_af/1000
# SanMar_kaf = SanMar_af/1000

# # 1) Past/Future Temp, Precip, San Marcial
# box_dfList = [avg_precip,avg_temp,SanMar_kaf]
# box_dfNames = ['Average Precipitation (ft/yr)','Average Temperature (degrees C)','Average Streamflow (kaf/yr)']  
# boxwplot(box_dfList,box_dfNames)

# # 1.5 Print some stats on percent change

# pchangevars=[EBTas_degC,EBPr_ft,SanMar_af,Cab_Release_kaf]
# pchangenames=['EB Temp (degC)','EB Pr (ft/yr)','Streamflow (kaf)','Caballo Release (kaf)']
# pchange(pchangevars,pchangenames)
# pchange(box_dfList,box_dfNames)

# # 2) Across all models, whats the liklihood different amounts will be released over time?
# drought2(Cab_Release_af,pctList=[1,.5,.2],startYr=1971,endYr=2070,clist=['#01665e','#d8b365','#8c510a'])

# # 2.5) Across years, what is the liklihood of X release volume
# resStats(Cab_Release_af,pctList=[1,.5,.2],startYr=2021,endYr=2070,clist=['#01665e','#d8b365','#8c510a'])
# #resStats(Cab_Release_af,pctList=[1,.5,.2],startYr=1971,endYr=2020,clist=['#01665e','#d8b365','#8c510a'])

# # 3) Consecutive years of drought
# consec_drought(Cab_Release_af, pctList=[1,.5,.2],startYr=2021,endYr=2070,clist=['#01665e','#d8b365','#8c510a'])
# #consec_drought(Cab_Release_af, pctList=[.5,.2],startYr=1971,endYr=2020,clist=['#d8b365','#8c510a'])
# # 2013 flow 169158 AF->around 20% full allocation

# # 4) Correlations: Evap V/SA, Evap Vol/Temp, San Marcial/SA    
# dfList = [EB_SurfA_ac,EBTas_degC,Res_SEvap_af,SanMar_af]
# dfNames = ['EB Surface Area (af)','EB Temperature (degC)','Reservoir Surface Evap. (af/yr)','San Marcial (af)']    

# #corrMtx(dfList,dfNames,1971,2070)

# #corrMtx(box_dfList, box_dfNames, 2021, 2070)


# dfl1=[Res_SEvap_af,EB_SurfA_ac]
# dfn1=['Reservoir Surface Evaporation (af)','Reservoir Surface Area (acres)']
# dfl2=[Res_SEvap_af,EBTas_degC]
# dfn2=['Reservoir Surface Evaporation (af)', 'Elephant Butte Temperature (deg C)']
# dfl3=[SanMar_af,EB_SurfA_ac]
# dfn3=['San Marcial Flow (af)','Elephant Butte Surface Area (acres)']
# # corrPlt(dfl1,dfn1,startYr=1971,endYr=2020)
# # corrPlt(dfl2,dfn2,startYr=1971,endYr=2020)
# # corrPlt(dfl3,dfn3,startYr=1971,endYr=2020)
# # corrPlt(dfl1,dfn1,startYr=2021,endYr=2070)
# # corrPlt(dfl2,dfn2,startYr=2021,endYr=2070)
# # corrPlt(dfl3,dfn3,startYr=2021,endYr=2070)
# # corrPlt(dfl1,dfn1,startYr=1971,endYr=2070)
# # corrPlt(dfl2,dfn2,startYr=1971,endYr=2070)
# # corrPlt(dfl3,dfn3,startYr=1971,endYr=2070)


# # 5) Local/Upstream Contributions
# perChangePlt(Net_Local, 'Change in Net Local Reservoir Contributions (kaf/yr)', SanMar_af, 'Change in Upstream Reservoir Contributions (kaf/yr)', 'Local vs Upstream Change 1971-2020 vs 2021-2070', 1971, 2020, 2021, 2070)
# #perChangePlt(local_net_depth, 'Change in Net Local Reservoir Inputs (1000 ft)', SanMar_af, 'Change in Upstream Reservoir Inputs (kaf/yr)', 'Local vs Upstream Change 1971-2020 vs 2021-2070', 1971, 2020, 2021, 2070)



#-----------------Appendix/Supplemental-----------------------------#

#perChangePlt(EBTas_degC,'EB Avg Temp (deg C)', EBPr_ft,'EB Yrly Precip (ft/yr)', 'Change in Temp & Precip at EB 1950-1999 vs 2020-2070', 1971, 2020, 2021, 2070)

# Reservoir Volume
#varOverTimePlt(EB_Storage_af,plot_title='Elephant Butte Storage',x_label='Year',y_label='EB Storage (af)',startYr=2021,endYr=2070)
#pctOverTimePlt(EB_Storage_af,plot_title='Elephant Butte Storage',x_label='Year',y_label='EB Storage (af)',startYr=2021,endYr=2070)
# Reservoir Release
#varOverTimePlt(Cab_Gauge_af,plot_title='Caballo Releases',x_label='Year',y_label='Caballo Release (af)',startYr=2021,endYr=2070)
#pctOverTimePlt(Cab_Gauge_af,plot_title='Caballo Releases',x_label='Year',y_label='Caballo Release (af)',startYr=2021,endYr=2070)

#perChangePlt(CabTas_degC,'Cab Avg Temp', CabPr_ft,'Cab Yrly Precip', 'Change in Temp & Precip at Caballo 1950-1999 vs 2020-2070', 1950, 1999, 2020, 2070)

#perChangePlt(EBTas_degC,'EB Avg Temp (deg C)', EBPr_ft,'EB Yrly Precip (ft/yr)', 'Change in Temp & Precip at EB 1950-1999 vs 2020-2070', 1971, 2020, 2021, 2070)

#maybe a full correlation matrix idk