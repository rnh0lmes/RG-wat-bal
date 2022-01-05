# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 19:42:31 2020

@author: Robyn
"""

# ****Elephant Butte Reservoir Release Modeling**** #

# This script has been developed by Nolan Townsend, Luis Garnica Chavira, and Robyn Holmes
# under the supervision of Alex Mayer.


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#******************************MODEL START***********************************#
#****************************************************************************#

#******************************LOAD INPUT************************************#
# Load Excel File
fileName = 'Inputs\\input_parameters_projections.xlsx'   # File name

# Load each sheet as a pandas dataframe
ScalarInputs = pd.read_excel(fileName, sheet_name='ScalarInputs', index_col=0)
SanMar_cfs = pd.read_excel(fileName, sheet_name='SanMar_cfs', index_col=0)
CabPr_mmday = pd.read_excel(fileName, sheet_name='CabPr_mmday', index_col=0)
CabTas_degC = pd.read_excel(fileName, sheet_name='CabTas_degC', index_col=0)
EBPr_mmday = pd.read_excel(fileName, sheet_name='EBPr_mmday', index_col=0)
EBTas_degC = pd.read_excel(fileName, sheet_name='EBTas_degC', index_col=0)
OutputDescriptions = pd.read_excel(fileName, sheet_name='OutputDescriptions', index_col=0)

# Convert units
SanMar_af = SanMar_cfs*724.44792344617  # Convert CFS to af/year
CabPr_ft = CabPr_mmday*1.198302165      # Convert mm/day to ft/year
EBPr_ft = EBPr_mmday*1.198302165        # Convert mm/day to ft/year

#*****************************Extract Vaiables*******************************#
# Elephant Butte Reservoir
EBInitStorage_af = ScalarInputs.at['EBInitStorage_af', 'Value']          # Initial Storage Volume: 415000 acre-feet
EBMin = ScalarInputs.at['EBMin', 'Value']                  # EB Minimum Storage Volume: 17300 acre-feet
EBMax = ScalarInputs.at['EBMax', 'Value']                  # EB Maximum Storage Volume: 1.99E+06 acre-feet

EBA0 = ScalarInputs.at['EBA0', 'Value']                    # Hypsometric coefficient values for calc Elephant Butte surface area
EBA1 = ScalarInputs.at['EBA1', 'Value']
EBA2 = ScalarInputs.at['EBA2', 'Value']
EBA3 = ScalarInputs.at['EBA3', 'Value']
EBA4 = ScalarInputs.at['EBA4', 'Value']

# Caballo Reservoir
CabInitStorage_af = ScalarInputs.at['CabInitStorage_af', 'Value']       # Caballo Initial Storage Volume

CabA0 = ScalarInputs.at['CabA0', 'Value']                    # Hypsometric coefficient values for calc Caballo surface area
CabA1 = ScalarInputs.at['CabA1', 'Value']
CabA2 = ScalarInputs.at['CabA2', 'Value']
CabA3 = ScalarInputs.at['CabA3', 'Value']
CabA4 = ScalarInputs.at['CabA4', 'Value']

#Operating Constants
OPConst1 = ScalarInputs.at['OPConst1', 'Value']             # Constants for calculating desired release volume
OPConst2 = ScalarInputs.at['OPConst2', 'Value']
OPConst3 = ScalarInputs.at['OPConst3', 'Value']
FullAllocation_af = ScalarInputs.at['FullAllocation', 'Value']  # Average historical full allocation baseline over the whole region

#Other Coefficients
RunoffCoeff = ScalarInputs.at['RunoffCoeff', 'Value']        # Runoff Coefficient

EvapCoeffA = ScalarInputs.at['EvapCoeff', 'Value']           # Evaporation Coefficient A
EvapCoeffB = ScalarInputs.at['EvapInt', 'Value']             # Evaporation Coefficient B

HistoricTas = ScalarInputs.at['HistoricTas', 'Value']        # Average temperature over historical period (from temp-evap regression)
HistoricEvap = ScalarInputs.at['HistoricEvap', 'Value']      # Average evaperation over historical period (from temp-evap regression)

CabLandArea_ac = ScalarInputs.at['CabLandArea', 'Value']     # Caballo subwatershed land area (acres)
EBLandArea_ac = ScalarInputs.at['EBLandArea', 'Value']       # Elephant Butte subwatershed land area (acres)

#Timestep Info
StartYear = int(ScalarInputs.at['StartYear', 'Value'])       # Simulation start year
EndYear = int( ScalarInputs.at['EndYear', 'Value'])          # Simulation end year
Years = []                                              # Initialize simulation year array
for x in range(StartYear, EndYear+1):
    Years.append(x)
    
#***********************Extract Climate Scenario Info************************#
#Scenario Info
ClmSimNames = SanMar_af.columns.values.tolist()   # Creates a list of all climate scenarios

# Check all senario names from excel match
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

#***********Initialize Dataframes (df) for Variable Storage******************#
EB_Storage_af = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)         # Elephant Butte Storage (acre-feet)
EB_SurfA_ac = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)           # Elephant Butte Surface Area (acres)
Cab_Storage_af = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)        # Caballo Storage (acre-feet)
Cab_SurfA_ac = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)          # Caballo Surface Area (acres)
Res_SEvap_af = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)          # Reservoir Surface Evaporation
Res_SPrecip_af = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)        # Reservoir Direct Precipitation (af), Caballo + Elephant Butte
Excess_af = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)             # Excess (water that must be released to keep reservoir volume within capacity, not allocated to a user)
Cab_Gauge_af = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)          # Yearly Caballo Gauge (acre-feet) (allocated releases)
Cab_RO_af = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)             # Caballo Runoff (acre-feet)
DCab_Gauge_af = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)         # Desired Caballo Gauge (acre-feet)
Cab_Release_af = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)        # Gauge Flow at Caballo; = Caballo_Realease_af + Excess (acre-feet)
WBal_Check_af = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)         # Check if waterbalance = 0
SW_Evap_ft = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)            # Surface water evaporation depth (feet)
Allocation_Diff = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)       # Allocation Difference; = Cab_Gauge_af - DCab_Gauge_af
Net_Local_af = pd.DataFrame(columns=ClmSimNames, index=Years, dtype=np.float64)          # Net local reservoir flows; = P + RO - E

#***********************RESERVOIR OPERATION MODEL****************************#    
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
        i = 0
        while((EB_Storage_Diff > Epsilon) and (i < MaxIt)):
            if i == 0:
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
    
            i+=1
        # End iteration that solves for EB storage
        
        EB_Storage_af.at[Year, ClmSim] = EB_Storage_current_it       # Store EB Yearly Storage in dataframe
        Allocation_Diff.at[Year,ClmSim] = FullAllocation_af - Cab_Release_af.at[Year,ClmSim]  # Difference between Full Allocation and the actual release
        
        # Double Check Water Balance
        deltaS = 0
        if Year == StartYear:
            deltaS = EB_Storage_af.at[Year,ClmSim] - EBInitStorage_af
        else:
            deltaS = EB_Storage_af.at[Year,ClmSim] - EB_Storage_af.at[Year-1,ClmSim]
        WBal_Check_af.at[Year,ClmSim] = SanMar_af.at[Year, ClmSim] + Cab_RO_af.at[Year, ClmSim] - Res_SEvap_af.at[Year, ClmSim] + Res_SPrecip_af.at[Year, ClmSim] - Cab_Release_af.at[Year, ClmSim] - deltaS # Reservoir WB = inflows - outflows - delta S

    print('Completed:'+ClmSim)
    # End Year iteration
# End Climate Scenario iteration


#***********************Conversions, average subwatersheds*******************#
AvgTas_degC = (EBTas_degC*EBLandArea_ac+CabTas_degC*CabLandArea_ac)/(EBLandArea_ac+CabLandArea_ac)
AvgPr_ft = (EBPr_ft*EBLandArea_ac+CabPr_ft*CabLandArea_ac)/(EBLandArea_ac+CabLandArea_ac)
Cab_Release_kaf = Cab_Release_af/1000
SanMar_kaf = SanMar_af/1000
Net_Local_af = Cab_RO_af.sub(Res_SEvap_af).add(Res_SPrecip_af)     # Reservoir contributions from the local subwatersheds
local_net_depth = EBPr_ft.sub(SW_Evap_ft)
print('Describe Net Local, 2021-2070')
print(Net_Local_af.loc[2021:2070,:].melt().describe())

#***********************Export data to excel*********************************# 
writer = pd.ExcelWriter('Outputs\\ResModelOutput.xlsx')

OutputDescriptions.to_excel(writer, 'OutputDescriptions')
ScalarInputs.to_excel(writer, "Scalar Variables")
Allocation_Diff.to_excel(writer, 'Allocation_Diff')
AvgPr_ft.to_excel(writer, 'AvgPr_ft')
AvgTas_degC.to_excel(writer, 'AvgTas_degC')
CabPr_ft.to_excel(writer, 'CabPr_ft')
CabPr_mmday.to_excel(writer, 'CabPr_mm')
CabTas_degC.to_excel(writer, 'CabTas_degC')
Cab_Gauge_af.to_excel(writer, 'Cab_Gauge_af')
Cab_RO_af.to_excel(writer, 'Cab_RO_af')
Cab_Release_af.to_excel(writer, 'Cab_Release_af')
Cab_Release_kaf.to_excel(writer, 'Cab_Release_kaf')
Cab_Storage_af.to_excel(writer, 'Cab_Storage_af')
Cab_SurfA_ac.to_excel(writer, 'Cab_SurfA_ac')
DCab_Gauge_af.to_excel(writer, 'DCab_Gauge_af')
EBPr_ft.to_excel(writer, 'EBPr_ft')
EBPr_mmday.to_excel(writer, 'EBPr_mmday')
EBTas_degC.to_excel(writer, 'EBPr_degC')
EB_Storage_af.to_excel(writer, 'EB_Storage_af')
EB_SurfA_ac.to_excel(writer, 'EB_SurfA_ac')
Excess_af.to_excel(writer, 'Excess_af')
Net_Local_af.to_excel(writer, 'Net_Local_af')
Res_SEvap_af.to_excel(writer, 'Res_SEvap_af')
Res_SPrecip_af.to_excel(writer, 'Res_SPrecip_af')
SW_Evap_ft.to_excel(writer, 'SW_Evap_ft')
SanMar_af.to_excel(writer, 'SanMar_af')
SanMar_cfs.to_excel(writer, 'SanMar_cfs')
SanMar_kaf.to_excel(writer, 'SanMar_kaf')
WBal_Check_af.to_excel(writer, 'WBal_Check_af')

writer.save()

#*****************Statistics & Plotting Functions****************************#
def resStats(release_df, startYr=2021,endYr=2070, pctList=[.5,1],clist=['#8c510a','#d8b365','#c7eae5','#5ab4ac','#01665e']):
    '''
    Input a dataframe of reservoir releases from the model (start and end year optional)
    Outputs a table of reservoir statistics

    '''
    DesiredRelease = 790000
    nYears = endYr-startYr+1 # number of years considered
    
    release_df = release_df.loc[startYr:endYr,:]    #slice df to only include desired years
    
    ClmSimNames = release_df.columns.values.tolist()    #create a list of climate models
    ColNames = pctList
    stats_df = pd.DataFrame(index=ClmSimNames, columns=ColNames, dtype=np.float64)          #create new/empty df w scenarios as index


    #add col w fraction of years w full allocation released
    for i in range(len(pctList)):
        pctAlloc = pctList[i]*DesiredRelease
        for Scn in ClmSimNames:
            nAllocMet = release_df.apply(lambda x: True if x[Scn] < pctAlloc else False, axis=1)
            stats_df.at[Scn,ColNames[i]]=len(nAllocMet[nAllocMet == True].index)/nYears*100   #count # of years below DesiredRelease
        ColNames[i] = 'Percent of Years Below '+str(pctList[i]*100)+'% of Full Allocation' 
    stats_df.columns = ColNames
        
    avg = stats_df.mean(axis=0)
    print('Average something to do with reservoir....')
    print(avg)
    
    for i in range(len(ColNames)):
        print(ColNames[i])
        print(stats_df[ColNames[i]].describe(percentiles=(.05,.1,.25,.5,.75,.9,.95)))

    
def perChangePlt(xVar_df, xVar_name, yVar_df, yVar_name, histStartYr=1971, histEndYr=2020, futStartYr=2021, futEndYr=2070):
    '''
    Input two variables (model output format), plot labels, start and end years
    Plots period change for two variables, each point represents a climate simulation

    '''
    
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
    
    # Period Change Plots (dT/dP)
    plt.figure()
    pcp = sns.scatterplot(data=EB_delta, x=xVar_name, y=yVar_name, palette=['#01665e'],zorder=2.5)
    plt.tight_layout()

    Ystats = EB_delta[yVar_name].describe(percentiles=(.05,.1,.25,.5,.75,.9,.95))
    Xstats = EB_delta[xVar_name].describe(percentiles=(.05,.1,.25,.5,.75,.9,.95))
    print('Describe' + yVar_name)
    print(Ystats)
    print('Describe' + xVar_name)
    print(Xstats)
    
    pcp.axhline(0,color='tab:grey')
    pcp.axhline(Ystats['50%'], color='tab:gray',linestyle='dashed')
    pcp.axvline(0,color='tab:grey')
    pcp.axvline(Xstats['50%'], color='tab:grey',linestyle='dashed')
        
def inOutPieCharts(P_df,E_df,RO_df,Qin_df,Release_df,startYr,endYr):
    '''
    Pie chart of water balance terms.

    '''
    frames = [Qin_df,P_df,RO_df,Release_df,E_df]
    avgs = []
    colorsIn = sns.color_palette('colorblind')[7:10]
    colorsOut = sns.color_palette('colorblind')[3:5]
    
    for df in frames:
        avg = df.loc[startYr:endYr,:].mean().mean()
        avgs.append(avg)
    
    plt.figure()
    ins = avgs[0:3]
    insPct = [i/sum(ins)*100 for i in ins]
    inNames = ["Streamflow: {:0.0f} kAF ({:0.1f}%)".format(ins[0]/1000,insPct[0]),"Precipitation: {:0.0f} kAF ({:0.1f}%)".format(ins[1]/1000,insPct[1]),"Runoff: {:0.0f} kAF ({:0.1f}%)".format(ins[2]/1000,insPct[2])]
    #print("tot avg in " + str(sum(ins)) + str(ins))
    plt.pie(ins,labels=inNames,colors=colorsIn,wedgeprops = {'linewidth' : 2 , 'edgecolor':'black'})
    plt.title("Flow In ("+str(startYr)+"-"+str(endYr)+")")
    
    plt.figure()
    outs = avgs[3:5]
    outsPct = [i/sum(outs)*100 for i in outs]
    outNames = ["Releases: {:0.0f} kAF ({:0.1f}%)".format(outs[0]/1000,outsPct[0]),"Evaporation: {:0.0f} kAF ({:0.1f}%)".format(outs[1]/1000,outsPct[1])]
    print("tot avg out " + str(sum(outs)) + str(outs))
    plt.pie(outs,labels=outNames,colors=colorsOut,wedgeprops = {'linewidth' : 2,'edgecolor':'black'})
    plt.title("Flow Out ("+str(startYr)+"-"+str(endYr)+")")
    
def drought2(release_V, pctList=[.5,1], startYr=1971,endYr=2070,clist=['#8c510a','#d8b365','#c7eae5','#5ab4ac','#01665e']):
    '''
    Plots percent of simulations below thesholds over a range of years

    '''
    release_V = release_V.loc[startYr:endYr,:]      # Slice requested time
    simYears = release_V.index.values.tolist()
    nSims = len(release_V.columns.values)
    colNames = []
    
    for i in pctList:
        name = str(int(i*100))+'% of Full Allocation'
        colNames.append(name)
        
    #Create dataframe to store count
    stats_df = pd.DataFrame(columns=colNames, index=simYears, dtype=np.float64)
    
    for i in range(len(pctList)):
        threshold = pctList[i]*790000
        for Year in simYears:
            nAllocMet = release_V.apply(lambda x: True if x[Year] < threshold else False, axis=0)
            stats_df.at[Year,colNames[i]]=len(nAllocMet[nAllocMet == True].index)/nSims*100   #count
            
            
    ## Export for alex:
    fracFullAlloc = release_V/FullAllocation_af
    w = pd.ExcelWriter('C:\\Users\\robyn\\Documents\\Grad_School\\Research\\swim-python\\Outputs\\fig6.xlsx')
    stats_df.to_excel(w,"fig6")
    fracFullAlloc.to_excel(w,"fractionFullAlloc")
    release_V.to_excel(w,"releaseVolume")
    w.save()
              
    plt.figure()
    sns.lineplot(data=stats_df)
    plt.xlabel('Year')
    plt.ylabel('Simulations Below Threshold (%)')
    plt.legend(loc='upper left',title="Threshold Volume")
    plt.ylim([0,105])
    plt.tight_layout()
    
    #filled version
    a = stats_df.iloc[:,0]
    b = stats_df.iloc[:,1]
    c = stats_df.iloc[:,2]
    
    filled25 = c
    filled50 = b - filled25
    filled100 = a - filled50
    
    filledColors = [clist[-1],clist[-2],clist[-3]]
    plt.figure()
    plt.stackplot(simYears, filled25, filled50, filled100, colors=filledColors)
    plt.xlabel('Year')
    plt.ylabel('Simulations Below Threshold (%)')
    plt.legend(["0% to 25% of Full Allocation", "25% to 50% of Full Allocation", "50% to 100% of Full Allocation"],loc='upper left',title="Threshold Volume")
    plt.tight_layout()
    
    d2 = stats_df
    d2_past = d2.loc[1971:2020,:]
    d2_fut = d2.loc[2021:2070.:]
    print('Full Alloc')
    L1=d2_past['100% of Full Allocation'].describe()
    print(L1)
    L2=d2_fut['100% of Full Allocation'].describe()
    plt.axhline(L2['50%'],1971,2021,color=clist[0])
    print(L2)
    print('half alloc')
    L3=d2_past['50% of Full Allocation'].describe()
    print(L3)
    L4=d2_fut['50% of Full Allocation'].describe()
    print(L4['50%'])
    print(L4)
    print('25% alloc')
    L5 = d2_past['25% of Full Allocation'].describe()
    print(L5)
    L6 = d2_fut['25% of Full Allocation'].describe()
    print(L6)    
 
def consec_drought(release_V,startYr=2021,endYr=2070,pctList=[.2,.5],clist=['#8c510a','#d8b365','#c7eae5','#5ab4ac','#01665e']):

    release_V= release_V.loc[startYr:endYr,:]   #Slice requested time period

    simYears = release_V.index.values.tolist()
    Sims = release_V.columns.values.tolist()
    colNames = []
    
    for i in range(len(pctList)):
        colNames.append(str(pctList[i]*100)+'%')
    df = pd.DataFrame(columns=colNames,dtype=np.float64)  
    
    #past
    for i in range(len(pctList)):
        # Create lists of index values
        threshold = pctList[i]*790000
        #Iterate through dataframe
        for Sim in Sims:
            count = 0
            consecYears = []
            for Year in simYears:
                if(release_V.at[Year,Sim] < threshold):
                    count+=1
                elif(count>0):
                    count=0
                consecYears.append(count)
            if len(consecYears)>0:        
                maximum = max(consecYears)
            else:
                maximum = 0
            df.at[Sim,colNames[i]]=maximum
    avg = df.mean(axis=0)
    print('Describe consecutive drought:')
    print(avg)
    
    for i in range(len(colNames)):
        print(colNames[i])
        print(df[colNames[i]].describe(percentiles=(.05,.1,.25,.5,.75,.9,.95)))

def consec_drought2(release_V,pctList=[.2,.5],clist=['#8c510a','#d8b365','#c7eae5']):  

    release_V_past= release_V.loc[1971:2020,:]   #Slice requested time period
    release_V_fut= release_V.loc[2021:2070,:]   #Slice requested time period
    
    pastYears = release_V_past.index.values.tolist()
    futYears = release_V_fut.index.values.tolist()
    Sims = release_V.columns.values.tolist()
    
    colNames = ['Consecutive Years Below Threshold','Year Range','Threshold Volume']
    df = pd.DataFrame(columns=colNames,dtype=np.float64) 
    
    fracAlloc = []
    
    for i in range(len(pctList)):
        fracAlloc.append(str(int(pctList[i]*100))+'% of Full Allocation')
    
    #past
    for i in range(len(pctList)):
        # Create lists of index values
        threshold = pctList[i]*790000
        #Iterate through dataframe
        for Sim in Sims:
            count = 0
            consecYears = []
            for Year in pastYears:  ##change
                if(release_V.at[Year,Sim] < threshold):
                    count+=1
                elif(count>0):
                    count=0
                consecYears.append(count)
            if len(consecYears)>0:        
                maximum = max(consecYears)
            else:
                maximum = 0
            newRow = [maximum,'1971-2020',fracAlloc[i]]
            df.loc[len(df.index)] = newRow
    
    #fut
    for i in range(len(pctList)):
        # Create lists of index values
        threshold = pctList[i]*790000
        #Iterate through dataframe
        for Sim in Sims:
            count = 0
            consecYears = []
            for Year in futYears:  ##change
                if(release_V.at[Year,Sim] < threshold):
                    count+=1
                elif(count>0):
                    count=0
                consecYears.append(count)
            if len(consecYears)>0:        
                maximum = max(consecYears)
            else:
                maximum = 0
            newRow = [maximum,'2021-2070',fracAlloc[i]]
            df.loc[len(df.index)] = newRow
            
    plt.figure()
    sns.stripplot(x="Year Range",y="Consecutive Years Below Threshold",data=df,size=8,jitter=.25,hue="Threshold Volume",dodge=True,linewidth=1,alpha=.5)
    print(df.groupby(["Year Range","Threshold Volume"])['Consecutive Years Below Threshold'].describe())
       
def reliability(release_df,pctList=[.2,.5,1],clist=['#8c510a','#d8b365','#c7eae5','#5ab4ac','#01665e']):  

    
    DesiredRelease = 790000
    nYears = 50   # number of years considered
    
    past_df = release_df.loc[1971:2020,:]    #slice df to only include desired years
    fut_df = release_df.loc[2021:2070,:]
    
    ClmSimNames = release_df.columns.values.tolist()    #create a list of climate models
    ColNames = ['Fraction of Years Below Threshold','Year Range','Threshold Volume']
    #df = pd.DataFrame(columns=ColNames, dtype=np.float64)          #create new/empty df w scenarios as index

    fracAlloc = []
    for i in range(len(pctList)):
        fracAlloc.append(str(int(pctList[i]*100))+'% of Full Allocation')
    
    data = []
    #past
    #add cols w fraction of years w full allocation released
    for i in range(len(pctList)):
        pctAlloc = pctList[i]*DesiredRelease
        for Scn in ClmSimNames:
            #past
            nAllocMet = past_df.apply(lambda x: True if x[Scn] < pctAlloc else False, axis=1)
            frac=len(nAllocMet[nAllocMet == True].index)/nYears   #count # of years below DesiredRelease
            data.append([frac,'1971-2020',fracAlloc[i]])
            #future
            nAllocMet = fut_df.apply(lambda x: True if x[Scn] < pctAlloc else False, axis=1)
            frac=len(nAllocMet[nAllocMet == True].index)/nYears   #count # of years below DesiredRelease
            data.append([frac,'2021-2070',fracAlloc[i]])

    df = pd.DataFrame(data,columns=ColNames)

    plt.figure()
    sns.stripplot(x="Year Range",y="Fraction of Years Below Threshold",data=df,size=8,jitter=.25,hue="Threshold Volume",dodge=True,linewidth=1,alpha=.5)  
    print(df.groupby(["Year Range","Threshold Volume"])["Fraction of Years Below Threshold"].describe())

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
        reshaped_df = reshapeVarBox(dfList[i],dfNames[i],startYr,endYr)
        if(first==True):
            allvars_df = reshaped_df
            first = False
        else:
            allvars_df=pd.merge(allvars_df,reshaped_df,how='left',on=['Simulation'])
    allvars_df['Year Range'] = str(startYr)+'-'+str(endYr)
    return(allvars_df)

def boxwplot(dfList,dfNames, histStart=1971, histEnd = 2020, futStart=2021, futEnd=2070):
    '''
    Box and whisker plot for past and future periods

    '''
    # average, add cols, join
    hist_joined_df = joinVarsBox(dfList,dfNames,histStart,histEnd) 
    fut_joined_df = joinVarsBox(dfList,dfNames,futStart,futEnd) 
    joined_df = pd.concat([hist_joined_df,fut_joined_df])

    #plot
    nVars=len(dfNames)
    fig, axes = plt.subplots(1, nVars,figsize=(10, 5))
      
    for i in range(0,(nVars)):
        sns.boxplot(ax=axes[i], data=joined_df, x='Year Range', y=dfNames[i])     # All simulations
    plt.tight_layout()

def pchange(dfList,dfNames,histStart=1971,histEnd=2020,futStart=2021,futEnd=2070):
    '''
    Outputs a table of percent change past to future periods
    '''
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
        

#--------------------OUTPUT FOR REPORT------------------------------#
# pd.options.display.max_columns = None # (uncomment this line to print full dataframes in console)
plt.style.use('tableau-colorblind10')
sns.set_context('talk')    # poster (huge), talk (big), notebook (medium), paper (small)


# 1) Past/Future Temp, Precip, San Marcial
box_dfList = [AvgPr_ft,AvgTas_degC,SanMar_kaf]
box_dfNames = ['Average Precipitation (ft/yr)','Average Temperature (degrees C)','Average Streamflow (kAF/yr)']  
boxwplot(box_dfList,box_dfNames)

# 1.5 Print some stats on percent change
plt.style.use('seaborn-colorblind')
pchangevars=[EBTas_degC,EBPr_ft,SanMar_kaf,Cab_Release_kaf,Cab_Gauge_af]
pchangenames=['EB Temp (degC)','EB Pr (ft/yr)','Streamflow (kAF)','Caballo Release (kAF)','Caballo Gauge']
pchange(pchangevars,pchangenames)
pchange(box_dfList,box_dfNames)

# 2) Across all models, whats the liklihood different amounts will be released over time?
drought2(Cab_Release_af,pctList=[1,.5,.25],startYr=1971,endYr=2070,clist=['#01665e','#d8b365','#8c510a'])

# 2.5) Across years, what is the liklihood of X release volume?

resStats(Cab_Release_af,pctList=[1,.5,.25],startYr=2021,endYr=2070,clist=['#01665e','#d8b365','#8c510a'])
#resStats(Cab_Release_af,pctList=[1,.5,.25],startYr=1971,endYr=2020,clist=['#01665e','#d8b365','#8c510a'])
reliability(Cab_Release_af, pctList=[1,.5,.25])

# 3) Consecutive years of drought
consec_drought(Cab_Release_af, pctList=[1,.5,.25],startYr=2021,endYr=2070,clist=['#01665e','#d8b365','#8c510a'])
#consec_drought(Cab_Release_af, pctList=[1,.5,.25],startYr=1971,endYr=2020,clist=['#01665e','#d8b365','#8c510a'])
## 2013 flow 169158 AF->around 20% full allocation
consec_drought2(Cab_Release_af, pctList=[1,.5,.25],clist=['#01665e','#d8b365','#8c510a'])

# 4) Correlations: Evap V/SA, Evap Vol/Temp, San Marcial/SA    
#dfList = [EB_SurfA_ac,EBTas_degC,Res_SEvap_af,SanMar_af]
#dfNames = ['EB Surface Area (af)','EB Temperature (degC)','Reservoir Surface Evap. (af/yr)','San Marcial (af)']    

# 5) Local/Upstream Contributions 
perChangePlt(Net_Local_af/1000, 'Change in Local Fluxes (kAF/yr)', SanMar_af/1000, 'Change in Streamflow (kAF/yr)')
perChangePlt(AvgTas_degC, 'Change in Local Temperature (deg C)', SanMar_af/1000, 'Change in Streamflow (kAF/yr)')
perChangePlt(AvgPr_ft, 'Change in Local Precipitation (ft/yr)', SanMar_af/1000, 'Change in Streamflow (kAF/yr)')

# 6) Comparative W-B terms
inOutPieCharts(Res_SPrecip_af,Res_SEvap_af,Cab_RO_af,SanMar_af,Cab_Release_af,startYr=1971,endYr=2020) 
inOutPieCharts(Res_SPrecip_af,Res_SEvap_af,Cab_RO_af,SanMar_af,Cab_Release_af,startYr=2021,endYr=2070)

# Print Some Stats
def printStats(df):
    df_past = df.loc[1971:2020,:]
    df_fut = df.loc[2021:2070,:]
    df_list = [df_past,df_fut]
    for frame in df_list:
        mean = frame.mean()
        stats = mean.describe(percentiles=[.05,.1,.25,.5,.75,.9,.95])
        print(stats)
        
print('AVERAGE PRECIP')    
printStats(AvgPr_ft*12)
print('AVERAGE TEMPERATURE')
printStats(AvgTas_degC)
print('STREAMFLOW')
printStats(SanMar_af)
print('RELEASES')
printStats(Cab_Release_kaf)


print('surf evap avg 1971-2070:' + str(Res_SEvap_af.loc[1971:2070].mean().mean()))
print('surf precip avg 1971-2070:' + str(Res_SPrecip_af.loc[1971:2070].mean().mean()))
print('ro avg 1971-2070:'+ str(Cab_RO_af.loc[1971:2070].mean().mean()))

print('surf evap avg 2021-2070:' + str(Res_SEvap_af.loc[1971:2020].mean().mean()))
print('surf precip avg 2021-2070:' + str(Res_SPrecip_af.loc[1971:2020].mean().mean()))
print('ro avg 2021-2070:'+ str(Cab_RO_af.loc[1971:2020].mean().mean()))

print('surf evap avg 2021-2070:' + str(Res_SEvap_af.loc[2021:2070].mean().mean()))
print('surf precip avg 2021-2070:' + str(Res_SPrecip_af.loc[2021:2070].mean().mean()))
print('ro avg 2021-2070:'+ str(Cab_RO_af.loc[2021:2070].mean().mean()))
