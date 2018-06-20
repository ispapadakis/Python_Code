# -*- coding: utf-8 -*-
# ---
# title: "Credit Limit Optimization Model"
# author: "Yanni Papadakis"
# ---
# Reimplementing R - lpSolveAPI Based - Model


# Import PuLP modeler functions

import pulp as mp

# INPUT PARAMETERS --------------------------------------------------------

# Parameter File 1
# <File Contents>
# size          Applicant Company Size (based on Number of Employees)
# risk          Applicant Company Risk Segment (based on Scorecard)
# credlim       Applicant Company Credit Limit Assigned (based on Historical Accounts Receivable Total)
# n             Exp Cases in Segment
# wotot         Exp Total Writeoff Amount in Segment
# wodiff_mr     Reduction in Exp Writeoff Amount in Segment After Manual Review
# expneed       Expected Total Credit Need Given Credit Limit
# exprev        Expected Revenue from Segment
# cum_n         Cumulative Exp Cases (Sums Segments with Lower Limit)
# cum_wotot     Cumulative Writeoff Total (Sums Segments with Lower Limit)
# cum_wodiff_mr Cumulative Reduction in Exp Writeoff Amount After Manual Review (Sums Segments with Lower Limit)
# cum_expneed   Cumulative Expected Credit Need Given Credit Limit (Sums Segments with Lower Limit)
# cum_exprev    Cumulative Expected Revenue from Segment (Sums Segments with Lower Limit)
# over_n        Total Exp Cases with Needs Over Limit
with open("clo_pulp_data.csv") as f:
    # Load Title Line
    ttl = f.readline().rstrip().split(',')
    varType = [str,str,float,int,float,float,float,float,int,float,float,float,float,float]
    rawData = dict()
    for line in f:
        lstr = line.rstrip().split(',')
        l = list(map(lambda f,x: f(x),varType,lstr))
        rawData[tuple(l[:3])] = dict(zip(ttl[3:],l[3:]))

# Scale Parameters in File 1
#   by a thousand
varsScale = {x:1e-3 for x in 
   ("n","cum_n","over_n")}
#   by a million
varsScale.update({x:1e-6 for x in 
    ('wotot','wodiff_mr','expneed','exprev','cum_wotot','cum_wodiff_mr','cum_expneed','cum_exprev')})

def scaleElemByFactor(_d, factorDict):
    from copy import deepcopy
    d = deepcopy(_d)
    for k in factorDict:
        d[k] *= factorDict[k]
    return d

scaledData = dict()
for k in rawData:
    scaledData[k] = scaleElemByFactor(rawData[k], varsScale)
  
# Parameter File 2
# <File Contents>
# size      Company Size in Segment (based on Number of Employees)
# risk      Company Risk in Segment (based on Scorecard)
# wor       Writeoff Ratio
# rvr       Revenue Ratio
# mrwor     Writeoff Ratio if Reviewd Manually
# n_tot     Exp Total Cases Per Year
with open("clo_pulp_pars.csv") as f:
    # Load Title Line
    paramNames = f.readline().rstrip().split(',')
    paramData = dict()
    paramType = [str,str,float,float,float,int]
    for line in f:
        lstr = line.rstrip().split(',')
        l = list(map(lambda f,x: f(x),paramType,lstr))
        paramData[tuple(l[:2])] = dict(zip(paramNames[2:],l[2:]))

# Other Parameters

# ManualReview Percent Limit
#   Assume 50% Acceptance Rate
mrLimit = 0.05 * 0.50

# Max Expected Writeoff Percent Limit
maxwoLimit = 0.5

# Include Cases if Revenue Rate > min(Writeoff Rate, MR Writeoff Rate) [MR Writeoff Rate is Always Min]

# Combinations Accepted
combAcceptable = [comb for comb in paramData if paramData[comb]['mrworate'] < paramData[comb]['revrate']]

# Size Segments
sizeseg = sorted({s for s,_ in combAcceptable})

# Risk Segments
riskseg = list('HML')
nRiskSeg = len(riskseg)

# Reduce Data Frame to Acceptable Cases Only - Use Scaled Values
modelData = dict((comb,scaledData[comb]) for comb in scaledData if comb[:2] in combAcceptable)

# List of Model Cases
casesInModel = modelData.keys()
# Cases By (Size,Risk) Combination
caseComb = dict()
for comb in combAcceptable:
    caseComb[comb] = sorted(case for case in casesInModel if case[:2] == comb)

                
# Max Total Clients to Manually Review (as limit of acceptable cases)
am = int( mrLimit * sum(modelData[case]['n'] for case in modelData) / varsScale['n']) * varsScale['n']

# Max Expected Writeoff (as limit out of all current cases)
wr = int(maxwoLimit * sum(rawData[case]['wotot'] for case in rawData)) * varsScale['wotot']


# Problem Variable Definition  
# Maximize Revenue Given Limit in Expected Writeoff   
prob = mp.LpProblem("Credit Limit Optimization",mp.LpMaximize)

# Declare Variables

# Binary Variable X: Accept Group Automatically
x = mp.LpVariable.dicts("In",casesInModel,0,1,mp.LpBinary)

# Binary Variable Y: Direct Group to Manual Review
y = mp.LpVariable.dicts("MR",casesInModel,0,1,mp.LpBinary)

# Declare Objective Function (Needs to be first equation in model)

# OBJECTIVE: Maximize Revenue by Segment Selection
prob += sum(modelData[case]['exprev'] * x[case] for case in casesInModel), "Exp Revenue Objective"

# SUBJECT TO:

# Constraint: Maximum Manual Review
# Scale Up to Help Solver Recognize Constraint
prob += sum(modelData[case]['over_n'] * y[case] for case in casesInModel) <= am, "Lim Manual Total"

# Constraint: Select One Only X Per (Size,Risk) Combination
for comb in combAcceptable:
    prob += sum(x[case] for case in caseComb[comb]) <= 1, "Lim 1 X {}".format(comb)

# Constraint: Select One Only Y Per (Size,Risk) Combination
for comb in combAcceptable:
    prob += sum(y[case] for case in caseComb[comb]) <= 1, "Lim 1 Y {}".format(comb)

# Constraint: Selected Y Level Exceeds X Level (MR After Auto Selection)
for comb in combAcceptable:
    prob += sum(case[2] * (x[case] - y[case]) for case in caseComb[comb]) <= 0, "X < Y {}".format(comb)

# Constraint: Maximum Total Risk Exposure
prob += sum([modelData[case]['cum_wotot'] * x[case] - modelData[case]['cum_wodiff_mr'] * y[case] for case in casesInModel]) <= wr, "Lim WO Total Auto"

# Constraints: Risk Ranking - CL(L) > CL(M) > CL(H)
# CL(M) > CL(H)
for size in sizeseg:
    if (size,'M') in combAcceptable and (size,'H') in combAcceptable:
        prob += sum(case[2] * x[case] for case in caseComb[(size,'M')]) >= \
                   sum(case[2] * x[case] for case in caseComb[(size,'H')]) , "RR M>H {}".format(size)
# CL(L) > CL(M)
for size in sizeseg:
    if (size,'L') in combAcceptable and (size,'M') in combAcceptable:
        prob += sum(case[2] * x[case] for case in caseComb[(size,'L')]) >= \
                   sum(case[2] * x[case] for case in caseComb[(size,'M')]) , "RR L>M {}".format(size)

# Constraints: Size Ranking - CL(Size Up) > CL(Size)
for risk in riskseg:
    for size,sizeUp in zip(sizeseg,sizeseg[1:]):
        if (size,risk) in combAcceptable and (sizeUp,risk) in combAcceptable:
            prob += sum(case[2] * x[case] for case in caseComb[(sizeUp,risk)]) >= \
                       sum(case[2] * x[case] for case in caseComb[(size,risk)]) , "SR {}>{} {}".format(sizeUp,size,risk)

# The problem data is written to an .lp file
prob.writeLP("clo_pulp.lp")

# The problem is solved using PuLP's choice of Solver
prob.solve()

# Obtain Solution
for v in prob.variables():
    if v.varValue > 0: print(v.name, "=", v.varValue)

# Problem Status
print("\nStatus:", mp.LpStatus[prob.status],', Solution Time: {:.1f} sec'.format(prob.solutionTime))
print('Problem Size: {} Vars, {} Constraints / Obj Bound Reached:{:.9}'.format(len(x)+len(y),len(prob.constraints),prob.objective.value()))
print('\n\nProblem:',prob.name)
print('Objective Value: ${:.2f} million'.format(prob.objective.value()))
    
# Solution Reports
solnAuto = {i[:2]:i[2] for i in x if x[i].varValue}
solnMR = {i[:2]:i[2] for i in x if y[i].varValue}
tblAuto = [[solnAuto[(s,r)] if (s,r) in solnAuto else 0.0 for r in riskseg] for s in sizeseg]
tblMR = [[solnMR[(s,r)] if (s,r) in solnMR else 0.0 for r in riskseg] for s in sizeseg]

fmtTtl  = '{:3s} ' + ' {:>6}'*nRiskSeg
fmtLine = '{:3s} ' + ' {:6.0f}'*nRiskSeg
print('\nOptimal Credit Limits AUTO')
print(fmtTtl.format(' ',*riskseg))
for i,s in enumerate(sizeseg):
    print(fmtLine.format(s,*tblAuto[i]))
print('\nOptimal Credit Limits MR')
print(fmtTtl.format(' ',*riskseg))
for i,s in enumerate(sizeseg):
    print(fmtLine.format(s,*tblMR[i]))
print('\nPopulation Distribution')
print(fmtTtl.format(' ',*riskseg))
for i,s in enumerate(sizeseg):
    print(fmtLine.format(s,*[paramData[(s,r)]['n_tot'] for r in riskseg]))
    
print('\nManual Review Requirement (1,000)')
optmr = sum(modelData[case]['over_n'] * y[case].varValue for case in casesInModel)
print('Optimal {:.3} vs Limit {:.3}'.format(optmr,am))
    
print('\nTotal Exposure to Writeoffs ($M)')
optwo = sum([modelData[case]['cum_wotot'] * x[case].varValue - modelData[case]['cum_wodiff_mr'] * y[case].varValue for case in casesInModel])
print('Optimal {:.3} vs Limit {:.3}'.format(optwo,wr))

# Report
'''

Status: Optimal , Solution Time: 0.2 sec
Problem Size: 216 Vars, 82 Constraints / Obj Bound Reached:1.41385974


Problem: Credit Limit Optimization
Objective Value: $1.41 million

Optimal Credit Limits AUTO
          H      M      L
S0        0     50     50
S1        0     50     50
S2        0     50     50
S3       50     50     50
S4       50   9999   9999
S5     9999   9999   9999
S6     9999   9999   9999

Optimal Credit Limits MR
          H      M      L
S0        0   9999   9999
S1        0   9999   9999
S2        0   9999   9999
S3     9999   9999   9999
S4     9999   9999   9999
S5     9999   9999   9999
S6     9999   9999   9999

Population Distribution
          H      M      L
S0      385   1068   1492
S1      935   1293    386
S2     1466   2203    221
S3      725    819     51
S4      980   1184     46
S5      218    279      6
S6      386    900     18

Manual Review Requirement (1,000)
Optimal 0.242 vs Limit 0.306

Total Exposure to Writeoffs ($M)
Optimal 0.472 vs Limit 0.703
'''
