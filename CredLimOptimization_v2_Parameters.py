# MODEL PARAMETERS

# epsilon (small number << 1)
epsilon = 1e-3

# Manual Review Parameters
#   ManualReview Percent Limit
mrLimit = 0.10
#   Acceptance Rate Varies by Risk Rating
mrAcceptRate = {'H':0.50, 'M':0.75, 'L':0.90}

# Max Expected Writeoff Percent Limit
maxwoLimit = 0.50

# Industry Segments
induseg = ['N1', 'N2', 'N3']

# Size Segments
sizeseg = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']

# Risk Segments
riskseg = ['H', 'M', 'L']

# Data File
# Data File Name
inputFile = "csv/clo_pulp_data_v2.csv"
# Data File Metadata
varType = [str,str,str,int,int,float,float,float,float,int,float,float,float,float,int,float,float,float,float]
index_slice = slice(None,4)
var_slice = slice(4,None)

# Scale Parameters
thouScale = ("n","cum_n","over_n")
millScale = ('wotot','wotot_mr','expneed','exprev','cum_wotot','cum_wotot_mr','cum_expneed','cum_exprev','over_wotot','over_wotot_mr','over_expneed','over_exprev')

# Results File
resultsFile = "csv/clo_pulp_data_v2_results.csv"
