# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:32:13 2015

@author: bolaka
"""

# imports
import os
import re
import csv
import pandas as pd
import numpy as np
import math
# from fuzzywuzzy import fuzz
from mlclassificationlibs import *
# from wordcloud import WordCloud, STOPWORDS
import string
from scipy.stats import itemfreq

#class CVXText:

def setPath(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # setting work directory
    os.chdir(directory)

def rows(data):
    return len(set(data.index))

def normalizeColumns(data, columnsToNormalize):
    columns = list(names(data))
    temp = pd.DataFrame(data=np.zeros((0,len(columns))), columns=columns)
    
    for key, group in data.groupby(data.index):
        if group[columnsToNormalize[0]].count() > 1:
            for column in columnsToNormalize:
                # normalize the column
                maxim = group[column].max()
                minim = group[column].min()
                if minim == maxim:
                    print(key, 'tossed off')
                    continue
                daysfromspud = group[column].values
                group[column + '_norm'] = [ (x - minim) / (maxim - minim) for x in daysfromspud]
            
            temp = temp.append(group)
        
    return temp

def extractMudWtFeatures(data,name, featuresToSave, owner):
    print('Extract mud wt. features for',name)
    data['SUMMARYOPS'].fillna('NA', inplace=True)
    
    # diameter
    rules = []
    depths = []
    rates = []
    mud_wt = []
    viscosity = []
    pH = []
    bitDiam = []
   
    # TODO
    wob = []
    psi = []
    jetsize1 = []
    jetsize2 = []
    rpm = []
    mudused = []
    for cmt in data['SUMMARYOPS']:
        depthAtStart1 = re.search("^\s*([\d|,]{3,})'",cmt, re.I) # \s*[\w|\W]\s*drilling\s*([\d|.|-]{1,3})\s*'/hr
#        depthAtStart2 = re.search("^\s*([\d|,]{3,})'\s*[a-z|,]+\s*@?\s*(\d{1,3})'?\s*(fph)",cmt.lower()) #[a-z|,|\s|@]
        depthAtStart2 = re.search("^\s*([\d|,]{3,})'\s*[a-z|,|\s|@]*(\d{1,3})\s*('|fph|ft|rop|hr)",cmt, re.I) #[a-z|,|\s|@]
        depthAtStart3 = re.search("^\s*([\d|,]{3,})'.*hole\s*@\s*(\d{1,3})\s*('|fph|ft|rop|hr)",cmt, re.I) #[a-z|,|\s|@]
        
        depthAtStartROP1 = re.search("^\s*p\.o\.:?\s*drilling\s*(\d{1,3})\s*'/hr", cmt, re.M|re.I)
        depthAtStartROP2 = re.search("^\s*p\.o\.\s*drilling\s*(\d{1,3})\s*(-|to)\s*(\d{1,3})'/hr", cmt, re.M|re.I)
        
        drillbitDiam1 = re.search('^\s*bit\s*#\d,\s*(\d{1,2}[\s|-]\d\/\d)",', cmt, re.M|re.I)        
        drillbitDiam2 = re.search('^\s*bit\s*#\d,\s*(\d{1,2}[.]\d{2,3})",', cmt, re.M|re.I)   
        drillbitDiam3 = re.search('^\s*bit\s*#\d,\s*(\d{1,2})",', cmt, re.M|re.I)        
        
        atter = re.search("(drlg|drilling|drill)\s*[at|@|#|-|to|.]?\s*[\d|,]{3,}'",cmt.lower())
        
        atter4 = re.search("^(drlg|drilling|drill)\s*[a-z|/|\s|@]*([\d|,|.|-]{1,3})\s*('/hr|fph|ft/hr)[a-z|/|\s|@]*([\d|,]+)'",cmt.lower())         
        
        atter2 = re.search("^(drlg|drilling|drill)\s*[a-z|/|\s|@]*([\d|,|.|-]{1,3})\s*('/hr|'?\s*f?p?hr?|fph|'?\s*ft\s*/?\s*hr|'\s*per\s*hr)",cmt.lower()) #('/hr|ft/hr|fph)
        
        atter0 = re.search(".*hole[a-z|\s|@|,]*([\d|,]{3,})\s*('/hr|'?\s*fph|ft/hr|'\s*hr|'\s*per\shour|'\s*phr|'\s*rop)",cmt, re.I) 
        atter1 = re.search(".*hole[\s|@|,]*([\d|,]{4,})\s*'",cmt.lower()) #a-z|
        atter3 = re.search("^(drlg|drilling|drill).*hole[\s|a-z|@]*([\d|,|.|-]+)\s*('\s*/hour|'\s*/hr|'?\s*fph|ft\s*/hr)",cmt.lower()) 
        
        ahead = re.search("^(drlg|drilling|drill).?\s*[a-z|\s|,|@]*([\d|,]{3,})'\s*[a-z|@|,]*\s*(\d{1,3})\s*('|fph|ft)",cmt, re.I) # (@|at)?\s*
        ahead1 = re.search("^(drlg|drilling|drill).?\s*[a-z|\s|,|@|/]*([\d|,]{3,})'",cmt, re.I)
        fromregex = re.search("\s*(drlg|drill|drilling)\s*[a-z|,|@|\.|-|/]*\s*([\d|,]{3,}'?)\s*([to|-]{1})\s*([\d|,]{3,}'?)",cmt, re.I)
        diaregex = re.search("^\s*(drlg|drilling|drill)\s*(\d\s*\d/\d)\"[a-z|@|\.|\s]*([\d|,]{3,})'",cmt, re.I) #DRLG 8 3/4'' VERT @ 2029'   
        depthregex = re.search("depth\s*:\s*([\d|,]{2,})'?", cmt, re.I)
#        aheadopt = re.search("^(drlg|drilling).?\s*[a-z|\s|,|@]*([\d|,]{3,}')?[a-z|@|\s|,]*(\d{2,3})\s*('|fph|ft)",cmt.lower())

        outat = re.search("\s+out\s*(at|@)\s*([\d|,]{3,})'?",cmt.lower()) # Out at|@ 3445'
        inatmade = re.search("in\s*(at|@)\s*([\d|,]+)'?\.?\s*(made|drilled)\s*([\d|,]+)'",cmt.lower()) # In at|@ 475'.  Made 175
        csgsetat = re.search("csg\s*set\s*at\s*([\d|,]+\.?\d+)'?\.?",cmt.lower()) # Csg set at 10,250'.
        simpleat = re.search("@\s*([\d|,]{3,6})'", cmt, re.I)
        
        weight = re.search("mud wt.\s*([\d|.]+)",cmt, re.I)
        beforevisregex = re.search("\s*(\d{2})\s*vis\s+", cmt, re.I)
        vis = re.search("vis(\.|:)?\s*(\d{2})(,|\s*)",cmt, re.I)        
        ph = re.search("\s+ph:?\s*(\d{1,2})",cmt, re.I)   
        wobregex1 = re.search(",\s*wob\s*(\d+)\s*k", cmt, re.I)
        wobregex2 = re.search(",\s*wob\s*(\d+)\s*(-|to)\s*(\d+)\s*", cmt, re.M|re.I)

        wobregex3 = re.search(",\s*bit\s*weight\s*(\d+)\s*", cmt, re.I) # , bit weight 45K,
        wobregex4 = re.search(",\s*bit\s*weight\s*(\d+)\s*(-|to)\s*(\d+)\s*", cmt, re.M|re.I) # , bit weight 40-45K,
        
        wobregex5 = re.search(",\s*bit\s*wt\s*(\d+)\s*", cmt, re.I) # , bit weight 45K,
        
        psiregex = re.search("\s*pump\s*psi\s*(\d+)#", cmt, re.I)
        jetregex1 = re.search("\s*jet\s*size\s*(\d+)\s*x\s*(\d+)\.", cmt, re.I) # Jet size 3 x 14.
        jetregex2 = re.search("\s*jet\s*size\s*(\d+)\s*x\s*(\d+)\s*,\s*(\d+)", cmt, re.I) #  Jet size 1 x 9, 4 x 10, 4 x 11.
        jetregex3 = re.search("\s*jet\s*size\s*(\d+)\s*x\s*(\d+)\s*,\s*(\d+)\s*x\s*(\d+)\s*,\s*(\d+)\s*x\s*(\d+)\.", cmt, re.I) #  Jet size 1 x 9, 4 x 10, 4 x 11.
        jetregex4 = re.search("\s*jet\s*size\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\.", cmt, re.I) # Jet size 12, 12, 13.
        
        mudusedregex = re.search("mud\s*used\s*:(.*)\.", cmt, re.I)        
        
        rule = None
        depth = 0
        rate = None
        bitDia = None
        if depthAtStart2:
#            print(depthAtStart2.group(), '========> ', depthAtStart2.group(1), '     ', depthAtStart2.group(2)) # 
            depth = re.sub(",", '', depthAtStart2.group(1)).strip()
            rate = depthAtStart2.group(2)
            rule = 'depthAtStart2'
        elif depthAtStart3:
#            print(depthAtStart3.group())
            depth = re.sub(",", '', depthAtStart3.group(1)).strip()
            rate = depthAtStart3.group(2)
            rule = 'depthAtStart3'
        elif depthAtStart1:
            depth = re.sub(",", '', depthAtStart1.group(1)).strip()
            if depthAtStartROP2:
                rate = (float(depthAtStartROP2.group(1)) + float(depthAtStartROP2.group(3))) / 2.0
            elif depthAtStartROP1:
                rate = depthAtStartROP1.group(1)
#            print(depthAtStart1.group(), rate)
            rule = 'depthAtStart1'
        elif depthregex:
#            print(depthregex.group(), depthregex.group(1))
            depth = re.sub(",", '', depthregex.group(1)).strip()
            rule = 'depthregex'
        elif outat:
#            outat1 = re.findall("[\d|,]{3,}'", outat1.group())
            depth = re.sub(",", '', outat.group(2)).strip()
            rule = 'outat'
        elif inatmade:
#            inatmade1 = re.findall("[\d|,]+'", inatmade1.group())
            inat = float(re.sub(",", '', inatmade.group(2)).strip())
            made = float(re.sub(",", '', inatmade.group(4)).strip())
            depth = (inat + made)
            rule = 'inatmade'
        elif csgsetat:
#            print(csgsetat)
            depth = float(re.sub(",", '', csgsetat.group(1)).strip())
            rule = 'csgsetat'
        elif ahead:
#            print(ahead.group(), '                tvd=====', ahead.group(2), 'fph======', ahead.group(3))
            depth = re.sub(",", '', ahead.group(2)).strip()
            rate = ahead.group(3)
            rule = 'ahead'
        elif atter0:
#            print(atter0.group(), 'traitor! ', atter0.group(1))
            rate = atter0.group(1)
            rule = 'atter0'
        elif atter1:
#            print(atter1.group(), 'tvd --------------- ', atter1.group(1))
            depth = re.sub(",", '', atter1.group(1)).strip()
            rule = 'atter1'
        elif atter3:
#            print(atter3.group())
            minmax = atter3.group(2).split('-')
            if len(minmax) > 1:
#                print('only fph ===========', atter2.group())
                if minmax[0] and minmax[1]:
                    mini = float(minmax[0])
                    maxi = float(minmax[1])
                    rate = (mini + maxi) / 2.0
                elif minmax[1]:
                    rate = float(minmax[1])
            else:
                rate = re.sub("-", '', atter3.group(2)).strip()
            
            rule = 'atter3'
        elif atter4:
#            print(atter4.group(), '             fph=', atter4.group(2), ' tvd = ', atter4.group(4))
            depth = re.sub(",", '', atter4.group(4)).strip()
            rate = atter4.group(2)
            rule = 'atter4'
        elif atter2:
            minmax = atter2.group(2).split('-')
            if len(minmax) > 1:
#                print('only fph ===========', atter2.group())
                if minmax[0] and minmax[1]:
                    mini = float(minmax[0])
                    maxi = float(minmax[1])
                    rate = (mini + maxi) / 2.0
                elif minmax[1]:
                    rate = float(minmax[1])
            else:
                rate = re.sub("-", '', atter2.group(2)).strip()
#            print(atter2.group(), '            ROP=', rate)
            rule = 'atter2'
        elif fromregex:
            depth1 = float(re.sub("[,|']", '', fromregex.group(2)).strip())
            depth2 = float(re.sub("[,|']", '', fromregex.group(4)).strip())
            depth = max(depth1, depth2)
#            print(fromregex.group(), depth)
            rule = 'fromregex'
        elif ahead1:
#            print(ahead1.group(), ' tvd ===== ', ahead1.group(2))
            depth = re.sub(",", '', ahead1.group(2)).strip()
            rule = 'ahead1'
        elif diaregex:
#            print(diaregex.group(), diaregex.group(3))
            depth = re.sub(",", '', diaregex.group(3)).strip()
            rule = 'diaregex'
        elif depthAtStartROP2:
            rate = (float(depthAtStartROP2.group(1)) + float(depthAtStartROP2.group(3))) / 2.0
#            print(depthAtStartROP2.group(), '  ROP=', rate)
            rule = 'depthAtStartROP2'
        elif depthAtStartROP1:
            rate = depthAtStartROP1.group(1)
#            print(depthAtStartROP1.group(), '  ROP=', rate)
            rule = 'depthAtStartROP1'
        elif atter:
            depth = re.findall("[\d|,]{3,}'", atter.group())
            depth = re.sub("['|,]", '', depth[0]).strip()
#            print('old one ------------ ', atter.group(), 'tvd=', depth)
            rule = 'atter'
#        elif simpleat:
#            print(simpleat.group(), '  tvd=', simpleat.group(1))
##            if aheadopt.group(2) is not None:
#            depth = re.sub(",", '', simpleat.group(1)).strip()
##            rate = aheadopt.group(3)
#            rule = 'simpleat'
        
        
        depths.append(depth)
        rates.append(rate)
        rules.append(rule)
        
        if drillbitDiam1:
            fractions = re.split('-|/|\s',drillbitDiam1.group(1))
#            print(fractions)
            if (len(fractions) > 1):
                fractions = [int(s) for s in fractions]
                bitDia = fractions[0] + (fractions[1] / fractions[2] )
#                print(drillbitDiam1.group(1), '=========== >', bitDia)
        elif drillbitDiam2:
            bitDia = float(drillbitDiam2.group(1))
#            print(drillbitDiam2.group(1), '=========== >', bitDia)
        elif drillbitDiam3:
            bitDia = float(drillbitDiam3.group(1))
#            print(drillbitDiam3.group(1), '=========== >', bitDia)
        bitDiam.append(bitDia)
        
        w = float('nan')
        if weight:
#            w = re.sub('[a-z]+.\s*', '', weight[0]).strip()
            w = weight.group(1)
        mud_wt.append(float(w))
        
        v = float('nan')
        if vis:
#            vis = re.sub('[a-z]+.\s*', '', vis[0]).strip()
#            print(vis.group(), 'vis ======= ', vis.group(2))
            v = vis.group(2)        
        elif beforevisregex:
#            print(beforevisregex.group(), 'beforevisregex ======= ', beforevisregex.group(1))
            v = beforevisregex.group(1)

        viscosity.append(float(v))
        
        p = float('nan')
        if ph:
#            print(ph.group(), 'ph ======= ', ph.group(1))
#            ph = re.sub('[a-z]+.\s*', '', ph[0]).strip()
#            if ph:
#                p = ph
            p = ph.group(1)
        pH.append(float(p))
        
        temp = float('nan')
        if wobregex1:
#            print(wobregex1.group(), '--------->', wobregex1.group(1))
            temp = wobregex1.group(1)
        elif wobregex2:
            temp = ( float(wobregex2.group(1)) + float(wobregex2.group(3)) ) / 2.0
#            print(wobregex2.group(), '--------->', temp)
        elif wobregex3:
#            print(wobregex3.group(), '--------->', wobregex3.group(1))
            temp = wobregex3.group(1)
        elif wobregex4:
            temp = ( float(wobregex4.group(1)) + float(wobregex4.group(3)) ) / 2.0
#            print(wobregex4.group(), '--------->', temp)
        elif wobregex5:
#            print(wobregex5.group(), '--------->', wobregex5.group(1))
            temp = wobregex5.group(1)
        wob.append(float(temp))
        
        temp = float('nan')
        if psiregex:
#            print(psiregex.group(), '============>', psiregex.group(1))
            temp = psiregex.group(1)
        psi.append(temp)
        
        temp1 = temp2 = float('nan')
        if jetregex3:
#            print(jetregex3.group()) #, '===>', jetregex2.group(1), jetregex2.group(2), jetregex2.group(3)
            temp1 = ( float(jetregex3.group(1)) + float(jetregex3.group(3)) + float(jetregex3.group(5)) ) / 3.0
            temp2 = ( float(jetregex3.group(2)) + float(jetregex3.group(4)) + float(jetregex3.group(6)) ) / 3.0  
#            print(jetregex3.group(), '===>', temp1, 'x', temp2)
        elif jetregex4:
            temp2 = ( float(jetregex4.group(1)) + float(jetregex4.group(2)) + float(jetregex4.group(3)) ) / 3.0
#            print(jetregex4.group(), temp2) #, '===>', jetregex2.group(1), jetregex2.group(2), jetregex2.group(3)
        elif jetregex2:
            temp1 = jetregex2.group(1)
            temp2 = ( float(jetregex2.group(2)) + float(jetregex2.group(3)) ) / 2.0
#            print(jetregex2.group(), '===>', temp1, 'x', temp2)
        
        elif jetregex1:
#            print(jetregex1.group(), '============>', jetregex1.group(1), 'x', jetregex1.group(2))
            temp1 = jetregex1.group(1)
            temp2 = jetregex1.group(2)
        
        jetsize1.append(temp1)
        jetsize2.append(temp2)
        
        temp = None
        if mudusedregex:
            muds = [re.sub("[-|\d]", '', mud).strip() for mud in mudusedregex.group(1).split(',')] 
#            print(', '.join(muds))
            temp = mudusedregex.group(1)
        mudused.append(temp)
    
    featuresToSave.append('rule')
    data['rule'] = rules    
    
    featuresToSave.append('depth')
    data['depth'] = depths
    
    featuresToSave.append('ROP')
    data['ROP'] = rates

    featuresToSave.append('mud-wt')    
    data['mud-wt'] = mud_wt
    
    featuresToSave.append('viscosity')    
    data['viscosity'] = viscosity
    
    featuresToSave.append('pH')    
    data['pH'] = pH
    
    featuresToSave.append('WOB')    
    data['WOB'] = wob
    
    featuresToSave.append('pump psi')    
    data['pump psi'] = psi
    
    featuresToSave.append('mud used')    
    data['mud used'] = mudused
    
    featuresToSave.append('diameter')
    data['diameter'] = bitDiam
    
    featuresToSave.append('jet size 1')
    data['jet size 1'] = jetsize1
    
    featuresToSave.append('jet size 2')
    data['jet size 2'] = jetsize2
    
#    data = data[featuresToSave]
    data['type'] = owner
    data.to_csv(name, sep=',', encoding='utf-8')
    return data

#def extractMudWtFeatures(data,name, featuresToSave, owner):
#    print('Extract mud wt. features for',name)
#    data['SUMMARYOPS'].fillna('NA', inplace=True)
#    
#    # diameter
#    rules = []
#    depths = []
#    rates = []
#    mud_wt = []
#    viscosity = []
#    pH = []
#   
#    # TODO
#    wob = []
#    rpm = []
#    for cmt in data['SUMMARYOPS']:
#        starter = re.findall("^\s*[\d|,]+'",cmt)
#
#        atter = re.search("(drlg|drilling)\s*[at|@|#|-|to|.]?\s*[\d|,]{3,}'",cmt.lower())
#        
#        atter2 = re.search("(drlg|drilling)\s*[\w|/|\s]*@?\s*([\d|,|.|-]+)\s*('|fph|ft)?",cmt.lower()) #('/hr|ft/hr|fph)
#        
#        ahead = re.search("(drlg|drilling).?\s*[\w|/|\s]*@?\s*(@|at)?\s*([\d|,|.]{3,})'?\s*,?\s*[a-z|@]*\s*(\d{1,3})\s*('|fph|ft)?",cmt.lower())
#        
#        aheadopt = re.search("(drlg|drilling).?\s*[a-z|@]*\s*(@|at)?\s*([\d+|,|.])?'?[\s|,]*[a-z|@]*\s*(\d{2,3})\s*('|fph|ft)?",cmt.lower())
#
#        outat = re.search("out\s*(at|@)\s*[\d|,]{3,}'",cmt.lower()) # Out at|@ 3445'
##        outat2 = re.search("out\s*@\s*[\d|,]{3,}'",cmt.lower()) # Out @ 3445'
#        
#        inatmade = re.search("in\s*(at|@)\s*[\d|,]+'\.?\s*(made|drilled)\s*[\d|,]+'",cmt.lower()) # In at|@ 475'.  Made 175
##        inatmade2 = re.search("in\s*@\s*[\d|,]+'\.?\s*made\s*[\d|,]+'",cmt.lower()) # In @ 475'.  Made 175
#        
#        csgsetat = re.search("csg\s*set\s*at\s*[\d|,]+\.?\d+'\.?",cmt.lower()) # Csg set at 10,250'.
#        weight = re.findall("mud wt.\s*[\d|.]+",cmt.lower())
#        vis = re.findall("vis\s*[\d|.]+",cmt.lower())        
#        ph = re.findall("ph\s*[\d|.]+",cmt.lower())        
#        
#        rule = None
#        depth = 0
#        rate = 0
#        if len(starter) > 0:
##            print('starter =========== ', starter[0])
#            starter = re.sub("['|,]", '', starter[0]).strip()
#            depth = starter
#            rule = 'starter'
#        elif ahead:
##            print('both =========== ', ahead.group(), 'tvd=',  ahead.group(3), 'fph=', ahead.group(4))
#            if ahead.group(3) is not None:
#                depth = re.sub("[.|,]", '', ahead.group(3)).strip()
#            rate = re.sub("[.|,]", '', ahead.group(4)).strip()
#            rule = 'ahead'
#        elif atter:
#            depth = re.findall("[\d|,]{3,}'", atter.group())
#            depth = re.sub("['|,]", '', depth[0]).strip()
##            print('old one ------------ ', atter.group(), 'tvd=', depth)
#            rule = 'atter'
#        elif atter2:
#            minmax = atter2.group(2).split('-')
#            if len(minmax) > 1:
#                print('only fph ===========', atter2.group())
#                if minmax[0] and minmax[1]:
#                    mini = float(minmax[0])
#                    maxi = float(minmax[1])
#                    rate = (mini + maxi) / 2.0
#                elif minmax[1]:
#                    rate = float(minmax[1])
#            else:
#                rate = re.sub("-", '', atter2.group(2)).strip()
#            
#            rule = 'atter2'
#        elif aheadopt:
##            print('aheadopt =========== ', aheadopt.group(), 'tvd=',  aheadopt.group(3), 'fph=', aheadopt.group(4))
#            if aheadopt.group(3) is not None:
#                depth = re.sub("[.|,]", '', aheadopt.group(3)).strip()
#            rate = re.sub("[.|,]", '', aheadopt.group(4)).strip()
#            rule = 'aheadopt'
#        elif outat1:
#            outat1 = re.findall("[\d|,]{3,}'", outat1.group())
#            outat1 = re.sub("['|,]", '', outat1[0]).strip()
#            depth = outat1
#            rule = 'outat1'
#        elif outat2:
#            outat2 = re.findall("[\d|,]{3,}'", outat2.group())
#            outat2 = re.sub("['|,]", '', outat2[0]).strip()
#            depth = outat2
#            rule = 'outat2'
#        elif inatmade1:
#            inatmade1 = re.findall("[\d|,]+'", inatmade1.group())
#            inat = float(re.sub("['|,]", '', inatmade1[0]).strip())
#            made = float(re.sub("['|,]", '', inatmade1[1]).strip())
#            depth = (inat + made)
#            rule = 'inatmade1'
#        elif inatmade2:
#            inatmade2 = re.findall("[\d|,]+'", inatmade2.group())
#            inat = float(re.sub("['|,]", '', inatmade2[0]).strip())
#            made = float(re.sub("['|,]", '', inatmade2[1]).strip())
#            depth = (inat + made)
#            rule = 'inatmade2'
#        elif csgsetat:
##            print(csgsetat)
##            csgsetat = re.findall("[\d|,]{3,}'", csgsetat.group())
#            csgsetat = re.findall("[\d|,]+\.?\d+'", csgsetat.group())
#            csgsetat = float(re.sub("['|,]", '', csgsetat[0]).strip())
#            
#            depth = csgsetat
#            rule = 'csgsetat'
#        
#        depths.append(depth)
#        rates.append(rate)
#        rules.append(rule)
#        
#        w = float('nan')
#        if len(weight) > 0:
#            w = re.sub('[a-z]+.\s*', '', weight[0]).strip()
#        mud_wt.append(float(w))
#        
#        v = float('nan')
#        if len(vis) > 0:
#            vis = re.sub('[a-z]+.\s*', '', vis[0]).strip()
#            if vis:
#                v = vis
#        viscosity.append(float(v))
#        
#        p = float('nan')
#        if len(ph) > 0:
#            ph = re.sub('[a-z]+.\s*', '', ph[0]).strip()
#            if ph:
#                p = ph
#        pH.append(float(p))
#    
#    featuresToSave.append('rule')
#    data['rule'] = rules    
#    
#    featuresToSave.append('depth')
#    data['depth'] = depths
#    
#    featuresToSave.append('ROP')
#    data['ROP'] = rates
#
#    featuresToSave.append('mud-wt')    
#    data['mud-wt'] = mud_wt
#    
#    featuresToSave.append('viscosity')    
#    data['viscosity'] = viscosity
#    
#    featuresToSave.append('pH')    
#    data['pH'] = pH
#    
#    data = data[featuresToSave]
#    data['type'] = owner
#    data.to_csv(name, sep=',', encoding='utf-8')
#    return data

#def extractMudWtFeatures(data,name, featuresToSave):
#    print('Extract regex features for',name)
#    data['SUMMARYOPS'].fillna('NA', inplace=True)
#    
#    # diameter
#    depths = []
#    mud_wt = []
#    for cmt in data['SUMMARYOPS']:
#        starter = re.findall("^\s*[\d|,]+'",cmt)
#        atter = re.findall("DRLG(.*?)[@|AT|#]\s[\d|,]{3,}'",cmt.upper())
#        weight = re.findall("mud wt.\s*[\d|.]+",cmt.lower())
#        print(starter, atter, weight)
#        depth = 0
#        if len(starter) > 0:
#            depth = starter[0]
#        elif len(atter) > 0:
#            atter = re.sub('[A-Z|@]\s+', '', atter[0]).strip()
#            depth = atter
#            
#        depths.append(depth)
#        
#        w = ''
#        if len(weight) > 0:
#            w = re.sub('[a-z]+.\s*', '', weight[0]).strip()
#        mud_wt.append(w)
#    
#    data['drill_depth_by_day'] = depths
#    data['mud_wt'] = mud_wt
#    
#    data = data[featuresToSave]
#    data.to_csv(name, sep=',', encoding='utf-8')

def extractFeatures(data,name,bag):
    print('Extract features for',name)
    data['SUMMARYOPS'].fillna('NA', inplace=True)
    indx = data.index
    data['wellid'] = pd.Categorical.from_array(indx).labels

    data['nlines'] = [len(cmt.splitlines()) for cmt in data['SUMMARYOPS']]
    data['nwords'] = [len(cmt.split()) for cmt in data['SUMMARYOPS']]
    data['starts-with-DRI..'] = [cmt.upper().startswith( 'DRI' ) for cmt in data['SUMMARYOPS']]
    
    # diameter
    diametersCsg = []
    for cmt in data['SUMMARYOPS']:
        diaFrac = re.findall('R[U|A]N\s+\d{1,2}[\s|-]\d\/\d', cmt.upper() )
        diaDec = re.findall('R[U|A]N\s+\d{1,2}[.]\d{2,3}', cmt.upper() )
#        print(diaFrac, diaDec)
        dia = ''
        if len(diaFrac) > 0:
            dia = re.sub('[A-Z|&.]+', '', diaFrac[0]).strip()
        elif len(diaDec) > 0:
            dia = re.sub('[A-Z|&]+', '', diaDec[0]).strip()
#        print(dia)
        dia = re.sub('[A-Z|&,]+', '', dia).strip()
        if dia:
            fractions = re.split('-|/|\s',dia)
#            print(fractions)
            if (len(fractions) > 1):
                fractions = [int(s) for s in fractions]
                diam = fractions[0] + (fractions[1] / fractions[2] )
            else:
                diam = float(dia)
        else:
            diam = 0
        diametersCsg.append(diam)
    
    data['CASING_DIAM'] = diametersCsg
    for word in bag:
#        print(word)
        
        ratio = [fuzz.partial_token_set_ratio( re.sub(r'[\s"\\]', ' ', cmt).strip().upper(), word ) for cmt in data['SUMMARYOPS']]
        data['is' + word] = np.array([r > 70 for r in ratio]) * 1
    data.to_csv(name, sep=',', encoding='utf-8')

def drawTimelines(owner, data, onlyPerfect):
    normalized_depth = 10500
    barWidth=2
    
    # define colors
    colorCOOP2 = '#FBA073'  # COOP 2-string - light orange
    colorCOOP3 = '#F12D0A'  # COOP 3-string - dark orange
    colorNOJV2 = '#9ECAF5'  # NOJV 2-string - light blue
    colorNOJV3 = '#0A34F1' # NOJV 3-string - deep blue    
    
    print('Processing ' + owner + ' wells:')
    
    # drop rows with missing
    data = data.dropna()
    
    # drop duplicate wells
#    data[idCol] = data.index
#    data.drop_duplicates(subset=idCol, take_last=True, inplace=True)
#    del data[idCol]
    
    colnames = names(data)
    rowslen = str(len(data.index))
    
    isThreeString = False
    if any("Inter" in s for s in colnames):
        isThreeString = True
        
    print(rowslen,isThreeString)

    # set the colors based on string & owner
    if owner == 'COOP' and isThreeString:
        groupColor = colorCOOP3
    elif owner == 'COOP' and isThreeString != True:
        groupColor = colorCOOP2
    elif owner == 'NOJV' and isThreeString: 
        groupColor = colorNOJV3
    elif owner == 'NOJV' and isThreeString != True:
        groupColor = colorNOJV2
        
    # the normalized drilling times
    normSDtimes = []
    normPDtimes = []
    if isThreeString:
        normIDtimes = []
    totalDtimes = []
        
    # the normalized drilling TVDs
    normSDtvds = []
    normPDtvds = []
    if isThreeString:
        normIDtvds = []
        
    # normalize time and TVD per well    
    for index, row in data.iterrows():
        
        # normalize drilling TVDs
        try:
            # normalization factor
            scale = normalized_depth/row['TVD']
            surfaceTVD = float(row['Surf string TVD'])
            productionTVD = float(row['Prod string TVD'])
            if isThreeString:
                interTVD = float(row['Inter string TVD'])
                
        except ValueError:
            print(row)
            
        # normalize drilling times
        normSDtimes.append(row['Surf D days'] * scale)
        normPDtimes.append(row['Prod D days'] * scale)
        if isThreeString:
            normIDtimes.append(row['Inter D days'] * scale)
            totalDtimes.append(row['Surf D days'] * scale + row['Prod D days'] * scale + row['Inter D days'] * scale + row['Surf C days'] + row['Inter C days'] +  row['Prod C days'])
        else:
            totalDtimes.append(row['Surf D days'] * scale + row['Prod D days'] * scale + row['Surf C days'] * scale +  row['Prod C days'])
            
            
        normSDtvds.append(surfaceTVD * scale)
        normPDtvds.append(productionTVD * scale)
        if isThreeString:
            normIDtvds.append(interTVD * scale)
            
          
            
    # add the new normalized drilling times to data
    data['normalized surface drill times'] = normSDtimes
    if isThreeString:
        data['normalized intermediate drill times'] = normIDtimes
    data['normalized production drill times'] = normPDtimes
    data['Total Drilling Times'] = totalDtimes
    
    # add the new normalized drilling TVDs to data
    data['normalized surface drill TVDs'] = normSDtvds
    if isThreeString:
        data['normalized intermediate drill TVDs'] = normIDtvds
    data['normalized production drill TVDs'] = normPDtvds
    
    if onlyPerfect == False:
        # plot the timelines
        count = 0
        for index, row in data.iterrows():
        
            if any(pd.isnull(row)):
                continue
        
            x = []
            x.append(0)
            x.append(row['normalized surface drill times'])
            x.append(x[len(x)-1] + row['Surf C days'])
            if isThreeString:
                x.append(x[len(x)-1] + row['normalized intermediate drill times'])
                x.append(x[len(x)-1] + row['Inter C days'])
            x.append(x[len(x)-1] + row['normalized production drill times'])
            x.append(x[len(x)-1] + row['Prod C days'])
            
            y = []
            y.append(0)
            y.append(-row['normalized surface drill TVDs'])
            y.append(-row['normalized surface drill TVDs'])
            if isThreeString:
                y.append(-row['normalized intermediate drill TVDs'])
                y.append(-row['normalized intermediate drill TVDs'])
            y.append(-row['normalized production drill TVDs'])
            y.append(-row['normalized production drill TVDs'])
            
            if isThreeString:
                label = rowslen + ' ' + owner + ' 3-string wells'
            else:
                label = rowslen + ' ' + owner + ' 2-string wells'
            if count == 1:
                plt.plot(x, y, linestyle='--', color=groupColor, label=label)  
            else:
                plt.plot(x, y, linestyle='--', color=groupColor)
            
            count += 1
    
    # plotting the perfect well
    bestSDwell = data.ix[data['normalized surface drill times'].argmin()]
    bestSDtime = bestSDwell['normalized surface drill times']
    bestStvd = bestSDwell['normalized surface drill TVDs']
    bestSCtime = min(data['Surf C days'])
    if isThreeString:
        bestIDwell = data.ix[data['normalized intermediate drill times'].argmin()]
        bestIDtime = bestIDwell['normalized intermediate drill times']
        bestItvd = bestIDwell['normalized intermediate drill TVDs']
        bestICtime = min(data['Inter C days'])
        
    bestPDwell = data.ix[data['normalized production drill times'].argmin()]
    bestPDtime = bestPDwell['normalized production drill times']
    bestPtvd = bestPDwell['normalized production drill TVDs']
    bestPCtime = min(data['Prod C days'])
    
    x = []
    x.append(0)
    x.append(bestSDtime)
    x.append(x[len(x)-1] + bestSCtime)
    if isThreeString:
        x.append(x[len(x)-1] + bestIDtime)
        x.append(x[len(x)-1] + bestICtime)
    x.append(x[len(x)-1] + bestPDtime)
    x.append(x[len(x)-1] + bestPCtime)
    
    y = []
    y.append(0)
    y.append(-bestStvd)
    y.append(-bestStvd)
    if isThreeString:
        y.append(-bestItvd)
        y.append(-bestItvd)
    y.append(-bestPtvd)
    y.append(-bestPtvd)
    
    if isThreeString:
        label = 'Perfect Well ' + owner + ' 3-string'
    else:
        label = 'Perfect Well ' + owner + ' 2-string'
    plt.plot(x, y, color=groupColor, linewidth=3, label = label)
    
    plt.xlabel('Time (days)')
    plt.ylabel('Depth (feet)')
    plt.title('Time/depth plots')
    plt.legend()
    
    plotTitle = 'groupNperfect-well-' + owner + '3-string'
    if isThreeString and onlyPerfect:
        plotTitle = 'perfect-well-' + owner + '3-string'    
    elif isThreeString != True and onlyPerfect:
        plotTitle = 'perfect-well-' + owner + '2-string'
    elif isThreeString != True and onlyPerfect != True:
        plotTitle = 'groupNperfect-well-' + owner + '2-string'
    
    F = plt.gcf() 
    DefaultSize = F.get_size_inches()
    F.set_size_inches( (DefaultSize[0]*2, DefaultSize[1]*2) )
    F.savefig(plotTitle + '.png', dpi = (200)) 
    print("Saving " + plotTitle)
    plt.show()
    
    # rename the columns and save the data
    mappings = {
        'Surf D days' : 'Actual Drilling time for surface string (days)',
        'Prod D days' : 'Actual Drilling time for production string (days)',
        'Surf C days' : 'Actual Case and cement time for surface string',
        'Prod C days' : 'Actual Case and cement time for production string',
        'Surf string TVD' : 'Actual Drilling depth for surface string (feet)',
        'Prod string TVD' : 'Actual Drilling depth for production string (feet)',
        'Inter D days' : 'Actual Drilling time for intermediate string (days)',
        'Inter C days' : 'Actual Case and cement time for intermediate string',
        'Inter string TVD' : 'Actual Drilling depth for intermediate string (feet)',
        'normalized surface drill times' : 'Normalized drill time for surface string',
        'normalized intermediate drill times' : 'Normalized drill time for intermediate string',
        'normalized production drill times' : 'Normalized drill time for production string',
        'normalized surface drill TVDs' : 'Normalized drill depth for surface string',
        'normalized intermediate drill TVDs' : 'Normalized drill depth for intermediate string',
        'normalized production drill TVDs' : 'Normalized drill depth for production string'
    };
    data = data.rename(columns=mappings)
    
    # reorder the columns
    colorder3 = ['TVD','Total Drilling Times','Actual Drilling time for surface string (days)',
                'Actual Drilling depth for surface string (feet)',
                'Actual Case and cement time for surface string',
                'Actual Drilling time for intermediate string (days)',
                'Actual Drilling depth for intermediate string (feet)',
                'Actual Case and cement time for intermediate string',
                'Actual Drilling time for production string (days)',
                'Actual Drilling depth for production string (feet)',
                'Actual Case and cement time for production string',
                'Normalized drill time for surface string',
                'Normalized drill depth for surface string',
                'Normalized drill time for intermediate string',
                'Normalized drill depth for intermediate string',
                'Normalized drill time for production string',
                'Normalized drill depth for production string']
        
    colorder2 = ['TVD','Total Drilling Times','Actual Drilling time for surface string (days)',
                'Actual Drilling depth for surface string (feet)',
                'Actual Case and cement time for surface string',
                'Actual Drilling time for production string (days)',
                'Actual Drilling depth for production string (feet)',
                'Actual Case and cement time for production string',
                'Normalized drill time for surface string',
                'Normalized drill depth for surface string',
                'Normalized drill time for production string',
                'Normalized drill depth for production string']
    
    drillingTimes2 = ['Total Drilling Times','Normalized drill time for surface string',
                          'Normalized drill time for production string']
    drillingTimes3 = ['Total Drilling Times','Normalized drill time for surface string',
                          'Normalized drill time for intermediate string',
                          'Normalized drill time for production string']
                
    if owner == 'NOJV':
        colorder2.append('RIGIDENTIFIER')
        colorder3.append('RIGIDENTIFIER')
        
    if isThreeString:
        data = data[colorder3]    
    else:
        data = data[colorder2]    
    
    print('Successfully normalized the wells with new columns as ', names(data))
    
    if isThreeString:
        data.to_csv(owner + 'normalized3.csv', sep=',', encoding='utf-8')
    else:
        data.to_csv(owner + 'normalized2.csv', sep=',', encoding='utf-8')
    
    if isThreeString:
        times = data[drillingTimes3]    
    else:
        times = data[drillingTimes2]        
    
    return (times)

def aggregateWells(owner, prefix,combined,code1,code2, testing):
    ID = 'WELLID'
    two_string_wells = []
    three_string_wells = []
    tossed_out_wells = []
    
    twoStrings = None
    threeStrings = None
    tossed = None
    
    testing[combined] = testing[code1] + ' ' + testing[code2]
    testWells200 = testing.groupby(testing.index)

    for name, group in testWells200:
#        print(name)
        extractDepth = False
        if 'TVD' in group.columns:   
            extractDepth = True
        
        rigIden = set(group['RIGSCALC'])
        strings = group[combined].value_counts()
        names = strings.index.values
        
        correct2 = ['SURFACE DRILLING','SURFACE CASING','PRODUCTION DRILLING','PRODUCTION CASING']
        match2 = set(names) & set(correct2)
        correct3 = ['SURFACE DRILLING','SURFACE CASING', 'INTERMEDIATE DRILLING', 'INTERMEDIATE CASING',
                    'PRODUCTION DRILLING', 'PRODUCTION CASING']
        match3 = set(names) & set(correct3)    
        previousRow = None             
        
        if len(names) == 4 and len(match2) == 4:
            depths = {'SURFACE' : float('nan'), 'PRODUCTION' : float('nan') }       
            
            if extractDepth and prefix != 'all wells':
                # extract depths for 2-string
#                print('2-string')
                
                # extract depths for 3-string
                for index, row in group.iterrows():
#                    print('checking', row['TVD'], row['RIGDAYSCALC'], depths[row[code1]])
                    if row[code2] == 'CASING' and np.isnan(depths[row[code1]]):  
                        if pd.isnull(row['TVD']) and previousRow is not None:
#                            depth = {}
                            depths[row[code1]] = previousRow['TVD']
#                            print('previous=', depths[row[code1]])
#                            depths.append(depth)
                        else:
#                            depth = {}
                            depths[row[code1]] = row['TVD']
#                            print('current=', depths[row[code1]])
#                            depths.append(depth)
                    previousRow = row
                print(depths)
                
            if group['TVD'].max() == depths['PRODUCTION']:
                dict1 = {}
                dict1['#'] = len(two_string_wells) + 1
                dict1[ID] = name
                dict1['TVD'] = group['TVD'].max()
                dict1['Surf D days'] = strings['SURFACE DRILLING']
                dict1['Surf C days'] = strings['SURFACE CASING']
                dict1['Prod D days'] = strings['PRODUCTION DRILLING']
                dict1['Prod C days'] = strings['PRODUCTION CASING']
                dict1['Surf string TVD'] = depths['SURFACE']
                dict1['Prod string TVD'] = depths['PRODUCTION']
                dict1['RIGIDENTIFIER'] = [', '.join(rig for rig in rigIden if rig is not np.nan) for rig in rigIden]
                two_string_wells.append(dict1)
            else:
                dict1 = {}
                dict1['#'] = len(tossed_out_wells) + 1
                dict1[ID] = name
                dict1['TVD'] = group['TVD'].max()
                dict1['RIGIDENTIFIER'] = [', '.join(rig for rig in rigIden if rig is not np.nan) for rig in rigIden]
                dict1['Total Drilling Time'] = round(group['DAYSFROMSPUDCALC'].max())
                
                tossed_out_wells.append(dict1)
            
        elif len(names) == 6 and len(match3) == 6:
            depths = {'SURFACE' : float('nan'), 'INTERMEDIATE' : float('nan'), 'PRODUCTION' : float('nan') }
            
            if extractDepth and prefix != 'all wells':
#                print('3-string')
                
                # extract depths for 3-string
                for index, row in group.iterrows():
#                    print('checking', row['TVD'], row['RIGDAYSCALC'], depths[row[code1]])
                    if row[code2] == 'CASING' and np.isnan(depths[row[code1]]):
                        if pd.isnull(row['TVD']) and previousRow is not None:
#                            depth = {}
                            depths[row[code1]] = previousRow['TVD']
#                            print('previous=', depths[row[code1]])
#                            depths.append(depth)
                        else:
#                            depth = {}
                            depths[row[code1]] = row['TVD']
#                            print('current=', depths[row[code1]])
#                            depths.append(depth)
                    previousRow = row
                print(depths)
            
            if group['TVD'].max() == depths['PRODUCTION']:
                dict1 = {}
                dict1['#'] = len(three_string_wells) + 1
                dict1[ID] = name
                dict1['TVD'] = group['TVD'].max()
                dict1['Surf D days'] = strings['SURFACE DRILLING']
                dict1['Surf C days'] = strings['SURFACE CASING']
                dict1['Inter D days'] = strings['INTERMEDIATE DRILLING']
                dict1['Inter C days'] = strings['INTERMEDIATE CASING']
                dict1['Prod D days'] = strings['PRODUCTION DRILLING']
                dict1['Prod C days'] = strings['PRODUCTION CASING']
                dict1['Surf string TVD'] = depths['SURFACE']
                dict1['Inter string TVD'] = depths['INTERMEDIATE']
                dict1['Prod string TVD'] = depths['PRODUCTION']
                dict1['RIGIDENTIFIER'] = [', '.join(rig for rig in rigIden if rig is not np.nan) for rig in rigIden]
                three_string_wells.append(dict1)
            else:
                dict1 = {}
                dict1['#'] = len(tossed_out_wells) + 1
                dict1[ID] = name
                dict1['TVD'] = group['TVD'].max()
                dict1['RIGIDENTIFIER'] = [', '.join(rig for rig in rigIden if rig is not np.nan) for rig in rigIden]
                dict1['Total Drilling Time'] = round(group['DAYSFROMSPUDCALC'].max())
                
                tossed_out_wells.append(dict1)
        
        else:
            depths = {'PRODUCTION' : 'NA'}
            
#            if extractDepth:
#                print('tossed')
                
            dict1 = {}
            dict1['#'] = len(tossed_out_wells) + 1
            dict1[ID] = name
            dict1['TVD'] = group['TVD'].max()
            dict1['RIGIDENTIFIER'] = [', '.join(rig for rig in rigIden if rig is not np.nan) for rig in rigIden]
            dict1['Total Drilling Time'] = round(group['DAYSFROMSPUDCALC'].max())
            
            tossed_out_wells.append(dict1)

    if len(two_string_wells) > 0:
        twoStrings = pd.DataFrame(two_string_wells)   
        twoStrings.set_index([ID],inplace=True)
        count2 = len(set(twoStrings.index))
        print('\n', count2, '2-string',owner, prefix, 'wells extracted')
        twoStrings.to_csv(prefix + '-' + owner + '-2string.csv', sep=',', encoding='utf-8')

    if len(three_string_wells) > 0:
        threeStrings = pd.DataFrame(three_string_wells)   
        threeStrings.set_index([ID],inplace=True)
        count3 = len(set(threeStrings.index))
        print(count3, '3-string', owner, prefix, 'wells extracted')
        threeStrings.to_csv(prefix + '-' + owner + '-3string.csv', sep=',', encoding='utf-8')
    
    if len(tossed_out_wells) > 0:
        tossed = pd.DataFrame(tossed_out_wells)   
        tossed.set_index([ID],inplace=True)
        countTossed = len(set(tossed.index))
        print(countTossed, owner, prefix, 'wells tossed','\n\n')
        tossed.to_csv(prefix + '-' + owner + '-tossed.csv', sep=',', encoding='utf-8')
    
    result = {}
    if twoStrings is not None:
        result['2-string-wells'] = twoStrings
        
    if threeStrings is not None:
        result['3-string-wells'] = threeStrings
    
    if tossed is not None:
        result['tossed'] = tossed
    
    return result
    
def drawWordCloud(textColumn):
    textColumn.fillna('NA', inplace=True)
    text = ''
    for comment in textColumn:
        if comment != 'NA':
#            lines = comment.splitlines()
            text += ''.join(comment.lower())
#    print(text)
    more_stopwords = list(string.ascii_lowercase)
#    stopwords = STOPWORDS.union(more_stopwords)
#    print(stopwords)
    wordcloud = WordCloud(max_words=30,width=3000,height=2000,stopwords=STOPWORDS.union(more_stopwords),background_color='white').generate(text)
    # Open a plot of the generated image.
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    return( { 'freq' : pd.DataFrame(wordcloud.words_), 'text' : text } )

def orderFormations(zones, zoneid, zonename, depthTopCol, depthBtmCol,):
    # we need to order the zone categories for labeling
    codes = zones.groupby([zoneid])
    zoneDepths = []
    for code, group in codes:
        btm = group[depthBtmCol].mean()
        top = group[depthTopCol].mean()
        name = group[zonename].value_counts().idxmax() 
#        score = top + btm / 2
#        score = btm
#        score = top
        zoneDepths.append( (code,name,top,btm) )
    
    sorteds = pd.DataFrame(sorted(zoneDepths, key=lambda zone: zone[2]), columns=['zone', 'name', 'top', 'btm'])
    return sorteds

def addFormations(data, zones, depthCol, depthTopCol, depthBtmCol, idcol, threshold, zoneid, zonename):
    sorteds = orderFormations(zones, zoneid, zonename, depthTopCol, depthBtmCol)
    zonesOrder = sorteds['zone'].values
    mappings = {}
    mappings['before formations start'] = 0
    for i, zone in enumerate(zonesOrder):
#        print(i, zone)
        mappings[zone] = i + 1
    mappings['beyond 200 ft of formations'] = len(zonesOrder)    
    
    sorteds.to_csv('zone-orders.csv', sep=',', encoding='utf-8',header=True,index=False)
    zoneGrps = zones.groupby([idcol])    
    
    # new data frame to hold the formations appended
    columns = list(names(data))
    columns.append('formation')
    formations = pd.DataFrame(data=np.zeros((0,len(columns))), columns=columns)
    
    for index, group in data.groupby(data.index):
        if index in zoneGrps.groups:
            zoneinfo = zoneGrps.get_group(index)
            
            # group by depthtop & depthbtm
            layers = zoneinfo.groupby([depthTopCol, depthBtmCol], sort=False)        
#            print('well' ,index, 'has', len(layers.groups), 'in formations data')
            
            layersList = []
            for idx, row in group.iterrows():
                depth = row[depthCol]
    
                count = 0
                prevLayer = None
                for (top, btm), layer in layers:
#                    print('depth = ', depth, ', top = ', top, ', bottom = ', btm) #prevLayer, count, 
                    if depth < top:
                        if count == 0:
                            # means upper strata
                            layersList.append('before formations start')
                        else:
                            # means missed layer; need to pick previous layer
                            layersList.append(prevLayer)
                        break
                    elif depth > top and depth <= btm:
                        zone = ', '.join( set(layer[zoneid].values) )
                        layersList.append(zone)
                        break
                    elif depth > btm and count == (len(layers.groups) - 1):
                        if (depth - btm < threshold):
                            layersList.append(', '.join( set(layer[zoneid].values) ))
                        else:
                            layersList.append('beyond 200 ft of formations')
                    count += 1
                    prevLayer = ', '.join( set(layer[zoneid].values) )
    
    
#            print(len(group[depthCol]), len(layersList), layersList )
            group['formation'] = layersList
            formations = formations.append(group)
    
#    categorical = pd.Categorical.from_array(formations['formation'])
#    formations['formation_code'] = categorical.labels
#    print(categorical.levels.values)
#    print(categorical.labels)
#    print(zonesOrder)
    formations['formation_code'] = formations['formation'].map(mappings)
    return formations

def prepareSeriesGeos(data, colx, coly, colformation, colLost, removeMissing, outliers):
    df = data
#    df = df.loc[condition]    
    
    if removeMissing:
        df = df.loc[ df[colx] > 0 ]
        df = df.loc[ df[coly] > 0 ]
        
        # remove outliers
        print('removing outliers beyond ', outliers, ' standard deviations...')
        df = df.loc[ np.abs( df[coly] - df[coly].mean() ) <= ( outliers * df[coly].std() ) ] #keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
        
    x = df[colx].values
    y = df[coly].values
    metric = df[colLost].values
    geo = df[colformation].values
    
    return pd.DataFrame({'x': pd.Series(x),'y':pd.Series(y), 'metric': metric, 'geo': geo })

def prepareSeries1(data, colx, coly, metric, removeMissing, outliers):
    df = data
#    df = df.loc[condition]    
    
    if removeMissing:
        df = df.loc[ df[colx] > 0 ]
        df = df.loc[ df[coly] > 0 ]
        
        # remove outliers
        print('removing outliers beyond ', outliers, ' standard deviations...')
        df = df.loc[ np.abs( df[coly] - df[coly].mean() ) <= ( outliers * df[coly].std() ) ] #keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
        
    x = df[colx].values
    y = df[coly].values
    metric = df[metric].values
    
    return pd.DataFrame({'x': pd.Series(x),'y':pd.Series(y), 'metric': metric })

def prepareSeries(data, colx, coly, condition, removeMissing, outliers):
    df = data
    df = df.loc[condition]    
    
    if removeMissing:
        df = df.loc[ df[colx] > 0 ]
        df = df.loc[ df[coly] > 0 ]
        
        # remove outliers
        print('removing outliers beyond ', outliers, ' standard deviations...')
        df = df.loc[ np.abs( df[coly] - df[coly].mean() ) <= ( outliers * df[coly].std() ) ] #keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
    
    x = df[colx].values
    y = df[coly].values
    return x, y

# plot a multi-series scatter plot
def multiseriesScatter(x0, y0, x1, y1, series0Label, series1Label, plotTitle, ylimits, xlimits):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.scatter(x0, y0, s=10, c='b', marker="o", label=series0Label) #
    ax.scatter(x1, y1, c='r', marker="o", label=series1Label) #s=10, 
    ax.set_ylim(ylimits)
    ax.set_xlim(xlimits)
    plt.legend(loc='upper left')
    plt.title(plotTitle)
    
    F = plt.gcf() 
    DefaultSize = F.get_size_inches()
    F.set_size_inches( (DefaultSize[0]*2, DefaultSize[1]*2) )
    F.savefig(plotTitle + '.png', dpi = (200)) 
    print("Saving " + plotTitle)
    plt.show()

def plotDensityMatrix1(data, deltaX, nameY, deltaY, prefix):
    print('depth vs ', nameY)
    x = data['x'].values
    y = data['y'].values
    lost = data['metric'].values
    
    minX = int(min(x))
    maxX = int(max(x))
    minY = int(min(y))
    maxY = int(max(y))
    print('x between ', minX, ' & ', maxX, '; y between ', minY, ' and ', maxY)
    
#    binsX = math.floor((maxX - minX) / deltaX)
#    histX = np.histogram(x, binsX)[1]
#    print('no of bins in x = ', binsX, ' at:\n', histX.tolist())
#    binnedX = np.digitize(x, histX)
#    orderedX = np.unique(binnedX)
    orderedX = np.unique(x)  
    if deltaX > 0:
        orderedX = list(range(minX, maxX, deltaX))
    labelsX = []
    print('classes in x = ',len(orderedX))
    
    labelsY = []
    orderedY = np.unique(y)    
    if deltaY > 0:
        orderedY = list(range(minY, maxY, deltaY))
    
    print('there should be ', len(orderedX), ' columns and ', len(orderedY), ' rows')
    d = pd.DataFrame({'x': pd.Series(x),'y':pd.Series(y), 'lost':pd.Series(lost) }) #'y-bins':pd.Series(binnedY),'x-bins': pd.Series(binnedX), 
#    d.to_csv(nameY + '-mappings.csv', sep=',', encoding='utf-8', header=True,index=False)
    
    totals = np.zeros( shape=(len(orderedY),len(orderedX)) ) #( len(orderedY), len(orderedX) )
    lostmuds = np.zeros( shape=(len(orderedY),len(orderedX)) ) #( len(orderedY), len(orderedX) )

    for i, y in enumerate(orderedY):
        labelY = "%.2f" % y
        conditionY = (d['y'] == y)
        if deltaY > 0:
            labelY = str(y) + " to " + str(y + deltaY)
            conditionY = (d['y'] >= y) & (d['y'] < (y + deltaY))
            if ( (i + 1) == len(orderedY)):
                conditionY = (d['y'] >= y) & (d['y'] <= (y + deltaY))

        labelsY.append(labelY)
        for j, x in enumerate(orderedX):
            conditionX = (d['x'] >= x) & (d['x'] < (x + deltaX))
            
            if ( (j + 1) == len(orderedX) ):
                conditionX = (d['x'] >= x) & (d['x'] <= (x + deltaX))      
            
            if i == 0:
                labelX = str(x) + "' to " + str(x + deltaX) + "'"
#                print(labelX)
                labelsX.append(labelX)
            
            nolostmud = d[conditionY & conditionX & (d['lost'] == 0)]
            lostmud = d[conditionY & conditionX & (d['lost'] == 1)]  
            red = len(lostmud)
            blue = len(nolostmud)
            total = red + blue
#            print('(',y,',',x,')',' at ','(',i,',',j,')', ' red = ', red, ' total = ', total ) # ' blue = ', blue, 
#            temp.append( str(red) + ' ; ' + str(blue) ) # (red / total) * 100
#            temp.append( total ) # (red / total) * 100
            if (total > 0):
#                matrix[i - 1][j - 1] = (red / total) * 100.0
                totals[i][j] = total
                lostmuds[i][j] = (red / total) #  red
#                temp.append( (red / total) * 100 ) # 
#            else:
#                temp.append(0)
#        matrix.append(temp)
    
#    with open(nameY + '-matrix.csv', 'a') as outcsv:   
#        writer = csv.writer(outcsv, delimiter=',') # quoting=csv.QUOTE_MINIMAL, quotechar='', lineterminator='\n'
#        for item in matrix:
#            writer.writerow(item)    

    column_labels = labelsY # orderedY
    row_labels = labelsX # orderedX
    data = lostmuds
    
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Reds, edgecolors='k') # plt.cm.Blues
    
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if totals[y, x] > 0:
                text = '%.0f' % totals[y, x]
                if data[y, x] > 0:
                    text = '%.0f' % (data[y, x] * totals[y, x]) + '/' + '%.0f' % totals[y, x] # 
                plt.text(x + 0.5, y + 0.5, text,
                     horizontalalignment='center',
                     verticalalignment='center',
                     )

#    plt.colorbar(heatmap)   
    
    # Here we use a text command instead of the title
    # to avoid collision between the x-axis tick labels
    # and the normal title position
#    plt.text(0.5,1.08,'depth vs ' + nameY,
#             fontsize=25,
#             horizontalalignment='center',
#             transform=ax.transAxes
#             )
    
    # x axis label
#    plt.text(0.5,1.14,'depths',
#             fontsize=20,
#             horizontalalignment='center',
#             transform=ax.transAxes
#             )
     
    # standard axis elements
    plt.ylabel(nameY,fontsize=20)
    plt.xlabel('depths',fontsize=20)
#    plt.title('depth vs' + nameY)
    
    fig = plt.gcf()
    fig.set_size_inches(len(orderedX)/1.5,len(orderedY)/1.5)    #15
    fig.set_dpi(300)
    
    # turn off the frame
    ax.set_frame_on(False)
    
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
    
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    
    # rotate the 
    plt.xticks(rotation=90)    
    
    ax.grid(False)

    # Turn off all the ticks
    ax = plt.gca()
    
    for t in ax.xaxis.get_major_ticks(): 
        t.tick1On = False 
        t.tick2On = False 
    for t in ax.yaxis.get_major_ticks(): 
        t.tick1On = False 
        t.tick2On = False      
    
#    plt.subplots_adjust(bottom=0.05, top=0.1)
    fig.savefig(nameY + '-' + prefix + '-matrix.png')  #, bbox_extra_artists=[xlabel], bbox_inches='tight'
#    plt.show()     
    
#    return ( {'m' : matrix, 'xlabels' : , 'ylabels' :  } )
    
def plotDensityMatrixGeos(data, deltaX, nameY, deltaY, formationDepths, prefix):
    print('geo vs ', nameY)
    x = data['x'].values
    y = data['y'].values
    lost = data['metric'].values
    geo = data['geo'].values
    
#    minX = int(min(x))
#    maxX = int(max(x))
    minY = int(min(y))
    maxY = int(max(y))
    
#    orderedX = list(range(minX, maxX, deltaX))
    orderedX = formationDepths.zone.values
    labelsX = formationDepths.name.values
#    print('classes in x = ',len(orderedX))
    
    labelsY = []
    orderedY = np.unique(y)    
    if deltaY > 0:
        orderedY = list(range(minY, maxY, deltaY))
    
    print('there should be ', len(orderedX), ' columns and ', len(orderedY), ' rows')
    d = pd.DataFrame({'x': pd.Series(x),'y':pd.Series(y), 'lost':pd.Series(lost), 'geo':pd.Series(geo) }) #'y-bins':pd.Series(binnedY),'x-bins': pd.Series(binnedX), 
    
    totals = np.zeros( shape=(len(orderedY),len(orderedX)) ) #( len(orderedY), len(orderedX) )
    lostmuds = np.zeros( shape=(len(orderedY),len(orderedX)) ) #( len(orderedY), len(orderedX) )

    for i, y in enumerate(orderedY):
        labelY = y
        conditionY = (d['y'] == y)
        if deltaY > 0:
            labelY = str(y) + " to " + str(y + deltaY)
            conditionY = (d['y'] >= y) & (d['y'] < (y + deltaY))
            if ( (i + 1) == len(orderedY)):
                conditionY = (d['y'] >= y) & (d['y'] <= (y + deltaY))

        labelsY.append(labelY)
#        for j, x in enumerate(orderedX):
        for j, x in enumerate(orderedX):
#            conditionX = (d['x'] >= x) & (d['x'] < (x + deltaX))
#            conditionX = (d['geo'] == x)
#            print(x, type(x), d['geo'].dtype)
            conditionX = (d['geo'].str.contains(x, regex=False))
            
#            if ( (j + 1) == len(orderedX) ):
#                conditionX = (d['x'] >= x) & (d['x'] <= (x + deltaX))      
            
#            if i == 0:
##                labelX = str(x) + "' to " + str(x + deltaX) + "'"
#                labelX = x
#                print(labelX)
#                labelsX.append(labelX)
            
            nolostmud = d[conditionY & conditionX & (d['lost'] == 0)]
            lostmud = d[conditionY & conditionX & (d['lost'] == 1)]  
            red = len(lostmud)
            blue = len(nolostmud)
            total = red + blue
#            print('(',y,',',x,')',' at ','(',i,',',j,')', ' red = ', red, ' total = ', total )
            if (total > 0):
                totals[i][j] = total
                lostmuds[i][j] = (red / total) #  red
    
    column_labels = labelsY # orderedY
    row_labels = labelsX # orderedX
    data = lostmuds
    
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Reds, edgecolors='k') # plt.cm.Blues
    
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if totals[y, x] > 0:
                text = '%.0f' % totals[y, x]
                if data[y, x] > 0:
                    text = '%.0f' % (data[y, x] * totals[y, x]) + '/' + '%.0f' % totals[y, x] # 
                plt.text(x + 0.5, y + 0.5, text,
                     horizontalalignment='center',
                     verticalalignment='center',
                     )

#    plt.colorbar(heatmap)   
    
    # Here we use a text command instead of the title
    # to avoid collision between the x-axis tick labels
    # and the normal title position
#    plt.text(0.5,1.08,'depth vs ' + nameY,
#             fontsize=25,
#             horizontalalignment='center',
#             transform=ax.transAxes
#             )
    
    # x axis label
#    plt.text(0.5,1.14,'depths',
#             fontsize=20,
#             horizontalalignment='center',
#             transform=ax.transAxes
#             )
     
    # standard axis elements
    plt.ylabel(nameY,fontsize=20)
    plt.xlabel('depths',fontsize=20)
#    plt.title('depth vs' + nameY)
    
    fig = plt.gcf()
    fig.set_size_inches(len(orderedX)/1.5,len(orderedY)/1.5)    #15
    fig.set_dpi(300)
    
    # turn off the frame
    ax.set_frame_on(False)
    
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
    
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    
    # rotate the 
    plt.xticks(rotation=90)    
    
    ax.grid(False)

    # Turn off all the ticks
    ax = plt.gca()
    
    for t in ax.xaxis.get_major_ticks(): 
        t.tick1On = False 
        t.tick2On = False 
    for t in ax.yaxis.get_major_ticks(): 
        t.tick1On = False 
        t.tick2On = False      
    
#    plt.subplots_adjust(bottom=0.05, top=0.1)
    fig.savefig(nameY + '-' + prefix + '-matrix.png')  #, bbox_extra_artists=[xlabel], bbox_inches='tight'
#    plt.show()     
    
#    return ( {'m' : matrix, 'xlabels' : , 'ylabels' :  } )
    
def plotDensityMatrix(data, nameX, nameY, deltaX, deltaY, prefix):
    print(nameX, ' vs ', nameY)
    x = data['x'].values
    y = data['y'].values
    metric = data['metric'].values
    
    minX = int(min(x))
    maxX = int(max(x))
    minY = int(min(y))
    maxY = int(max(y))
    print('x between ', minX, ' & ', maxX, '; y between ', minY, ' and ', maxY)
    
    orderedX = np.unique(x)  
    if deltaX > 0:
        orderedX = list(range(minX, maxX, deltaX))
    labelsX = []
    print('classes in x = ',len(orderedX))
    
    labelsY = []
    orderedY = np.unique(y)    
    if deltaY > 0:
        orderedY = list(range(minY, maxY, deltaY))
    
    print('there should be ', len(orderedX), ' columns and ', len(orderedY), ' rows')
    d = pd.DataFrame({'x': pd.Series(x),'y':pd.Series(y), 'metric':pd.Series(metric) })
    
    totals = np.zeros( shape=(len(orderedY),len(orderedX)) )

    for i, y in enumerate(orderedY):
        labelY = "%.0f" % y
        conditionY = (d['y'] == y)
        if deltaY > 0:
            labelY = str(y) + " to " + str(y + deltaY)
            conditionY = (d['y'] >= y) & (d['y'] < (y + deltaY))
            if ( (i + 1) == len(orderedY)):
                conditionY = (d['y'] >= y) & (d['y'] <= (y + deltaY))

        labelsY.append(labelY)
        for j, x in enumerate(orderedX):
            conditionX = (d['x'] == x)
            if deltaX > 0:
                conditionX = (d['x'] >= x) & (d['x'] < (x + deltaX))
                if ( (j + 1) == len(orderedX) ):
                    conditionX = (d['x'] >= x) & (d['x'] <= (x + deltaX))      
            
            if i == 0:
                labelX  = "%.0f" % x
                if deltaX > 0:
                    labelX = str(x) + "' to " + str(x + deltaX) + "'"     
                labelsX.append(labelX)
            
            metric_value = d.loc[conditionY & conditionX, 'metric'].sum()
#            print(metric_value, type(metric_value))
#            print('(',y,',',x,')',' at ','(',i,',',j,')', ' metric = ', metric_value)
            if (metric_value > 0):
                totals[i][j] = metric_value
    
    column_labels = labelsY 
    row_labels = labelsX 
    data = (totals - totals.mean()) / (totals.max() - totals.min())
#    print(data)
    
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Reds, edgecolors='k') # plt.cm.Blues
    
    for y in range(totals.shape[0]):
        for x in range(totals.shape[1]):
            if totals[y, x] > 0:
#                print('(',y,',',x,')', ' metric = ', totals[y, x])
                text = '%.0f' % totals[y, x]
                plt.text(x + 0.5, y + 0.5, text,
                     horizontalalignment='center',
                     verticalalignment='center',
                     )

    plt.colorbar(heatmap)   
    
    # Here we use a text command instead of the title
    # to avoid collision between the x-axis tick labels
    # and the normal title position
#    plt.text(0.5,1.08,'depth vs ' + nameY,
#             fontsize=25,
#             horizontalalignment='center',
#             transform=ax.transAxes
#             )
    
    # x axis label
#    plt.text(0.5,1.14,'depths',
#             fontsize=20,
#             horizontalalignment='center',
#             transform=ax.transAxes
#             )
     
    # standard axis elements
    plt.ylabel(nameY,fontsize=20)
    plt.xlabel(nameX,fontsize=20)
#    plt.title('depth vs' + nameY)
    
    fig = plt.gcf()
    fig.set_size_inches(len(orderedX)/1.5,len(orderedY)/1.5)    #15
    fig.set_dpi(300)
    
    # turn off the frame
    ax.set_frame_on(False)
    
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
    
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    
    # rotate the 
#    plt.xticks(rotation=90)    
    
    ax.grid(False)

    # Turn off all the ticks
    ax = plt.gca()
    
    for t in ax.xaxis.get_major_ticks(): 
        t.tick1On = False 
        t.tick2On = False 
    for t in ax.yaxis.get_major_ticks(): 
        t.tick1On = False 
        t.tick2On = False      
    
#    plt.subplots_adjust(bottom=0.05, top=0.1)
    fig.savefig(nameX + 'vs' + nameY + '-' + prefix + '.png')  #, bbox_extra_artists=[xlabel], bbox_inches='tight'
#    plt.show()     
