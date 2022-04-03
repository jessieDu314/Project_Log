# %%%% clean 2
import pandas as pd
from datetime import datetime
import re
dta_houses  = pd.read_csv("dta_airbnb_clean1.csv").drop(['Unnamed: 0'],axis=1)

# %%%%
# already cleaned
dta_clean = dta_houses[['headline','current_price','original_price']].copy()

# fees
tmp = dta_houses[['clean_fee','service_fee','occupancy_fee']].copy()

for i in range(len(tmp)):
    if tmp.iloc[i].isna().any() == True and tmp.iloc[i].isna().all() == False:
        tmp.iloc[i] = tmp.iloc[i].fillna(0)    
        
# self check in
tmp = dta_houses.self_checkin.fillna('0')
tmp = tmp.apply(lambda x: re.sub(r'([A-z]+)',r'1',x))

# parking
dta_clean['free_P_on_property'] = '0'
dta_clean['free_P_on_property'] = dta_clean.free_P_on_property[dta_clean.amenity.str.contains('Free parking on premises').copy()==True].replace('0','1')
dta_clean['free_P_on_property'] = dta_clean['free_P_on_property'].fillna('0')

# private entrance
dta_clean['private_entrance'] = '0'
dta_clean['private_entrance'] = dta_clean.private_entrance[dta_clean.amenity.str.contains('Private entrance')==True].replace('0','1')
dta_clean['private_entrance'] = dta_clean['private_entrance'].fillna('0')

# scenic view
scenic = []
for i in range(len(dta_houses)):
    view = []
    views = re.findall('([A-z]+)( view,)',dta_houses.amenity[i])
    for j in views:
        view += [j[0]]
    scenic.append(','.join(view))
    
# pets
dta_clean['pets'] = '0'
dta_clean['pets'] = dta_clean.pets[dta_clean.amenity.str.contains('Pets')==True].replace('0','1')
dta_clean['pets'] = dta_clean['pets'].fillna('0')

#language
tmp = dta_houses.language.fillna('')
for i in range(len(tmp)):
    if tmp[i] == '':
        language = 0
    else:
        if 'English' in tmp[i]:
            language = len(tmp[i].split(',')) - 1
        else:
            language = len(tmp[i].split(','))
            

# cancel policy                                                                                                                                  
tmp = dta_houses.free_cancel.fillna('NAN')
tmp = tmp.apply(lambda x: re.sub(r'(for )|(before )',r'',x))
tmp = tmp.apply(lambda x: re.sub(r'48 hours',r'1',x))
for i in range(len(tmp)):
    try:
        tmp[i] = datetime.strptime(tmp[i],'%b %d').strftime('%m-%d')
    except:
        pass

tmp2 = dta_houses.check_in
tmp2 = pd.DataFrame(tmp2.apply(lambda x: datetime.strptime(x,'%m/%d/%Y').strftime('%m-%d')))
tmp2['cancel'] = tmp.copy()
cancel_window = []
for i in range(len(tmp2)):
    try:
        k = str(datetime.strptime(tmp2['check_in'][i], '%m-%d')-datetime.strptime(tmp2['cancel'][i],'%m-%d'))
        k = re.findall('([0-9]+ day[s]?)',k)[0]
    except:
        k = tmp2['cancel'][i]
    cancel_window.append(k)
    
# host join year
for i in range(len(dta_houses)):
    try:
        year = datetime.strptime(dta_houses.join_date[i],'%b-%y').strftime('%Y')
    except:
        try:
            year = datetime.strptime(dta_houses.join_date[i],'%B %Y').strftime('%Y')
        except:
            year = ''