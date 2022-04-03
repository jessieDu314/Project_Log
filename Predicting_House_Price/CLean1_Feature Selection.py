# %%%%
import pandas as pd
import re
dta_houses = pd.read_csv("dta_airbnb_raw.csv").drop(['Unnamed: 0'],axis=1)

dta_houses = dta_houses[dta_houses.headline.isna() == False]
dta_houses = dta_houses[dta_houses.price.isna() == False]
dta_houses = dta_houses[dta_houses.amenities.isna() == False]

dta_houses    = dta_houses.reset_index()
del dta_houses['index']


# %%%%
dta_houses_clean = dta_houses[['headline']].copy()

# price
dta_houses_clean = dta_houses.price.str.extract('([0-9]+)( per night)')[0][0]

# fees
for i in range(len(dta_houses)):
    payment = re.sub(r'\\n',r" ",dta_houses.price.str.split(', \''))[i][0]
for j in payment:     
    try:
        cleaning = re.findall('(Cleaning)([A-z ]+\$)([0-9]+)',j)[0][2]
    except:
        cleaning  = ''
    try:
        service = re.findall('(Service)([A-z ]+\$)([0-9]+)',j)[0][2]
    except:
        service   = ''
    try:
        occupancy = re.findall('(Occupancy)([A-z ]+\$)([0-9]+)',j)[0][2]
    except:
        occupancy = ''   
        
# registration
for j in payment:
    try:
        c_in  = re.findall('(CHECK-IN )([0-9\/?]+)',j)[0][1]
    except:
        c_in = ''
    try:
        c_out = re.findall('(CHECKOUT )([0-9\/?]+)',j)[0][1]
    except:
        c_out = ''
        
# house_info
# type
for i in range(len(dta_houses)):
    h_type = re.findall('(.*)( hosted by )(.*)',dta_houses.house_info.str.split('\n')[i][0])[0][0]

# other house_info
for j in range(len(dta_houses)):    
    basics  = dta_houses.house_info.str.split('\n')[j][1]
    try:
        guest   = re.findall('([0-9]+)( guest[s]?)',basics)[0][0]
    except:
        guest   = ''
    try:
        bedroom = re.findall('([0-9]+ [A-z]+? bedroom[s]?|[0-9]+ bedroom[s]?)',basics)[0]
    except:
        bedroom = '' 
    try:    
        beds    = re.findall('([0-9]+)( [A-z]+)?( beds? )',basics)[0][0]
    except:
        beds    = ''
    try:        
        baths   = re.findall('([0-9.]+)( bath[s]?| [A-z]+ bath[s]?)',basics)[0][0]
    except:
        baths   = ''
        
# highlight
for i in range(len(dta_houses)):
    try:    
        highlight_text = re.sub(r'\\n',r" ",dta_houses.highlight[i].split(', \'')[0])
    except:
         highlight_text = ''
    enhanced_c = 0
    if 'Enhanced Clean' in highlight_text:
        enhanced_c = 1
    try:
        self_c   = re.findall('( in with the )([A-z]+)',highlight_text)[0][1]
    except:        
        self_c   = ''
    try:
        free_can = re.findall('(cancellation )(.*)(")',highlight_text)[0][1]
    except:
        free_can = ''
        
# sleep area
for i in range(len(dta_houses)):
    try:
        sleep_a = re.sub(r'\n',r'.',dta_houses.sleep_area[i])
    except:
        sleep_a = ''
    try:     
        sleep = re.sub(r"Where you'll sleep,",r'',sleep_a)
    except:
        try:
            sleep = re.sub(r"Sleeping arranarrangements,",r'',sleep_a)
        except: 
            try:
                sleep = re.sub(r"1 of 2 pages,1 / 2,",r'',sleep_a)    
            except:
                sleep = sleep_a    

# amenities
for i in range(len(dta_houses)):   
    amenity = re.sub(r'\n',r',',dta_houses.amenities[i])
    if 'Unavailable' in amenity:  
        try:
            offer       = re.findall('(What this place offers,)(.*)(Not included,)(.*)',amenity)[0][1]
        except:
            offer       = re.findall('(.*)(Not [A-z]+,)(.*)',amenity)[0][0]
        try:
            not_offer   = re.findall('(What this place offers,)(.*)(Not included,)(.*)',amenity)[0][3]
        except:
            not_offer   = re.findall('(.*)(Not [A-z]+,)(.*)',amenity)[0][2]
            
    else:
        try:
            offer       = re.findall('(What this place offers,)(.*)',amenity)[0][1]
        except:
            offer       = amenity
        not_offer   = ''  
        
# rating overview
for i in range(len(dta_houses)):  
    try:
        rating_text = dta_houses.rating[i].split('\n')[1]
    except:
        rating_text = ''
    try:
        rating = re.findall('([0-9].+)( \· )',rating_text)[0][0]
    except:
        rating = ''
    try:
        review_num = re.findall('([0-9]+)( review[s]?)',rating_text)[0][0]
    except:
        review_num = ''
        
# sub_ratings
for i in range(len(dta_houses)):
    try:
        sub_rating = re.sub(r'\n',r' ',dta_houses.sub_ratings[i])
        clean      = re.findall('(Cleanliness )([0-9.]+)',sub_rating)[0][1]
        accu       = re.findall('(Accuracy )([0-9.]+)',sub_rating)[0][1]
        commu      = re.findall('(Communication )([0-9.]+)',sub_rating)[0][1]
        loc        = re.findall('(Location )([0-9.]+)',sub_rating)[0][1]
        chek       = re.findall('(Check-in )([0-9.]+)',sub_rating)[0][1]
        val        = re.findall('(Value )([0-9.]+)',sub_rating)[0][1]
    except:
        clean      = ''
        accu       = ''
        commu      = ''
        loc        = ''
        chek       = ''
        val        = ''
        
# host
for i in range(len(dta_houses)):
    try:
        name  = re.findall('(Hosted by )(.*)(Joined)',dta_houses.host[i])[0][1]
    except:
        name = ''
    try:
        join  = re.findall('(Joined in )([A-z]+ [0-9]{4})',dta_houses.host[i])[0][1]
    except:
        join = ''
    try:
        host_r_num = re.findall('\d{4}(.*)',re.findall('(Joined in )([A-z]+ )(.*)( Reviews)',dta_houses.host[i])[0][2])[0]
    except:
        host_r_num = 0
    try:
        if 'Identity' in dta_houses.host[i]:        
            identity = 1
        else:
            identity = 0
    except:
        identity = 0
    try:
        if 'Superhost' in dta_houses.host[i]:
            super_h = 1
        else:
            super_h = 0
    except:
        super_h = 0
    try:
        lan = re.findall('(Languages?: )(.*)(Response rate)',dta_houses.host[i])[0][1]
    except:
        try:
            lan = re.findall('(Languages?: )(.*)(Response time)',dta_houses.host[i])[0][1]
        except:
            try:
                lan = re.findall('(Languages?: )(.*)(Contact)',dta_houses.host[i])[0][1]            
            except:
                lan = ''
    try:
        res_rate = re.findall('(Response rate: )([0-9]+%)',dta_houses.host[i])[0][1]
    except:
        res_rate = ''
    try:
        res_time = re.findall('(Response time: )([A-z ]+)(Contact)',dta_houses.host[i])[0][1]
    except:
        res_time = ''