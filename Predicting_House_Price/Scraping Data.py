# %%%% import library
import time
import datetime
import pandas as pd
import re
from bs4 import BeautifulSoup
from selenium import webdriver
# %%%% Part I:  Find the links for all the available houses of each city
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 5)
pd.set_option('display.width',800)
path   = '/Users/Desktop/Brandeis/chromedriver'

driver = webdriver.Chrome(path)

# links of cities
city = ['new-york','los-angeles','san-francisco','chicago','washington','san-diego','las-vegas','san-jose','miami','boston',
        'houston','atlanta','phoenix','dallas','seattle','denver','austin','orlando','honolulu','philadelphia',
        'minneapolis','Portland','Baltimore','Provo','Raleigh','Sacramento','Nashville','San-Antonio','Salt-Lake-City','Tampa',
        'New-Orleans','Santa-Rosa','Madison','Charlotte','Bridgeport','Omaha','Colorado-Springs','St-Louis','Oxnard','Columbus',
        'Tucson','Virginia-Beach','Kansas-City','Durham','Charleston','Cape-Coral','Portland','New-Haven','El-Paso','Richmond']
city = [x.lower() for x in city]

# 15 page for each city
links_to_scrape  = []
for i in city:
    for j in range(15):
        template = "https://www.airbnb.com/s/"+i+"/homes?adults=2&checkin=&checkout=&tab_id=home_tab&refinement_paths%5B%5D=%2Fhomes&flexible_trip_dates%5B%5D=december&flexible_trip_dates%5B%5D=january&flexible_trip_dates%5B%5D=february&flexible_trip_dates%5B%5D=march&date_picker_type=flexible_dates&source=structured_search_input_header&search_type=search_query&flexible_trip_lengths%5B%5D=weekend_trip"
        if j > 0:
            more_page = "&items_offset="+str(j*20)+"&section_offset=2"
            template  = template + more_page                    
        links_to_scrape.append(template)

# scrapping urls
url_to_scrape = []
for l in range(len(links_to_scrape)):
    driver.get(links_to_scrape[l])
    time.sleep(15)
    elems = driver.find_elements_by_xpath("//a[@href]")
    for elem in elems:
        url_to_scrape.append(elem.get_attribute("href"))
        
dta_url_1 = pd.DataFrame(url_to_scrape)
dta_url_1 = dta_url_1[pd.DataFrame(url_to_scrape)[0].str.contains('https://www.airbnb.com/rooms/')]
dta_url_1 = dta_url_1[dta_url_1.duplicated()]

driver.close()
              
# %%%% Part II: Find features listed on the website that could be used for prediction
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 5)
pd.set_option('display.width',800)
path    = '/Users/Desktop/Brandeis/chromedriver'

driver  = webdriver.Chrome(path)
dta_url = pd.read_csv('urls.csv').drop(['Unnamed: 0'],axis = 1)
dta_url = dta_url.drop(1).reset_index()

# set empty data frame
dta_houses3              = pd.DataFrame()
dta_missing              = pd.DataFrame()
dta_saltlaksandmore      = pd.DataFrame()

houses                   = {}
houses['scrapping_date'] = []
houses['url']            = []
houses['headline']       = []
houses['house_info']     = []
houses['location']       = []
houses['highlight']      = []
houses['sleep_area']     = []
houses['amenities']      = []
houses['amenity_number'] = []
houses['price']          = []
houses['rating']         = []
houses['sub_ratings']    = []
houses['host']           = []

# scraping features
for url in dta_url.iloc[:,0].to_list():    
    driver.get(url)
    time.sleep(5)
    # date and url        
    houses['scrapping_date'] = datetime.datetime.now()
    houses['url']            = driver.current_url
    
    # headline
    try:
        houses['headline'] = driver.find_elements_by_xpath('//div[@class="_b8stb0"]')[0].text
    except:
        houses['headline'] = ''
    
    # house type
    try:
        houses['house_info'] = driver.find_elements_by_xpath('//div[@class="_tqmy57"]')[0].text
    except:
        houses['house_info'] = ''

    # location
    try:
        houses['location'] = driver.find_elements_by_xpath('//span[@class="_pbq7fmm"]')[0].text
    except:
        houses['location'] = ''
    
    # highlight
    try:
        houses['highlight'] = [driver.find_elements_by_xpath('//div[@data-plugin-in-point-id="HIGHLIGHTS_DEFAULT"]')[0].text,
                           driver.find_elements_by_xpath('//div[@data-plugin-in-point-id="HIGHLIGHTS_DEFAULT"]')[0].get_attribute('innerHTML')]
    except:
        houses['highlight'] = ''
    
    # sleep_area
    try:
        houses['sleep_area'] = driver.find_elements_by_xpath('//div[@class="_7ts20mk"]')[0].text
    except:
        try:            
            houses['sleep_area'] = driver.find_elements_by_xpath('//section[@role="region"]')[0].text
        except:
            houses['sleep_area'] = ''
    
    # amenities(normal+plus)
    try:
        houses['amenity_number'] = driver.find_elements_by_xpath('//div[@class="b6xigss dir dir-ltr"]')[0].text
    except:
        try:
            houses['amenity_number'] = driver.find_elements_by_xpath('//div[@class="_1dnh28d"]')[0].text
        except:
            houses['amenity_number'] = ''
    
    try:
        am_block = driver.find_elements_by_xpath('//div[@data-section-id="AMENITIES_DEFAULT"]')[0].text
        if am_block.find('Show all') > 0:
            driver.find_elements_by_xpath('(//a[@class="b1sec48q v7aged4 dir dir-ltr"])[1]')[0].click()
            time.sleep(4)
            houses['amenities'] = driver.find_elements_by_xpath('//div[@class="_17itzz4"]')[0].text
            driver.find_elements_by_xpath('//div[@class="_pa35zs"]')[0].click()
            time.sleep(4)
        else:
            houses['amenities'] = driver.find_elements_by_xpath('//div[@data-section-id="AMENITIES_DEFAULT"]')[0].text  
    except:        
        try:
            am_block = driver.find_elements_by_xpath('//div[@data-section-id="AMENITIES_PLUS"]')[0].text
            if am_block.find('Show all') > 0:
                for link in driver.find_elements_by_xpath('//a[@class="_3ly6pcs"]'):
                    if 'amenities' in link.text:
                        link.click()
                        time.sleep(4)
                houses['amenities'] = driver.find_elements_by_xpath('//div[@class="_17itzz4"]')[0].text
                driver.find_elements_by_xpath('//div[@class="_pa35zs"]')[0].click()
                time.sleep(4)  
            else:
                houses['amenities'] = driver.find_elements_by_xpath('//div[@data-section-id="AMENITIES_PLUS" ]')[0].text                   
        except:               
            houses['amenities'] = ''
            
            
    # price & fees
    try:
        for p in range(len(driver.find_elements_by_xpath('//span[@class="_14tkmhr"]'))):
            if len(re.findall(pattern="price de", string = driver.find_elements_by_xpath('//span[@class="_14tkmhr"]')[p].get_attribute('innerHTML'))) > 0:
                driver.find_elements_by_xpath('//span[@class="_14tkmhr"]')[p].click()
                time.sleep(4)
    except:
        pass
    try:
        houses['price'] = [driver.find_elements_by_xpath('//div[@class="_ud8a1c"]')[0].text,
                           driver.find_elements_by_xpath('//div[@class="_ud8a1c"]')[0].get_attribute('innerHTML')]
    except:
        houses['price'] = ''
    
    # ratings
    try:
        houses['rating'] = driver.find_elements_by_xpath('//span[@class="ttu4pt2 dir dir-ltr"]')[0].text
    except:
        houses['rating'] = ''
        
    # sub-ratings
    try:
        houses['sub_ratings'] = driver.find_elements_by_xpath('//div[@class="r1xdm4l6 dir dir-ltr"]')[0].text
    except:
        houses['sub_ratings'] = ''
    
    
    # house host    
    try:
        houses['host'] = BeautifulSoup(driver.find_elements_by_xpath('//div[@data-section-id="HOST_PROFILE_DEFAULT"]')[0].get_attribute('innerHTML')).text        
    except:
        houses['host'] = ''
            
driver.close()            
            
            
