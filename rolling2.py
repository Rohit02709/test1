import requests
import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import timedelta,datetime

st.set_page_config(layout="wide")   
pnl_url=r'http://performance.squareoffbots.com/assets/json/sqbots_allData_21052021.json'
cap_url=r'http://performance.squareoffbots.com/assets/json/newCAp21052021.json'

pnl_dic=requests.get(pnl_url).json()
cap_dic=requests.get(cap_url).json()


retDic={}
for day in pnl_dic:
    if day in cap_dic:
        for strat in pnl_dic[day]:
            if strat in cap_dic[day]:
                if isinstance(cap_dic[day][strat],str):
                    cap_dic[day][strat]=cap_dic[day][strat].replace('e','')
                if isinstance(pnl_dic[day][strat],str):
                    pnl_dic[day][strat]=pnl_dic[day][strat].replace('e','')
                if float(cap_dic[day][strat])>0:
                    ret=(float(pnl_dic[day][strat])/float(cap_dic[day][strat]))*100
                    if strat not in retDic:
                        retDic[strat]={}
                    retDic[strat][day]=ret


retDf=pd.DataFrame.from_dict(retDic)
retDf.fillna(0,inplace=True)
retDf['date']=retDf.index
retDf['pd_date']=pd.to_datetime(retDf['date'],format='%Y-%m-%d')
retDf.sort_values('pd_date',inplace=True)
lastDate=pd.to_datetime(retDf.tail(1)['pd_date'].values[0]).strftime('%Y%m%d')



allStrats=list(retDic.keys())
for strat in allStrats:
    retDf['rolling7'+strat]=retDf[strat].rolling(7).sum()
    retDf['average7'+strat]=retDf['rolling7'+strat]/7
    retDf['rolling14'+strat]=retDf[strat].rolling(14).sum()
    retDf['average14'+strat]=retDf['rolling14'+strat]/7
retDf['date']=retDf.index
col1, col2 = st.columns(2)
stratName=col1.selectbox('Select a Strategy',tuple(allStrats[:1]+["All"]+allStrats[1:]))

timePeriod=col2.selectbox('Time Period',(7,14))
w1=650;w2=1200
h1=600;h2=600
fig=px.line(retDf,x="date",y=[f'rolling{timePeriod}{stratName}',f'average{timePeriod}{stratName}'],title=stratName+' '+str(timePeriod)+' days (% Returns)',width=w2, height=h2)
col1.plotly_chart(fig)