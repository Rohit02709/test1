import requests
import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import timedelta,datetime
st.set_page_config(layout="wide")   
st.title("**♟**SQUAREOFF BOTS ROLLING RETURNS**♟**")
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




col1, col2 = st.columns(2)
allStrats=list(retDic.keys())

topDate=retDf.iloc[0]['date']



stratName=col1.selectbox('Select a Strategy',tuple(allStrats[:1]+["All"]+allStrats[1:]))

timePeriod=col2.selectbox('Time Period',(7,15,30,60,100,300,'All'))
if timePeriod=='All':
    timePeriod=len(retDf)
firstDate=datetime.strptime(lastDate,'%Y%m%d')-timedelta(days=int(timePeriod))

retDf=retDf[retDf['pd_date']>=firstDate]
retDf.reset_index(drop=True,inplace=True)
# for aS in allStrats:  
#     retDf[aS]=retDf[aS]-retDf.loc[0][aS]
for aS in allStrats:  
    retDf['cum_'+aS]=retDf[aS].cumsum()
for aS in allStrats:  
    retDf['cum_'+aS]=retDf['cum_'+aS]-retDf.loc[0]['cum_'+aS]

w1=750;w2=500
h1=600;h2=600
px.defaults.template = "plotly_dark"
plotly_theme="seaborn"
retDf['color']="green"
retDf['size']="5"
st.markdown("""<style>
    .big-font{
        font-size: 30px;
        font-weight: bold;
    }
</style>""",unsafe_allow_html=True)
if stratName=="All":
    counter=0
    for stratName in allStrats:
        counter+=1
        fig=px.line(retDf,x="date",y='cum_'+stratName,title=stratName+' '+str(timePeriod)+' days (% Returns)',width=w2, height=h2,template=plotly_theme)
        if counter%2==0:
            col1.plotly_chart(fig)
        else:
            col2.plotly_chart(fig)
else:
    cumfig=px.line(retDf,x="date",y='cum_'+stratName,title=stratName+' '+str(timePeriod)+' days (% Returns)',width=w1, height=h1,template=plotly_theme)
    finalCumPnl=retDf.tail(1)['cum_'+stratName].values[0]
    fig=px.scatter(retDf,x="date",y=stratName,title=stratName+' '+str(timePeriod)+' days (% Returns)',width=w1, height=h1)
    fig.update_traces(marker=dict(size=15,
                              line=dict(width=10,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    cumfig.data[0].line.color = 'rgb(10, 126, 116)'
    col2.markdown('<br><br><br><br>',unsafe_allow_html=True)
    col2.markdown('<p class="big-font">'+f'{stratName} {timePeriod} days Returns : {finalCumPnl:.2f} %' +'</p>',unsafe_allow_html=True)
    col2.markdown('<p class="big-font">'+f'{stratName} {timePeriod} days Pnl Per 300000 : Rs. {finalCumPnl*3000:.2f} '+'</p>',unsafe_allow_html=True)
    col1.plotly_chart(cumfig)
    col1.plotly_chart(fig)
# fig.update_layout(   xaxis=dict(
#         showline=True,
#         showgrid=False,
#         showticklabels=True,
#         linecolor='white',
#         linewidth=5,
#         ticks='outside',
#         tickfont=dict(
#             family='Arial',
#             size=14,
#             color='rgb(82, 82, 82)',
#         ),
#     ),
#     yaxis=dict(
#         showgrid=False,
#         zeroline=False,
#         showline=False,
#         showticklabels=False,
#     ),
#     autosize=False,
#     margin=dict(
#         autoexpand=False,
#         l=100,
#         r=20,
#         t=110,
#     ),
#     showlegend=False,
#   )



    