"""
my activity dasbboard with streamlit
"""

#Import packages
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
from PIL import Image
plt.set_loglevel('WARNING')

#To organize pages
st.set_page_config(layout="wide")

#Download all csv into df
running = pd.read_csv("running.csv", sep=';')
#drop columns not number types for correlation later
running_new = running.copy()
col_object = running_new.select_dtypes(include=["object"]).columns.tolist()
for ele in col_object:
    running_new.drop([ele], axis=1, inplace=True)

floors = pd.read_csv("Floors.csv", sep=',')
sleep = pd.read_csv("Sleep.csv", sep=',')
steps = pd.read_csv("Steps.csv", sep=',')

#My sidebars : parameters
st.sidebar.markdown('<p class="header-style">NAME : Thuy AUFRERE</p>',
                     unsafe_allow_html=True
                     )

#Create pages from 0 to 3
selected_pages = st.sidebar.selectbox("Select the activity dashboard",
                                       ["Data Overwiew",
                                        "Running",
                                        "Walking (steps, floors) and Sleep"]
                                       )
#Prepare each pages dashboard from 0 to 2

#Page 0 : owerview
if selected_pages == "Data Overwiew":
    #0 - My welcome page : explaination about my dashboard
    st.title("MY HEALTH DASHBOARD")
    st.write("This dashboard shows some KPI regarding my health activities.")
    st.write("Data has been downloaded from [Garmin Connect] (https://connect.garmin.com/signin/) and the code for the dashboard can be found on my github : [Here] (https://github.com/T-AUF/PSBX).")

    #Download image
    pic_run = Image.open("Run.PNG")
    pic_steps = Image.open("Steps.PNG")
    pic_sleep = Image.open("Sleep.PNG")
    pic_walk = Image.open("Walk.PNG")

    col_run, col_steps, col_walk , col_sleep = st.columns(4)

    col_run.header("Running Activity")
    col_run.image(pic_run)

    col_steps.header("Floors Climbed")
    col_steps.image(pic_steps)

    col_walk.header("Walking Activity")
    col_walk.image(pic_walk)

    col_sleep.header("Sleeping Activity")
    col_sleep.image(pic_sleep)

    #download data
    st.subheader("My Garmin Data")
    with st.expander("See running data"):
        st.write("Below is my dataframe regarding my running activity")
        st.dataframe(running)
        
    with st.expander("See walking data"):
        st.write("Below is my dataframe regarding my walking activity")
        st.dataframe(steps)
        
    with st.expander("See floors climbed data"):
        st.write("Below is my dataframe regarding my floors climbed activity")
        st.dataframe(floors) 
        
    with st.expander("See sleeping data"):
        st.write("Below is my dataframe regarding my sleeping activity")
        st.dataframe(sleep)
    
    st.write("👍✔️😍 To help me to code : I highly recommend those websites for [Matplotlib with STREAMLIT](https://pythonwife.com/matplotlib-with-streamlit/), [Seaborn with STREAMLIT](https://pythonwife.com/seaborn-with-streamlit/) and the documentation of [STREAMLIT](https://docs.streamlit.io/).")
        
#Page 1 : Running
elif selected_pages == "Running":
    st.header("Running Activity")
    
    st.markdown("**Maximum** values : what is information ?")
    #create 6 columns for max
    distance, ville, duree, calories, pace = st.columns(5)
    
    distance.metric(label="Distance", value=str(running["Distance"].max())+" km")
    ville.metric(label="City", value=running.loc[running["Distance"].idxmax()]["Title"])
    duree.metric(label="Time", value=running.loc[running["Distance"].idxmax()]["Time"])
    calories.metric(label="Calories", value=running.loc[running["Distance"].idxmax()]["Calories"])
    pace.metric(label="Avg Pace", value=("{:,.2f}".format(running.loc[running["Distance"].idxmax()]["Avg Pace"]))+" km/hrs")
    
    st.markdown("**Minimum** values : what is information ?")
    #create 6 columns for min
    distance1, ville1, duree1, calories1, pace1 = st.columns(5)

    distance1.metric(label="Distance", value=str(running["Distance"].min())+" km")
    ville1.metric(label="City", value=running.loc[running["Distance"].idxmin()]["Title"]) 
    duree1.metric(label="Time", value=running.loc[running["Distance"].idxmin()]["Time"])
    calories1.metric(label="Calories", value=running.loc[running["Distance"].idxmin()]["Calories"])
    pace1.metric(label="Avg Pace", value= ("{:,.2f}".format(running.loc[running["Distance"].idxmin()]["Avg Pace"]))+" km/hrs")
    
    #Add comment area
    st.text_area("Comment on max and min values", 
                  '''
                  ⚠️ It looks like the minimum values had a issue during the recording.''',
                  height = 1
                   )
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write("What my running habits are ?")
    #plot days of running
    cols_var = ["activityMonthName", "activityDayName", "activityLevel"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.7, hspace=0.3)

    for i, ax in enumerate(axes.ravel()):
        if i > 3:
            ax.set_visible(False)
            continue
        sns.countplot(y = cols_var[i], data=running, ax=ax)
    st.pyplot(fig)
    
    #Add comment area
    st.text_area("Comment between date",  '''
                  It seems I run more from March to May than in summer (maybe because of the heat).
                  I run most of time on Saturday and Sunday.
                  I often run above 150 BPM.'''
                  )
    st.write(" ")
    st.write(" ")
    st.write(" ")
    
    #Add repartion year
    st.write("Repartition running between date and year")
    
    comp_date, comp_year = st.columns(2)
    with comp_date:
        st.write("Date from feb 2019 to dec 2021")
        fig = plt.figure(figsize=(15, 6))
        sns.lineplot(x = "Date", y = "Distance", data = running)
        st.pyplot(fig)
    
    with comp_year:
        st.write("Year 2019 to 2021")
        fig = plt.figure(figsize=(15, 6))
        sns.stripplot(x = "activityYear", y = "Distance", data = running)
        st.pyplot(fig)
    
    #Add comment area
    st.text_area("Comment on my habits",  '''
                  I runned more in 2019 than 2021.''',
                  height = 1
                  )
    
    st.write(" ")
    st.write(" ")
    st.write(" ")
    
    #Add heart plot
    st.write("My Heart Rate repartition")
    
    heart, comments_h = st.columns((2, (1)))
    with heart:
        fig = plt.figure(figsize = (20,6))
        sns.histplot(data = running, x = "Avg HR").set(title="Average HR repartition")
        st.pyplot(fig)
    
    with comments_h:
        st.text_area("Comment on HR repartition", 
                      '''
                      Most probable value during the run would be around 160 BPM.''',
                      height = 1
                     )             
    st.write(" ")
    st.write(" ")
    st.write(" ")              
    st.write("Let's see correlation between features")

    fig = plt.figure(figsize=(25, 6))
    # define the mask to set the values in the upper triangle to True
    mask = np.triu(np.ones_like(running_new.corr()))
    
    #cvim/max : delimiter, annot = yes and cmap = color type
    heatmap = sns.heatmap(running_new.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16)
    st.pyplot(fig)
    
    corr_distance, comment_c = st.columns((2, (1)))
    with corr_distance:
        fig1 = plt.figure(figsize=(12, 6))
        heatmap = sns.heatmap(running_new.corr()[['Distance']].sort_values(by='Distance', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
        heatmap.set_title('Features Correlating Distance', fontdict={'fontsize':12}, pad=8)
        st.pyplot(fig1)
    
    with comment_c:
        st.text_area("Comment on correlation",  '''
                      Distance is highly correlated with activity minute, elapsed time and calories. More I run kilometers, more my running time, colaories and elapsed time increase''',
                      height = 1
                     )
    st.write(" ")
    st.write(" ")
    st.write("Map : where do I run?")
    # Other method
    # df = pd.DataFrame(data=zip(running.lat.tolist(), running.lon.tolist()),columns=['lat', 'lon'])
    # st.map(df, zoom= 5, use_container_width=True)
    
    view = pdk.ViewState(latitude=46.227638, longitude=2.213749, zoom=4.8)
    ville = pdk.Layer(
                        "ScatterplotLayer",
                        data=running,
                        pickable=False,
                        opacity=0.3,
                        stroked=True,
                        filled=True,
                        radius_scale=10,
                        radius_min_pixels=5,
                        radius_max_pixels=60,
                        line_width_min_pixels=1,
                        get_position=["lon", "lat"],
                        get_fill_color=[252, 136, 3],
                        get_line_color=[255,0,0],
                        tooltip="test test"
                     )
    mappy = pdk.Deck(
                layers=[ville],
                initial_view_state=view,
                map_style="mapbox://styles/mapbox/light-v10"
                )
    st.pydeck_chart(mappy)
    
    
#Page 2 : steps and floors
elif selected_pages == "Walking (steps, floors) and Sleep":
    st.header("Steps, Floors Climbed and Sleep Activities")
    
    st.subheader("Floors climbed")
    st.markdown("**Maximum** et Mininum")
    #create 4 columns for max, min, mean
    floors_max, floors_min , floors_moy , day = st.columns(4)
    
    floors_max.metric(label="Max in a month", value=str(floors["Floors Climbed1"].max())+" floors")
    floors_min.metric(label="Min in a month", value=str(floors["Floors Climbed1"].min())+" floors")
    floors_moy.metric(label="Mean in a month", value=str(floors["Floors Climbed1"].mean())+" floors/month")
    day.metric(label="Mean per day", value=str(floors["Floors Climbed1"].mean()/30)+ " floors/day")
 
    st.write(" ")
    st.write(" ")
    st.write(" ")              
    st.write("Repartition by year")
    # Add evolution year with up/down floors climbed
    fig = plt.figure(figsize=(20, 8))

    neg = floors["Floors Descended"].tolist()
    pos = floors["Floors Climbed1"].tolist()
    dd = floors["mm-yyyy"].tolist()

    plt.bar(dd, neg, facecolor='#9999ff', edgecolor='black')
    plt.bar(dd, pos, facecolor='#ff9999', edgecolor='black')

    st.pyplot(fig)
    
    st.markdown("**Maximum** et Mininum STEPS")
    #create 4 columns for max, min, mean
    steps_max, steps_min , steps_moy , day = st.columns(4)
    
    steps_max.metric(label="Max in a month", value=str(steps["Actual"].max())+" steps")
    steps_min.metric(label="Min in a month", value=str(steps["Actual"].min())+" steps")
    steps_moy.metric(label="Mean in a month", value=("{:,.1f}".format(steps["Actual"].mean())))
    day.metric(label="Mean per day", value=("{:,.1f}".format(steps["Actual"].mean()/30)))

    st.write(" ")
    st.write(" ")
    st.write(" ")
    
    #Add steps
    st.subheader("Steps")
    st.write("Repartition by year")
    month_st , year_st = st.columns((2, (1)))
    
    # By year
    with month_st:
        months = steps["Steps"].unique().tolist()
        s = steps.groupby(['Steps'])["Actual"].sum().tolist()
     
        fig = plt.figure(figsize=(12, 6))
        plt.bar(months, s, color=['C4'])
        st.pyplot(fig)
 
    with year_st:
        labels = steps["year"].unique().tolist()
        s1 = steps.groupby(['year'])["Actual"].sum().tolist()

        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.pie(s1, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        st.pyplot(fig1)
    
    st.write(" ")
    st.write(" ")
    st.write(" ")    
    #Add sleep
    st.subheader("Sleep")
    st.write("Repartition sleep time")
    
    month_sl , year_sl, max_, min_ = st.columns((6, 1,1,1))
    
    with month_sl:
        fig = plt.figure(figsize=(10, 3))
        #plot repartition time of sleep
        sns.set(rc={'xtick.labelsize':6,'ytick.labelsize':8})
        sns.stripplot(x="Sleep Time", y="Hrs", data=sleep)
        #add mean line
        ymed = sleep["Hrs"].median()
        x = plt.gca().axes.get_xlim()
        plt.plot(x, len(x) * [ymed], sns.xkcd_rgb["pale red"])
     
        st.pyplot(fig)
    
    with year_sl:
        value=sleep["Hrs"].mean()
        value=("{:,.2f}".format(value))
        #year_sl.metric(label="Mean of sleep", value=str(sleep["Hrs"].mean())+ " hrs")
        year_sl.metric(label="Mean of sleep - hrs", value=value)
    
    with max_ :
        value=sleep["Hrs"].max()
        value=("{:,.2f}".format(value))
        #year_sl.metric(label="Mean of sleep", value=str(sleep["Hrs"].mean())+ " hrs")
        year_sl.metric(label="Max of sleep - hrs", value=value)
        
    with min_ :
        value=sleep["Hrs"].min()
        value=("{:,.2f}".format(value))
        year_sl.metric(label="Min of sleep - hrs", value=value)