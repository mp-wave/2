import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from matplotlib.patches import Arc
from matplotlib.colors import TwoSlopeNorm
import matplotlib.image as image
import matplotlib.cbook as cbook
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import streamlit as st
from streamlit.server.server import Server
from streamlit.legacy_caching.hashing import _CodeHasher
from io import BytesIO



zo = 12

st.set_page_config(layout="wide", page_title='Wave Sports Science App', initial_sidebar_state='collapsed')


def main():
    state = _get_state()
    pages = {
        "Training PDF": TrainingPDF,
        "Match PDF": MatchPDF,
        "ZScore Download": ZScore
    }

    # st.sidebar.title("Page Filters")
    page = st.sidebar.radio("Select Page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()


def display_state_values(state):
    st.write("Input state:", state.input)
    st.write("Slider state:", state.slider)
    # st.write("Radio state:", state.radio)
    st.write("Checkbox state:", state.checkbox)
    st.write("Selectbox state:", state.selectbox)
    st.write("Multiselect state:", state.multiselect)

    for i in range(3):
        st.write(f"Value {i}:", state[f"State value {i}"])

    if st.button("Clear state"):
        state.clear()


def multiselect(label, options, default, format_func=str):
    """multiselect extension that enables default to be a subset list of the list of objects
     - not a list of strings

     Assumes that options have unique format_func representations

     cf. https://github.com/streamlit/streamlit/issues/352
     """
    options_ = {format_func(option): option for option in options}
    default_ = [format_func(option) for option in default]
    selections = st.multiselect(
        label, options=list(options_.keys()), default=default_, format_func=format_func
    )
    return [options_[format_func(selection)] for selection in selections]


# selections = multiselect("Select", options=[Option1, Option2], default=[Option2])


class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = st._get_script_run_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


datafile = cbook.get_sample_data('/Users/michaelpoma/Documents/Python/PDF Pics/Wave PDF Header.png', asfileobj=False)
im = image.imread(datafile)

# fontPathBold = "/Users/michaelpoma/Library/Fonts/EBGaramond-Bold.ttf"
# fontPathNBold = "/Users/michaelpoma/Library/Fonts/EBGaramond-Medium.ttf"
fontPathBold = "/Users/michaelpoma/Library/Fonts/PTSans-Bold.ttf"
fontPathNBold = "/Users/michaelpoma/Library/Fonts/PTSans-Regular.ttf"
titlepage = fm.FontProperties(fname=fontPathBold, size=72)
headers = fm.FontProperties(fname=fontPathBold, size=46)
footers = fm.FontProperties(fname=fontPathNBold, size=24)
metrics1 = fm.FontProperties(fname=fontPathBold, size=20)
metrics2 = fm.FontProperties(fname=fontPathBold, size=24)
numbers = fm.FontProperties(fname=fontPathBold, size=22)
tabletitle = fm.FontProperties(fname=fontPathBold, size=19)


def TrainingPDF(state):
    uploaded_file = st.file_uploader("Choose a File")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)

    df['Full Name'] = df['Player First Name'] + ' ' + df['Player Last Name']
    # df['Full Name'] = df['Player Display Name']

    df['Time In Heart Rate Zone 6'] = df['Time In Heart Rate Zone 6'] = pd.to_datetime(df['Time In Heart Rate Zone 6'])
    df['extracted_minute_timestamp'] = df['Time In Heart Rate Zone 6'].dt.minute
    df['extracted_seconds_timestamp'] = df['Time In Heart Rate Zone 6'].dt.second
    df['HRZ6_total_seconds'] = (df.extracted_minute_timestamp * 60) + df.extracted_seconds_timestamp
    df['HRZ6_minutes'] = round(df.HRZ6_total_seconds / 60, 2)


    sessiondate = st.date_input(
        "Date of the Training Session",
        datetime.date(2022, 2, 1))

    Session = df['Session Week Number'].mean()
    #Session = st.number_input('Insert Session Week Number')

    test = df[(df['Session Week Number'] == Session)]
    test = test.groupby(["Full Name"], as_index=False).agg({"Total Distance": 'sum',
                                                            'Distance Per Min': 'mean', 'Max Speed': 'max',
                                                            'Sprints': 'sum', 'High Speed Running (Absolute)': 'sum',
                                                            'Number Of High Intensity Bursts': 'sum', 'Explosive Distance (Absolute)':'sum',
                                                            'Accelerations':'sum','Decelerations':'sum','HRZ6_minutes':'sum'}).sort_values(
        by='Full Name', ascending=True)
    player_names = test['Full Name'].to_list()
    #print(test)

    User = st.text_input('Insert Name of User to Save PDF')
    download_location = '/Users/'+str(User)+'/Downloads/'
    if st.button('Click Here to Start Generating PDF - Download to Follow'):
        with PdfPages(str(download_location) + str(sessiondate) + ' - ' + 'Session Report.pdf') as pdf:

            fig = plt.figure(figsize=(32, 20))
            newax = fig.add_axes([0, 0, 1, 1], zorder=-1)
            titlepic = cbook.get_sample_data(
                '/Users/michaelpoma/Documents/Python/PDF Pics/PDF Report Title/Wave PDF Report Title.png',
                asfileobj=False)
            titleim = image.imread(titlepic)
            newax.imshow(titleim, aspect='auto')
            newax.axis('off')
            plt.axis('off')
            fig.text(0.675, 0.18, str(sessiondate) + " Training Report", fontproperties=titlepage, color='#21C6D9',
                     horizontalalignment='center', verticalalignment='center',
                     bbox=dict(facecolor='#032E62', alpha=0.75, edgecolor='#21C6D9', lw=3.5))
            fig.text(0.675, 0.12, str('Whole Team'), fontproperties=titlepage,
                     horizontalalignment='center', verticalalignment='center', color='#032E62',
                     bbox=dict(facecolor='#21C6D9', alpha=0.85, edgecolor='#032E62', lw=3.5))
            pdf.savefig()
            plt.close()

            fig, ax = plt.subplots(figsize=(32, 20))
            gs = gridspec.GridSpec(1, 1, wspace=0.1)
            ax1 = plt.subplot(gs[:, :])
            speed = test
            speedplot = pd.DataFrame(data=speed,
                                     columns=['Full Name', 'Max Speed']).sort_values(by=['Max Speed'], ascending=False)
            # print(speedplot)
            speedplot = speedplot.set_index('Full Name')

            speedplot = speedplot.plot(kind='bar', ax=ax1, rot=0,
                                       color={"Max Speed": "#FC1896"}, fontsize=14)
            plt.xticks(rotation=45)
            speedplot.set_ylim((22, 34))
            for container in ax1.containers:
                ax1.bar_label(container, fontsize=14)
            ax1.set_ylabel('km/h', fontsize=12)
            newax = fig.add_axes([0, 0.925, 1.15, 0.2], zorder=-1)
            fig.text(0.5, 0.94, str('Max Speed Across Team\n' + str(sessiondate) + ' Training Session'),
                     horizontalalignment='center', color='white', fontproperties=headers, zorder=zo + 2)
            newax.imshow(im, aspect='auto')
            newax.axis('off')
            fig.text(0.075, 0.01, 'Max Speed Graph', horizontalalignment='center', color='black',
                     fontproperties=footers,
                     zorder=zo)
            fig.text(0.9, 0.01, str(sessiondate) + ' Training Report', horizontalalignment='center', color='black',
                     fontproperties=footers, zorder=zo)
            pdf.savefig()
            plt.close()

            fig, ax = plt.subplots(figsize=(32, 20))
            gs = gridspec.GridSpec(1, 1, wspace=0.1)
            ax1 = plt.subplot(gs[:, :])
            speed = test
            speedplot = pd.DataFrame(data=speed,
                                     columns=['Full Name', 'Number Of High Intensity Bursts']).sort_values(
                by=['Number Of High Intensity Bursts'], ascending=False)
            # print(speedplot)
            speedplot = speedplot.set_index('Full Name')

            speedplot = speedplot.plot(kind='bar', ax=ax1, rot=0,
                                       color={'Number Of High Intensity Bursts': "#FC1896"}, fontsize=14)
            plt.xticks(rotation=45)
            for container in ax1.containers:
                ax1.bar_label(container, fontsize=14)
            newax = fig.add_axes([0, 0.925, 1.15, 0.2], zorder=-1)
            fig.text(0.5, 0.94, str('High Intensity Bursts Across Team\n' + str(sessiondate) + ' Training Session'),
                     horizontalalignment='center', color='white', fontproperties=headers, zorder=zo + 2)
            newax.imshow(im, aspect='auto')
            newax.axis('off')
            fig.text(0.075, 0.01, 'High Intensity Bursts Graph', horizontalalignment='center', color='black',
                     fontproperties=footers,
                     zorder=zo)
            fig.text(0.9, 0.01, str(sessiondate) + ' Training Report', horizontalalignment='center', color='black',
                     fontproperties=footers, zorder=zo)
            pdf.savefig()
            plt.close()

            fig, ax = plt.subplots(figsize=(32, 20))
            gs = gridspec.GridSpec(1, 1, wspace=0.1)
            ax1 = plt.subplot(gs[:, :])
            speed = test
            speedplot = pd.DataFrame(data=speed,
                                     columns=['Full Name', 'HRZ6_minutes']).sort_values(by=['HRZ6_minutes'],
                                                                                        ascending=False)
            print(speedplot)
            speedplot = speedplot.set_index('Full Name')

            speedplot = speedplot.plot(kind='bar', ax=ax1, rot=0,
                                       color={'HRZ6_minutes': "#FC1896"}, fontsize=14)
            plt.xticks(rotation=45)
            for container in ax1.containers:
                ax1.bar_label(container, fontsize=14)
            ax1.set_ylabel('Minutes', fontsize=12)
            newax = fig.add_axes([0, 0.925, 1.15, 0.2], zorder=-1)
            fig.text(0.5, 0.94, str('Heart Rate Zone 6 Across Team\n' + str(sessiondate) + ' Training Session'),
                     horizontalalignment='center', color='white', fontproperties=headers, zorder=zo + 2)
            newax.imshow(im, aspect='auto')
            newax.axis('off')
            fig.text(0.075, 0.01, 'Time In Heart Rate Zone 6 Graph', horizontalalignment='center', color='black',
                     fontproperties=footers,
                     zorder=zo)
            fig.text(0.9, 0.01, str(sessiondate) + ' Training Report', horizontalalignment='center', color='black',
                     fontproperties=footers, zorder=zo)
            pdf.savefig()
            plt.close()

            for player_name in player_names:
                fig, ax = plt.subplots(figsize=(32, 20))
                gs = gridspec.GridSpec(3, 2, wspace=0.1)
                ax1 = plt.subplot(gs[0, :])
                rec1 = plt.Rectangle((0.325, .825), .3, .1, ls='-', color='#21C6D9', zorder=zo, alpha=1)
                rec2 = plt.Rectangle((0.275, .925), .2, -.9, ls='-', color='#032E62', zorder=zo, alpha=1)
                ax1.add_artist(rec1)
                ax1.add_artist(rec2)
                ax1.text(0.55, .875, "Output", fontproperties=tabletitle, color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.375, .875, "STATSports Metrics", fontproperties=tabletitle, color='#FC1896',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.375, .775, "Total Distance", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.375, .675, "Distance per Min", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.375, 0.575, "Max Speed", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.375, 0.475, "Sprints", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.375, 0.375, "High Speed Running", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.375, 0.275, "Explosive Distance", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.375, 0.175, "High Intensity Bursts", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.375, 0.075, "Accelerations/Decelerations", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                output = test[test['Full Name'] == player_name]
                ax1.text(0.55, 0.775, str(sum(output['Total Distance'])) + ' m', fontproperties=numbers,
                         color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)
                ax1.text(0.55, 0.675, str(round(sum(output['Distance Per Min']), 1)) + ' m', fontproperties=numbers,
                         color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)
                ax1.text(0.55, 0.575, str(round(sum(output['Max Speed']), 1)) + ' km/h', fontproperties=numbers,
                         color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)
                ax1.text(0.55, 0.475, str(sum(output['Sprints'])), fontproperties=numbers, color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)
                ax1.text(0.55, 0.375, str(sum(output['High Speed Running (Absolute)'])) + ' m', fontproperties=numbers,
                         color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)
                ax1.text(0.55, 0.275, str(sum(output['Explosive Distance (Absolute)'])) + ' m', fontproperties=numbers,
                         color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)
                ax1.text(0.55, 0.175, str(sum(output['Number Of High Intensity Bursts'])), fontproperties=numbers,
                         color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)
                ax1.text(0.55, 0.075, str(sum(output['Accelerations'])) + ' / ' + str(sum(output['Decelerations'])),
                         fontproperties=numbers,
                         color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)

                ax1.axis('off')

                ax2 = plt.subplot(gs[1:, 0])
                distance = test[test['Full Name'] == player_name]
                distanceplot = pd.DataFrame(data=distance,
                                            columns=['Full Name', 'Total Distance', 'High Speed Running (Absolute)'])
                print(distanceplot)
                distanceplot = distanceplot.set_index('Full Name')
                distanceplot = distanceplot.plot(kind='bar', ax=ax2, secondary_y='High Speed Running (Absolute)', rot=0,
                                                 color={"Total Distance": "#032E62",
                                                        "High Speed Running (Absolute)": "#21C6D9"}, fontsize=20)
                distanceplot.right_ax.set_ylim((0, 1000))
                distanceplot.right_ax.set_ylabel('High Speed Running', fontsize=12)
                distanceplot.set_ylabel('Total Distance', fontsize=12)
                distanceplot.set_ylim((2000, 9000))
                distanceplot.set_xlabel('')
                # horizfull_pitch('none', 'black', ax2)

                ax3 = plt.subplot(gs[1:, 1])
                aceldecel = test[test['Full Name'] == player_name]
                aceldecelplot = pd.DataFrame(data=aceldecel,
                                             columns=['Full Name', 'Accelerations', 'Decelerations'])
                print(aceldecelplot)
                aceldecelplot = aceldecelplot.set_index('Full Name')
                accelerationdecelerations = aceldecelplot.plot(kind='bar', ax=ax3, rot=0,
                                                               color={"Accelerations": "#032E62",
                                                                      "Decelerations": "#21C6D9"}, fontsize=20)
                accelerationdecelerations.set_xlabel('')
                accelerationdecelerations.yaxis.tick_right()
                accelerationdecelerations.yaxis.set_label_position("right")
                accelerationdecelerations.set_ylabel('Accels/Decels', fontsize=12)
                plt.yticks(fontsize=20)

                # ax5 = plt.subplot(gs[-1, :])
                # horizfull_pitch('none', 'black', ax5)

                newax = fig.add_axes([0, 0.925, 1.15, 0.2], zorder=-1)
                fig.text(0.5, 0.94, str(player_name) + '\n' + str(sessiondate) + ' Training Session',
                         horizontalalignment='center', color='white', fontproperties=headers, zorder=zo + 2)
                newax.imshow(im, aspect='auto')
                newax.axis('off')
                fig.text(0.075, 0.01, str(player_name), horizontalalignment='center', color='black',
                         fontproperties=footers,
                         zorder=zo)
                fig.text(0.9, 0.01, str(sessiondate) + ' Training Report', horizontalalignment='center', color='black',
                         fontproperties=footers, zorder=zo)
                pdf.savefig()
                plt.close()
                print(str(player_name) + ' done')
            st.success('PDF Successfully Downloaded to ' + str(download_location))

    else:
        pass

def MatchPDF(state):
    uploaded_file = st.file_uploader("Choose a File")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Drill Title'] = ['First' if x == 'FIRST HALF' else 'Second' for x in df['Secondary Label']]
        df['Full Name'] = df['Player First Name'] + ' ' + df['Player Last Name']
        st.write(df)

    #df['Full Name'] = df['Player First Name'] + ' ' + df['Player Last Name']
    # df['Full Name'] = df['Player Display Name']

    df['Time In Heart Rate Zone 6'] = df['Time In Heart Rate Zone 6'] = pd.to_datetime(df['Time In Heart Rate Zone 6'])
    df['extracted_minute_timestamp'] = df['Time In Heart Rate Zone 6'].dt.minute
    df['extracted_seconds_timestamp'] = df['Time In Heart Rate Zone 6'].dt.second
    df['HRZ6_total_seconds'] = (df.extracted_minute_timestamp * 60) + df.extracted_seconds_timestamp
    df['HRZ6_minutes'] = round(df.HRZ6_total_seconds / 60, 2)


    matchdate = st.date_input(
        "Date of the Match",
        datetime.date(2022, 4, 1))

    opp = st.text_input('Input Name of Opposition')

    #Session = df['Session Week Number'].mean()
    #Session = st.number_input('Insert Session Week Number')

    #test = df[(df['Session Week Number'] == Session)]
    testhalf = df.groupby(["Full Name","Drill Title"], as_index=False).agg({"Total Distance": 'sum',
                                                            'Distance Per Min': 'mean', 'Max Speed': 'max',
                                                            'Sprints': 'sum', 'High Speed Running (Absolute)': 'sum',
                                                            'Number Of High Intensity Bursts': 'sum','Explosive Distance (Absolute)':'sum',
                                                            'accels/min':'sum','decels/min':'sum','HRZ6_minutes':'sum'}).sort_values(
        by='Full Name', ascending=True)

    testfull = df.groupby(["Full Name"], as_index=False).agg({"Total Distance": 'sum',
                                                            'Distance Per Min': 'mean', 'Max Speed': 'max',
                                                            'Sprints': 'sum', 'High Speed Running (Absolute)': 'sum',
                                                            'Number Of High Intensity Bursts': 'sum','Explosive Distance (Absolute)':'sum',
                                                            'accels/min':'sum','decels/min':'sum','HRZ6_minutes':'sum'}).sort_values(
        by='Full Name', ascending=True)
    player_names = testfull['Full Name'].to_list()
    #print(test)

    User = st.text_input('Insert Name of User to Save PDF')
    download_location = '/Users/'+str(User)+'/Downloads/'
    if st.button('Click Here to Start Generating PDF - Download to Follow'):
        with PdfPages(str(download_location) + str(matchdate) + ' v ' + str(opp) + ' Match Report.pdf') as pdf:
            fig = plt.figure(figsize=(32, 20))
            newax = fig.add_axes([0, 0, 1, 1], zorder=-1)
            titlepic = cbook.get_sample_data(
                '/Users/michaelpoma/Documents/Python/PDF Pics/PDF Report Title/Wave PDF Report Title.png',
                asfileobj=False)
            titleim = image.imread(titlepic)
            newax.imshow(titleim, aspect='auto')
            newax.axis('off')
            plt.axis('off')
            fig.text(0.675, 0.18, str(matchdate) + " Match Report", fontproperties=titlepage, color='#21C6D9',
                     horizontalalignment='center', verticalalignment='center',
                     bbox=dict(facecolor='#032E62', alpha=0.75, edgecolor='#21C6D9', lw=3.5))
            fig.text(0.675, 0.12, str('Wave') + ' v ' + str(opp), fontproperties=titlepage,
                     horizontalalignment='center', verticalalignment='center', color='#032E62',
                     bbox=dict(facecolor='#21C6D9', alpha=0.85, edgecolor='#032E62', lw=3.5))
            pdf.savefig()
            plt.close()

            fig, ax = plt.subplots(figsize=(32, 20))
            gs = gridspec.GridSpec(1, 1, wspace=0.1)
            ax1 = plt.subplot(gs[:, :])
            speed = testfull
            speedplot = pd.DataFrame(data=speed,
                                     columns=['Full Name', 'Max Speed']).sort_values(by=['Max Speed'], ascending=False)
            # print(speedplot)
            speedplot = speedplot.set_index('Full Name')

            speedplot = speedplot.plot(kind='bar', ax=ax1, rot=0,
                                       color={"Max Speed": "#FC1896"}, fontsize=14)
            plt.xticks(rotation=45)
            speedplot.set_ylim((22, 32))
            for container in ax1.containers:
                ax1.bar_label(container, fontsize=14)
            ax1.set_ylabel('km/h', fontsize=12)
            newax = fig.add_axes([0, 0.925, 1.15, 0.2], zorder=-1)
            fig.text(0.5, 0.94, str('Max Speed Across Team\n' + str(matchdate) + ' v ' + str(opp)),
                     horizontalalignment='center', color='white', fontproperties=headers, zorder=zo + 2)
            newax.imshow(im, aspect='auto')
            newax.axis('off')
            fig.text(0.075, 0.01, 'Max Speed Graph', horizontalalignment='center', color='black',
                     fontproperties=footers,
                     zorder=zo)
            fig.text(0.9, 0.01, str(matchdate) + ' Match Report', horizontalalignment='center', color='black',
                     fontproperties=footers, zorder=zo)
            pdf.savefig()
            plt.close()

            fig, ax = plt.subplots(figsize=(32, 20))
            gs = gridspec.GridSpec(1, 1, wspace=0.1)
            ax1 = plt.subplot(gs[:, :])
            speed = testfull
            speedplot = pd.DataFrame(data=speed,
                                     columns=['Full Name', 'Number Of High Intensity Bursts']).sort_values(
                by=['Number Of High Intensity Bursts'], ascending=False)
            # print(speedplot)
            speedplot = speedplot.set_index('Full Name')

            speedplot = speedplot.plot(kind='bar', ax=ax1, rot=0,
                                       color={'Number Of High Intensity Bursts': "#FC1896"}, fontsize=14)
            plt.xticks(rotation=45)
            for container in ax1.containers:
                ax1.bar_label(container, fontsize=14)
            newax = fig.add_axes([0, 0.925, 1.15, 0.2], zorder=-1)
            fig.text(0.5, 0.94, str('High Intensity Bursts Across Team\n' + str(matchdate) + ' v ' + str(opp)),
                     horizontalalignment='center', color='white', fontproperties=headers, zorder=zo + 2)
            newax.imshow(im, aspect='auto')
            newax.axis('off')
            fig.text(0.075, 0.01, 'High Intensity Bursts Graph', horizontalalignment='center', color='black',
                     fontproperties=footers,
                     zorder=zo)
            fig.text(0.9, 0.01, str(matchdate) + ' Match Report', horizontalalignment='center', color='black',
                     fontproperties=footers, zorder=zo)
            pdf.savefig()
            plt.close()

            fig, ax = plt.subplots(figsize=(32, 20))
            gs = gridspec.GridSpec(1, 1, wspace=0.1)
            ax1 = plt.subplot(gs[:, :])
            speed = testfull
            speedplot = pd.DataFrame(data=speed,
                                     columns=['Full Name', 'HRZ6_minutes']).sort_values(by=['HRZ6_minutes'],
                                                                                        ascending=False)
            print(speedplot)
            speedplot = speedplot.set_index('Full Name')

            speedplot = speedplot.plot(kind='bar', ax=ax1, rot=0,
                                       color={'HRZ6_minutes': "#FC1896"}, fontsize=14)
            plt.xticks(rotation=45)
            for container in ax1.containers:
                ax1.bar_label(container, fontsize=14)
            ax1.set_ylabel('Minutes', fontsize=12)
            newax = fig.add_axes([0, 0.925, 1.15, 0.2], zorder=-1)
            fig.text(0.5, 0.94, str('Heart Rate Zone 6 Across Team\n' + str(matchdate) + ' v ' + str(opp)),
                     horizontalalignment='center', color='white', fontproperties=headers, zorder=zo + 2)
            newax.imshow(im, aspect='auto')
            newax.axis('off')
            fig.text(0.075, 0.01, 'Time In Heart Rate Zone 6 Graph', horizontalalignment='center', color='black',
                     fontproperties=footers,
                     zorder=zo)
            fig.text(0.9, 0.01, str(matchdate) + ' Match Report', horizontalalignment='center', color='black',
                     fontproperties=footers, zorder=zo)
            pdf.savefig()
            plt.close()

            for player_name in player_names:
                fig, ax = plt.subplots(figsize=(32, 20))
                gs = gridspec.GridSpec(3, 2, wspace=0.1)
                ax1 = plt.subplot(gs[0, :])
                rec1 = plt.Rectangle((0.125, .825), .3, .1, ls='-', color='#21C6D9', zorder=zo, alpha=1)
                rec2 = plt.Rectangle((0.075, .925), .2, -.9, ls='-', color='#032E62', zorder=zo, alpha=1)
                ax1.add_artist(rec1)
                ax1.add_artist(rec2)
                rec3 = plt.Rectangle((0.65, .825), .3, .1, ls='-', color='#21C6D9', zorder=zo, alpha=1)
                rec4 = plt.Rectangle((0.6, .925), .2, -.9, ls='-', color='#032E62', zorder=zo, alpha=1)
                ax1.add_artist(rec3)
                ax1.add_artist(rec4)

                ax1.text(0.35, .875, "Output", fontproperties=tabletitle, color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.175, .875, "STATSports Metrics - First Half", fontproperties=tabletitle, color='#FC1896',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.175, .775, "Total Distance", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.175, .675, "Distance per Min", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.175, 0.575, "Max Speed", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.175, 0.475, "Sprints", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.175, 0.375, "High Speed Running", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.175, 0.275, "Explosive Distance", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.175, 0.175, "High Intensity Bursts", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.175, 0.075, "Accels/Decels per Minute", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)

                output = testhalf[(testhalf['Full Name'] == player_name) & (testhalf['Drill Title'] == 'First')]
                ax1.text(0.35, 0.775, str(sum(output['Total Distance'])) + ' m', fontproperties=numbers,
                         color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)
                ax1.text(0.35, 0.675, str(round(sum(output['Distance Per Min']), 1)) + ' m', fontproperties=numbers,
                         color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)
                ax1.text(0.35, 0.575, str(round(sum(output['Max Speed']), 1)) + ' km/h', fontproperties=numbers,
                         color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)
                ax1.text(0.35, 0.475, str(sum(output['Sprints'])), fontproperties=numbers, color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)
                ax1.text(0.35, 0.375, str(sum(output['High Speed Running (Absolute)'])) + ' m', fontproperties=numbers,
                         color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)
                ax1.text(0.35, 0.275, str(sum(output['Explosive Distance (Absolute)'])) + ' m', fontproperties=numbers,
                         color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)
                ax1.text(0.35, 0.175, str(sum(output['Number Of High Intensity Bursts'])), fontproperties=numbers,
                         color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)
                ax1.text(0.35, 0.075,
                         str(round(sum(output['accels/min']), 2)) + ' / ' + str(round(sum(output['decels/min']), 2)),
                         fontproperties=numbers,
                         color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)

                ax1.text(0.875, .875, "Output", fontproperties=tabletitle, color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.7, .875, "STATSports Metrics - Second Half", fontproperties=tabletitle, color='#FC1896',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.7, .775, "Total Distance", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.7, .675, "Distance per Min", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.7, 0.575, "Max Speed", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.7, 0.475, "Sprints", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.7, 0.375, "High Speed Running", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.7, 0.275, "Explosive Distance", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.7, 0.175, "High Intensity Bursts", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)
                ax1.text(0.7, 0.075, "Accels/Decels per Minute", fontproperties=metrics2, color='#21C6D9',
                         horizontalalignment='center', verticalalignment='center', zorder=zo + 2)

                output = testhalf[(testhalf['Full Name'] == player_name) & (testhalf['Drill Title'] == 'Second')]
                ax1.text(0.875, 0.775, str(sum(output['Total Distance'])) + ' m', fontproperties=numbers,
                         color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)
                ax1.text(0.875, 0.675, str(round(sum(output['Distance Per Min']), 1)) + ' m', fontproperties=numbers,
                         color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)
                ax1.text(0.875, 0.575, str(round(sum(output['Max Speed']), 1)) + ' km/h', fontproperties=numbers,
                         color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)
                ax1.text(0.875, 0.475, str(sum(output['Sprints'])), fontproperties=numbers, color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)
                ax1.text(0.875, 0.375, str(sum(output['High Speed Running (Absolute)'])) + ' m', fontproperties=numbers,
                         color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)
                ax1.text(0.875, 0.275, str(sum(output['Explosive Distance (Absolute)'])) + ' m', fontproperties=numbers,
                         color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)
                ax1.text(0.875, 0.175, str(sum(output['Number Of High Intensity Bursts'])), fontproperties=numbers,
                         color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)
                ax1.text(0.875, 0.075,
                         str(round(sum(output['accels/min']), 2)) + ' / ' + str(round(sum(output['decels/min']), 2)),
                         fontproperties=numbers,
                         color='#032E62',
                         horizontalalignment='center', verticalalignment='center', zorder=12)

                ax1.axis('off')

                ax2 = plt.subplot(gs[1:, 0])
                distance = testfull[testfull['Full Name'] == player_name]
                distanceplot = pd.DataFrame(data=distance,
                                            columns=['Full Name', 'Total Distance', 'High Speed Running (Absolute)'])
                print(distanceplot)
                distanceplot = distanceplot.set_index('Full Name')
                distanceplot = distanceplot.plot(kind='bar', ax=ax2, secondary_y='High Speed Running (Absolute)', rot=0,
                                                 color={"Total Distance": "#032E62",
                                                        "High Speed Running (Absolute)": "#21C6D9"}, fontsize=20)
                distanceplot.right_ax.set_ylim((0, 1000))
                distanceplot.right_ax.set_ylabel('High Speed Running', fontsize=12)
                distanceplot.set_ylabel('Total Distance', fontsize=12)
                distanceplot.set_ylim((2000, 9000))
                distanceplot.set_xlabel('')
                # horizfull_pitch('none', 'black', ax2)

                ax3 = plt.subplot(gs[1:, 1])
                aceldecel = testfull[testfull['Full Name'] == player_name]
                aceldecelplot = pd.DataFrame(data=aceldecel,
                                             columns=['Full Name', 'accels/min', 'decels/min'])
                print(aceldecelplot)
                aceldecelplot = aceldecelplot.set_index('Full Name')
                accelerationdecelerations = aceldecelplot.plot(kind='bar', ax=ax3, rot=0,
                                                               color={"accels/min": "#032E62", "decels/min": "#21C6D9"},
                                                               fontsize=20)
                accelerationdecelerations.set_xlabel('')
                accelerationdecelerations.yaxis.tick_right()
                accelerationdecelerations.yaxis.set_label_position("right")
                accelerationdecelerations.set_ylabel('Accels/Decels', fontsize=12)
                plt.yticks(fontsize=20)

                # ax5 = plt.subplot(gs[-1, :])
                # horizfull_pitch('none', 'black', ax5)

                newax = fig.add_axes([0, 0.925, 1.15, 0.2], zorder=-1)
                fig.text(0.5, 0.94, str(player_name) + '\n ' + str(matchdate) + ' v ' + str(opp),
                         horizontalalignment='center', color='white', fontproperties=headers, zorder=zo + 2)
                newax.imshow(im, aspect='auto')
                newax.axis('off')
                fig.text(0.075, 0.01, str(player_name), horizontalalignment='center', color='black',
                         fontproperties=footers,
                         zorder=zo)
                fig.text(0.9, 0.01, str(matchdate) + ' Match Report', horizontalalignment='center', color='black',
                         fontproperties=footers, zorder=zo)
                pdf.savefig()
                plt.close()
                print(str(player_name) + ' done')
            st.success('PDF Successfully Downloaded to ' + str(download_location))

    else:
        pass

def ZScore(state):
    st.markdown('## Conditional Formatted Dataframe')
    st.text('Sortable Table - Colored by ZScore by Positional Group within League - Use Download Button to View Easiest in Excel with Panes Frozen')
    st.markdown('#')

    weeknumber = int(st.number_input('Input Week Number'))
    uploaded_file = st.file_uploader("Choose a File")
    if uploaded_file is not None:
        td = pd.read_excel(uploaded_file, sheet_name='Total Distance',
                           index_col='Player Name ')
        hsr = pd.read_excel(uploaded_file, sheet_name='High Speed Running ',
                            index_col='Player Name ')
        sd = pd.read_excel(uploaded_file, sheet_name='Sprint Distance ',
                           index_col='Player Name ')
        td['MeanWeek'] = round(td.iloc[:, 1:].mean(axis=1), 2)
        td['StdWeek'] = round(td.iloc[:, 1:].std(axis=1), 2)
        td['ZScore'] = round((td['Week ' + str(weeknumber)] - td.MeanWeek) / td.StdWeek, 2)
        sd['MeanWeek'] = round(sd.iloc[:, 1:].mean(axis=1),2)
        sd['StdWeek'] = round(sd.iloc[:, 1:].std(axis=1),2)
        sd['ZScore'] = round((sd['Week '+str(weeknumber)] - sd.MeanWeek) / sd.StdWeek,2)
        hsr['MeanWeek'] = round(hsr.iloc[:, 1:].mean(axis=1), 2)
        hsr['StdWeek'] = round(hsr.iloc[:, 1:].std(axis=1), 2)
        hsr['ZScore'] = round((hsr['Week ' + str(weeknumber)] - hsr.MeanWeek) / hsr.StdWeek, 2)
        #st.write(td)
        #st.write(hsr)
        #st.write(sd)


    col_list = ['ZScore']

    sd_df = (sd.style.background_gradient(vmin=-2, vmax=2,
                                           cmap=sns.color_palette("seismic_r", as_cmap=True),
                                           subset=col_list))

    st.markdown('Sprint Distance Data')
    st.dataframe(sd_df, width=1280, height=768)
    td_df = (td.style.background_gradient(vmin=-2, vmax=2,
                                           cmap=sns.color_palette("seismic_r", as_cmap=True),
                                           subset=col_list))

    st.markdown('Total Distance Data')
    st.dataframe(td_df, width=1280, height=768)
    hsr_df = (hsr.style.background_gradient(vmin=-2, vmax=2,
                                           cmap=sns.color_palette("seismic_r", as_cmap=True),
                                           subset=col_list))

    st.markdown('High Speed Running Data')
    st.dataframe(hsr_df, width=1280, height=768)

    fn = 'Week ' + str(weeknumber) + ' ZScore DataFrame.xlsx'

    def to_excel(df1, df2, df3):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df1.to_excel(writer, sheet_name='High Speed Running')
        df2.to_excel(writer, sheet_name='Total Distance')
        df3.to_excel(writer, sheet_name='Sprint Distance')
        workbook = writer.book
        #worksheet = writer.sheets['Sheet1']
        format1 = workbook.add_format({'num_format': '0.00'})
        #worksheet.set_column('A:A', None, format1)
        #worksheet.freeze_panes('C2')
        writer.save()
        processed_data = output.getvalue()
        return processed_data

    df_xlsx = to_excel(hsr_df, td_df, sd_df)
    st.download_button(label='Download Data as XLSX',
                       data=df_xlsx,
                       file_name=fn)


if __name__ == "__main__":
    main()



# MatchPlayerPDF('NWSL Fall Series', 20201005, 3775665, 'Houston Dash', 'Houston Dash', 'North Carolina Courage')
# MatchPlayerPDF('NWSL Challenge Cup', 20210416, 3787416, 'Seattle Reign', 'Seattle Reign', 'Houston Dash')
# to run : streamlit run "/Users/michaelpoma/Documents/Python/Codes/Wave Sports Science App.py"
