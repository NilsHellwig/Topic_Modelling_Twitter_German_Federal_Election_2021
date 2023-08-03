import matplotlib.font_manager as font_manager
import pandas as pd

FONT_PATH = "fonts/LinLibertine_R.ttf"

def add_missing_months(df):
    # Erstelle einen leeren DataFrame mit allen Monaten des Jahres
    all_months = pd.DataFrame({'month': range(1, 13)})
    
    # Führe einen Left Join durch, um die fehlenden Monate hinzuzufügen
    merged_df = all_months.merge(df, on='month', how='left')
    
    # Fülle die fehlenden Werte mit 0
    merged_df['count'] = merged_df['count'].fillna(0)
    
    return merged_df

def setup_font(plt):
    font_manager.fontManager.addfont(FONT_PATH)
    prop = font_manager.FontProperties(fname=FONT_PATH)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['font.sans-serif'] = prop.get_name()
    return plt

def plot_topic_subplots(df_topics, n_topics, topic_count, filename="results/top5_topics"):
    print(topic_count[0])
    import matplotlib.pyplot as plt
    
    num_top_words = 6
    num_columns = 4
    num_rows = n_topics // num_columns + 1
    
    setup_font(plt)

    fig = plt.figure(figsize=(25, 4*num_rows))

    for i in range(n_topics):
        if i >= num_rows * num_columns:
            break

        words = df_topics["topic_" + str(i) + "_word"].tolist()[:num_top_words]
        scores = df_topics["topic_" + str(i) + "_score"].tolist()[:num_top_words]

        sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        words = [words[j] for j in sorted_indices]
        scores = [scores[j] for j in sorted_indices]

        ax = fig.add_subplot(num_rows, num_columns, i+1)
        scores.reverse() 
        bars = ax.barh(range(num_top_words), scores, color='white', edgecolor='black')

        biggest_bar = max([x.get_width() for x in bars])

        for bar in bars:
            width = bar.get_width()
            if width > biggest_bar / 5:
                ax.text(width - (0.02 * biggest_bar), bar.get_y() + bar.get_height() / 2 - 0.05, f'{width:.3f}', ha='right', va='center', color='black')
            else:
                ax.text(width + (0.02 * biggest_bar), bar.get_y() + bar.get_height() / 2, f'{width:.3f}', ha='left', va='center')

        words = list(reversed(words))

        ax.set_yticks(range(num_top_words))
        ax.set_yticklabels(words)
        ax.set_title(f"Topic {i+1} (n={topic_count[i]})", fontweight='bold')
        ax.set_xlabel("Score")

    plt.subplots_adjust(wspace=0.65, hspace=0.5) 
    plt.gcf().set_size_inches(25, 4*num_rows)  
    
    for f_type in [".svg", ".png", ".pdf"]:
        plt.savefig(filename+f_type, dpi=300, bbox_inches='tight')
        
    plt.show()
    

def plot_timeseries_subplots(df_topics, n_topics, topics_over_time, filename="results/timeseries_topics"):
    import matplotlib.pyplot as plt

    num_plots_per_row = 5
    num_rows = n_topics // num_plots_per_row + (n_topics % num_plots_per_row > 0)
    num_columns = min(n_topics, num_plots_per_row)
    
    setup_font(plt)

    fig = plt.figure(figsize=(25, 4*num_rows))

    for i in range(n_topics):
        top_3 = list(df_topics["topic_"+str(i)+"_word"][:3])
        if i >= num_rows * num_columns:
            break
    
        ax = fig.add_subplot(num_rows, num_columns, i+1)
        topic_data = topics_over_time[topics_over_time["Topic"] == i]

        for month in range(1, 13):
            if month not in topic_data["Timestamp"].to_list():
                topic_data = topic_data.append({"Topic": i, "Frequency": 0, "Timestamp": month}, ignore_index=True)
    
        topic_data = topic_data.sort_values("Timestamp")
    
        timestamps = topic_data["Timestamp"].tolist()
        frequencies = topic_data["Frequency"].tolist()
        
        ax.plot(timestamps, frequencies, color='black')

        ax.set_title(f"Topic {i+1}\n" + ','.join(top_3))
        ax.set_xlabel("Month")
        ax.set_ylabel("Frequency")
        ax.set_xticks(range(1, 13))  # Setze Achsenticks für alle Werte von 1 bis 12
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)  # Entferne Tick-Markierungen

    plt.tight_layout()
    
    for f_type in [".svg", ".png", ".pdf"]:
        plt.savefig(filename+f_type, dpi=300, bbox_inches='tight')
        
    plt.show()
    
def plot_timeseries_sentiment_subplots(df_topics, n_topics, document_info, topics_over_time, filename="results/timeseries_sentiment_subplots"):
    import matplotlib.pyplot as plt

    num_plots_per_row = 4
    num_rows = n_topics // num_plots_per_row + (n_topics % num_plots_per_row > 0)
    num_columns = min(n_topics, num_plots_per_row)
    
    setup_font(plt)
    
    fig = plt.figure(figsize=(25, 4*num_rows))
    
    for i in range(n_topics):
        top_3 = list(df_topics["topic_"+str(i)+"_word"][:3])
        if i >= num_rows * num_columns:
            break
        ax = fig.add_subplot(num_rows, num_columns, i+1)
    
        sentiments = document_info["sentiment"].unique()
    
        topic_counts = document_info[document_info["Topic"] == i].groupby("month").agg({"sentiment": "first", "Topic": "size"}).reset_index()
        topic_counts.columns = ["month", "sentiment", "count"]
        topic_counts = add_missing_months(topic_counts)
        ax.plot(topic_counts["month"], topic_counts["count"], color="grey", linestyle='--')
        
        for sentiment in sentiments:
            topic_counts = document_info[(document_info["Topic"] == i) & (document_info["sentiment"] == sentiment)]
            topic_counts = topic_counts.groupby("month").size().reset_index(name="count")
            topic_counts = add_missing_months(topic_counts)
   
            ax.plot(topic_counts["month"], topic_counts["count"], label=sentiment, color='red' if sentiment == 1 else 'green' if sentiment == 0 else 'orange')
        
    
        ax.set_xlabel("Month")
        ax.set_ylabel("# Tweets")
        ax.set_xticks(range(1, 13))
        ax.set_title(f"Topic {i+1}\n" + ','.join(top_3))
    
    plt.subplots_adjust(wspace=0.4, hspace=0.55) 
    
    for f_type in [".svg", ".png", ".pdf"]:
        plt.savefig(filename+f_type, dpi=300, bbox_inches='tight')
        
    plt.show()