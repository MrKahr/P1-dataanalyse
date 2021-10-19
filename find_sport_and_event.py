
# ! Find all sports
def find_sport(df):
    # Drop all duplicates in the sports column 
    df_sport = df["Sport"].drop_duplicates()

    return df_sport


# ! Find all events for a given sport
def find_event(df):
    # Drop all duplicates event and becomes df containing player index and event
    df_event = df['Event'].drop_duplicates()

    return df_event