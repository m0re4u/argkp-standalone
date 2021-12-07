import pandas as pd

def convert_csv(infile):
    df = pd.read_csv(infile, index_col=0)
    new_df = pd.DataFrame({
        'english': df['content_translated'],
        'extracted_from': 'pro',
        'project': 0,
        'quality_scores': df['Quality estimate'],
    })
    new_df.to_csv('nu_nl_data.csv')



if __name__ == "__main__":
    convert_csv('translated_filtered_qualityrated.csv')