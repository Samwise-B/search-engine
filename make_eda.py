import pandas as pd

# Example data similar to the one provided
data = {
    'query': ['what is rba', 'was ronald reagan a democrat'],
    'passages': [
        {
            'is_selected': [0, 0, 0, 0, 0, 1],
            'passage_text': [
                "Since 2007, the RBA's outstanding reputation has been affected by the 'Securency' or NPA scandal.",
                "The Reserve Bank of Australia (RBA) came into being on 14 January 1960 as Australia 's central bank.",
                'RBA Recognized with the 2014 Microsoft US Regional Partner of the ... by PR Newswire.',
                'The inner workings of a rebuildable atomizer are surprisingly simple.',
                'Results-Based Accountability® (also known as RBA) is a disciplined way of thinking.',
                'RBA uses a data-driven, decision-making process to help communities and organizations.'
            ],
            'url': [
                'https://en.wikipedia.org/wiki/Reserve_Bank_of_Australia',
                'https://en.wikipedia.org/wiki/Reserve_Bank_of_Australia',
                'http://acronyms.thefreedictionary.com/RBA',
                'https://www.slimvapepen.com/rebuildable-atomizer-rba/',
                'http://rba-africa.com/about/what-is-rba/',
                'http://resultsleadership.org/what-is-results-based-accountability-rba/'
            ]
        },
        {
            'is_selected': [0, 1, 0],
            'passage_text': [ ],
            'url': [
                'http://www.history.com/topics/us-presidents/ronald-reagan',
                'https://en.wikipedia.org/wiki/Reagan_Democrat',
                'http://www.answers.com/Q/Was_Ronald_Reagan_a_republican_or_a_democrat'
            ]
        }
    ]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Save the DataFrame to a Parquet file
df.to_parquet('nested_data_example.parquet')

print("Parquet file created successfully.")
