from pynytimes import NYTAPI
import datetime
nyt = NYTAPI("GCk7Zf9wACZ4EcAABGu7qeKuA69xiYLe", parse_dates=True)
import csv


articles = nyt.article_search(
    query = "Supreme Court",
    results = 5000,
    dates = {
        "begin": datetime.datetime(1940, 1, 1),
        "end": datetime.datetime(2022, 6, 1)
    },
    options = {
        "sort": "relevant"
    }
)


print(articles[0])
x = ['pub_date', 'lead_paragraph','snippet','abstract']

# open the file in the write mode
with open('data/nyt.csv', 'w') as f:
    # create the csv writer
    writer = csv.writer(f)
    writer.writerow(str(x))#'pub_date,lead-paragraph,snippet,abstract,print-page,print-section')
    for i in range(len(articles)):
        # write a row to the csv file
        a = articles[i]
        row = ""
        for j in range(len(x)):
            row += str(a[x[j]]) +','
        row = row[:-2]
        writer.writerow(row)

