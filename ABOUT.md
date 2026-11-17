### What is FasalScope?

A small, complete data-science product, built for the course **1MTAIDS 33 Data Science** (Department of Applied Electronics, Sant Gadge Baba Amravati University) by **Dr. Nilesh P. Ingle**.

It joins up the whole course in one app:

| Part of the app | Course idea |
|---|---|
| Photo → colour numbers | turning raw data into features |
| SVM, Naive Bayes, logistic regression | classification, probability, Bayes' rule |
| Uploads saved to MongoDB | NoSQL data storage |
| Community dashboard, Rain Board | statistics and visualisation with Bokeh |
| Model Inspector | thresholds, confusion matrix, overfitting |

**Honest limits.** The models only count colours. Lighting, background and banana variety all change the numbers, so treat the verdict as a classroom demonstration, not advice. The training photos show the same bananas many times on the same table, so the accuracy shown is optimistic: real phone photos are harder.

**Privacy.** Uploads store six colour numbers, a tiny 96-pixel thumbnail, the first name you typed and the time. Please photograph fruit only.

**Data credits.**
- Banana photos (training data and the two example photos): *Ripe Unripe Banana Dataset*, Mendeley Data, CC BY 4.0.
- Rainfall: India Meteorological Department subdivision rainfall 1901–2015, via data.gov.in.

**Code:** https://github.com/learndataa/fasalscope (MIT licence).
